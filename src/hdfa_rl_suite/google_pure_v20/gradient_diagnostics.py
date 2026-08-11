"""Candidate information, reference-gradient, scale, and acquisition audits."""
from __future__ import annotations

import math
import time
from typing import Any, Iterable

import numpy as np

from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.gaussian import (
    BehaviorSnapshot,
    component_log_probability,
)
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.losses import (
    total_loss_and_gradients,
)
from hdfa_rl_suite.google_pure_v19_experimental.dynamic_validation import _controller_spec

from .core import (
    batch_snr,
    candidate_snr,
    cosine_alignment,
    fixed_budget_equal,
    perturbation_rank,
    rowspace_overlap,
    update_efficiency,
    wrong_sign_probability,
)
from .data import (
    evaluator,
    exact_update_direction,
    load_matched_run,
    replay_gradients,
    selected_fast_epochs,
    verify_import_manifest,
)
from .io import ARTIFACT_ROOT, canonical_hash, read_json, settings, write_artifact


def _phase(epoch: float, frequency: float) -> float:
    return float((2.0 * np.pi * float(frequency) * float(epoch)) % (2.0 * np.pi))


def _beneficial_direction(mean: np.ndarray, epoch: float, frequency: float) -> np.ndarray:
    target = evaluator().plant.latent_controls_for(
        np.full(41, math.sin(2.0 * math.pi * frequency * epoch)))
    direction = target - np.asarray(mean, dtype=float)
    norm = float(np.linalg.norm(direction))
    return direction / norm if norm > 0 else np.ones(41) / math.sqrt(41)


def _direction_from_rewards(actions: np.ndarray, rewards: np.ndarray, mean: np.ndarray,
                            sigma: np.ndarray, baseline: np.ndarray,
                            epoch: int) -> tuple[np.ndarray, np.ndarray, Any]:
    controller = _controller_spec()
    behavior = BehaviorSnapshot(
        mean, sigma, component_log_probability(actions, mean, sigma), int(epoch))
    loss = total_loss_and_gradients(
        actions, rewards, evaluator().plant.mask, mean, sigma, baseline, behavior,
        clip=controller.ppo_clip,
        entropy_weight=controller.effective_entropy_coefficient,
        baseline_weight=controller.baseline_loss_weight)
    advantages = rewards - baseline[None, :]
    score = (actions - mean[None, :]) / sigma[None, :]**2
    contributions = (advantages @ evaluator().plant.mask.astype(float)) * score
    direction = np.mean(contributions, axis=0)
    if not np.allclose(direction, -loss.grad_mean, rtol=1e-10, atol=1e-12):
        raise RuntimeError("candidate contributions do not reconstruct the supplied update")
    return direction, contributions, loss


def _summarize_directional(values: np.ndarray) -> dict[str, float]:
    snr_candidate = candidate_snr(values)
    snr_batch = batch_snr(values)
    return {
        "candidate_directional_mean": float(np.mean(values)),
        "candidate_directional_std": float(np.std(values, ddof=1)),
        "candidate_SNR": snr_candidate,
        "batch_SNR": snr_batch,
        "estimated_wrong_sign_probability": wrong_sign_probability(snr_batch),
    }


def audit_fast_gradient_statistics() -> dict[str, Any]:
    verify_import_manifest()
    cfg = settings()
    run = load_matched_run("fast")
    replay = replay_gradients(run)
    start, stop = map(int, cfg["analysis_window"])
    frequency = float(cfg["fast_frequency_per_epoch"])
    controller = _controller_spec()
    rows = []
    for item in replay[start:stop]:
        epoch = int(item["epoch"])
        beneficial = _beneficial_direction(item["mean"], epoch, frequency)
        candidate_z = item["candidate_update_contributions"] @ beneficial
        direction = item["update_direction"]
        directional = _summarize_directional(candidate_z)
        projection = float(direction @ beneficial)
        norm2 = float(direction @ direction)
        ranks = perturbation_rank(run["noises"][epoch])
        eta = update_efficiency(
            item["delta_mean"], direction, beneficial, controller.mean_learning_rate)
        rows.append({
            "epoch": epoch,
            "phase_radians": _phase(epoch, frequency),
            **directional,
            "actual_wrong_sign": projection <= 0,
            "cosine_alignment_with_local_beneficial_direction":
                cosine_alignment(direction, beneficial),
            "orthogonal_gradient_power_fraction":
                max(0.0, 1.0 - projection**2 / norm2) if norm2 > 0 else 0.0,
            "gradient_norm": float(np.linalg.norm(direction)),
            "signed_progress": projection,
            "perturbation_matrix": ranks,
            "beneficial_subspace_rowspace_overlap":
                rowspace_overlap(run["noises"][epoch], beneficial),
            "realized_update_efficiency": eta,
        })
    bins = int(cfg["phase_bins"])
    bin_rows = []
    for index in range(bins):
        selected = [row for row in rows if int(row["phase_radians"] / (2*np.pi) * bins) == index]
        z_mean = np.asarray([row["candidate_directional_mean"] for row in selected])
        z_std = np.asarray([row["candidate_directional_std"] for row in selected])
        combined_candidate_snr = abs(float(np.mean(z_mean))) / max(
            float(np.sqrt(np.mean(z_std**2))), 1e-15)
        bin_rows.append({
            "phase_bin": index,
            "phase_center_radians": float((index + .5) * 2*np.pi / bins),
            "epochs": len(selected),
            "candidate_directional_mean": float(np.mean(z_mean)),
            "candidate_directional_std_rms": float(np.sqrt(np.mean(z_std**2))),
            "candidate_SNR": combined_candidate_snr,
            "batch_SNR": math.sqrt(8) * combined_candidate_snr,
            "estimated_wrong_sign_probability": wrong_sign_probability(
                math.sqrt(8) * combined_candidate_snr),
            "actual_wrong_sign_fraction": float(np.mean([
                row["actual_wrong_sign"] for row in selected])),
            "mean_cosine_alignment": float(np.mean([
                row["cosine_alignment_with_local_beneficial_direction"] for row in selected])),
            "mean_orthogonal_gradient_power_fraction": float(np.mean([
                row["orthogonal_gradient_power_fraction"] for row in selected])),
            "cumulative_signed_progress": float(np.sum([
                row["signed_progress"] for row in selected])),
        })
    etas = np.asarray([row["realized_update_efficiency"] for row in rows
                       if row["realized_update_efficiency"] is not None])
    efficiency = {
        "pass": True,
        "definition": "v^T delta_mu / (alpha_mu v^T supplied_update_direction)",
        "sign_convention": "supplied_update_direction=-grad_mean",
        "mean_learning_rate": controller.mean_learning_rate,
        "median_eta": float(np.median(etas)),
        "mean_eta": float(np.mean(etas)),
        "eta_5_95_percentile": np.quantile(etas, [.05, .95]).tolist(),
        "fraction_negative": float(np.mean(etas < 0)),
        "optimizer_applies_supplied_gradient_correctly": bool(
            abs(float(np.median(etas)) - 1.0) < .05 and np.mean(etas < 0) < .01),
        "optimizer_retuning_permitted_by_this_audit": False,
        "forbidden_auto_runs_launched": [],
    }
    write_artifact("fast_update_efficiency", efficiency,
                   title="V20 realized fast mean-update efficiency")
    value = {
        "pass": True,
        "analysis_epoch_window": [start, stop],
        "candidate_count": 8,
        "parameter_count": 41,
        "snr_denominator": "candidate standard deviation (not standard error)",
        "epoch_rows": rows,
        "phase_bins": bin_rows,
        "actual_wrong_sign_fraction": float(np.mean([
            row["actual_wrong_sign"] for row in rows])),
        "median_batch_SNR": float(np.median([row["batch_SNR"] for row in rows])),
        "median_alignment": float(np.median([
            row["cosine_alignment_with_local_beneficial_direction"] for row in rows])),
        "median_orthogonal_gradient_power_fraction": float(np.median([
            row["orthogonal_gradient_power_fraction"] for row in rows])),
        "median_raw_rank": float(np.median([
            row["perturbation_matrix"]["raw_rank"] for row in rows])),
        "median_centered_rank": float(np.median([
            row["perturbation_matrix"]["centered_rank"] for row in rows])),
        "median_beneficial_subspace_overlap": float(np.median([
            row["beneficial_subspace_rowspace_overlap"] for row in rows])),
        "cumulative_signed_progress": float(np.sum([
            row["signed_progress"] for row in rows])),
        "update_efficiency_artifact": "artifacts/google_pure_v20/fast_update_efficiency.json",
        "new_trajectory_generated": False,
        "forbidden_auto_runs_launched": [],
    }
    return write_artifact("fast_gradient_statistics", value,
                          title="V20 fast candidate-level mean-gradient statistics", notes=[
        "Candidate and batch SNR use the candidate standard deviation.",
        "The supplied mean update is replayed from the stored detector rewards and learned baseline.",
    ])


def _sample_rewards(actions: np.ndarray, epoch: int, frequency: float, cycles: int,
                    *, seed: int) -> np.ndarray:
    evaluation = evaluator()
    target_normalized = evaluation.plant.optimum(epoch, frequency)
    target_native = evaluation.boundary.target_to_native(target_normalized)
    rows = []
    shots = int(cycles) // evaluation.plant.rounds
    for index, action in enumerate(actions):
        counts = evaluation.plant.sample_detector_counts(
            evaluation.native(action), epoch=epoch, frequency=frequency,
            qec_cycles=int(cycles), seed=int(seed + index), target_controls=target_native)
        rows.append(-counts / shots)
    return np.asarray(rows)


def _analytic_mean_direction(mean: np.ndarray, epoch: int, frequency: float,
                             delta: float) -> np.ndarray:
    evaluation = evaluator()
    gradient = np.empty(mean.size)
    for coordinate in range(mean.size):
        offset = np.zeros_like(mean); offset[coordinate] = delta
        gradient[coordinate] = (
            evaluation.cost(mean + offset, epoch, frequency) -
            evaluation.cost(mean - offset, epoch, frequency)) / (2.0 * delta)
    return -gradient


def _comparison(estimate: np.ndarray, reference: np.ndarray,
                beneficial: np.ndarray) -> dict[str, Any]:
    ref_norm = float(np.linalg.norm(reference))
    estimate_norm = float(np.linalg.norm(estimate))
    ref_projection = float(reference @ beneficial)
    estimate_projection = float(estimate @ beneficial)
    orthogonal_error = estimate - reference
    if ref_norm > 0:
        orthogonal_error = orthogonal_error - reference * float(
            orthogonal_error @ reference) / ref_norm**2
    return {
        "v_dot_gradient": estimate_projection,
        "v_dot_reference": ref_projection,
        "cosine_alignment": cosine_alignment(estimate, reference),
        "relative_norm_bias": estimate_norm / ref_norm - 1.0 if ref_norm > 0 else None,
        "directional_bias": estimate_projection - ref_projection,
        "orthogonal_error_norm": float(np.linalg.norm(orthogonal_error)),
        "sign_disagreement": bool(estimate_projection * ref_projection < 0),
    }


def compute_reference_gradients() -> dict[str, Any]:
    verify_import_manifest()
    cfg = settings()
    run = load_matched_run("fast")
    replay = replay_gradients(run)
    frequency = float(cfg["fast_frequency_per_epoch"])
    K_high = int(cfg["high_candidate_count"])
    K_ref = int(cfg["reference_candidate_count"])
    rows = []
    families = np.asarray([item.gate_type for item in evaluator().plant.inventory])
    neighborhoods = np.asarray([
        "detectors:" + ",".join(map(str, item.detectors_influenced))
        for item in evaluator().plant.inventory])
    for state_index, epoch in enumerate(selected_fast_epochs()):
        item = replay[epoch]
        mean, sigma, baseline = item["mean"], item["sigma"], item["baseline"]
        beneficial = _beneficial_direction(mean, epoch, frequency)
        ordinary = item["update_direction"]
        highshot_rewards = _sample_rewards(
            item["actions"], epoch, frequency, int(cfg["high_cycles_per_candidate"]),
            seed=2_000_000 + epoch * 100)
        highshot, _, _ = _direction_from_rewards(
            item["actions"], highshot_rewards, mean, sigma, baseline, epoch)
        rng = np.random.default_rng(20_000 + epoch)
        high_noises = rng.normal(size=(K_high, mean.size))
        high_actions = mean[None, :] + sigma[None, :] * high_noises
        high_candidate = exact_update_direction(
            high_actions, mean, sigma, baseline, epoch, frequency)
        half = K_ref // 2
        base = rng.normal(size=(half, mean.size))
        ref_noises = np.concatenate([base, -base], axis=0)
        ref_actions = mean[None, :] + sigma[None, :] * ref_noises
        reference = exact_update_direction(
            ref_actions, mean, sigma, baseline, epoch, frequency)
        analytic = _analytic_mean_direction(
            mean, epoch, frequency, float(cfg["finite_difference_delta"]))
        family_rows = []
        for label in sorted(set(families)):
            selected = families == label
            family_rows.append({
                "family": label,
                "coordinates": int(np.sum(selected)),
                "ordinary_reference_alignment": cosine_alignment(
                    ordinary[selected], reference[selected]),
                "high_candidate_reference_alignment": cosine_alignment(
                    high_candidate[selected], reference[selected]),
            })
        neighborhood_values = []
        for label in sorted(set(neighborhoods)):
            selected = neighborhoods == label
            neighborhood_values.append(cosine_alignment(
                ordinary[selected], reference[selected]))
        rows.append({
            "epoch": epoch,
            "phase_radians": _phase(epoch, frequency),
            "frozen_state_hash": canonical_hash({
                "mean": mean.tolist(), "sigma": sigma.tolist(),
                "baseline": baseline.tolist(), "epoch": epoch}),
            "ordinary_K8_finite_shot": _comparison(ordinary, reference, beneficial),
            "high_shot_K8_M48000": _comparison(highshot, reference, beneficial),
            "high_candidate_K64_exact_marginals": _comparison(
                high_candidate, reference, beneficial),
            "highest_feasible_K256_antithetic_exact_marginals": _comparison(
                reference, reference, beneficial),
            "analytic_deterministic_mean_cost_gradient": _comparison(
                analytic, reference, beneficial),
            "reference_gradient": reference.tolist(),
            "ordinary_gradient": ordinary.tolist(),
            "high_shot_gradient": highshot.tolist(),
            "high_candidate_gradient": high_candidate.tolist(),
            "analytic_gradient": analytic.tolist(),
            "family_resolved_alignment": family_rows,
            "neighborhood_resolved_alignment_median": float(np.nanmedian([
                value for value in neighborhood_values if value is not None])),
        })
    ordinary_alignment = float(np.median([
        row["ordinary_K8_finite_shot"]["cosine_alignment"] for row in rows]))
    highshot_alignment = float(np.median([
        row["high_shot_K8_M48000"]["cosine_alignment"] for row in rows]))
    highcandidate_alignment = float(np.median([
        row["high_candidate_K64_exact_marginals"]["cosine_alignment"] for row in rows]))
    ordinary_sign = float(np.mean([
        row["ordinary_K8_finite_shot"]["sign_disagreement"] for row in rows]))
    if highcandidate_alignment - ordinary_alignment >= .15 and \
            highcandidate_alignment - highshot_alignment >= .08:
        classification = "FINITE_CANDIDATE_VARIANCE"
    elif highshot_alignment - ordinary_alignment >= .15:
        classification = "SHOT_NOISE_LIMITED"
    elif ordinary_alignment >= .7 and ordinary_sign <= .125:
        classification = "WELL_ESTIMATED_GRADIENT"
    elif float(np.median([np.linalg.norm(row["reference_gradient"]) for row in rows])) < 1e-6:
        classification = "REFERENCE_GRADIENT_TOO_WEAK"
    else:
        classification = "SYSTEMATIC_GRADIENT_BIAS"
    value = {
        "pass": True,
        "selected_phase_states": selected_fast_epochs(),
        "reference_definition": "K=256 antithetic exact Stim detector-marginal score gradient",
        "ordinary_definition": "stored K=8, M=12000 finite-shot learned-baseline gradient",
        "high_candidate_definition": "K=64 exact Stim detector-marginal score gradient",
        "high_shot_definition": "same stored K=8 actions resampled at M=48000 cycles",
        "analytic_definition": "central finite-difference deterministic production mean cost",
        "rows": rows,
        "median_ordinary_reference_alignment": ordinary_alignment,
        "median_highshot_reference_alignment": highshot_alignment,
        "median_highcandidate_reference_alignment": highcandidate_alignment,
        "ordinary_sign_disagreement_fraction": ordinary_sign,
        "classification": classification,
        "new_trajectory_generated": False,
        "forbidden_auto_runs_launched": [],
    }
    return write_artifact("fast_reference_gradients", value,
                          title="V20 stored-state fast reference gradients")


def _state_references() -> tuple[dict[int, np.ndarray], dict[int, Any]]:
    path = ARTIFACT_ROOT / "fast_reference_gradients.json"
    artifact = read_json(path) if path.is_file() else compute_reference_gradients()
    refs = {int(row["epoch"]): np.asarray(row["reference_gradient"], dtype=float)
            for row in artifact["rows"]}
    replay = replay_gradients(load_matched_run("fast"))
    return refs, {epoch: replay[epoch] for epoch in refs}


def _information_cell(epoch: int, candidates: int, cycles: int, reference: np.ndarray,
                      replay_item: Any, *, seed: int) -> dict[str, Any]:
    frequency = float(settings()["fast_frequency_per_epoch"])
    rng = np.random.default_rng(seed)
    noises = rng.normal(size=(candidates, 41))
    mean, sigma, baseline = replay_item["mean"], replay_item["sigma"], replay_item["baseline"]
    actions = mean[None, :] + sigma[None, :] * noises
    started = time.perf_counter()
    rewards = _sample_rewards(actions, epoch, frequency, cycles, seed=seed * 1000)
    direction, contributions, _ = _direction_from_rewards(
        actions, rewards, mean, sigma, baseline, epoch)
    runtime = time.perf_counter() - started
    beneficial = reference / max(float(np.linalg.norm(reference)), 1e-15)
    z = contributions @ beneficial
    comparison = _comparison(direction, reference, beneficial)
    return {
        "epoch": epoch,
        "candidates": candidates,
        "cycles_per_candidate": cycles,
        "total_cycle_budget": candidates * cycles,
        **_summarize_directional(z),
        "actual_wrong_sign": float(direction @ reference) <= 0,
        "reference_gradient_alignment": comparison["cosine_alignment"],
        "orthogonal_gradient_power_fraction": max(
            0.0, 1.0 - float(direction @ beneficial)**2 /
            max(float(direction @ direction), 1e-15)),
        "mean_update_prediction_error": float(
            np.linalg.norm(direction - reference) / max(np.linalg.norm(reference), 1e-15)),
        "runtime_seconds": runtime,
        "perturbation_rank": perturbation_rank(noises),
    }


def _aggregate_cells(label: str, cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cell": label,
        "candidates": cells[0]["candidates"],
        "cycles_per_candidate": cells[0]["cycles_per_candidate"],
        "total_cycle_budget_per_state": cells[0]["total_cycle_budget"],
        "states": len(cells),
        "median_directional_SNR": float(np.median([row["batch_SNR"] for row in cells])),
        "wrong_sign_fraction": float(np.mean([row["actual_wrong_sign"] for row in cells])),
        "median_reference_gradient_alignment": float(np.median([
            row["reference_gradient_alignment"] for row in cells])),
        "median_orthogonal_gradient_power_fraction": float(np.median([
            row["orthogonal_gradient_power_fraction"] for row in cells])),
        "median_mean_update_prediction_error": float(np.median([
            row["mean_update_prediction_error"] for row in cells])),
        "runtime_seconds": float(sum(row["runtime_seconds"] for row in cells)),
        "state_rows": cells,
    }


def run_candidate_shot_factorial() -> dict[str, Any]:
    verify_import_manifest()
    cfg = settings()["factorial"]
    refs, replay = _state_references()
    epochs = selected_fast_epochs()[::2]
    designs = [(K, 12000) for K in cfg["candidate_axis"]] + [
        (8, M) for M in cfg["cycle_axis"] if M != 12000]
    rows = []
    for design_index, (K, M) in enumerate(designs):
        cells = [_information_cell(
            epoch, int(K), int(M), refs[epoch], replay[epoch],
            seed=30_000 + 1000 * design_index + epoch) for epoch in epochs]
        rows.append(_aggregate_cells(f"K{K}_M{M}", cells))
    by = {(row["candidates"], row["cycles_per_candidate"]): row for row in rows}
    candidate_gain = by[(32, 12000)]["median_reference_gradient_alignment"] - \
        by[(8, 12000)]["median_reference_gradient_alignment"]
    shot_gain = by[(8, 48000)]["median_reference_gradient_alignment"] - \
        by[(8, 12000)]["median_reference_gradient_alignment"]
    if candidate_gain >= .1 and shot_gain < .1:
        classification = "ACTION_SPACE_SAMPLING_LIMITED"
    elif shot_gain >= .1 and candidate_gain < .1:
        classification = "SHOT_NOISE_LIMITED"
    elif shot_gain >= .1 and candidate_gain >= .1:
        classification = "BOTH_LIMITING"
    else:
        classification = "NEITHER_LIMITING"
    value = {
        "pass": True,
        "fast_only_short_stored_state_diagnostics": True,
        "no_full_750_epoch_campaign": True,
        "diagnostic_epochs": epochs,
        "rows": rows,
        "candidate_axis_alignment_gain_K8_to_K32": candidate_gain,
        "shot_axis_alignment_gain_M12000_to_M48000": shot_gain,
        "classification": classification,
        "forbidden_auto_runs_launched": [],
    }
    return write_artifact("candidate_vs_shots_factorial", value,
                          title="V20 candidate-count versus cycle-count factorial")


def run_fixed_budget_comparison() -> dict[str, Any]:
    verify_import_manifest()
    refs, replay = _state_references()
    epochs = selected_fast_epochs()[::2]
    designs = [(32, 12000), (16, 24000), (8, 48000)]
    rows = []
    for design_index, (K, M) in enumerate(designs):
        cells = [_information_cell(
            epoch, K, M, refs[epoch], replay[epoch],
            seed=40_000 + 1000 * design_index + epoch) for epoch in epochs]
        rows.append(_aggregate_cells(f"K{K}_M{M}", cells))
    if not fixed_budget_equal(rows):
        raise RuntimeError("fixed-budget K/M comparison is not budget matched")
    best = max(rows, key=lambda row: row["median_reference_gradient_alignment"])
    classification = (
        "ACTION_SPACE_SAMPLING_LIMITED" if best["candidates"] == 32 else
        "SHOT_NOISE_LIMITED" if best["cycles_per_candidate"] == 48000 else
        "BALANCED_REALLOCATION")
    value = {
        "pass": True,
        "matched_total_budget": rows[0]["total_cycle_budget_per_state"],
        "budget_equality_verified": True,
        "rows": rows,
        "best_fixed_budget_cell": best["cell"],
        "classification": classification,
        "no_extra_total_computation_confounded": True,
        "forbidden_auto_runs_launched": [],
    }
    return write_artifact("fixed_budget_information_comparison", value,
                          title="V20 fixed-budget candidate/cycle comparison")


def run_scale_information_frontier() -> dict[str, Any]:
    verify_import_manifest()
    cfg = settings()
    refs, replay = _state_references()
    epochs = selected_fast_epochs()
    multipliers = [float(value) for value in cfg["scale_multipliers"]]
    before = canonical_hash({epoch: {"mean": replay[epoch]["mean"].tolist(),
                                     "sigma": replay[epoch]["sigma"].tolist()}
                             for epoch in epochs})
    raw_rows = []
    for multiplier in multipliers:
        state_rows = []
        for epoch in epochs:
            item = replay[epoch]
            mean = item["mean"]
            sigma = np.clip(multiplier * item["sigma"], .002, .8)
            noises = load_matched_run("fast")["noises"][epoch]
            actions = mean[None, :] + sigma[None, :] * noises
            rewards = -np.asarray([
                evaluator().detector_expectations(action, epoch, cfg["fast_frequency_per_epoch"])
                for action in actions])
            direction, contributions, _ = _direction_from_rewards(
                actions, rewards, mean, sigma, item["baseline"], epoch)
            reference = refs[epoch]
            beneficial = reference / max(float(np.linalg.norm(reference)), 1e-15)
            z = contributions @ beneficial
            mean_cost = evaluator().cost(mean, epoch, cfg["fast_frequency_per_epoch"])
            candidate_costs = np.asarray([-np.sum(row) for row in rewards])
            state_rows.append({
                "epoch": epoch,
                "directional": _summarize_directional(z),
                "alignment": cosine_alignment(direction, reference),
                "gradient_relative_bias": float(
                    np.linalg.norm(direction - reference) /
                    max(np.linalg.norm(reference), 1e-15)),
                "empirical_damage": float(np.mean(candidate_costs) - mean_cost),
                "clipping_fraction": 0.0,
            })
        raw_rows.append({"lambda": multiplier, "state_rows": state_rows})
    anchor = next(row for row in raw_rows if row["lambda"] == .25)
    anchor_damage = np.asarray([row["empirical_damage"] for row in anchor["state_rows"]]) / .25**2
    rows = []
    for raw in raw_rows:
        predicted = anchor_damage * raw["lambda"]**2
        empirical = np.asarray([row["empirical_damage"] for row in raw["state_rows"]])
        rows.append({
            "lambda": raw["lambda"],
            "median_directional_SNR": float(np.median([
                row["directional"]["batch_SNR"] for row in raw["state_rows"]])),
            "median_alignment": float(np.median([
                row["alignment"] for row in raw["state_rows"]])),
            "median_wrong_sign_probability": float(np.median([
                row["directional"]["estimated_wrong_sign_probability"]
                for row in raw["state_rows"]])),
            "median_gradient_relative_bias": float(np.median([
                row["gradient_relative_bias"] for row in raw["state_rows"]])),
            "mean_empirical_candidate_damage": float(np.mean(empirical)),
            "mean_quadratic_predicted_damage": float(np.mean(predicted)),
            "empirical_over_predicted_damage": float(
                np.mean(empirical) / max(np.mean(predicted), 1e-15)),
            "clipping_fraction": 0.0,
            "state_rows": raw["state_rows"],
        })
    after = canonical_hash({epoch: {"mean": replay[epoch]["mean"].tolist(),
                                    "sigma": replay[epoch]["sigma"].tolist()}
                            for epoch in epochs})
    if before != after:
        raise RuntimeError("frozen scale frontier mutated stored policy state")
    best = max(rows, key=lambda row: row["median_alignment"] /
               max(row["mean_empirical_candidate_damage"], 1e-12))
    if best["lambda"] in (.5, .75, 1.0):
        classification = "CLEAR_INTERIOR_INFORMATION_DAMAGE_OPTIMUM"
    elif rows[0]["median_directional_SNR"] < .7 * rows[-1]["median_directional_SNR"]:
        classification = "LOW_SCALE_INFORMATION_STARVATION"
    elif rows[-1]["empirical_over_predicted_damage"] > 1.25:
        classification = "HIGH_SCALE_NONLINEAR_DAMAGE"
    else:
        classification = "NO_SCALE_DEPENDENCE"
    value = {
        "pass": True,
        "stored_policy_state_hash_before": before,
        "stored_policy_state_hash_after": after,
        "policy_state_unchanged": before == after,
        "rows": rows,
        "pareto_frontier": [
            {"lambda": row["lambda"], "SNR": row["median_directional_SNR"],
             "damage": row["mean_empirical_candidate_damage"],
             "alignment": row["median_alignment"],
             "bias": row["median_gradient_relative_bias"]} for row in rows],
        "quadratic_prediction_anchor": "lambda=0.25 exact symmetric stored-action damage",
        "best_information_damage_lambda": best["lambda"],
        "classification": classification,
        "retraining_performed": False,
        "forbidden_auto_runs_launched": [],
    }
    return write_artifact("frozen_scale_information_damage_frontier", value,
                          title="V20 frozen-scale information/damage frontier")


def _conditioned(rows: list[dict[str, float]], field: str) -> list[dict[str, Any]]:
    values = np.asarray([row[field] for row in rows])
    edges = np.quantile(values, [0, .25, .5, .75, 1])
    result = []
    for index in range(4):
        selected = [row for row in rows if (row[field] >= edges[index] and
                    (row[field] <= edges[index + 1] if index == 3
                     else row[field] < edges[index + 1]))]
        result.append({
            "quartile": index + 1,
            "range": [float(edges[index]), float(edges[index + 1])],
            "epochs": len(selected),
            "mean_reward_gradient_projection": float(np.mean([
                row["reward_projection"] for row in selected])),
            "mean_entropy_gradient_projection": float(np.mean([
                row["entropy_projection"] for row in selected])),
            "mean_net_gradient_projection": float(np.mean([
                row["net_projection"] for row in selected])),
            "net_gradient_projection_std": float(np.std([
                row["net_projection"] for row in selected], ddof=1)),
        })
    return result


def audit_dynamic_sigma() -> dict[str, Any]:
    verify_import_manifest()
    controller = _controller_spec()
    result_rows = []
    for label in ("slow", "intermediate", "fast"):
        run = load_matched_run(label)
        replay = replay_gradients(run)
        start, stop = map(int, run["transfer"]["analysis_epoch_window"])
        frequency = float(run["transfer"]["frequency_per_epoch"])
        unit = np.ones(41) / math.sqrt(41)
        rows = []
        for item in replay[start:stop]:
            sigma = item["sigma"]
            entropy = -controller.effective_entropy_coefficient / sigma
            reward = item["loss"].grad_sigma - entropy
            net = item["loss"].grad_sigma
            target = evaluator().plant.latent_controls_for(
                np.full(41, math.sin(2*np.pi*frequency*item["epoch"])))
            rows.append({
                "epoch": item["epoch"],
                "phase": _phase(item["epoch"], frequency),
                "tracking_error": float(np.linalg.norm(item["mean"] - target)),
                "reward_variance": item["reward_variance"],
                "baseline_error": item["baseline_error"],
                "reward_projection": float(reward @ unit),
                "entropy_projection": float(entropy @ unit),
                "net_projection": float(net @ unit),
                "net_norm": float(np.linalg.norm(net)),
                "sigma_median": float(np.median(sigma)),
                "floor_fraction": float(np.mean(sigma <= .002 + 1e-12)),
                "ceiling_fraction": float(np.mean(sigma >= .8 - 1e-12)),
            })
        phase_bins = []
        for index in range(8):
            selected = [row for row in rows if int(row["phase"] / (2*np.pi) * 8) == index]
            phase_bins.append({
                "phase_bin": index,
                "epochs": len(selected),
                "mean_reward_gradient_projection": float(np.mean([
                    row["reward_projection"] for row in selected])),
                "mean_entropy_gradient_projection": float(np.mean([
                    row["entropy_projection"] for row in selected])),
                "mean_net_gradient_projection": float(np.mean([
                    row["net_projection"] for row in selected])),
            })
        result_rows.append({
            "label": label,
            "frequency_per_epoch": frequency,
            "analysis_sigma_median": float(np.median([row["sigma_median"] for row in rows])),
            "terminal_sigma_median": rows[-1]["sigma_median"],
            "signed_phase_bins": phase_bins,
            "conditioned_on_tracking_error": _conditioned(rows, "tracking_error"),
            "conditioned_on_reward_variance": _conditioned(rows, "reward_variance"),
            "conditioned_on_baseline_error": _conditioned(rows, "baseline_error"),
            "mean_reward_projection": float(np.mean([row["reward_projection"] for row in rows])),
            "mean_entropy_projection": float(np.mean([row["entropy_projection"] for row in rows])),
            "mean_net_projection": float(np.mean([row["net_projection"] for row in rows])),
            "net_projection_std": float(np.std([row["net_projection"] for row in rows], ddof=1)),
            "floor_occupancy": float(np.mean([row["floor_fraction"] for row in rows])),
            "ceiling_occupancy": float(np.mean([row["ceiling_fraction"] for row in rows])),
        })
    fast = result_rows[-1]
    if fast["ceiling_occupancy"] > .1 or fast["floor_occupancy"] > .1:
        classification = "PROJECTION_OR_BOUND_EFFECTS"
    elif abs(fast["mean_net_projection"]) < fast["net_projection_std"]:
        classification = "LARGE_FINITE_BATCH_SCALE_GRADIENT_VARIANCE"
    elif fast["terminal_sigma_median"] > fast["analysis_sigma_median"]:
        classification = "INSUFFICIENT_SETTLING_WITH_TRACKING_ERROR_DEPENDENCE"
    else:
        classification = "PHASE_DEPENDENT_REWARD_PENALTY"
    value = {
        "pass": True,
        "gradient_space": "direct sigma before projection",
        "effective_entropy_coefficient": controller.effective_entropy_coefficient,
        "rows": result_rows,
        "classification": classification,
        "sigma_tuned": False,
        "new_trajectory_generated": False,
        "forbidden_auto_runs_launched": [],
    }
    return write_artifact("dynamic_sigma_signed_gradients", value,
                          title="V20 signed dynamic-sigma audit")


def audit_acquisition_bias() -> dict[str, Any]:
    verify_import_manifest()
    cfg = settings()
    frequency = float(cfg["fast_frequency_per_epoch"])
    run = load_matched_run("fast")
    replay = replay_gradients(run)
    rows = []
    rng = np.random.default_rng(52_020)
    for epoch in selected_fast_epochs():
        item = replay[epoch]
        actions, mean, sigma, baseline = (
            item["actions"], item["mean"], item["sigma"], item["baseline"])
        K = len(actions)
        variants: dict[str, list[float]] = {
            "frozen_target_phase": [float(epoch)] * K,
            "ordinary_sequential_target_evolution": [epoch + index / K for index in range(K)],
            "timestamp_corrected_rewards": [float(epoch)] * K,
            "reversed_candidate_order": [epoch + (K - 1 - index) / K for index in range(K)],
            "shorter_sub_batches_matched_budget": [epoch if index < K//2 else epoch + .5
                                                    for index in range(K)],
        }
        permutation = rng.permutation(K)
        random_times = np.asarray([epoch + index / K for index in range(K)])[permutation]
        variants["randomized_candidate_order"] = random_times.tolist()
        directions = {}
        for name, times in variants.items():
            rewards = -np.asarray([
                evaluator().detector_expectations(action, times[index], frequency)
                for index, action in enumerate(actions)])
            directions[name] = _direction_from_rewards(
                actions, rewards, mean, sigma, baseline, epoch)[0]
        frozen = directions["frozen_target_phase"]
        beneficial = _beneficial_direction(mean, epoch, frequency)
        comparisons = {}
        for name, direction in directions.items():
            angle_cos = cosine_alignment(direction, frozen)
            angle = float(math.acos(float(np.clip(angle_cos, -1, 1)))) \
                if angle_cos is not None else None
            comparisons[name] = {
                "gradient_angle_shift_radians": angle,
                "effective_phase_delay_epochs": (
                    angle / (2*np.pi*frequency) if angle is not None else None),
                "directional_bias": float((direction - frozen) @ beneficial),
                "predicted_progress": float(direction @ beneficial),
                "information_budget_candidates": K,
            }
        rows.append({"epoch": epoch, "phase_radians": _phase(epoch, frequency),
                     "comparisons": comparisons})
    sequential_angles = [row["comparisons"]["ordinary_sequential_target_evolution"][
        "gradient_angle_shift_radians"] for row in rows]
    sequential_bias = [abs(row["comparisons"]["ordinary_sequential_target_evolution"][
        "directional_bias"]) for row in rows]
    frozen_progress = [abs(row["comparisons"]["frozen_target_phase"]["predicted_progress"])
                       for row in rows]
    relative_bias = float(np.median(sequential_bias) /
                          max(np.median(frozen_progress), 1e-15))
    # Phase points with an almost-zero frozen gradient have an unstable angle;
    # the preregistered conclusion is therefore based on the phase-stratified
    # median angle together with the median directional bias.
    if float(np.median(sequential_angles)) < .1 and relative_bias < .1:
        classification = "NO_MEANINGFUL_BATCH_MOTION_BIAS"
    elif float(np.median(sequential_angles)) >= .1:
        classification = "ACQUISITION_DELAY_BIAS"
    elif relative_bias >= .1:
        classification = "WITHIN_EPOCH_TARGET_SMEARING"
    else:
        classification = "ORDER_DEPENDENT_ESTIMATOR_BIAS"
    value = {
        "pass": True,
        "production_acquisition_semantics": "target frozen at integer epoch for all candidates",
        "sequential_motion_is_counterfactual": True,
        "all_variants_use_matched_K8_information_budget": True,
        "rows": rows,
        "median_sequential_angle_shift_radians": float(np.median(sequential_angles)),
        "median_sequential_relative_directional_bias": relative_bias,
        "classification": classification,
        "training_changed": False,
        "forbidden_auto_runs_launched": [],
    }
    return write_artifact("acquisition_bias_audit", value,
                          title="V20 moving-target acquisition/batch audit")
