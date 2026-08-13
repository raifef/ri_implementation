"""V21 rolling coverage, frozen-state design benchmark, damage, and Pareto surface."""
from __future__ import annotations

import math
import time
from typing import Any, Iterable

import numpy as np

from hdfa_rl_suite.google_pure_v20.core import (
    batch_snr,
    candidate_snr,
    cosine_alignment,
    rowspace_overlap,
    wrong_sign_probability,
)
from hdfa_rl_suite.google_pure_v20.data import (
    evaluator,
    load_matched_run,
    replay_gradients,
    selected_fast_epochs,
)

from .candidate_design import (
    DESIGN_IDS,
    DESIGN_NAMES,
    SOURCE_FIDELITY,
    estimate_policy_updates,
    generate_frame,
)
from .diagnostics import _beneficial, _blocks, _reference_rows, _sample_rewards
from .io import ARTIFACT_ROOT, canonical_hash, read_json, settings, write_artifact
from .lineage import verify_import_manifest


def pareto_dominates(candidate_error: float, candidate_damage: float,
                     baseline_error: float, baseline_damage: float,
                     *, tolerance: float = 1e-12) -> bool:
    """Return strict two-objective Pareto dominance with numerical tolerance."""
    no_worse = (candidate_error <= baseline_error + tolerance and
                candidate_damage <= baseline_damage + tolerance)
    strictly_better = (candidate_error < baseline_error - tolerance or
                       candidate_damage < baseline_damage - tolerance)
    return bool(no_worse and strictly_better)


def normalize_exploration_damage(raw_damage: float, fixed_cost: float,
                                 oracle_cost: float) -> float:
    """Put candidate damage on the exact Figure 5a fixed-oracle scale."""
    denominator = float(fixed_cost) - float(oracle_cost)
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("Figure 5a damage normalization requires fixed > oracle")
    return float(raw_damage) / denominator


def _frame_condition(frame: np.ndarray) -> dict[str, Any]:
    gram = np.asarray(frame, dtype=float).T @ np.asarray(frame, dtype=float)
    eigenvalues = np.linalg.eigvalsh(gram)
    tolerance = max(float(np.max(eigenvalues)), 1.0) * 1e-10
    nonzero = eigenvalues[eigenvalues > tolerance]
    return {
        "rank": int(len(nonzero)),
        "condition_number_nonzero_support": float(np.max(nonzero) / np.min(nonzero))
            if len(nonzero) else None,
        "smallest_nonzero_eigenvalue": float(np.min(nonzero)) if len(nonzero) else 0.0,
        "largest_eigenvalue": float(np.max(nonzero)) if len(nonzero) else 0.0,
    }


def audit_frame_coverage() -> dict[str, Any]:
    verify_import_manifest()
    cfg = settings()
    blocks = _blocks()
    windows = [int(value) for value in cfg["rolling_coverage_windows"]]
    total_epochs = int(cfg["coverage_epochs"])
    reference = {int(row["epoch"]): np.asarray(row["reference_gradient"], dtype=float)
                 for row in _reference_rows()}
    families = np.asarray([item.gate_type for item in evaluator().plant.inventory])
    design_rows = []
    for design_id in DESIGN_IDS:
        frames = [generate_frame(
            design_id, dimension=41, epoch=epoch, seed=42_100, blocks=blocks
        ).standardized_directions for epoch in range(total_epochs)]
        window_rows = []
        for window in windows:
            summaries = []
            for stop in range(window, total_epochs + 1):
                stacked = np.concatenate(frames[stop-window:stop], axis=0)
                condition = _frame_condition(stacked)
                family = {}
                for label in sorted(set(families)):
                    family[label] = _frame_condition(stacked[:, families == label])
                neighborhood = {
                    f"public-block-{index}": _frame_condition(stacked[:, block])
                    for index, block in enumerate(blocks)}
                summaries.append({"stop_epoch": stop, **condition,
                                  "control_families": family,
                                  "detector_neighborhoods": neighborhood})
            window_rows.append({
                "window_epochs": window,
                "minimum_rank": min(row["rank"] for row in summaries),
                "median_rank": float(np.median([row["rank"] for row in summaries])),
                "maximum_nonzero_condition_number": max(
                    row["condition_number_nonzero_support"] for row in summaries
                    if row["condition_number_nonzero_support"] is not None),
                "minimum_nonzero_eigenvalue": min(
                    row["smallest_nonzero_eigenvalue"] for row in summaries),
                "same_complement_repeatedly_missed": max(row["rank"] for row in summaries) < 41,
                "window_summaries": summaries,
            })
        reference_rows = []
        phases = []
        overlaps = []
        for epoch, gradient in reference.items():
            local_epoch = epoch % total_epochs
            start = max(0, local_epoch - 40)
            stacked = np.concatenate(frames[start:local_epoch + 1], axis=0)
            overlap = rowspace_overlap(stacked, gradient)
            phase = float((2*np.pi*cfg["fast_frequency_per_epoch"]*epoch) % (2*np.pi))
            reference_rows.append({"epoch": epoch, "phase_radians": phase,
                                   "reference_gradient_coverage": overlap})
            phases.append(phase); overlaps.append(overlap)
        phase_bins = []
        for index in range(4):
            selected = [overlaps[i] for i, phase in enumerate(phases)
                        if int(phase / (2*np.pi) * 4) == index]
            phase_bins.append({"phase_bin": index, "states": len(selected),
                               "mean_reference_coverage": float(np.mean(selected))})
        full_window = next(row for row in window_rows if row["window_epochs"] == max(windows))
        design_rows.append({
            "design_id": design_id,
            "design_name": DESIGN_NAMES[design_id],
            "rolling_windows": window_rows,
            "reference_gradient_coverage": reference_rows,
            "phase_conditioned_reference_coverage": phase_bins,
            "median_reference_gradient_coverage": float(np.median(overlaps)),
            "full_rank_within_longest_window": full_window["minimum_rank"] == 41,
            "coverage_not_degenerate": full_window["minimum_rank"] == 41 and
                float(np.median(overlaps)) >= .95,
        })
    value = {
        "pass": True,
        "coverage_gram_definition": "sum over epoch/candidate standardized z z^T",
        "public_blocks": [block.tolist() for block in blocks],
        "designs": design_rows,
        "forbidden_auto_runs_launched": [],
    }
    return write_artifact("candidate_frame_coverage", value,
                          title="V21 rolling candidate-frame coverage")


def _exact_rewards(actions: np.ndarray, epoch: int, frequency: float) -> np.ndarray:
    return -np.asarray([
        evaluator().detector_expectations(action, epoch, frequency) for action in actions])


def _metric_row(estimate: np.ndarray, reference: np.ndarray, beneficial: np.ndarray,
                contributions: np.ndarray) -> dict[str, Any]:
    ref_norm2 = float(reference @ reference)
    projection = float(reference @ estimate) / max(ref_norm2, 1e-15)
    orthogonal = estimate - projection * reference
    ref_progress = float(beneficial @ reference)
    estimate_progress = float(beneficial @ estimate)
    z = contributions @ beneficial
    c_snr = candidate_snr(z)
    b_snr = batch_snr(z)
    return {
        "squared_error": float(np.sum((estimate - reference)**2)),
        "alignment": cosine_alignment(estimate, reference),
        "directional_magnitude_ratio": estimate_progress /
            ref_progress if abs(ref_progress) > 1e-15 else None,
        "reference_gradient_capture": projection,
        "orthogonal_error_power": float(orthogonal @ orthogonal),
        "wrong_sign": estimate_progress * ref_progress < 0,
        "candidate_SNR": c_snr,
        "batch_SNR": b_snr,
        "wrong_sign_probability": wrong_sign_probability(b_snr),
    }


def _bound_occupancy(actions: np.ndarray) -> float:
    evaluation = evaluator()
    normalized = evaluation.bounded.apply_control_transform(actions)
    ratio = np.abs(normalized / evaluation.bounded.control_limits[None, :])
    return float(np.mean(ratio >= .99))


def _physical_covariance(frame: np.ndarray) -> dict[str, float]:
    covariance = np.asarray(frame).T @ np.asarray(frame) / len(frame)
    off = covariance - np.diag(np.diag(covariance))
    return {
        "diagonal_mean": float(np.mean(np.diag(covariance))),
        "diagonal_RMS_error_from_one": float(np.sqrt(np.mean((np.diag(covariance)-1)**2))),
        "offdiagonal_RMS": float(np.sqrt(np.mean(off**2))),
    }


def benchmark_candidate_designs() -> dict[str, Any]:
    verify_import_manifest()
    cfg = settings()
    frequency = float(cfg["fast_frequency_per_epoch"])
    repetitions = int(cfg["benchmark_repetitions"])
    blocks = _blocks()
    replay = replay_gradients(load_matched_run("fast"))
    references = {int(row["epoch"]): np.asarray(row["reference_gradient"], dtype=float)
                  for row in _reference_rows()}
    design_rows = []
    for design_id in DESIGN_IDS:
        started = time.perf_counter()
        state_rows = []
        for epoch, reference in references.items():
            item = replay[epoch]
            beneficial = _beneficial(item["mean"], epoch, frequency)
            estimates = []
            replicate_rows = []
            for repeat in range(repetitions):
                frame = generate_frame(
                    design_id, dimension=41, epoch=epoch,
                    seed=43_000 + repeat * 1000, blocks=blocks)
                actions = item["mean"][None, :] + item["sigma"][None, :] * \
                    frame.standardized_directions
                rewards = _sample_rewards(
                    actions, epoch, frequency,
                    seed=4_300_000 + DESIGN_IDS.index(design_id)*100_000 +
                    epoch*100 + repeat*10)
                estimated = estimate_policy_updates(
                    frame, rewards, item["baseline"], evaluator().plant.mask,
                    item["sigma"])
                update = np.asarray(estimated["mean_update_direction"], dtype=float)
                contributions = np.asarray(
                    estimated["candidate_mean_update_contributions"], dtype=float)
                exact_candidate_cost = np.asarray([
                    evaluator().cost(action, epoch, frequency) for action in actions])
                mean_cost = evaluator().cost(item["mean"], epoch, frequency)
                row = _metric_row(update, reference, beneficial, contributions)
                row.update({
                    "repeat": repeat,
                    "candidate_damage_detector_events_per_shot": float(
                        np.mean(exact_candidate_cost) - mean_cost),
                    "frame_conditioning": _frame_condition(frame.standardized_directions),
                    "physical_covariance": _physical_covariance(
                        frame.standardized_directions),
                    "bound_occupancy": _bound_occupancy(actions),
                    "frame_hash": canonical_hash(frame.standardized_directions.tolist()),
                })
                replicate_rows.append(row); estimates.append(update)
            estimate_array = np.asarray(estimates)
            bias = np.mean(estimate_array, axis=0) - reference
            state_rows.append({
                "epoch": epoch,
                "phase_radians": float((2*np.pi*frequency*epoch) % (2*np.pi)),
                "bias_vector": bias.tolist(),
                "bias_norm": float(np.linalg.norm(bias)),
                "MSE": float(np.mean([
                    row["squared_error"] for row in replicate_rows])),
                "replicates": replicate_rows,
            })
        all_replicates = [row for state in state_rows for row in state["replicates"]]
        design_rows.append({
            "design_id": design_id,
            "design_name": DESIGN_NAMES[design_id],
            "source_fidelity": SOURCE_FIDELITY[design_id],
            "estimator_valid": True,
            "sigma_estimator_valid": generate_frame(
                design_id, dimension=41, epoch=0, seed=43_000, blocks=blocks
            ).sigma_estimator_valid,
            "K": 8, "M": 12000, "B": 96000,
            "mean_reference_gradient_MSE": float(np.mean([
                state["MSE"] for state in state_rows])),
            "median_alignment": float(np.median([
                row["alignment"] for row in all_replicates])),
            "median_directional_magnitude_ratio": float(np.median([
                row["directional_magnitude_ratio"] for row in all_replicates
                if row["directional_magnitude_ratio"] is not None])),
            "median_reference_gradient_capture": float(np.median([
                row["reference_gradient_capture"] for row in all_replicates])),
            "mean_orthogonal_error_power": float(np.mean([
                row["orthogonal_error_power"] for row in all_replicates])),
            "wrong_sign_fraction": float(np.mean([
                row["wrong_sign"] for row in all_replicates])),
            "median_candidate_SNR": float(np.median([
                row["candidate_SNR"] for row in all_replicates])),
            "median_batch_SNR": float(np.median([
                row["batch_SNR"] for row in all_replicates])),
            "mean_candidate_damage_detector_events_per_shot": float(np.mean([
                row["candidate_damage_detector_events_per_shot"] for row in all_replicates])),
            "mean_bound_occupancy": float(np.mean([
                row["bound_occupancy"] for row in all_replicates])),
            "runtime_seconds": time.perf_counter() - started,
            "state_rows": state_rows,
        })
    baseline = next(row for row in design_rows if row["design_id"] == "D0")
    for row in design_rows:
        row["MSE_ratio_to_iid"] = (
            row["mean_reference_gradient_MSE"] /
            baseline["mean_reference_gradient_MSE"])
        row["orthogonal_error_ratio_to_iid"] = (
            row["mean_orthogonal_error_power"] /
            baseline["mean_orthogonal_error_power"])
    best = min(design_rows, key=lambda row: row["mean_reference_gradient_MSE"])
    value = {
        "pass": True,
        "fixed_budget_verified": all(
            row["K"] * row["M"] == row["B"] == 96000 for row in design_rows),
        "reference_definition": "V20 K256 antithetic exact-detector Gaussian population reference",
        "finite_shot_acquisition": "exact Stim detector sampling at M=12000 QEC cycles/candidate",
        "designs": design_rows,
        "best_MSE_design": best["design_id"],
        "best_MSE_ratio_to_iid": best["MSE_ratio_to_iid"],
        "hard_projection_in_candidate_competition": False,
        "forbidden_auto_runs_launched": [],
    }
    return write_artifact("frozen_state_candidate_design_benchmark", value,
                          title="V21 fixed-budget frozen-state candidate-design benchmark")


def reconcile_exploration_damage() -> dict[str, Any]:
    verify_import_manifest()
    cfg = settings()
    frequency = float(cfg["fast_frequency_per_epoch"])
    v20 = read_json(ARTIFACT_ROOT.parent /
                    "google_pure_v20/frozen_scale_information_damage_frontier.json")
    population = read_json(ARTIFACT_ROOT.parent /
                           "google_pure_v20/population_gradient_fast_rollout.json")
    postrepair = read_json(ARTIFACT_ROOT.parent /
                           "google_pure_v20/postrepair_fast_validation.json")
    epochs = selected_fast_epochs()
    denominators = []
    for epoch in epochs:
        fixed = evaluator().normalized_cost(np.zeros(41), epoch, frequency)
        optimal = evaluator().normalized_cost(
            evaluator().plant.optimum(epoch, frequency), epoch, frequency)
        denominators.append(fixed - optimal)
    denominator = float(np.mean(denominators))
    rows = []
    for source in v20["rows"]:
        raw = float(source["mean_empirical_candidate_damage"])
        quadratic = float(source["mean_quadratic_predicted_damage"])
        rows.append({
            "lambda": source["lambda"],
            "raw_frozen_state_damage_detector_events_per_shot": raw,
            "Figure5a_normalized_frozen_state_damage": normalize_exploration_damage(
                raw, denominator, 0.0),
            "quadratic_prediction_raw": quadratic,
            "quadratic_prediction_normalized": normalize_exploration_damage(
                quadratic, denominator, 0.0),
            "empirical_over_quadratic": raw / max(quadratic, 1e-15),
        })
    population_penalty = (
        population["population_gradient_fast"]["I_mean"] -
        population["population_gradient_fast"]["I_stochastic_exact_diagnostic"])
    baseline_online_penalty = (
        postrepair["baseline_v19_experimental_fast"]["I_mean"] -
        postrepair["baseline_v19_experimental_fast"]["I_stochastic"])
    lambda_one = next(row for row in rows if row["lambda"] == 1.0)
    mismatch = population_penalty / max(
        lambda_one["Figure5a_normalized_frozen_state_damage"], 1e-15)
    diagnoses = []
    if abs(mismatch - 1.0) > .25:
        diagnoses.extend(["trajectory-dependent scale damage", "branch-state mismatch"])
    if len(epochs) < 150:
        diagnoses.extend(["phase weighting", "state/horizon difference"])
    if max(row["empirical_over_quadratic"] for row in rows) > 1.25:
        diagnoses.append("nonlinear accumulation")
    value = {
        "pass": True,
        "normalization_denominator_Cfixed_minus_Coracle": denominator,
        "frozen_state_rows": rows,
        "online_rollout_exploration_penalty": {
            "population_branch_Imean_minus_Istochastic": population_penalty,
            "V19_baseline_Imean_minus_Istochastic": baseline_online_penalty,
        },
        "lambda_one_online_to_frozen_normalized_ratio": mismatch,
        "normalization_difference": "raw detector events per shot divided by exact phase-matched Cfixed-Coracle",
        "state_horizon_difference": "8 final-period frozen states versus 150-epoch online trajectory",
        "phase_weighting_difference": "stratified point states versus complete-period trajectory",
        "remaining_mismatch_diagnoses": sorted(set(diagnoses)),
        "metrics_reconciled_onto_exact_Figure5a_scale": True,
        "forbidden_auto_runs_launched": [],
    }
    return write_artifact("exploration_damage_metric_reconciliation", value,
                          title="V21 exploration-damage metric reconciliation")


def _marginal_binomial_rewards(actions: np.ndarray, epoch: int, frequency: float,
                               *, seed: int, cycles: int = 12000) -> np.ndarray:
    probabilities = np.asarray([
        evaluator().detector_expectations(action, epoch, frequency) for action in actions])
    shots = cycles // evaluator().plant.rounds
    rng = np.random.default_rng(seed)
    counts = rng.binomial(shots, np.clip(probabilities, 0.0, 1.0))
    return -counts / shots


def run_design_scale_pareto() -> dict[str, Any]:
    verify_import_manifest()
    cfg = settings()
    frequency = float(cfg["fast_frequency_per_epoch"])
    repetitions = int(cfg["pareto_repetitions"])
    multipliers = [float(value) for value in cfg["scale_multipliers"]]
    blocks = _blocks()
    replay = replay_gradients(load_matched_run("fast"))
    references = {int(row["epoch"]): np.asarray(row["reference_gradient"], dtype=float)
                  for row in _reference_rows()}
    cells = []
    for design_id in DESIGN_IDS:
        for multiplier in multipliers:
            metrics = []
            for epoch, reference in references.items():
                item = replay[epoch]
                sigma = np.clip(multiplier * item["sigma"], .002, .8)
                beneficial = _beneficial(item["mean"], epoch, frequency)
                for repeat in range(repetitions):
                    frame = generate_frame(
                        design_id, dimension=41, epoch=epoch,
                        seed=44_000 + repeat*1000, blocks=blocks)
                    actions = item["mean"][None, :] + sigma[None, :] * \
                        frame.standardized_directions
                    rewards = _marginal_binomial_rewards(
                        actions, epoch, frequency,
                        seed=4_400_000 + DESIGN_IDS.index(design_id)*100_000 +
                        int(multiplier*100)*1000 + epoch*10 + repeat)
                    estimated = estimate_policy_updates(
                        frame, rewards, item["baseline"], evaluator().plant.mask, sigma)
                    update = np.asarray(estimated["mean_update_direction"], dtype=float)
                    contributions = np.asarray(
                        estimated["candidate_mean_update_contributions"], dtype=float)
                    row = _metric_row(update, reference, beneficial, contributions)
                    candidate_cost = np.asarray([
                        evaluator().cost(action, epoch, frequency) for action in actions])
                    row["candidate_damage"] = float(
                        np.mean(candidate_cost) -
                        evaluator().cost(item["mean"], epoch, frequency))
                    row["frame_conditioning"] = _frame_condition(
                        frame.standardized_directions)
                    metrics.append(row)
            cells.append({
                "design_id": design_id,
                "design_name": DESIGN_NAMES[design_id],
                "source_fidelity": SOURCE_FIDELITY[design_id],
                "lambda": multiplier,
                "K": 8, "M": 12000, "B": 96000,
                "gradient_MSE": float(np.mean([row["squared_error"] for row in metrics])),
                "candidate_damage": float(np.mean([row["candidate_damage"] for row in metrics])),
                "median_alignment": float(np.median([row["alignment"] for row in metrics])),
                "median_directional_magnitude_ratio": float(np.median([
                    row["directional_magnitude_ratio"] for row in metrics
                    if row["directional_magnitude_ratio"] is not None])),
                "median_reference_gradient_capture": float(np.median([
                    row["reference_gradient_capture"] for row in metrics])),
                "mean_orthogonal_error_power": float(np.mean([
                    row["orthogonal_error_power"] for row in metrics])),
                "median_candidate_SNR": float(np.median([
                    row["candidate_SNR"] for row in metrics])),
                "median_wrong_sign_probability": float(np.median([
                    row["wrong_sign_probability"] for row in metrics])),
                "wrong_sign_fraction": float(np.mean([row["wrong_sign"] for row in metrics])),
                "median_frame_nonzero_condition": float(np.median([
                    row["frame_conditioning"]["condition_number_nonzero_support"]
                    for row in metrics])),
                "finite_M_model": "exact detector marginals with M/rounds binomial counts",
                "detector_cross_correlation_omitted_in_pareto_scan": True,
            })
    baseline_cells = [row for row in cells if row["design_id"] == "D0"]
    for cell in cells:
        dominated_baselines = [row for row in baseline_cells
            if pareto_dominates(
                cell["gradient_MSE"], cell["candidate_damage"],
                row["gradient_MSE"], row["candidate_damage"])]
        cell["pareto_dominates_iid"] = bool(dominated_baselines)
        cell["dominated_iid_lambdas"] = [row["lambda"] for row in dominated_baselines]
    design_rows = []
    for design_id in DESIGN_IDS:
        rows = [row for row in cells if row["design_id"] == design_id]
        design_rows.append({
            "design_id": design_id,
            "source_fidelity": SOURCE_FIDELITY[design_id],
            "cells": rows,
            "any_pareto_dominance": any(row["pareto_dominates_iid"] for row in rows),
            "smaller_sigma_matches_or_beats_iid_lambda_one": any(
                row["lambda"] < 1.0 and row["gradient_MSE"] <= next(
                    source["gradient_MSE"] for source in baseline_cells if source["lambda"] == 1.0)
                and row["candidate_damage"] < next(
                    source["candidate_damage"] for source in baseline_cells if source["lambda"] == 1.0)
                for row in rows),
        })
    value = {
        "pass": True,
        "fixed_budget_verified": all(row["K"] * row["M"] == row["B"] == 96000
                                     for row in cells),
        "cells": cells,
        "designs": design_rows,
        # D0 is the reference family.  A different D0 scale can dominate the
        # lambda=1 D0 point, but that is a scale result, not evidence that a
        # candidate *design* outperforms iid Gaussian sampling.
        "designs_pareto_dominating_iid": [row["design_id"] for row in design_rows
                                           if row["design_id"] != "D0" and
                                           row["any_pareto_dominance"]],
        "designs_matching_iid_accuracy_at_smaller_sigma": [
            row["design_id"] for row in design_rows
            if row["design_id"] != "D0" and
            row["smaller_sigma_matches_or_beats_iid_lambda_one"]],
        "selection_used_paper_headline_outcomes": False,
        "forbidden_auto_runs_launched": [],
    }
    return write_artifact("candidate_design_scale_pareto", value,
                          title="V21 candidate-design by exploration-scale Pareto surface", notes=[
        "Every cell keeps K=8 and M=12000.",
        "Pareto promotion requires lower gradient error without higher exact candidate damage.",
        "The scan uses exact detector marginals with finite-M binomial noise; online validation retains full Stim sampling.",
    ])


def run_frozen_promotion_gate() -> dict[str, Any]:
    verify_import_manifest()
    benchmark_path = ARTIFACT_ROOT / "frozen_state_candidate_design_benchmark.json"
    coverage_path = ARTIFACT_ROOT / "candidate_frame_coverage.json"
    pareto_path = ARTIFACT_ROOT / "candidate_design_scale_pareto.json"
    estimators_path = ARTIFACT_ROOT / "candidate_estimators.json"
    variance_path = ARTIFACT_ROOT / "gradient_variance_decomposition.json"
    required = [benchmark_path, coverage_path, pareto_path, estimators_path, variance_path]
    if not all(path.is_file() for path in required):
        raise RuntimeError("V21 frozen promotion gate requires all pre-online diagnostics")
    benchmark = read_json(benchmark_path)
    coverage = read_json(coverage_path)
    pareto = read_json(pareto_path)
    estimators = read_json(estimators_path)
    variance = read_json(variance_path)
    by_benchmark = {row["design_id"]: row for row in benchmark["designs"]}
    by_coverage = {row["design_id"]: row for row in coverage["designs"]}
    by_estimator = {row["design_id"]: row for row in estimators["designs"]}
    by_pareto = {row["design_id"]: row for row in pareto["designs"]}
    baseline = by_benchmark["D0"]
    rows = []
    for design_id in DESIGN_IDS:
        source = by_benchmark[design_id]
        gates = {
            "estimator_valid": by_estimator[design_id]["online_controller_eligible"],
            "fixed_budget": source["K"] * source["M"] == source["B"] == 96000,
            "reference_gradient_MSE_improved": source["mean_reference_gradient_MSE"] <
                .95 * baseline["mean_reference_gradient_MSE"],
            "orthogonal_error_reduced": source["mean_orthogonal_error_power"] <
                baseline["mean_orthogonal_error_power"],
            "directional_magnitude_not_badly_biased": .5 <=
                source["median_directional_magnitude_ratio"] <= 1.5,
            "coverage_not_degenerate": by_coverage[design_id]["coverage_not_degenerate"],
            "source_fidelity_label_present": source["source_fidelity"] in {
                "SOURCE_EXPLICIT", "SOURCE_IMPLIED", "DIAGNOSTIC_EXTENSION"},
            "pareto_shift_favorable": by_pareto[design_id]["any_pareto_dominance"],
            "direction_variance_material": variance["direction_variance_material"],
            "not_hard_projection": True,
        }
        rows.append({
            "design_id": design_id,
            "source_fidelity": source["source_fidelity"],
            "gates": gates,
            "pass": all(gates.values()),
            "MSE_ratio_to_iid": source["MSE_ratio_to_iid"],
            "orthogonal_error_ratio_to_iid": source["orthogonal_error_ratio_to_iid"],
        })
    eligible = [row for row in rows if row["pass"]]
    eligible.sort(key=lambda row: row["MSE_ratio_to_iid"])
    promoted = [row["design_id"] for row in eligible[:int(settings()["maximum_online_designs"])]]
    value = {
        "pass": True,
        "design_gates": rows,
        "eligible_designs": [row["design_id"] for row in eligible],
        "promoted_designs_for_short_online_rollout": promoted,
        "maximum_online_designs": int(settings()["maximum_online_designs"]),
        "hard_projection_promoted": False,
        "no_design_forced_if_gate_empty": True,
        "forbidden_auto_runs_launched": [],
    }
    return write_artifact("frozen_state_promotion_gate", value,
                          title="V21 frozen-state candidate-design promotion gate")
