"""Reduced physically matched V16 causal validation experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hdfa_rl_suite.google_pure_source_exact.figure5a.acquisition import run_cell
from hdfa_rl_suite.google_pure_source_exact.figure5a.contracts import AcquisitionMode, Figure5aProtocol
from hdfa_rl_suite.google_pure_source_exact.figure5a.validation import build_plant, dependency_hashes
from hdfa_rl_suite.google_pure_source_exact.paper_families.common import (
    SparseControlPlant,
    _sparse_source_loss,
)
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.contracts import PositivityGuard
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.gaussian import DirectSigmaGaussianPolicy
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.optimizer import (
    DirectSigmaOptimizer,
    OptimizerConfig,
)
from hdfa_rl_suite.google_pure_source_exact.source_normalization import SourceNormalizationBoundary
from hdfa_rl_suite.google_pure_source_exact.step_response_130.plant import SourceStepPlant
from hdfa_rl_suite.google_pure_v7.figure5.accounting import detector_factors, total_controls
from hdfa_rl_suite.google_pure_v12.directional import reference_directional_curvature

from .contracts import NONFINAL
from .coordinate import run_covariance_fixture
from .io import ARTIFACT_ROOT, ROOT, atomic_json, atomic_text, canonical_hash, config, read_json
from .optimizer_audits import freeze_optimizer


def _frozen() -> dict[str, Any]:
    path = ARTIFACT_ROOT / "frozen_source_normalized_optimizer.json"
    return read_json(path) if path.is_file() else freeze_optimizer()


def _optimizer_config(values: dict[str, Any], *, v16: bool) -> OptimizerConfig:
    if v16:
        return OptimizerConfig(
            mean_learning_rate=float(values["mean_learning_rate"]),
            sigma_learning_rate=float(values["sigma_learning_rate"]),
            baseline_learning_rate=float(values["baseline_learning_rate"]),
            momentum=float(values["momentum"]),
            minimum_sigma=float(values["minimum_sigma"]),
            maximum_sigma=float(values["maximum_sigma"]),
            positivity_guard=PositivityGuard(values["positivity_guard"]),
        )
    return OptimizerConfig(.08, .02, .08, momentum=0.0, minimum_sigma=.002,
                           maximum_sigma=.8,
                           positivity_guard=PositivityGuard.PROJECTED_GRADIENT)


def _coupled_binomial(probabilities: dict[str, np.ndarray], cycles: int, seed: int) -> dict[str, np.ndarray]:
    """Exact common-uniform binomial coupling without allocating cycle-level tapes."""
    names = tuple(probabilities)
    values = np.stack([np.asarray(probabilities[name], dtype=float) for name in names])
    if values.ndim != 3 or np.any(values < 0) or np.any(values > 1):
        raise ValueError("coupled probabilities must have shape arms x candidates x detectors")
    result = {name: np.zeros(values.shape[1:], dtype=np.int64) for name in names}
    rng = np.random.default_rng(int(seed))
    for candidate in range(values.shape[1]):
        for detector in range(values.shape[2]):
            order = np.argsort(values[:, candidate, detector], kind="stable")
            sorted_probability = values[order, candidate, detector]
            intervals = np.diff(np.concatenate(([0.0], sorted_probability, [1.0])))
            bins = rng.multinomial(int(cycles), intervals)
            cumulative = np.cumsum(bins[:-1])
            for rank, arm_index in enumerate(order):
                result[names[int(arm_index)]][candidate, detector] = cumulative[rank]
    return result


def _tau(progress: list[float]) -> dict[str, float | None]:
    values = np.asarray(progress, dtype=float)
    residual = 1.0 - values
    selected = np.flatnonzero((values >= .05) & (values <= .85) & (residual > 0))
    if selected.size < 8:
        return {"tau_epochs": None, "tau_fit_r_squared": None, "tau_identifiable": False}
    x = selected.astype(float)
    y = np.log(residual[selected])
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    denominator = float(np.sum((y - y.mean())**2))
    r_squared = 1.0 - float(np.sum((y - fitted)**2)) / denominator if denominator > 0 else 0.0
    identifiable = bool(slope < 0 and r_squared >= .8)
    return {"tau_epochs": float(-1.0 / slope) if identifiable else None,
            "tau_fit_r_squared": float(r_squared), "tau_identifiable": identifiable}


def _response_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    finals = np.asarray([row["final_target_fraction"] for row in rows])
    t50 = [row["t50_epochs"] for row in rows if row["t50_epochs"] is not None]
    t632 = [row["t63_2_epochs"] for row in rows if row["t63_2_epochs"] is not None]
    taus = [row["tau_epochs"] for row in rows if row["tau_epochs"] is not None]
    return {
        "run_count": len(rows),
        "median_final_target_fraction": float(np.median(finals)),
        "median_t50_epochs": float(np.median(t50)) if t50 else None,
        "median_t63_2_epochs": float(np.median(t632)) if t632 else None,
        "median_tau_epochs": float(np.median(taus)) if taus else None,
        "median_final_native_update_norm": float(np.median([row["native_update_norm"][-1] for row in rows])),
        "median_gradient_snr": float(np.nanmedian([np.nanmedian(row["gradient_snr"]) for row in rows])),
        "median_final_native_sigma": float(np.median([row["native_sigma_median"][-1] for row in rows])),
    }


def _run_step_seed(seed: int) -> list[dict[str, Any]]:
    cfg = config()["matched_step"]
    frozen = _frozen()
    plant = SourceStepPlant(onset_epoch=0)
    scale_v12 = np.sqrt(reference_directional_curvature() / plant.sensitivity)
    scale_v15 = np.sqrt(.01 / plant.sensitivity)
    native_target = np.zeros(plant.controls)
    native_target[0] = scale_v15[0] * plant.target_delta
    native_sigma = scale_v15 * float(frozen["initial_sigma"])
    arm_specs = {
        "B_V12_DEVELOPMENT_DIAGNOSTIC": (scale_v12, False, .001),
        "C_V15_INHERITED_OPTIMIZER": (scale_v15, False, .001),
        "D_V16_FROZEN_OPTIMIZER": (scale_v15, True, float(frozen["entropy_coefficient"])),
    }
    states: dict[str, dict[str, Any]] = {}
    for name, (scale, is_v16, entropy_weight) in arm_specs.items():
        normalized_mean = np.zeros(plant.controls)
        normalized_sigma = native_sigma / scale
        policy = DirectSigmaGaussianPolicy(normalized_mean, normalized_sigma, seed=seed)
        states[name] = {
            "scale": scale, "policy": policy,
            "optimizer": DirectSigmaOptimizer(plant.controls, plant.detectors,
                                                _optimizer_config(frozen, v16=is_v16)),
            "baseline": np.zeros(plant.detectors), "entropy": entropy_weight,
            "progress": [], "native_update_norm": [], "gradient_snr": [],
            "native_sigma": [], "candidate_damage": [], "gradient_norm": [],
            "z_hashes": [],
        }
    owners = np.arange(plant.controls, dtype=np.int64) % plant.detectors
    for epoch in range(int(cfg["epochs"])):
        probabilities, batches = {}, {}
        for name, state in states.items():
            batch = state["policy"].sample(int(cfg["candidates"]))
            batches[name] = batch
            native_actions = batch.actions * state["scale"][None, :]
            probabilities[name] = np.asarray([
                plant.expected_edr(action, epoch, target_controls=native_target)
                for action in native_actions])
            state["z_hashes"].append(canonical_hash(batch.standardized_noise.tolist()))
        counts = _coupled_binomial(
            probabilities, int(cfg["cycles_per_candidate"]),
            int(canonical_hash(["v16-matched-step", seed, epoch])[:16], 16))
        for name, state in states.items():
            policy = state["policy"]
            batch = batches[name]
            rewards = -counts[name] / float(cfg["cycles_per_candidate"])
            score = (batch.actions[:, 0] - policy.mean[0]) / policy.sigma[0]**2
            influences = (rewards[:, owners[0]] - state["baseline"][owners[0]]) * score
            standard_error = float(np.std(influences, ddof=1) / np.sqrt(len(influences)))
            snr = float(abs(np.mean(influences)) / standard_error) if standard_error > 0 else np.nan
            loss = _sparse_source_loss(
                batch.actions, rewards, owners, policy.mean, policy.sigma, state["baseline"],
                batch.behavior, clip=.2, entropy_weight=state["entropy"], baseline_weight=.2)
            before_native = state["scale"] * policy.mean
            state["optimizer"].step(
                policy.mean, policy.sigma, state["baseline"], loss["grad_mean"],
                loss["grad_sigma"], loss["grad_baseline"], mean_bounds=(-2.0, 2.0))
            policy.policy_version += 1
            after_native = state["scale"] * policy.mean
            state["progress"].append(float(after_native[0] / native_target[0]))
            state["native_update_norm"].append(float(np.linalg.norm(after_native - before_native)))
            state["gradient_snr"].append(snr)
            state["native_sigma"].append(float(np.median(state["scale"] * policy.sigma)))
            mean_probability = plant.expected_edr(after_native, epoch, target_controls=native_target)
            state["candidate_damage"].append(float(np.mean(probabilities[name]) - np.mean(mean_probability)))
            state["gradient_norm"].append(float(np.linalg.norm(loss["grad_mean"])))
    rows = []
    for name, state in states.items():
        progress = state["progress"]
        t50 = np.flatnonzero(np.asarray(progress) >= .5)
        t632 = np.flatnonzero(np.asarray(progress) >= .632)
        rows.append({
            "branch": name, "seed": int(seed), "epochs": int(cfg["epochs"]),
            "candidate_count": int(cfg["candidates"]),
            "cycles_per_candidate": int(cfg["cycles_per_candidate"]),
            "native_initial_mean_hash": canonical_hash(np.zeros(plant.controls).tolist()),
            "native_initial_covariance_hash": canonical_hash(np.square(native_sigma).tolist()),
            "native_target_hash": canonical_hash(native_target.tolist()),
            "native_step_size": float(native_target[0]),
            "normalized_target_coordinate": float(native_target[0] / state["scale"][0]),
            "standard_normal_tape_hashes": state["z_hashes"],
            "qec_tape_rule": "sha256(v16-matched-step,seed,epoch) common-uniform multinomial coupling",
            "target_relative_progress": progress,
            "final_target_fraction": float(progress[-1]),
            "t50_epochs": int(t50[0]) if t50.size else None,
            "t63_2_epochs": int(t632[0]) if t632.size else None,
            **_tau(progress),
            "native_update_norm": state["native_update_norm"],
            "gradient_snr": state["gradient_snr"],
            "normalized_gradient_norm": state["gradient_norm"],
            "native_sigma_median": state["native_sigma"],
            "candidate_excess_edr": state["candidate_damage"],
            "controller_target_access": False,
            "source_budget_run": False,
            **NONFINAL,
        })
    return rows


def run_matched_step() -> dict[str, Any]:
    covariance = run_covariance_fixture()
    cfg = config()["matched_step"]
    rows = [row for seed in cfg["seeds"] for row in _run_step_seed(int(seed))]
    branches = sorted({row["branch"] for row in rows})
    summaries = {branch: _response_summary([row for row in rows if row["branch"] == branch])
                 for branch in branches}
    paired_step_tapes = all(len({tuple(row["standard_normal_tape_hashes"])
                                 for row in rows if row["seed"] == seed}) == 1
                            for seed in cfg["seeds"])
    v15 = summaries["C_V15_INHERITED_OPTIMIZER"]
    v16 = summaries["D_V16_FROZEN_OPTIMIZER"]
    stronger = bool(v16["median_final_target_fraction"] >
                    v15["median_final_target_fraction"] + .05)
    if v16["median_t50_epochs"] is not None and v15["median_t50_epochs"] is not None:
        stronger = stronger and v16["median_t50_epochs"] < v15["median_t50_epochs"]
    plant = SourceStepPlant(onset_epoch=0)
    scale_a = np.ones(plant.controls)
    scale_b = np.sqrt(reference_directional_curvature() / plant.sensitivity)
    scale_c = np.sqrt(.01 / plant.sensitivity)
    native_before = np.zeros(plant.controls)
    native_after = np.zeros(plant.controls)
    native_after[0] = scale_c[0] * plant.target_delta
    target_contract = {
        "native_target_before_step": native_before.tolist(),
        "native_target_after_step": native_after.tolist(),
        "native_step_vector": (native_after - native_before).tolist(),
        "normalized_target_before": {
            "A_IDENTITY": (native_before / scale_a).tolist(),
            "B_V12_DEVELOPMENT_DIAGNOSTIC": (native_before / scale_b).tolist(),
            "C_V15_SOURCE_NORMALIZED": (native_before / scale_c).tolist(),
        },
        "normalized_target_after": {
            "A_IDENTITY": (native_after / scale_a).tolist(),
            "B_V12_DEVELOPMENT_DIAGNOSTIC": (native_after / scale_b).tolist(),
            "C_V15_SOURCE_NORMALIZED": (native_after / scale_c).tolist(),
        },
        "native_targets_identical_across_abc": True,
    }
    result = {
        "schema_version": "google-pure-v16-physically-matched-step.v1",
        "coordinate_equivalence_fixture_hash": canonical_hash(covariance),
        "rows": rows, "summaries": summaries,
        "physical_target_contract": target_contract,
        "all_native_starts_identical": len({row["native_initial_mean_hash"] for row in rows}) == 1,
        "all_native_covariances_identical": len({row["native_initial_covariance_hash"] for row in rows}) == 1,
        "all_native_targets_identical": len({row["native_target_hash"] for row in rows}) == 1,
        "random_tapes_paired": paired_step_tapes,
        "v16_materially_stronger_than_v15_same_plant": stronger,
        "v12_role": "COORDINATE_COVARIANCE_DEVELOPMENT_DIAGNOSTIC_ONLY",
        "optimizer_retuned_after_matched_result": False,
        "paper_130_epoch_target_used_for_selection": False,
        "classification": "REDUCED_PHYSICALLY_MATCHED_CAUSAL_VALIDATION",
        "optimizer_bundle_hash": _frozen()["optimizer_bundle_hash"],
        "pass": bool(covariance["pass"] and stronger),
        **NONFINAL,
    }
    atomic_json(ARTIFACT_ROOT / "matched_step" / "comparison.json", result)
    atomic_text(ARTIFACT_ROOT / "matched_step" / "comparison.md",
                "# Physically matched step\n\nA/B covariance is established separately. This comparison holds native start, covariance, target, plant, candidate tape, and QEC tape fixed; V12 remains a development diagnostic only.")
    return result


def _figure5b_plant() -> tuple[SparseControlPlant, SourceNormalizationBoundary]:
    cfg = config()["matched_figure5b"]
    distance, parameters = int(cfg["distance"]), int(cfg["parameters_per_gate"])
    plant = SparseControlPlant(distance, total_controls(distance, parameters),
                               detector_factors(distance), seed=9100 + 101 * distance + parameters)
    boundary = SourceNormalizationBoundary.from_training_objective(
        "FIGURE5B_SPARSE_SCALING", plant.connected_objective_curvature,
        control_ids=plant.control_ids)
    return plant, boundary


def _run_figure5b_seed(seed: int) -> list[dict[str, Any]]:
    cfg, frozen = config()["matched_figure5b"], _frozen()
    plant, boundary = _figure5b_plant()
    scale_v15 = boundary.native_scale
    scale_v12 = np.sqrt(reference_directional_curvature() / plant.connected_objective_curvature)
    rng = np.random.default_rng(seed + 100000)
    initial_x_v15 = rng.choice((-1.0, 1.0), plant.controls) * rng.uniform(.45, .65, plant.controls)
    initial_native = scale_v15 * initial_x_v15
    native_sigma = scale_v15 * float(frozen["initial_sigma"])
    target = np.zeros(plant.controls)
    specs = {
        "B_V12_DEVELOPMENT_DIAGNOSTIC": (scale_v12, False, .001),
        "C_V15_INHERITED_OPTIMIZER": (scale_v15, False, .001),
        "D_V16_FROZEN_OPTIMIZER": (scale_v15, True, float(frozen["entropy_coefficient"])),
    }
    states = {}
    for name, (scale, is_v16, beta) in specs.items():
        policy = DirectSigmaGaussianPolicy(initial_native / scale, native_sigma / scale, seed=seed)
        states[name] = {
            "scale": scale, "policy": policy,
            "optimizer": DirectSigmaOptimizer(plant.controls, plant.detectors,
                                                _optimizer_config(frozen, v16=is_v16)),
            "baseline": np.zeros(plant.detectors), "entropy": beta,
            "records": [], "z_hashes": [],
        }
    for epoch in range(int(cfg["epochs"])):
        probabilities, batches = {}, {}
        for name, state in states.items():
            batch = state["policy"].sample(int(cfg["candidates"]))
            batches[name] = batch
            native_actions = batch.actions * state["scale"][None, :]
            probabilities[name] = plant.expected_detector_rates(native_actions, target)
            state["z_hashes"].append(canonical_hash(batch.standardized_noise.tolist()))
        counts = _coupled_binomial(
            probabilities, int(cfg["cycles_per_candidate"]),
            int(canonical_hash(["v16-matched-figure5b", seed, epoch])[:16], 16))
        for name, state in states.items():
            policy, batch = state["policy"], batches[name]
            before_native = state["scale"] * policy.mean
            before = plant.performance(before_native, target)
            rewards = -counts[name] / float(cfg["cycles_per_candidate"])
            loss = _sparse_source_loss(
                batch.actions, rewards, plant.control_detector, policy.mean, policy.sigma,
                state["baseline"], batch.behavior, clip=.2,
                entropy_weight=state["entropy"], baseline_weight=.2)
            state["optimizer"].step(
                policy.mean, policy.sigma, state["baseline"], loss["grad_mean"],
                loss["grad_sigma"], loss["grad_baseline"], mean_bounds=(-2.0, 2.0))
            policy.policy_version += 1
            after_native = state["scale"] * policy.mean
            after = plant.performance(after_native, target)
            denominator = before["lambda"] - before["lambda_star"]
            fractional = ((before["lambda"] - after["lambda"]) / denominator
                          if denominator != 0 else np.nan)
            candidate_physical = float(np.mean(probabilities[name]))
            state["records"].append({
                "epoch": epoch,
                "lambda": before["lambda"], "lambda_next": after["lambda"],
                "lambda_star": before["lambda_star"],
                "fractional_residual_reduction": float(fractional),
                "physical_error": before["physical_error"],
                "edr": before["physical_error"],
                "logical_error": before["logical_error"],
                "normalized_gradient_norm": float(np.linalg.norm(loss["grad_mean"])),
                "native_gradient_norm": float(np.linalg.norm(loss["grad_mean"] / state["scale"])),
                "normalized_update_norm": float(np.linalg.norm(policy.mean - before_native / state["scale"])),
                "native_update_norm": float(np.linalg.norm(after_native - before_native)),
                "normalized_sigma_median": float(np.median(policy.sigma)),
                "native_sigma_median": float(np.median(state["scale"] * policy.sigma)),
                "candidate_damage": candidate_physical - before["physical_error"],
            })
    rows = []
    for name, state in states.items():
        fractions = np.asarray([record["fractional_residual_reduction"] for record in state["records"]])
        rows.append({
            "branch": name, "seed": int(seed),
            "native_start_hash": canonical_hash(initial_native.tolist()),
            "native_covariance_hash": canonical_hash(np.square(native_sigma).tolist()),
            "native_optimum_hash": canonical_hash(target.tolist()),
            "plant_hash": plant.plant_hash, "graph_hash": plant.graph_hash,
            "standard_normal_tape_hashes": state["z_hashes"],
            "qec_tape_rule": "sha256(v16-matched-figure5b,seed,epoch) common-uniform multinomial coupling",
            "records": state["records"],
            "median_fractional_residual_reduction": float(np.nanmedian(fractions)),
            "positive_fractional_reduction_fraction": float(np.mean(fractions > 0)),
            "sustained_positive_last_quarter": bool(np.nanmedian(fractions[-max(4, len(fractions)//4):]) > 0),
            **NONFINAL,
        })
    return rows


def run_matched_figure5b() -> dict[str, Any]:
    cfg = config()["matched_figure5b"]
    rows = [row for seed in cfg["seeds"] for row in _run_figure5b_seed(int(seed))]
    starts = {(row["seed"], row["native_start_hash"]) for row in rows}
    covariances = {(row["seed"], row["native_covariance_hash"]) for row in rows}
    summaries = {}
    for branch in sorted({row["branch"] for row in rows}):
        selected = [row for row in rows if row["branch"] == branch]
        summaries[branch] = {
            "median_fractional_residual_reduction": float(np.median([
                row["median_fractional_residual_reduction"] for row in selected])),
            "median_positive_fraction": float(np.median([
                row["positive_fractional_reduction_fraction"] for row in selected])),
            "sustained_positive_run_count": sum(row["sustained_positive_last_quarter"] for row in selected),
            "run_count": len(selected),
        }
    v16 = summaries["D_V16_FROZEN_OPTIMIZER"]
    paired_figure5b_tapes = all(len({tuple(row["standard_normal_tape_hashes"])
                                     for row in rows if row["seed"] == seed}) == 1
                                for seed in cfg["seeds"])
    same_optima = all(len({row["native_optimum_hash"] for row in rows
                           if row["seed"] == seed}) == 1 for seed in cfg["seeds"])
    same_plants = len({(row["plant_hash"], row["graph_hash"]) for row in rows}) == 1
    result = {
        "schema_version": "google-pure-v16-physically-matched-figure5b.v1",
        "condition": {"distance": cfg["distance"], "parameters_per_gate": cfg["parameters_per_gate"]},
        "rows": rows, "summaries": summaries,
        "same_native_start_per_seed": len(starts) == len(cfg["seeds"]),
        "same_native_covariance_per_seed": len(covariances) == len(cfg["seeds"]),
        "same_native_optimum_plant_evaluation_and_tapes": bool(
            paired_figure5b_tapes and same_optima and same_plants),
        "main_metric": "fractional residual reduction r_t=(Lambda_t-Lambda_t+1)/(Lambda_t-Lambda_star)",
        "raw_delta_lambda_is_not_main_metric": True,
        "v16_sustained_positive_residual_reduction":
            v16["sustained_positive_run_count"] == v16["run_count"],
        "optimizer_retuned_after_matched_result": False,
        "classification": "REDUCED_PHYSICALLY_MATCHED_D3_P1_DIAGNOSTIC",
        "optimizer_bundle_hash": _frozen()["optimizer_bundle_hash"],
        "pass": bool(v16["sustained_positive_run_count"] == v16["run_count"]),
        **NONFINAL,
    }
    atomic_json(ARTIFACT_ROOT / "matched_figure5b" / "comparison.json", result)
    atomic_text(ARTIFACT_ROOT / "matched_figure5b" / "comparison.md",
                "# Physically matched Figure 5b d=3, P=1\n\nThe primary metric is fractional residual reduction. Native start, covariance, optimum, plant, evaluation and random tapes are paired. This is not final Figure 5b evidence.")
    return result


def _run_recovery() -> dict[str, Any]:
    frozen = _frozen()
    epochs = int(config()["reduced_acceptance"]["recovery_epochs"])
    plant = SparseControlPlant(5, 924, 24, seed=10100, curvature=.004)
    boundary = SourceNormalizationBoundary.from_training_objective(
        "RANDOMIZED_RECOVERY_AFTER_SPOIL", plant.connected_objective_curvature,
        control_ids=plant.control_ids)
    scale = boundary.native_scale
    rng = np.random.default_rng(81901)
    spoiled_x = rng.choice((-1.0, 1.0), plant.controls) * .45
    initial_native = scale * spoiled_x
    native_sigma = scale * float(frozen["initial_sigma"])
    target = np.zeros(plant.controls)
    specs = {"V15_INHERITED": (False, .001),
             "V16_FROZEN": (True, float(frozen["entropy_coefficient"]))}
    states = {}
    for name, (is_v16, beta) in specs.items():
        policy = DirectSigmaGaussianPolicy(initial_native / scale, native_sigma / scale, seed=81901)
        states[name] = {"policy": policy,
                        "optimizer": DirectSigmaOptimizer(plant.controls, plant.detectors,
                                                            _optimizer_config(frozen, v16=is_v16)),
                        "baseline": np.zeros(plant.detectors), "entropy": beta,
                        "progress": []}
    initial_norm = float(np.linalg.norm(initial_native))
    for epoch in range(epochs):
        probabilities, batches = {}, {}
        for name, state in states.items():
            batch = state["policy"].sample(16); batches[name] = batch
            probabilities[name] = plant.expected_detector_rates(batch.actions * scale[None, :], target)
        counts = _coupled_binomial(probabilities, 6000,
                                   int(canonical_hash(["v16-recovery", epoch])[:16], 16))
        for name, state in states.items():
            policy, batch = state["policy"], batches[name]
            rewards = -counts[name] / 6000.0
            loss = _sparse_source_loss(batch.actions, rewards, plant.control_detector,
                                       policy.mean, policy.sigma, state["baseline"], batch.behavior,
                                       clip=.2, entropy_weight=state["entropy"], baseline_weight=.2)
            state["optimizer"].step(policy.mean, policy.sigma, state["baseline"],
                                    loss["grad_mean"], loss["grad_sigma"], loss["grad_baseline"],
                                    mean_bounds=(-2.0, 2.0))
            policy.policy_version += 1
            residual = float(np.linalg.norm(scale * policy.mean) / initial_norm)
            state["progress"].append(1.0 - residual)
    summaries = {name: {"final_recovery_fraction": float(state["progress"][-1]),
                        "trajectory": state["progress"]}
                 for name, state in states.items()}
    return {"summaries": summaries,
            "v16_materially_stronger": summaries["V16_FROZEN"]["final_recovery_fraction"] >
                                      summaries["V15_INHERITED"]["final_recovery_fraction"] + .03,
            "native_start_covariance_plant_and_tapes_matched": True}


def _run_figure5a_reduced() -> dict[str, Any]:
    frozen = _frozen()
    source_config = read_json(ROOT / "configs/google_pure_source_exact/figure5a.json")
    plant = build_plant(source_config)
    reduced = config()["reduced_acceptance"]
    protocol = Figure5aProtocol(
        AcquisitionMode.VALIDATION, int(reduced["figure5a_epochs"]),
        int(reduced["figure5a_candidates"]), int(reduced["figure5a_cycles_per_candidate"]),
        int(source_config["plant"]["circuit_rounds"]),
    )
    optimizer_cfg = _optimizer_config(frozen, v16=True)
    rows = []
    for frequency in reduced["figure5a_frequencies"]:
        token = str(frequency).replace(".", "p")
        campaign_root = ARTIFACT_ROOT / "reduced_acceptance" / "figure5a" / "v16-acceptance-v2"
        checkpoint = campaign_root / f"checkpoint-{token}.json"
        provenance_path = campaign_root / f"acquisition-provenance-{token}.json"
        checkpoint_preexisted = checkpoint.is_file()
        result = run_cell(
            protocol=protocol, plant=plant, frequency=float(frequency),
            entropy_weight=float(frozen["entropy_coefficient"]), seed=81921,
            optimizer_config=optimizer_cfg, initial_sigma=float(frozen["initial_sigma"]),
            checkpoint_path=checkpoint, dependency_hashes=dependency_hashes(ROOT, source_config),
            controller_hash=frozen["optimizer_bundle_hash"], clip=float(frozen["ppo_clip"]),
            baseline_weight=float(frozen["baseline_loss_weight"]), resume=checkpoint.is_file(),
            source_budget_profile="V16_REDUCED_DEVELOPMENT_ACCEPTANCE",
        )
        if provenance_path.is_file():
            acquisition_provenance = read_json(provenance_path)
            if acquisition_provenance.get("optimizer_bundle_hash") != frozen["optimizer_bundle_hash"] or \
                    acquisition_provenance.get("protocol_hash") != protocol.protocol_hash:
                raise RuntimeError("V16 Figure 5a reduced acquisition provenance changed")
        else:
            acquisition_provenance = {
                "schema_version": "google-pure-v16-reduced-figure5a-provenance.v1",
                "frequency": float(frequency),
                "protocol_hash": protocol.protocol_hash,
                "optimizer_bundle_hash": frozen["optimizer_bundle_hash"],
                "fresh_acquisition_at_creation": not checkpoint_preexisted,
                "source_budget_profile": "V16_REDUCED_DEVELOPMENT_ACCEPTANCE",
                **NONFINAL,
            }
            if not acquisition_provenance["fresh_acquisition_at_creation"]:
                raise RuntimeError("new V16 acceptance provenance cannot adopt a pre-existing checkpoint")
            atomic_json(provenance_path, acquisition_provenance)
        optimum = np.asarray([record["optimum"] for record in result["epoch_records"]], dtype=float)
        learned = np.asarray([record["post_update_normalized_mean"]
                              for record in result["epoch_records"]], dtype=float)
        target = np.repeat(optimum[:, None], learned.shape[1], axis=1)
        fixed_squared_error = float(np.sum(target**2))
        learned_squared_error = float(np.sum((learned - target)**2))
        tracking_improvement = (1.0 - learned_squared_error / fixed_squared_error
                                if fixed_squared_error > 0 else None)
        rows.append({
            "frequency": float(frequency),
            "learned_mean_source_ratio": result["learned_mean_ratio"]["source_ratio"],
            "stochastic_source_ratio": result["stochastic_ratio"]["source_ratio"],
            "normalized_tracking_rmse": float(np.sqrt(np.mean((learned - target)**2))),
            "fixed_normalized_tracking_rmse": float(np.sqrt(np.mean(target**2))),
            "tracking_improvement_vs_fixed": tracking_improvement,
            "tracking_metric_uses_hidden_optimum_for_evaluation_only": True,
            "controller_target_access": False,
            "candidate_qec_cycles": result["candidate_qec_cycles"],
            "control_count": result["control_count"],
            "plant_hash": result["plant_hash"],
            "optimizer_bundle_hash": frozen["optimizer_bundle_hash"],
            "v15_boundary_hash": result["boundary_transform_hash"],
            "fresh_acquisition_at_campaign_creation":
                acquisition_provenance["fresh_acquisition_at_creation"],
            "checkpoint_reused_this_call": checkpoint_preexisted,
        })
    improvements = [row["tracking_improvement_vs_fixed"] for row in rows]
    directional = all(value is not None and np.isfinite(value) for value in improvements)
    if directional:
        directional = bool(improvements[0] >= improvements[1])
    count_ratios = [row["learned_mean_source_ratio"] for row in rows]
    count_direction = all(value is not None and np.isfinite(value) for value in count_ratios)
    if count_direction:
        count_direction = bool(count_ratios[0] >= count_ratios[1])
    return {"rows": rows, "slow_frequency_tracks_at_least_as_well_as_fast": directional,
            "expected_slow_fast_direction_pass": directional,
            "finite_shot_aggregate_ratio_direction_pass": count_direction,
            "acceptance_metric": "normalized tracking improvement versus fixed policy",
            "finite_shot_ratio_retained_but_not_used_for_tiny_budget_direction_gate": True,
            "fresh_reduced_acquisition_provenance_pass": all(
                row["fresh_acquisition_at_campaign_creation"] for row in rows),
            "figure5c_executed": False, "natural_drift_executed": False}


def run_reduced_acceptance() -> dict[str, Any]:
    """Run only the post-freeze reduced gate; never widen thresholds on failure."""
    step = run_matched_step()
    figure5b = run_matched_figure5b()
    recovery = _run_recovery()
    figure5a = _run_figure5a_reduced()
    gates = {
        "step_materially_stronger_than_v15_same_plant": step["v16_materially_stronger_than_v15_same_plant"],
        "recovery_materially_stronger_than_v15_same_plant": recovery["v16_materially_stronger"],
        "figure5a_slow_fast_expected_direction": figure5a["expected_slow_fast_direction_pass"],
        "figure5a_fresh_reduced_acquisition_provenance":
            figure5a["fresh_reduced_acquisition_provenance_pass"],
        "figure5b_sustained_positive_residual_reduction": figure5b["v16_sustained_positive_residual_reduction"],
        "figure5c_untouched": not figure5a["figure5c_executed"],
        "natural_drift_not_run": not figure5a["natural_drift_executed"],
    }
    result = {
        "schema_version": "google-pure-v16-reduced-acceptance.v1",
        "gates": gates,
        "step": step, "recovery": recovery, "figure5a": figure5a, "figure5b": figure5b,
        "thresholds_relaxed_after_failure": False,
        "dynamic_or_headline_tuning_after_freeze": False,
        "source_budget_run_launched": False,
        "reference_campaign_launched": False,
        "pass": all(gates.values()),
        "failure_interpretation": None if all(gates.values()) else
            "V16 optimizer diagnosis is not accepted; failed gates remain blockers and were not relaxed.",
        **NONFINAL,
    }
    atomic_json(ARTIFACT_ROOT / "reduced_acceptance" / "result.json", result)
    return result


__all__ = ["run_matched_step", "run_matched_figure5b", "run_reduced_acceptance"]
