"""Fast-only local-population-gradient rollout and causal classification."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from hdfa_rl_suite.google_pure_v19_experimental.dynamic_validation import _controller_spec

from .analysis import sinusoidal_transfer
from .data import evaluator, load_matched_run, verify_import_manifest
from .io import ARTIFACT_ROOT, read_json, settings, write_artifact


def _trajectory_summary(epochs: np.ndarray, normalized_means: np.ndarray,
                        mean_costs: np.ndarray, stochastic_costs: np.ndarray,
                        fixed_costs: np.ndarray, optimal_costs: np.ndarray,
                        frequency: float) -> dict[str, Any]:
    denominator = float(np.mean(fixed_costs) - np.mean(optimal_costs))
    if denominator <= 0:
        raise RuntimeError("population trajectory has a nonpositive evaluator denominator")
    scalar = np.mean(normalized_means, axis=1)
    transfer = sinusoidal_transfer(epochs, scalar, frequency)
    phase = 2*np.pi*frequency*epochs
    design = np.column_stack([np.ones_like(epochs), np.sin(phase), np.cos(phase)])
    fit = design @ np.linalg.lstsq(design, scalar, rcond=None)[0]
    residual = scalar - fit
    orthogonal = normalized_means - scalar[:, None]
    target = np.sin(phase)
    return {
        "I_mean": (float(np.mean(fixed_costs)) - float(np.mean(mean_costs))) / denominator,
        "I_stochastic_exact_diagnostic": (
            float(np.mean(fixed_costs)) - float(np.mean(stochastic_costs))) / denominator,
        "gain": transfer["gain"],
        "phase_lag_radians": transfer["phase_lag_radians"],
        "fundamental_I_prediction": transfer["fundamental_quadratic_I_prediction"],
        "DC_offset": transfer["dc_offset"],
        "harmonic_and_nonperiodic_power": float(np.mean(residual**2)),
        "orthogonal_diffusion_power": float(np.mean(orthogonal**2)),
        "tracking_error_RMS": float(np.sqrt(np.mean((normalized_means - target[:, None])**2))),
        "update_lag_epochs": transfer["phase_lag_radians"] / (2*np.pi*frequency),
        "raw_mean_cost_detector_events_per_shot": float(np.mean(mean_costs)),
        "raw_stochastic_cost_detector_events_per_shot": float(np.mean(stochastic_costs)),
        "raw_fixed_cost_detector_events_per_shot": float(np.mean(fixed_costs)),
        "raw_optimal_cost_detector_events_per_shot": float(np.mean(optimal_costs)),
    }


def run_population_gradient_fast() -> dict[str, Any]:
    verify_import_manifest()
    cfg = settings()
    run = load_matched_run("fast")
    evaluation = evaluator()
    controller = _controller_spec()
    epochs_total = int(cfg["population_rollout_epochs"])
    transient = int(cfg["postrepair"]["transient_epochs"])
    frequency = float(cfg["fast_frequency_per_epoch"])
    decomposition_path = ARTIFACT_ROOT / "fast_mean_cost_decomposition.json"
    if not decomposition_path.is_file():
        from .analysis import decompose_fast_mean_cost
        decompose_fast_mean_cost()
    decomposition = read_json(decomposition_path)
    phase_epochs = np.asarray(decomposition["hessian_phase_state_epochs"], dtype=int)
    phase_hessians = np.asarray(decomposition["hessian_diagonal_phase_states"], dtype=float)
    period = int(round(1.0 / frequency))
    mean = np.zeros(41, dtype=float)
    rows = []
    for epoch in range(epochs_total):
        normalized = evaluation.plant.apply_control_transform(mean)
        target = evaluation.plant.optimum(epoch, frequency)
        phase_distance = np.abs((phase_epochs % period) - epoch % period)
        phase_distance = np.minimum(phase_distance, period - phase_distance)
        hessian = phase_hessians[int(np.argmin(phase_distance))]
        transform_jacobian = 1.0 - np.tanh(mean / evaluation.plant.control_limits)**2
        gradient = 2.0 * hessian * (normalized - target) * transform_jacobian
        sigma = np.asarray(run["records"][epoch]["behavior_sigma"], dtype=float)
        noises = run["noises"][epoch]
        stochastic_actions = mean[None, :] + sigma[None, :] * noises
        mean_cost = evaluation.cost(mean, epoch, frequency)
        stochastic_cost = float(np.mean([
            evaluation.cost(action, epoch, frequency) for action in stochastic_actions]))
        fixed_cost = evaluation.normalized_cost(np.zeros(41), epoch, frequency)
        optimal_cost = evaluation.normalized_cost(target, epoch, frequency)
        next_mean = np.clip(mean - controller.mean_learning_rate * gradient, -2.0, 2.0)
        rows.append({
            "epoch": epoch,
            "normalized_mean": normalized.tolist(),
            "latent_mean": mean.tolist(),
            "replayed_sigma": sigma.tolist(),
            "population_gradient": gradient.tolist(),
            "delta_mean": (next_mean - mean).tolist(),
            "mean_cost": mean_cost,
            "stochastic_cost": stochastic_cost,
            "fixed_cost": fixed_cost,
            "optimal_cost": optimal_cost,
        })
        mean = next_mean
    selected = rows[transient:]
    epochs = np.asarray([row["epoch"] for row in selected], dtype=float)
    population = _trajectory_summary(
        epochs,
        np.asarray([row["normalized_mean"] for row in selected]),
        np.asarray([row["mean_cost"] for row in selected]),
        np.asarray([row["stochastic_cost"] for row in selected]),
        np.asarray([row["fixed_cost"] for row in selected]),
        np.asarray([row["optimal_cost"] for row in selected]),
        frequency)

    baseline_records = run["records"][transient:epochs_total]
    baseline_means = np.asarray([row["normalized_behavior_mean"] for row in baseline_records])
    baseline_mean_costs = np.asarray([
        evaluation.normalized_cost(baseline_means[index], row["epoch"], frequency)
        for index, row in enumerate(baseline_records)])
    baseline_stochastic_costs = np.asarray([
        np.mean([evaluation.cost(
            np.asarray(row["latent_behavior_mean"]) +
            np.asarray(row["behavior_sigma"]) * noise, row["epoch"], frequency)
                 for noise in run["noises"][row["epoch"]]])
        for row in baseline_records])
    baseline_fixed = np.asarray([
        evaluation.normalized_cost(np.zeros(41), row["epoch"], frequency)
        for row in baseline_records])
    baseline_optimal = np.asarray([
        evaluation.normalized_cost(evaluation.plant.optimum(row["epoch"], frequency),
                                   row["epoch"], frequency)
        for row in baseline_records])
    baseline = _trajectory_summary(
        epochs, baseline_means, baseline_mean_costs, baseline_stochastic_costs,
        baseline_fixed, baseline_optimal, frequency)
    sampling_failure = population["I_mean"] > 0 and baseline["I_mean"] < 0
    value = {
        "pass": True,
        "campaign_scope": "FAST_ONLY_300_EPOCH_DEVELOPMENT_ROLLOUT",
        "mean_gradient": "phase-interpolated exact-Stim local Hessian population gradient",
        "population_gradient_approximation": (
            "analytic local quadratic gradient; exact Stim evaluator retained for all reported costs"),
        "same_target_trajectory": True,
        "same_mean_learning_rate": True,
        "same_update_cadence": True,
        "same_normalization": True,
        "same_plant": True,
        "same_public_analogue_scale_treatment": (
            "V19 fast sigma trajectory replayed exactly; sigma is not updated by this diagnostic"),
        "population_gradient_removes_finite_candidate_mean_noise_only": True,
        "analysis_epoch_window": [transient, epochs_total],
        "ordinary_v19_fast_same_window": baseline,
        "population_gradient_fast": population,
        "sampling_information_failure_pattern": sampling_failure,
        "deterministic_bandwidth_failure_pattern": population["I_mean"] < 0,
        "trajectory_rows": rows,
        "slow_intermediate_rerun": False,
        "finite_shot_acquisition_launched": False,
        "forbidden_auto_runs_launched": [],
    }
    return write_artifact("population_gradient_fast_rollout", value,
                          title="V20 population-gradient fast rollout")


def classify_root_cause() -> dict[str, Any]:
    verify_import_manifest()
    dependencies = {
        "decomposition": "fast_mean_cost_decomposition.json",
        "geometry": "transfer_evaluation_geometry_audit.json",
        "gradients": "fast_gradient_statistics.json",
        "reference": "fast_reference_gradients.json",
        "factorial": "candidate_vs_shots_factorial.json",
        "fixed_budget": "fixed_budget_information_comparison.json",
        "scale": "frozen_scale_information_damage_frontier.json",
        "dynamic_sigma": "dynamic_sigma_signed_gradients.json",
        "acquisition": "acquisition_bias_audit.json",
        "population": "population_gradient_fast_rollout.json",
    }
    missing = [name for name in dependencies.values()
               if not (ARTIFACT_ROOT / name).is_file()]
    if missing:
        raise RuntimeError(f"root-cause classification requires completed diagnostics: {missing}")
    values = {key: read_json(ARTIFACT_ROOT / name) for key, name in dependencies.items()}
    gates = {
        "orthogonal_cost_demonstrated": values["decomposition"]["classification"] ==
            "ORTHOGONAL_MEAN_DIFFUSION_DOMINATES",
        "transfer_geometry_not_primary": values["geometry"]["classification"] !=
            "TRANSFER_EVALUATION_GEOMETRY_MISMATCH",
        "ordinary_reference_gradient_poor": values["reference"][
            "classification"] in {"FINITE_CANDIDATE_VARIANCE", "SYSTEMATIC_GRADIENT_BIAS"},
        "candidate_or_fixed_budget_evidence": values["factorial"]["classification"] ==
            "ACTION_SPACE_SAMPLING_LIMITED" or values["fixed_budget"]["classification"] ==
            "ACTION_SPACE_SAMPLING_LIMITED",
        "optimizer_path_correct": read_json(
            ARTIFACT_ROOT / "fast_update_efficiency.json")[
                "optimizer_applies_supplied_gradient_correctly"] is True,
        "batch_motion_not_primary": values["acquisition"]["classification"] ==
            "NO_MEANINGFUL_BATCH_MOTION_BIAS",
        "population_gradient_succeeds": values["population"][
            "sampling_information_failure_pattern"] is True,
    }
    if gates["population_gradient_succeeds"] and gates["ordinary_reference_gradient_poor"] and \
            (gates["candidate_or_fixed_budget_evidence"] or
             gates["orthogonal_cost_demonstrated"]):
        primary = "FINITE_CANDIDATE_DIRECTIONAL_FAILURE"
        secondary = ["ORTHOGONAL_MEAN_DIFFUSION"] if gates[
            "orthogonal_cost_demonstrated"] else []
    elif gates["orthogonal_cost_demonstrated"] and gates["population_gradient_succeeds"]:
        primary, secondary = "ORTHOGONAL_MEAN_DIFFUSION", []
    elif not gates["transfer_geometry_not_primary"]:
        primary, secondary = "TRANSFER_EVALUATION_GEOMETRY_MISMATCH", []
    elif not gates["batch_motion_not_primary"]:
        primary, secondary = "MOVING_TARGET_BATCH_BIAS", []
    elif values["population"]["deterministic_bandwidth_failure_pattern"]:
        primary, secondary = "DETERMINISTIC_FAST_BANDWIDTH_LIMIT", []
    else:
        primary, secondary = "MULTIPLE_CAUSES", []
    permitted = primary in {"FINITE_CANDIDATE_DIRECTIONAL_FAILURE",
                            "ORTHOGONAL_MEAN_DIFFUSION"}
    value = {
        "pass": permitted,
        "primary_classification": primary,
        "secondary_causes": secondary,
        "causal_gates": gates,
        "diagnostic_artifacts": dependencies,
        "repair_permitted": permitted,
        "permitted_single_repair": (
            "PUBLIC_FIGURE5A_SHARED_SUBSPACE_MEAN_GRADIENT_PROJECTION" if permitted else None),
        "root_cause_emitted_before_repair": True,
        "mean_learning_rate_change_permitted": False,
        "sigma_tuning_permitted": False,
        "forbidden_auto_runs_launched": [],
    }
    return write_artifact("root_cause_classification", value,
                          title="V20 fast-mean root-cause classification")
