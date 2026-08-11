"""Stored-trajectory V20 mean-cost and transfer-geometry diagnostics."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .core import (
    decompose_mean_trace,
    fundamental_improvement,
    quadratic_component_accounting,
)
from .data import evaluator, load_matched_run, selected_fast_epochs, verify_import_manifest
from .io import ARTIFACT_ROOT, read_json, settings, write_artifact


def sinusoidal_transfer(epochs: np.ndarray, values: np.ndarray,
                        frequency: float) -> dict[str, float]:
    time = np.asarray(epochs, dtype=float)
    response = np.asarray(values, dtype=float)
    phase = 2.0 * np.pi * float(frequency) * time
    design = np.column_stack([np.ones_like(time), np.sin(phase), np.cos(phase)])
    fit = np.linalg.lstsq(design, response, rcond=None)[0]
    gain = float(np.hypot(fit[1], fit[2]))
    lag = float(-math.atan2(fit[2], fit[1]))
    return {
        "dc_offset": float(fit[0]),
        "sine_coefficient": float(fit[1]),
        "cosine_coefficient": float(fit[2]),
        "gain": gain,
        "phase_lag_radians": lag,
        "fundamental_quadratic_I_prediction": fundamental_improvement(gain, lag),
    }


def _phase_hessians(frequency: float, *, delta: float) -> tuple[list[int], np.ndarray]:
    evaluation = evaluator()
    epochs = selected_fast_epochs()
    rows = []
    for epoch in epochs:
        target = evaluation.plant.optimum(epoch, frequency)
        center = evaluation.normalized_cost(target, epoch, frequency)
        diagonal = np.empty(41)
        for coordinate in range(41):
            offset = np.zeros(41); offset[coordinate] = delta
            diagonal[coordinate] = (
                evaluation.normalized_cost(target + offset, epoch, frequency) - 2.0 * center +
                evaluation.normalized_cost(target - offset, epoch, frequency)) / (2.0 * delta**2)
        rows.append(diagonal)
    return epochs, np.asarray(rows)


def _nearest_phase_hessian(epochs: np.ndarray, selected: list[int], hessians: np.ndarray,
                           period: int) -> np.ndarray:
    selected_phase = np.asarray(selected) % period
    result = []
    for epoch in epochs.astype(int):
        distance = np.abs(selected_phase - epoch % period)
        distance = np.minimum(distance, period - distance)
        result.append(hessians[int(np.argmin(distance))])
    return np.asarray(result)


def decompose_fast_mean_cost() -> dict[str, Any]:
    verify_import_manifest()
    cfg = settings()
    run = load_matched_run("fast")
    start, stop = map(int, cfg["analysis_window"])
    records = run["records"][start:stop]
    epochs = np.arange(start, stop, dtype=float)
    frequency = float(cfg["fast_frequency_per_epoch"])
    means = np.asarray([row["normalized_behavior_mean"] for row in records], dtype=float)
    decomposition = decompose_mean_trace(
        means, epochs, frequency, harmonics=int(cfg["decomposition_harmonics"]))
    reconstructions = {
        "A_fundamental_only": decomposition["fundamental"],
        "B_fundamental_plus_DC": decomposition["fundamental"] + decomposition["dc"],
        "C_plus_harmonics": decomposition["fundamental"] + decomposition["dc"] +
            decomposition["harmonic"],
        "D_full_driven_subspace": decomposition["fundamental"] + decomposition["dc"] +
            decomposition["harmonic"] + decomposition["transient"],
        "E_full_41d_mean": decomposition["full"],
    }
    evaluation = evaluator()
    fixed = np.zeros(41)
    target = np.asarray([evaluation.plant.optimum(int(epoch), frequency) for epoch in epochs])
    fixed_costs = np.asarray([
        evaluation.normalized_cost(fixed, epoch, frequency) for epoch in epochs])
    optimal_costs = np.asarray([
        evaluation.normalized_cost(target[index], epoch, frequency)
        for index, epoch in enumerate(epochs)])
    denominator = float(np.mean(fixed_costs) - np.mean(optimal_costs))
    if denominator <= 0:
        raise RuntimeError("exact fast production normalization denominator is not positive")
    stage_rows = []
    previous_i: float | None = None
    first_i: float | None = None
    stage_costs: dict[str, np.ndarray] = {}
    for name, trace in reconstructions.items():
        costs = np.asarray([
            evaluation.normalized_cost(trace[index], epoch, frequency)
            for index, epoch in enumerate(epochs)])
        stage_costs[name] = costs
        mean_cost = float(np.mean(costs))
        improvement = (float(np.mean(fixed_costs)) - mean_cost) / denominator
        if first_i is None:
            first_i = improvement
        incremental = 0.0 if previous_i is None else previous_i - improvement
        stage_rows.append({
            "stage": name,
            "raw_cost_detector_events_per_shot": mean_cost,
            "raw_cost_expected_production_mean_stream_counts": mean_cost * len(epochs) * 8 * 4000,
            "normalized_I": improvement,
            "incremental_penalty": incremental,
            "fraction_of_A_to_E_unexplained_penalty_recovered": None,
        })
        previous_i = improvement
    total_unexplained = float(stage_rows[0]["normalized_I"] - stage_rows[-1]["normalized_I"])
    for row in stage_rows:
        row["fraction_of_A_to_E_unexplained_penalty_recovered"] = (
            (stage_rows[0]["normalized_I"] - row["normalized_I"]) / total_unexplained
            if abs(total_unexplained) > 1e-15 else 0.0)
    sequential = {
        "DC_penalty": stage_rows[0]["normalized_I"] - stage_rows[1]["normalized_I"],
        "harmonic_penalty": stage_rows[1]["normalized_I"] - stage_rows[2]["normalized_I"],
        "transient_penalty": stage_rows[2]["normalized_I"] - stage_rows[3]["normalized_I"],
        "orthogonal_penalty": stage_rows[3]["normalized_I"] - stage_rows[4]["normalized_I"],
    }
    sequential["cross_term_penalty"] = total_unexplained - sum(sequential.values())

    selected, hessian_rows = _phase_hessians(
        frequency, delta=float(cfg["finite_difference_delta"]))
    hessian_trace = _nearest_phase_hessian(
        epochs, selected, hessian_rows, int(round(1.0 / frequency)))
    quadratic = quadratic_component_accounting({
        "fundamental_tracking_error": decomposition["fundamental"] - target,
        "DC": decomposition["dc"],
        "harmonics": decomposition["harmonic"],
        "transient": decomposition["transient"],
        "orthogonal": decomposition["orthogonal"],
    }, hessian_trace)
    exact_total_error = float(np.mean(stage_costs["E_full_41d_mean"] - optimal_costs))
    quadratic["exact_simulator_total_excess_cost"] = exact_total_error
    quadratic["quadratic_over_exact_excess_cost"] = (
        quadratic["total"] / exact_total_error if exact_total_error > 0 else None)
    penalties = {key: abs(float(value)) for key, value in sequential.items()
                 if key != "cross_term_penalty"}
    ordered = sorted(penalties, key=penalties.get, reverse=True)
    dominant = ordered[0]
    fraction = penalties[dominant] / max(sum(penalties.values()), 1e-15)
    class_map = {
        "DC_penalty": "DC_OFFSET_DOMINATES",
        "harmonic_penalty": "HARMONICS_DOMINATE",
        "transient_penalty": "TRANSIENT_CONTAMINATION_DOMINATES",
        "orthogonal_penalty": "ORTHOGONAL_MEAN_DIFFUSION_DOMINATES",
    }
    classification = class_map[dominant] if fraction >= .55 else "MULTIPLE_COMPONENTS"
    status = read_json(
        ARTIFACT_ROOT.parent / "google_pure_v19/experimental_public_analogue_matched/status.json")
    matched_fast = next(row for row in status["rows"] if row["label"] == "fast")
    transfer = matched_fast["mean_transfer_regression"]
    transfer_prediction = fundamental_improvement(
        transfer["gain"], transfer["phase_lag_radians"])
    value = {
        "pass": True,
        "input": "V19 matched fast stored trajectory",
        "analysis_epoch_window": [start, stop],
        "complete_periods": (stop - start) / round(1.0 / frequency),
        "production_finite_shot_I_mean": matched_fast["stream_decomposition"]["I_mean"],
        "reported_scalar_transfer_prediction": transfer_prediction,
        "reported_missing_normalized_cost": transfer_prediction -
            matched_fast["stream_decomposition"]["I_mean"],
        "exact_fundamental_to_full_missing_cost": total_unexplained,
        "exact_simulator_ablation": True,
        "exact_normalization": {
            "C_fixed_detector_events_per_shot": float(np.mean(fixed_costs)),
            "C_optimal_detector_events_per_shot": float(np.mean(optimal_costs)),
            "denominator": denominator,
        },
        "stages": stage_rows,
        "sequential_component_penalties": sequential,
        "quadratic_local_metric": quadratic,
        "hessian_coordinate_space": "V15_NORMALIZED_CONTROL_COORDINATES",
        "hessian_phase_state_epochs": selected,
        "hessian_diagonal_phase_states": hessian_rows.tolist(),
        "classification": classification,
        "dominant_component_fraction": fraction,
        "decomposition_conservation_max_abs": float(np.max(np.abs(
            sum(decomposition[key] for key in (
                "dc", "fundamental", "harmonic", "transient", "orthogonal")) - means))),
        "new_trajectory_generated": False,
        "forbidden_auto_runs_launched": [],
    }
    return write_artifact("fast_mean_cost_decomposition", value,
                          title="V20 fast mean-cost decomposition", notes=[
        f"Exact ablations classify the missing mean cost as `{classification}`.",
        "A through E use the same exact Stim detector-marginal evaluator and normalization.",
    ])


def _topology_regions() -> list[str]:
    rows = evaluator().plant.inventory
    centers = np.asarray([float(np.mean(item.qubits)) for item in rows])
    cuts = np.quantile(centers, [1 / 3, 2 / 3])
    return ["low-qubit-region" if center <= cuts[0] else
            "middle-qubit-region" if center <= cuts[1] else "high-qubit-region"
            for center in centers]


def audit_transfer_geometry() -> dict[str, Any]:
    verify_import_manifest()
    decomposition_path = ARTIFACT_ROOT / "fast_mean_cost_decomposition.json"
    decomposition = read_json(decomposition_path) if decomposition_path.is_file() \
        else decompose_fast_mean_cost()
    cfg = settings()
    run = load_matched_run("fast")
    start, stop = map(int, cfg["analysis_window"])
    records = run["records"][start:stop]
    epochs = np.arange(start, stop, dtype=float)
    frequency = float(cfg["fast_frequency_per_epoch"])
    trace = np.asarray([row["normalized_behavior_mean"] for row in records])
    hessian = np.mean(np.asarray(decomposition["hessian_diagonal_phase_states"]), axis=0)
    target_direction = np.mean(trace, axis=1)
    weighted_direction = (trace @ hessian) / float(np.sum(hessian))
    unweighted = sinusoidal_transfer(epochs, target_direction, frequency)
    weighted = sinusoidal_transfer(epochs, weighted_direction, frequency)
    plant = evaluator().plant
    families = np.asarray([item.gate_type for item in plant.inventory])
    regions = np.asarray(_topology_regions())
    family_rows = []
    region_rows = []
    target = np.asarray([plant.optimum(int(epoch), frequency) for epoch in epochs])
    error = trace - target
    coordinate_cost = np.mean(hessian[None, :] * error**2, axis=0)
    total_cost = float(np.sum(coordinate_cost))
    for label in sorted(set(families)):
        selected = families == label
        row = sinusoidal_transfer(epochs, np.mean(trace[:, selected], axis=1), frequency)
        row.update({"family": label, "coordinates": int(np.sum(selected)),
                    "quadratic_cost_fraction": float(np.sum(coordinate_cost[selected]) /
                                                       total_cost)})
        family_rows.append(row)
    for label in sorted(set(regions)):
        selected = regions == label
        row = sinusoidal_transfer(epochs, np.mean(trace[:, selected], axis=1), frequency)
        row.update({"region": label, "coordinates": int(np.sum(selected)),
                    "quadratic_cost_fraction": float(np.sum(coordinate_cost[selected]) /
                                                       total_cost)})
        region_rows.append(row)
    q = decomposition["quadratic_local_metric"]
    driven = q["self_costs"]["fundamental_tracking_error"] + q["self_costs"]["DC"] + \
        q["self_costs"]["harmonics"] + q["self_costs"]["transient"]
    orthogonal = q["self_costs"]["orthogonal"]
    high_family = max(family_rows, key=lambda row: row["quadratic_cost_fraction"])
    contradiction_disappears = (
        weighted["fundamental_quadratic_I_prediction"] < 0 and
        unweighted["fundamental_quadratic_I_prediction"] > 0)
    classification = ("TRANSFER_EVALUATION_GEOMETRY_MISMATCH" if contradiction_disappears
                      else "TRANSFER_WEIGHTING_DOES_NOT_EXPLAIN_SIGN_CONTRADICTION")
    value = {
        "pass": True,
        "unweighted_target_direction_transfer": unweighted,
        "hessian_weighted_transfer": weighted,
        "family_resolved_transfer": family_rows,
        "region_resolved_transfer": region_rows,
        "driven_subspace_cost_fraction": driven / max(driven + orthogonal, 1e-15),
        "orthogonal_cost_fraction": orthogonal / max(driven + orthogonal, 1e-15),
        "high_curvature_family": high_family["family"],
        "high_curvature_family_cost_fraction": high_family["quadratic_cost_fraction"],
        "positive_to_negative_contradiction_disappears_under_weighting":
            contradiction_disappears,
        "classification": classification,
        "training_changed": False,
        "forbidden_auto_runs_launched": [],
    }
    return write_artifact("transfer_evaluation_geometry_audit", value,
                          title="V20 transfer/evaluation geometry audit")
