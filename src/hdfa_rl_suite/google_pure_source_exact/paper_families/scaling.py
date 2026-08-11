"""Direct-sigma sparse scaling acquisitions for Figure 5b and Figure 5c."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from hdfa_rl_suite.google_pure_v7.figure5.accounting import (
    detector_factors,
    physical_qubits,
    total_controls,
)
from hdfa_rl_suite.google_pure_v7.config import repository_root

from .common import SparseControlPlant, canonical_hash, run_direct_sigma_trace


def _checkpoint(protocol: Mapping[str, Any], condition: Mapping[str, Any]) -> Path:
    identity = canonical_hash({"protocol_hash": protocol["protocol_hash"],
                               "condition": dict(condition)})[:24]
    return repository_root() / "artifacts/google_pure_source_exact/paper_families/checkpoints/scaling" / f"{identity}.json"


def acquire_scaling_condition(protocol: Mapping[str, Any],
                              condition: Mapping[str, Any]) -> dict[str, Any]:
    family = str(protocol["experiment_family"])
    distance = int(condition["distance"])
    parameters_per_gate = int(condition["parameters_per_gate"])
    seed = int(condition["seed"])
    controls = total_controls(distance, parameters_per_gate)
    detectors = detector_factors(distance)
    plant = SparseControlPlant(distance, controls, detectors,
                               seed=9100 + 101 * distance + parameters_per_gate)
    rng = np.random.default_rng(seed)
    initial_mean = rng.choice((-1.0, 1.0), controls) * rng.uniform(.45, .75, controls)
    config = protocol["config"]
    result = run_direct_sigma_trace(
        plant=plant, protocol_hash=str(protocol["protocol_hash"]), seed=seed,
        epochs=int(config["epochs"]), candidates=int(config["candidates"]),
        cycles_per_candidate=int(config["cycles_per_candidate"]),
        entropy_weight=float(config["entropy_coefficient"]),
        checkpoint=_checkpoint(protocol, condition), target_at_epoch=lambda _: np.zeros(controls),
        initial_mean=initial_mean, experiment_family=family,
        fresh_acquisition_required=bool(protocol.get("fresh_acquisition_required", False)),
        source_budget_profile=str(protocol.get("source_budget_profile", protocol["mode"])))
    records = result["records"]
    lambdas = np.asarray([row["learned"]["lambda"] for row in records])
    lambda_star = float(records[0]["oracle"]["lambda_star"])
    ratio = lambdas / lambda_star
    physical = [row["learned"]["physical_error"] for row in records]
    logical = [row["learned"]["logical_error"] for row in records]
    logical_fixed = [row["fixed"]["logical_error"] for row in records]
    logical_candidate = [float(np.clip(
        .01 * (plant.threshold_physical_error / max(row["candidate_physical_error"], 1e-12)) **
        (-(distance + 1) / 2), 1e-12, 1.0)) for row in records]
    logical_floor = float(records[0]["oracle"]["logical_floor"])
    x = 1.0 - ratio[:-1]
    y = 100.0 * np.diff(lambdas) / lambda_star
    fit_min = float(config.get("local_fit_min_distance", 1e-4))
    fit_max = float(config.get("local_fit_max_distance", .7))
    keep = (x > fit_min) & (x < fit_max) & np.isfinite(x) & np.isfinite(y)
    if np.any(keep) and float(np.dot(x[keep], x[keep])) > 0:
        slope = float(np.dot(x[keep], y[keep]) / np.dot(x[keep], x[keep]))
        prediction = slope * x[keep]
        denominator = float(np.sum((y[keep] - np.mean(y[keep])) ** 2))
        r_squared = 1.0 - float(np.sum((y[keep] - prediction) ** 2)) / denominator if denominator else 0.0
    else:
        # A reduced smoke trace can contain no points in the preregistered local
        # window.  Preserve that as a zero-information fit rather than emitting
        # non-JSON NaNs; reference validation rejects its R-squared value.
        slope, r_squared = 0.0, 0.0
    return {
        "distance": distance, "parameters_per_gate": parameters_per_gate, "seed": seed,
        "total_controls": controls, "physical_qubits": physical_qubits(distance),
        "detectors": detectors, "plant_instance_hash": result["plant_hash"],
        "graph_instance_hash": result["graph_hash"], "candidate_qec_cycles": result["candidate_qec_cycles"],
        "controller_mode": "PAPER_DIRECT_SIGMA", "parameterization": "direct_sigma",
        "ratio_clipping_mode": "SOURCE_ELEMENTWISE_COORDINATE_CLIPPING",
        "baseline_mode": "JOINT_LEARNED_DETECTOR_BASELINE",
        "direct_sigma_epoch_count": len(records), "nonzero_qec_cycles_executed": result["candidate_qec_cycles"] > 0,
        "controller_target_access": result["controller_observed_target"],
        "v15_provenance": {key: result[key] for key in (
            "implementation_version", "sensitivity_map_hash", "sensitivity_definition_hash",
            "calibration_bundle_hash", "detector_degree_audit_hash", "boundary_transform_hash",
            "boundary_transform_name", "boundary_apply_count", "control_order_hash",
            "expanded_scale_hash", "fresh_acquisition", "reused_shard_ids", "source_budget_profile")},
        "boundary_trace": result["boundary_trace"],
        "dense_parameter_matrix_allocated": result["dense_parameter_matrix_allocated"],
        "logical_floor": logical_floor, "logical_initial": logical[0],
        "lambda_star": lambda_star, "gamma_times_100": slope,
        "convergence_fit_r_squared": r_squared,
        "paper_physical_error_axis_present": True, "paper_logical_error_axis_present": True,
        "paper_log_axes": True, "epoch_colour_present": True,
        "irreducible_floor_bars_present": True,
        "source_structure_match": True, "paper_comparable": False,
        "blocking_reasons": ["the proprietary Figure 5 scaling plant and optimizer hyperparameters are unavailable"],
        "trajectory": {
            "epoch": list(range(len(records))), "physical_error": physical,
            "logical_learned": logical, "logical_candidate": logical_candidate,
            "logical_fixed": logical_fixed, "logical_oracle": [logical_floor] * len(records),
            "logical_floor": [logical_floor] * len(records), "lambda": lambdas.tolist(),
            "lambda_ratio": ratio.tolist(), "x_distance": x.tolist(),
            "normalized_speed": y.tolist(), "fit_mask": keep.astype(int).tolist(),
            "mean_sigma": [row["mean_sigma"] for row in records],
            "component_clip_fraction": [row["component_clip_fraction"] for row in records],
        },
    }
