"""Direct-sigma recovery after a randomized distance-5 policy spoil."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from hdfa_rl_suite.google_pure_v7.config import repository_root

from .common import SparseControlPlant, canonical_hash, run_direct_sigma_trace


def _checkpoint(protocol: Mapping[str, Any], condition: Mapping[str, Any]) -> Path:
    identity = canonical_hash({"protocol_hash": protocol["protocol_hash"], "condition": dict(condition)})[:24]
    return repository_root() / "artifacts/google_pure_source_exact/paper_families/checkpoints/recovery" / f"{identity}.json"


def acquire_recovery_condition(protocol: Mapping[str, Any],
                               condition: Mapping[str, Any]) -> dict[str, Any]:
    config = protocol["config"]
    seed, severity = int(condition["seed"]), float(condition["severity"])
    controls = int(config["controls"])
    plant = SparseControlPlant(5, controls, 24, seed=10_100, curvature=.004)
    rng = np.random.default_rng(seed)
    spoiled = np.zeros(controls)
    selected = rng.choice(controls, max(1, controls // 2), replace=False)
    spoiled[selected] = rng.choice((-1.0, 1.0), len(selected)) * severity
    result = run_direct_sigma_trace(
        plant=plant, protocol_hash=str(protocol["protocol_hash"]), seed=seed,
        epochs=int(config["epochs"]), candidates=int(config["candidates"]),
        cycles_per_candidate=int(config["cycles_per_candidate"]),
        entropy_weight=float(config["entropy_coefficient"]), checkpoint=_checkpoint(protocol, condition),
        target_at_epoch=lambda _: np.zeros(controls), initial_mean=spoiled,
        experiment_family="RANDOMIZED_RECOVERY_AFTER_SPOIL",
        fresh_acquisition_required=bool(protocol.get("fresh_acquisition_required", False)),
        source_budget_profile=str(protocol.get("source_budget_profile", protocol["mode"])))
    logical = np.asarray([row["learned"]["logical_error"] for row in result["records"]])
    floor = float(result["records"][0]["oracle"]["logical_floor"])
    excess = np.maximum(logical - floor, 0.0)
    initial = float(max(np.mean(excess[:max(1, min(5, len(excess)))]), np.finfo(float).tiny))
    sustained = min(int(config["sustained_epochs"]), len(excess))
    moving = np.convolve(excess, np.ones(sustained) / sustained, mode="valid")
    hits = np.flatnonzero(moving <= .1 * initial)
    recovery_epoch = int(hits[0] + sustained - 1) if hits.size else None
    return {
        "seed": seed, "severity": severity, "distance": 5, "control_count": controls,
        "randomized_fraction": .5, "spoil_vector_hash": canonical_hash(spoiled.tolist()),
        "initial_excess": initial, "recovery_epoch": recovery_epoch,
        "censored": recovery_epoch is None, "not_a_step_response": True,
        "controller_mode": "PAPER_DIRECT_SIGMA", "parameterization": "direct_sigma",
        "ratio_clipping_mode": "SOURCE_ELEMENTWISE_COORDINATE_CLIPPING",
        "baseline_mode": "JOINT_LEARNED_DETECTOR_BASELINE",
        "candidate_qec_cycles": result["candidate_qec_cycles"],
        "plant_instance_hash": result["plant_hash"], "graph_instance_hash": result["graph_hash"],
        "controller_target_access": result["controller_observed_target"],
        "v15_provenance": {key: result[key] for key in (
            "implementation_version", "sensitivity_map_hash", "sensitivity_definition_hash",
            "calibration_bundle_hash", "detector_degree_audit_hash", "boundary_transform_hash",
            "boundary_transform_name", "boundary_apply_count", "control_order_hash",
            "expanded_scale_hash", "fresh_acquisition", "reused_shard_ids", "source_budget_profile")},
        "boundary_trace": result["boundary_trace"],
        "trajectory": {"learned_mean_excess_logical_risk": excess.tolist(),
                       "learned_mean_logical_error": logical.tolist(),
                       "fixed_logical_error": [row["fixed"]["logical_error"] for row in result["records"]],
                       "mean_policy_sigma": [row["mean_sigma"] for row in result["records"]],
                       "candidate_physical_error": [row["candidate_physical_error"] for row in result["records"]]},
        "source_structure_match": True, "paper_comparable": False,
        "blocking_reasons": ["the randomized Willow policy and proprietary 924-control detector map are unavailable"],
    }
