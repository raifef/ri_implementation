"""Paper-family wrapper around the 924-coordinate source step analogue."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.contracts import PositivityGuard
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.optimizer import OptimizerConfig
from hdfa_rl_suite.google_pure_source_exact.step_response_130.acquisition import run_step_analogue
from hdfa_rl_suite.google_pure_source_exact.step_response_130.contracts import StepProtocol
from hdfa_rl_suite.google_pure_v7.config import repository_root

from .common import canonical_hash, controller_config


def _checkpoint(protocol: Mapping[str, Any], condition: Mapping[str, Any]) -> Path:
    identity = canonical_hash({"protocol_hash": protocol["protocol_hash"], "condition": dict(condition)})[:24]
    return repository_root() / "artifacts/google_pure_source_exact/paper_families/checkpoints/step" / f"{identity}.json"


def acquire_step_condition(protocol: Mapping[str, Any], condition: Mapping[str, Any]) -> dict[str, Any]:
    config = protocol["config"]
    seed, severity = int(condition["seed"]), float(condition["severity"])
    source = StepProtocol(
        profile=f"paper-family-{protocol['mode']}", distance=5, controls=int(config["controls"]),
        candidates_per_epoch=int(config["candidates"]),
        cycles_per_candidate=int(config["cycles_per_candidate"]),
        epochs=int(config["epochs"]), onset_epoch=int(config["onset_epoch"]),
        direction_coordinate=int(config["direction_coordinate"]),
        target_delta_normalized=severity, seed=seed,
        certification=protocol["mode"] in {"reference", "paper-scale"})
    value = controller_config()
    optimizer = OptimizerConfig(
        float(value["mean_learning_rate"]), float(value["sigma_learning_rate"]),
        float(value["baseline_learning_rate"]), minimum_sigma=float(value["minimum_sigma"]),
        maximum_sigma=float(value["maximum_sigma"]),
        positivity_guard=PositivityGuard(value["positivity_guard"]))
    result = run_step_analogue(
        source, _checkpoint(protocol, condition), optimizer,
        initial_sigma=float(value["initial_sigma"]), entropy_weight=float(config["entropy_coefficient"]),
        baseline_weight=float(value["baseline_weight"]), clip=float(value["ppo_clip"]), resume=True,
        checkpoint_every_candidates=int(config.get("checkpoint_every_candidates", 1)),
        compact_records=bool(config.get("compact_records", False)),
        experiment_family="STEP_RESPONSE_INJECTED_DRIFT",
        fresh_acquisition_required=bool(protocol.get("fresh_acquisition_required", False)),
        source_budget_profile=str(protocol.get("source_budget_profile", protocol["mode"])))
    records = result["records"]
    direction = source.direction_coordinate
    mean_response = [row["mean_after_direction"] if "mean_after_direction" in row else
                     row["mean_after"][direction] for row in records]
    sigma_response = [row["sigma_direction"] if "sigma_direction" in row else
                      row["sigma"][direction] for row in records]
    return {
        "seed": seed, "severity": severity, "distance": 5, "control_count": source.controls,
        "onset_epoch": source.onset_epoch, "response": result["response"],
        "controller_mode": "PAPER_DIRECT_SIGMA", "parameterization": "direct_sigma",
        "ratio_clipping_mode": "SOURCE_ELEMENTWISE_COORDINATE_CLIPPING",
        "baseline_mode": "JOINT_LEARNED_DETECTOR_BASELINE",
        "candidate_qec_cycles": source.epochs * source.candidates_per_epoch * source.cycles_per_candidate,
        "checkpoint_every_candidates":result["checkpoint_every_candidates"],
        "record_storage":result["record_storage"],
        "controller_target_access": False, "controller_direction_access": False,
        "policy_spoil_applied": False, "plant_instance_hash": result["plant_hash"],
        "v15_provenance": {key: result[key] for key in (
            "implementation_version", "sensitivity_map_hash", "sensitivity_definition_hash",
            "calibration_bundle_hash", "detector_degree_audit_hash", "boundary_transform_hash",
            "boundary_transform_name", "boundary_apply_count", "control_order_hash",
            "expanded_scale_hash", "fresh_acquisition", "reused_shard_ids", "source_budget_profile")},
        "boundary_trace": result["boundary_trace"],
        "trajectory": {
            "normalized_projected_policy_response": [value / source.target_delta_normalized
                                                       for value in mean_response],
            "learned_mean_edr": [row["learned_mean_edr"] for row in records],
            "fixed_edr": [row["fixed_edr"] for row in records],
            "oracle_edr": [row["oracle_edr"] for row in records],
            "mean_policy_sigma": sigma_response,
            "direction_snr": [row["direction_snr"] for row in records],
        },
        "source_structure_match": True, "paper_comparable": False,
        "blocking_reasons": ["the proprietary Willow 924-control inventory and detector mask are unavailable"],
    }
