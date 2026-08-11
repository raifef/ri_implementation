"""Figure 5a steerability conditions through the amended 41-control Stim backend."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from hdfa_rl_suite.google_pure_source_exact.figure5a.acquisition import run_cell
from hdfa_rl_suite.google_pure_source_exact.figure5a.contracts import AcquisitionMode, Figure5aProtocol, canonical_hash
from hdfa_rl_suite.google_pure_source_exact.figure5a.validation import build_plant, dependency_hashes
from hdfa_rl_suite.google_pure_source_exact.identity import build_direct_sigma_identity
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.contracts import PositivityGuard
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.optimizer import (
    GradientClippingMode,
    OptimizerConfig,
)
from hdfa_rl_suite.google_pure_v7.config import repository_root


def acquire_condition(protocol: Mapping[str, Any], condition: Mapping[str, Any]) -> dict[str, Any]:
    root = repository_root()
    source_config = json.loads((root / "configs/google_pure_source_exact/figure5a.json").read_text(encoding="utf-8"))
    plant = build_plant(source_config); config = protocol["config"]
    mode = AcquisitionMode.REFERENCE if protocol["mode"] in {"reference", "paper-scale"} else AcquisitionMode(protocol["mode"])
    if mode == AcquisitionMode.REFERENCE and not source_config["controller"]["reference_hyperparameters_frozen"]:
        raise RuntimeError(
            "reference Figure 5a is blocked until clipping and learning rates are frozen "
            "from the 50-candidate development ladder")
    source_protocol = Figure5aProtocol(mode, int(config["epochs"]), int(config["candidates"]),
                                      int(config["cycles_per_candidate"]), int(source_config["plant"]["circuit_rounds"]))
    value = source_config["controller"]
    clipping = value["gradient_clipping"]
    optimizer = OptimizerConfig(float(value["mean_learning_rate"]), float(value["sigma_learning_rate"]),
        float(value["baseline_learning_rate"]), minimum_sigma=float(value["minimum_sigma"]),
        maximum_sigma=float(value["maximum_sigma"]), positivity_guard=PositivityGuard(value["positivity_guard"]),
        gradient_clipping_mode=GradientClippingMode(clipping["selected_mode"]),
        gradient_clip_threshold=float(clipping["selected_threshold"]))
    identity = build_direct_sigma_identity(root)
    cell_id = canonical_hash({"protocol_hash":protocol["protocol_hash"],"condition":dict(condition)})[:24]
    checkpoint = root / "artifacts/google_pure_source_exact/paper_families/checkpoints/figure5a" / f"{cell_id}.json"
    result = run_cell(protocol=source_protocol, plant=plant, frequency=float(condition["frequency"]),
        entropy_weight=float(condition["entropy_coefficient"]), seed=int(condition["seed"]),
        optimizer_config=optimizer, initial_sigma=float(value["initial_sigma"]), checkpoint_path=checkpoint,
        dependency_hashes=dependency_hashes(root,source_config), controller_hash=identity["controller_hash"],
        clip=float(value["ppo_clip"]), baseline_weight=float(value["baseline_weight"]), resume=checkpoint.exists(),
        checkpoint_every_candidates=int(config.get("checkpoint_every_candidates", 1)),
        fresh_acquisition_required=bool(protocol.get("fresh_acquisition_required", False)),
        source_budget_profile=str(protocol.get("source_budget_profile", protocol["mode"])))
    records=result["epoch_records"]
    return {"seed":int(condition["seed"]),"frequency":float(condition["frequency"]),
        "entropy_coefficient":float(condition["entropy_coefficient"]),
        "improvement_candidate":float(result["stochastic_ratio"]["source_ratio"]),
        "improvement_mean":float(result["learned_mean_ratio"]["source_ratio"]),
        "controller_mode":"PAPER_DIRECT_SIGMA","parameterization":"direct_sigma",
        "ratio_clipping_mode":"SOURCE_ELEMENTWISE_COORDINATE_CLIPPING",
        "baseline_mode":"JOINT_LEARNED_DETECTOR_BASELINE","control_count":41,
        "plant_instance_hash":plant.plant_hash,"graph_instance_hash":canonical_hash(plant.mask.astype(int).tolist()),
        "candidate_qec_cycles":result["candidate_qec_cycles"],"four_stream_qec_cycles":result["four_stream_qec_cycles"],
        "checkpoint_every_candidates":result["checkpoint_every_candidates"],
        "source_coordinate_provenance": {key: result[key] for key in (
            "implementation_version", "coordinate_contract", "action_execution",
            "plant_boundary_execution", "likelihood_space", "entropy_space",
            "empirical_relative_normalization_applied", "mean_bounds_applied",
            "control_order_hash", "fresh_acquisition", "reused_shard_ids",
            "source_budget_profile")},
        "normalization_contract":"(C_fixed-C_candidates)/(C_fixed-C_optimal)",
        "mean_policy_reported_separately":True,"source_structure_match":True,"paper_comparable":False,
        "blocking_reasons":["the proprietary Figure 5a plant ensemble and optimizer hyperparameters are unavailable"],
        "trajectory":{"epoch":[row["epoch"] for row in records],
            "learned_mean":[row["post_update_mean"][0] for row in records],
            "optimum":[row["optimum"] for row in records],
            "mean_sigma":[float(np.mean(row["post_update_sigma"])) for row in records]}}


def validation(rows: list[dict[str, Any]], mode: str) -> tuple[bool, list[str], dict[str, Any]]:
    reasons = []
    if any(row.get("normalization_contract") != "(C_fixed-C_candidates)/(C_fixed-C_optimal)" for row in rows): reasons.append("wrong candidate normalization")
    if any(not row.get("mean_policy_reported_separately") for row in rows): reasons.append("candidate/mean policy conflation")
    if any(row.get("controller_mode")!="PAPER_DIRECT_SIGMA" or row.get("parameterization")!="direct_sigma" for row in rows):
        reasons.append("amended direct-sigma controller did not execute")
    if mode in {"reference","paper-scale"}:
        from .direct_path import integration_manifest
        integration=integration_manifest()
        if any(row.get("plant_instance_hash")!=integration["plant_hash"] for row in rows):
            reasons.append("Figure 5a did not execute the integrated 41-parameter Stim plant")
        if any(row.get("graph_instance_hash")!=integration["graph_hash"] for row in rows):
            reasons.append("Figure 5a did not execute the integrated Stim-derived graph")
    values = [float(row["improvement_candidate"]) for row in rows]
    if mode in {"reference", "paper-scale"} and values and not min(values) <= 0 <= max(values): reasons.append("zero contour is not bracketed")
    return not reasons, reasons, {"candidate_improvement_range": [min(values), max(values)] if values else None,
                                  "zero_contour_bracketed": bool(values and min(values) <= 0 <= max(values)),
                                  "paper_critical_frequency": 1/150,
                                  "source_structure_match":all(row.get("source_structure_match") for row in rows),
                                  "paper_comparable":False,
                                  "blocking_reasons":["the proprietary Figure 5a simulator is unavailable"]}
