"""Immediate V15 execution-path, normalization, and freshness repair.

All artifacts are development diagnostics.  No function in this module consumes
held-out seeds, launches a source-budget run, or promotes simulator evidence.
"""
from __future__ import annotations

from pathlib import Path
import json
import time
from typing import Any, Mapping

import numpy as np

from hdfa_rl_suite.google_pure_paper_reproduction.experiment_families import ExperimentFamily
from hdfa_rl_suite.google_pure_paper_reproduction.paper_figures import acquire, build_protocol, merge_protocol
from hdfa_rl_suite.google_pure_paper_reproduction.storage import initialise_layout
from hdfa_rl_suite.google_pure_source_exact.figure5a.validation import build_plant
from hdfa_rl_suite.google_pure_source_exact.paper_families.common import (
    SparseControlPlant,
    _sparse_source_loss,
    controller_config,
    optimizer_config,
)
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.gaussian import (
    DirectSigmaGaussianPolicy,
)
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.optimizer import (
    DirectSigmaOptimizer,
)
from hdfa_rl_suite.google_pure_source_exact.source_normalization import (
    IMPLEMENTATION_VERSION,
    SourceNormalizationBoundary,
    canonical_hash,
    file_hash,
    require_v15_boundary_provenance,
    source_normalization_inputs,
)
from hdfa_rl_suite.google_pure_source_exact.step_response_130.plant import SourceStepPlant
from hdfa_rl_suite.google_pure_v7.config import repository_root
from hdfa_rl_suite.google_pure_v7.figure5.accounting import detector_factors, total_controls
from hdfa_rl_suite.google_pure_v12.directional import reference_directional_curvature

from .io import atomic_json, atomic_text


ROOT = repository_root()
OUTPUT = ROOT / "artifacts/google_pure_v15/immediate_execution_audit"
NONFINAL = {
    "scientifically_valid": False,
    "final_evidence": False,
    "paper_equivalence_claim_permitted": False,
    "heldout_seeds_consumed": False,
    "long_run_auto_launched": False,
}
F = ExperimentFamily


def _code_inputs() -> dict[str, str]:
    relative = (
        "src/hdfa_rl_suite/google_pure_source_exact/source_normalization.py",
        "src/hdfa_rl_suite/google_pure_source_exact/figure5a/acquisition.py",
        "src/hdfa_rl_suite/google_pure_source_exact/step_response_130/acquisition.py",
        "src/hdfa_rl_suite/google_pure_source_exact/paper_families/recovery.py",
        "src/hdfa_rl_suite/google_pure_source_exact/paper_families/scaling.py",
        "src/hdfa_rl_suite/google_pure_paper_reproduction/panel_a.py",
        "src/hdfa_rl_suite/google_pure_paper_reproduction/step_response.py",
        "src/hdfa_rl_suite/google_pure_paper_reproduction/randomized_recovery.py",
        "src/hdfa_rl_suite/google_pure_paper_reproduction/panel_b.py",
        "src/hdfa_rl_suite/google_pure_paper_reproduction/storage.py",
        "src/hdfa_rl_suite/google_pure_paper_reproduction/hourly_workflow.py",
    )
    return {path: file_hash(ROOT / path) for path in relative}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_execution_path() -> dict[str, Any]:
    """Freeze exact inputs and prove whether pre-repair plots exercised V15."""
    manifests = sorted((ROOT / "artifacts/google_pure_paper_reproduction/manifests").glob("*_merge.json"))
    rows = []
    for path in manifests:
        value = _load_json(path)
        provenance = value.get("provenance", {})
        rows.append({
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": file_hash(path),
            "experiment_family": value.get("experiment_family"),
            "protocol_hash": value.get("protocol_hash"),
            "implementation_version": provenance.get("implementation_version"),
            "controller_mode": provenance.get("controller_mode"),
            "sensitivity_map_hash": provenance.get("sensitivity_map_hash"),
            "boundary_transform_hash": provenance.get("boundary_transform_hash"),
            "fresh_acquisition": provenance.get("fresh_acquisition"),
            "v15_execution_provenance_complete": all(provenance.get(key) not in {None, ""} for key in (
                "implementation_version", "sensitivity_map_hash", "sensitivity_definition_hash",
                "calibration_bundle_hash", "detector_degree_audit_hash",
                "boundary_transform_hash", "experiment_driver_hash")),
        })
    legacy = [row for row in rows if not row["v15_execution_provenance_complete"]]
    inputs = {
        "schema_version": "google-pure-v15-immediate-execution-inputs.v1",
        "implementation_version": IMPLEMENTATION_VERSION,
        "lineage_sequence": ["V12", "V13", "V15"],
        "v14_exists": False,
        "source_normalization_inputs": source_normalization_inputs(),
        "code_inputs": _code_inputs(),
        "prior_merge_manifests": rows,
        "prior_manifest_count": len(rows),
        "prior_legacy_or_unverifiable_manifest_count": len(legacy),
        "conclusion": (
            "PREVIOUS_RESULTS_DID_NOT_PROVE_V15_DRIVER_EXECUTION" if legacy
            else "ALL_DISCOVERED_RESULTS_HAVE_V15_EXECUTION_PROVENANCE"),
        **NONFINAL,
    }
    inputs["execution_inputs_hash"] = canonical_hash(
        {key: value for key, value in inputs.items() if key != "execution_inputs_hash"})
    atomic_json(OUTPUT / "execution_inputs.json", inputs)
    lines = [
        "# V15 immediate execution inputs", "",
        f"Conclusion: **{inputs['conclusion']}**", "",
        "The final source-normalization implementation is V15. The intentional lineage is V12 → V13 → V15; there is no V14.", "",
        f"Discovered merge manifests: {len(rows)}; missing complete V15 execution lineage: {len(legacy)}.", "",
        "A legacy plot is not evidence that the current normalization boundary reached its experiment driver.",
    ]
    atomic_text(OUTPUT / "execution_inputs.md", "\n".join(lines))
    return inputs


def _step_boundary() -> SourceNormalizationBoundary:
    plant = SourceStepPlant()
    return SourceNormalizationBoundary.from_training_objective(
        F.STEP_RESPONSE_INJECTED_DRIFT.value, plant.sensitivity,
        control_ids=[f"step:control:{index}" for index in range(plant.controls)])


def _recovery_boundary() -> tuple[SparseControlPlant, SourceNormalizationBoundary]:
    plant = SparseControlPlant(5, 924, 24, seed=10_100, curvature=.004)
    return plant, SourceNormalizationBoundary.from_training_objective(
        F.RANDOMIZED_RECOVERY_AFTER_SPOIL.value, plant.connected_objective_curvature,
        control_ids=plant.control_ids)


def _figure5b_boundary() -> tuple[SparseControlPlant, SourceNormalizationBoundary]:
    distance, parameters = 3, 1
    plant = SparseControlPlant(distance, total_controls(distance, parameters),
                               detector_factors(distance), seed=9100 + 101 * distance + parameters)
    return plant, SourceNormalizationBoundary.from_training_objective(
        F.FIGURE5B_SPARSE_SCALING.value, plant.connected_objective_curvature,
        control_ids=plant.control_ids)


def _trace(name: str, boundary: SourceNormalizationBoundary) -> dict[str, Any]:
    candidate = np.zeros(len(boundary.control_ids))
    candidate[0] = .25
    if len(candidate) > 7:
        candidate[7] = -.125
    result = boundary.trace(candidate, indices=np.flatnonzero(candidate).tolist())
    result.update({"diagnostic": name, "expected_formula": "u_i = u0_i + s_i*x_i", **NONFINAL})
    atomic_json(OUTPUT / "boundary_traces" / f"{name}.json", result)
    return result


def trace_boundary_step() -> dict[str, Any]:
    return _trace("step", _step_boundary())


def trace_boundary_recovery() -> dict[str, Any]:
    return _trace("recovery", _recovery_boundary()[1])


def trace_boundary_figure5b() -> dict[str, Any]:
    return _trace("figure5b_d3_p1", _figure5b_boundary()[1])


def compare_v12_v15_scales() -> dict[str, Any]:
    reference = reference_directional_curvature()
    step = SourceStepPlant()
    step_v12 = np.sqrt(reference / step.sensitivity)
    step_v15 = _step_boundary().native_scale
    recovery, recovery_boundary = _recovery_boundary()
    recovery_v12 = np.sqrt(reference / recovery.connected_objective_curvature)
    rows = []
    for family, v12, v15 in (
        (F.STEP_RESPONSE_INJECTED_DRIFT.value, step_v12, step_v15),
        (F.RANDOMIZED_RECOVERY_AFTER_SPOIL.value, recovery_v12, recovery_boundary.native_scale),
    ):
        rows.append({
            "experiment_family": family,
            "control_count": len(v12),
            "v12_scale_min": float(np.min(v12)), "v12_scale_median": float(np.median(v12)),
            "v12_scale_max": float(np.max(v12)),
            "v15_scale_min": float(np.min(v15)), "v15_scale_median": float(np.median(v15)),
            "v15_scale_max": float(np.max(v15)),
            "v15_over_v12_min": float(np.min(v15 / v12)),
            "v15_over_v12_max": float(np.max(v15 / v12)),
            "maps_identical": bool(np.array_equal(v12, v15)),
            "difference_source": "V12 uses Figure5a outcome-derived reference curvature; V15 uses source-defined 0.01 EDR fraction",
        })
    result = {
        "schema_version": "google-pure-v15-scale-comparison.v1",
        "v12_reference_directional_curvature": reference,
        "v15_reference_edr_fraction": .01,
        "rows": rows,
        "v12_source_equivalence_established": False,
        **NONFINAL,
    }
    atomic_json(OUTPUT / "v12_v15_scale_comparison.json", result)
    return result


def audit_calibration_objective() -> dict[str, Any]:
    step = _step_boundary()
    recovery, recovery_map = _recovery_boundary()
    figure5b, figure5b_map = _figure5b_boundary()
    rows = []
    for name, plant, boundary in (
        ("step", SourceStepPlant(), step),
        ("recovery", recovery, recovery_map),
        ("figure5b_d3_p1", figure5b, figure5b_map),
    ):
        conditioned = boundary.native_objective_curvature * np.square(boundary.native_scale)
        rows.append({
            "driver": name,
            "control_count": len(conditioned),
            "training_objective": "sum connected detector rewards, then one mean over candidates",
            "minimum_conditioned_curvature": float(np.min(conditioned)),
            "maximum_conditioned_curvature": float(np.max(conditioned)),
            "target_conditioned_curvature": .01,
            "exact_objective_match": bool(np.allclose(conditioned, .01, rtol=0, atol=2e-15)),
            **boundary.provenance_fields(),
        })
    result = {
        "schema_version": "google-pure-v15-calibration-objective-audit.v1",
        "rows": rows,
        "all_drivers_calibrated_to_the_objective_consumed_by_training": all(
            row["exact_objective_match"] for row in rows),
        "extra_detector_degree_factor_applied": False,
        **NONFINAL,
    }
    atomic_json(OUTPUT / "calibration_objective_audit.json", result)
    return result


def _run_step_scale_arm(scale: np.ndarray, branch: str, *, seed: int,
                        epochs: int, candidates: int, cycles: int) -> dict[str, Any]:
    plant = SourceStepPlant(onset_epoch=60)
    policy_cfg = controller_config()
    policy = DirectSigmaGaussianPolicy(
        np.zeros(plant.controls), np.full(plant.controls, float(policy_cfg["initial_sigma"])),
        seed=int(seed))
    optimizer = DirectSigmaOptimizer(plant.controls, plant.detectors, optimizer_config())
    baseline = np.zeros(plant.detectors)
    owners = np.arange(plant.controls, dtype=np.int64) % plant.detectors
    normalized_target = np.zeros(plant.controls)
    normalized_target[plant.direction_coordinate] = plant.target_delta
    native_target = scale * normalized_target
    progress, candidate_damage = [], []
    gradient_projection, gradient_projection_se = [], []
    for epoch in range(epochs):
        batch = policy.sample(candidates)
        active_target = native_target if epoch >= plant.onset_epoch else np.zeros(plant.controls)
        native_actions = batch.actions * scale[None, :]
        probabilities = np.asarray([
            plant.expected_edr(row, epoch, target_controls=active_target)
            for row in native_actions])
        stream = int(canonical_hash(["v15-paired-abc-step", seed, epoch])[:16], 16)
        counts = np.random.default_rng(stream).binomial(cycles, probabilities)
        rewards = -counts / float(cycles)
        # At the fresh-policy update the importance ratio is one. Per-candidate
        # influence values expose the sign and uncertainty of the step direction.
        score0 = (batch.actions[:, 0] - policy.mean[0]) / (policy.sigma[0] ** 2)
        influence = (rewards[:, owners[0]] - baseline[owners[0]]) * score0
        gradient_projection.append(float(np.mean(influence)))
        gradient_projection_se.append(float(np.std(influence, ddof=1) / np.sqrt(candidates)))
        loss = _sparse_source_loss(
            batch.actions, rewards, owners, policy.mean, policy.sigma, baseline,
            batch.behavior, clip=float(policy_cfg["ppo_clip"]),
            entropy_weight=.001, baseline_weight=float(policy_cfg["baseline_weight"]))
        optimizer.step(policy.mean, policy.sigma, baseline,
                       loss["grad_mean"], loss["grad_sigma"], loss["grad_baseline"],
                       mean_bounds=(-2.0, 2.0))
        policy.policy_version += 1
        progress.append(float(policy.mean[0] / plant.target_delta))
        candidate_damage.append(float(np.mean(probabilities)))
    eligible = np.asarray(progress[plant.onset_epoch:])
    crossings = np.flatnonzero(eligible >= .5)
    return {
        "normalization_branch": branch, "seed": int(seed), "epochs": epochs,
        "onset_epoch": plant.onset_epoch, "candidate_count_per_epoch": candidates,
        "cycles_per_candidate": cycles,
        "final_target_fraction": float(progress[-1]),
        "response_time_50_epochs_after_onset": int(crossings[0]) if crossings.size else None,
        "target_relative_progress": progress,
        "candidate_mean_edr": candidate_damage,
        "directional_gradient_projection": gradient_projection,
        "directional_gradient_projection_standard_error": gradient_projection_se,
        "native_per_normalized_min": float(np.min(scale)),
        "native_per_normalized_max": float(np.max(scale)),
        "scale_hash": canonical_hash(scale.tolist()),
        "controller_mode": "PAPER_DIRECT_SIGMA", "parameterization": "direct_sigma",
        "controller_hyperparameter_hash": canonical_hash(policy_cfg),
        "standard_normal_tape_id": canonical_hash(["DirectSigmaGaussianPolicy", int(seed)]),
        "count_random_stream_rule": "sha256(v15-paired-abc-step,seed,epoch)",
        "controller_target_access": False, "normalization_application_count": 0 if branch.startswith("A_") else 1,
        **NONFINAL,
    }


def _step_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    finals = np.asarray([row["final_target_fraction"] for row in rows])
    times = [row["response_time_50_epochs_after_onset"] for row in rows
             if row["response_time_50_epochs_after_onset"] is not None]
    return {
        "run_count": len(rows), "median_final_target_fraction": float(np.median(finals)),
        "minimum_final_target_fraction": float(np.min(finals)),
        "maximum_final_target_fraction": float(np.max(finals)),
        "response_identified_count": len(times),
        "median_response_time_50_epochs_after_onset": float(np.median(times)) if times else None,
        "median_final_directional_gradient_projection": float(np.median([
            row["directional_gradient_projection"][-1] for row in rows])),
        "median_final_gradient_projection_standard_error": float(np.median([
            row["directional_gradient_projection_standard_error"][-1] for row in rows])),
    }


def run_abc_step() -> dict[str, Any]:
    """Run a paired, equal-budget A/B/C test with the actual production V15 map."""
    epochs, candidates, cycles = 120, 16, 36_000
    seeds = (57301, 57302, 57303)
    plant = SourceStepPlant(onset_epoch=60)
    v12_scale = np.sqrt(reference_directional_curvature() / plant.sensitivity)
    boundary = _step_boundary()
    branch_scales = (
        ("A_LEGACY_UNSCALED", np.ones(plant.controls)),
        ("B_V12_OUTCOME_DERIVED_BOUNDARY", v12_scale),
        ("C_V15_SOURCE_NORMALIZATION_BOUNDARY", boundary.native_scale),
    )
    rows = [_run_step_scale_arm(scale, branch, seed=seed, epochs=epochs,
                                candidates=candidates, cycles=cycles)
            for branch, scale in branch_scales for seed in seeds]
    for row in rows:
        if row["normalization_branch"] == "C_V15_SOURCE_NORMALIZATION_BOUNDARY":
            row.update(boundary.provenance_fields())
    summaries = {branch: _step_summary([
        row for row in rows if row["normalization_branch"] == branch])
        for branch, _ in branch_scales}
    result = {
        "schema_version": "google-pure-v15-abc-step.v1",
        "branches": {
            "A": "legacy unscaled path", "B": "exact V12 outcome-derived map",
            "C": "final V15 source-normalization boundary"},
        "paired_seed_sets_identical": True,
        "controller_hyperparameters_identical": True,
        "normalization_is_the_only_branch_difference": True,
        "epochs": epochs, "candidate_count_per_epoch": candidates,
        "cycles_per_candidate": cycles, "seeds": list(seeds),
        "rows": rows, "summaries": summaries,
        "v12_v15_scale_comparison_hash": canonical_hash(compare_v12_v15_scales()),
        "production_v15_boundary_hash": boundary.boundary_transform_hash,
        "classification": "REDUCED_DEVELOPMENT_DIRECTIONAL_DIAGNOSTIC",
        **NONFINAL,
    }
    atomic_json(OUTPUT / "abc_step" / "comparison.json", result)
    return result


def _figure5b_gradient_probe(scale: np.ndarray, branch: str) -> dict[str, Any]:
    plant, _ = _figure5b_boundary()
    config = controller_config()
    seed, candidates, cycles = 54201, 10, 10_000
    rng = np.random.default_rng(seed)
    initial = rng.choice((-1.0, 1.0), plant.controls) * rng.uniform(.45, .75, plant.controls)
    policy = DirectSigmaGaussianPolicy(initial, np.full(plant.controls, float(config["initial_sigma"])), seed=seed)
    optimizer = DirectSigmaOptimizer(plant.controls, plant.detectors, optimizer_config())
    baseline = np.zeros(plant.detectors)
    batch = policy.sample(candidates)
    native = batch.actions * scale[None, :]
    counts = plant.sample_counts(native, np.zeros(plant.controls), cycles=cycles,
                                 seed=int(canonical_hash(["v15-abc-figure5b", seed])[:16], 16))
    rewards = -counts / float(cycles)
    loss = _sparse_source_loss(
        batch.actions, rewards, plant.control_detector, policy.mean, policy.sigma, baseline,
        batch.behavior, clip=float(config["ppo_clip"]), entropy_weight=.001,
        baseline_weight=float(config["baseline_weight"]))
    direction = -initial / np.linalg.norm(initial)
    projection = float(np.dot(-np.asarray(loss["grad_mean"]), direction))
    before = plant.performance(initial * scale, np.zeros(plant.controls))
    optimizer.step(policy.mean, policy.sigma, baseline,
                   loss["grad_mean"], loss["grad_sigma"], loss["grad_baseline"],
                   mean_bounds=(-2.0, 2.0))
    after = plant.performance(policy.mean * scale, np.zeros(plant.controls))
    return {
        "branch": branch, "seed": seed, "candidate_count": candidates,
        "cycles_per_candidate": cycles,
        "directional_ascent_projection": projection,
        "gradient_projection_standard_error": None,
        "candidate_mean_detector_event_rate": float(np.mean(counts) / cycles),
        "lambda_before": before["lambda"], "lambda_after_one_update": after["lambda"],
        "one_update_lambda_improvement": float(before["lambda"] - after["lambda"]),
        "common_standard_normal_tape": True, "common_count_seed": True,
    }


def run_abc_figure5b() -> dict[str, Any]:
    plant, boundary = _figure5b_boundary()
    v12 = np.sqrt(reference_directional_curvature() / plant.connected_objective_curvature)
    rows = [
        _figure5b_gradient_probe(np.ones(plant.controls), "A_LEGACY_UNSCALED"),
        _figure5b_gradient_probe(v12, "B_V12_OUTCOME_DERIVED"),
        _figure5b_gradient_probe(boundary.native_scale, "C_V15_SOURCE_NORMALIZED"),
    ]
    result = {
        "schema_version": "google-pure-v15-abc-figure5b.v1",
        "condition": {"distance": 3, "parameters_per_gate": 1},
        "rows": rows,
        "controller_hyperparameters_identical": True,
        "paired_randomness": True,
        "normalization_is_the_only_branch_difference": True,
        "classification": "ONE_UPDATE_DIRECTIONAL_DIAGNOSTIC_NOT_FIGURE5B_EVIDENCE",
        **boundary.provenance_fields(), **NONFINAL,
    }
    atomic_json(OUTPUT / "abc_figure5b" / "comparison.json", result)
    return result


def audit_shard_freshness() -> dict[str, Any]:
    shard_paths = sorted((ROOT / "artifacts/google_pure_paper_reproduction/synthetic_reproduction").glob(
        "**/shards/*.json"))
    rows = []
    for path in shard_paths:
        value = _load_json(path)
        provenance = value.get("provenance", {})
        complete = True
        try:
            require_v15_boundary_provenance(provenance)
        except RuntimeError:
            complete = False
        rows.append({
            "path": path.relative_to(ROOT).as_posix(), "shard_id": value.get("shard_id"),
            "protocol_hash": value.get("identity", {}).get("protocol_hash"),
            "implementation_version": provenance.get("implementation_version"),
            "fresh_acquisition": provenance.get("fresh_acquisition"),
            "v15_provenance_complete": complete,
        })
    result = {
        "schema_version": "google-pure-v15-shard-freshness-audit.v1",
        "shard_count": len(rows),
        "v15_complete_shard_count": sum(row["v15_provenance_complete"] for row in rows),
        "fresh_v15_shard_count": sum(row["v15_provenance_complete"] and row["fresh_acquisition"] for row in rows),
        "legacy_or_unverifiable_shard_count": sum(not row["v15_provenance_complete"] for row in rows),
        "rows": rows, **NONFINAL,
    }
    atomic_json(OUTPUT / "shard_freshness_audit.json", result)
    return result


def _tiny_config(family: str) -> dict[str, Any]:
    common = {
        "profile_name": "V15_IMMEDIATE_DRIVER_INTEGRATION",
        "paper_equivalence_claim_permitted": False,
        "evidence_scope": "TINY_EXECUTION_PATH_PROOF_ONLY",
    }
    if family == F.FIGURE5A_REAL_TIME_STEERING.value:
        return {**common, "epochs": 1, "candidates": 3, "cycles_per_candidate": 30,
                "controls": 41, "frequencies": [.001, 1 / 150],
                "entropy_coefficients": [.01], "seeds": [53101]}
    if family == F.FIGURE5B_SPARSE_SCALING.value:
        return {**common, "epochs": 2, "candidates": 3, "cycles_per_candidate": 300,
                "entropy_coefficient": .001, "distances": [3], "parameters_per_gate": [1],
                "seeds": [54101], "local_fit_min_distance": 1e-4, "local_fit_max_distance": .7}
    if family == F.RANDOMIZED_RECOVERY_AFTER_SPOIL.value:
        return {**common, "epochs": 3, "candidates": 3, "cycles_per_candidate": 300,
                "controls": 924, "entropy_coefficient": .001, "sustained_epochs": 2,
                "severities": [.45], "seeds": [56101]}
    if family == F.STEP_RESPONSE_INJECTED_DRIFT.value:
        return {**common, "epochs": 25, "onset_epoch": 3, "direction_coordinate": 0,
                "candidates": 3, "cycles_per_candidate": 300, "controls": 924,
                "entropy_coefficient": .001, "severities": [.5], "seeds": [57101]}
    raise ValueError(family)


def _tiny_protocol(family: str, run_id: str) -> dict[str, Any]:
    config_path = OUTPUT / "generated_configs" / f"{family.lower()}-{run_id}.json"
    atomic_json(config_path, _tiny_config(family))
    return build_protocol(
        family, mode="smoke", config_path=config_path,
        workflow_mode="SMOKE_ACQUISITION", acquisition_run_id=run_id,
        fresh_acquisition_required=True)


def verify_driver_integration() -> dict[str, Any]:
    run_id = f"v15-driver-proof-{time.time_ns()}"
    families = (
        F.STEP_RESPONSE_INJECTED_DRIFT.value,
        F.RANDOMIZED_RECOVERY_AFTER_SPOIL.value,
        F.FIGURE5A_REAL_TIME_STEERING.value,
        F.FIGURE5B_SPARSE_SCALING.value,
    )
    rows = []
    for family in families:
        protocol = _tiny_protocol(family, run_id)
        result = acquire(protocol)
        merged = merge_protocol(protocol)
        rows.append({
            "experiment_family": family, "protocol_hash": protocol["protocol_hash"],
            "experiment_driver_hash": protocol["experiment_driver_hash"],
            "sensitivity_map_hash": protocol["sensitivity_map_hash"],
            "completed_this_call": result["completed_this_call"],
            "fresh_acquisition": result["fresh_acquisition"],
            "reused_shard_ids": result["reused_shard_ids"],
            "merged_shards": merged["merged_shards"],
            "condition_count": protocol["condition_count"],
            "merge_fresh_acquisition": merged["fresh_acquisition"],
            "pass": result["completed_this_call"] == protocol["condition_count"] and
                    result["fresh_acquisition"] and not result["reused_shard_ids"] and
                    merged["complete"] and merged["fresh_acquisition"],
        })
    wrong_hash_rejected = False
    boundary = _step_boundary()
    try:
        boundary.apply(np.zeros(len(boundary.control_ids)), sensitivity_map_hash="wrong-hash")
    except RuntimeError:
        wrong_hash_rejected = True
    result = {
        "schema_version": "google-pure-v15-driver-integration.v1",
        "run_id": run_id, "rows": rows,
        "wrong_sensitivity_hash_rejected_before_plant": wrong_hash_rejected,
        "all_required_drivers_executed_v15": all(row["pass"] for row in rows),
        "pass": wrong_hash_rejected and all(row["pass"] for row in rows),
        "figure5c_fitting_modified": False,
        **NONFINAL,
    }
    atomic_json(OUTPUT / "driver_integration.json", result)
    return result


def freeze_execution_contract() -> dict[str, Any]:
    inputs = audit_execution_path()
    payload = {
        "schema_version": "google-pure-v15-frozen-execution-contract.v1",
        "implementation_version": IMPLEMENTATION_VERSION,
        "lineage_sequence": ["V12", "V13", "V15"], "v14_exists": False,
        "execution_inputs_hash": inputs["execution_inputs_hash"],
        "code_inputs": _code_inputs(),
        "source_normalization_inputs": source_normalization_inputs(),
        "required_fresh_fields": ["fresh_acquisition", "reused_shard_ids", "source_budget_profile"],
        "figure5c_fit_frozen": True,
        **NONFINAL,
    }
    payload["execution_contract_hash"] = canonical_hash(
        {key: value for key, value in payload.items() if key != "execution_contract_hash"})
    path = OUTPUT / "frozen_execution_contract.json"
    if path.is_file() and _load_json(path) != payload:
        existing = _load_json(path)
        archive = OUTPUT / "superseded_execution_contracts" / (
            f"{existing.get('execution_contract_hash', canonical_hash(existing))}.json")
        atomic_json(archive, existing)
    atomic_json(path, payload)
    return payload


def run_reduced_postfix_validation() -> dict[str, Any]:
    started = time.perf_counter()
    driver = verify_driver_integration()
    figure5a_driver = next(row for row in driver["rows"]
                           if row["experiment_family"] == F.FIGURE5A_REAL_TIME_STEERING.value)
    result = {
        "schema_version": "google-pure-v15-reduced-postfix-validation.v1",
        "driver_integration": driver,
        "step_abc": run_abc_step(),
        "figure5b_abc": run_abc_figure5b(),
        "scale_comparison": compare_v12_v15_scales(),
        "objective_audit": audit_calibration_objective(),
        "shard_freshness": audit_shard_freshness(),
        "boundary_traces": {
            "step": trace_boundary_step(), "recovery": trace_boundary_recovery(),
            "figure5b": trace_boundary_figure5b()},
        "figure5a_slow_and_fast_cells_executed": bool(
            figure5a_driver["condition_count"] == 2 and figure5a_driver["merged_shards"] == 2),
        "elapsed_seconds": time.perf_counter() - started,
        "classification": "REDUCED_POST_FIX_DEVELOPMENT_VALIDATION",
        **NONFINAL,
    }
    result["pass"] = bool(driver["pass"] and result["objective_audit"][
        "all_drivers_calibrated_to_the_objective_consumed_by_training"])
    atomic_json(OUTPUT / "reduced_postfix_validation.json", result)
    return result


__all__ = [
    "audit_execution_path", "trace_boundary_step", "trace_boundary_recovery",
    "trace_boundary_figure5b", "run_abc_step", "compare_v12_v15_scales",
    "audit_calibration_objective", "run_abc_figure5b", "audit_shard_freshness",
    "verify_driver_integration", "freeze_execution_contract",
    "run_reduced_postfix_validation",
]
