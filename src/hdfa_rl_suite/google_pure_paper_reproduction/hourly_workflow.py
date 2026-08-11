"""Bounded, validation-only execution of every public-paper experiment family.

This profile preserves the amended controller path, physical plant identities, source
per-candidate cycle count, full scaling geometry, paired natural-drift estimator, and
censoring-aware endpoints. It deliberately reduces scan density, candidate batches,
and replication. It validates whether the execution path is healthy; it cannot
establish paper-equivalent performance.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path
import time
import traceback
from typing import Any, Mapping
from enum import StrEnum

from hdfa_rl_suite.google_pure_source_exact.figure5a.validation import physical_preflight
from hdfa_rl_suite.google_pure_v7.config import canonical_hash, repository_root

from .claim_registry import build_claim_registry
from .comparison_metrics import family_checks
from .direct_path import integration_manifest, protocol_identity_reasons
from .experiment_families import ExperimentFamily
from .paper_figures import acquire, build_protocol, merge_protocol, plot_protocol
from .paper_tables import build_values_table
from .public_data import reproduce_public_data
from .reporting import audit_all, status
from .side_by_side import compare_panel
from .source_registry import build_source_contract
from .storage import atomic_json, initialise_layout, load_protocol
from .validation import validate_protocol


F = ExperimentFamily
PROFILE_NAME = "ONE_HOUR_SCIENTIFIC_VALIDATION"
PROFILE_MODE = "validation"
MAX_SAFE_WORKERS = 6
TARGET_WALL_MINUTES = 60
ESTIMATED_WALL_MINUTES = (35, 55)
CALIBRATION_SECONDS = {
    "figure5a_representative_cell": 305.92,
    "figure5b_distance15_p30_cell": 60.74,
    "natural_drift_full_cell": 11.38,
    "step_response_full_cell": 133.06,
}
CONFIG_DIRECTORY = Path("configs/google_pure_paper_reproduction/one_hour_validation")


class WorkflowMode(StrEnum):
    ANALYSIS_ONLY = "ANALYSIS_ONLY"
    SMOKE_ACQUISITION = "SMOKE_ACQUISITION"
    ONE_HOUR_FRESH_ACQUISITION = "ONE_HOUR_FRESH_ACQUISITION"
    REFERENCE_ACQUISITION = "REFERENCE_ACQUISITION"


@dataclass(frozen=True)
class FamilyProfile:
    family: str
    config_name: str
    panel: str | None = None


FAMILY_PROFILES = (
    FamilyProfile(F.FIGURE5A_REAL_TIME_STEERING.value, "figure5a.json", "figure5a"),
    FamilyProfile(F.FIGURE5B_SPARSE_SCALING.value, "figure5b.json", "figure5b"),
    FamilyProfile(F.FIGURE5C_CONVERGENCE_LAW.value, "figure5c.json", "figure5c"),
    FamilyProfile(F.NATURAL_DRIFT_SPECTRAL_SUPPRESSION.value, "natural_drift.json"),
    FamilyProfile(F.RANDOMIZED_RECOVERY_AFTER_SPOIL.value, "randomized_recovery.json"),
    FamilyProfile(F.STEP_RESPONSE_INJECTED_DRIFT.value, "step_response.json"),
)


def _profile_config_path(root: Path, profile: FamilyProfile) -> Path:
    path = root / CONFIG_DIRECTORY / profile.config_name
    if not path.exists():
        raise RuntimeError(f"missing one-hour profile configuration: {path}")
    return path


def build_one_hour_protocols(root: Path | None = None, *,
                             workflow_mode: str = WorkflowMode.ONE_HOUR_FRESH_ACQUISITION.value,
                             acquisition_run_id: str | None = None) -> list[dict[str, Any]]:
    """Freeze all family protocols in validation mode with disjoint seeds."""
    base = repository_root() if root is None else Path(root)
    selected = WorkflowMode(workflow_mode)
    if selected is WorkflowMode.ANALYSIS_ONLY:
        return [load_protocol(profile.family, PROFILE_MODE) for profile in FAMILY_PROFILES]
    run_mode = {
        WorkflowMode.SMOKE_ACQUISITION: "smoke",
        WorkflowMode.ONE_HOUR_FRESH_ACQUISITION: PROFILE_MODE,
        WorkflowMode.REFERENCE_ACQUISITION: "reference",
    }[selected]
    fresh = selected in {WorkflowMode.ONE_HOUR_FRESH_ACQUISITION,
                         WorkflowMode.REFERENCE_ACQUISITION}
    run_id = acquisition_run_id or (
        f"{selected.value}-{time.time_ns()}" if fresh else "REUSABLE_SMOKE_ACQUISITION")
    protocols = [build_protocol(
        profile.family, mode=run_mode,
        config_path=(_profile_config_path(base, profile)
                     if selected is WorkflowMode.ONE_HOUR_FRESH_ACQUISITION else None),
        workflow_mode=selected.value, acquisition_run_id=run_id,
        fresh_acquisition_required=fresh)
        for profile in FAMILY_PROFILES]
    for protocol in protocols:
        config = protocol["config"]
        if selected is WorkflowMode.ONE_HOUR_FRESH_ACQUISITION:
            if config.get("profile_name") != PROFILE_NAME:
                raise RuntimeError("one-hour protocol lost its profile identity")
            if config.get("paper_equivalence_claim_permitted") is not False:
                raise RuntimeError("one-hour protocol must prohibit paper-equivalence claims")
            if protocol["mode"] != PROFILE_MODE:
                raise RuntimeError("one-hour protocol must remain validation-only")
        reasons = protocol_identity_reasons(protocol)
        if reasons:
            raise RuntimeError("amended execution-path preflight failed: " + "; ".join(reasons))
    return protocols


def profile_plan(protocols: list[Mapping[str, Any]], *, max_workers: int) -> dict[str, Any]:
    """Return the immutable cost and evidence boundary before acquisition."""
    if not 1 <= int(max_workers) <= MAX_SAFE_WORKERS:
        raise ValueError(f"max_workers must lie in [1, {MAX_SAFE_WORKERS}]")
    families = []
    for protocol in protocols:
        config = protocol["config"]
        boundaries = (int(protocol["condition_count"]) * int(config["epochs"]) *
                      int(config["candidates"]))
        families.append({
            "experiment_family": protocol["experiment_family"],
            "protocol_hash": protocol["protocol_hash"],
            "conditions": protocol["condition_count"],
            "epochs": int(config["epochs"]),
            "candidates_per_epoch": int(config["candidates"]),
            "cycles_per_candidate": int(config["cycles_per_candidate"]),
            "candidate_boundaries": boundaries,
            "candidate_qec_cycles": boundaries * int(config["cycles_per_candidate"]),
            "checkpoint_every_candidates": int(config.get("checkpoint_every_candidates", 1)),
            "record_storage": "compact_directional" if config.get("compact_records") else "full_policy",
            "evidence_scope": config.get("evidence_scope", "NONFINAL_PUBLIC_ANALOGUE"),
            "workflow_mode": protocol["workflow_mode"],
            "fresh_acquisition_required": protocol["fresh_acquisition_required"],
            "acquisition_run_id": protocol["acquisition_run_id"],
        })
    payload = {
        "schema_version": "google-paper-one-hour-plan.v1",
        "profile_name": PROFILE_NAME,
        "mode": PROFILE_MODE,
        "max_workers": int(max_workers),
        "target_wall_minutes": TARGET_WALL_MINUTES,
        "estimated_wall_minutes": list(ESTIMATED_WALL_MINUTES),
        "estimate_basis": "direct timings of reusable validation cells on the target workstation",
        "calibration_observed_seconds": dict(CALIBRATION_SECONDS),
        "families": families,
        "condition_count": sum(row["conditions"] for row in families),
        "candidate_boundaries": sum(row["candidate_boundaries"] for row in families),
        "candidate_qec_cycles": sum(row["candidate_qec_cycles"] for row in families),
        "final_evidence": False,
        "paper_equivalence_claim_permitted": False,
        "literal_paper_budget_preserved": False,
        "scientific_use": "bounded development validation and long-run go/no-go decision",
        "limitations": [
            "reduced scan density",
            "reduced candidate batch size",
            "two or three development seeds per experiment family",
            "validation checkpoints may flush once per epoch while preserving deterministic replay",
            "non-public proprietary simulator and hardware traces remain unavailable",
        ],
    }
    payload["plan_hash"] = canonical_hash(payload)
    return payload


def _acquire_one(task: tuple[dict[str, Any], int]) -> dict[str, Any]:
    """Execute exactly one deterministic condition in a spawned process."""
    protocol, condition_index = task
    try:
        result = acquire(protocol, max_shards=1, worker_index=condition_index,
                         worker_count=int(protocol["condition_count"]))
    except Exception as error:
        return {
            "ok": False,
            "failure": {
                "experiment_family": protocol["experiment_family"],
                "protocol_hash": protocol["protocol_hash"],
                "condition_index": int(condition_index),
                "condition": dict(protocol["conditions"][condition_index]),
                "exception_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        }
    return {"ok": True, "result": result}


def _run_acquisitions(protocols: list[dict[str, Any]], *, max_workers: int) -> dict[str, Any]:
    tasks = [(protocol, index) for protocol in protocols
             for index in range(int(protocol["condition_count"]))]
    totals = {protocol["experiment_family"]: int(protocol["condition_count"])
              for protocol in protocols}
    completed = {family: 0 for family in totals}
    new_shards = 0
    reused_shards = 0
    new_candidate_qec_cycles = 0
    new_candidate_boundaries = 0
    started = time.perf_counter()
    # Six processes match the physical-core count measured on the target workstation.
    # This bounded pool avoids the oversubscription and stale-worker behaviour of the
    # previous eleven-process launch while retaining dynamic load balancing.
    pool = ProcessPoolExecutor(max_workers=max_workers)
    futures = {pool.submit(_acquire_one, task): task for task in tasks}
    try:
        for future in as_completed(futures):
            protocol, condition_index = futures[future]
            family = protocol["experiment_family"]
            try:
                envelope = future.result()
            except Exception as error:
                envelope = {
                    "ok": False,
                    "failure": {
                        "experiment_family": family,
                        "protocol_hash": protocol["protocol_hash"],
                        "condition_index": int(condition_index),
                        "condition": dict(protocol["conditions"][condition_index]),
                        "exception_type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    },
                }
            if not envelope["ok"]:
                failure = envelope["failure"]
                for pending in futures:
                    if pending is not future:
                        pending.cancel()
                failure_manifest = {
                    "schema_version": "google-paper-one-hour-worker-failure.v1",
                    "profile_name": PROFILE_NAME,
                    "failed_unix_seconds": time.time(),
                    "elapsed_seconds": time.perf_counter() - started,
                    "max_workers": int(max_workers),
                    "completed_conditions": dict(completed),
                    "failure": failure,
                    "remaining_work_cancelled": True,
                    "final_evidence": False,
                    "paper_equivalence_claim_permitted": False,
                }
                failure_manifest["failure_hash"] = canonical_hash(failure_manifest)
                atomic_json(initialise_layout() / "reports" /
                            "one_hour_validation_worker_failure.json", failure_manifest)
                pool.shutdown(wait=True, cancel_futures=True)
                pool = None
                raise RuntimeError(
                    f"worker failed for {family} condition {condition_index}: "
                    f"{failure['exception_type']}: {failure['message']}")
            result = envelope["result"]
            if result["completed_this_call"] + result["preexisting_assigned_shards"] != 1:
                raise RuntimeError(f"condition did not complete or resume cleanly: {family}")
            completed[family] += 1
            new_shards += int(result["completed_this_call"])
            reused_shards += int(result["preexisting_assigned_shards"])
            config = protocol["config"]
            new_candidate_boundaries += (int(result["completed_this_call"]) *
                                         int(config["epochs"]) * int(config["candidates"]))
            new_candidate_qec_cycles += (int(result["completed_this_call"]) *
                                         int(config["epochs"]) * int(config["candidates"]) *
                                         int(config["cycles_per_candidate"]))
            print(f"[{(time.perf_counter()-started)/60:6.1f} min] "
                  f"{family}: {completed[family]}/{totals[family]}", flush=True)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    return {
        "acquisition_seconds": time.perf_counter() - started,
        "new_shards": new_shards,
        "reused_shards": reused_shards,
        "new_candidate_boundaries": new_candidate_boundaries,
        "new_candidate_qec_cycles": new_candidate_qec_cycles,
    }


def _scientific_preflight(root: Path) -> dict[str, Any]:
    integration = integration_manifest()
    source_config = json.loads(
        (root / "configs/google_pure_source_exact/figure5a.json").read_text(encoding="utf-8"))
    plant = physical_preflight(root, source_config)
    if not integration.get("pass") or not plant.get("pass"):
        raise RuntimeError("one-hour acquisition blocked by direct-path or physical preflight")
    return {
        "integration_manifest_hash": integration["manifest_hash"],
        "controller_hash": integration["controller_hash"],
        "controller_code_hash": integration["controller_code_hash"],
        "controller_mode": integration["controller_mode"],
        "parameterization": integration["parameterization"],
        "plant_preflight_hash": canonical_hash(plant),
        "plant_hash": plant["plant_hash"],
        "graph_hash": plant["mask_hash"],
        "pass": True,
    }


def run_one_hour_validation(*, max_workers: int = MAX_SAFE_WORKERS,
                            workflow_mode: str = WorkflowMode.ONE_HOUR_FRESH_ACQUISITION.value,
                            acquisition_run_id: str | None = None) -> dict[str, Any]:
    """Run all families, then merge, validate, plot and compare without promotion."""
    if not 1 <= int(max_workers) <= MAX_SAFE_WORKERS:
        raise ValueError(f"max_workers must lie in [1, {MAX_SAFE_WORKERS}]")
    root = repository_root()
    selected = WorkflowMode(workflow_mode)
    started_wall = time.time()
    started = time.perf_counter()
    preflight = _scientific_preflight(root)
    build_claim_registry()
    build_source_contract()
    reproduce_public_data(recompute=False)
    protocols = build_one_hour_protocols(
        root, workflow_mode=selected.value, acquisition_run_id=acquisition_run_id)
    plan = profile_plan(protocols, max_workers=max_workers)
    output_root = initialise_layout()
    atomic_json(output_root / "reports" / "one_hour_validation_plan.json", plan)
    print(json.dumps({"profile": PROFILE_NAME, "plan_hash": plan["plan_hash"],
                      "estimated_wall_minutes": plan["estimated_wall_minutes"],
                      "max_workers": max_workers}, sort_keys=True), flush=True)
    if selected is WorkflowMode.ANALYSIS_ONLY:
        acquisition_stats = {
            "acquisition_seconds": 0.0, "new_shards": 0, "reused_shards": 0,
            "new_candidate_boundaries": 0, "new_candidate_qec_cycles": 0,
        }
    else:
        acquisition_stats = _run_acquisitions(protocols, max_workers=max_workers)

    analysis_started = time.perf_counter()
    family_results = []
    for profile, protocol in zip(FAMILY_PROFILES, protocols):
        merged = merge_protocol(protocol)
        validation = validate_protocol(protocol)
        plotted = plot_protocol(protocol)
        comparison = (compare_panel(profile.panel, protocol) if profile.panel else
                      {"numeric_checks": family_checks(profile.family, validation),
                       "verdict": "VALIDATION_ONLY"})
        family_results.append({
            "experiment_family": profile.family,
            "protocol_hash": protocol["protocol_hash"],
            "merged_shards": merged["merged_shards"],
            "complete": merged["complete"],
            "scientifically_valid": validation["scientifically_valid"],
            "status": validation["status"],
            "final_evidence": validation["final_evidence"],
            "paper_comparable": validation["paper_comparable"],
            "blocking_reasons": validation["blocking_reasons"],
            "plot_files": plotted["files"],
            "comparison_verdict": comparison.get("verdict"),
        })

    build_values_table()
    audit = audit_all()
    current_status = status()
    analysis_seconds = time.perf_counter() - analysis_started
    elapsed = time.perf_counter() - started
    all_complete = all(row["complete"] for row in family_results)
    all_validation_passed = all(row["scientifically_valid"] for row in family_results)
    result = {
        "schema_version": "google-paper-one-hour-validation.v1",
        "profile_name": PROFILE_NAME,
        "workflow_mode": selected.value,
        "plan_hash": plan["plan_hash"],
        "started_unix_seconds": started_wall,
        "elapsed_seconds": elapsed,
        "elapsed_minutes": elapsed / 60.0,
        **acquisition_stats,
        "analysis_seconds": analysis_seconds,
        "analysis_only": selected is WorkflowMode.ANALYSIS_ONLY,
        "target_wall_minutes": TARGET_WALL_MINUTES,
        "within_estimated_wall_window": elapsed / 60.0 <= ESTIMATED_WALL_MINUTES[1],
        "max_workers": int(max_workers),
        "preflight": preflight,
        "families": family_results,
        "all_families_complete": all_complete,
        "all_validation_gates_passed": all_validation_passed,
        "workflow_pass": all_complete and all_validation_passed,
        "final_evidence": False,
        "paper_equivalence_claim_permitted": False,
        "evidence_layer": "EXECUTED_REPOSITORY_DEVELOPMENT_VALIDATION",
        "scientific_conclusion": (
            "IMPLEMENTATION_PATH_READY_FOR_SELECTIVE_CONFIRMATION" if all_validation_passed
            else "ONE_HOUR_VALIDATION_FOUND_SCIENTIFIC_FAILURES"),
        "audit_pass": audit.get("status") == "PASS",
        "status_snapshot_hash": canonical_hash(current_status),
    }
    atomic_json(output_root / "reports" / "one_hour_validation_manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run every paper family in a bounded validation-only profile")
    parser.add_argument("--max-workers", type=int, default=MAX_SAFE_WORKERS,
                        help=f"spawned processes, capped at {MAX_SAFE_WORKERS}")
    parser.add_argument("--plan-only", action="store_true",
                        help="write and print the immutable plan without acquisition")
    parser.add_argument("--workflow-mode", choices=[item.value for item in WorkflowMode],
                        default=WorkflowMode.ONE_HOUR_FRESH_ACQUISITION.value)
    parser.add_argument("--run-id", default=None,
                        help="explicit acquisition identity; fresh modes otherwise generate one")
    args = parser.parse_args()
    if args.plan_only:
        protocols = build_one_hour_protocols(
            workflow_mode=args.workflow_mode, acquisition_run_id=args.run_id)
        result = profile_plan(protocols, max_workers=args.max_workers)
        atomic_json(initialise_layout() / "reports" / "one_hour_validation_plan.json", result)
    else:
        result = run_one_hour_validation(max_workers=args.max_workers,
                                         workflow_mode=args.workflow_mode,
                                         acquisition_run_id=args.run_id)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    if not args.plan_only and not result["workflow_pass"]:
        raise SystemExit(2)


__all__ = [
    "CALIBRATION_SECONDS", "ESTIMATED_WALL_MINUTES", "FAMILY_PROFILES", "MAX_SAFE_WORKERS",
    "PROFILE_MODE", "PROFILE_NAME", "WorkflowMode", "build_one_hour_protocols", "profile_plan",
    "run_one_hour_validation",
]


if __name__ == "__main__":
    main()
