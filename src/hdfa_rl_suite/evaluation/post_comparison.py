"""Development-only deterministic replay for post-comparison diagnosis.

This path intentionally executes just the staged controller for named scenario/seed
conditions.  It neither reads nor copies the 254 MB acceptance report and is never an
authoritative comparison arm.  Its compact timelines are causal debugging evidence.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping, Sequence

from hdfa_rl_suite.baselines.controllers import PredictiveHDFARLArm
from hdfa_rl_suite.common import deterministic_hash
from hdfa_rl_suite.evaluation.benchmark import BenchmarkRunner
from hdfa_rl_suite.evaluation.launch import load_launch_definition
from hdfa_rl_suite.product import QECOperabilityError, RecoveryCertificationError


LEGACY_REPORT_FILE_SHA256 = (
    "bfbe966dacc4f4dde986a158bd09c37cf02de0e8cc2637ebfecee89ff392a6f5")
LEGACY_REPORT_HASH = (
    "45b36b2e9c8e2d6bc5d5173710b5bfb512a346651269de1882a03bd5f823b773")
DEFAULT_CASES = (("nested_common", 102), ("unknown", 105))


def _compact_interval(interval: int, elapsed_s: float, output) -> dict:
    evidence = dict(output.stage_evidence or {})
    regions = evidence.get("regions", {})
    triggering = {}
    for region_id, row in regions.items():
        predictive = row.get("posterior_predictive", {})
        residuals = predictive.get("standardized_residuals", {})
        triggering[region_id] = tuple(
            detector for detector, value in sorted(
                residuals.items(), key=lambda item: abs(float(item[1])), reverse=True)
            if abs(float(value)) >= 1.0)
    authorizations = evidence.get("authorizations", ())
    final_authorization = authorizations[-1] if authorizations else {}
    return {
        "interval": interval,
        "simulated_time_s": elapsed_s,
        "newly_acquired_cycles": output.total_qec_cycles,
        "causal_observation": evidence.get("causal_observation"),
        "feedback_observation": evidence.get("feedback_observation"),
        "regions": regions,
        "triggering_detectors_by_region": triggering,
        "supervisor": {
            "mode": output.lifecycle_mode,
            "authorization": output.authorization,
            "reason": final_authorization.get("reason"),
            "transition": final_authorization.get("transition"),
            "rollback_required": final_authorization.get("rollback_required", False),
            "authorization_log": authorizations,
        },
        "pending_reentry": evidence.get("pending_reentry"),
        "structured_reentry_request": evidence.get("structured_reentry_request"),
        "bootstrap_reason": output.bootstrap_reason,
        "bootstrap_count": output.bootstrap_count,
        "bootstrap_evidence": output.bootstrap_evidence,
        "policy": {
            "confirmed": evidence.get("confirmed_policy"),
            "applied_policy_hashes": evidence.get("applied_policy_hashes"),
            "recent_transactions": evidence.get("latest_policy_transactions"),
        },
        "rollback": {
            "legacy_summary": evidence.get("rollback"),
            "outcomes": evidence.get("rollback_outcomes"),
            "physical_failures": evidence.get("physical_rollback_failures"),
        },
        "regional_recovery": evidence.get("regional_recovery"),
        "forecast_mpc": {
            region_id: {
                key: row.get(key) for key in (
                    "forecast_invalidity_reasons", "forecast_validity_horizon_s",
                    "mpc_status", "mpc_action", "mpc_active_constraints",
                    "mpc_policy_id", "mpc_policy_hash",
                    "mpc_reference_policy_id", "mpc_reference_policy_hash")
            } for region_id, row in regions.items()
        },
        "residual": {
            "result": evidence.get("residual_result"),
            "candidate_count": evidence.get("residual_candidate_count"),
            "candidate_trajectories": output.candidate_trajectories,
            "exploration_damage": output.exploration_damage,
        },
        "compute_timing": evidence.get("compute_timing"),
        "lifecycle_violations": output.lifecycle_violations,
        "replay_hash": output.replay_hash,
    }


def replay_case(scenario_id: str, seed: int, *, launch_path: str | Path
                ) -> Mapping[str, object]:
    definition = load_launch_definition(launch_path)
    config = replace(
        definition.config, authoritative=False, logical_shots_per_interval=8)
    scenarios = {item.scenario_id: item for item in definition.scenarios()}
    scenario = scenarios[scenario_id]
    runner = BenchmarkRunner(config, (scenario,))
    prepared = runner._prepare_matched_state(scenario, seed)
    device = prepared.device.clone()
    arm = PredictiveHDFARLArm(
        seed=seed, residual=True, candidate_count=4,
        candidate_cycles=config.candidate_cycles,
        bootstrap_config=runner._bootstrap_config())
    arm.prepare(device, prepared.bootstrap)
    timeline = []
    status, reason = "completed", None
    failed_bootstrap = None
    for interval in range(config.intervals):
        try:
            output = arm.run_interval(device, config.cycles_per_interval, interval)
        except QECOperabilityError as error:
            status, reason = "censored", str(error)
            failed_bootstrap = error.result.to_dict()
            break
        except RecoveryCertificationError as error:
            status, reason = "censored", str(error)
            failed_bootstrap = {
                "evidence_phase": "online-disturbance-aware-recovery",
                "request": asdict(error.request),
                "outcome": asdict(error.outcome) if error.outcome else None,
            }
            break
        except Exception as error:  # diagnostic evidence must preserve unexpected faults
            status = "missing"
            reason = f"{type(error).__name__}: {error}"
            break
        timeline.append(_compact_interval(interval, device.now_s, output))
    return {
        "schema_version": "post-comparison-timeline.v1",
        "evidence_role": "development-only deterministic replay; not acceptance evidence",
        "scenario_id": scenario_id,
        "seed": seed,
        "status": status,
        "reason": reason,
        "completed_intervals": len(timeline),
        "disturbance_realization_id": device.disturbance_realization_id,
        "stationary_baseline_observation_hash": prepared.baseline.observation_hash,
        "matched_initial_state": asdict(prepared.device.counterfactual_state_fingerprint()),
        "timeline": timeline,
        "failed_bootstrap_evidence": failed_bootstrap,
        "case_hash": deterministic_hash({
            "scenario_id": scenario_id, "seed": seed, "status": status,
            "reason": reason, "timeline": timeline,
            "failed_bootstrap_evidence": failed_bootstrap,
        }),
    }


def _root_cause_markdown(cases: Sequence[Mapping[str, object]]) -> str:
    by_id = {str(item["scenario_id"]): item for item in cases}
    nested = by_id.get("nested_common", {})
    unknown = by_id.get("unknown", {})
    return f"""# Post-comparison deterministic root-cause report

Evidence status: development diagnosis only. The immutable v1 result remains an
authoritative scientific rejection. File SHA-256: `{LEGACY_REPORT_FILE_SHA256}`;
internal report hash: `{LEGACY_REPORT_HASH}`.

## `nested_common/102`

Post-repair replay status: `{nested.get('status')}` after
`{nested.get('completed_intervals')}` retained intervals. Earliest causal divergence is
interval 1: one regional Stage-3 unknown-model posterior dominates while the remaining
regions stay below the unknown threshold, but Stage 7 classifies the maximum as a broad
unknown event. The unsafe assumption is the product-level reduction of all OOD evidence
to a scalar `ReentryReason`: it discards locality, rolls the whole policy back, and then
runs the stationary global Stage-0 calibrator inside an active disturbance. The same
pattern recurs at interval 6. The next global bootstrap is therefore asked to recover a
moving device under a contract designed for stationary initialization; it fails the QEC
and sensitivity gates and censors cleanly. The gates themselves did not fail incorrectly.

## `unknown/105`

Post-repair replay status: `{unknown.get('status')}` with
`{unknown.get('completed_intervals')}` retained intervals. Earliest causal divergence is
interval 15, where multi-region/OOD evidence correctly revokes model authority. Global
re-entry at interval 16 succeeds, but its fresh supervisor enters `DEGRADED`. During the
minimum-dwell interval, `tick` returns predictive approval even though the transition out
of `DEGRADED` was suppressed; the subsequent mode-permission check converts this into a
rollback. The rollback always targets the original bootstrap snapshot and judges physical
restoration against a fixed detector-rate band, despite the changed disturbance context.
This creates the interval-18 and interval-20 rollback-validation failures and repeated
global re-entry. Transaction restoration and physical restoration are conflated in one
`lifecycle_violations` string, obscuring—but not hiding—the two physical failures.

## Required repair boundary

The repair must preserve every hard gate. It must carry structured regional evidence to
re-entry, validate affected plus boundary detectors, keep broad OOD global, make mode
approval consistent with the actually entered state, select a recent scope-compatible
confirmed snapshot, and report transactional restoration separately from uncertainty-
aware physical restoration.

## Repair verification

The post-repair bundle is not a reinterpretation of v1. `nested_common/102` uses an
online disturbance-aware regional evidence phase with affected-plus-boundary validation
and frozen unaffected controls. `unknown/105` remains global from causal multi-region
unknown-model evidence, but global OOD recovery is now an explicitly named active-
disturbance phase rather than stationary Stage 0. Both transaction and physical rollback
statuses remain separately serialized. Every retained interval carries the versioned
critical-path timing record. The exact completion, safety and bounded-entry conditions
are executable preflight regressions; the immutable v1 rejection remains unchanged.
"""


def write_diagnostic_bundle(output_dir: str | Path, *,
                            launch_path: str | Path =
                            "experiments/physical_validation/authoritative-comparison-v1.json",
                            cases: Sequence[tuple[str, int]] = DEFAULT_CASES) -> tuple[Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    results = [replay_case(scenario, seed, launch_path=launch_path)
               for scenario, seed in cases]
    payload = {
        "schema_version": "post-comparison-diagnostic-bundle.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "legacy_report_file_sha256": LEGACY_REPORT_FILE_SHA256,
        "legacy_report_hash": LEGACY_REPORT_HASH,
        "launch_path": str(launch_path).replace("\\", "/"),
        "cases": results,
    }
    payload["bundle_hash"] = deterministic_hash(payload)
    timeline_path = target / "failure-timelines.json"
    report_path = target / "root-cause-report.md"
    timeline_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    report_path.write_text(_root_cause_markdown(results), encoding="utf-8")
    return timeline_path, report_path
