"""Exact post-comparison recovery regressions used to authorize long acquisition."""
from __future__ import annotations

from dataclasses import replace
import statistics

from hdfa_rl_suite.evaluation.benchmark import BenchmarkRunner
from hdfa_rl_suite.evaluation.launch import load_launch_definition
from hdfa_rl_suite.product import HDFAProductController, ProductLoopConfig
from hdfa_rl_suite.recovery import (
    PhysicalRestorationStatus, RecoveryScope, ReentryReason,
    TransactionRestorationStatus,
)
from hdfa_rl_suite.simulator import (
    DriftKind, LatentProcessSpec, ScalableQECDevice, SimulatorConfig,
)
from hdfa_rl_suite.stage0 import ScalableBootstrapConfig

from .common import ValidationCheck, ValidationReport, all_passed, finalize_report


def _run_case(scenario_id: str, seed: int):
    definition = load_launch_definition(
        "experiments/physical_validation/authoritative-comparison-v1.json")
    config = replace(
        definition.config, authoritative=False, logical_shots_per_interval=8)
    scenario = next(item for item in definition.scenarios()
                    if item.scenario_id == scenario_id)
    runner = BenchmarkRunner(config, (scenario,))
    prepared = runner._prepare_matched_state(scenario, seed)
    return runner._run_arm(
        scenario, seed, "predictive_hdfa_residual_rl",
        runner.arm_factories["predictive_hdfa_residual_rl"], prepared)


def _is_regional(value: object) -> bool:
    return str(value).lower().endswith("regional")


def _ready_product(seed: int) -> HDFAProductController:
    device = ScalableQECDevice(SimulatorConfig(
        qubit_count=3, seed=seed, cycle_period_s=1e-4,
        processes=(LatentProcessSpec(
            "stationary", DriftKind.CONSTANT, {}, amplitude=0.0),)))
    product = HDFAProductController(device, seed=seed, config=ProductLoopConfig(
        residual_candidate_count=4, residual_candidate_cycles=4,
        rollback_validation_cycles=64, rollback_snapshot_max_age_s=1.0,
        bootstrap=ScalableBootstrapConfig(
            characterization_shots=64, validation_cycles=64,
            target_posterior_stddev=.07, qec_detector_rate_limit=.20)))
    product._validation_control_output = product.run_interval(64).control
    return product


def _rollback_fault_evidence() -> dict[str, object]:
    product = _ready_product(8801)
    regional = product.request_reentry(
        ReentryReason.OOD_RECALIBRATION, interval=2,
        scope=RecoveryScope.REGIONAL,
        triggering_evidence={"causal_locality": True},
        affected_regions=("q0",), affected_controls=("drive:q0",),
        boundary_detectors=("D1",), common_mode_probability=.05,
        unknown_model_probability=.4)
    same = product.request_reentry(
        ReentryReason.OOD_RECALIBRATION, interval=2,
        scope=RecoveryScope.REGIONAL,
        triggering_evidence={"causal_locality": True},
        affected_regions=("q0",), affected_controls=("drive:q0",),
        boundary_detectors=("D1",), common_mode_probability=.05,
        unknown_model_probability=.4)
    idempotent = same.request_id == regional.request_id
    product._validated_snapshots = [
        replace(item, validated_at_s=product.device.now_s-10.0)
        for item in product._validated_snapshots]
    stale_rejected = product._select_rollback_snapshot(regional) is None

    transaction_product = _ready_product(8802)
    request = transaction_product.request_reentry(
        ReentryReason.FAILED_ROLLBACK, interval=2, scope=RecoveryScope.GLOBAL,
        triggering_evidence={"fault": "acknowledgement_hash"})
    before = transaction_product.device.acquire(64, retain_records=False)
    original_ack = transaction_product.device.await_policy_acknowledgement

    def mismatched_ack():
        return replace(original_ack(), policy_hash="injected-mismatched-hash")

    transaction_product.device.await_policy_acknowledgement = mismatched_ack
    transaction_violations: list[str] = []
    transaction_physical: list[str] = []
    _, transaction = transaction_product._rollback(
        request, before, transaction_product._validation_control_output,
        [], [], transaction_violations, transaction_physical)

    physical_product = _ready_product(8803)
    request = physical_product.request_reentry(
        ReentryReason.FAILED_ROLLBACK, interval=2, scope=RecoveryScope.GLOBAL,
        triggering_evidence={"fault": "unsafe_restored_telemetry"})
    before = physical_product.device.acquire(64, retain_records=False)
    original_acquire = physical_product.device.acquire

    def unsafe_acquire(cycles, *args, **kwargs):
        batch = original_acquire(cycles, *args, **kwargs)
        return replace(batch, detector_events=batch.detector_exposures)

    physical_product.device.acquire = unsafe_acquire
    physical_violations: list[str] = []
    physical_failures: list[str] = []
    _, physical = physical_product._rollback(
        request, before, physical_product._validation_control_output,
        [], [], physical_violations, physical_failures)
    return {
        "idempotent_request": idempotent,
        "stale_regional_snapshot_rejected": stale_rejected,
        "ack_hash_transaction_status": transaction.transaction_status.value,
        "ack_hash_physical_status": transaction.physical_status.value,
        "ack_hash_lifecycle_violations": tuple(transaction_violations),
        "unsafe_transaction_status": physical.transaction_status.value,
        "unsafe_physical_status": physical.physical_status.value,
        "unsafe_lifecycle_violations": tuple(physical_violations),
        "unsafe_physical_failures": tuple(physical_failures),
        "passed": (
            idempotent and stale_rejected
            and transaction.transaction_status is TransactionRestorationStatus.FAILED
            and transaction.physical_status is PhysicalRestorationStatus.NOT_EVALUATED
            and bool(transaction_violations) and not transaction_physical
            and physical.transaction_status is TransactionRestorationStatus.CONFIRMED
            and physical.physical_status is PhysicalRestorationStatus.FAILED
            and not physical_violations and bool(physical_failures)),
    }


def run_post_comparison_validation() -> ValidationReport:
    nested_metric, nested_rows, _ = _run_case("nested_common", 102)
    unknown_metric, unknown_rows, _ = _run_case("unknown", 105)
    tail_metrics = [nested_metric]
    tail_metrics.extend(_run_case("nested_common", seed)[0]
                        for seed in (101, 103, 104, 105))
    tail_metrics.extend(_run_case("ou_step", seed)[0]
                        for seed in (101, 102, 103, 104, 105))
    nested_recoveries = [row.regional_recovery for row in nested_rows
                         if row.regional_recovery]
    nested_requests = [row.reentry_request for row in nested_rows
                       if row.reentry_request]
    unknown_requests = [row.reentry_request for row in unknown_rows
                        if row.reentry_request]
    nested_safe = (
        nested_metric.completion_status == "completed"
        and nested_metric.lifecycle_violation_count == 0
        and nested_metric.physical_rollback_failure_count == 0
        and nested_metric.bootstrap_count == 1
        and bool(nested_recoveries)
        and all(item.get("passed") for item in nested_recoveries))
    regional_contract = bool(nested_recoveries) and all(
        item.get("gate_results", {}).get("boundary_validation")
        and item.get("gate_results", {}).get("unaffected_policy_frozen")
        and item.get("boundary_detectors")
        and item.get("frozen_controls")
        for item in nested_recoveries)
    locality = bool(nested_requests) and any(
        _is_regional(item.get("scope")) for item in nested_requests)
    unknown_safe = (
        unknown_metric.completion_status == "completed"
        and unknown_metric.lifecycle_violation_count == 0
        and unknown_metric.physical_rollback_failure_count == 0
        and unknown_metric.bootstrap_count <= 5
        and unknown_metric.rollback_count <= 4)
    broad_global = all(
        not _is_regional(item.get("scope")) for item in unknown_requests)
    tail_summary = {}
    for scenario_id in ("ou_step", "nested_common"):
        values = [item for item in tail_metrics if item.scenario_id == scenario_id]
        endpoints = [next(endpoint for endpoint in item.recovery_endpoints
                          if endpoint.target_fraction == .90) for item in values]
        intervals_or_censor = [
            endpoint.intervals_after_peak
            if endpoint.status == "reached" and endpoint.intervals_after_peak is not None
            else 32 for endpoint in endpoints]
        tail_summary[scenario_id] = {
            "run_count": len(values),
            "completion_fraction": sum(item.completion_status == "completed"
                                       for item in values)/max(1, len(values)),
            "observed_90pct_fraction": sum(endpoint.status == "reached"
                                           for endpoint in endpoints)/max(1, len(endpoints)),
            "median_90pct_intervals_or_censor": statistics.median(intervals_or_censor),
            "maximum_90pct_intervals_or_censor": max(intervals_or_censor),
            "safety_failures": sum(
                item.lifecycle_violation_count+item.physical_rollback_failure_count
                for item in values),
        }
    tail_latency_passed = all(
        row["run_count"] == 5 and row["completion_fraction"] == 1.0
        and row["observed_90pct_fraction"] >= .8
        and row["median_90pct_intervals_or_censor"] <= 8
        and row["safety_failures"] == 0
        for row in tail_summary.values())
    rollback_faults = _rollback_fault_evidence()
    checks = (
        ValidationCheck(
            "nested_common_102_regional_recovery", nested_safe,
            {"status": nested_metric.completion_status,
             "bootstrap_count": nested_metric.bootstrap_count,
             "rollback_count": nested_metric.rollback_count,
             "physical_failures": nested_metric.physical_rollback_failure_count},
            "the exact former censor completes through a certified regional recovery",
            "The second localized unknown event must not invoke stationary global Stage 0."),
        ValidationCheck(
            "regional_boundary_and_freeze_contract", regional_contract and locality,
            {"recovery_count": len(nested_recoveries),
             "requests": nested_requests},
            "locality is positive, boundary detectors are held out, and unaffected controls remain frozen",
            "Testing fewer detectors cannot weaken multiplicity or hide cross-region damage."),
        ValidationCheck(
            "unknown_105_bounded_global_recovery", unknown_safe,
            {"status": unknown_metric.completion_status,
             "bootstrap_count": unknown_metric.bootstrap_count,
             "rollback_count": unknown_metric.rollback_count,
             "physical_failures": unknown_metric.physical_rollback_failure_count},
            "the exact former lifecycle failure completes without transaction or physical restoration failure",
            "Recovery entries and rollback attempts must remain explicitly bounded."),
        ValidationCheck(
            "broad_ood_no_regional_shortcut", broad_global,
            {"requests": unknown_requests},
            "multi-region/repeated broad OOD remains on the global recovery path",
            "Truth isolation forbids a scenario-name exception; only causal regional evidence is used."),
        ValidationCheck(
            "rollback_transaction_physical_fault_separation",
            bool(rollback_faults["passed"]), rollback_faults,
            "idempotence, stale scope, acknowledgement/hash and unsafe telemetry faults all fail closed",
            "Transactional corruption remains a lifecycle failure; unsafe restored telemetry remains an explicit physical failure."),
        ValidationCheck(
            "ou_nested_development_tail_latency", tail_latency_passed,
            tail_summary,
            "each OU-step/nested family completes, has zero safety failures, >=80% observed 90% recovery and median <=8 intervals",
            "Censored runs are charged at the 32-interval limit; no complete-case deletion is permitted."),
    )
    return finalize_report(ValidationReport(
        "post-comparison-recovery-validation.v1",
        "post_comparison_recovery", all_passed(checks), checks,
        tuple({"scenario": row.scenario_id, "seed": row.seed,
               "arm": row.arm, "interval": row.interval,
               "mode": row.lifecycle_mode,
               "reentry": row.reentry_request,
               "regional_recovery": row.regional_recovery,
               "physical_failures": row.physical_rollback_failures}
              for row in (*nested_rows, *unknown_rows)),
        {"development_only": True, "source_report_immutable": True,
         "rollback_fault_evidence": rollback_faults},
    ))
