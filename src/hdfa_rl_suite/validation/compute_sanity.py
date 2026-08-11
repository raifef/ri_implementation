"""Symmetric critical-path timing and censoring-aware RMST validation."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from hdfa_rl_suite.evaluation.benchmark import (
    BenchmarkConfig, BenchmarkRunner, _restricted_mean_time,
    default_benchmark_scenarios,
)
from hdfa_rl_suite.evaluation.evidence import validate_report_payload

from .common import ValidationCheck, ValidationReport, all_passed, finalize_report


def run_compute_accounting_validation() -> ValidationReport:
    config = BenchmarkConfig(
        qubit_count=3, intervals=3, cycles_per_interval=64, seeds=(31, 37),
        candidate_cycles=32, logical_shots_per_interval=8,
        cycle_period_s=1e-5,
        bootstrap_characterization_shots=64,
        bootstrap_validation_cycles=64,
        bootstrap_target_stddev=.07,
        bootstrap_qec_rate_limit=.20,
        authoritative=False)
    scenario = default_benchmark_scenarios(3)[:1]
    initial = BenchmarkRunner(config, scenario)
    factories = {name: initial.arm_factories[name] for name in (
        "full_control_detector_rl", "predictive_hdfa_residual_rl")}
    evaluation_runner = BenchmarkRunner(config, scenario, factories)
    report = evaluation_runner.run()
    trajectories = tuple(report.trajectories)
    timing_complete = all(
        row.timing is not None and not row.timing.validate()
        for row in trajectories)
    symmetric = all(
        row.timing is not None and set(row.timing.stage_compute_s)
        and row.timing.clock_policy == "serial-hybrid-clock-critical-path.v1"
        for row in trajectories)
    gates = {gate.gate_id: gate for gate in report.gates}
    rmst_gate = gates["compute_aware_rmst_net_convergence_gain"]
    tail_gate = gates["compute_aware_e2e_tail_noninferiority"]
    gate_evaluable = (
        rmst_gate.status in {"pass", "fail"}
        and tail_gate.status in {"pass", "fail"}
        and rmst_gate.estimand.get("observed_only") is True
        and rmst_gate.estimand.get("independent_seed_count") == 2)
    censored_reference = _restricted_mean_time(
        ((1.0, True), (3.0, False)), 3.0)
    censored_faster = _restricted_mean_time(
        ((.5, True), (3.0, False)), 3.0)
    censoring_math = 0 <= censored_faster < censored_reference <= 3.0

    base_payload = report.to_dict()
    fault_expectations = {
        "missing_timing": "online_timing_missing",
        "negative_timing": "online_timing_negative",
        "nonmonotonic_timing": "timing_nonmonotonic",
        "changed_clock_domain": "timing_clock_domain_changed",
        "excluded_online_work": "online_compute_excluded",
        "double_counted_work": "timing_double_count",
        "target_extrapolation": "compute_target_extrapolated",
        "complete_case_deletion": "compute_complete_case_deletion",
        "insufficient_seed_clusters": "compute_seed_clusters_insufficient",
        "safety_censor_promoted": "compute_safety_censor_promoted",
    }

    def inject(name: str) -> dict:
        payload = deepcopy(base_payload)
        central = next(row for row in payload["trajectories"]
                       if row["arm"] in {
                           "full_control_detector_rl",
                           "predictive_hdfa_residual_rl"})
        timing = central["timing"]
        compute_gate = next(gate for gate in payload["gates"]
                            if gate["gate_id"] ==
                            "compute_aware_rmst_net_convergence_gain")
        if name == "missing_timing":
            central["timing"] = None
        elif name == "negative_timing":
            timing["qec_acquisition_s"] = -1.0
        elif name == "nonmonotonic_timing":
            timing["critical_path_events"][0]["sequence"] = 7
        elif name == "changed_clock_domain":
            timing["critical_path_events"][0]["clock_domain"] = "device:unknown"
        elif name == "excluded_online_work":
            timing["online_compute_critical_s"] += 1.0
        elif name == "double_counted_work":
            timing["critical_path_events"][0]["overlaps_event_ids"] = ("duplicate",)
        elif name == "target_extrapolation":
            compute_gate["estimand"]["observed_only"] = False
        elif name == "complete_case_deletion":
            compute_gate["estimand"]["included_pair_count"] -= 1
        elif name == "insufficient_seed_clusters":
            compute_gate["estimand"]["independent_seed_count"] = 1
        elif name == "safety_censor_promoted":
            compute_gate["estimand"]["staged_safety_censoring"] = True
            compute_gate["status"] = "pass"
        return payload

    caught_faults = {}
    for name, expected_code in fault_expectations.items():
        issues = validate_report_payload(inject(name))
        codes = {issue.code for issue in issues}
        caught_faults[name] = {
            "expected_code": expected_code, "observed_codes": tuple(sorted(codes)),
            "detected": expected_code in codes,
        }

    # Deliberately charge an extreme staged online-compute burden and recompute the
    # frozen estimand.  The primary gate must fail, not discard the affected pair.
    inflated = []
    for metric in report.metrics:
        if metric.arm == "predictive_hdfa_residual_rl":
            endpoints = tuple(replace(
                endpoint,
                e2e_time_s=(None if endpoint.e2e_time_s is None
                            else endpoint.e2e_time_s+100.0),
                censoring_e2e_time_s=(None if endpoint.censoring_e2e_time_s is None
                                      else endpoint.censoring_e2e_time_s+100.0),
                e2e_components_s={
                    **endpoint.e2e_components_s,
                    "online_compute_critical_s": float(
                        endpoint.e2e_components_s.get(
                            "online_compute_critical_s", 0.0))+100.0,
                }) for endpoint in metric.recovery_endpoints)
            metric = replace(metric, recovery_endpoints=endpoints)
        inflated.append(metric)
    indexed = {(row.scenario_id, row.seed, row.arm): row for row in inflated}
    inflated_pairs = [(
        indexed.get((item.scenario_id, seed, "full_control_detector_rl")),
        indexed.get((item.scenario_id, seed, "predictive_hdfa_residual_rl")))
        for item in scenario for seed in config.seeds]
    inflated_gate, _ = evaluation_runner._compute_aware_gates(inflated_pairs)
    caught_faults["intentionally_inflated_staged_compute"] = {
        "status": inflated_gate.status, "detected": inflated_gate.status != "pass"}
    fault_matrix_passed = all(row["detected"] for row in caught_faults.values())
    checks = (
        ValidationCheck(
            "symmetric_online_timing_complete", timing_complete and symmetric,
            {"trajectory_count": len(trajectories),
             "invalid": [row.timing.validate() if row.timing else ("missing",)
                         for row in trajectories]},
            "both central arms carry complete non-negative monotonic critical-path schedules",
            "QEC, diagnostics, actuation, online compute, host overhead and offline work stay separate."),
        ValidationCheck(
            "compute_aware_rmst_and_tail_evaluable", gate_evaluable,
            {"rmst_status": rmst_gate.status, "tail_status": tail_gate.status,
             "estimand": rmst_gate.estimand},
            "all matched timed pairs enter a seed-clustered censoring-aware RMST and tail estimand",
            "Reached-only medians and complete-case deletion are forbidden."),
        ValidationCheck(
            "censored_rmst_numerical_reference", censoring_math,
            {"reference_rmst": censored_reference,
             "faster_rmst": censored_faster},
            "known event/censoring examples produce ordered restricted means",
            "Non-recovery remains in the risk set through the common horizon."),
        ValidationCheck(
            "compute_evidence_fault_matrix", fault_matrix_passed,
            caught_faults,
            "timing, censoring, clustering and inflated-compute faults all fail closed",
            "The primary estimand cannot pass through missing/excluded work or complete-case deletion."),
    )
    return finalize_report(ValidationReport(
        "compute-accounting-validation.v1", "compute_accounting",
        all_passed(checks), checks,
        tuple({"arm": row.arm, "scenario": row.scenario_id, "seed": row.seed,
               "timing_invalidity": row.timing_invalidity_reasons}
              for row in report.metrics),
        {"evidence_layer": "development timing validation; not a performance result",
         "fault_count": len(caught_faults)},
    ))
