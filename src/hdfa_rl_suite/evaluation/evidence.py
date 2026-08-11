"""Evidence-layer and scientific-report contracts."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Mapping, Sequence


class EvidenceLayer(str, Enum):
    PUBLISHED_HARDWARE_EVIDENCE = "published_hardware_evidence"
    DECLARED_SURROGATE = "declared_surrogate"
    EXECUTED_REPOSITORY_SIMULATION = "executed_repository_simulation"
    CIRCUIT_LEVEL_LOGICAL_ADAPTER = "circuit_level_logical_adapter"
    MEASURED_DEPLOYMENT_RESULT = "measured_deployment_result"


@dataclass(frozen=True)
class EvidenceRecord:
    result_id: str
    layer: EvidenceLayer
    description: str
    measurement_role: str
    source: str
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReportContractIssue:
    code: str
    message: str


REQUIRED_DEVELOPMENT_FIGURES = (
    "latent_optimum_and_fixed_mismatch",
    "fixed_periodic_oracle_trajectories",
    "rl_mean_policy_trajectory",
    "exploratory_aggregate_and_damage",
    "active_risk_set_and_censoring",
    "logical_versus_detector_relation",
    "cycle_candidate_budget",
    "lifecycle_mode_and_reentry_burden",
)


def validate_evidence_records(records: Sequence[EvidenceRecord]) -> tuple[ReportContractIssue, ...]:
    issues: list[ReportContractIssue] = []
    if not records:
        return (ReportContractIssue("evidence_registry_missing",
                                    "every scientific report requires an evidence registry"),)
    for record in records:
        text = f"{record.description} {record.source}".lower()
        if (record.layer is EvidenceLayer.EXECUTED_REPOSITORY_SIMULATION
                and ("willow measurement" in text or "measured on willow" in text)):
            issues.append(ReportContractIssue(
                "simulation_promoted_to_willow",
                f"{record.result_id} describes repository simulation as Willow measurement"))
        if (record.layer is EvidenceLayer.DECLARED_SURROGATE
                and record.measurement_role in {"executed_control_count", "measured_control_count"}):
            issues.append(ReportContractIssue(
                "surrogate_promoted_to_executed_controls",
                f"{record.result_id} presents a surrogate count as executed controls"))
        if record.measurement_role == "short_pipeline_probe" and "convergence" in text:
            issues.append(ReportContractIssue(
                "pipeline_probe_promoted_to_convergence",
                f"{record.result_id} promotes a short throughput probe to a convergence study"))
        if record.measurement_role == "candidate_average" and "learned" in text:
            issues.append(ReportContractIssue(
                "candidate_average_promoted_to_learned_policy",
                f"{record.result_id} labels candidate-average performance as learned-policy performance"))
    return tuple(issues)


def validate_report_payload(payload: Mapping[str, object], *,
                            require_figures: bool = False) -> tuple[ReportContractIssue, ...]:
    """Validate recovery, sample accounting, censoring, and evidence semantics."""
    issues: list[ReportContractIssue] = []
    typed_records: list[EvidenceRecord] = []
    for item in payload.get("evidence_records", ()):
        if isinstance(item, EvidenceRecord):
            typed_records.append(item)
        elif isinstance(item, Mapping):
            try:
                layer_value = item["layer"]
                layer = (layer_value if isinstance(layer_value, EvidenceLayer)
                         else EvidenceLayer(str(layer_value)))
                typed_records.append(EvidenceRecord(
                    str(item["result_id"]), layer,
                    str(item["description"]), str(item["measurement_role"]),
                    str(item["source"]), tuple(item.get("limitations", ())),
                ))
            except (KeyError, TypeError, ValueError):
                issues.append(ReportContractIssue(
                    "evidence_record_invalid", "an evidence record is incomplete or uses an unknown layer"))
    issues.extend(validate_evidence_records(typed_records))

    for metric in payload.get("metrics", ()):
        if not isinstance(metric, Mapping):
            continue
        for field in ("qec_cycles", "candidate_evaluations", "candidate_cycles",
                      "mean_policy_evaluation_cycles"):
            if field not in metric:
                issues.append(ReportContractIssue(
                    "native_qec_accounting_missing",
                    f"metric {metric.get('arm', 'unknown')} lacks {field}"))
        if metric.get("candidate_evaluations", 0) and metric.get("candidate_budget_class") not in {
                "paper-scale", "validated-reduced-budget", "high-shot-reference",
                "reduced-budget-candidate", "smoke-test-only"}:
            issues.append(ReportContractIssue(
                "candidate_budget_unlabelled", "adaptive candidate data lack a declared budget class"))
        if (metric.get("mean_policy_detector_event_rate") is None
                and metric.get("candidate_evaluations", 0)):
            issues.append(ReportContractIssue(
                "mean_candidate_metric_conflation", "adaptive metrics lack independent mean-policy evaluation"))
        if metric.get("completion_status") == "censored" and not metric.get("censoring_reason"):
            issues.append(ReportContractIssue(
                "censoring_reason_missing", "censored runs require a declared reason and retained risk-set status"))
        for endpoint in metric.get("recovery_endpoints", ()):
            if not isinstance(endpoint, Mapping):
                continue
            if endpoint.get("target_fraction") == .90:
                if endpoint.get("status") == "reached" and endpoint.get("detector_cycles") is None:
                    issues.append(ReportContractIssue(
                        "invalid_90pct_extrapolation", "90% recovery is labelled reached without an observed cycle"))
                if endpoint.get("status") not in {"reached", "censored", "missing"}:
                    issues.append(ReportContractIssue(
                        "recovery_status_invalid", "recovery must distinguish reached, censored, and missing"))

    for summary in payload.get("recovery_summaries", ()):
        if (isinstance(summary, Mapping) and summary.get("censored_count", 0)
                and summary.get("complete_case_superiority", False)):
            issues.append(ReportContractIssue(
                "informative_censoring_promoted",
                "a censored complete-case summary cannot be promoted as a treatment effect"))

    central_arms = {"full_control_detector_rl", "predictive_hdfa_no_residual",
                    "predictive_hdfa_residual_rl"}
    for trajectory in payload.get("trajectories", ()):
        if not isinstance(trajectory, Mapping) or trajectory.get("arm") not in central_arms:
            continue
        timing = trajectory.get("timing")
        if not isinstance(timing, Mapping):
            issues.append(ReportContractIssue(
                "online_timing_missing",
                f"central arm {trajectory.get('arm')} lacks symmetric online timing"))
            continue
        required = (
            "qec_acquisition_s", "diagnostic_downtime_s",
            "actuation_acknowledgement_s", "online_compute_critical_s",
            "simulator_kernel_host_s", "total_observed_host_wall_s",
            "offline_logical_evaluation_s", "offline_report_analysis_s",
            "stage_compute_s", "critical_path_events", "timing_complete",
            "clock_policy")
        if any(field not in timing for field in required):
            issues.append(ReportContractIssue(
                "online_timing_incomplete", "a central timing record lacks required components"))
            continue
        durations = [timing.get(field) for field in required[:8]]
        if any(not isinstance(value, (int, float)) or value < 0
               or not math.isfinite(float(value)) for value in durations):
            issues.append(ReportContractIssue(
                "online_timing_negative", "timing durations must be finite and non-negative"))
        if (timing.get("schema_version") != "online-critical-path-timing.v1"
                or timing.get("clock_policy") != "serial-hybrid-clock-critical-path.v1"):
            issues.append(ReportContractIssue(
                "timing_clock_domain_changed",
                "central arms require the frozen monotonic hybrid-clock policy"))
        stage_compute = timing.get("stage_compute_s")
        if (not isinstance(stage_compute, Mapping)
                or abs(sum(float(value) for value in stage_compute.values())
                       - float(timing.get("online_compute_critical_s", 0.0))) > 1e-8):
            issues.append(ReportContractIssue(
                "online_compute_excluded",
                "online critical compute differs from the complete stage decomposition"))
        if not timing.get("timing_complete"):
            issues.append(ReportContractIssue(
                "online_timing_incomplete", "timing record is explicitly incomplete"))
        events = timing.get("critical_path_events")
        if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
            issues.append(ReportContractIssue(
                "critical_path_schedule_missing", "timing requires a timestamped schedule"))
        else:
            typed_events = [event for event in events if isinstance(event, Mapping)]
            sequences = [event.get("sequence") for event in typed_events]
            if len(typed_events) != len(events) or sequences != list(range(len(events))):
                issues.append(ReportContractIssue(
                    "timing_nonmonotonic", "critical-path events are not monotonic"))
            if any(event.get("overlaps_event_ids") for event in typed_events):
                issues.append(ReportContractIssue(
                    "timing_double_count", "unresolved event overlap would double-count E2E time"))
            expected_domains = {
                "online_compute": "host:perf_counter_ns",
                "qec_acquisition": "simulated-device-time-s",
                "diagnostic_downtime": "simulated-device-time-s",
                "actuation_acknowledgement": "simulated-device-time-s",
            }
            if any(event.get("clock_domain") != expected_domains.get(event.get("component"))
                   for event in typed_events):
                issues.append(ReportContractIssue(
                    "timing_clock_domain_changed",
                    "critical-path components were measured in an unexpected clock domain"))
            component_fields = {
                "qec_acquisition": "qec_acquisition_s",
                "diagnostic_downtime": "diagnostic_downtime_s",
                "actuation_acknowledgement": "actuation_acknowledgement_s",
                "online_compute": "online_compute_critical_s",
            }
            for component, field in component_fields.items():
                scheduled = sum(float(event.get("duration_s", 0.0))
                                for event in typed_events
                                if event.get("component") == component
                                and event.get("on_critical_path")
                                and not event.get("excluded_as_offline"))
                if abs(scheduled-float(timing.get(field, 0.0))) > 1e-8:
                    issues.append(ReportContractIssue(
                        "timing_schedule_mismatch",
                        f"critical-path schedule differs from declared {field}"))
        for endpoint in next((metric.get("recovery_endpoints", ())
                              for metric in payload.get("metrics", ())
                              if isinstance(metric, Mapping)
                              and metric.get("scenario_id") == trajectory.get("scenario_id")
                              and metric.get("seed") == trajectory.get("seed")
                              and metric.get("arm") == trajectory.get("arm")), ()):
            if not isinstance(endpoint, Mapping):
                continue
            components = endpoint.get("e2e_components_s")
            if endpoint.get("timing_status") == "valid" and isinstance(components, Mapping):
                expected = sum(float(components.get(field, 0.0)) for field in (
                    "qec_acquisition_s", "diagnostic_downtime_s",
                    "actuation_acknowledgement_s", "online_compute_critical_s"))
                observed = (endpoint.get("e2e_time_s") if endpoint.get("status") == "reached"
                            else endpoint.get("censoring_e2e_time_s"))
                if not isinstance(observed, (int, float)) or abs(float(observed)-expected) > 1e-8:
                    issues.append(ReportContractIssue(
                        "endpoint_timing_formula_mismatch",
                        "recovery endpoint does not equal the frozen E2E component sum"))

    compute_gates = [gate for gate in payload.get("gates", ())
                     if isinstance(gate, Mapping)
                     and gate.get("gate_id") == "compute_aware_rmst_net_convergence_gain"]
    if compute_gates:
        gate = compute_gates[0]
        estimand = gate.get("estimand")
        if not isinstance(estimand, Mapping):
            issues.append(ReportContractIssue(
                "compute_estimand_missing", "compute-aware gate lacks its frozen estimand"))
        else:
            if not estimand.get("observed_only"):
                issues.append(ReportContractIssue(
                    "compute_target_extrapolated", "compute recovery target must be observed only"))
            if not estimand.get("rmst_horizon_s"):
                issues.append(ReportContractIssue(
                    "rmst_horizon_missing", "compute-aware RMST requires one common horizon"))
            if int(estimand.get("independent_seed_count", 0)) < 2:
                issues.append(ReportContractIssue(
                    "compute_seed_clusters_insufficient",
                    "compute-aware uncertainty requires independent seed clusters"))
            if (estimand.get("cluster_unit") != "independent disturbance seed"
                    or estimand.get("complete_case_deletion") is not False
                    or int(estimand.get("included_pair_count", -1))
                    != int(estimand.get("declared_pair_count", -2))):
                issues.append(ReportContractIssue(
                    "compute_complete_case_deletion",
                    "all declared matched pairs must enter the seed-clustered estimand"))
            if estimand.get("staged_safety_censoring") and gate.get("status") == "pass":
                issues.append(ReportContractIssue(
                    "compute_safety_censor_promoted",
                    "staged safety censoring cannot produce a passing compute gate"))

    config = payload.get("config", {})
    estimator_v2 = (isinstance(config, Mapping)
                    and config.get("estimator_schema_version") == "estimators.v2")
    estimator_fields = {
        "worst_matched_ratio", "median_matched_ratio",
        "cluster_aggregate_ratio", "cluster_aggregate_ci95",
        "rmst_difference", "rmst_ci95", "tail_difference", "tail_ci95",
        "gate_decision_statistic", "gate_threshold", "gate_status",
    }
    for gate in payload.get("gates", ()):
        if not isinstance(gate, Mapping):
            continue
        estimators = gate.get("estimators")
        if estimator_v2 and not isinstance(estimators, Mapping):
            issues.append(ReportContractIssue(
                "estimator_bundle_missing",
                f"gate {gate.get('gate_id')} lacks the estimators.v2 bundle"))
            continue
        if not isinstance(estimators, Mapping):
            continue
        missing_fields = estimator_fields-set(estimators)
        extra_fields = set(estimators)-estimator_fields
        if missing_fields or extra_fields:
            issues.append(ReportContractIssue(
                "estimator_schema_mismatch",
                f"gate {gate.get('gate_id')} estimator fields differ: "
                f"missing={sorted(missing_fields)} extra={sorted(extra_fields)}"))
        embedded_cis = any(isinstance(estimators.get(name), Mapping) for name in (
            "cluster_aggregate_ci95", "rmst_ci95", "tail_ci95"))
        if gate.get("confidence_interval") is not None and embedded_cis:
            issues.append(ReportContractIssue(
                "estimator_ci_ambiguous",
                f"gate {gate.get('gate_id')} attaches a top-level CI under estimators.v2"))
        if estimators.get("gate_status") != gate.get("status"):
            issues.append(ReportContractIssue(
                "estimator_gate_status_mismatch",
                f"gate {gate.get('gate_id')} stores inconsistent statuses"))
        for value_name, ci_name in (
                ("cluster_aggregate_ratio", "cluster_aggregate_ci95"),
                ("rmst_difference", "rmst_ci95"),
                ("tail_difference", "tail_ci95")):
            ci = estimators.get(ci_name)
            value = estimators.get(value_name)
            if isinstance(ci, Mapping):
                estimate = ci.get("estimate")
                if (not isinstance(value, (int, float))
                        or not isinstance(estimate, (int, float))
                        or abs(float(value)-float(estimate)) > 1e-12):
                    issues.append(ReportContractIssue(
                        "estimator_ci_mismatch",
                        f"{ci_name} is not the interval for {value_name} in "
                        f"gate {gate.get('gate_id')}"))
        if (estimators.get("worst_matched_ratio") is not None
                and isinstance(estimators.get("cluster_aggregate_ci95"), Mapping)
                and gate.get("measured_ratio") == estimators.get("worst_matched_ratio")
                and gate.get("confidence_interval") is not None):
            issues.append(ReportContractIssue(
                "worst_ratio_with_aggregate_ci",
                f"gate {gate.get('gate_id')} displays an aggregate CI as worst-pair uncertainty"))

        estimand = gate.get("estimand", {})
        if isinstance(estimand, Mapping):
            if (estimand.get("post_hoc_horizon")
                    and estimand.get("analysis_role") == "confirmatory"):
                issues.append(ReportContractIssue(
                    "post_hoc_horizon_promoted",
                    f"gate {gate.get('gate_id')} promotes a post-hoc horizon"))
            if (estimand.get("complete_case_deletion") is True
                    and gate.get("status") == "pass"):
                issues.append(ReportContractIssue(
                    "censored_complete_case_promoted",
                    f"gate {gate.get('gate_id')} passes after complete-case deletion"))

    for diagnostic in payload.get("diagnostics", ()):
        if not isinstance(diagnostic, Mapping):
            continue
        if diagnostic.get("deduplicated") and not diagnostic.get("scenario_id"):
            issues.append(ReportContractIssue(
                "deduplicated_diagnostic_scenario_missing",
                "deduplicated diagnostics must retain scenario identity"))
        if diagnostic.get("status") in {"fail", "failure"} and (
                diagnostic.get("scenario_id") is None
                or diagnostic.get("seed") is None):
            issues.append(ReportContractIssue(
                "failure_identity_missing",
                "failure diagnostics require scenario and seed identity"))

    if require_figures:
        available = {str(item.get("figure_id")) for item in payload.get("figures", ())
                     if isinstance(item, Mapping)}
        for figure_id in REQUIRED_DEVELOPMENT_FIGURES:
            if figure_id not in available:
                issues.append(ReportContractIssue(
                    "required_figure_missing", f"required figure {figure_id!r} is missing"))
    return tuple(issues)


def canonical_benchmark_evidence() -> tuple[EvidenceRecord, ...]:
    return (
        EvidenceRecord(
            "detector_control_comparison",
            EvidenceLayer.EXECUTED_REPOSITORY_SIMULATION,
            "Matched trajectories executed by the repository simulator.",
            "controller_comparison", "hdfa_rl_suite.evaluation.benchmark",
            ("not a Willow reproduction", "not a measured deployment result")),
        EvidenceRecord(
            "logical_memory_evidence",
            EvidenceLayer.CIRCUIT_LEVEL_LOGICAL_ADAPTER,
            "Stim rotated-memory samples decoded by PyMatching MWPM.",
            "logical_evaluation", "hdfa_rl_suite.logical.surface_code",
            ("evaluation-only simulator control-error mapping",)),
        EvidenceRecord(
            "architecture_acceptance_targets",
            EvidenceLayer.DECLARED_SURROGATE,
            "Predeclared compute-aware RMST, tail, 5x/2x and no-regression hypotheses evaluated on simulator trajectories; the former 10x candidate ratio is secondary.",
            "acceptance_target", "00_Architecture_Overview_Revised.docx",
            ("targets are not assumed results",)),
    )
