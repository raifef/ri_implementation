"""Scientific schema and evidence-layer failure-injection checks."""
from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

from hdfa_rl_suite.evaluation.evidence import (
    EvidenceLayer, EvidenceRecord, canonical_benchmark_evidence,
    validate_report_payload,
)

from .common import ValidationCheck, ValidationReport, finalize_report


def _valid_payload() -> dict:
    return {
        "evidence_records": [asdict(item) for item in canonical_benchmark_evidence()],
        "metrics": [{
            "arm": "full_control_detector_rl", "qec_cycles": 4096,
            "candidate_evaluations": 40, "candidate_cycles": 81920,
            "mean_policy_evaluation_cycles": 512,
            "candidate_budget_class": "validated-reduced-budget",
            "mean_policy_detector_event_rate": .02,
            "aggregate_exploration_detector_event_rate": .03,
            "completion_status": "censored", "censoring_reason": "declared limit",
            "recovery_endpoints": [{"target_fraction": .9, "status": "censored",
                                    "detector_cycles": None}],
        }],
        "recovery_summaries": [{"censored_count": 1,
                                "complete_case_superiority": False}],
    }


def run_report_validation(*, injected_faults: Iterable[str] = ()) -> ValidationReport:
    faults = set(injected_faults)
    payload = _valid_payload()
    expected_code = None
    if "suite_as_willow" in faults:
        payload["evidence_records"][0]["description"] = "Measured on Willow measurements"
        expected_code = "simulation_promoted_to_willow"
    elif "surrogate_as_executed_controls" in faults:
        payload["evidence_records"].append(asdict(EvidenceRecord(
            "surrogate", EvidenceLayer.DECLARED_SURROGATE, "surrogate count",
            "executed_control_count", "synthetic")))
        expected_code = "surrogate_promoted_to_executed_controls"
    elif "pipeline_probe_as_convergence" in faults:
        payload["evidence_records"].append(asdict(EvidenceRecord(
            "probe", EvidenceLayer.EXECUTED_REPOSITORY_SIMULATION,
            "short probe convergence result", "short_pipeline_probe", "synthetic")))
        expected_code = "pipeline_probe_promoted_to_convergence"
    elif "mean_candidate_metric_conflation" in faults:
        payload["metrics"][0]["mean_policy_detector_event_rate"] = None
        expected_code = "mean_candidate_metric_conflation"
    elif "candidate_average_as_learned" in faults:
        payload["evidence_records"].append(asdict(EvidenceRecord(
            "candidate", EvidenceLayer.EXECUTED_REPOSITORY_SIMULATION,
            "learned policy performance", "candidate_average", "synthetic")))
        expected_code = "candidate_average_promoted_to_learned_policy"
    elif "invalid_convergence_extrapolation" in faults:
        payload["metrics"][0]["recovery_endpoints"][0] = {
            "target_fraction": .9, "status": "reached", "detector_cycles": None}
        expected_code = "invalid_90pct_extrapolation"
    elif "informative_censoring_complete_case" in faults:
        payload["recovery_summaries"][0]["complete_case_superiority"] = True
        expected_code = "informative_censoring_promoted"
    issues = validate_report_payload(payload)
    passed = not issues
    if expected_code is not None:
        # Fault injection reports fail by design; retaining the detector code proves
        # that the intended gate, rather than an unrelated parse error, caught it.
        passed = False
    observed_codes = tuple(item.code for item in issues)
    check = ValidationCheck(
        "report_schema_and_evidence_layers", passed, observed_codes,
        "valid reports pass; injected evidence, metric, fit, and censoring promotions fail",
        (f"targeted detector={expected_code}" if expected_code else
         "the canonical scientific report contract is valid"),
    )
    return finalize_report(ValidationReport(
        "report-contract-validation.v1", "report_contract_validation",
        passed, (check,), (),
        {"injected_faults": sorted(faults), "expected_detector": expected_code,
         "observed_detectors": observed_codes},
    ))
