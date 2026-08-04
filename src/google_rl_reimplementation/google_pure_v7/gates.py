"""Three-layer scientific gate contract."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .reporting import write_report


@dataclass(frozen=True)
class ScientificGate:
    artifact_complete: bool
    mechanism_valid: bool
    performance_pass: bool
    blocking_reasons: tuple[str, ...] = ()

    @property
    def passes(self) -> bool:
        return self.artifact_complete and self.mechanism_valid and self.performance_pass and not self.blocking_reasons

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["blocking_reasons"] = list(self.blocking_reasons)
        value["gate_pass"] = self.passes
        return value


def gate_from_result(result: dict[str, Any], *, required_fields: Iterable[str], mechanism_checks: Iterable[bool],
                     performance_checks: Iterable[bool], mechanism_reasons: Iterable[str] = (),
                     performance_reasons: Iterable[str] = ()) -> ScientificGate:
    complete = all(field in result for field in required_fields)
    mechanisms = tuple(bool(value) for value in mechanism_checks)
    performances = tuple(bool(value) for value in performance_checks)
    reasons: list[str] = []
    if not complete:
        reasons.append("required artifact fields missing")
    reasons.extend(reason for ok, reason in zip(mechanisms, mechanism_reasons) if not ok)
    reasons.extend(reason for ok, reason in zip(performances, performance_reasons) if not ok)
    return ScientificGate(complete, complete and all(mechanisms), complete and all(performances), tuple(reasons))


def write_scientific_gate_contract() -> dict[str, Any]:
    payload = {
        "schema_version": "google-pure-v7-scientific-gates.v1",
        "layers": {
            "artifact_complete": "expected file exists, schema is valid, and required fields are present",
            "mechanism_valid": "estimator and experiment are mathematically and scientifically well-defined",
            "performance_pass": "predeclared quantitative scientific thresholds are met",
        },
        "global_rule": "all_required_artifacts_complete AND all_primary_mechanisms_valid AND all_primary_performance_gates_pass",
        "status_semantics": {
            "study_without_survivor": "STUDY_COMPLETE_NO_PASSING_CONFIGURATION",
            "invalid_estimator": "INVALID_DIAGNOSTIC",
            "artifact_status_pass_is_not_a_gate": True,
        },
        "certification_seeds_consumed": False,
        "status": "PASS",
    }
    return write_report("scientific_gate_contract", payload, "Scientific Gate Contract")
