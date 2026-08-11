"""Claim-specific numeric comparisons; no master scalar certification."""
from __future__ import annotations

from typing import Any, Mapping


def compare_to_anchor(value: float | None, anchor: float | None, *, relative_tolerance: float = .25) -> dict[str, Any]:
    if anchor is None:
        return {"verdict": "NOT_PUBLICLY_IDENTIFIABLE", "value": value, "anchor": None}
    if value is None:
        return {"verdict": "NOT_YET_RUN", "value": None, "anchor": anchor}
    error = abs(value-anchor)/max(abs(anchor), 1e-15)
    return {"verdict": "WITHIN_TOLERANCE_SYNTHETIC_MATCH" if error <= relative_tolerance else "MISMATCH",
            "value": value, "anchor": anchor, "relative_error": error, "relative_tolerance": relative_tolerance}


def family_checks(family: str, validation: Mapping[str, Any]) -> list[dict[str, Any]]:
    metrics = validation.get("metrics", {})
    if family == "FIGURE5A_REAL_TIME_STEERING":
        return [{"quantity": "critical_frequency_epochs_inverse", **compare_to_anchor(metrics.get("estimated_critical_frequency"), 1/150)}]
    if family == "FIGURE5B_SPARSE_SCALING":
        return [{"quantity": "distance_15_p30_controls", **compare_to_anchor(metrics.get("distance_15_p30_controls"), 38670, relative_tolerance=0)}]
    if family == "FIGURE5C_CONVERGENCE_LAW":
        cvs = metrics.get("gamma_distance_cv_by_p", [])
        return [{"quantity": "distance_independence_cv", "value": max(cvs) if cvs else None, "anchor": "<=0.15",
                 "verdict": "WITHIN_TOLERANCE_SYNTHETIC_MATCH" if cvs and max(cvs) <= .15 else "NOT_YET_RUN"}]
    if family == "NATURAL_DRIFT_SPECTRAL_SUPPRESSION":
        return [{"quantity": "low_frequency_suppression_db", **compare_to_anchor(metrics.get("median_suppression_db"), 4.0, relative_tolerance=.35)}]
    if family == "RANDOMIZED_RECOVERY_AFTER_SPOIL":
        return [{"quantity": "recovery_epoch", **compare_to_anchor(metrics.get("median_recovery_epoch"), 1000, relative_tolerance=.5)}]
    if family == "STEP_RESPONSE_INJECTED_DRIFT":
        return [{"quantity": "response_time_90_epochs", **compare_to_anchor(metrics.get("median_response_time_90_epochs"), 130, relative_tolerance=.35)}]
    return []

