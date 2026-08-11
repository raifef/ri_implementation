"""Frozen V12 scientific classifications and fail-closed evidence contracts."""
from __future__ import annotations

V12_SCHEMA = "google-pure-v12-directional-lineage.v1"

CURRENT_CLASSIFICATIONS = {
    "FIGURE5A_REAL_TIME_STEERING": "PARTIAL",
    "FIGURE5B_SPARSE_SCALING": "INVALID_DIAGNOSTIC",
    "FIGURE5C_CONVERGENCE_LAW": "INVALID_DIAGNOSTIC",
    "NATURAL_DRIFT_SPECTRAL_SUPPRESSION": "PARTIAL",
    "RANDOMIZED_RECOVERY_AFTER_SPOIL": "FAILED",
    "STEP_RESPONSE_INJECTED_DRIFT": "FAILED",
    "DIRECT_SIGMA_INTEGRATION": "OPERATIONAL",
    "FINAL_GOOGLE_BASELINE": "NOT_READY",
    "HDFA_COMPARISON": "NOT_READY",
}

DIAGNOSTIC_CASES = (
    "BEST_SLOW_FIGURE5A",
    "FAILED_STEP_RESPONSE",
    "FAILED_RANDOMIZED_RECOVERY",
)

NONFINAL_FIELDS = {
    "final_evidence": False,
    "paper_equivalence_claim_permitted": False,
    "scientifically_valid": False,
    "staged_controller_run": False,
}


def fail_closed_status(*, gates: dict[str, bool], classifications: dict[str, str] | None = None) -> dict:
    """Return a non-promotable status even when all development gates pass."""
    return {
        "schema_version": V12_SCHEMA,
        "classifications": dict(CURRENT_CLASSIFICATIONS if classifications is None else classifications),
        "gates": {str(key): bool(value) for key, value in gates.items()},
        "development_gates_passed": bool(gates) and all(gates.values()),
        **NONFINAL_FIELDS,
    }
