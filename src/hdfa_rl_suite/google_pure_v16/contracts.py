"""Immutable V16 vocabulary and evidence boundary."""
from __future__ import annotations

V16_SCHEMA = "google-pure-v16-source-normalized-optimizer.v1"

SOURCE_LITERAL = "SOURCE_LITERAL"
SOURCE_DERIVED = "SOURCE_DERIVED"
SOURCE_REFERENCED_PRIMARY_METHOD = "SOURCE_REFERENCED_PRIMARY_METHOD"
SOURCE_UNSPECIFIED_PREREGISTERED = "SOURCE_UNSPECIFIED_PREREGISTERED"
LEGACY_INHERITED = "LEGACY_INHERITED"

SOURCE_CLASSES = {
    SOURCE_LITERAL,
    SOURCE_DERIVED,
    SOURCE_REFERENCED_PRIMARY_METHOD,
    SOURCE_UNSPECIFIED_PREREGISTERED,
    LEGACY_INHERITED,
}

NONFINAL = {
    "final_evidence": False,
    "scientifically_valid": False,
    "paper_comparable": False,
    "paper_equivalence_claim_permitted": False,
    "reference_evidence_complete": False,
    "heldout_seeds_consumed": False,
    "long_run_auto_launched": False,
}

HYPOTHESIS_STATUSES = {
    "EXECUTION_PATH_FAILURE": "RULED_OUT",
    "V15_BOUNDARY_APPLICATION_FAILURE": "RULED_OUT",
    "CALIBRATION_TRAINING_OBJECTIVE_MISMATCH":
        "RULED_OUT_IN_CURRENT_SYNTHETIC_DRIVERS",
    "PRIMARY_ACTIVE_HYPOTHESIS":
        "OPTIMIZER_POLICY_HYPERPARAMETERS_NOT_CONSISTENT_WITH_FINAL_SOURCE_NORMALIZATION",
    "SECONDARY_ACTIVE_HYPOTHESIS":
        "CURRENT_ABC_COMPARISONS_ARE_NOT_PHYSICALLY_MATCHED_ACROSS_PARAMETERIZATIONS",
}


def nonfinal(value: dict) -> dict:
    """Attach the evidence boundary without permitting promotion by a caller."""
    return {"schema_version": V16_SCHEMA, **value, **NONFINAL}
