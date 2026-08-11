"""Immutable V17 vocabulary and evidence boundary."""
from __future__ import annotations

V17_SCHEMA = "google-pure-v17-figure5a-dynamics.v1"

NONFINAL = {
    "final_evidence": False,
    "scientifically_valid": False,
    "paper_comparable": False,
    "paper_equivalence_claim_permitted": False,
    "reference_evidence_complete": False,
    "heldout_seeds_consumed": False,
    "source_budget_auto_launched": False,
    "long_run_auto_launched": False,
}


def nonfinal(value: dict) -> dict:
    """Attach the V17 evidence boundary without allowing caller promotion."""
    return {"schema_version": V17_SCHEMA, **value, **NONFINAL}
