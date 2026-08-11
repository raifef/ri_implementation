"""Immutable V18 vocabulary and evidence boundary."""
from __future__ import annotations

V18_SCHEMA = "google-pure-v18-quick-figure5a-identification.v1"

NONFINAL = {
    "final_evidence": False,
    "scientifically_valid": False,
    "paper_comparable": False,
    "paper_equivalence_claim_permitted": False,
    "reference_evidence_complete": False,
    "heldout_seeds_consumed": False,
    "source_budget_auto_launched": False,
    "reference_campaign_auto_launched": False,
    "long_run_auto_launched": False,
}


def nonfinal(value: dict) -> dict:
    return {"schema_version": V18_SCHEMA, **value, **NONFINAL}
