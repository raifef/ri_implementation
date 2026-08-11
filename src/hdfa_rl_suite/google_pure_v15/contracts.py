"""Immutable vocabulary for the V15 closure programme."""
from __future__ import annotations

V15_SCHEMA = "google-pure-v15-complete-open-issue-closure.v1"

CONFIRMED_FAULT = "CONFIRMED_FAULT"
CONFIRMED_ARCHITECTURAL_LIMITATION = "CONFIRMED_ARCHITECTURAL_LIMITATION"
UNRESOLVED_RISK = "UNRESOLVED_RISK"
RULED_OUT = "RULED_OUT_IN_CURRENT_SIMULATOR"
RESOLVED = "RESOLVED"
SOURCE_NON_IDENTIFIABLE = "SOURCE_NON_IDENTIFIABLE"
HARDWARE_ONLY_UNTESTED = "HARDWARE_ONLY_UNTESTED"

ALLOWED_STATUSES = {
    CONFIRMED_FAULT, CONFIRMED_ARCHITECTURAL_LIMITATION, UNRESOLVED_RISK,
    RULED_OUT, RESOLVED, SOURCE_NON_IDENTIFIABLE, HARDWARE_ONLY_UNTESTED,
}
TERMINAL_STATUSES = {RULED_OUT, RESOLVED, SOURCE_NON_IDENTIFIABLE, HARDWARE_ONLY_UNTESTED}

PUBLIC_MATCHED = "PUBLIC_SOURCE_IDENTIFIABLE_AND_MATCHED"
PUBLIC_MISMATCHED = "PUBLIC_SOURCE_IDENTIFIABLE_AND_MISMATCHED"
PUBLIC_NON_IDENTIFIABLE = "PUBLIC_SOURCE_NON_IDENTIFIABLE"
HARDWARE_UNTESTED = "HARDWARE_ONLY_UNTESTED"
SOURCE_GAP_CLASSES = {PUBLIC_MATCHED, PUBLIC_MISMATCHED, PUBLIC_NON_IDENTIFIABLE, HARDWARE_UNTESTED}

ISSUES = {
    "A": "V12 curvature substitution",
    "B": "mathematical target s_i",
    "C": "detector-degree double count",
    "D": "operating-point dependence",
    "E": "cross-control coupling",
    "F": "calibration uncertainty",
    "G": "calibration leakage",
    "H": "Figure 5b boundary map",
    "I": "broad curvature distribution",
    "J": "non-diagonal Hessian",
    "K": "slow modes",
    "L": "graph-gradient dilution",
    "M": "information dilution",
    "N": "correlations and effective sample size",
    "O": "direct-sigma floor",
    "P": "mean versus scale conditioning",
    "Q": "finite horizon versus plateau",
    "R": "EDR, physical-error and logical-error alignment",
    "S": "Figure 5a batch latency",
    "T": "step-response source comparability",
    "U": "natural-drift statistical power",
    "V": "PPO lifecycle",
    "W": "state and candidate provenance",
    "X": "offline decoder steering",
    "Y": "immutable held-out reference gate",
    "Z": "proprietary and hardware non-identifiability",
}

NONFINAL = {
    "final_evidence": False,
    "scientifically_valid": False,
    "paper_comparable": False,
    "paper_equivalence_claim_permitted": False,
    "reference_evidence_complete": False,
    "comparison_permitted": False,
}


def nonfinal(value: dict) -> dict:
    """Attach the evidence boundary without allowing callers to override it."""
    return {"schema_version": V15_SCHEMA, **value, **NONFINAL}
