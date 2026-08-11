"""Recovery from a deliberately spoiled policy, distinct from a drift step."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from hdfa_rl_suite.google_pure_source_exact.paper_families.recovery import acquire_recovery_condition


def acquire_condition(protocol: Mapping[str, Any], condition: Mapping[str, Any]) -> dict[str, Any]:
    return acquire_recovery_condition(protocol, condition)


def validation(rows: list[dict[str, Any]], mode: str) -> tuple[bool, list[str], dict[str, Any]]:
    reasons = []
    if any(not row.get("not_a_step_response") or row["randomized_fraction"] <= 0 for row in rows): reasons.append("recovery/step-response family conflation")
    if any(row.get("controller_mode") != "PAPER_DIRECT_SIGMA" or row.get("parameterization") != "direct_sigma" for row in rows):
        reasons.append("amended direct-sigma controller did not execute")
    observed = [row["recovery_epoch"] for row in rows if row["recovery_epoch"] is not None]
    censored = len(rows)-len(observed)
    if censored:
        reasons.append(f"recovery threshold not reached in {censored}/{len(rows)} runs")
    outcome = "RECOVERY_NOT_REACHED_WITHIN_HORIZON" if censored == len(rows) else (
        "PARTIAL_RECOVERY_WITH_CENSORING" if censored else "RECOVERY_REACHED")
    return not reasons, reasons, {"median_recovery_epoch": float(np.median(observed)) if observed else None,
        "paper_anchor_epochs": 1000, "censored_count": censored, "run_count": len(rows),
        "reached_fraction": len(observed)/len(rows) if rows else 0.0,
        "recovery_threshold": "sustained excess logical-risk proxy <= 10% of initial excess",
        "outcome": outcome, "median_identifiable": bool(observed) and censored == 0,
        "source_structure_match": all(row.get("source_structure_match") for row in rows),
        "paper_comparable": False,
        "blocking_reasons": ["the randomized Willow policy and detector mapping are unavailable"]}
