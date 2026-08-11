"""Injected optimum-step response, kept separate from policy spoil recovery."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from hdfa_rl_suite.google_pure_source_exact.paper_families.step import acquire_step_condition


def acquire_condition(protocol: Mapping[str, Any], condition: Mapping[str, Any]) -> dict[str, Any]:
    return acquire_step_condition(protocol, condition)


def validation(rows: list[dict[str, Any]], mode: str) -> tuple[bool, list[str], dict[str, Any]]:
    reasons = []
    if any(row.get("policy_spoil_applied") for row in rows): reasons.append("step/policy-spoil family conflation")
    if any(row.get("controller_mode") != "PAPER_DIRECT_SIGMA" or row.get("parameterization") != "direct_sigma" for row in rows):
        reasons.append("amended direct-sigma controller did not execute")
    responses = [row["response"] for row in rows]
    censored = sum(item["response_time_90_epochs"] is None for item in responses)
    if censored: reasons.append(f"injected-target 90% crossing not reached in {censored}/{len(rows)} runs")
    if any(item["response_fraction_of_injected_target"] < .9 for item in responses):
        reasons.append("one or more learned-mean traces did not achieve 90% of the injected target")
    if any(not item.get("fit_valid", False) for item in responses):
        reasons.append("one or more exponential fits failed the credibility gate")
    times = [item["response_time_90_epochs"] for item in responses if item["response_time_90_epochs"] is not None]
    fractions = [item["response_fraction_of_injected_target"] for item in responses]
    return not reasons, reasons, {"median_response_time_90_epochs": float(np.median(times)) if times else None,
        "paper_anchor_epochs": 130, "censored_count": censored,
        "median_final_response_fraction": float(np.median(fractions)) if fractions else None,
        "crossing_definition": "fraction_of_injected_target_excursion",
        "outcome": "STEP_TARGET_NOT_REACHED_WITHIN_HORIZON" if censored else "STEP_TARGET_REACHED",
        "source_structure_match": all(row.get("source_structure_match") for row in rows),
        "paper_comparable": False,
        "blocking_reasons": ["the proprietary Willow 924-control inventory and detector map are unavailable"]}
