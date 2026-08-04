"""Figure 5a sampled-candidate steerability conditions."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from google_rl_reimplementation.google_pure_v7.figure5.panel_a import _condition as v7_condition


def acquire_condition(protocol: Mapping[str, Any], condition: Mapping[str, Any]) -> dict[str, Any]:
    legacy_plan = {"config": protocol["config"]}
    arrays, metadata = v7_condition(legacy_plan, dict(condition))
    return {**metadata, "seed": int(condition["seed"]), "frequency": float(condition["frequency"]),
            "entropy_coefficient": float(condition["entropy_coefficient"]),
            "trajectory": {key: np.asarray(value).tolist() for key, value in arrays.items()},
            "normalization_contract": "(C_fixed-C_candidates)/(C_fixed-C_optimal)",
            "mean_policy_reported_separately": True}


def validation(rows: list[dict[str, Any]], mode: str) -> tuple[bool, list[str], dict[str, Any]]:
    reasons = []
    if any(row.get("normalization_contract") != "(C_fixed-C_candidates)/(C_fixed-C_optimal)" for row in rows): reasons.append("wrong candidate normalization")
    if any(not row.get("mean_policy_reported_separately") for row in rows): reasons.append("candidate/mean policy conflation")
    values = [float(row["improvement_candidate"]) for row in rows]
    if mode in {"reference", "paper-scale"} and values and not min(values) <= 0 <= max(values): reasons.append("zero contour is not bracketed")
    return not reasons, reasons, {"candidate_improvement_range": [min(values), max(values)] if values else None,
                                  "zero_contour_bracketed": bool(values and min(values) <= 0 <= max(values)),
                                  "paper_critical_frequency": 1/150}
