"""Figure 5c source-axis convergence-law trajectories and fits."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from hdfa_rl_suite.google_pure_source_exact.paper_families.scaling import acquire_scaling_condition


def acquire_condition(protocol: Mapping[str, Any], condition: Mapping[str, Any]) -> dict[str, Any]:
    result = acquire_scaling_condition(protocol, condition)
    result.update({"source_x_axis": "1-Lambda/Lambda*",
                   "source_y_axis": "1e2 d_t Lambda/Lambda*"})
    return result


def validation(rows: list[dict[str, Any]], mode: str) -> tuple[bool, list[str], dict[str, Any]]:
    reasons, cvs = [], []
    if any(not np.isfinite(row["gamma_times_100"]) for row in rows): reasons.append("non-finite local convergence fit")
    if any(row.get("controller_mode") != "PAPER_DIRECT_SIGMA" or row.get("parameterization") != "direct_sigma" for row in rows):
        reasons.append("amended direct-sigma controller did not execute")
    if any(row.get("convergence_fit_r_squared", -np.inf) < .8 or
           row.get("gamma_times_100", 0.0) == 0.0 or
           sum(row.get("trajectory", {}).get("fit_mask", [])) < 2 for row in rows):
        reasons.append("one or more local convergence fits failed the preregistered R-squared gate")
    for p in sorted({row["parameters_per_gate"] for row in rows}):
        means = [np.mean([row["gamma_times_100"] for row in rows if row["parameters_per_gate"] == p and row["distance"] == d])
                 for d in sorted({row["distance"] for row in rows if row["parameters_per_gate"] == p})]
        if len(means) > 1 and np.mean(means): cvs.append(float(np.std(means, ddof=1)/abs(np.mean(means))))
    if mode in {"reference", "paper-scale"} and any(value > .15 for value in cvs): reasons.append("distance-independence tolerance exceeded")
    return not reasons, reasons, {"gamma_distance_cv_by_p": cvs, "beam_parameters_per_gate": [1, 10, 30],
        "source_structure_match": all(row.get("source_structure_match") for row in rows),
        "paper_comparable": False,
        "blocking_reasons": ["the proprietary Figure 5 scaling simulator is unavailable"]}
