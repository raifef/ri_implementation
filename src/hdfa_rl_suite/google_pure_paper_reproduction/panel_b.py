"""Figure 5b sparse scaling through the amended direct-sigma backend."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from hdfa_rl_suite.google_pure_v7.figure5.accounting import total_controls
from hdfa_rl_suite.google_pure_source_exact.paper_families.scaling import acquire_scaling_condition


def acquire_condition(protocol: Mapping[str, Any], condition: Mapping[str, Any]) -> dict[str, Any]:
    return acquire_scaling_condition(protocol, condition)


def validation(rows: list[dict[str, Any]], mode: str) -> tuple[bool, list[str], dict[str, Any]]:
    reasons = []
    if total_controls(15, 30) != 38670: reasons.append("d15 P30 control-count contract failed")
    if any(row["logical_floor"] >= row["logical_initial"] for row in rows): reasons.append("logical floor must be independent and below the initial value")
    if any(row["dense_parameter_matrix_allocated"] for row in rows): reasons.append("dense scaling allocation detected")
    if any(row.get("controller_mode") != "PAPER_DIRECT_SIGMA" or row.get("parameterization") != "direct_sigma" for row in rows):
        reasons.append("amended direct-sigma controller did not execute")
    if any(row.get("controller_target_access") for row in rows): reasons.append("controller received hidden target")
    for row in rows:
        trajectory = row.get("trajectory", {})
        logical = np.asarray(trajectory.get("logical_learned", []), dtype=float)
        if logical.size < 2:
            reasons.append("missing epoch-resolved logical-error trajectory")
            continue
        excess = max(float(logical[0] - row["logical_floor"]), np.finfo(float).tiny)
        if float((logical[0] - logical[-1]) / excess) < .05:
            reasons.append("floor-normalized logical-error progress did not reach the preregistered 5% visibility gate")
    if mode in {"reference", "paper-scale"} and any(not row.get("paper_physical_error_axis_present") or
            not row.get("paper_logical_error_axis_present") or not row.get("irreducible_floor_bars_present") or
            not row.get("paper_log_axes") or not row.get("epoch_colour_present") for row in rows):
        reasons.append("paper physical-error/logical-error panel geometry is incomplete")
    return not reasons, reasons, {"distance_15_p30_controls": 38670, "threshold_physical_error": 1.79e-3,
        "dense_parameter_matrix_allocated": False,
        "panel_label": "SOURCE_STRUCTURED_SYNTHETIC_SCALING_ANALOGUE", "paper_panel_geometry_complete": not reasons,
        "source_structure_match": not reasons, "paper_comparable": False,
        "blocking_reasons": ["the proprietary Figure 5 scaling simulator is unavailable"]}
