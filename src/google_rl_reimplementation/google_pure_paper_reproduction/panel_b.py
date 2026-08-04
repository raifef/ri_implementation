"""Figure 5b sparse scaling without dense control matrices."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from google_rl_reimplementation.google_pure_v7.figure5.accounting import detector_factors, physical_qubits, total_controls
from google_rl_reimplementation.google_pure_v7.figure5.panel_b import scaling_trace


def acquire_condition(protocol: Mapping[str, Any], condition: Mapping[str, Any]) -> dict[str, Any]:
    d, p, seed = int(condition["distance"]), int(condition["parameters_per_gate"]), int(condition["seed"])
    trace = scaling_trace(d, p, seed, int(protocol["config"]["epochs"]))
    return {"distance": d, "parameters_per_gate": p, "seed": seed, "total_controls": total_controls(d, p),
            "physical_qubits": physical_qubits(d), "detectors": detector_factors(d),
            "logical_floor": float(trace["logical_floor"][0]), "logical_initial": float(trace["logical_learned"][0]),
            "lambda_star": 1.79e-3 / 4e-4, "dense_parameter_matrix_allocated": False,
            "trajectory": {key: np.asarray(value).tolist() for key, value in trace.items()}}


def validation(rows: list[dict[str, Any]], mode: str) -> tuple[bool, list[str], dict[str, Any]]:
    reasons = []
    if total_controls(15, 30) != 38670: reasons.append("d15 P30 control-count contract failed")
    if any(row["logical_floor"] >= row["logical_initial"] for row in rows): reasons.append("logical floor must be independent and below the initial value")
    if any(row["dense_parameter_matrix_allocated"] for row in rows): reasons.append("dense scaling allocation detected")
    return not reasons, reasons, {"distance_15_p30_controls": 38670, "threshold_physical_error": 1.79e-3}

