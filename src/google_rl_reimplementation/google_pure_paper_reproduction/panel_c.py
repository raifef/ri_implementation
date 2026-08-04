"""Figure 5c source-axis convergence-law trajectories and fits."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from google_rl_reimplementation.google_pure_v7.figure5.accounting import total_controls
from google_rl_reimplementation.google_pure_v7.figure5.panel_b import scaling_trace


def acquire_condition(protocol: Mapping[str, Any], condition: Mapping[str, Any]) -> dict[str, Any]:
    d, p, seed = int(condition["distance"]), int(condition["parameters_per_gate"]), int(condition["seed"])
    base = scaling_trace(d, p, seed, int(protocol["config"]["epochs"])); ratio = np.asarray(base["lambda_ratio"])
    x, y = 1-ratio[:-1], 100*np.diff(ratio)
    keep = (x > float(protocol["config"].get("local_fit_min_distance", 1e-4))) & (x < float(protocol["config"].get("local_fit_max_distance", .7)))
    slope = float(np.dot(x[keep], y[keep]) / np.dot(x[keep], x[keep])) if np.any(keep) else float("nan")
    return {"distance": d, "parameters_per_gate": p, "seed": seed, "total_controls": total_controls(d, p),
            "gamma_times_100": slope, "source_x_axis": "1-Lambda/Lambda*", "source_y_axis": "1e2 d_t Lambda/Lambda*",
            "trajectory": {"x_distance": x.tolist(), "normalized_speed": y.tolist(), "fit_mask": keep.astype(int).tolist(),
                           "lambda_ratio": ratio.tolist(), "epoch": np.asarray(base["epoch"]).tolist()}}


def validation(rows: list[dict[str, Any]], mode: str) -> tuple[bool, list[str], dict[str, Any]]:
    reasons, cvs = [], []
    if any(not np.isfinite(row["gamma_times_100"]) for row in rows): reasons.append("non-finite local convergence fit")
    for p in sorted({row["parameters_per_gate"] for row in rows}):
        means = [np.mean([row["gamma_times_100"] for row in rows if row["parameters_per_gate"] == p and row["distance"] == d])
                 for d in sorted({row["distance"] for row in rows if row["parameters_per_gate"] == p})]
        if len(means) > 1 and np.mean(means): cvs.append(float(np.std(means, ddof=1)/abs(np.mean(means))))
    if mode in {"reference", "paper-scale"} and any(value > .15 for value in cvs): reasons.append("distance-independence tolerance exceeded")
    return not reasons, reasons, {"gamma_distance_cv_by_p": cvs, "beam_parameters_per_gate": [1, 10, 30]}

