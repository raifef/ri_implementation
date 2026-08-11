"""Target-relative step metrics and source-comparable exponential fitting."""
from __future__ import annotations

from typing import Any

import numpy as np

from .contracts import NONFINAL, V13_SCHEMA
from .io import ARTIFACT_ROOT, atomic_json, atomic_text, config, read_json


THRESHOLDS = (0.5, 0.632, 0.9, 0.95)


def _crossing(values: np.ndarray, threshold: float) -> int | None:
    indices = np.flatnonzero(values >= threshold)
    return int(indices[0]) if indices.size else None


def _fit_curve(values: np.ndarray) -> dict[str, Any]:
    time = np.arange(len(values), dtype=float)
    best: tuple[float, float, float, np.ndarray] | None = None
    # Log-spaced grid is deterministic, robust to censored late responses, and does
    # not need an optimizer whose version would become another hidden dependency.
    for tau in np.geomspace(.25, max(1.0, 20.0 * len(values)), 3000):
        basis = 1.0 - np.exp(-time / tau)
        denominator = float(np.dot(basis, basis))
        if denominator <= 0:
            continue
        asymptote = float(np.dot(basis, values) / denominator)
        residual = values - asymptote * basis
        sse = float(np.dot(residual, residual))
        if best is None or sse < best[0]:
            best = (sse, float(tau), asymptote, residual)
    if best is None:
        raise RuntimeError("step fit has no admissible timescale")
    sse, tau, asymptote, residual = best
    total = float(np.sum(np.square(values - np.mean(values))))
    r_squared = 1.0 - sse / total if total > 0 else 1.0
    return {"tau_epochs": tau, "asymptotic_target_fraction": asymptote,
            "r_squared": r_squared, "residuals": residual.tolist(),
            "residual_rms": float(np.sqrt(np.mean(np.square(residual))))}


def fit_step_response() -> dict[str, Any]:
    path = ARTIFACT_ROOT / "step_validation/runs.json"
    if not path.is_file():
        comparison = read_json(ARTIFACT_ROOT / "sensitivity_calibration/comparison.json")
        rows = [row for row in comparison["rows"]
                if row.get("normalization_branch") == "C_V13_SOURCE_LITERAL_BOUNDARY"
                and row.get("family") == "STEP_RESPONSE_INJECTED_DRIFT"]
    else:
        rows = read_json(path)["rows"]
    if not rows:
        raise RuntimeError("no V13 step-validation traces exist")
    onset = int(rows[0]["onset_epoch"])
    traces = np.asarray([[point["target_relative_progress"] for point in row["trace"]][onset:]
                         for row in rows], dtype=float)
    if traces.ndim != 2 or traces.shape[1] < 3:
        raise RuntimeError("step response needs at least three post-onset epochs")
    median_trace = np.median(traces, axis=0)
    fit = _fit_curve(median_trace)
    crossing_rows = []
    for row, trace in zip(rows, traces):
        crossing_rows.append({"seed": row["seed"], **{
            f"t{str(level * 100).replace('.', '_')}_epochs": _crossing(trace, level)
            for level in THRESHOLDS}})
    empirical = {}
    for level in THRESHOLDS:
        key = f"t{str(level * 100).replace('.', '_')}_epochs"
        identified = [row[key] for row in crossing_rows if row[key] is not None]
        empirical[key] = float(np.median(identified)) if identified else None
        empirical[key + "_identified_runs"] = len(identified)
    repetitions = int(config()["step_fit"]["bootstrap_repetitions"])
    rng = np.random.default_rng(int(config()["step_fit"]["bootstrap_seed"]))
    bootstrap = []
    for _ in range(repetitions):
        sampled = traces[rng.integers(0, len(traces), len(traces))]
        bootstrap.append(_fit_curve(np.median(sampled, axis=0)))
    tau_values = np.asarray([row["tau_epochs"] for row in bootstrap])
    asymptotes = np.asarray([row["asymptotic_target_fraction"] for row in bootstrap])
    fit["tau_interval_95"] = np.quantile(tau_values, [.025, .975]).tolist()
    fit["asymptote_interval_95"] = np.quantile(asymptotes, [.025, .975]).tolist()
    predicted = {}
    for level in THRESHOLDS:
        if fit["asymptotic_target_fraction"] > level:
            predicted[f"t{str(level * 100).replace('.', '_')}_epochs"] = float(
                -fit["tau_epochs"] * np.log(1.0 - level / fit["asymptotic_target_fraction"]))
        else:
            predicted[f"t{str(level * 100).replace('.', '_')}_epochs"] = None
    t632 = empirical["t63_2_epochs"]
    consistency = (t632 is not None and abs(t632 - fit["tau_epochs"]) /
                   max(fit["tau_epochs"], 1e-30) <=
                   float(config()["step_fit"]["timescale_consistency_relative_tolerance"]))
    gates = {
        "fit_r_squared": fit["r_squared"] >= float(config()["step_fit"]["minimum_r_squared"]),
        "timescale_internal_consistency": bool(consistency),
        "t90_is_target_relative": all(row["t90_0_epochs"] is None or
                                      traces[index, row["t90_0_epochs"]] >= .9
                                      for index, row in enumerate(crossing_rows)),
        "all_runs_use_same_onset": all(int(row["onset_epoch"]) == onset for row in rows),
    }
    result = {"schema_version": V13_SCHEMA, "run_count": len(rows), "onset_epoch": onset,
              "threshold_definition": "FIRST_POST_ONSET_EPOCH_WITH_TARGET_RELATIVE_PROGRESS_AT_LEAST_THRESHOLD",
              "crossings": crossing_rows, "empirical_median_crossings": empirical,
              "exponential_model": "R(t)=R_inf*(1-exp(-t/tau))", "fit": fit,
              "predicted_crossings": predicted, "gates": gates, "fit_valid": all(gates.values()),
              "paper_comparison": {"quantity_compared": "tau_epochs",
                                   "paper_characteristic_time_epochs":
                                       int(config()["step_fit"]["paper_characteristic_time_epochs"]),
                                   "ratio_v13_to_paper_anchor": fit["tau_epochs"] /
                                       int(config()["step_fit"]["paper_characteristic_time_epochs"]),
                                   "equivalence_claim": False},
              **NONFINAL}
    atomic_json(ARTIFACT_ROOT / "step_validation/fit.json", result)
    atomic_text(ARTIFACT_ROOT / "step_validation/fit.md", "\n".join([
        "# V13 step-response fit", "",
        f"Exponential tau: **{fit['tau_epochs']:.3f} epochs**",
        f"Asymptotic target fraction: **{fit['asymptotic_target_fraction']:.4f}**",
        f"Fit valid: **{result['fit_valid']}**", "",
        "All crossings are fractions of the injected target, never fractions of the observed final excursion.",
        "No paper-equivalence claim is permitted.",
    ]))
    return result


def validate_step_fit() -> dict[str, Any]:
    result = read_json(ARTIFACT_ROOT / "step_validation/fit.json")
    failures = [name for name, passed in result["gates"].items() if not passed]
    value = {"schema_version": V13_SCHEMA, "pass": not failures,
             "failures": failures, "fit_hash_fields_present": all(
                 key in result["fit"] for key in ("tau_epochs", "tau_interval_95",
                                                   "asymptotic_target_fraction", "residuals")),
             **NONFINAL}
    atomic_json(ARTIFACT_ROOT / "step_validation/fit_validation.json", value)
    return value

