"""Exponential response fit with nonparametric, censoring, and rejection diagnostics."""
from __future__ import annotations

from typing import Any, Iterable
import numpy as np


def _crossing(y: np.ndarray, onset: int, y0: float, amplitude: float, fraction: float) -> dict[str, Any]:
    threshold = y0 + fraction * amplitude
    after = y[onset:]
    hits = np.flatnonzero(after >= threshold) if amplitude >= 0 else np.flatnonzero(after <= threshold)
    return {"fraction": fraction, "epoch": None if hits.size == 0 else int(onset + hits[0]),
            "latency_epochs": None if hits.size == 0 else int(hits[0]), "censored": hits.size == 0}


def _fit_grid(y: np.ndarray, onset: int, start: int, stop: int) -> dict[str, float]:
    t = np.arange(start, stop, dtype=float) - onset
    yy = y[start:stop]
    taus = np.geomspace(1.0, max(2.0, 5.0 * (stop - onset)), 800)
    best: tuple[float, float, float, float] | None = None
    for tau in taus:
        x = 1.0 - np.exp(-np.maximum(t, 0.0) / tau)
        design = np.column_stack((np.ones_like(x), x))
        beta, *_ = np.linalg.lstsq(design, yy, rcond=None)
        residual = yy - design @ beta
        sse = float(residual @ residual)
        if best is None or sse < best[0]:
            best = (sse, float(beta[0]), float(beta[1]), float(tau))
    assert best is not None
    sse, baseline, amplitude, tau = best
    denom = float(np.sum((yy - np.mean(yy)) ** 2))
    return {"sse": sse, "baseline": baseline, "amplitude": amplitude, "tau_epochs": tau,
            "r_squared": 1.0 - sse / denom if denom > 0 else 0.0}


def estimate_response(trace: Iterable[float], *, onset_epoch: int, bootstrap_seed: int = 0,
                      bootstrap_samples: int = 300) -> dict[str, Any]:
    y = np.asarray(tuple(trace), dtype=float)
    if y.ndim != 1 or y.size < onset_epoch + 20 or onset_epoch < 3 or not np.all(np.isfinite(y)):
        raise ValueError("response trace is invalid or too short")
    fit = _fit_grid(y, onset_epoch, onset_epoch, y.size)
    plateau = float(np.mean(y[max(onset_epoch, int(0.9 * y.size)):]))
    achieved_fraction = abs((plateau - fit["baseline"]) / fit["amplitude"]) if abs(fit["amplitude"]) > 1e-12 else 0.0
    windows = []
    for trim in (0, max(1, (y.size - onset_epoch) // 10), max(2, (y.size - onset_epoch) // 5)):
        windows.append(_fit_grid(y, onset_epoch, onset_epoch + trim, y.size))
    sensitivity = max(abs(row["tau_epochs"] - fit["tau_epochs"]) for row in windows) / max(fit["tau_epochs"], 1e-12)
    rng = np.random.default_rng(bootstrap_seed)
    predicted = fit["baseline"] + fit["amplitude"] * (1 - np.exp(-np.maximum(np.arange(y.size)-onset_epoch, 0)/fit["tau_epochs"]))
    residual = y - predicted
    boot = []
    for _ in range(bootstrap_samples):
        sample = predicted + rng.choice(residual, size=y.size, replace=True)
        boot.append(_fit_grid(sample, onset_epoch, onset_epoch, y.size)["tau_epochs"])
    tau_ci = [float(x) for x in np.quantile(boot, [0.025, 0.975])]
    bound_saturated = fit["tau_epochs"] <= 1.02 or fit["tau_epochs"] >= 4.9 * (y.size - onset_epoch)
    reasons = []
    if bound_saturated: reasons.append("tau_bound_saturation")
    if achieved_fraction < 0.8: reasons.append("no_identifiable_plateau")
    if abs(fit["amplitude"]) < max(1e-8, 3 * float(np.std(y[:onset_epoch]))): reasons.append("amplitude_not_identifiable")
    if sensitivity > 0.25: reasons.append("fit_window_sensitive")
    if fit["r_squared"] < 0.8: reasons.append("poor_exponential_fit")
    crossings = [_crossing(y, onset_epoch, fit["baseline"], fit["amplitude"], f) for f in (0.5, 0.75, 0.9)]
    # The public response coordinate is already normalized by the injected target.
    # These observed crossings must therefore use 0.5/0.75/0.9 of that target, not
    # 90% of whatever excursion happened to be observed in a failed run.
    target_crossings = [_crossing(y, onset_epoch, 0.0, 1.0, f) for f in (0.5, 0.75, 0.9)]
    response_fraction = float(plateau)
    return {"fit_valid": not reasons, "rejection_reasons": reasons, "tau_epochs": fit["tau_epochs"],
            "tau_ci_95": tau_ci, "baseline": fit["baseline"], "amplitude": fit["amplitude"],
            "plateau": plateau, "achieved_response_fraction": achieved_fraction,
            "r_squared": fit["r_squared"], "fit_window_sensitivity": sensitivity,
            "crossings": crossings, "fit_relative_crossings": crossings,
            "target_relative_crossings": target_crossings,
            "response_fraction_of_injected_target": response_fraction,
            "response_time_90_epochs": target_crossings[-1]["latency_epochs"],
            "response_classification": "TARGET_90_NOT_REACHED_WITHIN_HORIZON"
                if target_crossings[-1]["censored"] else "TARGET_90_REACHED",
            "requires_extension": bool(reasons or any(x["censored"] for x in target_crossings))}
