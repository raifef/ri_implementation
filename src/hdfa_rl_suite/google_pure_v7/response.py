"""Parametric and non-parametric step-response diagnostics."""
from __future__ import annotations

from typing import Any

import numpy as np


def _first_crossing(values: np.ndarray, threshold: float, *, increasing: bool) -> int | None:
    indices = np.flatnonzero(values >= threshold) if increasing else np.flatnonzero(values <= threshold)
    return int(indices[0]) if len(indices) else None


def estimate_step_response(response: np.ndarray, *, onset_epoch: int, target: float,
                           settling_relative_tolerance: float = 0.05,
                           sustained_epochs: int = 25, tau_grid_points: int = 500,
                           tau_max: float | None = None) -> dict[str, Any]:
    values = np.asarray(response, dtype=float)
    if values.ndim != 1 or onset_epoch <= 0 or onset_epoch >= len(values) - 4 or not np.all(np.isfinite(values)):
        raise ValueError("valid scalar response with an interior onset is required")
    pre = float(np.mean(values[max(0, onset_epoch - 10):onset_epoch]))
    post = values[onset_epoch:]
    final_width = max(10, len(post) // 10)
    final = float(np.mean(post[-final_width:]))
    achieved_change = final - pre
    target_change = float(target - pre)
    if abs(target_change) <= 1e-12:
        raise ValueError("target must differ from the pre-step response")
    # Paper-relevant crossings are fractions of the injected target excursion,
    # never fractions of whatever final excursion happened to be observed.
    fractions = (post - pre) / target_change
    achieved_fraction = achieved_change / target_change
    response_times = {
        "response_time_50_epochs": _first_crossing(fractions, 0.5, increasing=True),
        "response_time_63_2_epochs": _first_crossing(fractions, 0.632, increasing=True),
        "response_time_90_epochs": _first_crossing(fractions, 0.9, increasing=True),
    }
    tolerance = settling_relative_tolerance * abs(target_change)
    settled = None
    for index in range(max(0, len(post) - sustained_epochs + 1)):
        if np.all(np.abs(post[index:index+sustained_epochs] - target) <= tolerance):
            settled = index
            break
    overshoot = max(0.0, float(np.max(post - target))) if target_change > 0 else max(0.0, float(np.max(target - post)))
    errors = target - post
    time = np.arange(len(post), dtype=float)
    maximum_tau = float(tau_max or max(10.0, 2.0 * len(post)))
    tau_grid = np.geomspace(1.0, maximum_tau, tau_grid_points)
    fits: list[tuple[float, float, np.ndarray]] = []
    for tau in tau_grid:
        basis = np.column_stack((np.ones(len(time)), np.exp(-time / tau)))
        beta, _, rank, _ = np.linalg.lstsq(basis, errors, rcond=None)
        if rank == 2:
            residual = errors - basis @ beta
            fits.append((float(np.sum(residual**2)), float(tau), beta))
    if not fits:
        fit = {"valid": False, "reason": "rank deficient exponential profile"}
    else:
        best_sse, best_tau, beta = min(fits, key=lambda row: row[0])
        variance = best_sse / max(1, len(time) - 3)
        threshold = best_sse + 3.841458820694124 * variance
        accepted = [tau for sse, tau, _ in fits if sse <= threshold]
        fitted = beta[0] + beta[1] * np.exp(-time / best_tau)
        total = float(np.sum((errors - np.mean(errors))**2))
        credibility_reasons = []
        if best_tau <= 1.02 or best_tau >= .98 * maximum_tau: credibility_reasons.append("tau_bound_saturation")
        fit_r_squared = 1.0 - best_sse / total if total > 0 else 1.0
        if fit_r_squared < .8: credibility_reasons.append("poor_exponential_fit")
        if achieved_fraction < .8: credibility_reasons.append("target_plateau_not_identified")
        fit = {
            "valid": not credibility_reasons,
            "credibility_reasons": credibility_reasons,
            "tau_epochs": best_tau,
            "tau_profile_confidence_interval_95_epochs": [float(min(accepted)), float(max(accepted))],
            "e_infinity": float(beta[0]),
            "e_zero_minus_e_infinity": float(beta[1]),
            "fit_r_squared": fit_r_squared,
            "fit_sse": best_sse,
        }
    return {
        "pre_step_response": pre,
        "final_response": final,
        "target_response": float(target),
        "final_residual": float(target - final),
        "achieved_change": achieved_change,
        "response_fraction_of_injected_target": float(achieved_fraction),
        "crossing_definition": "fraction_of_injected_target_excursion",
        **response_times,
        "settling_time_95_epochs": settled,
        "settling_tolerance_absolute": tolerance,
        "overshoot": overshoot,
        "integrated_absolute_tracking_error": float(np.sum(np.abs(errors))),
        "exponential_fit": fit,
        "response_classification": "SETTLED_TO_TARGET" if settled is not None else (
            "TARGET_90_NOT_REACHED_WITHIN_HORIZON" if response_times["response_time_90_epochs"] is None
            else "TARGET_CROSSED_BUT_NOT_SETTLED"),
    }
