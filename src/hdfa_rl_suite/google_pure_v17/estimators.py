"""Pure estimators used by V17 diagnostics and permanent fixtures."""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def sinusoid(frequency: float, epochs: np.ndarray, phase: float = 0.0) -> np.ndarray:
    """Source-defined trajectory: frequency is cycles per epoch, not radians per epoch."""
    return np.sin(2.0 * np.pi * float(frequency) * np.asarray(epochs, dtype=float) + float(phase))


def measured_period_from_zero_crossings(values: np.ndarray, epochs: np.ndarray) -> dict[str, Any]:
    """Estimate period from like-oriented, linearly interpolated zero crossings."""
    y = np.asarray(values, dtype=float)
    t = np.asarray(epochs, dtype=float)
    if y.ndim != 1 or t.shape != y.shape or y.size < 3:
        raise ValueError("period estimator requires aligned one-dimensional arrays")
    crossings: list[dict[str, float | str]] = []
    for index in range(y.size - 1):
        left, right = float(y[index]), float(y[index + 1])
        if left == 0.0:
            crossing = float(t[index])
        elif left * right < 0.0:
            crossing = float(t[index] - left * (t[index + 1] - t[index]) / (right - left))
        else:
            continue
        slope = right - left
        orientation = "positive" if slope > 0 else "negative"
        if not crossings or abs(crossing - float(crossings[-1]["epoch"])) > 1e-9:
            crossings.append({"epoch": crossing, "orientation": orientation})
    periods = []
    for orientation in ("positive", "negative"):
        selected = [float(row["epoch"]) for row in crossings if row["orientation"] == orientation]
        periods.extend(np.diff(selected).tolist())
    return {"zero_crossings": crossings,
            "measured_period_epochs": float(np.median(periods)) if periods else None,
            "period_samples": periods}


def complete_period_window(frequency: float, *, burn_in_periods: int,
                           evaluation_periods: int, phase: float = 0.0) -> dict[str, Any]:
    period = 1.0 / float(frequency)
    rounded = int(round(period))
    if not np.isclose(period, rounded, rtol=0, atol=1e-10):
        raise ValueError("an exact integer-epoch complete-period window is unavailable")
    start = int(burn_in_periods) * rounded
    stop = start + int(evaluation_periods) * rounded
    return {
        "frequency_per_epoch": float(frequency), "period_epochs": float(period),
        "burn_in_epochs": start, "evaluation_start_epoch": start,
        "evaluation_stop_epoch_exclusive": stop, "evaluation_epochs": stop - start,
        "complete_post_burnin_periods": int(evaluation_periods),
        "fractional_periods": 0.0, "phase_radians": float(phase),
    }


def estimate_sinusoidal_transfer(epochs: np.ndarray, output: np.ndarray, frequency: float,
                                 *, minimum_cycles: float = 1.0,
                                 maximum_condition_number: float = 1e5) -> dict[str, Any]:
    t = np.asarray(epochs, dtype=float)
    y = np.asarray(output, dtype=float)
    if t.ndim != 1 or y.shape != t.shape or t.size < 4:
        raise ValueError("transfer fit requires aligned one-dimensional observations")
    omega = 2.0 * np.pi * float(frequency)
    design = np.column_stack([np.sin(omega * t), np.cos(omega * t), np.ones(t.size)])
    condition = float(np.linalg.cond(design))
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ beta
    residual = y - fitted
    dof = max(1, t.size - design.shape[1])
    variance = float(residual @ residual / dof)
    covariance = variance * np.linalg.pinv(design.T @ design)
    a, b, offset = map(float, beta)
    gain = float(np.hypot(a, b))
    phase = float(math.atan2(b, a))
    if gain > 0:
        gradient_gain = np.asarray([a / gain, b / gain, 0.0])
        gradient_phase = np.asarray([-b / gain**2, a / gain**2, 0.0])
        gain_se = float(np.sqrt(max(0.0, gradient_gain @ covariance @ gradient_gain)))
        phase_se = float(np.sqrt(max(0.0, gradient_phase @ covariance @ gradient_phase)))
    else:
        gain_se = phase_se = float("inf")
    observed_cycles = float((t[-1] - t[0] + 1.0) * frequency)
    identifiable = bool(observed_cycles >= minimum_cycles and
                        condition <= maximum_condition_number and np.isfinite(gain_se))
    return {
        "sine_coefficient": a, "cosine_coefficient": b, "offset": offset,
        "gain": gain, "phase_radians": phase, "phase_lag_radians": -phase,
        "gain_standard_error": gain_se, "phase_standard_error": phase_se,
        "gain_confidence_interval_95": [max(0.0, gain - 1.96 * gain_se), gain + 1.96 * gain_se],
        "phase_confidence_interval_95": [phase - 1.96 * phase_se, phase + 1.96 * phase_se],
        "residual_rms": float(np.sqrt(np.mean(residual**2))),
        "residual_lag1_autocorrelation": (float(np.corrcoef(residual[:-1], residual[1:])[0, 1])
                                          if residual.size > 2 and np.std(residual) > 0 else None),
        "design_condition_number": condition, "observed_cycles": observed_cycles,
        "identifiable": identifiable,
    }


def estimate_pure_delay(input_values: np.ndarray, output_values: np.ndarray,
                        *, maximum_delay: int) -> dict[str, Any]:
    x = np.asarray(input_values, dtype=float)
    y = np.asarray(output_values, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("delay estimator inputs must be aligned vectors")
    rows = []
    for delay in range(int(maximum_delay) + 1):
        left = x[:len(x) - delay] if delay else x
        right = y[delay:] if delay else y
        correlation = float(np.corrcoef(left, right)[0, 1]) if np.std(left) and np.std(right) else 0.0
        rows.append({"delay_epochs": delay, "correlation": correlation})
    best = max(rows, key=lambda row: row["correlation"])
    return {"estimated_delay_epochs": best["delay_epochs"], "maximum_correlation": best["correlation"],
            "profile": rows}


def _step_model(t: np.ndarray, gain: float, delay: float, tau: float) -> np.ndarray:
    active = np.maximum(np.asarray(t, dtype=float) - float(delay), 0.0)
    return float(gain) * (1.0 - np.exp(-active / float(tau)))


def fit_step_transfer(epochs: np.ndarray, response: np.ndarray, *, fixed_gain: float | None = None,
                      fixed_delay: float | None = None) -> dict[str, Any]:
    """Profile-grid fit of K[1-exp(-(t-Delta)+/tau)] with no SciPy dependency."""
    t = np.asarray(epochs, dtype=float)
    y = np.asarray(response, dtype=float)
    delays = np.asarray([fixed_delay], dtype=float) if fixed_delay is not None else np.linspace(0.0, 40.0, 41)
    taus = np.geomspace(5.0, 2000.0, 300)
    best: tuple[float, float, float, float] | None = None
    profile_delay = []
    for delay in delays:
        local_best = None
        active = np.maximum(t - delay, 0.0)
        for tau in taus:
            basis = 1.0 - np.exp(-active / tau)
            if fixed_gain is None:
                denominator = float(basis @ basis)
                gain = float(basis @ y / denominator) if denominator > 0 else 0.0
                gain = float(np.clip(gain, 0.0, 3.0))
            else:
                gain = float(fixed_gain)
            residual = y - gain * basis
            sse = float(residual @ residual)
            candidate = (sse, gain, float(delay), float(tau))
            if local_best is None or candidate[0] < local_best[0]:
                local_best = candidate
            if best is None or candidate[0] < best[0]:
                best = candidate
        assert local_best is not None
        profile_delay.append({"delay_epochs": float(delay), "minimum_sse": local_best[0]})
    assert best is not None
    sse, gain, delay, tau = best
    fitted = _step_model(t, gain, delay, tau)
    residual = y - fitted
    active = np.maximum(t - delay, 0.0)
    switched = (t > delay).astype(float)
    exponential = np.exp(-active / tau)
    jacobian = np.column_stack([
        1.0 - exponential,
        -gain * exponential * switched / tau,
        -gain * exponential * active / tau**2,
    ])
    variance = sse / max(1, len(t) - 3)
    covariance = variance * np.linalg.pinv(jacobian.T @ jacobian)
    standard_errors = np.sqrt(np.maximum(0.0, np.diag(covariance)))
    threshold = sse + 3.841458820694124 * variance
    accepted_delays = [row["delay_epochs"] for row in profile_delay if row["minimum_sse"] <= threshold]
    return {
        "gain": gain, "delay_epochs": delay, "tau_epochs": tau, "sse": sse,
        "rms_residual": float(np.sqrt(np.mean(residual**2))),
        "residual_lag1_autocorrelation": (float(np.corrcoef(residual[:-1], residual[1:])[0, 1])
                                          if residual.size > 2 and np.std(residual) > 0 else None),
        "covariance": covariance.tolist(), "standard_errors": standard_errors.tolist(),
        "confidence_intervals_95": {
            "gain": [gain - 1.96 * standard_errors[0], gain + 1.96 * standard_errors[0]],
            "delay_epochs": [delay - 1.96 * standard_errors[1], delay + 1.96 * standard_errors[1]],
            "tau_epochs": [tau - 1.96 * standard_errors[2], tau + 1.96 * standard_errors[2]],
        },
        "profile_delay": profile_delay,
        "profile_likelihood_95": {
            "delay_epochs": ([min(accepted_delays), max(accepted_delays)] if accepted_delays else None),
            "criterion": "SSE <= SSE_min + chi2_0.95(df=1)*residual_variance",
        },
        "fitted": fitted.tolist(), "residuals": residual.tolist(),
        "fixed_gain": fixed_gain, "fixed_delay": fixed_delay,
    }


def fit_two_timescale_step(epochs: np.ndarray, response: np.ndarray) -> dict[str, Any]:
    """Diagnostic nonnegative two-pole step fit with one shared integer delay."""
    t = np.asarray(epochs, dtype=float)
    y = np.asarray(response, dtype=float)
    taus = np.geomspace(8.0, 800.0, 32)
    best: tuple[float, float, float, float, float, float] | None = None
    for delay in range(0, 11):
        active = np.maximum(t - delay, 0.0)
        basis = np.asarray([1.0 - np.exp(-active / tau) for tau in taus])
        for left in range(len(taus) - 1):
            for right in range(left + 1, len(taus)):
                design = np.column_stack([basis[left], basis[right]])
                weights = np.maximum(0.0, np.linalg.lstsq(design, y, rcond=None)[0])
                fitted = design @ weights
                sse = float(np.sum((y - fitted) ** 2))
                candidate = (sse, float(weights[0]), float(weights[1]),
                             float(taus[left]), float(taus[right]))
                if best is None or candidate[0] < best[0]:
                    best = candidate + (float(delay),)
    assert best is not None
    sse, fast_weight, slow_weight, fast_tau, slow_tau, delay = best
    gain = fast_weight + slow_weight
    fitted = (fast_weight * (1.0 - np.exp(-np.maximum(t - delay, 0.0) / fast_tau)) +
              slow_weight * (1.0 - np.exp(-np.maximum(t - delay, 0.0) / slow_tau)))
    return {"gain": float(gain), "delay_epochs": float(delay),
            "fast_tau_epochs": float(fast_tau), "slow_tau_epochs": float(slow_tau),
            "fast_weight_fraction": float(fast_weight / gain) if gain else None,
            "slow_weight_fraction": float(slow_weight / gain) if gain else None,
            "sse": float(sse), "rms_residual": float(np.sqrt(np.mean((y - fitted) ** 2))),
            "aic": float(len(y) * np.log(max(sse / len(y), np.finfo(float).tiny)) + 2 * 5),
            "fitted": fitted.tolist(),
            "identifiable": bool(slow_tau < float(taus[-1]) and gain < 2.5),
            "boundary_limited": bool(slow_tau == float(taus[-1]) or gain >= 2.5)}


def analytic_first_order_transfer(gain: float, delay: float, tau: float,
                                  frequency: float) -> dict[str, float]:
    omega = 2.0 * np.pi * float(frequency)
    magnitude = float(gain / np.sqrt(1.0 + (omega * tau) ** 2))
    phase = float(-omega * delay - np.arctan(omega * tau))
    return {"gain": magnitude, "phase_radians": phase, "phase_lag_radians": -phase,
            "angular_frequency_radians_per_epoch": omega}


def paired_acceptance_v2(units: list[dict[str, Any]], *, delta_min: float,
                         confidence: float = .95, bootstrap_draws: int = 10000,
                         seed: int = 82917) -> dict[str, Any]:
    """Validate complete matched units and form a paired bootstrap lower bound."""
    required = {"condition", "seed", "phase_radians", "budget_hash", "crn_hash",
                "cycles_per_candidate", "burn_in_epochs", "evaluation_periods",
                "complete_periods", "I_stochastic"}
    errors = []
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    invalid_keys: set[tuple[Any, ...]] = set()
    for index, unit in enumerate(units):
        missing = sorted(required - set(unit))
        if missing:
            errors.append(f"unit:{index}:missing:{','.join(missing)}")
            continue
        if unit["condition"] not in {"slow", "fast"}:
            errors.append(f"unit:{index}:invalid-condition")
            continue
        key = (unit["seed"], unit["phase_radians"], unit["budget_hash"], unit["crn_hash"],
               unit["cycles_per_candidate"], unit["burn_in_epochs"], unit["evaluation_periods"])
        if int(unit["evaluation_periods"]) <= 0:
            errors.append(f"unit:{index}:nonpositive-evaluation-periods")
            invalid_keys.add(key)
        if int(unit["complete_periods"]) != int(unit["evaluation_periods"]):
            errors.append(f"unit:{index}:incomplete-period-window")
            invalid_keys.add(key)
        grouped.setdefault(key, {})[unit["condition"]] = unit
    deltas = []
    for key, pair in grouped.items():
        if set(pair) != {"slow", "fast"}:
            errors.append(f"pair:{key}:missing-condition")
            continue
        if key in invalid_keys:
            continue
        deltas.append(float(pair["slow"]["I_stochastic"] - pair["fast"]["I_stochastic"]))
    if len(deltas) < 2:
        errors.append("at-least-two-complete-pairs-required")
    lower = None
    if not errors:
        values = np.asarray(deltas, dtype=float)
        rng = np.random.default_rng(int(seed))
        indices = rng.integers(0, len(values), size=(int(bootstrap_draws), len(values)))
        bootstrap_means = np.mean(values[indices], axis=1)
        lower = float(np.quantile(bootstrap_means, 1.0 - float(confidence)))
    return {
        "valid": not errors, "validation_errors": errors,
        "paired_complete_unit_count": len(deltas), "paired_deltas": deltas,
        "mean_delta_I": float(np.mean(deltas)) if deltas else None,
        "lower_confidence_bound": lower, "confidence": float(confidence),
        "bootstrap_draws": int(bootstrap_draws) if not errors else 0,
        "delta_min": float(delta_min),
        "pass": lower is not None and lower > float(delta_min),
    }
