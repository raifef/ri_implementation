"""Optimum-normalized sine tracking estimator with fit uncertainty."""
from __future__ import annotations

from typing import Any

import numpy as np

from .reporting import write_report


class InvalidSineDiagnostic(ValueError):
    """Raised when gain or phase is not scientifically identifiable."""


def classify_bandwidth_cutoff(omega_tau: list[float], gains: list[float], *, threshold: float = 1.0/np.sqrt(2.0)) -> dict[str, Any]:
    if not omega_tau or len(omega_tau) != len(gains) or any(value is None or not np.isfinite(value) for value in gains):
        raise InvalidSineDiagnostic("null or non-finite bandwidth gain is invalid")
    if any(right <= left for left, right in zip(omega_tau, omega_tau[1:])):
        raise InvalidSineDiagnostic("omega-tau grid must be strictly increasing")
    crossing = next((index for index, gain in enumerate(gains) if gain <= threshold), None)
    if crossing == 0:
        return {"classification":"BELOW_SWEEP","omega_tau_upper_bound":omega_tau[0],"value":omega_tau[0]}
    if crossing is None:
        return {"classification":"ABOVE_SWEEP","omega_tau_lower_bound":omega_tau[-1],"value":omega_tau[-1]}
    x0,x1=omega_tau[crossing-1],omega_tau[crossing]; y0,y1=gains[crossing-1],gains[crossing]
    if np.isclose(y1,y0): raise InvalidSineDiagnostic("bandwidth interpolation is degenerate")
    return {"classification":"INTERPOLATED","value":float(x0+(threshold-y0)*(x1-x0)/(y1-y0))}


def wrap_phase(phase: float) -> float:
    wrapped = (float(phase) + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.pi if np.isclose(wrapped, -np.pi) else wrapped)


def fit_sine_tracking(
    time_epochs: np.ndarray,
    learned_mean: np.ndarray,
    *,
    optimum_amplitude: float,
    omega_radians_per_epoch: float,
    burn_in_epochs: int,
    minimum_complete_periods: int = 3,
    documented_drift_tape: bool = True,
) -> dict[str, Any]:
    """Fit c+a*sin(wt)+b*cos(wt); gain denominator is always the moving optimum."""
    time = np.asarray(time_epochs, dtype=float)
    values = np.asarray(learned_mean, dtype=float)
    if time.ndim != 1 or values.shape != time.shape or len(time) < 4 or not np.all(np.isfinite(values)):
        raise InvalidSineDiagnostic("finite aligned scalar time series required")
    if optimum_amplitude <= 0 or not np.isfinite(optimum_amplitude):
        raise InvalidSineDiagnostic("moving-optimum amplitude must be positive and finite")
    if omega_radians_per_epoch <= 0 or not np.isfinite(omega_radians_per_epoch):
        raise InvalidSineDiagnostic("known sine frequency must be positive and finite")
    if not documented_drift_tape:
        raise InvalidSineDiagnostic("analysis requires a documented drift tape")
    selected = time >= float(burn_in_epochs)
    post_time, post_values = time[selected], values[selected]
    if len(post_time) < 4:
        raise InvalidSineDiagnostic("burn-in leaves insufficient samples")
    sample_spacing = float(np.median(np.diff(post_time)))
    if sample_spacing <= 0 or not np.allclose(np.diff(post_time), sample_spacing, rtol=1e-8, atol=1e-10):
        raise InvalidSineDiagnostic("analysis samples must be uniformly spaced")
    period = 2.0 * np.pi / omega_radians_per_epoch
    available_duration = len(post_time) * sample_spacing
    complete_periods = int(np.floor(available_duration / period + 1e-12))
    if complete_periods < minimum_complete_periods:
        raise InvalidSineDiagnostic("fewer than three complete post-burn-in periods")
    analysis_duration = complete_periods * period
    analysis_count = int(np.floor(analysis_duration / sample_spacing + 1e-12))
    if analysis_count < 4:
        raise InvalidSineDiagnostic("complete-period analysis window is too short")
    t = post_time[:analysis_count]
    y = post_values[:analysis_count]
    design = np.column_stack((np.ones(len(t)), np.sin(omega_radians_per_epoch * t),
                              np.cos(omega_radians_per_epoch * t)))
    beta, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank != 3:
        raise InvalidSineDiagnostic("sine fit is rank deficient")
    residual = y - design @ beta
    degrees_of_freedom = len(y) - 3
    if degrees_of_freedom <= 0:
        raise InvalidSineDiagnostic("fit uncertainty cannot be estimated")
    gram_inverse = np.linalg.inv(design.T @ design)
    residual_variance = float(np.sum(residual**2) / degrees_of_freedom)
    covariance = residual_variance * gram_inverse
    if not np.all(np.isfinite(covariance)):
        raise InvalidSineDiagnostic("fit covariance is non-finite")
    offset, sine_coefficient, cosine_coefficient = map(float, beta)
    learned_amplitude = float(np.hypot(sine_coefficient, cosine_coefficient))
    gain = learned_amplitude / float(optimum_amplitude)
    phase = wrap_phase(np.arctan2(cosine_coefficient, sine_coefficient))
    lag = -phase / omega_radians_per_epoch
    residual_amplitude = float(abs(optimum_amplitude - learned_amplitude * np.exp(-1j * phase)))
    if not all(np.isfinite(value) for value in (learned_amplitude, gain, phase, lag, residual_amplitude)):
        raise InvalidSineDiagnostic("gain or phase diagnostic is non-finite")
    amplitude_gradient = np.asarray([sine_coefficient, cosine_coefficient]) / max(learned_amplitude, 1e-15)
    phase_gradient = np.asarray([-cosine_coefficient, sine_coefficient]) / max(learned_amplitude**2, 1e-30)
    coefficient_covariance = covariance[1:3, 1:3]
    amplitude_variance = max(0.0, float(amplitude_gradient @ coefficient_covariance @ amplitude_gradient))
    phase_variance = max(0.0, float(phase_gradient @ coefficient_covariance @ phase_gradient))
    amplitude_se = float(np.sqrt(amplitude_variance))
    phase_se = float(np.sqrt(phase_variance))
    z = 1.959963984540054
    total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - float(np.sum(residual**2)) / total if total > 0 else 1.0
    return {
        "status": "VALID_DIAGNOSTIC",
        "denominator_source": "moving_optimum_amplitude_not_fixed_policy",
        "optimum_amplitude": float(optimum_amplitude),
        "learned_mean_amplitude": learned_amplitude,
        "amplitude_gain": gain,
        "phase_convention": "mu=c+a*sin(omega*t)+b*cos(omega*t); phase=atan2(b,a) in (-pi,pi]",
        "phase_radians": phase,
        "phase_lag_epochs": lag,
        "residual_amplitude": residual_amplitude,
        "fit_offset": offset,
        "fit_sine_coefficient": sine_coefficient,
        "fit_cosine_coefficient": cosine_coefficient,
        "fit_r_squared": r_squared,
        "fit_covariance": covariance.tolist(),
        "confidence_intervals_95": {
            "learned_mean_amplitude": [max(0.0, learned_amplitude - z * amplitude_se), learned_amplitude + z * amplitude_se],
            "amplitude_gain": [max(0.0, gain - z * amplitude_se / optimum_amplitude), gain + z * amplitude_se / optimum_amplitude],
            "phase_radians_unwrapped_local": [phase - z * phase_se, phase + z * phase_se],
        },
        "number_of_complete_periods": complete_periods,
        "period_epochs": period,
        "burn_in_epochs": int(burn_in_epochs),
        "analysis_window_epochs": float(analysis_count * sample_spacing),
        "analysis_samples": int(analysis_count),
        "discarded_post_burn_in_samples": int(len(post_time) - analysis_count),
        "drift_tape_documented": True,
    }


def validate_sine_estimator(seed: int = 7101) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    period = 40.0
    omega = 2.0 * np.pi / period
    optimum_amplitude = 0.2
    known_gain = 0.65
    known_phase = -0.37
    time = np.arange(280, dtype=float)
    values = 0.03 + optimum_amplitude * known_gain * np.sin(omega * time + known_phase)
    values += rng.normal(scale=0.001, size=len(time))
    fit = fit_sine_tracking(time, values, optimum_amplitude=optimum_amplitude,
                            omega_radians_per_epoch=omega, burn_in_epochs=40)
    gain_error = abs(fit["amplitude_gain"] - known_gain)
    phase_error = abs(wrap_phase(fit["phase_radians"] - known_phase))
    zero_rejected = False
    try:
        fit_sine_tracking(time, values, optimum_amplitude=0.0, omega_radians_per_epoch=omega, burn_in_epochs=40)
    except InvalidSineDiagnostic:
        zero_rejected = True
    mechanism = gain_error < 0.02 and phase_error < 0.03 and zero_rejected
    payload = {
        "schema_version": "google-pure-v7-sine-estimator-validation.v1",
        "synthetic_truth": {"gain": known_gain, "phase_radians": known_phase},
        "estimate": fit,
        "gain_absolute_error": gain_error,
        "phase_absolute_error": phase_error,
        "zero_optimum_amplitude_rejected": zero_rejected,
        "fixed_policy_amplitude_used_as_denominator": False,
        "artifact_complete": True,
        "mechanism_valid": mechanism,
        "performance_pass": mechanism,
        "blocking_reasons": [] if mechanism else ["synthetic gain/phase validation tolerance failed"],
        "certification_seeds_consumed": False,
        "status": "PASS" if mechanism else "INVALID_DIAGNOSTIC",
    }
    return write_report("sine_estimator_validation", payload, "Sine Estimator Validation")
