"""Small, dependency-light time-series and spectral estimators."""

from __future__ import annotations

import numpy as np


def autocorrelation(values: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 3 or not 0 <= max_lag < len(x):
        raise ValueError("invalid time series or max_lag")
    x = x - np.mean(x)
    denominator = float(np.dot(x, x))
    if denominator <= 0:
        return np.r_[1.0, np.zeros(max_lag)]
    return np.array([float(np.dot(x[: len(x) - lag], x[lag:]) / denominator) for lag in range(max_lag + 1)])


def periodogram(values: np.ndarray, *, sample_spacing: float = 1.0, detrend: bool = True) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 4 or sample_spacing <= 0:
        raise ValueError("invalid periodogram input")
    if detrend:
        grid = np.arange(len(x), dtype=float)
        design = np.column_stack([np.ones(len(x)), grid])
        x = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    transform = np.fft.rfft(x)
    power = sample_spacing * np.abs(transform) ** 2 / len(x)
    if len(power) > 2:
        power[1:-1] *= 2.0
    return np.fft.rfftfreq(len(x), d=sample_spacing), power


def power_ratio_db(numerator_power: float | np.ndarray, denominator_power: float | np.ndarray) -> float | np.ndarray:
    """Convert a *power* ratio to dB with 10 log10 (not 20 log10)."""

    numerator = np.maximum(np.asarray(numerator_power, dtype=float), np.finfo(float).tiny)
    denominator = np.maximum(np.asarray(denominator_power, dtype=float), np.finfo(float).tiny)
    result = 10.0 * np.log10(numerator / denominator)
    return float(result) if result.ndim == 0 else result


def low_frequency_fraction(frequencies: np.ndarray, power: np.ndarray, *, upper_quantile: float = 0.1) -> float:
    if not 0 < upper_quantile <= 1:
        raise ValueError("upper_quantile must be in (0, 1]")
    positive = frequencies > 0
    f = frequencies[positive]
    p = power[positive]
    if len(f) == 0 or np.sum(p) <= 0:
        return 0.0
    cutoff = np.quantile(f, upper_quantile)
    return float(np.sum(p[f <= cutoff]) / np.sum(p))

