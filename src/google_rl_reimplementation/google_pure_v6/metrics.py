"""Unambiguous suppression/residual metric conventions."""
from __future__ import annotations

from typing import Any

import numpy as np


def stability_metrics(fixed: np.ndarray, learned_mean: np.ndarray, *, identifiable: bool = True) -> dict[str, Any]:
    fixed_std = float(np.std(np.asarray(fixed, dtype=float), ddof=1))
    mean_std = float(np.std(np.asarray(learned_mean, dtype=float), ddof=1))
    if not identifiable or fixed_std <= 0 or mean_std <= 0:
        return {
            "status": "NON_IDENTIFIABLE_FIXED_VARIANCE",
            "fixed_ler_std": fixed_std,
            "learned_mean_ler_std": mean_std,
            "stability_suppression_factor_fixed_over_mean": None,
            "stability_residual_ratio_mean_over_fixed": None,
            "stability_factor_orientation": "fixed_std_over_learned_mean_std",
        }
    suppression = fixed_std / mean_std
    return {
        "status": "IDENTIFIABLE",
        "fixed_ler_std": fixed_std,
        "learned_mean_ler_std": mean_std,
        "stability_suppression_factor_fixed_over_mean": suppression,
        "stability_residual_ratio_mean_over_fixed": 1.0 / suppression,
        "stability_factor_orientation": "fixed_std_over_learned_mean_std",
    }


def spectral_metrics(fixed_power: float, learned_mean_power: float) -> dict[str, Any]:
    if fixed_power <= 0 or learned_mean_power <= 0:
        raise ValueError("spectral powers must be positive")
    gain = float(10.0 * np.log10(fixed_power / learned_mean_power))
    return {
        "low_frequency_suppression_db_fixed_over_mean": gain,
        "low_frequency_residual_db_mean_over_fixed": -gain,
        "spectral_gain_orientation": "10log10(fixed_power/learned_mean_power)",
    }
