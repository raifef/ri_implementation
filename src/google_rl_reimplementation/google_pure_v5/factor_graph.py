"""Sparse detector-control factor-graph operations."""
from __future__ import annotations

import numpy as np


def validate_mask(mask: np.ndarray, control_count: int | None = None) -> np.ndarray:
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2 or not value.any(axis=1).all():
        raise ValueError("every detector must connect to at least one control")
    if control_count is not None and value.shape[1] != control_count:
        raise ValueError("factor graph control dimension mismatch")
    return value


def compose_detector_local_ratios(
    clipped_component_ratios: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Product of already element-wise-clipped control ratios for each detector."""
    ratios = np.asarray(clipped_component_ratios, dtype=float)
    graph = validate_mask(mask, ratios.shape[1])
    if np.any(ratios <= 0):
        raise ValueError("importance ratios must be positive")
    return np.exp(np.log(ratios) @ graph.astype(float).T)


def global_policy_ratio(component_ratios: np.ndarray) -> np.ndarray:
    return np.prod(np.asarray(component_ratios, dtype=float), axis=1)
