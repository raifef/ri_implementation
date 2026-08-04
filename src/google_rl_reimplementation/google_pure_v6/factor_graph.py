"""Detector-local current-over-collection importance ratios."""
from __future__ import annotations

import numpy as np

from .policy import component_log_probability


def validate_mask(mask: np.ndarray, control_count: int | None = None) -> np.ndarray:
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2 or not value.any(axis=1).all():
        raise ValueError("every detector requires at least one connected control")
    if control_count is not None and value.shape[1] != control_count:
        raise ValueError("factor-graph control dimension mismatch")
    return value


def local_log_ratios(
    latent_actions: np.ndarray,
    current_mean: np.ndarray,
    current_log_scale: np.ndarray,
    collection_component_log_probability: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    graph = validate_mask(mask, np.asarray(latent_actions).shape[1])
    current = component_log_probability(latent_actions, current_mean, current_log_scale)
    collection = np.asarray(collection_component_log_probability, dtype=float)
    if current.shape != collection.shape:
        raise ValueError("collection log-probability shape mismatch")
    return (current - collection) @ graph.astype(float).T


def local_importance_ratios(*args: object, log_guard: float = 40.0) -> np.ndarray:
    logs = local_log_ratios(*args)
    return np.exp(np.clip(logs, -float(log_guard), float(log_guard)))


def global_importance_ratio(
    latent_actions: np.ndarray,
    current_mean: np.ndarray,
    current_log_scale: np.ndarray,
    collection_component_log_probability: np.ndarray,
) -> np.ndarray:
    current = component_log_probability(latent_actions, current_mean, current_log_scale)
    return np.exp(np.clip(np.sum(current - collection_component_log_probability, axis=1), -40.0, 40.0))
