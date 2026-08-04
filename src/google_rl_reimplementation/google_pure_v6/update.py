"""Literal detector-local PPO objective and its analytic score-function gradient."""
from __future__ import annotations

from typing import Any

import numpy as np

from .factor_graph import local_importance_ratios, validate_mask
from .policy import gaussian_scores


def ppo_objective_and_gradient(
    latent_actions: np.ndarray,
    advantages: np.ndarray,
    mask: np.ndarray,
    current_mean: np.ndarray,
    current_log_scale: np.ndarray,
    collection_component_log_probability: np.ndarray,
    *,
    clip: float,
    entropy_coefficient: float,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, Any]]:
    """Mean over candidates of the detector sum; entropy is added once per control."""
    actions = np.asarray(latent_actions, dtype=float)
    advantages = np.asarray(advantages, dtype=float)
    graph = validate_mask(mask, actions.shape[1])
    if advantages.shape != (actions.shape[0], graph.shape[0]):
        raise ValueError("advantage shape mismatch")
    if not 0.0 < clip < 1.0:
        raise ValueError("PPO clip must be in (0,1)")
    ratios = local_importance_ratios(
        actions, current_mean, current_log_scale, collection_component_log_probability, graph
    )
    clipped = np.clip(ratios, 1.0 - clip, 1.0 + clip)
    raw_terms = ratios * advantages
    clipped_terms = clipped * advantages
    terms = np.minimum(raw_terms, clipped_terms)
    active = ((advantages >= 0.0) & (ratios <= 1.0 + clip)) | (
        (advantages < 0.0) & (ratios >= 1.0 - clip)
    )
    local_weights = np.where(active, advantages * ratios, 0.0) / actions.shape[0]
    control_weights = local_weights @ graph.astype(float)
    score_mean, score_log_scale = gaussian_scores(actions, current_mean, current_log_scale)
    grad_mean = np.sum(control_weights * score_mean, axis=0)
    grad_log_scale = np.sum(control_weights * score_log_scale, axis=0)
    entropy = float(np.sum(current_log_scale + 0.5 * np.log(2.0 * np.pi * np.e)))
    objective = float(np.mean(np.sum(terms, axis=1))) + float(entropy_coefficient) * entropy
    grad_log_scale += float(entropy_coefficient)
    degree = graph.sum(axis=0).astype(int)
    return objective, grad_mean, grad_log_scale, {
        "ratio_mean": float(ratios.mean()),
        "ratio_min": float(ratios.min()),
        "ratio_max": float(ratios.max()),
        "clip_fraction": float(1.0 - active.mean()),
        "entropy": entropy,
        "detector_normalization": "sum_over_detectors_then_mean_over_candidates",
        "control_detector_degree": degree.tolist(),
        "effective_entropy_gradient": np.full(actions.shape[1], entropy_coefficient).tolist(),
    }


def legacy_v5_component_clipped_objective_and_gradient(
    latent_actions: np.ndarray,
    advantages: np.ndarray,
    mask: np.ndarray,
    current_mean: np.ndarray,
    current_log_scale: np.ndarray,
    collection_mean: np.ndarray,
    collection_log_scale: np.ndarray,
    *,
    clip: float,
    entropy_coefficient: float,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, Any]]:
    """Frozen diagnostic reproduction of v5; never the production v6 objective."""
    from .policy import component_log_probability

    actions = np.asarray(latent_actions, dtype=float)
    graph = validate_mask(mask, actions.shape[1])
    advantages = np.asarray(advantages, dtype=float)
    log_ratio = component_log_probability(actions, current_mean, current_log_scale) - component_log_probability(
        actions, collection_mean, collection_log_scale
    )
    component_ratio = np.exp(np.clip(log_ratio, -40.0, 40.0))
    component_clipped = np.clip(component_ratio, 1.0 - clip, 1.0 + clip)
    local_ratio = np.exp(np.log(component_clipped) @ graph.astype(float).T)
    active = (component_ratio > 1.0 - clip) & (component_ratio < 1.0 + clip)
    weighted = advantages * local_ratio
    control_weights = (weighted @ graph.astype(float)) * active / actions.shape[0]
    score_mean, score_scale = gaussian_scores(actions, current_mean, current_log_scale)
    gm = np.sum(control_weights * score_mean, axis=0)
    gs = np.sum(control_weights * score_scale, axis=0) + entropy_coefficient
    entropy = float(np.sum(current_log_scale + 0.5 * np.log(2.0 * np.pi * np.e)))
    return float(np.mean(np.sum(weighted, axis=1)) + entropy_coefficient * entropy), gm, gs, {
        "ratio_mean": float(local_ratio.mean()), "clip_fraction": float(1.0 - active.mean()),
        "objective_mode": "legacy_v5_component_clipping",
    }


def sgd_ascent_step(
    mean: np.ndarray,
    log_scale: np.ndarray,
    grad_mean: np.ndarray,
    grad_log_scale: np.ndarray,
    *,
    mean_learning_rate: float,
    scale_learning_rate: float,
    normalized_bounds: tuple[float, float],
    scale_bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.clip(mean + mean_learning_rate * grad_mean, *normalized_bounds),
        np.clip(log_scale + scale_learning_rate * grad_log_scale, np.log(scale_bounds[0]), np.log(scale_bounds[1])),
    )
