"""Paper-literal local clipped-ratio objective and analytic gradient."""
from __future__ import annotations

import numpy as np

from .factor_graph import compose_detector_local_ratios, validate_mask
from .policy import component_log_probability


def clipped_objective_and_gradient(
    actions: np.ndarray,
    advantages: np.ndarray,
    mask: np.ndarray,
    mean: np.ndarray,
    log_scale: np.ndarray,
    old_mean: np.ndarray,
    old_log_scale: np.ndarray,
    *,
    clip: float,
    entropy_coefficient: float,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, float]]:
    """Supplement Eqs. 17-22 with element-wise clipping of chi near one."""
    actions = np.asarray(actions, dtype=float)
    advantages = np.asarray(advantages, dtype=float)
    graph = validate_mask(mask, actions.shape[1])
    if advantages.shape != (actions.shape[0], graph.shape[0]):
        raise ValueError("advantage and factor-graph shapes are inconsistent")
    if not 0.0 < clip < 1.0:
        raise ValueError("PPO clip must lie between zero and one")
    log_ratio = component_log_probability(actions, mean, log_scale) - component_log_probability(
        actions, old_mean, old_log_scale
    )
    component_ratio = np.exp(np.clip(log_ratio, -40.0, 40.0))
    clipped_component = np.clip(component_ratio, 1.0 - clip, 1.0 + clip)
    local_ratio = compose_detector_local_ratios(clipped_component, graph)
    weighted = advantages * local_ratio
    objective = float(np.mean(np.sum(weighted, axis=1)))
    active = (component_ratio > 1.0 - clip) & (component_ratio < 1.0 + clip)
    control_weight = (weighted @ graph.astype(float)) / actions.shape[0]
    control_weight *= active
    delta = actions - mean[None, :]
    inv_var = np.exp(-2.0 * log_scale)
    grad_mean = np.sum(control_weight * delta * inv_var[None, :], axis=0)
    grad_log_scale = np.sum(
        control_weight * (delta * delta * inv_var[None, :] - 1.0), axis=0
    )
    entropy = float(np.sum(log_scale + 0.5 * np.log(2.0 * np.pi * np.e)))
    objective += float(entropy_coefficient) * entropy
    grad_log_scale += float(entropy_coefficient)
    return objective, grad_mean, grad_log_scale, {
        "component_ratio_mean": float(component_ratio.mean()),
        "component_clip_fraction": float(1.0 - active.mean()),
        "detector_local_ratio_mean": float(local_ratio.mean()),
        "entropy": entropy,
    }


def sgd_ascent_step(
    mean: np.ndarray,
    log_scale: np.ndarray,
    grad_mean: np.ndarray,
    grad_log_scale: np.ndarray,
    *,
    mean_learning_rate: float,
    scale_learning_rate: float,
    bounds: tuple[float, float],
    scale_bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    new_mean = np.clip(mean + mean_learning_rate * grad_mean, bounds[0], bounds[1])
    new_log_scale = np.clip(
        log_scale + scale_learning_rate * grad_log_scale,
        np.log(scale_bounds[0]),
        np.log(scale_bounds[1]),
    )
    return new_mean, new_log_scale
