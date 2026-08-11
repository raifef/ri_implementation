"""Supplement VIII.C losses and direct-sigma analytic gradients."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .gaussian import BehaviorSnapshot, component_log_probability, entropy, gaussian_scores
from .contracts import (
    JOINT_LEARNED_DETECTOR_BASELINE,
    NON_SOURCE_EMA_BASELINE_ABLATION,
    NON_SOURCE_PPO_ABLATION,
    SOURCE_ELEMENTWISE_COORDINATE_CLIPPING,
    require_loss_semantics,
)


@dataclass(frozen=True)
class LossResult:
    total: float
    policy: float
    baseline: float
    entropy: float
    grad_mean: np.ndarray
    grad_sigma: np.ndarray
    grad_baseline: np.ndarray
    diagnostics: dict[str, Any]


def _mask(value: np.ndarray, controls: int) -> np.ndarray:
    mask = np.asarray(value, dtype=bool)
    if mask.ndim != 2 or mask.shape[1] != controls or not mask.any(axis=1).all():
        raise ValueError("each detector must have at least one aligned control")
    return mask


def total_loss_and_gradients(
    actions: np.ndarray,
    rewards: np.ndarray,
    mask: np.ndarray,
    mean: np.ndarray,
    sigma: np.ndarray,
    baseline: np.ndarray,
    behavior: BehaviorSnapshot,
    *,
    clip: float,
    policy_weight: float = 1.0,
    baseline_weight: float = 1.0,
    entropy_weight: float = 1.0,
    ratio_clipping_mode: str = SOURCE_ELEMENTWISE_COORDINATE_CLIPPING,
    baseline_mode: str = JOINT_LEARNED_DETECTOR_BASELINE,
    paper_mode: bool = True,
) -> LossResult:
    """Return Eq. (22) and analytic gradients for gradient descent.

    Collection parameters and advantages are immutable values. Thus the behavior
    policy receives no gradient and the baseline is optimized only by Eq. (19).
    """
    samples = np.asarray(actions, dtype=float)
    reward = np.asarray(rewards, dtype=float)
    mu = np.asarray(mean, dtype=float)
    sd = np.asarray(sigma, dtype=float)
    base = np.asarray(baseline, dtype=float)
    graph = _mask(mask, len(mu))
    if samples.ndim != 2 or samples.shape[1:] != mu.shape:
        raise ValueError("action shape mismatch")
    if reward.shape != (len(samples), graph.shape[0]) or base.shape != (graph.shape[0],):
        raise ValueError("reward or baseline shape mismatch")
    if behavior.component_log_probability.shape != samples.shape:
        raise ValueError("behavior snapshot is not aligned to actions")
    if not 0.0 < clip < 1.0:
        raise ValueError("clip must be in (0,1)")
    ratio_clipping_mode, baseline_mode = require_loss_semantics(
        ratio_clipping_mode, baseline_mode, paper_mode=paper_mode)

    advantages = reward - base[None, :]
    current_logp = component_log_probability(samples, mu, sd)
    coordinate_log_ratio = current_logp - behavior.component_log_probability
    lower_log, upper_log = np.log1p(-clip), np.log1p(clip)
    boundary_tolerance = 8.0 * np.finfo(float).eps
    graph_float = graph.astype(float)
    if ratio_clipping_mode == SOURCE_ELEMENTWISE_COORDINATE_CLIPPING:
        clipped_coordinate_log_ratio = np.clip(coordinate_log_ratio, lower_log, upper_log)
        masked_log_ratio = clipped_coordinate_log_ratio @ graph_float.T
        coordinate_active = (coordinate_log_ratio > lower_log + boundary_tolerance) & \
                            (coordinate_log_ratio < upper_log - boundary_tolerance)
        detector_active = np.ones_like(masked_log_ratio, dtype=bool)
    else:
        unclipped_masked_log_ratio = coordinate_log_ratio @ graph_float.T
        masked_log_ratio = np.clip(unclipped_masked_log_ratio, lower_log, upper_log)
        detector_active = (unclipped_masked_log_ratio > lower_log + boundary_tolerance) & \
                          (unclipped_masked_log_ratio < upper_log - boundary_tolerance)
        coordinate_active = np.ones_like(coordinate_log_ratio, dtype=bool)
    if np.any(masked_log_ratio > np.log(np.finfo(float).max)) or \
            np.any(masked_log_ratio < np.log(np.finfo(float).tiny)):
        raise FloatingPointError("masked coordinate-ratio product is outside float64 range")
    local_ratio = np.exp(masked_log_ratio)
    policy_loss = -float(policy_weight) * float(np.mean(np.sum(advantages * local_ratio, axis=1)))

    detector_weight = advantages * local_ratio * detector_active
    control_weights = (detector_weight @ graph_float) * coordinate_active / len(samples)
    score_mean, score_sigma = gaussian_scores(samples, mu, sd)
    grad_mean = -float(policy_weight) * np.sum(control_weights * score_mean, axis=0)
    reward_grad_sigma = -float(policy_weight) * np.sum(control_weights * score_sigma, axis=0)

    if baseline_mode == JOINT_LEARNED_DETECTOR_BASELINE:
        baseline_loss = float(baseline_weight) * float(np.mean(np.sum(advantages**2, axis=1)))
        grad_baseline = 2.0 * float(baseline_weight) * np.mean(base[None, :] - reward, axis=0)
    else:
        baseline_loss = 0.0
        grad_baseline = np.zeros_like(base)
    policy_entropy = entropy(sd)
    entropy_loss = -float(entropy_weight) * policy_entropy
    entropy_grad_sigma = -float(entropy_weight) / sd
    grad_sigma = reward_grad_sigma + entropy_grad_sigma
    return LossResult(policy_loss + baseline_loss + entropy_loss, policy_loss, baseline_loss,
                      entropy_loss, grad_mean, grad_sigma, grad_baseline,
                      {"entropy": policy_entropy,
                       "ratio_clipping_mode": ratio_clipping_mode,
                       "baseline_mode": baseline_mode,
                       "paper_mode": bool(paper_mode),
                       "coordinate_ratios_clipped_before_sparse_product":
                           ratio_clipping_mode == SOURCE_ELEMENTWISE_COORDINATE_CLIPPING,
                       "component_clip_fraction": float(1.0 - coordinate_active.mean()),
                       "detector_clip_fraction": float(1.0 - detector_active.mean()),
                       "tensor_shapes": {"actions": list(samples.shape),
                                         "current_log_probabilities": list(current_logp.shape),
                                         "behavior_log_probabilities": list(behavior.component_log_probability.shape),
                                         "coordinate_ratios": list(coordinate_log_ratio.shape),
                                         "mask": list(graph.shape), "masked_detector_ratios": list(local_ratio.shape),
                                         "advantages": list(advantages.shape)},
                       "ratio_mean": float(local_ratio.mean()),
                       "reward_sigma_gradient_norm": float(np.linalg.norm(reward_grad_sigma)),
                       "entropy_sigma_gradient_norm": float(np.linalg.norm(entropy_grad_sigma)),
                       "behavior_snapshot_writeable": bool(
                           behavior.mean.flags.writeable or behavior.sigma.flags.writeable
                           or behavior.component_log_probability.flags.writeable)})


def ema_baseline_update(baseline: np.ndarray, rewards: np.ndarray, coefficient: float,
                        *, paper_mode: bool = False) -> np.ndarray:
    """Explicit non-source ablation; never callable as a paper-mode baseline."""
    require_loss_semantics(SOURCE_ELEMENTWISE_COORDINATE_CLIPPING,
                           NON_SOURCE_EMA_BASELINE_ABLATION, paper_mode=paper_mode)
    base = np.asarray(baseline, dtype=float)
    reward = np.asarray(rewards, dtype=float)
    if reward.ndim != 2 or base.shape != (reward.shape[1],) or not 0.0 < coefficient <= 1.0:
        raise ValueError("invalid EMA baseline inputs")
    return (1.0 - coefficient) * base + coefficient * reward.mean(axis=0)
