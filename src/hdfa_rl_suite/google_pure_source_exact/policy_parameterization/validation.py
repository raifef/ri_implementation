"""Independent numerical checks for direct-sigma policy mathematics."""
from __future__ import annotations

from typing import Callable

import numpy as np

from .comparison import compare_positivity_guards
from .gaussian import BehaviorSnapshot, DirectSigmaGaussianPolicy, component_log_probability, entropy, gaussian_scores
from .losses import ema_baseline_update, total_loss_and_gradients
from .contracts import NON_SOURCE_PPO_ABLATION, SOURCE_ELEMENTWISE_COORDINATE_CLIPPING


def finite_difference(function: Callable[[np.ndarray], float], value: np.ndarray,
                      epsilon: float = 1e-6) -> np.ndarray:
    result = np.zeros_like(np.asarray(value, dtype=float))
    for index in range(len(result)):
        plus, minus = np.asarray(value, dtype=float).copy(), np.asarray(value, dtype=float).copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        result[index] = (function(plus) - function(minus)) / (2.0 * epsilon)
    return result


def mathematical_audit(seed: int = 40001) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    mean = np.asarray([0.1, -0.2, 0.05])
    sigma = np.asarray([0.4, 0.7, 0.25])
    actions = rng.normal(mean, sigma, size=(9, 3))
    score_mu, score_sigma = gaussian_scores(actions, mean, sigma)
    numeric_mu = np.vstack([finite_difference(
        lambda value, row=row: float(component_log_probability(row[None, :], value, sigma).sum()), mean)
        for row in actions])
    numeric_sigma = np.vstack([finite_difference(
        lambda value, row=row: float(component_log_probability(row[None, :], mean, value).sum()), sigma)
        for row in actions])
    entropy_numeric = finite_difference(lambda value: entropy(value), sigma)
    entropy_analytic = 1.0 / sigma

    mask = np.asarray([[1, 1, 0], [0, 1, 1]], dtype=bool)
    policy = DirectSigmaGaussianPolicy(mean, sigma, seed=seed)
    batch = policy.sample(len(actions), standardized_noise=(actions - mean) / sigma)
    rewards = rng.normal(size=(len(actions), 2))
    baseline = np.asarray([0.03, -0.04])
    kwargs = dict(actions=actions, rewards=rewards, mask=mask, baseline=baseline,
                  behavior=batch.behavior, clip=0.2, policy_weight=0.8,
                  baseline_weight=0.3, entropy_weight=0.07)
    analytic = total_loss_and_gradients(mean=mean, sigma=sigma, **kwargs)
    numeric_total_mu = finite_difference(
        lambda value: total_loss_and_gradients(mean=value, sigma=sigma, **kwargs).total, mean)
    numeric_total_sigma = finite_difference(
        lambda value: total_loss_and_gradients(mean=mean, sigma=value, **kwargs).total, sigma)
    numeric_baseline = finite_difference(
        lambda value: total_loss_and_gradients(actions=actions, rewards=rewards, mask=mask,
            mean=mean, sigma=sigma, baseline=value, behavior=batch.behavior, clip=0.2,
            policy_weight=0.0, baseline_weight=0.3, entropy_weight=0.0).total, baseline)
    errors = {
        "log_probability_mu": float(np.max(np.abs(score_mu - numeric_mu))),
        "log_probability_sigma": float(np.max(np.abs(score_sigma - numeric_sigma))),
        "entropy_sigma": float(np.max(np.abs(entropy_analytic - entropy_numeric))),
        "total_loss_mu": float(np.max(np.abs(analytic.grad_mean - numeric_total_mu))),
        "total_loss_sigma": float(np.max(np.abs(analytic.grad_sigma - numeric_total_sigma))),
        "baseline_loss": float(np.max(np.abs(analytic.grad_baseline - numeric_baseline))),
    }
    tolerance = 2e-5
    before = entropy(sigma)
    after = entropy(sigma - 0.001 * (-0.07 / sigma))
    return {"seed": seed, "certification_seed_consumed": False, "errors": errors,
            "tolerance": tolerance, "finite_difference_pass": max(errors.values()) < tolerance,
            "negative_entropy_descent_increases_entropy": after > before,
            "behavior_snapshot_immutable": not batch.behavior.mean.flags.writeable
                and not batch.behavior.sigma.flags.writeable
                and not batch.behavior.component_log_probability.flags.writeable,
            "positivity_guard_comparison": compare_positivity_guards()}


def source_loss_semantics_audit(seed: int = 40002) -> dict[str, object]:
    """Independent hand/finite-difference gates for Supplement Eqs. 17-19."""
    rng = np.random.default_rng(seed)
    mean, sigma = np.zeros(3), np.ones(3)
    actions = np.asarray([[0.15, -0.2, 0.3], [-0.1, 0.25, 0.2]])
    current = component_log_probability(actions, mean, sigma)
    behavior = BehaviorSnapshot(mean, sigma, current - np.log(1.15), 11)
    rewards = np.asarray([[1.0], [0.7]])
    mask = np.ones((1, 3), dtype=bool)
    common = dict(actions=actions, rewards=rewards, mask=mask, mean=mean, sigma=sigma,
                  baseline=np.zeros(1), behavior=behavior, clip=0.2,
                  baseline_weight=0.0, entropy_weight=0.0, paper_mode=False)
    source = total_loss_and_gradients(
        **common, ratio_clipping_mode=SOURCE_ELEMENTWISE_COORDINATE_CLIPPING)
    aggregate = total_loss_and_gradients(
        **common, ratio_clipping_mode=NON_SOURCE_PPO_ABLATION)
    hand_source = -float(rewards.mean()) * 1.15 ** 3
    hand_aggregate = -float(rewards.mean()) * 1.2

    detector_rewards = rng.normal(loc=np.asarray([0.3, -0.2]), scale=0.4, size=(64, 2))
    detector_mean = detector_rewards.mean(axis=0)
    baseline_actions = rng.normal(size=(64, 2))
    baseline_behavior = DirectSigmaGaussianPolicy(np.zeros(2), np.ones(2), seed=seed).sample(
        64, standardized_noise=baseline_actions).behavior
    baseline_common = dict(actions=baseline_actions, rewards=detector_rewards,
                           mask=np.eye(2, dtype=bool), mean=np.zeros(2), sigma=np.ones(2),
                           behavior=baseline_behavior, clip=0.2, policy_weight=0.0,
                           baseline_weight=0.2, entropy_weight=0.0)
    baseline_at_optimum = total_loss_and_gradients(baseline=detector_mean, **baseline_common)
    start = np.asarray([0.1, -0.1])
    baseline_analytic = total_loss_and_gradients(baseline=start, **baseline_common)
    baseline_numeric = finite_difference(
        lambda value: total_loss_and_gradients(baseline=value, **baseline_common).total, start)
    return {
        "seed": seed, "certification_seed_consumed": False,
        "coordinate_clip_before_product": source.diagnostics[
            "coordinate_ratios_clipped_before_sparse_product"],
        "source_hand_error": abs(source.policy - hand_source),
        "aggregate_hand_error": abs(aggregate.policy - hand_aggregate),
        "multi_coordinate_non_equivalence": source.policy != aggregate.policy,
        "paper_ratio_mode": source.diagnostics["ratio_clipping_mode"],
        "baseline_mode": baseline_analytic.diagnostics["baseline_mode"],
        "baseline_component_count": len(baseline_analytic.grad_baseline),
        "baseline_optimum_gradient_norm": float(np.linalg.norm(baseline_at_optimum.grad_baseline)),
        "baseline_finite_difference_error": float(np.max(np.abs(
            baseline_analytic.grad_baseline - baseline_numeric))),
        "pass": bool(source.diagnostics["coordinate_ratios_clipped_before_sparse_product"]
                     and abs(source.policy - hand_source) < 1e-14
                     and abs(aggregate.policy - hand_aggregate) < 1e-14
                     and source.policy != aggregate.policy
                     and np.linalg.norm(baseline_at_optimum.grad_baseline) < 1e-14
                     and np.max(np.abs(baseline_analytic.grad_baseline - baseline_numeric)) < 2e-7),
    }


def baseline_dynamics_audit(seed: int = 40003, epochs: int = 80) -> dict[str, object]:
    """Common-batch learned-loss versus EMA ablation comparison without future data."""
    rng = np.random.default_rng(seed)
    detector_count, batch_size = 3, 64
    learned = np.zeros(detector_count)
    ema = np.zeros(detector_count)
    learning_rate, loss_weight = 0.1, 0.2
    equivalent_ema_coefficient = 2.0 * learning_rate * loss_weight
    learned_rows, ema_rows = [], []
    for epoch in range(epochs):
        expected = np.asarray([0.3, -0.1, 0.2])
        if epoch >= epochs // 2:
            expected = expected + 0.15 * np.sin(2.0 * np.pi * (epoch - epochs // 2) / 20.0)
        rewards = rng.normal(expected, 0.25, size=(batch_size, detector_count))
        preupdate = learned.copy()
        gradient = 2.0 * loss_weight * np.mean(learned[None, :] - rewards, axis=0)
        learned -= learning_rate * gradient
        ema = ema_baseline_update(ema, rewards, equivalent_ema_coefficient)
        learned_rows.append({"epoch": epoch, "preupdate": preupdate.tolist(),
                             "postupdate": learned.tolist(),
                             "advantage_second_moment": float(np.mean((rewards - preupdate) ** 2))})
        ema_rows.append({"epoch": epoch, "postupdate": ema.tolist(),
                         "advantage_second_moment": float(np.mean((rewards -
                             np.asarray(ema_rows[-1]["postupdate"]) if ema_rows else rewards) ** 2))})
    agreement = float(np.max(np.abs(learned - ema)))
    first = float(np.mean([row["advantage_second_moment"] for row in learned_rows[:10]]))
    final = float(np.mean([row["advantage_second_moment"] for row in learned_rows[-10:]]))
    return {"seed": seed, "certification_seed_consumed": False,
            "learned_loss_weight": loss_weight, "learned_learning_rate": learning_rate,
            "ema_ablation_equivalent_coefficient": equivalent_ema_coefficient,
            "learned_trajectory": learned_rows, "ema_ablation_trajectory": ema_rows,
            "final_parameter_agreement": agreement,
            "early_advantage_second_moment": first, "late_advantage_second_moment": final,
            "source_consistent_variance": final < first,
            "causal_preupdate_baseline_logged": True,
            "pass": bool(agreement < 1e-14 and final < first)}
