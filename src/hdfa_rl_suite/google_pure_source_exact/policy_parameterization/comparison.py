"""Matched direct-sigma versus explicitly non-paper log-sigma development runs."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .contracts import DIRECT_SIGMA_PARAMETERIZATION, NON_PAPER_LOG_SIGMA_ABLATION, PositivityGuard
from .gaussian import DirectSigmaGaussianPolicy, entropy
from .losses import total_loss_and_gradients
from .optimizer import DirectSigmaOptimizer, OptimizerConfig


def quadratic_rewards(actions: np.ndarray, optimum: np.ndarray, curvature: np.ndarray) -> np.ndarray:
    delta = np.asarray(actions) - np.asarray(optimum)[None, :]
    return -0.5 * np.asarray(curvature)[None, :] * delta**2


def _trajectory(*, seed: int, profile: Mapping[str, Any], regime: str,
                parameterization: str) -> dict[str, Any]:
    dimension, epochs = int(profile["dimension"]), int(profile["epochs"])
    candidates = int(profile["candidates_per_epoch"])
    curvature = np.full(dimension, float(profile["curvature"]))
    rng = np.random.default_rng(seed + (0 if regime == "stationary" else 1_000_000))
    mean = np.full(dimension, float(profile["initial_mean"]))
    sigma = np.full(dimension, float(profile["initial_sigma"]))
    baseline = np.zeros(dimension)
    mask = np.eye(dimension, dtype=bool)
    optimizer = DirectSigmaOptimizer(dimension, dimension, OptimizerConfig(
        mean_learning_rate=float(profile["mean_learning_rate"]),
        sigma_learning_rate=float(profile["sigma_learning_rate"]),
        baseline_learning_rate=float(profile["baseline_learning_rate"]),
        minimum_sigma=float(profile["minimum_sigma"]), maximum_sigma=float(profile["maximum_sigma"]),
        positivity_guard=PositivityGuard(profile["positivity_guard"])))
    log_sigma = np.log(sigma.copy())
    series: dict[str, list[float]] = {name: [] for name in (
        "mean_sigma", "entropy", "mean_tracking_error", "mean_policy_edr", "candidate_edr",
        "reward_sigma_gradient_norm", "entropy_sigma_gradient_norm", "fraction_at_positivity_guard")}
    for epoch in range(epochs):
        if regime == "stationary":
            optimum = np.full(dimension, float(profile["stationary_optimum"]))
            entropy_weight = float(profile["stationary_entropy_weight"])
        else:
            phase = 2.0 * np.pi * epoch / float(profile["drift_period_epochs"])
            optimum = np.full(dimension, float(profile["drift_amplitude"]) * np.sin(phase))
            entropy_weight = float(profile["nonstationary_entropy_weight"])
        noise = rng.normal(size=(candidates, dimension))
        policy = DirectSigmaGaussianPolicy(mean, sigma, seed=seed)
        batch = policy.sample(candidates, standardized_noise=noise)
        rewards = quadratic_rewards(batch.actions, optimum, curvature)
        result = total_loss_and_gradients(batch.actions, rewards, mask, mean, sigma, baseline,
                                          batch.behavior, clip=float(profile["ppo_clip"]),
                                          entropy_weight=entropy_weight,
                                          baseline_weight=float(profile["baseline_weight"]))
        if parameterization == DIRECT_SIGMA_PARAMETERIZATION:
            diagnostic = optimizer.step(mean, sigma, baseline, result.grad_mean, result.grad_sigma,
                                        result.grad_baseline, mean_bounds=tuple(profile["mean_bounds"]))
        elif parameterization == NON_PAPER_LOG_SIGMA_ABLATION:
            mean -= float(profile["mean_learning_rate"]) * result.grad_mean
            mean[:] = np.clip(mean, *tuple(profile["mean_bounds"]))
            log_sigma -= float(profile["sigma_learning_rate"]) * (result.grad_sigma * sigma)
            log_sigma[:] = np.clip(log_sigma, np.log(float(profile["minimum_sigma"])),
                                   np.log(float(profile["maximum_sigma"])))
            sigma[:] = np.exp(log_sigma)
            baseline -= float(profile["baseline_learning_rate"]) * result.grad_baseline
            diagnostic = {"fraction_at_positivity_guard": float(np.mean(
                sigma <= float(profile["minimum_sigma"]))) }
        else:
            raise ValueError("unknown comparison parameterization")
        values = (float(np.mean(sigma)), entropy(sigma), float(np.linalg.norm(mean - optimum)),
                  float(np.mean(0.5 * curvature * (mean - optimum) ** 2)), float(np.mean(-rewards)),
                  result.diagnostics["reward_sigma_gradient_norm"],
                  result.diagnostics["entropy_sigma_gradient_norm"],
                  diagnostic["fraction_at_positivity_guard"])
        for key, value in zip(series, values):
            series[key].append(value)
    return {"seed": seed, "regime": regime, "parameterization": parameterization,
            "optimized_scale_variable": "sigma" if parameterization == DIRECT_SIGMA_PARAMETERIZATION else "log_sigma",
            "trajectory": series, "initial_sigma": float(profile["initial_sigma"]),
            "final_sigma": series["mean_sigma"][-1], "final_tracking_error": series["mean_tracking_error"][-1],
            "mean_candidate_edr": float(np.mean(series["candidate_edr"])),
            "mean_policy_edr": float(np.mean(series["mean_policy_edr"])),
            "fraction_at_positivity_guard": float(np.mean(series["fraction_at_positivity_guard"]))}


def run_matched_seed(seed: int, profile: Mapping[str, Any]) -> dict[str, Any]:
    rows = [_trajectory(seed=seed, profile=profile, regime=regime, parameterization=parameterization)
            for regime in ("stationary", "nonstationary")
            for parameterization in (DIRECT_SIGMA_PARAMETERIZATION, NON_PAPER_LOG_SIGMA_ABLATION)]
    stationary = next(row for row in rows if row["regime"] == "stationary" and row["optimized_scale_variable"] == "sigma")
    moving = next(row for row in rows if row["regime"] == "nonstationary" and row["optimized_scale_variable"] == "sigma")
    return {"seed": int(seed), "common_random_numbers": True, "rows": rows,
            "gates": {"stationary_sigma_shrank": stationary["final_sigma"] < 0.8 * stationary["initial_sigma"],
                      "nonstationary_sigma_finite": bool(np.isfinite(moving["final_sigma"]) and
                                                         moving["final_sigma"] > float(profile["minimum_sigma"])),
                      "direct_guard_not_dominant": moving["fraction_at_positivity_guard"] < 0.05}}


def compare_positivity_guards() -> dict[str, Any]:
    rows = []
    for guard in PositivityGuard:
        mean, sigma, baseline = np.zeros(3), np.full(3, 0.2), np.zeros(2)
        optimizer = DirectSigmaOptimizer(3, 2, OptimizerConfig(
            0.1, 1.0, 0.1, minimum_sigma=0.01, maximum_sigma=2.0, positivity_guard=guard))
        diagnostic = optimizer.step(mean, sigma, baseline, np.zeros(3), np.full(3, 1e6), np.zeros(2))
        rows.append({"guard": guard.value, "sigma": sigma.tolist(), "positive": bool(np.all(sigma > 0)),
                     **diagnostic})
    return {"source_identifiability": "SOURCE_UNSPECIFIED_PREREGISTERED",
            "selected": PositivityGuard.PROJECTED_GRADIENT.value, "rows": rows,
            "all_direct_and_positive": all(row["positive"] for row in rows)}
