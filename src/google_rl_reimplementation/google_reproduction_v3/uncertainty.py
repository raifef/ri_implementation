"""Deterministic uncertainty and effective-sample-size utilities."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .estimators import repository_decay_estimator


def percentile_interval(values: Sequence[float], level: float = 0.95) -> tuple[float, float]:
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    alpha = (1.0 - level) / 2.0
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))


def parametric_decay_bootstrap(
    rounds: Sequence[int],
    errors: Sequence[int],
    shots: Sequence[int],
    *,
    seed: int = 73001,
    replicates: int = 500,
) -> dict[str, object]:
    estimate = repository_decay_estimator(rounds, errors, shots)
    r = np.asarray(rounds, dtype=int)
    n = np.asarray(shots, dtype=int)
    probabilities = 0.5 * (1.0 - np.exp(estimate.intercept + estimate.slope * r))
    probabilities = np.clip(probabilities, 0.0, 0.5)
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(replicates):
        simulated = rng.binomial(n, probabilities)
        samples.append(repository_decay_estimator(r, simulated, n).logical_error_per_cycle)
    low, high = percentile_interval(samples)
    return {
        "method": "parametric_binomial_bootstrap_of_full_decay_fit",
        "seed": seed,
        "replicates": replicates,
        "standard_error": float(np.std(samples, ddof=1)),
        "confidence_level": 0.95,
        "confidence_interval": [low, high],
    }


def block_bootstrap(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    block_size: int,
    replicates: int = 500,
    seed: int = 73002,
) -> np.ndarray:
    x = np.asarray(values)
    if x.ndim != 1 or block_size <= 0 or block_size > len(x):
        raise ValueError("invalid block bootstrap arguments")
    rng = np.random.default_rng(seed)
    block_starts = np.arange(0, len(x) - block_size + 1)
    blocks_needed = int(np.ceil(len(x) / block_size))
    results = np.empty(replicates)
    for index in range(replicates):
        starts = rng.choice(block_starts, blocks_needed, replace=True)
        sample = np.concatenate([x[start : start + block_size] for start in starts])[: len(x)]
        results[index] = statistic(sample)
    return results


def effective_sample_size(sample_count: int, autocorrelations: Sequence[float]) -> float:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    positive_sum = 0.0
    for value in list(autocorrelations)[1:]:
        if value <= 0:
            break
        positive_sum += value
    return float(sample_count / max(1.0, 1.0 + 2.0 * positive_sum))
