"""Independent logical-memory and detector-statistic estimators."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .schemas import LogicalEstimate


def memory_failure_to_error_per_cycle(failure_probability: float, rounds: int) -> float:
    """Invert p_fail = (1 - (1 - 2e)^rounds) / 2 exactly."""

    if not 0.0 <= failure_probability <= 0.5:
        raise ValueError("memory failure probability must be in [0, 0.5]")
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    return 0.5 * (1.0 - max(0.0, 1.0 - 2.0 * failure_probability) ** (1.0 / rounds))


def error_per_cycle_to_memory_failure(error_per_cycle: float, rounds: int) -> float:
    if not 0.0 <= error_per_cycle <= 0.5 or rounds <= 0:
        raise ValueError("invalid logical error per cycle or rounds")
    return 0.5 * (1.0 - (1.0 - 2.0 * error_per_cycle) ** rounds)


def _observations(
    rounds: Sequence[int], errors: Sequence[int], shots: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r = np.asarray(rounds, dtype=float)
    k = np.asarray(errors, dtype=float)
    n = np.asarray(shots, dtype=float)
    if not (r.ndim == k.ndim == n.ndim == 1 and len(r) == len(k) == len(n) and len(r) >= 2):
        raise ValueError("rounds, errors, and shots must be equal-length vectors with at least two points")
    if np.any(r <= 0) or np.any(n <= 0) or np.any(k < 0) or np.any(k > n / 2):
        raise ValueError("invalid memory-decay observations")
    failure = k / n
    expectation = np.clip(1.0 - 2.0 * failure, 1e-12, 1.0)
    return r, k, n, expectation


def repository_decay_estimator(
    rounds: Sequence[int], errors: Sequence[int], shots: Sequence[int]
) -> LogicalEstimate:
    """Repository estimator: heteroscedastic weighted log-expectation fit.

    The free intercept absorbs state-preparation and measurement contrast.  The
    slope gives the per-cycle decay and is converted with the exact parity law.
    """

    r, _, n, expectation = _observations(rounds, errors, shots)
    target = np.log(expectation)
    variance = np.maximum(1e-15, (1.0 - expectation**2) / (n * expectation**2))
    weights = 1.0 / variance
    design = np.column_stack([np.ones(len(r)), r])
    normal = design.T @ (weights[:, None] * design)
    beta = np.linalg.solve(normal, design.T @ (weights * target))
    residual = target - design @ beta
    dof = max(1, len(r) - 2)
    dispersion = max(1.0, float(np.sum(weights * residual**2) / dof))
    covariance = np.linalg.inv(normal) * dispersion
    slope = float(beta[1])
    epc = 0.5 * (1.0 - math.exp(slope))
    slope_se = math.sqrt(max(0.0, float(covariance[1, 1])))
    epc_se = 0.5 * math.exp(slope) * slope_se
    return LogicalEstimate(epc, float(beta[0]), slope, epc_se, "weighted_logical_expectation_decay", len(r))


def independent_nonlinear_decay_estimator(
    rounds: Sequence[int], errors: Sequence[int], shots: Sequence[int], *, max_iterations: int = 50
) -> LogicalEstimate:
    """Independent nonlinear WLS fit in expectation space.

    This avoids the log transform used by the repository estimator and is an
    intentionally independent cross-check of its normalization.
    """

    r, _, n, observed = _observations(rounds, errors, shots)
    initial = repository_decay_estimator(rounds, errors, shots)
    amplitude = float(math.exp(initial.intercept))
    decay = max(1e-12, -initial.slope)
    weights = n / np.maximum(1e-12, 1.0 - observed**2)
    for _ in range(max_iterations):
        exponential = np.exp(-decay * r)
        predicted = amplitude * exponential
        residual = observed - predicted
        jacobian = np.column_stack([exponential, -amplitude * r * exponential])
        normal = jacobian.T @ (weights[:, None] * jacobian)
        step = np.linalg.solve(normal + np.eye(2) * 1e-15, jacobian.T @ (weights * residual))
        new_amplitude = float(np.clip(amplitude + step[0], 0.5, 1.5))
        new_decay = float(np.clip(decay + step[1], 1e-12, 1.0))
        if abs(new_amplitude - amplitude) + abs(new_decay - decay) < 1e-13:
            amplitude, decay = new_amplitude, new_decay
            break
        amplitude, decay = new_amplitude, new_decay
    exponential = np.exp(-decay * r)
    jacobian = np.column_stack([exponential, -amplitude * r * exponential])
    normal = jacobian.T @ (weights[:, None] * jacobian)
    residual = observed - amplitude * exponential
    dispersion = max(1.0, float(np.sum(weights * residual**2) / max(1, len(r) - 2)))
    covariance = np.linalg.inv(normal) * dispersion
    epc = 0.5 * (1.0 - math.exp(-decay))
    epc_se = 0.5 * math.exp(-decay) * math.sqrt(max(0.0, float(covariance[1, 1])))
    return LogicalEstimate(
        epc,
        math.log(amplitude),
        -decay,
        epc_se,
        "independent_nonlinear_expectation_decay",
        len(r),
    )


def detector_summary(detections: np.ndarray) -> dict[str, object]:
    if detections.ndim != 2 or detections.shape[0] < 4 or detections.shape[1] < 1:
        raise ValueError("detections must be a shots-by-detectors matrix")
    data = detections.astype(float, copy=False)
    rates = np.mean(data, axis=0)
    reward = np.mean(data, axis=1)
    # Full d7 covariance matrices are unnecessarily quadratic for the summary.
    # Use an evenly spaced, deterministic detector panel while retaining the
    # complete detector population for rate and reward statistics.
    covariance_panel = np.linspace(0, len(rates) - 1, min(128, len(rates)), dtype=int)
    panel = data[:, covariance_panel]
    panel = panel - np.mean(panel, axis=0)
    covariance = panel.T @ panel / max(1, len(data) - 1)
    off_diagonal = covariance[~np.eye(len(covariance_panel), dtype=bool)] if len(covariance_panel) > 1 else np.array([0.0])
    block_size = max(16, len(data) // 16)
    blocks = len(data) // block_size
    trimmed = data[: blocks * block_size].reshape(blocks, block_size, data.shape[1])
    block_rates = trimmed.mean(axis=1)
    observed_variance = block_rates.var(axis=0, ddof=1) if blocks > 1 else np.zeros_like(rates)
    binomial_variance = rates * (1.0 - rates) / block_size
    valid = binomial_variance > 1e-12
    dispersion = observed_variance[valid] / binomial_variance[valid]
    return {
        "shots": int(data.shape[0]),
        "detectors": int(data.shape[1]),
        "covariance_detector_panel": int(len(covariance_panel)),
        "detector_rate_mean": float(np.mean(rates)),
        "detector_rate_std": float(np.std(rates)),
        "detector_rate_quantiles": {
            str(q): float(np.quantile(rates, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
        "reward_mean": float(np.mean(reward)),
        "reward_variance": float(np.var(reward, ddof=1)),
        "mean_off_diagonal_covariance": float(np.mean(off_diagonal)),
        "mean_absolute_off_diagonal_covariance": float(np.mean(np.abs(off_diagonal))),
        "covariance_quantiles": {str(q): float(np.quantile(off_diagonal, q)) for q in (0.05, 0.5, 0.95)},
        "overdispersion_median": float(np.median(dispersion)) if len(dispersion) else None,
        "overdispersion_q90": float(np.quantile(dispersion, 0.9)) if len(dispersion) else None,
        "first_half_reward_mean": float(np.mean(reward[: len(reward) // 2])),
        "second_half_reward_mean": float(np.mean(reward[len(reward) // 2 :])),
    }
