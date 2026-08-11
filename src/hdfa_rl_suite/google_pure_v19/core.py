"""Pure mathematics used by the V19 diagnostic and its synthetic tests."""
from __future__ import annotations

import math
from typing import Callable, Mapping, Sequence

import numpy as np


IMPLEMENTED_SOURCE_STYLE_SCALE_OBJECTIVE = (
    "IMPLEMENTED_SOURCE_STYLE_OBJECTIVE_WITH_INHERITED_HYPERPARAMETERS"
)
PUBLIC_ANALOGUE_SCALE_OBJECTIVE = "PUBLIC_ANALOGUE_SCALE_OBJECTIVE"


def coordinate_quadratic_damage(hessian_diagonal: np.ndarray,
                                sigma: np.ndarray) -> np.ndarray:
    hessian = np.asarray(hessian_diagonal, dtype=float)
    scale = np.asarray(sigma, dtype=float)
    if hessian.ndim != 1 or scale.shape != hessian.shape:
        raise ValueError("hessian diagonal and sigma must be aligned vectors")
    if (not np.all(np.isfinite(hessian)) or not np.all(np.isfinite(scale)) or
            np.any(scale < 0)):
        raise ValueError("quadratic damage inputs must be finite with nonnegative sigma")
    return 0.5 * hessian * np.square(scale)


def quadratic_damage(hessian_diagonal: np.ndarray, sigma: np.ndarray) -> float:
    return float(np.sum(coordinate_quadratic_damage(hessian_diagonal, sigma)))


def aggregate_damage(coordinate_damage: np.ndarray,
                     labels: Sequence[str]) -> dict[str, float]:
    damage = np.asarray(coordinate_damage, dtype=float)
    if damage.ndim != 1 or len(labels) != damage.size:
        raise ValueError("damage and aggregation labels must align")
    result: dict[str, float] = {}
    for label, value in zip(labels, damage, strict=True):
        result[str(label)] = result.get(str(label), 0.0) + float(value)
    if not math.isclose(sum(result.values()), float(np.sum(damage)), rel_tol=0, abs_tol=1e-12):
        raise AssertionError("damage aggregation did not conserve the total")
    return result


def effective_dimension(nonnegative_damage: np.ndarray) -> float:
    values = np.maximum(np.asarray(nonnegative_damage, dtype=float), 0.0)
    denominator = float(np.sum(np.square(values)))
    return 0.0 if denominator == 0 else float(np.sum(values) ** 2 / denominator)


def cumulative_rank_curve(nonnegative_damage: np.ndarray) -> list[dict[str, float]]:
    values = np.sort(np.maximum(np.asarray(nonnegative_damage, dtype=float), 0.0))[::-1]
    total = float(np.sum(values))
    cumulative = np.cumsum(values)
    return [{"rank": int(index + 1), "fraction": (0.0 if total == 0 else float(value / total))}
            for index, value in enumerate(cumulative)]


def sigma_equilibrium(hessian_diagonal: np.ndarray, entropy_weight: float,
                      *, entropy_divisor: float = 1.0) -> np.ndarray:
    hessian = np.asarray(hessian_diagonal, dtype=float)
    beta = float(entropy_weight)
    divisor = float(entropy_divisor)
    if hessian.ndim != 1 or np.any(hessian <= 0) or not np.all(np.isfinite(hessian)):
        raise ValueError("sigma equilibrium requires a positive finite Hessian diagonal")
    if beta <= 0 or divisor <= 0:
        raise ValueError("entropy weight and divisor must be positive")
    return np.sqrt(beta / (divisor * hessian))


def classify_bound_activity(unconstrained_sigma: float, maximum_sigma: float,
                            observed_ceiling_occupancy: float = 0.0) -> str:
    equilibrium = float(unconstrained_sigma)
    bound = float(maximum_sigma)
    occupancy = float(observed_ceiling_occupancy)
    if min(equilibrium, bound) <= 0 or not 0 <= occupancy <= 1:
        raise ValueError("invalid bound-activity inputs")
    if equilibrium >= 1.5 * bound:
        return "strongly truncating the objective optimum"
    if equilibrium >= bound:
        return "equilibrium-limiting"
    if occupancy >= 0.01 or equilibrium >= 0.9 * bound:
        return "occasionally active"
    return "inactive"


def public_analogue_entropy_gradient(sigma: np.ndarray, entropy_weight: float,
                                     active_dimension: int) -> np.ndarray:
    scale = np.asarray(sigma, dtype=float)
    if scale.ndim != 1 or np.any(scale <= 0) or active_dimension <= 0:
        raise ValueError("public analogue entropy gradient inputs are invalid")
    return -float(entropy_weight) / (int(active_dimension) * scale)


def phase_bin_means(phases: np.ndarray, values: np.ndarray, bins: int) -> dict[str, np.ndarray]:
    phase = np.mod(np.asarray(phases, dtype=float), 2.0 * np.pi)
    data = np.asarray(values, dtype=float)
    if phase.ndim != 1 or data.shape[0] != phase.size or bins < 2:
        raise ValueError("phase/value rows must align and bins must exceed one")
    edges = np.linspace(0.0, 2.0 * np.pi, int(bins) + 1)
    indices = np.minimum(np.digitize(phase, edges, right=False) - 1, bins - 1)
    means = []
    counts = []
    for index in range(bins):
        selected = data[indices == index]
        counts.append(len(selected))
        means.append(np.full(data.shape[1:], np.nan) if len(selected) == 0 else np.mean(selected, axis=0))
    return {"centers": 0.5 * (edges[:-1] + edges[1:]),
            "counts": np.asarray(counts, dtype=int), "means": np.asarray(means)}


def phase_aligned_distance(left: np.ndarray, right: np.ndarray) -> float:
    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    if first.shape != second.shape or first.size == 0:
        raise ValueError("phase-aligned arrays must have the same nonempty shape")
    denominator = float(np.linalg.norm(first))
    return float("inf") if denominator == 0 else float(np.linalg.norm(second - first) / denominator)


def lambda_squared_fit(multipliers: np.ndarray, damage: np.ndarray) -> dict[str, float]:
    lam = np.asarray(multipliers, dtype=float)
    observed = np.asarray(damage, dtype=float)
    if lam.ndim != 1 or observed.shape != lam.shape or np.any(lam < 0):
        raise ValueError("lambda-squared fit inputs must be aligned and nonnegative")
    x = np.square(lam)
    denominator = float(x @ x)
    slope = 0.0 if denominator == 0 else float(x @ observed / denominator)
    fitted = slope * x
    residual = observed - fitted
    total = float(np.sum(np.square(observed - np.mean(observed))))
    r2 = 1.0 if total == 0 and np.allclose(residual, 0) else (
        float(1.0 - np.sum(np.square(residual)) / total) if total > 0 else float("-inf"))
    return {"slope": slope, "r_squared": r2,
            "maximum_absolute_residual": float(np.max(np.abs(residual)))}


def frozen_sigma_sweep(mean: np.ndarray, sigma: np.ndarray, noises: np.ndarray,
                       multipliers: Sequence[float],
                       cost: Callable[[np.ndarray], float]) -> dict[str, object]:
    """Evaluation-only sweep; inputs are copied and never updated."""
    mu = np.asarray(mean, dtype=float).copy()
    scale = np.asarray(sigma, dtype=float).copy()
    noise = np.asarray(noises, dtype=float)
    if mu.ndim != 1 or scale.shape != mu.shape or noise.ndim != 2 or noise.shape[1] != mu.size:
        raise ValueError("frozen sweep shapes are inconsistent")
    before_mean, before_sigma = mu.copy(), scale.copy()
    mean_cost = float(cost(mu))
    rows = []
    for multiplier in multipliers:
        candidate_costs = [float(cost(mu + float(multiplier) * scale * row)) for row in noise]
        rows.append({"lambda": float(multiplier), "C_mean": mean_cost,
                     "C_stochastic": float(np.mean(candidate_costs)),
                     "damage": float(np.mean(candidate_costs) - mean_cost)})
    return {"rows": rows,
            "policy_state_unchanged": bool(np.array_equal(mu, before_mean) and
                                            np.array_equal(scale, before_sigma)),
            "lambda_zero_equals_mean": bool(rows and rows[0]["lambda"] == 0.0 and
                                             rows[0]["C_stochastic"] == mean_cost)}


def aggregation_scaling_fixture(dimensions: Sequence[int], *, curvature: float,
                                sigma: float, entropy_weight: float) -> list[dict[str, float]]:
    """Independent one-detector/one-control copies expose exact dimensional scaling."""
    rows = []
    for dimension in dimensions:
        if dimension <= 0:
            raise ValueError("fixture dimension must be positive")
        reward_per_coordinate = float(curvature) * float(sigma)
        entropy_per_coordinate = -float(entropy_weight) / float(sigma)
        rows.append({
            "controls": int(dimension), "detectors": int(dimension),
            "reward_gradient_per_coordinate": reward_per_coordinate,
            "entropy_gradient_per_coordinate": entropy_per_coordinate,
            "reward_gradient_l1": int(dimension) * abs(reward_per_coordinate),
            "entropy_gradient_l1": int(dimension) * abs(entropy_per_coordinate),
            "per_coordinate_ratio": abs(entropy_per_coordinate / reward_per_coordinate),
        })
    return rows
