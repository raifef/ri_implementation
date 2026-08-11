"""Numerical primitives used by the V20 diagnostics and synthetic tests."""
from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any, Iterable

import numpy as np


def fundamental_improvement(gain: float, phase_lag_radians: float) -> float:
    """Complete-period quadratic improvement for a unit sinusoidal target."""
    gain = float(gain)
    return float(2.0 * gain * math.cos(float(phase_lag_radians)) - gain**2)


def candidate_snr(values: Iterable[float]) -> float:
    data = np.asarray(list(values), dtype=float)
    if data.size < 2:
        return float("nan")
    standard_deviation = float(np.std(data, ddof=1))
    if standard_deviation == 0:
        return float("inf") if float(np.mean(data)) != 0 else 0.0
    return abs(float(np.mean(data))) / standard_deviation


def batch_snr(values: Iterable[float]) -> float:
    data = np.asarray(list(values), dtype=float)
    return float(math.sqrt(data.size) * candidate_snr(data))


def wrong_sign_probability(snr: float) -> float:
    if math.isinf(float(snr)):
        return 0.0
    return float(NormalDist().cdf(-float(snr)))


def cosine_alignment(left: np.ndarray, right: np.ndarray) -> float | None:
    a = np.asarray(left, dtype=float).reshape(-1)
    b = np.asarray(right, dtype=float).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denominator) if denominator > 0 else None


def project_shared_subspace(gradient: np.ndarray) -> np.ndarray:
    """Orthogonally project a 41-vector onto the public shared-control mode."""
    value = np.asarray(gradient, dtype=float)
    if value.ndim != 1 or value.size == 0:
        raise ValueError("gradient must be a nonempty vector")
    return np.full_like(value, float(np.mean(value)))


def perturbation_rank(noises: np.ndarray) -> dict[str, int]:
    matrix = np.asarray(noises, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("perturbations must be a matrix")
    return {
        "raw_rank": int(np.linalg.matrix_rank(matrix)),
        "centered_rank": int(np.linalg.matrix_rank(matrix - np.mean(matrix, axis=0))),
        "candidate_count": int(matrix.shape[0]),
        "parameter_count": int(matrix.shape[1]),
    }


def rowspace_overlap(noises: np.ndarray, direction: np.ndarray) -> float:
    matrix = np.asarray(noises, dtype=float)
    vector = np.asarray(direction, dtype=float).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[1] != vector.size:
        raise ValueError("row space and direction are not aligned")
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return 0.0
    _, singular, right = np.linalg.svd(matrix, full_matrices=False)
    rank = int(np.sum(singular > np.finfo(float).eps * max(matrix.shape) * singular[0])) \
        if singular.size else 0
    if rank == 0:
        return 0.0
    projection = right[:rank] @ (vector / norm)
    return float(np.sum(projection**2))


def update_efficiency(delta_mean: np.ndarray, update_direction: np.ndarray,
                      beneficial_direction: np.ndarray, learning_rate: float) -> float | None:
    delta = np.asarray(delta_mean, dtype=float)
    supplied = np.asarray(update_direction, dtype=float)
    beneficial = np.asarray(beneficial_direction, dtype=float)
    denominator = float(learning_rate) * float(beneficial @ supplied)
    return float(beneficial @ delta / denominator) if abs(denominator) > 1e-15 else None


def decompose_mean_trace(trace: np.ndarray, epochs: np.ndarray, frequency: float,
                         *, harmonics: int = 4) -> dict[str, np.ndarray]:
    """Decompose a coordinate trace into shared DC/frequency/harmonic/residual modes."""
    values = np.asarray(trace, dtype=float)
    time = np.asarray(epochs, dtype=float)
    if values.ndim != 2 or len(time) != values.shape[0] or harmonics < 2:
        raise ValueError("invalid trace decomposition inputs")
    scalar = np.mean(values, axis=1)
    columns = [np.ones_like(time)]
    labels = ["dc"]
    for order in range(1, harmonics + 1):
        phase = 2.0 * np.pi * float(frequency) * float(order) * time
        columns.extend([np.sin(phase), np.cos(phase)])
        labels.extend([f"sin_{order}", f"cos_{order}"])
    design = np.column_stack(columns)
    coefficients = np.linalg.lstsq(design, scalar, rcond=None)[0]
    fitted = {label: design[:, index] * coefficients[index]
              for index, label in enumerate(labels)}
    dc_scalar = fitted["dc"]
    fundamental_scalar = fitted["sin_1"] + fitted["cos_1"]
    harmonic_scalar = sum((fitted[f"sin_{order}"] + fitted[f"cos_{order}"]
                           for order in range(2, harmonics + 1)),
                          np.zeros_like(scalar))
    transient_scalar = scalar - dc_scalar - fundamental_scalar - harmonic_scalar
    ones = np.ones((1, values.shape[1]))
    result = {
        "dc": dc_scalar[:, None] * ones,
        "fundamental": fundamental_scalar[:, None] * ones,
        "harmonic": harmonic_scalar[:, None] * ones,
        "transient": transient_scalar[:, None] * ones,
        "orthogonal": values - scalar[:, None] * ones,
        "full": values.copy(),
        "coefficients": coefficients,
    }
    reconstructed = sum(result[key] for key in (
        "dc", "fundamental", "harmonic", "transient", "orthogonal"))
    if not np.allclose(reconstructed, values, rtol=0, atol=2e-12):
        raise RuntimeError("mean-trace decomposition does not conserve the trace")
    return result


def quadratic_component_accounting(error_components: dict[str, np.ndarray],
                                   weights: np.ndarray) -> dict[str, Any]:
    """Return self and pair cross costs that exactly conserve diagonal quadratic cost."""
    keys = list(error_components)
    arrays = {key: np.asarray(error_components[key], dtype=float) for key in keys}
    hessian = np.asarray(weights, dtype=float)
    shape = next(iter(arrays.values())).shape
    if any(value.shape != shape for value in arrays.values()):
        raise ValueError("quadratic components have inconsistent shapes")
    if hessian.shape not in {(shape[1],), shape}:
        raise ValueError("quadratic weights must be coordinate or trace aligned")
    h = hessian[None, :] if hessian.ndim == 1 else hessian
    self_costs = {key: float(np.mean(np.sum(h * value**2, axis=1)))
                  for key, value in arrays.items()}
    cross_costs: dict[str, float] = {}
    for left_index, left in enumerate(keys):
        for right in keys[left_index + 1:]:
            cross_costs[f"{left}__x__{right}"] = float(
                2.0 * np.mean(np.sum(h * arrays[left] * arrays[right], axis=1)))
    total = float(np.mean(np.sum(h * sum(arrays.values())**2, axis=1)))
    conserved = float(sum(self_costs.values()) + sum(cross_costs.values()))
    return {
        "total": total,
        "self_costs": self_costs,
        "cross_costs": cross_costs,
        "cross_term_total": float(sum(cross_costs.values())),
        "conservation_error": conserved - total,
    }


def fixed_budget_equal(rows: Iterable[dict[str, Any]]) -> bool:
    budgets = {int(row["candidates"]) * int(row["cycles_per_candidate"]) for row in rows}
    return len(budgets) == 1
