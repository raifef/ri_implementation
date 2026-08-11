"""Factorized Gaussian complete-policy sampling in normalized coordinates."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np


def action_hash(action: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(action, dtype="<f8").tobytes(order="C")).hexdigest()


def component_log_probability(
    actions: np.ndarray, mean: np.ndarray, log_scale: np.ndarray
) -> np.ndarray:
    actions = np.asarray(actions, dtype=float)
    mean = np.asarray(mean, dtype=float)
    log_scale = np.asarray(log_scale, dtype=float)
    if actions.ndim != 2 or mean.shape != log_scale.shape or actions.shape[1:] != mean.shape:
        raise ValueError("Gaussian policy shapes are inconsistent")
    inv_var = np.exp(-2.0 * log_scale)
    return -0.5 * (
        (actions - mean[None, :]) ** 2 * inv_var[None, :]
        + 2.0 * log_scale[None, :]
        + np.log(2.0 * np.pi)
    )


@dataclass(frozen=True)
class CandidateBatch:
    candidate_ids: tuple[str, ...]
    action_hashes: tuple[str, ...]
    policy_version: int
    epoch: int
    collection_mean: np.ndarray
    collection_log_scale: np.ndarray
    normalized_actions: np.ndarray
    native_actions: np.ndarray


class FactorizedGaussianPolicy:
    """Independent mean/log-scale state with immutable sampled candidates."""

    def __init__(
        self,
        initial_mean: np.ndarray,
        *,
        initial_scale: float,
        normalized_bounds: tuple[float, float],
        native_sensitivity: np.ndarray,
        seed: int,
    ) -> None:
        self.mean = np.asarray(initial_mean, dtype=float).copy()
        self.log_scale = np.full(self.mean.shape, np.log(float(initial_scale)))
        self.lower, self.upper = map(float, normalized_bounds)
        self.native_sensitivity = np.asarray(native_sensitivity, dtype=float).copy()
        if self.mean.ndim != 1 or self.native_sensitivity.shape != self.mean.shape:
            raise ValueError("policy mean and sensitivity must be aligned vectors")
        if not self.lower < self.upper or np.any(self.native_sensitivity <= 0):
            raise ValueError("invalid policy bounds or native sensitivity")
        self.rng = np.random.default_rng(seed)

    @property
    def scale(self) -> np.ndarray:
        return np.exp(self.log_scale)

    def to_native(self, normalized: np.ndarray) -> np.ndarray:
        return np.asarray(normalized, dtype=float) * self.native_sensitivity

    def to_normalized(self, native: np.ndarray) -> np.ndarray:
        return np.asarray(native, dtype=float) / self.native_sensitivity

    def sample(self, count: int, *, policy_version: int, epoch: int) -> CandidateBatch:
        if count <= 0:
            raise ValueError("candidate count must be positive")
        actions = self.mean[None, :] + self.scale[None, :] * self.rng.normal(
            size=(count, len(self.mean))
        )
        actions = np.clip(actions, self.lower, self.upper)
        ids = tuple(f"v{policy_version}:e{epoch}:c{i}" for i in range(count))
        hashes = tuple(action_hash(row) for row in actions)
        return CandidateBatch(
            ids,
            hashes,
            policy_version,
            epoch,
            self.mean.copy(),
            self.log_scale.copy(),
            actions,
            self.to_native(actions),
        )
