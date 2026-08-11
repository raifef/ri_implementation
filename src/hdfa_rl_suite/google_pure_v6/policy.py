"""Factorized Gaussian complete-policy sampling with latent/applied separation."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from .units import CoordinateContract


def action_hash(action: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(action, dtype="<f8").tobytes(order="C")).hexdigest()


def component_log_probability(actions: np.ndarray, mean: np.ndarray, log_scale: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=float)
    mean = np.asarray(mean, dtype=float)
    log_scale = np.asarray(log_scale, dtype=float)
    if actions.ndim != 2 or mean.shape != log_scale.shape or actions.shape[1:] != mean.shape:
        raise ValueError("Gaussian policy shapes are inconsistent")
    z = (actions - mean[None, :]) * np.exp(-log_scale)[None, :]
    return -0.5 * z * z - log_scale[None, :] - 0.5 * np.log(2.0 * np.pi)


def gaussian_scores(actions: np.ndarray, mean: np.ndarray, log_scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    actions = np.asarray(actions, dtype=float)
    delta = actions - np.asarray(mean, dtype=float)[None, :]
    inverse_variance = np.exp(-2.0 * np.asarray(log_scale, dtype=float))
    return delta * inverse_variance[None, :], delta * delta * inverse_variance[None, :] - 1.0


@dataclass(frozen=True)
class CandidateBatch:
    candidate_ids: tuple[str, ...]
    latent_action_hashes: tuple[str, ...]
    applied_action_hashes: tuple[str, ...]
    policy_version: int
    epoch: int
    environment_time: int
    graph_version: str
    sensitivity_version: str
    collection_mean: np.ndarray
    collection_log_scale: np.ndarray
    collection_component_log_probability: np.ndarray
    latent_normalized_actions: np.ndarray
    applied_normalized_actions: np.ndarray
    applied_native_actions: np.ndarray


class FactorizedGaussianPolicy:
    def __init__(self, initial_mean: np.ndarray, coordinate_contract: CoordinateContract, *, initial_scale: float, seed: int) -> None:
        self.mean = np.asarray(initial_mean, dtype=float).copy()
        self.log_scale = np.full_like(self.mean, np.log(float(initial_scale)))
        self.coordinates = coordinate_contract
        if self.mean.shape != self.coordinates.native_offset.shape or initial_scale <= 0:
            raise ValueError("policy and coordinate shapes are inconsistent")
        self.rng = np.random.default_rng(seed)

    @property
    def scale(self) -> np.ndarray:
        return np.exp(self.log_scale)

    def sample(self, count: int, *, policy_version: int, epoch: int, environment_time: int, graph_version: str) -> CandidateBatch:
        if count <= 0:
            raise ValueError("candidate count must be positive")
        latent = self.mean[None, :] + self.scale[None, :] * self.rng.normal(size=(count, len(self.mean)))
        applied = self.coordinates.apply_bounds(latent)
        native = self.coordinates.to_native(applied)
        logp = component_log_probability(latent, self.mean, self.log_scale)
        ids = tuple(f"v{policy_version}:e{epoch}:t{environment_time}:c{i}" for i in range(count))
        return CandidateBatch(
            ids,
            tuple(action_hash(row) for row in latent),
            tuple(action_hash(row) for row in native),
            policy_version,
            epoch,
            environment_time,
            str(graph_version),
            self.coordinates.sensitivity_version,
            self.mean.copy(),
            self.log_scale.copy(),
            logp,
            latent,
            applied,
            native,
        )
