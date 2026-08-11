"""Factorized Gaussian mathematics expressed only in direct sigma."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np

from .contracts import DIRECT_SIGMA_PARAMETERIZATION


def _parameters(mean: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.asarray(mean, dtype=float)
    sd = np.asarray(sigma, dtype=float)
    if mu.ndim != 1 or sd.shape != mu.shape or not np.all(np.isfinite(mu)):
        raise ValueError("mean and sigma must be aligned finite vectors")
    if not np.all(np.isfinite(sd)) or np.any(sd <= 0):
        raise ValueError("direct sigma must be finite and strictly positive")
    return mu, sd


def component_log_probability(actions: np.ndarray, mean: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    mu, sd = _parameters(mean, sigma)
    samples = np.asarray(actions, dtype=float)
    if samples.ndim != 2 or samples.shape[1:] != mu.shape or not np.all(np.isfinite(samples)):
        raise ValueError("Gaussian action shape is inconsistent")
    z = (samples - mu[None, :]) / sd[None, :]
    return -0.5 * z * z - np.log(sd)[None, :] - 0.5 * np.log(2.0 * np.pi)


def gaussian_scores(actions: np.ndarray, mean: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu, sd = _parameters(mean, sigma)
    samples = np.asarray(actions, dtype=float)
    if samples.ndim != 2 or samples.shape[1:] != mu.shape:
        raise ValueError("Gaussian action shape is inconsistent")
    delta = samples - mu[None, :]
    return delta / sd[None, :] ** 2, delta**2 / sd[None, :] ** 3 - 1.0 / sd[None, :]


def entropy(sigma: np.ndarray) -> float:
    sd = np.asarray(sigma, dtype=float)
    if sd.ndim != 1 or np.any(sd <= 0) or not np.all(np.isfinite(sd)):
        raise ValueError("sigma must be a positive finite vector")
    return float(np.sum(np.log(sd) + 0.5 * np.log(2.0 * np.pi * np.e)))


def action_hash(action: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(action, dtype="<f8").tobytes(order="C")).hexdigest()


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=float).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class BehaviorSnapshot:
    mean: np.ndarray
    sigma: np.ndarray
    component_log_probability: np.ndarray
    policy_version: int
    parameterization: str = DIRECT_SIGMA_PARAMETERIZATION

    def __post_init__(self) -> None:
        mu, sd = _parameters(self.mean, self.sigma)
        logp = np.asarray(self.component_log_probability, dtype=float)
        if logp.ndim != 2 or logp.shape[1:] != mu.shape:
            raise ValueError("behavior log-probability shape mismatch")
        object.__setattr__(self, "mean", _readonly(mu))
        object.__setattr__(self, "sigma", _readonly(sd))
        object.__setattr__(self, "component_log_probability", _readonly(logp))


@dataclass(frozen=True)
class CandidateBatch:
    actions: np.ndarray
    standardized_noise: np.ndarray
    behavior: BehaviorSnapshot
    candidate_ids: tuple[str, ...]
    action_hashes: tuple[str, ...]


class DirectSigmaGaussianPolicy:
    """A policy whose actual trainable scale variable is sigma itself."""

    parameterization = DIRECT_SIGMA_PARAMETERIZATION

    def __init__(self, mean: np.ndarray, sigma: np.ndarray, *, seed: int = 0) -> None:
        mu, sd = _parameters(mean, sigma)
        self.mean = mu.copy()
        self.sigma = sd.copy()
        self.rng = np.random.default_rng(int(seed))
        self.policy_version = 0

    def sample(self, count: int, *, standardized_noise: np.ndarray | None = None) -> CandidateBatch:
        if count <= 0:
            raise ValueError("candidate count must be positive")
        shape = (count, len(self.mean))
        noise = self.rng.normal(size=shape) if standardized_noise is None else np.asarray(standardized_noise, dtype=float)
        if noise.shape != shape or not np.all(np.isfinite(noise)):
            raise ValueError("standardized noise shape mismatch")
        actions = self.mean[None, :] + self.sigma[None, :] * noise
        logp = component_log_probability(actions, self.mean, self.sigma)
        behavior = BehaviorSnapshot(self.mean, self.sigma, logp, self.policy_version)
        ids = tuple(f"v{self.policy_version}:c{i}" for i in range(count))
        return CandidateBatch(_readonly(actions), _readonly(noise), behavior, ids,
                              tuple(action_hash(row) for row in actions))

    def state_dict(self, *, optimizer_state: dict[str, Any] | None = None,
                   baseline: np.ndarray | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {"schema_version": "direct-sigma-checkpoint.v1",
                                 "parameterization": self.parameterization,
                                 "mean": self.mean.tolist(), "sigma": self.sigma.tolist(),
                                 "policy_version": self.policy_version,
                                 "rng_state": self.rng.bit_generator.state}
        if optimizer_state is not None:
            result["optimizer_state"] = optimizer_state
        if baseline is not None:
            value = np.asarray(baseline, dtype=float)
            if value.ndim != 1 or not np.all(np.isfinite(value)):
                raise ValueError("checkpoint baseline must be a finite vector")
            result["baseline"] = value.tolist()
        return result

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "DirectSigmaGaussianPolicy":
        forbidden = {"log_sigma", "log_scale", "eta"}.intersection(state)
        if forbidden or state.get("parameterization") != DIRECT_SIGMA_PARAMETERIZATION:
            raise ValueError("checkpoint is not a direct-sigma checkpoint")
        policy = cls(np.asarray(state["mean"]), np.asarray(state["sigma"]))
        policy.policy_version = int(state["policy_version"])
        policy.rng.bit_generator.state = state["rng_state"]
        return policy
