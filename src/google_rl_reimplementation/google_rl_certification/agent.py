"""Factorized-Gaussian, detector-local parameter-exploring policy gradient.

The equations follow Supplementary Information section VIII.  Exact optimizer,
hyperparameters, replay schedule, hardware normalization constants, and controller
implementation are not public; every such choice is explicit in the versioned config.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping, Sequence

import numpy as np

from .config import GoogleRLConfig


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_id: str
    detector_losses: np.ndarray


@dataclass(frozen=True)
class CandidateBatch:
    epoch: int
    policy_version: int
    candidate_ids: tuple[str, ...]
    actions_native: np.ndarray
    actions_normalized: np.ndarray
    collection_mean: np.ndarray
    collection_log_stddev: np.ndarray
    standardized_perturbations: np.ndarray
    clipped_fraction: float


def _stable_candidate_id(epoch: int, version: int, index: int,
                         action: np.ndarray) -> str:
    digest = hashlib.sha256(np.asarray(action, dtype="<f8").tobytes()).hexdigest()[:16]
    return f"google-rl:e{epoch}:v{version}:k{index}:{digest}"


class GaussianPolicyGradientAgent:
    """Public-structure RL agent operating in sensitivity-normalized coordinates."""

    def __init__(self, control_ids: Sequence[str], detector_ids: Sequence[str],
                 detector_control_mask: np.ndarray,
                 sensitivity_scales: Sequence[float],
                 initial_mean_native: Sequence[float], config: GoogleRLConfig,
                 *, seed: int = 0) -> None:
        self.control_ids = tuple(control_ids)
        self.detector_ids = tuple(detector_ids)
        self.mask = np.asarray(detector_control_mask, dtype=float)
        self.scales = np.asarray(sensitivity_scales, dtype=float)
        initial = np.asarray(initial_mean_native, dtype=float)
        expected = (len(self.detector_ids), len(self.control_ids))
        if self.mask.shape != expected:
            raise ValueError(f"detector-control mask has shape {self.mask.shape}, expected {expected}")
        if not np.all((self.mask == 0) | (self.mask == 1)):
            raise ValueError("detector-control mask must be binary")
        if initial.shape != (len(self.control_ids),) or self.scales.shape != initial.shape:
            raise ValueError("control means/scales do not match the control registry")
        if np.any(~np.isfinite(self.scales)) or np.any(self.scales <= 0):
            raise ValueError("sensitivity scales must be finite and positive")
        if np.any(self.mask.sum(axis=0) == 0):
            raise ValueError("every learned control must have at least one detector factor")
        if len(set(self.control_ids)) != len(self.control_ids) or len(set(self.detector_ids)) != len(self.detector_ids):
            raise ValueError("control and detector identifiers must be unique")
        self.config = config
        self.mean = initial / self.scales
        self.log_stddev = np.full_like(self.mean, math.log(config.policy.initial_stddev_normalized))
        self.baseline = np.zeros(len(self.detector_ids), dtype=float)
        self._baseline_initialized = False
        self._rng = np.random.default_rng(seed)
        self.version = 0
        self.epoch = 0
        self.last_gradient = np.zeros_like(self.mean)
        self.last_log_stddev_gradient = np.zeros_like(self.mean)
        self.last_update: dict[str, float | int | str] = {}

    @property
    def stddev(self) -> np.ndarray:
        return np.exp(self.log_stddev)

    @property
    def mean_native(self) -> np.ndarray:
        return self.mean * self.scales

    @property
    def stddev_native(self) -> np.ndarray:
        return self.stddev * self.scales

    @property
    def covariance_native(self) -> np.ndarray:
        return np.diag(self.stddev_native ** 2)

    def sample_candidates(self) -> CandidateBatch:
        count = self.config.sampling.candidates_per_epoch
        if self.config.sampling.candidate_design == "complete_antithetic_pairs":
            base = self._rng.normal(size=(count // 2, len(self.control_ids)))
            z = np.empty((count, len(self.control_ids)), dtype=float)
            z[0::2], z[1::2] = base, -base
        else:
            z = self._rng.normal(size=(count, len(self.control_ids)))
        sigma = self.stddev
        requested_delta = z * sigma[None, :]
        bound = self.config.safety.absolute_bound_normalized
        symmetric_headroom = np.minimum(bound - self.mean, bound + self.mean)
        headroom = np.minimum(symmetric_headroom,
                              self.config.safety.candidate_slew_normalized)
        delta = np.clip(requested_delta, -headroom[None, :], headroom[None, :])
        normalized = self.mean[None, :] + delta
        actions = normalized * self.scales[None, :]
        effective_z = np.divide(delta, sigma[None, :],
                                out=np.zeros_like(delta), where=sigma[None, :] > 0)
        clipped = float(np.mean(np.abs(delta-requested_delta) > 1e-14))
        ids = tuple(_stable_candidate_id(self.epoch, self.version, index, action)
                    for index, action in enumerate(actions))
        return CandidateBatch(
            self.epoch, self.version, ids, actions, normalized,
            self.mean.copy(), self.log_stddev.copy(), effective_z, clipped,
        )

    @staticmethod
    def _log_prob_components(actions: np.ndarray, mean: np.ndarray,
                             log_stddev: np.ndarray) -> np.ndarray:
        sigma = np.exp(log_stddev)
        z = (actions - mean[None, :]) / sigma[None, :]
        return -.5*z*z - log_stddev[None, :] - .5*math.log(2*math.pi)

    def update(self, batch: CandidateBatch,
               evaluations: Sequence[CandidateEvaluation]) -> Mapping[str, float | int | str]:
        if batch.policy_version != self.version or batch.epoch != self.epoch:
            raise ValueError("stale candidate batch cannot update the current policy")
        indexed: dict[str, np.ndarray] = {}
        for item in evaluations:
            if item.candidate_id in indexed:
                raise ValueError("duplicate candidate reward")
            loss = np.asarray(item.detector_losses, dtype=float)
            if loss.shape != (len(self.detector_ids),) or np.any(~np.isfinite(loss)):
                raise ValueError("candidate detector loss has invalid shape or values")
            indexed[item.candidate_id] = loss
        expected_ids = set(batch.candidate_ids)
        if set(indexed) != expected_ids:
            missing = sorted(expected_ids-set(indexed))
            extra = sorted(set(indexed)-expected_ids)
            raise ValueError(f"candidate/reward association mismatch; missing={missing}, extra={extra}")
        losses = np.stack([indexed[item] for item in batch.candidate_ids])
        rewards = -losses
        reward_mean = rewards.mean(axis=0)
        if not self._baseline_initialized:
            self.baseline = reward_mean.copy()
            self._baseline_initialized = True
        baseline_before = self.baseline.copy()
        advantages = rewards - baseline_before[None, :]
        old_log_prob = self._log_prob_components(
            batch.actions_normalized, batch.collection_mean,
            batch.collection_log_stddev)
        degree = np.maximum(self.mask.sum(axis=0), 1.0)
        final_policy_objective = 0.0
        final_clipped = 0.0
        final_grad_norm = 0.0
        for _ in range(self.config.policy.optimizer_steps):
            log_prob = self._log_prob_components(
                batch.actions_normalized, self.mean, self.log_stddev)
            component_ratio = np.exp(np.clip(log_prob-old_log_prob, -30., 30.))
            clipped_ratio = np.clip(
                component_ratio, 1-self.config.policy.ppo_clip,
                1+self.config.policy.ppo_clip)
            active = np.isclose(component_ratio, clipped_ratio, rtol=0., atol=0.).astype(float)
            local_log_ratio = np.log(clipped_ratio) @ self.mask.T
            local_ratio = np.exp(np.clip(local_log_ratio, -12., 12.))
            local_signal = ((advantages * local_ratio) @ self.mask) / degree[None, :]
            sigma = self.stddev
            z_current = (batch.actions_normalized-self.mean[None, :]) / sigma[None, :]
            score_mean = z_current / sigma[None, :]
            score_log_stddev = z_current*z_current - 1.
            grad_mean = np.mean(local_signal*score_mean*active, axis=0)
            grad_log_stddev = np.mean(local_signal*score_log_stddev*active, axis=0)
            grad_log_stddev += self.config.policy.entropy_weight
            combined_norm = float(np.sqrt(np.sum(grad_mean**2)+np.sum(grad_log_stddev**2)))
            scale = min(1., self.config.policy.gradient_clip/max(combined_norm, 1e-15))
            grad_mean *= scale
            grad_log_stddev *= scale
            mean_delta = np.clip(
                self.config.policy.mean_learning_rate*grad_mean,
                -self.config.safety.mean_slew_normalized,
                self.config.safety.mean_slew_normalized)
            self.mean = np.clip(
                self.mean+mean_delta,
                -self.config.safety.absolute_bound_normalized,
                self.config.safety.absolute_bound_normalized)
            self.log_stddev = np.clip(
                self.log_stddev+self.config.policy.log_stddev_learning_rate*grad_log_stddev,
                math.log(self.config.policy.minimum_stddev_normalized),
                math.log(self.config.policy.maximum_stddev_normalized))
            self.last_gradient = grad_mean.copy()
            self.last_log_stddev_gradient = grad_log_stddev.copy()
            final_policy_objective = float(np.mean(advantages*local_ratio))
            final_clipped = float(np.mean(component_ratio != clipped_ratio))
            final_grad_norm = combined_norm
        self.baseline += self.config.policy.baseline_learning_rate*(reward_mean-self.baseline)
        self.version += 1
        self.epoch += 1
        entropy = float(np.sum(self.log_stddev + .5*math.log(2*math.pi*math.e)))
        baseline_loss = float(np.mean((rewards-baseline_before[None, :])**2))
        self.last_update = {
            "policy_loss": -final_policy_objective,
            "baseline_loss": baseline_loss,
            "entropy_loss": -entropy,
            "entropy": entropy,
            "gradient_norm_before_clip": final_grad_norm,
            "gradient_clipped_fraction": final_clipped,
            "candidate_clipped_fraction": batch.clipped_fraction,
            "policy_version": self.version,
            "optimizer": self.config.policy.optimizer,
            "replay_batches_used": 1,
        }
        return dict(self.last_update)

    def state_record(self) -> Mapping[str, object]:
        return {
            "policy_version": self.version,
            "mean_native": dict(zip(self.control_ids, self.mean_native.tolist())),
            "stddev_native": dict(zip(self.control_ids, self.stddev_native.tolist())),
            "baseline_by_detector": dict(zip(self.detector_ids, self.baseline.tolist())),
            "candidate_design": self.config.sampling.candidate_design,
            "optimizer": self.config.policy.optimizer,
            "entropy_weight": self.config.policy.entropy_weight,
            "gradient_clip": self.config.policy.gradient_clip,
            "replay_epochs_configured": self.config.policy.replay_epochs,
        }
