"""Source-traceable reference implementation of the public policy-gradient path.

Source map (Sivak et al., arXiv:2511.08493v4 Supplement):

* Eqs. 10-11: diagonal Gaussian parameter-exploring policy and local reward.
* Eqs. 12-16: detector-vector baseline, advantages, and local policy ratios.
* Eqs. 17-20: empirical PPO-style clipped local objective.
* Eqs. 21-22: baseline MSE, entropy regularization, and total objective.
* Algorithm 1: sample complete policies, acquire detector outcomes, then update.

Numerical hyperparameters absent from the source live in
``configs/google_rl/source_unspecified_choices.yaml``.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

import numpy as np

from .config import ReferenceConfig


@dataclass(frozen=True)
class CandidateBatch:
    candidate_ids: tuple[str, ...]
    action_hashes: tuple[str, ...]
    policy_version: int
    epoch: int
    regime_id: str
    collection_mean: np.ndarray
    collection_log_stddev: np.ndarray
    actions_normalized: np.ndarray
    actions_native: np.ndarray
    standardized_perturbations: np.ndarray


@dataclass(frozen=True)
class DetectorEvidence:
    candidate_id: str
    candidate_action_hash: str
    detector_event_counts: np.ndarray
    effective_qec_cycles: int
    regime_id: str


@dataclass(frozen=True)
class _ReplaySample:
    epoch: int
    regime_id: str
    batch: CandidateBatch
    advantages: np.ndarray


def _normal_log_density(action: np.ndarray, mean: np.ndarray, log_stddev: np.ndarray) -> np.ndarray:
    variance = np.exp(2.0 * log_stddev)
    return -0.5 * (((action - mean) ** 2) / variance + 2.0 * log_stddev + np.log(2.0 * np.pi))


def action_hash(action_native: np.ndarray) -> str:
    canonical = np.asarray(action_native, dtype="<f8").tobytes(order="C")
    return hashlib.sha256(canonical).hexdigest()


def local_policy_ratios(
    actions: np.ndarray,
    mean: np.ndarray,
    log_stddev: np.ndarray,
    collection_mean: np.ndarray,
    collection_log_stddev: np.ndarray,
    detector_control_mask: np.ndarray,
) -> np.ndarray:
    """Return Supplement Eq. 16 local ratios, never a global-policy ratio."""
    mask = np.asarray(detector_control_mask, dtype=bool)
    current = _normal_log_density(actions, mean, log_stddev)
    old = _normal_log_density(actions, collection_mean, collection_log_stddev)
    local_log_ratio = (current[:, None, :] - old[:, None, :]) * mask[None, :, :]
    return np.exp(np.clip(local_log_ratio.sum(axis=2), -40.0, 40.0))


class ReferenceAgent:
    """Transparent, non-optimized implementation of the paper's local objective."""

    def __init__(
        self,
        control_ids: Iterable[str],
        detector_ids: Iterable[str],
        detector_control_mask: np.ndarray,
        sensitivity_native_per_normalized: np.ndarray,
        initial_mean_native: np.ndarray,
        config: ReferenceConfig,
        *,
        seed: int,
    ) -> None:
        self.control_ids = tuple(control_ids)
        self.detector_ids = tuple(detector_ids)
        if len(set(self.control_ids)) != len(self.control_ids) or len(set(self.detector_ids)) != len(self.detector_ids):
            raise ValueError("control and detector identifiers must be unique")
        self.mask = np.asarray(detector_control_mask, dtype=bool)
        expected = (len(self.detector_ids), len(self.control_ids))
        if self.mask.shape != expected:
            raise ValueError(f"detector-control mask has shape {self.mask.shape}, expected {expected}")
        if not self.mask.any(axis=1).all():
            raise ValueError("each detector needs at least one active local control factor")
        self.sensitivity = np.asarray(sensitivity_native_per_normalized, dtype=float).copy()
        initial = np.asarray(initial_mean_native, dtype=float)
        if initial.shape != self.sensitivity.shape or initial.shape != (len(self.control_ids),):
            raise ValueError("initial mean and sensitivity shapes must match controls")
        if not np.isfinite(self.sensitivity).all() or np.any(self.sensitivity <= 0):
            raise ValueError("sensitivities must be finite, positive native-units per normalized unit")
        self.config = config
        self.mean = initial / self.sensitivity
        self.log_stddev = np.full_like(self.mean, np.log(config.agent.initial_stddev_normalized))
        self.baseline = np.zeros(len(self.detector_ids), dtype=float)
        self.policy_version = 0
        self.epoch = 0
        self._rng = np.random.default_rng(seed)
        self._consumed: set[tuple[int, tuple[str, ...]]] = set()
        self._replay: list[_ReplaySample] = []

    @property
    def stddev(self) -> np.ndarray:
        return np.exp(self.log_stddev)

    @property
    def mean_native(self) -> np.ndarray:
        """Learned-mean policy; never an average of candidate outcomes."""
        return self.mean * self.sensitivity

    def sample_candidates(self, *, regime_id: str) -> CandidateBatch:
        """Sample complete independent policies (Supplement Algorithm 1)."""
        count = self.config.sampling.candidates_per_epoch
        z = self._rng.normal(size=(count, len(self.control_ids)))
        actions = self.mean + self.stddev * z
        bound = self.config.agent.absolute_bound_normalized
        actions = np.clip(actions, -bound, bound)
        ids = tuple(f"v{self.policy_version}:e{self.epoch}:c{i}" for i in range(count))
        native = actions * self.sensitivity[None, :]
        return CandidateBatch(
            ids,
            tuple(action_hash(row) for row in native),
            self.policy_version,
            self.epoch,
            str(regime_id),
            self.mean.copy(),
            self.log_stddev.copy(),
            actions,
            native,
            z,
        )

    def _aligned_rates(self, batch: CandidateBatch, evidence: Iterable[DetectorEvidence]) -> np.ndarray:
        items = tuple(evidence)
        ids = [item.candidate_id for item in items]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate candidate reward labels")
        by_id = {item.candidate_id: item for item in items}
        if set(by_id) != set(batch.candidate_ids):
            raise ValueError("missing, unknown, or shuffled-without-label candidate evidence")
        rows = []
        for index, candidate_id in enumerate(batch.candidate_ids):
            item = by_id[candidate_id]
            if item.candidate_action_hash != batch.action_hashes[index]:
                raise ValueError("candidate label/action provenance mismatch")
            if item.regime_id != batch.regime_id:
                raise ValueError("replay/evidence from an incompatible drift regime")
            counts = np.asarray(item.detector_event_counts, dtype=float)
            if counts.shape != (len(self.detector_ids),) or item.effective_qec_cycles <= 0:
                raise ValueError("detector evidence shape or cycle count is invalid")
            if np.any(counts < 0) or np.any(counts > item.effective_qec_cycles):
                raise ValueError("detector event counts are outside the binomial range")
            rows.append(counts / item.effective_qec_cycles)
        return np.asarray(rows)

    def update(self, batch: CandidateBatch, evidence: Iterable[DetectorEvidence]) -> dict[str, float]:
        """Apply masked clipped policy-gradient and vector-baseline updates."""
        key = (batch.policy_version, batch.candidate_ids)
        if batch.policy_version != self.policy_version or batch.epoch != self.epoch or key in self._consumed:
            raise ValueError("stale or already-consumed candidate batch")
        rates = self._aligned_rates(batch, evidence)
        rewards = -rates  # Supplement Eq. 11: local reward is negative detector outcome.
        advantages = rewards - self.baseline[None, :]  # Eqs. 12-14.
        clip = self.config.agent.ppo_clip
        detector_degree = np.maximum(self.mask.sum(axis=0), 1)
        replay = [sample for sample in self._replay if (
            sample.regime_id == batch.regime_id
            and batch.epoch - sample.epoch <= self.config.agent.replay_max_regime_age_epochs
        )]
        samples = [*replay, _ReplaySample(batch.epoch, batch.regime_id, batch, advantages.copy())]
        maximum_norm = 0.0
        final_ratios = np.ones((len(batch.candidate_ids), len(self.detector_ids)))
        for _optimizer_step in range(self.config.agent.optimizer_steps):
            mean_gradients = []
            scale_gradients = []
            for sample in samples:
                local_batch = sample.batch
                local_advantages = sample.advantages
                ratios = local_policy_ratios(
                    local_batch.actions_normalized,
                    self.mean,
                    self.log_stddev,
                    local_batch.collection_mean,
                    local_batch.collection_log_stddev,
                    self.mask,
                )
                if local_batch is batch:
                    final_ratios = ratios
                use_gradient = ((local_advantages >= 0) & (ratios <= 1 + clip)) | ((local_advantages < 0) & (ratios >= 1 - clip))
                weights = np.where(use_gradient, local_advantages * ratios, 0.0)
                delta = local_batch.actions_normalized - self.mean[None, :]
                inv_var = np.exp(-2.0 * self.log_stddev)
                score_mean = delta * inv_var[None, :]
                score_log_std = delta * delta * inv_var[None, :] - 1.0
                local_weights = np.einsum("nd,dc->nc", weights, self.mask.astype(float)) / detector_degree[None, :]
                mean_gradients.append(np.mean(local_weights * score_mean, axis=0))
                scale_gradients.append(np.mean(local_weights * score_log_std, axis=0))
            grad_mean = np.mean(mean_gradients, axis=0)
            grad_log_std = np.mean(scale_gradients, axis=0)
            entropy_scale = 1.0
            if self.config.agent.entropy_scale_mode == "mean_absolute_advantage":
                entropy_scale = max(float(np.mean(np.abs(advantages))), 1e-8)
            grad_log_std += self.config.agent.entropy_weight * entropy_scale  # Eqs. 21-22.
            combined = np.concatenate([grad_mean, grad_log_std])
            norm = float(np.linalg.norm(combined))
            maximum_norm = max(maximum_norm, norm)
            scale = min(1.0, self.config.agent.gradient_clip / max(norm, 1e-15))
            self.mean += self.config.agent.mean_learning_rate * grad_mean * scale
            self.mean = np.clip(
                self.mean,
                -self.config.agent.absolute_bound_normalized,
                self.config.agent.absolute_bound_normalized,
            )
            self.log_stddev += self.config.agent.log_stddev_learning_rate * grad_log_std * scale
            self.log_stddev = np.clip(
                self.log_stddev,
                np.log(self.config.agent.minimum_stddev_normalized),
                np.log(self.config.agent.maximum_stddev_normalized),
            )
        self.baseline += self.config.agent.baseline_learning_rate * (rewards.mean(axis=0) - self.baseline)
        self._replay.append(_ReplaySample(batch.epoch, batch.regime_id, batch, advantages.copy()))
        self._replay = self._replay[-self.config.agent.replay_capacity_epochs:]
        self._consumed.add(key)
        self.policy_version += 1
        self.epoch += 1
        return {
            "mean_reward": float(rewards.mean()),
            "gradient_norm_before_clip": maximum_norm,
            "mean_local_policy_ratio": float(final_ratios.mean()),
            "mean_stddev_normalized": float(self.stddev.mean()),
            "compatible_replay_epochs_used": float(len(replay)),
        }
