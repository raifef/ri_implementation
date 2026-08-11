"""Clean-room masked local PPO with explicit provenance and replay semantics."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable, Mapping

import numpy as np


def action_hash(action: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(action, dtype="<f8").tobytes(order="C")).hexdigest()


def normal_log_density(actions: np.ndarray, mean: np.ndarray, log_std: np.ndarray) -> np.ndarray:
    inv_var = np.exp(-2.0 * log_std)
    return -0.5 * ((actions - mean[None, :]) ** 2 * inv_var[None, :] + 2.0 * log_std[None, :] + np.log(2 * np.pi))


def local_policy_ratios(actions: np.ndarray, mean: np.ndarray, log_std: np.ndarray,
                        old_mean: np.ndarray, old_log_std: np.ndarray, mask: np.ndarray) -> np.ndarray:
    log_delta = normal_log_density(actions, mean, log_std) - normal_log_density(actions, old_mean, old_log_std)
    return np.exp(np.clip(log_delta @ np.asarray(mask, dtype=float).T, -40.0, 40.0))


def clipped_objective_and_gradient(
    actions: np.ndarray, advantages: np.ndarray, mask: np.ndarray,
    mean: np.ndarray, log_std: np.ndarray, old_mean: np.ndarray, old_log_std: np.ndarray,
    *, clip: float, entropy_weight: float = 0.0, detector_weights: np.ndarray | None = None,
    target_std: float | None = None, target_strength: float = 0.0,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, float]]:
    """Exact gradient of the empirical masked clipped surrogate objective."""
    actions = np.asarray(actions, dtype=float)
    advantages = np.asarray(advantages, dtype=float)
    mask_f = np.asarray(mask, dtype=float)
    if actions.ndim != 2 or advantages.shape != (len(actions), mask_f.shape[0]) or actions.shape[1] != mask_f.shape[1]:
        raise ValueError("masked PPO objective shapes are inconsistent")
    ratios = local_policy_ratios(actions, mean, log_std, old_mean, old_log_std, mask_f)
    weighted_adv = advantages.copy()
    if detector_weights is not None:
        weights = np.asarray(detector_weights, dtype=float)
        if weights.shape != (mask_f.shape[0],) or np.any(weights <= 0):
            raise ValueError("detector weights must be positive and aligned")
        weighted_adv *= weights[None, :] / weights.mean()
    clipped = np.clip(ratios, 1.0 - clip, 1.0 + clip)
    objective_terms = np.minimum(ratios * weighted_adv, clipped * weighted_adv)
    active = ((weighted_adv >= 0) & (ratios <= 1.0 + clip)) | ((weighted_adv < 0) & (ratios >= 1.0 - clip))
    local_weight = np.where(active, weighted_adv * ratios, 0.0) / objective_terms.size
    control_weight = local_weight @ mask_f
    delta = actions - mean[None, :]
    inv_var = np.exp(-2.0 * log_std)
    grad_mean = np.sum(control_weight * delta * inv_var[None, :], axis=0)
    grad_log_std = np.sum(control_weight * (delta * delta * inv_var[None, :] - 1.0), axis=0)
    objective = float(np.mean(objective_terms))
    if entropy_weight:
        objective += float(entropy_weight * np.mean(log_std + 0.5 * np.log(2 * np.pi * np.e)))
        grad_log_std += entropy_weight / len(log_std)
    if target_std is not None and target_strength:
        std = np.exp(log_std)
        objective -= float(0.5 * target_strength * np.mean((std - target_std) ** 2))
        grad_log_std -= target_strength * (std - target_std) * std / len(std)
    return objective, grad_mean, grad_log_std, {
        "ratio_mean":float(ratios.mean()), "ratio_max":float(ratios.max()),
        "clip_fraction":float(1.0 - active.mean()), "off_policy_fraction":float(np.mean(np.abs(ratios - 1.0) > 1e-10)),
    }


@dataclass(frozen=True)
class CandidateBatch:
    candidate_ids: tuple[str, ...]
    action_hashes: tuple[str, ...]
    policy_version: int
    epoch: int
    regime_id: str
    collection_mean: np.ndarray
    collection_log_std: np.ndarray
    actions: np.ndarray


@dataclass(frozen=True)
class DetectorEvidence:
    candidate_id: str
    action_hash: str
    detector_counts: np.ndarray
    effective_cycles: int
    regime_id: str


@dataclass(frozen=True)
class _Replay:
    epoch: int
    regime_id: str
    batch: CandidateBatch
    advantages: np.ndarray


class MaskedGaussianPPO:
    def __init__(self, mask: np.ndarray, initial_mean: np.ndarray, choices: Mapping[str, Any], *, seed: int,
                 detector_noise_variance: np.ndarray | None = None):
        self.mask = np.asarray(mask, dtype=bool)
        self.mean = np.asarray(initial_mean, dtype=float).copy()
        if self.mask.ndim != 2 or self.mean.shape != (self.mask.shape[1],) or not self.mask.any(axis=1).all():
            raise ValueError("invalid detector-control factorization")
        self.choices = dict(choices)
        self.log_std = np.full_like(self.mean, np.log(float(choices["initial_std"])))
        self.baseline = np.zeros(self.mask.shape[0])
        self.policy_version = 0
        self.epoch = 0
        self.rng = np.random.default_rng(seed)
        self._consumed: set[tuple[int, tuple[str, ...]]] = set()
        self._replay: list[_Replay] = []
        self.detector_weights = None
        if bool(choices.get("variance_weighted", False)):
            if detector_noise_variance is None:
                raise ValueError("variance-aware weighting requires declared detector noise")
            variance = np.asarray(detector_noise_variance, dtype=float)
            self.detector_weights = 1.0 / np.sqrt(np.maximum(variance, 1e-12))

    @property
    def std(self) -> np.ndarray:
        return np.exp(self.log_std)

    def sample(self, count: int, *, regime_id: str) -> CandidateBatch:
        z = self.rng.normal(size=(count, len(self.mean)))
        actions = np.clip(self.mean[None, :] + self.std[None, :] * z,
                          -float(self.choices["absolute_bound"]), float(self.choices["absolute_bound"]))
        ids = tuple(f"v{self.policy_version}:e{self.epoch}:c{i}" for i in range(count))
        return CandidateBatch(ids, tuple(action_hash(a) for a in actions), self.policy_version, self.epoch,
                              str(regime_id), self.mean.copy(), self.log_std.copy(), actions)

    def _rates(self, batch: CandidateBatch, evidence: Iterable[DetectorEvidence]) -> np.ndarray:
        items = tuple(evidence)
        by_id = {item.candidate_id:item for item in items}
        if len(by_id) != len(items) or set(by_id) != set(batch.candidate_ids):
            raise ValueError("candidate evidence labels are missing, duplicate, or unknown")
        rows = []
        for index, cid in enumerate(batch.candidate_ids):
            item = by_id[cid]
            if item.action_hash != batch.action_hashes[index]:
                raise ValueError("candidate-to-reward action provenance mismatch")
            if item.regime_id != batch.regime_id:
                raise ValueError("incompatible replay/evidence regime")
            counts = np.asarray(item.detector_counts, dtype=float)
            if counts.shape != (self.mask.shape[0],) or item.effective_cycles <= 0:
                raise ValueError("detector evidence shape or cycle count invalid")
            if np.any(counts < 0) or np.any(counts > item.effective_cycles):
                raise ValueError("detector counts outside binomial range")
            rows.append(counts / item.effective_cycles)
        return np.asarray(rows)

    def update(self, batch: CandidateBatch, evidence: Iterable[DetectorEvidence]) -> dict[str, float]:
        key = (batch.policy_version, batch.candidate_ids)
        if batch.policy_version != self.policy_version or batch.epoch != self.epoch or key in self._consumed:
            raise ValueError("stale or already-consumed policy batch")
        rates = self._rates(batch, evidence)
        rewards = -rates
        advantages = rewards - self.baseline[None, :]
        max_age = int(self.choices["replay_epochs"])
        compatible = [x for x in self._replay if x.regime_id == batch.regime_id and batch.epoch - x.epoch <= max_age]
        rejected_incompatible = sum(x.regime_id != batch.regime_id for x in self._replay)
        samples = [*compatible, _Replay(batch.epoch, batch.regime_id, batch, advantages.copy())]
        diag: dict[str, float] = {}
        maximum_norm = 0.0
        replay_ages: list[int] = []
        for _ in range(int(self.choices["optimizer_passes"])):
            gradients = []
            for item in samples:
                age = batch.epoch - item.epoch
                replay_ages.append(age)
                decay = float(self.choices.get("replay_decay", 1.0)) ** age
                _, gm, gs, local_diag = clipped_objective_and_gradient(
                    item.batch.actions, item.advantages, self.mask, self.mean, self.log_std,
                    item.batch.collection_mean, item.batch.collection_log_std,
                    clip=float(self.choices["ppo_clip"]), entropy_weight=float(self.choices["entropy_weight"]),
                    detector_weights=self.detector_weights, target_std=self.choices.get("target_std"),
                    target_strength=float(self.choices.get("target_strength", 0.0)),
                )
                gradients.append((decay, gm, gs))
                diag = local_diag
            denom = max(sum(x[0] for x in gradients), 1e-15)
            grad_mean = sum(w * gm for w, gm, _ in gradients) / denom
            grad_scale = sum(w * gs for w, _, gs in gradients) / denom
            if bool(self.choices.get("natural_mean", False)):
                grad_mean = grad_mean * np.exp(2.0 * self.log_std)
            combined = np.concatenate([grad_mean, grad_scale])
            norm = float(np.linalg.norm(combined))
            maximum_norm = max(maximum_norm, norm)
            clip_scale = min(1.0, float(self.choices["gradient_clip"]) / max(norm, 1e-15))
            self.mean += float(self.choices["mean_learning_rate"]) * grad_mean * clip_scale
            self.mean = np.clip(self.mean, -float(self.choices["absolute_bound"]), float(self.choices["absolute_bound"]))
            self.log_std += float(self.choices["scale_learning_rate"]) * grad_scale * clip_scale
            self.log_std = np.clip(self.log_std, np.log(float(self.choices["minimum_std"])), np.log(float(self.choices["maximum_std"])))
        self.baseline += float(self.choices["baseline_learning_rate"]) * (rewards.mean(axis=0) - self.baseline)
        self._replay.append(_Replay(batch.epoch, batch.regime_id, batch, advantages.copy()))
        self._replay = self._replay[-max(1, max_age + 2):]
        self._consumed.add(key)
        self.policy_version += 1
        self.epoch += 1
        return {
            **diag, "gradient_norm_before_clip":maximum_norm, "mean_std":float(self.std.mean()),
            "replay_epochs_used":float(len(compatible)), "mean_replay_age":float(np.mean(replay_ages)),
            "incompatible_replay_rejected":float(rejected_incompatible),
            "scale_floor_fraction":float(np.mean(self.std <= float(self.choices["minimum_std"]) * (1 + 1e-12))),
            "scale_ceiling_fraction":float(np.mean(self.std >= float(self.choices["maximum_std"]) * (1 - 1e-12))),
        }
