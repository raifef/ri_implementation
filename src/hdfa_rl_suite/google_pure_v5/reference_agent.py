"""Independent composition of the public detector-driven Google-style agent."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .baseline import DetectorBaseline
from .factor_graph import validate_mask
from .lifecycle import DetectorEvidence, PolicyLifecycle
from .policy import CandidateBatch, FactorizedGaussianPolicy
from .replay import FifoReplay, ReplayItem
from .reward import detector_rewards
from .update import clipped_objective_and_gradient, sgd_ascent_step


class PureGoogleReferenceAgent:
    """No predictor, model state, controller selection, or residual-control path."""

    def __init__(
        self,
        mask: np.ndarray,
        initial_mean: np.ndarray,
        native_sensitivity: np.ndarray,
        choices: Mapping[str, Any],
        *,
        seed: int,
    ) -> None:
        self.mask = validate_mask(mask, len(initial_mean))
        self.choices = dict(choices)
        self.lifecycle = PolicyLifecycle()
        self.policy = FactorizedGaussianPolicy(
            initial_mean,
            initial_scale=float(self.choices["initial_scale"]),
            normalized_bounds=tuple(self.choices["normalized_bounds"]),
            native_sensitivity=native_sensitivity,
            seed=seed,
        )
        self.baseline = DetectorBaseline(
            self.mask.shape[0], learning_rate=float(self.choices["baseline_learning_rate"])
        )
        self.replay = FifoReplay(int(self.choices["replay_capacity_epochs"]))

    @property
    def mean(self) -> np.ndarray:
        return self.policy.mean

    @property
    def log_scale(self) -> np.ndarray:
        return self.policy.log_scale

    @property
    def scale(self) -> np.ndarray:
        return self.policy.scale

    def sample(self, count: int) -> CandidateBatch:
        return self.policy.sample(
            count,
            policy_version=self.lifecycle.policy_version,
            epoch=self.lifecycle.epoch,
        )

    def update(
        self, batch: CandidateBatch, evidence: tuple[DetectorEvidence, ...]
    ) -> dict[str, float]:
        counts = self.lifecycle.validate(batch, evidence)
        cycles = evidence[0].effective_cycles
        rewards = detector_rewards(counts, cycles)
        frozen_baseline = self.baseline.snapshot()
        advantages = self.baseline.advantages(rewards, frozen=frozen_baseline)
        current = ReplayItem(batch, advantages.copy())
        update_items = (*self.replay.items(), current)
        diagnostic: dict[str, float] = {}
        objectives: list[float] = []
        for _ in range(int(self.choices["update_passes"])):
            mean_gradients = []
            scale_gradients = []
            for item in update_items:
                objective, grad_mean, grad_scale, diagnostic = clipped_objective_and_gradient(
                    item.batch.normalized_actions,
                    item.advantages,
                    self.mask,
                    self.policy.mean,
                    self.policy.log_scale,
                    item.batch.collection_mean,
                    item.batch.collection_log_scale,
                    clip=float(self.choices["ppo_clip"]),
                    entropy_coefficient=float(self.choices["entropy_coefficient"]),
                )
                objectives.append(objective)
                mean_gradients.append(grad_mean)
                scale_gradients.append(grad_scale)
            grad_mean = np.mean(mean_gradients, axis=0)
            grad_scale = np.mean(scale_gradients, axis=0)
            new_mean, new_log_scale = sgd_ascent_step(
                self.policy.mean,
                self.policy.log_scale,
                grad_mean,
                grad_scale,
                mean_learning_rate=float(self.choices["mean_learning_rate"]),
                scale_learning_rate=float(self.choices["scale_learning_rate"]),
                bounds=tuple(self.choices["normalized_bounds"]),
                scale_bounds=tuple(self.choices["scale_bounds"]),
            )
            self.policy.mean[:] = new_mean
            self.policy.log_scale[:] = new_log_scale
        self.baseline.update(rewards)
        self.replay.append(current)
        self.lifecycle.advance()
        return {
            **diagnostic,
            "objective": float(np.mean(objectives)),
            "mean_gradient_norm": float(np.linalg.norm(grad_mean)),
            "scale_gradient_norm": float(np.linalg.norm(grad_scale)),
            "baseline_loss_before_update": self.baseline.loss(rewards, value=frozen_baseline),
            "replay_batches_used": float(len(update_items) - 1),
            "mean_scale": float(np.mean(self.policy.scale)),
        }

    def reset(self, initial_mean: np.ndarray | None = None) -> None:
        if initial_mean is not None:
            value = np.asarray(initial_mean, dtype=float)
            if value.shape != self.policy.mean.shape:
                raise ValueError("reset mean shape mismatch")
            self.policy.mean[:] = value
        self.policy.log_scale.fill(np.log(float(self.choices["initial_scale"])))
        self.baseline.reset()
        self.replay.reset()
        self.lifecycle.reset()


def evidence_from_counts(
    batch: CandidateBatch, counts: np.ndarray, effective_cycles: int
) -> tuple[DetectorEvidence, ...]:
    values = np.asarray(counts)
    if values.shape[0] != len(batch.candidate_ids):
        raise ValueError("candidate/count row mismatch")
    return tuple(
        DetectorEvidence(candidate_id, batch.action_hashes[index], values[index], effective_cycles)
        for index, candidate_id in enumerate(batch.candidate_ids)
    )
