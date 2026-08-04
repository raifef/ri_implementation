"""Pure detector-local Gaussian PPO reference agent."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .baseline import DetectorBaseline
from .factor_graph import validate_mask
from .lifecycle import DetectorEvidence, PolicyLifecycle
from .policy import CandidateBatch, FactorizedGaussianPolicy
from .replay import FifoReplay, ReplayItem
from .units import CoordinateContract
from .update import legacy_v5_component_clipped_objective_and_gradient, ppo_objective_and_gradient, sgd_ascent_step


class PureGoogleV6Agent:
    def __init__(self, mask: np.ndarray, initial_mean: np.ndarray, coordinates: CoordinateContract,
                 choices: Mapping[str, Any], *, seed: int, objective_mode: str = "source_literal_ppo") -> None:
        self.mask = validate_mask(mask, len(initial_mean))
        self.choices = dict(choices)
        self.objective_mode = str(objective_mode)
        if self.objective_mode not in {"source_literal_ppo", "legacy_v5_component_clipping"}:
            raise ValueError("unknown objective mode")
        self.policy = FactorizedGaussianPolicy(initial_mean, coordinates, initial_scale=float(choices["initial_scale"]), seed=seed)
        self.baseline = DetectorBaseline(self.mask.shape[0], coefficient=float(choices["baseline_coefficient"]))
        self.replay = FifoReplay(int(choices["replay_capacity_epochs"]))
        self.lifecycle = PolicyLifecycle()

    @property
    def mean(self) -> np.ndarray:
        return self.policy.mean

    @property
    def scale(self) -> np.ndarray:
        return self.policy.scale

    def sample(self, count: int) -> CandidateBatch:
        return self.policy.sample(count, policy_version=self.lifecycle.policy_version, epoch=self.lifecycle.epoch,
                                  environment_time=self.lifecycle.environment_time, graph_version="graph-v6.1")

    def update(self, batch: CandidateBatch, evidence: tuple[DetectorEvidence, ...]) -> dict[str, Any]:
        counts, cycles = self.lifecycle.validate(batch, evidence)
        rewards = -counts / float(cycles)
        frozen_baseline = self.baseline.snapshot()
        advantages = self.baseline.advantages(rewards, frozen_baseline)
        current = ReplayItem(batch, rewards.copy(), advantages.copy(), frozen_baseline.copy())
        items = (*self.replay.items(), current)
        diagnostic: dict[str, Any] = {}
        for _ in range(int(self.choices["update_passes"])):
            mean_gradients, scale_gradients = [], []
            for item in items:
                if self.objective_mode == "source_literal_ppo":
                    _, gm, gs, diagnostic = ppo_objective_and_gradient(
                        item.batch.latent_normalized_actions, item.frozen_advantages, self.mask,
                        self.policy.mean, self.policy.log_scale,
                        item.batch.collection_component_log_probability,
                        clip=float(self.choices["ppo_clip"]), entropy_coefficient=float(self.choices["entropy_coefficient"]),
                    )
                else:
                    _, gm, gs, diagnostic = legacy_v5_component_clipped_objective_and_gradient(
                        item.batch.latent_normalized_actions, item.frozen_advantages, self.mask,
                        self.policy.mean, self.policy.log_scale, item.batch.collection_mean,
                        item.batch.collection_log_scale, clip=float(self.choices["ppo_clip"]),
                        entropy_coefficient=float(self.choices["entropy_coefficient"]),
                    )
                mean_gradients.append(gm)
                scale_gradients.append(gs)
            grad_mean = np.mean(mean_gradients, axis=0)
            grad_scale = np.mean(scale_gradients, axis=0)
            self.policy.mean[:], self.policy.log_scale[:] = sgd_ascent_step(
                self.policy.mean, self.policy.log_scale, grad_mean, grad_scale,
                mean_learning_rate=float(self.choices["mean_learning_rate"]),
                scale_learning_rate=float(self.choices["scale_learning_rate"]),
                normalized_bounds=tuple(self.choices["normalized_bounds"]),
                scale_bounds=tuple(self.choices["scale_bounds"]),
            )
        self.baseline.update(rewards)
        self.replay.append(current)
        self.lifecycle.advance()
        return {**diagnostic, "mean_gradient_norm": float(np.linalg.norm(grad_mean)),
                "scale_gradient_norm": float(np.linalg.norm(grad_scale)), "replay_batches_used": len(items)-1,
                "mean_scale": float(np.mean(self.scale)), "objective_mode": self.objective_mode}


def evidence_from_counts(batch: CandidateBatch, counts: np.ndarray, cycles: int) -> tuple[DetectorEvidence, ...]:
    return tuple(DetectorEvidence(cid, batch.applied_action_hashes[i], np.asarray(counts)[i], cycles,
                                  batch.environment_time) for i, cid in enumerate(batch.candidate_ids))
