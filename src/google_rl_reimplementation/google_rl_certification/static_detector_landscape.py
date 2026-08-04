"""Level-2 sparse detector-control likelihood with exact gradients."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .agent import CandidateEvaluation, GaussianPolicyGradientAgent
from .common import cosine_similarity, ranking_accuracy
from .config import GoogleRLConfig


@dataclass(frozen=True)
class SparseDetectorLandscape:
    control_ids: tuple[str, ...]
    detector_ids: tuple[str, ...]
    mask: np.ndarray
    sensitivity_scales: np.ndarray
    irreducible_floors: np.ndarray
    quadratic_weights: np.ndarray
    coupling_vectors: np.ndarray
    coupling_weights: np.ndarray
    optimum_normalized: Callable[[float], np.ndarray]
    maximum_probability: float = .45

    def _normalized(self, action_native: np.ndarray) -> np.ndarray:
        action = np.asarray(action_native, dtype=float)
        return action/self.sensitivity_scales

    def optimum_native(self, epoch: float) -> np.ndarray:
        return self.optimum_normalized(epoch)*self.sensitivity_scales

    def expected_rates(self, actions_native: np.ndarray, epoch: float = 0.) -> np.ndarray:
        actions = np.atleast_2d(np.asarray(actions_native, dtype=float))
        delta = self._normalized(actions)-self.optimum_normalized(epoch)[None, :]
        rates = self.irreducible_floors[None, :] + (delta*delta) @ self.quadratic_weights.T
        coupled = delta @ self.coupling_vectors.T
        rates += coupled*coupled*self.coupling_weights[None, :]
        return np.clip(rates, 1e-9, self.maximum_probability)

    def observe(self, actions_native: np.ndarray, cycles: int,
                rng: np.random.Generator, epoch: float = 0.) -> np.ndarray:
        probabilities = self.expected_rates(actions_native, epoch)
        return rng.binomial(cycles, probabilities)/float(cycles)

    def mean_rate(self, action_native: np.ndarray, epoch: float = 0.) -> float:
        return float(np.mean(self.expected_rates(np.asarray(action_native)[None, :], epoch)))

    def local_descent_gradient(self, action_native: np.ndarray,
                               epoch: float = 0.) -> np.ndarray:
        delta = self._normalized(np.asarray(action_native))[None, :]
        delta -= self.optimum_normalized(epoch)[None, :]
        detector_gradients = 2*self.quadratic_weights*delta
        coupled = delta @ self.coupling_vectors.T
        detector_gradients += (2*self.coupling_weights*coupled[0])[:, None]*self.coupling_vectors
        degree = np.maximum(self.mask.sum(axis=0), 1.)
        return -(detector_gradients*self.mask).sum(axis=0)/degree


def make_static_landscape(*, optimum_scale: float = 1.0) -> SparseDetectorLandscape:
    controls = ("drive:q0", "drive:q1", "coupling:q0-q1", "inactive:q2")
    detectors = ("d:q0", "d:q1", "d:shared", "d:coupler", "d:inactive")
    mask = np.asarray([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=float)
    weights = np.asarray([
        [.42, 0., 0., 0.],
        [0., .34, 0., 0.],
        [.12, .10, 0., 0.],
        [0., 0., .38, 0.],
        [0., 0., 0., 0.],
    ])
    coupling = np.zeros_like(weights)
    coupling[2, :2] = (.65, .35)
    target = optimum_scale*np.asarray([.30, -.27, .22, 0.])
    return SparseDetectorLandscape(
        controls, detectors, mask, np.asarray([.20, 2.0, .06, 1.5]),
        np.asarray([.012, .014, .013, .016, .011]), weights, coupling,
        np.asarray([0., 0., .10, 0., 0.]), lambda _epoch: target.copy())


def run_static_detector_certification(config: GoogleRLConfig, *, seed: int = 2207,
                                      epochs: int = 36) -> dict[str, Any]:
    landscape = make_static_landscape()
    initial = np.zeros(len(landscape.control_ids))
    agent = GaussianPolicyGradientAgent(
        landscape.control_ids, landscape.detector_ids, landscape.mask,
        landscape.sensitivity_scales, initial, config, seed=seed)
    rng = np.random.default_rng(seed+9_000_001)
    fixed_rate = landscape.mean_rate(initial)
    oracle_rate = landscape.mean_rate(landscape.optimum_native(0.))
    rows: list[dict[str, Any]] = []
    for epoch in range(epochs):
        mean_before = agent.mean_native.copy()
        truth = landscape.local_descent_gradient(mean_before, epoch)
        batch = agent.sample_candidates()
        expected = landscape.expected_rates(batch.actions_native, epoch)
        observed = landscape.observe(
            batch.actions_native,
            config.sampling.effective_cycles_per_candidate, rng, epoch)
        agent.update(batch, tuple(
            CandidateEvaluation(identifier, observed[index])
            for index, identifier in enumerate(batch.candidate_ids)))
        mean_rate = landscape.mean_rate(agent.mean_native, epoch)
        candidate_rate = float(np.mean(expected))
        rows.append({
            "epoch": epoch,
            "mean_native": agent.mean_native.tolist(),
            "stddev_native": agent.stddev_native.tolist(),
            "mean_policy_edr": mean_rate,
            "aggregate_exploration_edr": candidate_rate,
            "exploration_damage_edr": max(0., candidate_rate-landscape.mean_rate(mean_before, epoch)),
            "gradient_cosine_similarity": cosine_similarity(agent.last_gradient, truth),
            "reward_ranking_accuracy": ranking_accuracy(expected.mean(axis=1), observed.mean(axis=1)),
        })
    active_error = np.abs(
        agent.mean_native[:3]-landscape.optimum_native(0.)[:3]) / landscape.sensitivity_scales[:3]
    inactive_motion = abs(agent.mean_native[3]-initial[3])/landscape.sensitivity_scales[3]
    mean_cosine = float(np.mean([row["gradient_cosine_similarity"] for row in rows[:12]]))
    final_rate = rows[-1]["mean_policy_edr"]
    gates = {
        "mean_policy_converges": final_rate-oracle_rate < .25*(fixed_rate-oracle_rate),
        "positive_mean_gradient_alignment": mean_cosine > .55,
        "inactive_region_stable": inactive_motion < .035,
        "unequal_sensitivities_normalized": float(np.max(active_error)) < .16,
        "covariance_sensible": bool(
            np.all(agent.stddev >= config.policy.minimum_stddev_normalized-1e-12)
            and np.mean(agent.stddev[:3]) < config.policy.initial_stddev_normalized),
        "improves_over_fixed": final_rate < fixed_rate,
    }
    return {
        "schema_version": "google-rl-static-detector-certification.v1",
        "evidence_layer": "executed sparse detector-likelihood surrogate",
        "config_name": config.name,
        "fixed_edr": fixed_rate,
        "oracle_edr": oracle_rate,
        "final_mean_policy_edr": final_rate,
        "mean_gradient_cosine_similarity": mean_cosine,
        "inactive_region_motion_normalized": inactive_motion,
        "final_policy_distance_normalized": active_error.tolist(),
        "gates": gates,
        "passed": all(gates.values()),
        "trajectory": rows,
    }
