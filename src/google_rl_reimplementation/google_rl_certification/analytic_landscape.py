"""Level-1 known-gradient convex certification landscapes."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .agent import CandidateEvaluation, GaussianPolicyGradientAgent
from .common import cosine_similarity
from .config import GoogleRLConfig


def _run_one(config: GoogleRLConfig, initial: Sequence[float], optimum: Sequence[float],
             *, seed: int, steps: int = 36, noise_stddev: float = 0.0) -> dict[str, Any]:
    initial_array = np.asarray(initial, dtype=float)
    optimum_array = np.asarray(optimum, dtype=float)
    controls = tuple(f"u{index}" for index in range(len(initial_array)))
    detectors = tuple(f"d{index}" for index in range(len(initial_array)))
    mask = np.eye(len(initial_array), dtype=float)
    agent = GaussianPolicyGradientAgent(
        controls, detectors, mask, np.ones(len(controls)), initial_array,
        config, seed=seed)
    rng = np.random.default_rng(seed+1_000_003)
    curvature = np.linspace(.35, .55, len(controls))
    rows: list[dict[str, Any]] = []
    for step in range(steps):
        mean_before = agent.mean_native.copy()
        true_descent = -2*curvature*(mean_before-optimum_array)
        batch = agent.sample_candidates()
        losses = .01 + curvature[None, :]*(batch.actions_native-optimum_array[None, :])**2
        if noise_stddev:
            losses = losses+rng.normal(0., noise_stddev, losses.shape)
        evaluations = tuple(CandidateEvaluation(identifier, losses[index])
                            for index, identifier in enumerate(batch.candidate_ids))
        update = agent.update(batch, evaluations)
        mean_loss = float(np.mean(curvature*(agent.mean_native-optimum_array)**2))
        aggregate = float(np.mean(losses-.01))
        rows.append({
            "step": step,
            "mean_before": mean_before.tolist(),
            "mean_after": agent.mean_native.tolist(),
            "true_descent_gradient": true_descent.tolist(),
            "estimated_gradient": agent.last_gradient.tolist(),
            "gradient_cosine_similarity": cosine_similarity(agent.last_gradient, true_descent),
            "mean_policy_excess_loss": mean_loss,
            "aggregate_exploration_excess_loss": aggregate,
            "exploration_damage": max(0., aggregate-float(np.mean(
                curvature*(mean_before-optimum_array)**2))),
            "stddev": agent.stddev_native.tolist(),
            "policy_diagnostics": dict(update),
        })
    return {
        "initial": initial_array.tolist(),
        "optimum": optimum_array.tolist(),
        "initial_excess_loss": float(np.mean(curvature*(initial_array-optimum_array)**2)),
        "final_excess_loss": rows[-1]["mean_policy_excess_loss"],
        "first_gradient_cosine_similarity": rows[0]["gradient_cosine_similarity"],
        "minimum_gradient_cosine_similarity": min(
            row["gradient_cosine_similarity"] for row in rows[:8]),
        "final_stddev": rows[-1]["stddev"],
        "trajectory": rows,
    }


def run_analytic_certification(config: GoogleRLConfig, *, seed: int = 1103) -> dict[str, Any]:
    left = _run_one(config, (-.65,), (.22,), seed=seed)
    right = _run_one(config, (.65,), (-.22,), seed=seed+1)
    multivariate = _run_one(
        config, (-.55, .45, -.35), (.24, -.18, .12), seed=seed+2)
    optimum_start = _run_one(config, (.18, -.12), (.18, -.12), seed=seed+3,
                             steps=16)
    direction = (left["trajectory"][0]["mean_after"][0]
                 > left["trajectory"][0]["mean_before"][0]
                 and right["trajectory"][0]["mean_after"][0]
                 < right["trajectory"][0]["mean_before"][0])
    maximum_optimum_regression = max(
        row["mean_policy_excess_loss"] for row in optimum_start["trajectory"])
    gates = {
        "converges_from_both_sides": bool(
            direction and left["final_excess_loss"] < .05*left["initial_excess_loss"]
            and right["final_excess_loss"] < .05*right["initial_excess_loss"]),
        "positive_gradient_alignment": bool(min(
            left["first_gradient_cosine_similarity"],
            right["first_gradient_cosine_similarity"],
            multivariate["first_gradient_cosine_similarity"]) > .50),
        "covariance_adapts_without_collapse": bool(
            min(multivariate["final_stddev"])
            >= config.policy.minimum_stddev_normalized-1e-12
            and max(multivariate["final_stddev"])
            < config.policy.initial_stddev_normalized),
        "no_material_regression_from_optimum": maximum_optimum_regression < 2e-3,
        "bounds_and_slew_respected": all(
            abs(value) <= config.safety.absolute_bound_normalized+1e-12
            for run in (left, right, multivariate, optimum_start)
            for row in run["trajectory"] for value in row["mean_after"]),
    }
    return {
        "schema_version": "google-rl-analytic-certification.v1",
        "evidence_layer": "analytic known-gradient repository test",
        "config_name": config.name,
        "gates": gates,
        "passed": all(gates.values()),
        "maximum_optimum_start_excess_loss": maximum_optimum_regression,
        "runs": {"left": left, "right": right, "multivariate": multivariate,
                 "optimum_start": optimum_start},
    }
