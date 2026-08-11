"""Matched reduced-budget equivalence against the high-shot reference."""
from __future__ import annotations

from typing import Any

import numpy as np

from .agent import CandidateEvaluation, GaussianPolicyGradientAgent
from .analytic_landscape import run_analytic_certification
from .common import cosine_similarity, ranking_accuracy
from .config import GoogleRLConfig
from .drift_tracking import run_drift_tracking_certification
from .scaling_locality import run_scaling_locality
from .spoiled_policy_recovery import run_spoiled_policy_recovery
from .static_detector_landscape import make_static_landscape, run_static_detector_certification
from .steering_frequency import run_steering_frequency_sweep


TOLERANCES = {
    "minimum_reward_ranking_accuracy": .65,
    "minimum_gradient_cosine_similarity": .65,
    "maximum_harmful_update_probability": .15,
    "maximum_convergence_probability_difference": .15,
    "maximum_response_time_ratio": 2.0,
    "minimum_response_time_ratio": .5,
    "maximum_critical_period_difference_epochs": 75,
    "maximum_static_final_edr_difference": .002,
    "maximum_spoiled_final_edr_difference": .002,
    "maximum_slow_drift_edr_difference": .002,
    "maximum_exploration_damage_ratio": 2.0,
}


def _budget_probe(config: GoogleRLConfig, *, seed: int, trials: int = 24) -> dict[str, float]:
    landscape = make_static_landscape()
    truth = landscape.local_descent_gradient(np.zeros(len(landscape.control_ids)))
    rankings, cosines, harmful = [], [], []
    for trial in range(trials):
        agent = GaussianPolicyGradientAgent(
            landscape.control_ids, landscape.detector_ids, landscape.mask,
            landscape.sensitivity_scales, np.zeros(len(landscape.control_ids)),
            config, seed=seed+trial)
        rng = np.random.default_rng(seed+100_000+trial)
        batch = agent.sample_candidates()
        expected = landscape.expected_rates(batch.actions_native)
        observed = landscape.observe(
            batch.actions_native,
            config.sampling.effective_cycles_per_candidate, rng)
        agent.update(batch, tuple(CandidateEvaluation(identifier, observed[index])
                                  for index, identifier in enumerate(batch.candidate_ids)))
        rankings.append(ranking_accuracy(expected.mean(axis=1), observed.mean(axis=1)))
        cosine = cosine_similarity(agent.last_gradient, truth)
        cosines.append(cosine)
        harmful.append(float(np.dot(agent.last_gradient, truth)) <= 0.)
    return {
        "reward_ranking_accuracy": float(np.mean(rankings)),
        "gradient_cosine_similarity": float(np.mean(cosines)),
        "harmful_update_probability": float(np.mean(harmful)),
    }


def _run_suite(config: GoogleRLConfig, *, seed: int) -> dict[str, Any]:
    return {
        "analytic": run_analytic_certification(config, seed=seed+1),
        "static_detector": run_static_detector_certification(config, seed=seed+2),
        "spoiled_recovery": run_spoiled_policy_recovery(config, seed=seed+3),
        "drift_tracking": run_drift_tracking_certification(config, seed=seed+4),
        "steering_frequency": run_steering_frequency_sweep(config, seed=seed+5),
        "scaling_locality": run_scaling_locality(config, seed=seed+6),
        "budget_probe": _budget_probe(config, seed=seed+7),
    }


def _summary(suite: dict[str, Any]) -> dict[str, Any]:
    spoiled_90 = next(item for item in suite["spoiled_recovery"]["recovery_endpoints"]
                      if item["target_fraction"] == .90)
    return {
        "all_environments_passed": all(suite[name]["passed"] for name in (
            "analytic", "static_detector", "spoiled_recovery", "drift_tracking",
            "steering_frequency", "scaling_locality")),
        "analytic_final_excess_loss": suite["analytic"]["runs"]["multivariate"]["final_excess_loss"],
        "static_final_mean_policy_edr": suite["static_detector"]["final_mean_policy_edr"],
        "spoiled_final_mean_policy_edr": suite["spoiled_recovery"]["final_mean_policy_edr"],
        "spoiled_recovery_90_epochs": spoiled_90["epochs"],
        "spoiled_recovery_90_native_qec_cycles": spoiled_90["native_qec_cycles"],
        "slow_drift_mean_policy_edr": suite["drift_tracking"]["slow_drift"]["mean_policy_edr"],
        "slow_drift_aggregate_exploration_edr": suite["drift_tracking"]["slow_drift"]["aggregate_exploration_edr"],
        "step_response_epochs": suite["drift_tracking"]["step_response"]["characteristic_response_epochs"],
        "critical_steering_period_epochs": suite["steering_frequency"]["critical_period_epochs"],
        "maximum_scaling_remaining_fraction": max(
            row["affected_region_remaining_fraction"]
            for row in suite["scaling_locality"]["scaling_rows"]),
        **suite["budget_probe"],
    }


def run_budget_equivalence(high: GoogleRLConfig, reduced: GoogleRLConfig,
                           *, seed: int = 7701) -> dict[str, Any]:
    high_suite = _run_suite(high, seed=seed)
    reduced_suite = _run_suite(reduced, seed=seed)
    high_summary, reduced_summary = _summary(high_suite), _summary(reduced_suite)
    convergence_high = []
    convergence_reduced = []
    for offset in range(3):
        convergence_high.append(run_spoiled_policy_recovery(
            high, seed=seed+100+offset, epochs=40)["passed"])
        convergence_reduced.append(run_spoiled_policy_recovery(
            reduced, seed=seed+100+offset, epochs=40)["passed"])
    high_probability = float(np.mean(convergence_high))
    reduced_probability = float(np.mean(convergence_reduced))
    response_ratio = (reduced_summary["step_response_epochs"]
                      / max(high_summary["step_response_epochs"], 1))
    high_damage = (high_summary["slow_drift_aggregate_exploration_edr"]
                   - high_summary["slow_drift_mean_policy_edr"])
    reduced_damage = (reduced_summary["slow_drift_aggregate_exploration_edr"]
                      - reduced_summary["slow_drift_mean_policy_edr"])
    gates = {
        "high_shot_reference_passes_matched_suite": high_summary["all_environments_passed"],
        "reduced_passes_matched_suite": reduced_summary["all_environments_passed"],
        "reward_ranking_preserved": (
            reduced_summary["reward_ranking_accuracy"] >= TOLERANCES["minimum_reward_ranking_accuracy"]
            and reduced_summary["reward_ranking_accuracy"]
            >= high_summary["reward_ranking_accuracy"]-.15),
        "gradient_direction_preserved": (
            reduced_summary["gradient_cosine_similarity"] >= TOLERANCES["minimum_gradient_cosine_similarity"]
            and reduced_summary["gradient_cosine_similarity"]
            >= high_summary["gradient_cosine_similarity"]-.15),
        "harmful_update_probability_preserved": (
            reduced_summary["harmful_update_probability"]
            <= TOLERANCES["maximum_harmful_update_probability"]
            and reduced_summary["harmful_update_probability"]
            <= high_summary["harmful_update_probability"]+.10),
        "convergence_probability_preserved": (
            reduced_probability >= .75
            and high_probability-reduced_probability
            <= TOLERANCES["maximum_convergence_probability_difference"]),
        "final_mean_policy_quality_preserved": (
            abs(reduced_summary["static_final_mean_policy_edr"]
                - high_summary["static_final_mean_policy_edr"])
            <= TOLERANCES["maximum_static_final_edr_difference"]
            and abs(reduced_summary["spoiled_final_mean_policy_edr"]
                    - high_summary["spoiled_final_mean_policy_edr"])
            <= TOLERANCES["maximum_spoiled_final_edr_difference"]
            and abs(reduced_summary["slow_drift_mean_policy_edr"]
                    - high_summary["slow_drift_mean_policy_edr"])
            <= TOLERANCES["maximum_slow_drift_edr_difference"]),
        "response_time_in_epochs_preserved": (
            TOLERANCES["minimum_response_time_ratio"] <= response_ratio
            <= TOLERANCES["maximum_response_time_ratio"]),
        "steering_frequency_behavior_preserved": abs(
            reduced_summary["critical_steering_period_epochs"]
            - high_summary["critical_steering_period_epochs"])
        <= TOLERANCES["maximum_critical_period_difference_epochs"],
        "exploration_damage_ordering_preserved": (
            high_damage > 0 and reduced_damage > 0
            and reduced_damage <= TOLERANCES["maximum_exploration_damage_ratio"]*high_damage),
        "sparse_locality_preserved": (
            reduced_summary["maximum_scaling_remaining_fraction"]
            <= high_summary["maximum_scaling_remaining_fraction"]+.10),
    }
    equivalent = all(gates.values())
    return {
        "schema_version": "google-rl-budget-equivalence.v1",
        "evidence_layer": "matched repository surrogate certification; not hardware equivalence",
        "status": "REDUCED_BUDGET_EQUIVALENT" if equivalent else "REDUCED_BUDGET_NOT_EQUIVALENT",
        "high_shot_config": high.name,
        "reduced_budget_config": reduced.name,
        "tolerances": dict(TOLERANCES),
        "high_shot_summary": high_summary,
        "reduced_budget_summary": reduced_summary,
        "convergence_probability": {
            "high_shot": high_probability,
            "reduced_budget": reduced_probability,
            "seeds": [seed+100+offset for offset in range(3)],
        },
        "response_time_ratio": response_ratio,
        "exploration_damage_excess_edr": {
            "high_shot": high_damage, "reduced_budget": reduced_damage},
        "native_qec_cost_ratio_per_epoch": (
            reduced.native_qec_cycles_per_epoch/high.native_qec_cycles_per_epoch),
        "gates": gates,
        "passed": equivalent,
        "matched_environment_summaries": {
            "high_shot": high_summary, "reduced_budget": reduced_summary},
    }
