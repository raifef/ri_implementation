"""Level-3 deliberately spoiled calibration recovery."""
from __future__ import annotations

from typing import Any

import numpy as np

from .agent import CandidateEvaluation, GaussianPolicyGradientAgent
from .common import recovery_endpoints
from .config import GoogleRLConfig
from .static_detector_landscape import make_static_landscape


def run_spoiled_policy_recovery(config: GoogleRLConfig, *, seed: int = 3301,
                                epochs: int = 55) -> dict[str, Any]:
    landscape = make_static_landscape()
    spoiled_normalized = np.asarray([-.62, .56, -.48, 0.])
    spoiled = spoiled_normalized*landscape.sensitivity_scales
    agent = GaussianPolicyGradientAgent(
        landscape.control_ids, landscape.detector_ids, landscape.mask,
        landscape.sensitivity_scales, spoiled, config, seed=seed)
    rng = np.random.default_rng(seed+7_000_019)
    initial_rate = landscape.mean_rate(spoiled)
    oracle_rate = landscape.mean_rate(landscape.optimum_native(0.))
    denominator = max(initial_rate-oracle_rate, 1e-15)
    rows: list[dict[str, Any]] = []
    fractions: list[float] = []
    cumulative_damage_events = 0.0
    evaluation_cycles = config.mean_evaluation_qec_cycles
    for epoch in range(epochs):
        mean_before = agent.mean_native.copy()
        mean_before_rate = landscape.mean_rate(mean_before)
        batch = agent.sample_candidates()
        expected = landscape.expected_rates(batch.actions_native)
        observed = landscape.observe(
            batch.actions_native,
            config.sampling.effective_cycles_per_candidate, rng)
        agent.update(batch, tuple(
            CandidateEvaluation(identifier, observed[index])
            for index, identifier in enumerate(batch.candidate_ids)))
        expected_mean_rate = landscape.mean_rate(agent.mean_native)
        evaluation_due = epoch % config.sampling.mean_evaluation_period_epochs == 0
        evaluated_mean_rate = (float(rng.binomial(evaluation_cycles, expected_mean_rate)
                                     / evaluation_cycles)
                               if evaluation_due else None)
        aggregate_rate = float(np.mean(expected))
        damage_edr = max(0., aggregate_rate-mean_before_rate)
        cumulative_damage_events += (damage_edr
                                     * config.native_qec_cycles_per_epoch)
        fraction = min(1., max(0., (initial_rate-expected_mean_rate)/denominator))
        fractions.append(fraction)
        rows.append({
            "epoch": epoch,
            "mean_policy_edr": expected_mean_rate,
            "independent_evaluation_policy_edr": evaluated_mean_rate,
            "aggregate_exploration_edr": aggregate_rate,
            "exploration_damage_edr": damage_edr,
            "cumulative_exploration_excess_detector_events": cumulative_damage_events,
            "recovery_fraction": fraction,
            "mean_policy_native": agent.mean_native.tolist(),
            "stddev_native": agent.stddev_native.tolist(),
        })
    endpoints = recovery_endpoints(
        fractions, config.native_qec_cycles_per_epoch,
        config.sampling.candidates_per_epoch)
    final_distance = float(np.linalg.norm(
        (agent.mean_native-landscape.optimum_native(0.))/landscape.sensitivity_scales))
    reached_90 = next(item for item in endpoints if item["target_fraction"] == .90)
    gates = {
        "observed_50_percent_recovery": endpoints[0]["status"] == "reached",
        "observed_75_percent_recovery": endpoints[1]["status"] == "reached",
        "observed_90_percent_recovery": reached_90["status"] == "reached",
        "final_mean_near_optimum": final_distance < .14,
        "learned_mean_better_than_exploration": rows[-1]["mean_policy_edr"]
        < rows[-1]["aggregate_exploration_edr"],
    }
    return {
        "schema_version": "google-rl-spoiled-recovery.v1",
        "evidence_layer": "executed repository detector-likelihood surrogate",
        "config_name": config.name,
        "initial_mean_policy_edr": initial_rate,
        "oracle_edr": oracle_rate,
        "final_mean_policy_edr": rows[-1]["mean_policy_edr"],
        "final_policy_distance_normalized": final_distance,
        "recovery_endpoints": endpoints,
        "candidate_evaluations": epochs*config.sampling.candidates_per_epoch,
        "candidate_qec_cycles": epochs*config.native_qec_cycles_per_epoch,
        "cumulative_exploration_excess_detector_events": cumulative_damage_events,
        "gates": gates,
        "passed": all(gates.values()),
        "trajectory": rows,
    }
