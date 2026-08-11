"""Levels 3-5: no-drift, slow-drift and controlled-step validation."""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from .agent import CandidateEvaluation, GaussianPolicyGradientAgent
from .config import GoogleRLConfig
from .static_detector_landscape import SparseDetectorLandscape


def one_control_landscape(optimum: Callable[[float], float], *, curvature: float,
                          floor: float = .012) -> SparseDetectorLandscape:
    return SparseDetectorLandscape(
        ("drive:q0",), ("d:q0",), np.ones((1, 1)), np.ones(1),
        np.asarray([floor]), np.asarray([[curvature]]), np.zeros((1, 1)),
        np.zeros(1), lambda epoch: np.asarray([optimum(epoch)]))


def _run_agent(config: GoogleRLConfig, landscape: SparseDetectorLandscape,
               epochs: int, *, seed: int) -> dict[str, Any]:
    agent = GaussianPolicyGradientAgent(
        landscape.control_ids, landscape.detector_ids, landscape.mask,
        landscape.sensitivity_scales, landscape.optimum_native(0.), config,
        seed=seed)
    rng = np.random.default_rng(seed+8_000_009)
    rows: list[dict[str, Any]] = []
    for epoch in range(epochs):
        mean_before_rate = landscape.mean_rate(agent.mean_native, epoch)
        batch = agent.sample_candidates()
        expected = landscape.expected_rates(batch.actions_native, epoch)
        observed = landscape.observe(
            batch.actions_native,
            config.sampling.effective_cycles_per_candidate, rng, epoch)
        agent.update(batch, tuple(CandidateEvaluation(identifier, observed[index])
                                  for index, identifier in enumerate(batch.candidate_ids)))
        optimum = landscape.optimum_native(epoch)
        mean_rate = landscape.mean_rate(agent.mean_native, epoch)
        candidate_rate = float(np.mean(expected))
        rows.append({
            "epoch": epoch,
            "optimum": float(optimum[0]),
            "mean_policy": float(agent.mean_native[0]),
            "mean_policy_edr": mean_rate,
            "aggregate_exploration_edr": candidate_rate,
            "exploration_damage_edr": max(0., candidate_rate-mean_before_rate),
            "fixed_edr": landscape.mean_rate(np.zeros(1), epoch),
            "oracle_edr": landscape.mean_rate(optimum, epoch),
            "stddev": float(agent.stddev_native[0]),
        })
    return {"agent": agent, "trajectory": rows}


def run_no_drift_validation(config: GoogleRLConfig, *, seed: int = 4401,
                            epochs: int = 45) -> dict[str, Any]:
    landscape = one_control_landscape(lambda _epoch: 0., curvature=.08)
    run = _run_agent(config, landscape, epochs, seed=seed)
    rows = run["trajectory"]
    floor = landscape.mean_rate(np.zeros(1), 0.)
    maximum_excess = max(row["mean_policy_edr"]-floor for row in rows)
    return {
        "maximum_mean_policy_excess_edr": maximum_excess,
        "irreducible_floor": floor,
        "passed": maximum_excess < 8e-4,
        "trajectory": rows,
    }


def run_slow_drift_validation(config: GoogleRLConfig, *, seed: int = 4403,
                              epochs: int = 360, period_epochs: int = 300) -> dict[str, Any]:
    landscape = one_control_landscape(
        lambda epoch: .55*math.sin(2*math.pi*epoch/period_epochs), curvature=.08)
    run = _run_agent(config, landscape, epochs, seed=seed)
    rows = run["trajectory"]
    periodic_action = 0.0
    periodic_rates = []
    periodic_period = 60
    for epoch in range(epochs):
        if epoch % periodic_period == 0:
            periodic_action = float(landscape.optimum_native(epoch)[0])
        periodic_rates.append(landscape.mean_rate(np.asarray([periodic_action]), epoch))
    warm = max(30, epochs//6)
    mean_rate = float(np.mean([row["mean_policy_edr"] for row in rows[warm:]]))
    exploratory_rate = float(np.mean([row["aggregate_exploration_edr"] for row in rows[warm:]]))
    fixed_rate = float(np.mean([row["fixed_edr"] for row in rows[warm:]]))
    oracle_rate = float(np.mean([row["oracle_edr"] for row in rows[warm:]]))
    periodic_rate = float(np.mean(periodic_rates[warm:]))
    irreducible_floor = min(row["oracle_edr"] for row in rows)
    gates = {
        "fixed_degrades_under_persistent_drift": fixed_rate > oracle_rate+2e-3,
        "oracle_removes_controllable_degradation": (
            oracle_rate-irreducible_floor
            < .10*max(fixed_rate-irreducible_floor, 1e-15)),
        "rl_materially_outperforms_fixed": mean_rate < .75*fixed_rate,
        "mean_better_than_exploratory_aggregate": mean_rate < exploratory_rate,
        "approaches_controllable_optimum": mean_rate-oracle_rate < .25*(fixed_rate-oracle_rate),
        "detector_statistics_not_saturated": max(row["aggregate_exploration_edr"] for row in rows) < .20,
    }
    return {
        "schema_version": "google-rl-slow-drift.v1",
        "config_name": config.name,
        "period_epochs": period_epochs,
        "fixed_edr": fixed_rate,
        "periodic_recalibration_edr": periodic_rate,
        "oracle_edr": oracle_rate,
        "mean_policy_edr": mean_rate,
        "aggregate_exploration_edr": exploratory_rate,
        "gates": gates,
        "passed": all(gates.values()),
        "trajectory": rows,
    }


def run_step_response(config: GoogleRLConfig, *, seed: int = 4409,
                      epochs: int = 220, onset_epoch: int = 10) -> dict[str, Any]:
    amplitude = .55
    landscape = one_control_landscape(
        lambda epoch: 0. if epoch < onset_epoch else amplitude, curvature=.015)
    run = _run_agent(config, landscape, epochs, seed=seed)
    rows = run["trajectory"]
    response_level = amplitude*(1-math.exp(-1))
    reached = next((row["epoch"]-onset_epoch+1 for row in rows[onset_epoch:]
                    if row["mean_policy"] >= response_level), None)
    final_mean = float(np.mean([row["mean_policy"] for row in rows[-20:]]))
    gates = {
        "correct_response_direction": rows[onset_epoch+1]["mean_policy"] > rows[onset_epoch]["mean_policy"],
        "characteristic_response_observed": reached is not None,
        "stable_final_tracking": abs(final_mean-amplitude) < .12,
        "plausible_public_scale_without_exact_reproduction_claim": reached is not None and 5 <= reached <= 260,
    }
    return {
        "schema_version": "google-rl-step-response.v1",
        "config_name": config.name,
        "onset_epoch": onset_epoch,
        "step_amplitude": amplitude,
        "characteristic_response_epochs": reached,
        "public_anchor_epochs": 130,
        "public_anchor_comparison": "qualitative_only_different_surrogate_plant",
        "final_mean_policy": final_mean,
        "gates": gates,
        "passed": all(gates.values()),
        "trajectory": rows,
    }


def run_drift_tracking_certification(config: GoogleRLConfig, *, seed: int = 4400) -> dict[str, Any]:
    no_drift = run_no_drift_validation(config, seed=seed+1)
    slow = run_slow_drift_validation(config, seed=seed+3)
    step = run_step_response(config, seed=seed+9)
    return {
        "schema_version": "google-rl-drift-tracking-certification.v1",
        "evidence_layer": "executed declared detector-control drift surrogates",
        "config_name": config.name,
        "no_drift": no_drift,
        "slow_drift": slow,
        "step_response": step,
        "passed": bool(no_drift["passed"] and slow["passed"] and step["passed"]),
    }
