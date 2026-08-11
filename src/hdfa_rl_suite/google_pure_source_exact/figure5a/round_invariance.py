"""Preregistered circuit-depth nuisance analysis for Figure 5a."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np

from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.gaussian import (
    DirectSigmaGaussianPolicy,
)
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.losses import (
    total_loss_and_gradients,
)

from .contracts import canonical_hash
from .validation import build_plant


def _global_edr_from_reduced(plant: Any, rates: np.ndarray) -> float:
    weights = np.asarray([len(group) for group in plant.reward_component_raw_detectors], dtype=float)
    return float(np.dot(np.asarray(rates, dtype=float), weights) / plant.raw_detector_count)


def _improvement_ratio(fixed_edr: float, candidate_edr: float, oracle_edr: float) -> float:
    denominator = fixed_edr - oracle_edr
    if denominator <= 0:
        raise RuntimeError("round-invariance fixed/oracle EDR denominator is not positive")
    return float((fixed_edr - candidate_edr) / denominator)


def _objective_gradients(
    *, actions: np.ndarray, rewards: np.ndarray, plant: Any,
    mean: np.ndarray, sigma: np.ndarray, behavior: Any, clip: float,
) -> tuple[np.ndarray, np.ndarray]:
    loss = total_loss_and_gradients(
        actions, rewards, plant.mask, mean, sigma, np.zeros(plant.detector_count), behavior,
        clip=clip, policy_weight=1.0, baseline_weight=0.0, entropy_weight=0.0)
    # J is the expected sum of negative detector rates and is maximized.  The
    # optimizer minimizes -J, hence the sign reversal from loss gradients.
    return -loss.grad_mean, -loss.grad_sigma


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return None if denominator == 0.0 else float(np.dot(left, right) / denominator)


def _comparison(row: Mapping[str, Any], primary: Mapping[str, Any]) -> dict[str, Any]:
    grad_mu = np.asarray(row["grad_mu_J"], dtype=float)
    primary_mu = np.asarray(primary["grad_mu_J"], dtype=float)
    grad_sigma = np.asarray(row["grad_sigma_J"], dtype=float)
    primary_sigma = np.asarray(primary["grad_sigma_J"], dtype=float)
    return {
        "rounds": int(row["rounds"]),
        "EDR_absolute_delta": {
            stream: abs(float(row["EDR"][stream]) - float(primary["EDR"][stream]))
            for stream in ("fixed", "oracle", "candidate")
        },
        "grad_mu_J_cosine": _cosine(grad_mu, primary_mu),
        "grad_mu_J_relative_l2_delta": float(
            np.linalg.norm(grad_mu - primary_mu) / max(np.linalg.norm(primary_mu), np.finfo(float).tiny)),
        "grad_sigma_J_cosine": _cosine(grad_sigma, primary_sigma),
        "grad_sigma_J_relative_l2_delta": float(
            np.linalg.norm(grad_sigma - primary_sigma) /
            max(np.linalg.norm(primary_sigma), np.finfo(float).tiny)),
        "r_absolute_delta": abs(float(row["r"]) - float(primary["r"])),
    }


def plan_round_invariance(config: Mapping[str, Any]) -> dict[str, Any]:
    value = config["circuit_round_invariance"]
    rounds = tuple(int(item) for item in value["rounds"])
    candidates = int(value["candidates"])
    cycles = int(value["qec_cycles_per_policy"])
    if any(cycles % item for item in rounds):
        raise ValueError("round-invariance QEC budget must be divisible by every circuit depth")
    streams = 3  # fixed, oracle, and the frozen stochastic candidate batch
    return {
        "schema_version": "figure5a-round-invariance-plan.v1",
        "scientific_status": value["scientific_status"],
        "rounds": list(rounds), "primary_rounds": int(value["primary_rounds"]),
        "candidates": candidates, "qec_cycles_per_policy": cycles,
        "finite_qec_cycles": len(rounds) * streams * candidates * cycles,
        "Stim_shots": int(candidates * streams * cycles * sum(1 / item for item in rounds)),
        "estimated_circuit_compilations": len(rounds) * (candidates + 2),
        "metrics": list(value["metrics"]),
        "frozen_policy_across_rounds": True,
        "reference_seeds_consumed": False,
        "long_run_not_launched_by_plan": True,
        "plan_hash": canonical_hash(value),
    }


def run_round_invariance(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run exact and equal-QEC finite-shot comparisons at T=5,10,25,50."""
    value = config["circuit_round_invariance"]
    plan = plan_round_invariance(config)
    rounds = tuple(int(item) for item in value["rounds"])
    primary_rounds = int(value["primary_rounds"])
    candidates = int(value["candidates"])
    cycles = int(value["qec_cycles_per_policy"])
    seed = int(value["development_seed"])
    epoch = int(value["epoch"])
    frequency = float(value["frequency"])
    mean = np.zeros(41)
    sigma = np.full(41, float(config["controller"]["initial_sigma"]))
    batch = DirectSigmaGaussianPolicy(mean, sigma, seed=seed).sample(candidates)
    exact_rows: list[dict[str, Any]] = []
    finite_rows: list[dict[str, Any]] = []
    topology_rows: list[dict[str, Any]] = []
    for circuit_rounds in rounds:
        round_config = deepcopy(dict(config))
        round_config["plant"]["circuit_rounds"] = circuit_rounds
        plant = build_plant(round_config)
        target = plant.optimum(epoch, frequency)
        fixed_rates = plant.expected_reward_rates(
            np.zeros(41), epoch=epoch, frequency=frequency, target_controls=target)
        oracle_rates = plant.expected_reward_rates(
            target, epoch=epoch, frequency=frequency, target_controls=target)
        candidate_rates = np.asarray([
            plant.expected_reward_rates(action, epoch=epoch, frequency=frequency,
                                        target_controls=target)
            for action in batch.actions
        ])
        grad_mu, grad_sigma = _objective_gradients(
            actions=batch.actions, rewards=-candidate_rates, plant=plant,
            mean=mean, sigma=sigma, behavior=batch.behavior,
            clip=float(config["controller"]["ppo_clip"]))
        exact_edr = {
            "fixed": _global_edr_from_reduced(plant, fixed_rates),
            "oracle": _global_edr_from_reduced(plant, oracle_rates),
            "candidate": float(np.mean([
                _global_edr_from_reduced(plant, row) for row in candidate_rates])),
        }
        exact_rows.append({
            "rounds": circuit_rounds, "raw_detector_count": plant.raw_detector_count,
            "reward_component_count": plant.detector_count, "EDR": exact_edr,
            "grad_mu_J": grad_mu.tolist(), "grad_sigma_J": grad_sigma.tolist(),
            "grad_mu_J_l2_norm": float(np.linalg.norm(grad_mu)),
            "grad_sigma_J_l2_norm": float(np.linalg.norm(grad_sigma)),
            "r": _improvement_ratio(exact_edr["fixed"], exact_edr["candidate"],
                                    exact_edr["oracle"]),
        })

        aggregate_cycles = candidates * cycles
        fixed = plant.sample_detector_observation(
            np.zeros(41), epoch=epoch, frequency=frequency, qec_cycles=aggregate_cycles,
            seed=plant.stream_seed(seed, "round-invariance-fixed", epoch, 0),
            target_controls=target)
        oracle = plant.sample_detector_observation(
            target, epoch=epoch, frequency=frequency, qec_cycles=aggregate_cycles,
            seed=plant.stream_seed(seed, "round-invariance-oracle", epoch, 0),
            target_controls=target)
        sampled = [plant.sample_detector_observation(
            action, epoch=epoch, frequency=frequency, qec_cycles=cycles,
            seed=plant.stream_seed(seed, "round-invariance-candidate", epoch, index),
            target_controls=target) for index, action in enumerate(batch.actions)]
        finite_rewards = -np.asarray([item.reward_rates for item in sampled])
        finite_grad_mu, finite_grad_sigma = _objective_gradients(
            actions=batch.actions, rewards=finite_rewards, plant=plant,
            mean=mean, sigma=sigma, behavior=batch.behavior,
            clip=float(config["controller"]["ppo_clip"]))
        total_candidate_events = sum(item.raw_total for item in sampled)
        total_shots = aggregate_cycles // circuit_rounds
        finite_edr = {
            "fixed": float(fixed.raw_total / (total_shots * plant.raw_detector_count)),
            "oracle": float(oracle.raw_total / (total_shots * plant.raw_detector_count)),
            "candidate": float(total_candidate_events / (total_shots * plant.raw_detector_count)),
        }
        finite_rows.append({
            "rounds": circuit_rounds, "raw_detector_count": plant.raw_detector_count,
            "reward_component_count": plant.detector_count, "EDR": finite_edr,
            "grad_mu_J": finite_grad_mu.tolist(), "grad_sigma_J": finite_grad_sigma.tolist(),
            "grad_mu_J_l2_norm": float(np.linalg.norm(finite_grad_mu)),
            "grad_sigma_J_l2_norm": float(np.linalg.norm(finite_grad_sigma)),
            "r": _improvement_ratio(finite_edr["fixed"], finite_edr["candidate"],
                                    finite_edr["oracle"]),
            "fixed_raw_events": fixed.raw_total, "oracle_raw_events": oracle.raw_total,
            "candidate_raw_events": total_candidate_events,
        })
        topology_rows.append({
            "rounds": circuit_rounds, "parameter_ids": list(plant.parameter_ids),
            "mask": plant.mask.astype(int).tolist(),
            "reward_component_keys": list(plant.reward_component_keys),
        })

    primary_exact = next(row for row in exact_rows if row["rounds"] == primary_rounds)
    primary_finite = next(row for row in finite_rows if row["rounds"] == primary_rounds)
    exact_comparisons = [_comparison(row, primary_exact) for row in exact_rows]
    finite_comparisons = [_comparison(row, primary_finite) for row in finite_rows]
    reference_topology = topology_rows[rounds.index(primary_rounds)]
    topology_invariant = all(
        row["parameter_ids"] == reference_topology["parameter_ids"] and
        row["mask"] == reference_topology["mask"] and
        row["reward_component_keys"] == reference_topology["reward_component_keys"]
        for row in topology_rows)
    tolerance = value["near_invariance_definition"]
    nonprimary = [row for row in exact_comparisons if row["rounds"] != primary_rounds]
    exact_gate = topology_invariant and all(
        max(row["EDR_absolute_delta"].values()) <= float(tolerance["maximum_EDR_absolute_delta"]) and
        row["grad_mu_J_cosine"] is not None and
        row["grad_mu_J_cosine"] >= float(tolerance["minimum_gradient_cosine"]) and
        row["grad_sigma_J_cosine"] is not None and
        row["grad_sigma_J_cosine"] >= float(tolerance["minimum_gradient_cosine"]) and
        row["grad_mu_J_relative_l2_delta"] <= float(tolerance["maximum_gradient_relative_l2_delta"]) and
        row["grad_sigma_J_relative_l2_delta"] <= float(tolerance["maximum_gradient_relative_l2_delta"]) and
        row["r_absolute_delta"] <= float(tolerance["maximum_r_absolute_delta"])
        for row in nonprimary)
    result = {
        "schema_version": "figure5a-round-invariance.v1", "plan": plan,
        "objective_convention": "J is expected sum of reduced negative-EDR rewards and is maximized; entropy and baseline gradients are disabled",
        "equal_total_QEC_cycle_budget": True, "frozen_gaussian_actions_across_rounds": True,
        "exact": {"rows": exact_rows, "comparisons_to_primary": exact_comparisons},
        "finite_shot": {"rows": finite_rows, "comparisons_to_primary": finite_comparisons},
        "topology_invariant": topology_invariant,
        "near_invariance_definition": dict(tolerance),
        "exact_near_invariance_pass": bool(exact_gate),
        "interpretation": (
            "circuit_rounds may be treated as an implementation detail under the preregistered exact gate"
            if exact_gate else
            "circuit_rounds is a material nuisance parameter; retain depth-stratified results"),
    }
    result["result_hash"] = canonical_hash(result)
    return result
