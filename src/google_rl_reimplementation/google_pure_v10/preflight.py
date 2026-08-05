"""Independent physical and algorithmic preflight for comparative acquisitions."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from google_rl_reimplementation.google_pure_v6.plant import PureQuadraticPlant, default_spec
from google_rl_reimplementation.google_pure_v7.config import canonical_hash, repository_root
from google_rl_reimplementation.google_pure_v9.common import artifact_root as v9_artifact_root, read_json as read_v9_json

from .common import read_json, write_artifact
from .contracts import evidence_envelope


def preflight_gate(checks: Mapping[str, bool]) -> dict[str, Any]:
    required = {
        "plant_no_disturbance_sanity",
        "fixed_oracle_disturbance_sanity",
        "periodic_intermediate_sanity",
        "toy_reference_convergence",
        "positive_gradient_direction",
        "sample_budget_adequacy",
        "mean_exploration_separation",
        "policy_lifecycle_consistency",
        "matched_disturbance_realization",
        "complete_hashing",
    }
    missing = sorted(required - set(checks))
    if missing:
        raise ValueError(f"preflight check set is incomplete: {missing}")
    failures = [name for name in sorted(required) if not bool(checks[name])]
    return {"pass": not failures, "failed_checks": failures}


def _physical_checks() -> dict[str, bool | float | list[float]]:
    plant = PureQuadraticPlant(default_spec(6))
    direction = np.linspace(1.0, 0.45, plant.spec.control_count)
    direction /= np.linalg.norm(direction)
    base = plant.spec.base_optimum_normalized
    base_native = plant.spec.coordinates.to_native(base)
    stationary = [
        float(np.mean(plant.detector_rates_native(base_native[None, :], base_native)))
        for _ in range(5)
    ]
    amplitudes = np.asarray([0.05, 0.1, 0.2, 0.3])
    fixed_costs = []
    oracle_costs = []
    periodic_costs = []
    for amplitude in amplitudes:
        optimum = amplitude * direction
        optimum_native = plant.spec.coordinates.to_native(optimum)
        fixed_costs.append(float(np.mean(plant.detector_rates_native(base_native[None, :], optimum_native))))
        oracle_costs.append(float(np.mean(plant.detector_rates_native(optimum_native[None, :], optimum_native))))
        midpoint_native = plant.spec.coordinates.to_native(0.5 * optimum)
        periodic_costs.append(float(np.mean(plant.detector_rates_native(midpoint_native[None, :], optimum_native))))
    no_disturbance = bool(np.ptp(stationary) <= 1e-15)
    monotone = bool(np.all(np.diff(fixed_costs) > 0))
    ordering = bool(np.all(np.asarray(oracle_costs) <= np.asarray(periodic_costs)) and np.all(np.asarray(periodic_costs) <= np.asarray(fixed_costs)))
    return {
        "plant_no_disturbance_sanity": no_disturbance,
        "fixed_oracle_disturbance_sanity": monotone and all(fixed > oracle for fixed, oracle in zip(fixed_costs, oracle_costs)),
        "periodic_intermediate_sanity": ordering,
        "stationary_costs": stationary,
        "step_amplitudes": amplitudes.tolist(),
        "fixed_costs": fixed_costs,
        "periodic_costs": periodic_costs,
        "oracle_costs": oracle_costs,
    }


def _toy_and_gradient_checks() -> dict[str, Any]:
    optimum = 0.35
    final = []
    for start in (-0.8, 0.8):
        mean = start
        for _ in range(80):
            exact_gradient = -2.0 * (mean - optimum)
            mean += 0.05 * exact_gradient
        final.append(mean)
    toy_pass = all(abs(value - optimum) < 5e-4 for value in final)
    rng = np.random.default_rng(21401)
    dimensions = 6
    curvature = np.linspace(0.4, 1.0, dimensions)
    mean = np.linspace(-0.2, 0.25, dimensions)
    true_gradient = -2 * curvature * mean
    sigma = 0.04
    candidates = mean[None, :] + sigma * rng.normal(size=(4000, dimensions))
    rewards = -np.sum(curvature[None, :] * candidates ** 2, axis=1)
    score = (candidates - mean[None, :]) / sigma ** 2
    estimated_gradient = np.mean((rewards - np.mean(rewards))[:, None] * score, axis=0)
    cosine = float(np.dot(true_gradient, estimated_gradient) / (np.linalg.norm(true_gradient) * np.linalg.norm(estimated_gradient)))
    return {
        "toy_reference_convergence": toy_pass,
        "toy_final_means": final,
        "positive_gradient_direction": cosine > 0.95,
        "gradient_cosine_similarity": cosine,
    }


def _sample_budget_check() -> dict[str, Any]:
    plant = PureQuadraticPlant(default_spec(6))
    rng = np.random.default_rng(21411)
    candidate_action_scale = 0.2
    cycles_per_candidate = 100000
    actions = rng.normal(scale=candidate_action_scale, size=(48, plant.spec.control_count))
    optimum = np.zeros(plant.spec.control_count)
    native_actions = plant.spec.coordinates.to_native(actions)
    optimum_native = plant.spec.coordinates.to_native(optimum)
    true_cost = np.sum(plant.detector_rates_native(native_actions, optimum_native), axis=1)
    counts = plant.acquire_counts(native_actions, optimum_native, cycles=cycles_per_candidate, rng=rng)
    observed_cost = np.sum(counts, axis=1) / cycles_per_candidate
    true_rank = np.argsort(np.argsort(true_cost))
    observed_rank = np.argsort(np.argsort(observed_cost))
    rank_correlation = float(np.corrcoef(true_rank, observed_rank)[0, 1])
    top_quartile = set(np.argsort(true_cost)[:12])
    observed_top = set(np.argsort(observed_cost)[:12])
    ranking_accuracy = len(top_quartile & observed_top) / len(top_quartile)
    return {
        "sample_budget_adequacy": rank_correlation >= 0.8 and ranking_accuracy >= 0.75,
        "reward_rank_correlation": rank_correlation,
        "best_quartile_recovery": ranking_accuracy,
        "candidate_action_scale": candidate_action_scale,
        "cycles_per_candidate": cycles_per_candidate,
    }


def run_preflight() -> dict[str, Any]:
    physical = _physical_checks()
    algorithm = _toy_and_gradient_checks()
    budget = _sample_budget_check()
    root = repository_root()
    lifecycle = read_json(root / "artifacts" / "google_pure_v8" / "ppo_update_lifecycle_audit.json")
    baseline = read_json(root / "artifacts" / "google_pure_v8" / "baseline_freezing_audit.json")
    held_out_path = v9_artifact_root() / "stage_d_held_out_validation" / "results.json"
    selected_path = v9_artifact_root() / "selected_controller_contract.json"
    held_out = read_v9_json(held_out_path) if held_out_path.is_file() else {}
    selected = read_v9_json(selected_path) if selected_path.is_file() else {}
    cells = held_out.get("cells", [])
    checks = {
        "plant_no_disturbance_sanity": bool(physical["plant_no_disturbance_sanity"]),
        "fixed_oracle_disturbance_sanity": bool(physical["fixed_oracle_disturbance_sanity"]),
        "periodic_intermediate_sanity": bool(physical["periodic_intermediate_sanity"]),
        "toy_reference_convergence": bool(algorithm["toy_reference_convergence"]),
        "positive_gradient_direction": bool(algorithm["positive_gradient_direction"]),
        "sample_budget_adequacy": bool(budget["sample_budget_adequacy"]),
        "mean_exploration_separation": bool(cells) and all("five_policy" in row for row in cells),
        "policy_lifecycle_consistency": lifecycle.get("classification") == "PPO_CLIPPING_STRUCTURALLY_INACTIVE" and baseline.get("classification") == "PASS",
        "matched_disturbance_realization": bool(cells) and all(row.get("matched_policy_windows") for row in cells),
        "complete_hashing": bool(cells) and all(all(row.get(key) for key in ("controller_hash", "plant_hash", "graph_hash", "protocol_hash", "drift_tape_hash")) for row in cells),
    }
    gate = preflight_gate(checks)
    selected_controller = bool(selected.get("selected", False))
    full_benchmark = gate["pass"] and selected_controller and held_out.get("mode") in {"validation", "reference"}
    blockers = list(gate["failed_checks"])
    if not selected_controller:
        blockers.append("HELD_OUT_REFERENCE_CONTROLLER_NOT_SELECTED")
    if held_out.get("mode") not in {"validation", "reference"}:
        blockers.append("HELD_OUT_REFERENCE_MODE_NOT_EXECUTED")
    payload = {
        "schema_version": "google-pure-v10-preflight.v1",
        "checks": checks,
        "physical_diagnostics": physical,
        "algorithmic_diagnostics": algorithm,
        "sample_budget_diagnostics": budget,
        "preflight_gate_pass": gate["pass"],
        "full_benchmark_permitted": full_benchmark,
        "selected_controller": selected_controller,
        "held_out_mode": held_out.get("mode"),
        "configuration_hash": canonical_hash({"checks": checks, "held_out_hash": held_out.get("artifact_hash")}),
        **evidence_envelope(
            complete=True,
            mechanism_valid=gate["pass"],
            claim_supported=full_benchmark,
            paper_comparable=False,
            blocking_reasons=blockers,
        ),
    }
    return write_artifact("preflight_manifest", payload, "Comparative Benchmark Preflight")
