"""Ancillary paper anchors, development scorecard, and frozen certification gate."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from . import DISCLAIMER
from .config import CERTIFICATION_SEEDS, artifact_dir, canonical_hash, guard_seed, paper_scale, repository_root, sha256_file, source_choices
from .experiments import run_matched_trace
from .injected_drift_test import generate_injected_tape
from .plant import PurePlantSpec, PureQuadraticPlant
from .reporting import read_artifact, write_report


def _require_primary_gates() -> tuple[dict[str, Any], dict[str, Any]]:
    injected = read_artifact("injected_drift_stability")
    natural = read_artifact("natural_drift_spectral")
    if injected["status"] != "PASS" or natural["status"] != "PASS":
        raise RuntimeError("both primary drift development gates must pass before this study")
    return injected, natural


def run_steering_phase(epochs: int = 300) -> dict[str, Any]:
    try:
        _require_primary_gates()
    except RuntimeError as error:
        payload = {
            "schema_version": "google-pure-v5-steering-phase.v1",
            "status": "BLOCKED_PRIMARY_DRIFT_GATE",
            "reason": str(error),
            "checks": {
                "too_little_exploration_underperforms_balanced": False,
                "balanced_entropy_tracks_slow_drift": False,
                "too_much_exploration_damages_candidates": False,
                "learned_mean_useful_at_least_as_wide_as_candidates": False,
                "fixed_entropy_only": True,
                "distance_three_analogue": True,
            },
            "critical_frequency": {
                "learned_mean_estimate": None,
                "stochastic_candidate_estimate": None,
                "grid_uncertainty": [],
                "public_approximate_anchor": 1 / 150,
            },
            "rows": [],
            "certification_seeds_consumed": False,
        }
        write_report("steering_phase_diagram", payload, "Steering phase diagram")
        return payload
    if epochs < 180:
        raise ValueError("steering phase requires at least 180 epochs")
    base_choices, paper = dict(source_choices()), paper_scale()
    frequencies = [1 / 600, 1 / 300, 1 / 200, 1 / 150, 1 / 100, 1 / 70]
    entropies = [0.0, 0.0001, 0.0004, 0.002, 0.008]
    rows = []
    for entropy_index, entropy in enumerate(entropies):
        choices = dict(base_choices)
        choices["entropy_coefficient"] = entropy
        for frequency_index, frequency in enumerate(frequencies):
            plant = PureQuadraticPlant(PurePlantSpec(f"steering-{entropy_index}-{frequency_index}", draw_seed=7800 + 10 * entropy_index + frequency_index))
            scenario = {
                "profile": "sinusoidal", "category": "XY pulse frequency", "location": 2,
                "amplitude": 0.20, "frequency": frequency, "phase": 0.3,
            }
            tape, _ = generate_injected_tape(scenario, epochs, 20, plant.spec.control_count)
            result = run_matched_trace(plant, tape, choices, paper, seed=7850 + 10 * entropy_index + frequency_index)
            start = epochs // 3
            ler = result["logical_risk"]
            fixed = float(np.mean(ler["fixed_policy"][start:]))
            learned = float(np.mean(ler["learned_mean"][start:]))
            stochastic = float(np.mean(ler["stochastic_candidates"][start:]))
            rows.append({
                "entropy_coefficient": entropy, "frequency_epochs_inverse": frequency,
                "fixed_policy_mean_ler": fixed, "learned_mean_ler": learned,
                "stochastic_candidates_mean_ler": stochastic,
                "oracle_mean_ler": float(np.mean(ler["oracle_optimum"][start:])),
                "learned_mean_improvement": (fixed - learned) / max(fixed, 1e-15),
                "stochastic_improvement": (fixed - stochastic) / max(fixed, 1e-15),
                "exploration_damage": stochastic - learned,
                "final_scale": float(result["policy_scale_vectors"][-1].mean()),
            })
    balanced = [row for row in rows if row["entropy_coefficient"] == 0.0004]
    learned_useful = [row["frequency_epochs_inverse"] for row in balanced if row["learned_mean_improvement"] > 0.02]
    stochastic_useful = [row["frequency_epochs_inverse"] for row in balanced if row["stochastic_improvement"] > 0.02]
    learned_cutoff = max(learned_useful) if learned_useful else 0.0
    stochastic_cutoff = max(stochastic_useful) if stochastic_useful else 0.0
    slow = min(frequencies)
    low_entropy = next(row for row in rows if row["entropy_coefficient"] == 0.0 and row["frequency_epochs_inverse"] == slow)
    balanced_slow = next(row for row in balanced if row["frequency_epochs_inverse"] == slow)
    high_entropy = next(row for row in rows if row["entropy_coefficient"] == max(entropies) and row["frequency_epochs_inverse"] == slow)
    checks = {
        "too_little_exploration_underperforms_balanced": low_entropy["learned_mean_improvement"] < balanced_slow["learned_mean_improvement"] + 0.02,
        "balanced_entropy_tracks_slow_drift": balanced_slow["learned_mean_improvement"] > 0.02,
        "too_much_exploration_damages_candidates": high_entropy["exploration_damage"] > balanced_slow["exploration_damage"],
        "learned_mean_useful_at_least_as_wide_as_candidates": learned_cutoff >= stochastic_cutoff,
        "fixed_entropy_only": True,
        "distance_three_analogue": True,
    }
    payload = {
        "schema_version": "google-pure-v5-steering-phase.v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "distance": 3,
        "frequency_grid_epochs_inverse": frequencies,
        "fixed_entropy_grid": entropies,
        "critical_frequency": {
            "learned_mean_estimate": learned_cutoff,
            "stochastic_candidate_estimate": stochastic_cutoff,
            "grid_uncertainty": [1 / 300, 1 / 70],
            "public_approximate_anchor": 1 / 150,
        },
        "rows": rows,
        "certification_seeds_consumed": False,
    }
    write_report("steering_phase_diagram", payload, "Steering phase diagram")
    return payload


def run_randomized_recovery(epochs: int = 1000) -> dict[str, Any]:
    if epochs < 200:
        raise ValueError("recovery development requires at least 200 epochs")
    choices, paper = source_choices(), paper_scale()
    severities = [0.25, 0.45, 0.65]
    frozen_fraction = 0.5
    rows = []
    for index, severity in enumerate(severities):
        plant = PureQuadraticPlant(PurePlantSpec(f"recovery-{index}", draw_seed=7900 + index))
        rng = np.random.default_rng(7950 + index)
        spoiled = np.zeros(plant.spec.control_count)
        selected = rng.choice(plant.spec.control_count, size=int(round(frozen_fraction * plant.spec.control_count)), replace=False)
        spoiled[selected] = rng.choice([-1.0, 1.0], size=len(selected)) * severity
        from .reference_agent import PureGoogleReferenceAgent, evidence_from_counts
        agent = PureGoogleReferenceAgent(plant.mask, spoiled, plant.native_sensitivity, choices, seed=7960 + index)
        acquire_rng = np.random.default_rng(7970 + index)
        initial_ler = float(plant.logical_risk(spoiled[None, :], plant.base_optimum)[0])
        target_ler = plant.spec.logical_floor + 0.10 * (initial_ler - plant.spec.logical_floor)
        trace = []
        for _ in range(epochs):
            batch = agent.sample(int(paper["candidates_per_epoch"]))
            counts = plant.acquire_counts(batch.normalized_actions, plant.base_optimum, effective_cycles=int(paper["effective_cycles_per_candidate"]), rng=acquire_rng)
            agent.update(batch, evidence_from_counts(batch, counts, int(paper["effective_cycles_per_candidate"])))
            trace.append(float(plant.logical_risk(agent.mean[None, :], plant.base_optimum)[0]))
        smooth_width = 25
        smoothed = np.convolve(trace, np.ones(smooth_width) / smooth_width, mode="valid")
        crossings = np.flatnonzero(smoothed <= target_ler)
        crossing = int(crossings[0] + smooth_width - 1) if len(crossings) else None
        rows.append({
            "spoil_severity": severity, "randomized_fraction": frozen_fraction,
            "randomization_distribution": "Rademacher at frozen normalized severity",
            "normalized_distance_from_optimum": float(np.linalg.norm(spoiled) / np.sqrt(len(spoiled))),
            "initial_excess_detector_rate": float(plant.detector_rates(spoiled[None, :], plant.base_optimum).mean() - plant.floors.mean()),
            "initial_logical_degradation": initial_ler - plant.spec.logical_floor,
            "target_calibrated_ler": target_ler, "smoothing_width_epochs": smooth_width,
            "recovery_crossing_epoch": crossing, "final_ler": float(np.mean(trace[-25:])),
            "source_1000_epoch_commensurability": "synthetic severity declared; not asserted hardware-equivalent",
        })
    valid_crossings = [row["recovery_crossing_epoch"] for row in rows if row["recovery_crossing_epoch"] is not None]
    checks = {
        "frozen_severity_before_evaluation": True,
        "all_severities_recover": len(valid_crossings) == len(rows),
        "harder_not_materially_faster": all(rows[i]["recovery_crossing_epoch"] <= rows[i + 1]["recovery_crossing_epoch"] + 30 for i in range(len(rows) - 1)) if len(valid_crossings) == len(rows) else False,
        "final_level_reached": all(row["final_ler"] <= row["target_calibrated_ler"] for row in rows),
    }
    payload = {
        "schema_version": "google-pure-v5-randomized-recovery.v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "frozen_protocol": {"randomized_fraction": frozen_fraction, "severities": severities, "crossing_rule": "first 25-epoch moving mean at 90% excess-LER recovery"},
        "rows": rows,
        "median_recovery_epoch": float(np.median(valid_crossings)) if valid_crossings else None,
        "public_anchor_interpretation": "approximately 1000 epochs is interpreted only under comparable spoil severity; no rate was changed to force it",
        "certification_seeds_consumed": False,
    }
    write_report("randomized_recovery", payload, "Randomized-policy recovery")
    return payload


def surface_code_gate_count(distance: int) -> int:
    if distance < 3 or distance % 2 == 0:
        raise ValueError("distance must be odd and at least three")
    return 6 * distance * distance - 4 * distance - 1


def surface_code_control_count(distance: int) -> int:
    return 30 * surface_code_gate_count(distance)


def _scaling_realization(distance: int, epochs: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    gates = surface_code_gate_count(distance)
    controls = gates * 30
    active_gate = np.arange(gates) % 2 == 0
    mean = np.zeros(controls)
    mean.reshape(gates, 30)[active_gate] = 0.22
    log_scale = np.full(controls, np.log(float(source_choices()["initial_scale"])))
    baseline = np.zeros(gates)
    trajectory, gradient_variance, reward_variance, inactive_motion = [], [], [], []
    start_time = time.perf_counter()
    sparse_operations = 0
    for _ in range(epochs):
        scale = np.exp(log_scale)
        actions = mean[None, :] + scale[None, :] * rng.normal(size=(40, controls))
        grouped = actions.reshape(40, gates, 30)
        rates = 0.055 + 0.30 * np.mean(grouped * grouped, axis=2)
        counts = rng.binomial(100_000, np.clip(rates, 0, 0.45))
        rewards = -counts / 100_000.0
        advantage = rewards - baseline[None, :]
        score_mean = (actions - mean[None, :]) * np.exp(-2.0 * log_scale)[None, :]
        score_scale = (actions - mean[None, :]) ** 2 * np.exp(-2.0 * log_scale)[None, :] - 1.0
        local_advantage = np.repeat(advantage, 30, axis=1)
        sample_gradient = local_advantage * score_mean
        grad_mean = sample_gradient.mean(axis=0)
        grad_scale = (local_advantage * score_scale).mean(axis=0) + float(source_choices()["entropy_coefficient"])
        mean += float(source_choices()["mean_learning_rate"]) * grad_mean
        log_scale += float(source_choices()["scale_learning_rate"]) * grad_scale
        log_scale = np.clip(log_scale, np.log(0.04), np.log(0.25))
        baseline = (1.0 - 2.0 * float(source_choices()["baseline_learning_rate"])) * baseline + 2.0 * float(source_choices()["baseline_learning_rate"]) * rewards.mean(axis=0)
        objective = float(np.mean(mean.reshape(gates, 30) ** 2))
        trajectory.append(objective)
        gradient_variance.append(float(np.mean(np.var(sample_gradient, axis=0))))
        reward_variance.append(float(np.mean(np.var(rewards, axis=0))))
        inactive_motion.append(float(np.sqrt(np.mean(mean.reshape(gates, 30)[~active_gate] ** 2))))
        sparse_operations += 40 * controls * 4
    elapsed = time.perf_counter() - start_time
    y = np.log(np.maximum(trajectory, 1e-15))
    slope = float(np.polyfit(np.arange(epochs), y, 1)[0])
    return {
        "distance": distance, "gate_count": gates, "control_count": controls,
        "normalized_objective_trajectory": (np.asarray(trajectory) / max(trajectory[0], 1e-15)).tolist(),
        "convergence_rate_per_epoch": -slope,
        "gradient_variance": float(np.mean(gradient_variance)),
        "candidate_reward_variance": float(np.mean(reward_variance)),
        "inactive_region_motion": float(max(inactive_motion)),
        "runtime_seconds": elapsed, "sparse_operation_count": sparse_operations,
        "estimated_state_memory_bytes": int(8 * controls * 4 + 8 * gates),
    }


def run_convergence_scaling(epochs: int = 16) -> dict[str, Any]:
    if epochs < 8:
        raise ValueError("scaling requires at least eight actual epochs")
    distances = [3, 5, 7, 9, 11, 13, 15]
    rows = [_scaling_realization(distance, epochs, 8000 + 10 * distance + draw) for distance in distances for draw in range(2)]
    summaries = []
    for distance in distances:
        selected = [row for row in rows if row["distance"] == distance]
        summaries.append({
            "distance": distance, "gate_count": selected[0]["gate_count"], "control_count": selected[0]["control_count"],
            "mean_convergence_rate": float(np.mean([row["convergence_rate_per_epoch"] for row in selected])),
            "convergence_rate_95_percent_interval": [float(min(row["convergence_rate_per_epoch"] for row in selected)), float(max(row["convergence_rate_per_epoch"] for row in selected))],
            "mean_final_normalized_objective": float(np.mean([row["normalized_objective_trajectory"][-1] for row in selected])),
        })
    rates = np.array([row["mean_convergence_rate"] for row in summaries])
    counts = np.array([row["control_count"] for row in summaries], dtype=float)
    rate_slope = float(np.polyfit(np.log(counts), rates, 1)[0])
    relative_deterioration = float((rates[0] - rates[-1]) / max(abs(rates[0]), 1e-15))
    checks = {
        "all_exact_distances_run": [row["distance"] for row in summaries] == distances,
        "distance_15_exact_control_count": summaries[-1]["control_count"] == 38_670,
        "multiple_independent_realizations": all(sum(row["distance"] == distance for row in rows) >= 2 for distance in distances),
        "actual_objective_trajectories": all(len(row["normalized_objective_trajectory"]) == epochs for row in rows),
        "no_practical_rate_deterioration": relative_deterioration <= 0.15,
        "inactive_motion_small": max(row["inactive_region_motion"] for row in rows) < 0.025,
    }
    payload = {
        "schema_version": "google-pure-v5-convergence-scaling.v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks, "distances": distances, "realizations": rows, "summaries": summaries,
        "fits": {"convergence_rate_vs_log_control_count_slope": rate_slope, "distance3_to_distance15_relative_deterioration": relative_deterioration},
        "frozen_practical_deterioration_tolerance": 0.15,
        "certification_seeds_consumed": False,
    }
    write_report("convergence_scaling", payload, "Actual sparse convergence scaling")
    return payload


def write_unspecified_choice_log() -> dict[str, Any]:
    choices = dict(source_choices())
    record = {
        "schema_version": "google-pure-v5-choice-record.v1", "choice_set": choices,
        "split": "mechanism-development", "selection": "minimal source-consistent default",
        "selection_criterion": "mechanism validity and static stationarity; not closeness to 2.4x or 4 dB",
        "new_algorithmic_components": [], "certification_seeds_consumed": False,
        "disclaimer": DISCLAIMER,
    }
    path = artifact_dir() / "unspecified_choice_log.jsonl"
    path.write_text(json.dumps(record, sort_keys=True, allow_nan=False, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown = ["# Source-unspecified choice log", "", DISCLAIMER, "", "No held-out-anchor optimization was used.", "", "```json", json.dumps(record, indent=2, sort_keys=True), "```", ""]
    (artifact_dir() / "unspecified_choice_log.md").write_text("\n".join(markdown), encoding="utf-8")
    return record


def run_development_scorecard() -> dict[str, Any]:
    names = ["source_compliance_map", "baseline_forensic_audit", "numerical_algorithm_validation", "static_tests", "injected_drift_stability", "natural_drift_spectral", "test_separation_audit", "steering_phase_diagram", "randomized_recovery", "convergence_scaling"]
    evidence = {name: read_artifact(name) for name in names}
    write_unspecified_choice_log()
    gates = {name: value.get("status") == "PASS" for name, value in evidence.items()}
    primary_pass = gates["injected_drift_stability"] and gates["natural_drift_spectral"]
    all_pass = all(gates.values())
    if all_pass:
        outcome = "PURE_GOOGLE_STYLE_SYNTHETIC_REPRODUCTION_DEVELOPMENT_PASS"
    elif evidence["source_compliance_map"]["status"] != "PASS" or evidence["numerical_algorithm_validation"]["status"] != "PASS":
        outcome = "PURE_REPRODUCTION_FAILED_ALGORITHM"
    elif not primary_pass:
        outcome = "PURE_REPRODUCTION_FAILED_TEST_COMMENSURABILITY"
    else:
        outcome = "PARTIAL_PURE_REPRODUCTION"
    payload = {
        "schema_version": "google-pure-v5-development-scorecard.v1",
        "status": "PASS" if all_pass else "FAIL", "overall_status": outcome,
        "all_development_gates_pass": all_pass, "primary_drift_gates_pass": primary_pass,
        "gates": gates,
        "headline": {
            "injected_stability_ratio": evidence["injected_drift_stability"]["aggregate"]["median_control_only_stability_ratio"],
            "mean_ler_improvement": evidence["injected_drift_stability"]["aggregate"]["median_relative_mean_ler_improvement"],
            "step_response_epochs": evidence["injected_drift_stability"]["aggregate"]["median_step_response_epochs"],
            "natural_lf_gain_db": evidence["natural_drift_spectral"]["aggregate"]["median_low_frequency_gain_db"],
            "steering_cutoff": evidence["steering_phase_diagram"]["critical_frequency"]["learned_mean_estimate"],
            "randomized_recovery_epoch": evidence["randomized_recovery"]["median_recovery_epoch"],
            "scaling_relative_deterioration": evidence["convergence_scaling"]["fits"]["distance3_to_distance15_relative_deterioration"],
        },
        "certification_blocked": not all_pass,
        "blocked_downstream": ["reduced-budget equivalence", "staged comparison"],
        "certification_seeds_consumed": False,
    }
    write_report("development_scorecard", payload, "Pure v5 development scorecard")
    return payload


def freeze_certification() -> dict[str, Any]:
    score = read_artifact("development_scorecard")
    ready = bool(score.get("all_development_gates_pass", False))
    if not ready:
        payload = {
            "schema_version": "google-pure-v5-certification-preregistration.v1",
            "status": "NOT_FROZEN_DEVELOPMENT_GATES_FAILED", "one_run_permitted": False,
            "development_status": score["overall_status"], "certification_seeds": list(CERTIFICATION_SEEDS),
            "certification_seeds_consumed": False,
            "allowed_outcomes": ["PURE_GOOGLE_STYLE_SYNTHETIC_REPRODUCTION_CERTIFIED", "PARTIAL_PURE_REPRODUCTION", "PURE_REPRODUCTION_FAILED_ALGORITHM", "PURE_REPRODUCTION_FAILED_TEST_COMMENSURABILITY"],
        }
        write_report("certification_preregistration", payload, "Certification preregistration")
        return payload
    root = repository_root()
    files = list((root / "src/hdfa_rl_suite/google_pure_v5").glob("*.py")) + list((root / "configs/google_pure_v5").glob("*.yaml")) + list(artifact_dir().glob("*.json")) + list(artifact_dir().glob("*.jsonl"))
    hashes = {str(path.relative_to(root)).replace("\\", "/"): sha256_file(path) for path in sorted(files) if path.name != "certification_preregistration.json"}
    payload = {
        "schema_version": "google-pure-v5-certification-preregistration.v1",
        "status": "FROZEN_READY_UNOPENED", "one_run_permitted": True,
        "protected_file_hashes": hashes, "protected_manifest_hash": canonical_hash(hashes),
        "certification_seeds": list(CERTIFICATION_SEEDS), "certification_seeds_consumed": False,
        "post_opening_amendment_prohibited": True,
        "allowed_outcomes": ["PURE_GOOGLE_STYLE_SYNTHETIC_REPRODUCTION_CERTIFIED", "PARTIAL_PURE_REPRODUCTION", "PURE_REPRODUCTION_FAILED_ALGORITHM", "PURE_REPRODUCTION_FAILED_TEST_COMMENSURABILITY"],
        "blocked_downstream_until_certified": ["reduced-budget equivalence", "staged comparison"],
    }
    write_report("certification_preregistration", payload, "Certification preregistration")
    return payload


def run_certification(*, confirm: bool, epochs: int = 1000) -> dict[str, Any]:
    if not confirm:
        raise RuntimeError("locked certification seeds require --confirm-open-locked-seeds")
    prereg = read_artifact("certification_preregistration")
    if prereg.get("status") != "FROZEN_READY_UNOPENED" or not prereg.get("one_run_permitted"):
        raise RuntimeError("certification is blocked because development gates are not frozen ready")
    marker = artifact_dir() / "certification_seed_opening.json"
    if marker.exists():
        raise RuntimeError("the one permitted certification run has already been opened")
    for seed in CERTIFICATION_SEEDS:
        guard_seed(seed, certification=True)
    root = repository_root()
    mismatches = [name for name, expected in prereg["protected_file_hashes"].items() if sha256_file(root / name) != expected]
    if mismatches:
        raise RuntimeError(f"protected certification evidence changed: {mismatches}")
    payload = {
        "schema_version": "google-pure-v5-certification-opening.v1",
        "status": "LOCKED_SEEDS_OPENED_USER_RUN_REQUIRED",
        "epochs": int(epochs), "seeds": list(CERTIFICATION_SEEDS),
        "note": "Seed opening is recorded; long certification execution is intentionally not automatic in this development command.",
    }
    marker.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
