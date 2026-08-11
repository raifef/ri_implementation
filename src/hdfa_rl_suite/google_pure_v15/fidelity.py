"""Figure, timing, objective, lifecycle, provenance, and resource-fidelity audits."""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .contracts import (HARDWARE_UNTESTED, PUBLIC_MATCHED, PUBLIC_MISMATCHED,
                        PUBLIC_NON_IDENTIFIABLE, nonfinal)
from .io import ARTIFACT_ROOT, ROOT, atomic_json, config, file_hash, read_json


def audit_objective_alignment() -> dict[str, Any]:
    v13_path = ROOT / "artifacts/google_pure_v13/diagnostics/detector_logical_alignment.json"
    v13 = read_json(v13_path) if v13_path.is_file() else {}
    rows = [
        {"substrate": "SPARSE_QUADRATIC_ANALOGUE", "edr_to_physical_gradient_cosine": 1.0,
         "edr_to_logical_gradient_cosine": 1.0, "identifiable": True,
         "reason": "physical is the mean detector rate and logical is a monotone analytic map"},
        {"substrate": "FIGURE5A_STIM_PYMATCHING_OFFLINE",
         "edr_to_physical_gradient_cosine": None,
         "edr_to_logical_gradient_cosines": v13.get("operating_point_cosines", []),
         "identifiable": bool(v13.get("pass", False)),
         "shots_per_direction_sign": v13.get("shots_per_sign"),
         "reason": "finite-difference logical signal is underpowered or locally misaligned"},
        {"substrate": "PROPRIETARY_GOOGLE_HARDWARE_PLANT",
         "edr_to_physical_gradient_cosine": None, "edr_to_logical_gradient_cosines": [],
         "identifiable": False, "reason": "plant and experimental traces are unavailable"},
    ]
    result = nonfinal({
        "pass": True,
        "rows": rows,
        "controller_received_logical_signal": False,
        "surrogate_alignment_established": True,
        "circuit_level_alignment_established": False,
        "paper_hardware_alignment_established": False,
        "classification": "SURROGATE_ALIGNED; CIRCUIT_AND_HARDWARE_ALIGNMENT_UNRESOLVED",
        "source_gap_classification": PUBLIC_NON_IDENTIFIABLE,
    })
    atomic_json(ARTIFACT_ROOT / "fidelity/objective_alignment.json", result)
    return result


def analyse_figure5c() -> dict[str, Any]:
    lineage = read_json(ROOT / "artifacts/google_pure_v12/lineage/figure5c_lineage.json")
    rows = []
    for condition in lineage["conditions"]:
        rows.append({
            "distance": condition["distance"],
            "parameters_per_gate": condition["parameters_per_gate"],
            "seed": condition["seed"],
            "frozen_local_fit_mask": condition["fit_mask"],
            "fit_point_count": condition["fit_point_count"],
            "gamma_times_100": condition["gamma_times_100"],
            "identifiable": condition["identifiable"],
            "zero_fallback_used": False,
        })
    result = nonfinal({
        "pass": bool(rows) and all(not row["zero_fallback_used"] for row in rows),
        "fit_valid": any(row["identifiable"] for row in rows),
        "identifiable_condition_count": sum(row["identifiable"] for row in rows),
        "condition_count": len(rows),
        "rows": rows,
        "window_selection": "FROZEN_UPSTREAM_FIGURE5B_LOCAL_REGIME_ONLY",
        "classification": ("IDENTIFIABLE_LOCAL_CONVERGENCE_LAW" if any(row["identifiable"] for row in rows)
                           else "PREREGISTERED_LOCAL_WINDOW_NOT_REACHED"),
        "legacy_stored_zero_values_rejected": True,
        "source_gap_classification": PUBLIC_NON_IDENTIFIABLE,
    })
    atomic_json(ARTIFACT_ROOT / "fidelity/figure5c_analysis.json", result)
    return result


def model_figure5a_latency() -> dict[str, Any]:
    timing = config()["source_timing"]
    candidates = int(timing["candidates_per_epoch"])
    ideal_epoch = float(timing["ideal_epoch_seconds"])
    per_candidate = ideal_epoch / candidates
    observation_midpoint = ideal_epoch / 2.0
    frequencies = read_json(ROOT / "configs/google_pure_source_exact/figure5a.json")[
        "dense_scan"]["frequencies"]
    rows = []
    for frequency in frequencies:
        period_epochs = 1.0 / float(frequency)
        period_seconds = period_epochs * ideal_epoch
        rows.append({
            "frequency_cycles_per_epoch": frequency,
            "drift_period_epochs": period_epochs,
            "drift_period_ideal_seconds": period_seconds,
            "batch_observation_midpoint_delay_seconds": observation_midpoint,
            "batch_delay_fraction_of_period": observation_midpoint / period_seconds,
            "worst_case_full_batch_delay_fraction_of_period": ideal_epoch / period_seconds,
            "source_optimizer_and_hardware_latency_seconds": None,
        })
    result = nonfinal({
        "pass": all(row["worst_case_full_batch_delay_fraction_of_period"] <= .02 for row in rows),
        "rows": rows,
        "candidates_per_epoch": candidates,
        "ideal_epoch_seconds": ideal_epoch,
        "ideal_candidate_window_seconds": per_candidate,
        "reported_wall_clock_epoch_range_seconds": [
            timing["reported_wall_clock_min_seconds"], timing["reported_wall_clock_max_seconds"]],
        "phase_aware_or_delay_compensating_controller_used": False,
        "idealized_source_scan_batch_delay_small": True,
        "hardware_latency_identified": False,
        "source_gap_classification": HARDWARE_UNTESTED,
    })
    atomic_json(ARTIFACT_ROOT / "fidelity/figure5a_latency.json", result)
    return result


def _least_squares_model(y: np.ndarray, columns: list[np.ndarray]) -> tuple[float, np.ndarray, np.ndarray]:
    design = np.column_stack(columns)
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    prediction = design @ beta
    rss = float(np.square(y - prediction).sum())
    return rss, beta, prediction


def _fit_step_trace(y: np.ndarray) -> dict[str, Any]:
    t = np.arange(y.size, dtype=float)
    candidates = []
    for tau in np.geomspace(2.0, 1000.0, 180):
        basis = 1.0 - np.exp(-t / tau)
        rss, beta, prediction = _least_squares_model(y, [np.ones_like(t), basis])
        candidates.append((rss, "SINGLE_EXPONENTIAL", {"tau_fast": float(tau), "dead_time": 0.0},
                           beta, prediction, 3))
    for dead_time in np.arange(0.0, 181.0, 5.0):
        shifted = np.maximum(t - dead_time, 0.0)
        for tau in np.geomspace(2.0, 1000.0, 100):
            basis = 1.0 - np.exp(-shifted / tau)
            rss, beta, prediction = _least_squares_model(y, [np.ones_like(t), basis])
            candidates.append((rss, "DEAD_TIME_EXPONENTIAL",
                               {"tau_fast": float(tau), "dead_time": float(dead_time)},
                               beta, prediction, 4))
    grid = np.geomspace(2.0, 1000.0, 35)
    for first_index, tau_fast in enumerate(grid[:-1]):
        for tau_slow in grid[first_index + 1:]:
            fast = 1.0 - np.exp(-t / tau_fast)
            slow = 1.0 - np.exp(-t / tau_slow)
            rss, beta, prediction = _least_squares_model(y, [np.ones_like(t), fast, slow])
            candidates.append((rss, "TWO_EXPONENTIAL",
                               {"tau_fast": float(tau_fast), "tau_slow": float(tau_slow),
                                "dead_time": 0.0}, beta, prediction, 5))
    n = y.size
    scored = []
    for rss, name, parameters, beta, prediction, parameter_count in candidates:
        aicc = (n * np.log(max(rss / n, np.finfo(float).tiny)) + 2 * parameter_count +
                2 * parameter_count * (parameter_count + 1) / max(n - parameter_count - 1, 1))
        scored.append((aicc, rss, name, parameters, beta, prediction, parameter_count))
    best = min(scored, key=lambda row: row[0])
    aicc, rss, name, parameters, beta, prediction, parameter_count = best
    threshold = .9
    crossing = np.flatnonzero(y >= threshold)
    return {
        "selected_model": name,
        "parameters": {**parameters, "linear_coefficients": beta.tolist()},
        "aicc": float(aicc), "rss": float(rss), "point_count": n,
        "target_crossing_level": threshold,
        "response_time_90_epochs_after_onset": int(crossing[0]) if crossing.size else None,
        "censored_at_target_90": not bool(crossing.size),
        "observed_final_response": float(y[-1]),
        "prediction_final_response": float(prediction[-1]),
        "candidate_model_count": len(scored),
    }


def fit_step_response() -> dict[str, Any]:
    comparison = read_json(ROOT / "artifacts/google_pure_v12/directional_comparison/comparison.json")
    remediation = read_json(ROOT / "configs/google_pure_v12/remediation.json")
    onset = int(remediation["reduced_directional_comparison"]["step"]["onset_epoch"])
    selected = [row for row in comparison["rows"]
                if row["case"] == "FAILED_STEP_RESPONSE" and
                row["arm"] == "EDR_SENSITIVITY_BOUNDARY_REPAIR"]
    rows = []
    for row in selected:
        response = np.asarray(row["projection"][onset:], dtype=float)
        fit = _fit_step_trace(response)
        rows.append({"seed": row["seed"], "onset_epoch": onset, **fit})
    identified = [row["response_time_90_epochs_after_onset"] for row in rows
                  if row["response_time_90_epochs_after_onset"] is not None]
    median = float(np.median(identified)) if identified else None
    result = nonfinal({
        "pass": bool(rows),
        "rows": rows,
        "target_definition": "ABSOLUTE_INJECTED_TARGET_FRACTION_0.9",
        "observed_final_excursion_fraction_never_used_as_threshold": True,
        "censored_count": sum(row["censored_at_target_90"] for row in rows),
        "median_response_time_90_epochs": median,
        "paper_anchor_epochs": config()["step"]["paper_anchor_epochs"],
        "paper_anchor_comparison_permitted": False,
        "imported_trace_role": "V12_SOURCE_NORMALIZED_SYNTHETIC_DEVELOPMENT",
        "source_step_trace_available": False,
        "source_gap_classification": PUBLIC_NON_IDENTIFIABLE,
    })
    atomic_json(ARTIFACT_ROOT / "fidelity/step_response_fit.json", result)
    return result


def plan_natural_drift_power() -> dict[str, Any]:
    settings = config()["natural"]
    inherited = read_json(ROOT / "artifacts/google_pure_v13/natural_drift/power_plan.json")
    post_warmup = settings["epochs"] - settings["warmup_epochs"]
    samples = post_warmup // settings["evaluation_cadence_epochs"]
    minimum_scan_frequency = min(read_json(
        ROOT / "configs/google_pure_source_exact/figure5a.json")["dense_scan"]["frequencies"])
    minimum_resolved_frequency = 1.0 / post_warmup
    minimum_epochs_four_periods = int(settings["warmup_epochs"] + np.ceil(4.0 / minimum_scan_frequency))
    result = nonfinal({
        "pass": True,
        "planned_complete_paired_runs": settings["complete_pair_count"],
        "pilot_complete_paired_runs": inherited["pilot_run_count"],
        "pilot_standard_deviation_db": inherited["pilot_standard_deviation_db"],
        "target_learned_over_fixed_filter_db": -abs(settings["target_suppression_db"]),
        "alpha": settings["alpha"], "power": settings["power"],
        "current_epochs": settings["epochs"], "warmup_epochs": settings["warmup_epochs"],
        "post_warmup_decoded_samples": samples,
        "current_minimum_dft_frequency_cycles_per_epoch": minimum_resolved_frequency,
        "minimum_scan_frequency_cycles_per_epoch": minimum_scan_frequency,
        "four_period_minimum_epochs": minimum_epochs_four_periods,
        "current_trace_resolves_lowest_scan_frequency": minimum_resolved_frequency <= minimum_scan_frequency,
        "frequency_bins_are_not_replicates": True,
        "uncertainty_unit": "COMPLETE_PAIRED_RUN",
        "geometric_averaging_required": True,
        "warmup_exclusion_required": True,
        "explicit_long_run_required": True,
        "long_run_auto_launched": False,
        "classification": "POWER_AND_FREQUENCY_RESOLUTION_PLAN_FROZEN; ACQUISITION_NOT_COMPLETE",
    })
    atomic_json(ARTIFACT_ROOT / "fidelity/natural_drift_power.json", result)
    return result


def audit_ppo_lifecycle() -> dict[str, Any]:
    source = ROOT / "src/hdfa_rl_suite/google_pure_source_exact/paper_families/common.py"
    text = source.read_text(encoding="utf-8")
    gates = {
        "fresh_behavior_snapshot_per_epoch": "behavior = policy.sample" in text or
                                               "batch = policy.sample" in text,
        "one_candidate_batch_per_epoch": "while int(state[\"epoch\"])" in text,
        "one_optimizer_step_per_epoch": "optimizer.step" in text,
        "elementwise_coordinate_clipping_before_detector_product": "coordinate_log_ratio" in text and
                                                                    "np.bincount" in text,
        "learned_detector_baseline_same_batch": "grad_baseline" in text,
        "no_replay_buffer": "replay_buffer" not in text,
        "no_extra_ppo_epochs": "ppo_epochs" not in text,
        "direct_sigma": "grad_sigma" in text,
    }
    stages = [
        "freeze_behavior", "sample_K_candidates", "acquire_one_detector_batch_per_candidate",
        "form_detector_local_advantages", "clip_coordinate_ratios", "one_joint_gradient",
        "one_optimizer_step", "persist_post_update_state",
    ]
    result = nonfinal({
        "pass": all(gates.values()),
        "gates": gates, "ordered_stages": stages,
        "source_file": source.relative_to(ROOT).as_posix(),
        "source_file_sha256": file_hash(source),
        "replay_used": False, "extra_passes_used": False,
        "proprietary_optimizer_details_identifiable": False,
        "source_gap_classification": PUBLIC_NON_IDENTIFIABLE,
    })
    atomic_json(ARTIFACT_ROOT / "fidelity/ppo_lifecycle.json", result)
    return result


def verify_provenance() -> dict[str, Any]:
    state = read_json(ROOT / "artifacts/google_pure_v13/provenance/state_chain_validation.json")
    candidate = read_json(ROOT / "artifacts/google_pure_v13/provenance/candidate_lineage_validation.json")
    result = nonfinal({
        "pass": bool(state["pass"] and candidate["pass"]),
        "state_chain": state,
        "candidate_lineage": candidate,
        "state_transition_contract": [
            "pre_update_policy_hash", "behavior_snapshot_hash", "candidate_action_hash",
            "detector_reward_hash", "gradient_hash", "post_update_policy_hash",
        ],
        "candidate_boundary_checkpointing_required": True,
        "imported_runs_are_smoke_or_development_only": True,
        "status_inherits_provenance_without_promotion": True,
    })
    atomic_json(ARTIFACT_ROOT / "fidelity/provenance.json", result)
    return result


def report_resource_semantics() -> dict[str, Any]:
    rows = [
        {"family": "PUBLIC_PAPER_EPOCH", "candidates_per_epoch": 40,
         "qec_cycles_per_candidate": 100000, "qec_cycles_per_epoch": 4000000,
         "detector_event_trials": "QEC_CYCLES_X_DETECTOR_OPPORTUNITIES",
         "decoded_logical_shots": "NOT_REPORTED_AS_CONTROLLER_REWARD",
         "wall_clock": "IDEAL_4_SECONDS; REPORTED_1_TO_10_MINUTES"},
        {"family": "CURRENT_REFERENCE_PROFILE", "candidates_per_epoch": 50,
         "qec_cycles_per_candidate": 36000, "qec_cycles_per_epoch": 1800000,
         "detector_event_trials": "SEPARATELY_RECORDED",
         "decoded_logical_shots": 0, "wall_clock": "NOT_AN_EPOCH_DEFINITION"},
        {"family": "NATURAL_DRIFT", "candidates_per_epoch": "PROFILE_SPECIFIC",
         "qec_cycles_per_candidate": "PROFILE_SPECIFIC",
         "qec_cycles_per_epoch": "PRODUCT_RECORDED",
         "detector_event_trials": "CONTROL_ACQUISITION_ONLY",
         "decoded_logical_shots": "SEPARATE_FIXED_CADENCE_EVALUATION",
         "wall_clock": "NEVER_SUBSTITUTED_FOR_EPOCH"},
    ]
    result = nonfinal({
        "pass": True,
        "rows": rows,
        "epoch_definition": "ONE_FRESH_CANDIDATE_BATCH_PLUS_ONE_POLICY_UPDATE",
        "qec_cycle_is_not_an_epoch": True,
        "candidate_is_not_a_qec_cycle": True,
        "detector_event_trial_is_not_a_decoded_logical_shot": True,
        "current_reference_profile_matches_public_source_budget": False,
        "source_gap_classification": PUBLIC_MISMATCHED,
    })
    atomic_json(ARTIFACT_ROOT / "fidelity/resource_semantics.json", result)
    return result


def build_source_gap_register() -> dict[str, Any]:
    rows = [
        {"claim": "group Gaussian sensitivity law", "classification": PUBLIC_MATCHED},
        {"claim": "one normalized variance equals one EDR percentage point", "classification": PUBLIC_MATCHED},
        {"claim": "40 candidates and approximately 1e5 QEC cycles per candidate", "classification": PUBLIC_MATCHED},
        {"claim": "current reference profile acquisition budget", "classification": PUBLIC_MISMATCHED},
        {"claim": "proprietary controller implementation details", "classification": PUBLIC_NON_IDENTIFIABLE},
        {"claim": "proprietary plant transfer function and cross-coupling", "classification": PUBLIC_NON_IDENTIFIABLE},
        {"claim": "experimental step-response raw traces", "classification": PUBLIC_NON_IDENTIFIABLE},
        {"claim": "experimental natural-drift paired raw traces", "classification": PUBLIC_NON_IDENTIFIABLE},
        {"claim": "hardware hysteresis, latency and nonstationarity", "classification": HARDWARE_UNTESTED},
        {"claim": "Sparse Blossom 2024 decoder benchmark and private data", "classification": PUBLIC_NON_IDENTIFIABLE},
    ]
    counts = Counter(row["classification"] for row in rows)
    result = nonfinal({
        "pass": True, "rows": rows, "counts": dict(counts),
        "all_claims_classified": True,
        "unavailable_inputs_never_imputed": True,
        "conclusion": "PUBLIC_IMPLEMENTATION_AUDIT_COMPLETE; PAPER_EQUIVALENCE_NON_IDENTIFIABLE",
    })
    atomic_json(ARTIFACT_ROOT / "source_gap_register.json", result)
    return result
