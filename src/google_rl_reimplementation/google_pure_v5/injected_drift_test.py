"""Dedicated deliberate injected-drift stability experiment."""
from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np

from .accounting import acquisition_accounting
from .config import load_config, paper_scale, source_choices
from .experiments import percentile_interval, run_matched_trace
from .plant import PurePlantSpec, PureQuadraticPlant
from .reporting import read_artifact, write_report


DISTINCT_EVIDENCE_STATEMENT = (
    "The injected-drift stability statistic is derived from deliberate synthetic interventions "
    "and is not the same experiment as the natural-drift spectral analysis."
)


def generate_injected_tape(
    scenario: Mapping[str, Any], horizon: int, onset: int, control_count: int
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Generate only labelled step, sinusoidal, or stroboscopic interventions."""
    profile = str(scenario["profile"])
    if profile not in {"step", "sinusoidal", "stroboscopic"}:
        raise ValueError("unknown labelled injected profile")
    location = int(scenario["location"])
    category = str(scenario["category"])
    category_offset = {"CZ coupling strength": 0, "XY pulse amplitude": 1, "XY pulse frequency": 0}[category]
    primary = (2 * location + category_offset) % control_count
    affected = (primary, (primary + 1) % control_count) if category == "CZ coupling strength" else (primary,)
    t = np.arange(horizon, dtype=float)
    active_t = np.maximum(t - onset, 0.0)
    amplitude = float(scenario["amplitude"])
    if profile == "step":
        value = np.where(t >= onset, amplitude, 0.0)
    elif profile == "sinusoidal":
        value = np.where(
            t >= onset,
            amplitude * np.sin(2.0 * np.pi * float(scenario["frequency"]) * active_t + float(scenario["phase"])),
            0.0,
        )
    else:
        sinusoid = np.sin(2.0 * np.pi * float(scenario["frequency"]) * active_t + float(scenario["phase"]))
        value = np.where(t >= onset, amplitude * np.where(sinusoid >= 0.0, 1.0, -1.0), 0.0)
    tape = np.zeros((horizon, control_count), dtype=float)
    for index in affected:
        tape[:, index] = value
    return tape, affected


def _fit_step_response(error: np.ndarray, onset: int) -> dict[str, Any]:
    values = np.asarray(error, dtype=float)
    tail = values[onset:]
    asymptote = float(np.median(tail[-max(12, len(tail) // 8) :]))
    amplitude = max(float(tail[0] - asymptote), 1e-12)
    normalized = (tail - asymptote) / amplitude
    x = np.arange(len(tail), dtype=float)
    valid = (normalized > 0.05) & (normalized < 1.05)
    valid[:3] = False
    if valid.sum() < 5:
        tau, r2 = float("inf"), 0.0
    else:
        slope, intercept = np.polyfit(x[valid], np.log(normalized[valid]), 1)
        predicted = intercept + slope * x[valid]
        observed = np.log(normalized[valid])
        denom = float(np.sum((observed - observed.mean()) ** 2))
        r2 = 1.0 - float(np.sum((observed - predicted) ** 2)) / max(denom, 1e-15)
        tau = -1.0 / slope if slope < 0 else float("inf")
    crossing_indices = np.flatnonzero(normalized <= np.exp(-1.0))
    crossing = float(crossing_indices[0]) if len(crossing_indices) else float(len(tail))
    return {
        "onset_epoch": onset,
        "fitted_tail_start_epoch": onset + 3,
        "response_amplitude": amplitude,
        "characteristic_time_epochs": float(tau),
        "fit_r_squared": float(r2),
        "one_over_e_crossing_epochs": crossing,
        "asymptotic_error": asymptote,
    }


def run_injected_drift(epochs: int | None = None) -> dict[str, Any]:
    config = load_config("injected_drift_stability.yaml")
    horizon = int(config["horizon_epochs"] if epochs is None else epochs)
    if horizon < 120:
        raise ValueError("injected-drift development requires at least 120 epochs")
    onset = min(int(config["onset_epoch"]), max(20, horizon // 5))
    choices, paper = source_choices(), paper_scale()
    scenarios = []
    raw_hashes = []
    step_times = []
    for index, scenario in enumerate(config["scenarios"]):
        spec = PurePlantSpec(
            str(scenario["id"]), detector_count=int(config["detectors"]), control_count=int(config["controls"]),
            curvature=float(config["curvature"]), detector_floor=float(config["detector_floor"]),
            logical_floor=float(config["logical_floor"]), logical_gain=float(config["logical_gain"]),
            draw_seed=7100 + index,
        )
        plant = PureQuadraticPlant(spec)
        tape, affected = generate_injected_tape(scenario, horizon, onset, spec.control_count)
        raw_hash = hashlib.sha256(np.asarray(tape, dtype="<f8").tobytes()).hexdigest()
        raw_hashes.append(raw_hash)
        result = run_matched_trace(
            plant, tape, choices, paper, seed=int(config["development_seeds"][index])
        )
        ler = result["logical_risk"]
        detector = result["detector_rate"]
        start = max(0, onset // 2) if scenario["profile"] == "step" else int(horizon * float(config["primary_window_start_fraction"]))
        window = slice(start, None)
        fixed_std = float(np.std(ler["fixed_policy"][window], ddof=1))
        mean_std = float(np.std(ler["learned_mean"][window], ddof=1))
        stochastic_std = float(np.std(ler["stochastic_candidates"][window], ddof=1))
        fixed_mean = float(np.mean(ler["fixed_policy"][window]))
        learned_mean = float(np.mean(ler["learned_mean"][window]))
        stochastic_mean = float(np.mean(ler["stochastic_candidates"][window]))
        step_fit = None
        if scenario["profile"] == "step":
            error = np.mean(np.abs(result["learned_mean_vectors"][:, affected] - tape[:, affected]), axis=1)
            step_fit = _fit_step_response(error, onset)
            if np.isfinite(step_fit["characteristic_time_epochs"]):
                step_times.append(float(step_fit["characteristic_time_epochs"]))
        scenarios.append({
            "scenario_id": scenario["id"], "profile": scenario["profile"], "category": scenario["category"],
            "location": scenario["location"], "affected_controls": list(affected), "raw_trace_hash": raw_hash,
            "initial_ler": {name: float(values[0]) for name, values in ler.items()},
            "final_ler": {name: float(np.mean(values[-20:])) for name, values in ler.items()},
            "mean_ler": {"fixed_policy": fixed_mean, "learned_mean": learned_mean,
                         "stochastic_candidates": stochastic_mean, "oracle_optimum": float(np.mean(ler["oracle_optimum"][window]))},
            "ler_standard_deviation": {"fixed_policy": fixed_std, "learned_mean": mean_std,
                                       "stochastic_candidates": stochastic_std, "oracle_optimum": float(np.std(ler["oracle_optimum"][window], ddof=1))},
            "control_only_stability_ratio": fixed_std / max(mean_std, 1e-15),
            "stochastic_operational_stability_ratio": fixed_std / max(stochastic_std, 1e-15),
            "relative_mean_ler_improvement": (fixed_mean - learned_mean) / max(fixed_mean, 1e-15),
            "relative_detector_rate_improvement": float((np.mean(detector["fixed_policy"][window]) - np.mean(detector["learned_mean"][window])) / max(np.mean(detector["fixed_policy"][window]), 1e-15)),
            "exploration_damage": stochastic_mean - learned_mean,
            "step_response": step_fit,
        })
    ratios = [float(row["control_only_stability_ratio"]) for row in scenarios]
    improvements = [float(row["relative_mean_ler_improvement"]) for row in scenarios]
    ratio_band = list(map(float, config["stability_ratio_band"]))
    improvement_band = list(map(float, config["mean_ler_improvement_band"]))
    step_band = list(map(float, config["step_time_constant_band_epochs"]))
    median_ratio = float(np.median(ratios))
    median_improvement = float(np.median(improvements))
    median_step = float(np.median(step_times)) if step_times else float("inf")
    checks = {
        "all_primary_scenarios_correct_direction": all(value > 1.0 for value in ratios),
        "median_stability_source_band": ratio_band[0] <= median_ratio <= ratio_band[1],
        "material_mean_ler_improvement": improvement_band[0] <= median_improvement <= improvement_band[1],
        "step_response_source_band": step_band[0] <= median_step <= step_band[1],
        "all_profiles_present": {row["profile"] for row in scenarios} == {"step", "sinusoidal", "stroboscopic"},
        "multiple_locations": len({row["location"] for row in scenarios}) >= 3,
        "four_policy_classes_separate": all(set(row["mean_ler"]) == {"fixed_policy", "learned_mean", "stochastic_candidates", "oracle_optimum"} for row in scenarios),
        "exploration_damage_separate": all("exploration_damage" in row for row in scenarios),
    }
    accounting = acquisition_accounting(horizon * len(scenarios), paper, mean_evaluations=horizon * len(scenarios), fixed_evaluations=horizon * len(scenarios), logical_evaluations=4 * horizon * len(scenarios))
    payload = {
        "schema_version": "google-pure-v5-injected-drift.v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "distinct_evidence_statement": DISTINCT_EVIDENCE_STATEMENT,
        "primary_metric": "control-only LER standard-deviation ratio",
        "public_anchor_scope": "approximately 2.4x control steering only; excludes decoder steering",
        "aggregate": {
            "median_control_only_stability_ratio": median_ratio,
            "stability_ratio_95_percent_interval_across_scenarios": percentile_interval(ratios),
            "median_relative_mean_ler_improvement": median_improvement,
            "mean_ler_improvement_95_percent_interval_across_scenarios": percentile_interval(improvements),
            "median_step_response_epochs": median_step,
            "step_response_range_epochs": [float(min(step_times)), float(max(step_times))] if step_times else [],
        },
        "checks": checks,
        "scenarios": scenarios,
        "raw_trace_hashes": raw_hashes,
        "accounting": accounting,
        "decoder_steering_included": False,
        "certification_seeds_consumed": False,
    }
    write_report("injected_drift_stability", payload, "Injected-drift stability")
    return payload


def run_step_response() -> dict[str, Any]:
    try:
        injected = read_artifact("injected_drift_stability")
    except RuntimeError:
        injected = run_injected_drift()
    step_rows = [row for row in injected["scenarios"] if row["profile"] == "step"]
    payload = {
        "schema_version": "google-pure-v5-step-response.v1",
        "status": "PASS" if injected["checks"]["step_response_source_band"] else "FAIL",
        "shared_mechanism_only": True,
        "source_artifact": "injected_drift_stability.json",
        "fits": [row["step_response"] for row in step_rows],
        "aggregate": injected["aggregate"],
        "certification_seeds_consumed": False,
    }
    write_report("step_response", payload, "Injected-drift step-response mechanism check")
    return payload
