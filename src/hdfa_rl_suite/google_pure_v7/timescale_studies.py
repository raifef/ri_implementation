"""Long-step and timescale-matched sine/strobe production studies."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from hdfa_rl_suite.google_pure_v6.metrics import stability_metrics
from hdfa_rl_suite.google_pure_v6.plant import PureQuadraticPlant, default_spec, optimum_tape

from .config import canonical_hash, guard_seed, load_config
from .controller import CONTROLLER_MODE, require_resolved_controller
from .experiments import run_production_trace, trace_summary
from .reporting import read_artifact, write_report
from .response import estimate_step_response
from .sine import InvalidSineDiagnostic, classify_bandwidth_cutoff, fit_sine_tracking


def command_cost(*, epochs: int, candidates: int, cycles: int, controls: int = 6) -> dict[str, Any]:
    controller = require_resolved_controller()
    return {
        "resolved_controller_hash": controller["resolved_config_hash"],
        "expected_wall_time": "simulator-dependent; full acquisition intentionally user-triggered",
        "candidate_count": int(epochs * candidates),
        "epoch_count": int(epochs),
        "qec_cycle_count": int(epochs * candidates * cycles),
        "memory_estimate_bytes": int(max(1, epochs) * (controls * 16 + 256)),
        "disk_estimate_bytes": int(max(1, epochs) * (controls * 48 + 2048)),
        "certification_seed_status": "active seeds 12101-12112 untouched",
    }


def _direction(controls: int) -> np.ndarray:
    value = np.linspace(1.0, 0.45, controls)
    return value / np.linalg.norm(value)


def _integrated_excess_ratio(result: dict[str, Any], onset: int = 0) -> tuple[float | None, float, float]:
    fixed = result["logical_risk"]["fixed_policy"] - result["logical_risk"]["oracle_optimum"]
    mean = result["logical_risk"]["learned_mean"] - result["logical_risk"]["oracle_optimum"]
    denominator = float(np.sum(np.abs(fixed[onset:])))
    numerator = float(np.sum(np.abs(mean[onset:])))
    return (numerator / denominator if denominator > 1e-12 else None), numerator, denominator


def run_long_step(*, smoke: bool, execute: bool = False, seed: int = 7201,
                  epochs: int | None = None) -> dict[str, Any]:
    guard_seed(seed)
    controller = require_resolved_controller()
    config = load_config("long_step_response.yaml")
    if not smoke and not execute:
        raise RuntimeError("long step is a user-run experiment; pass --execute after reviewing the cost preflight")
    horizon = int(epochs or config["smoke_horizon_epochs" if smoke else "full_horizon_epochs"])
    candidates = int(config["candidates_per_epoch_smoke" if smoke else "candidates_per_epoch_full"])
    cycles = int(config["effective_cycles_per_candidate_smoke" if smoke else "effective_cycles_per_candidate_full"])
    plant = PureQuadraticPlant(default_spec(6))
    rows = []
    onset = int(round(float(config["onset_fraction"]) * horizon))
    for index, (label, amplitude) in enumerate((
        ("small_local_step", float(config["small_step_amplitude"])),
        ("moderate_local_step", float(config["moderate_step_amplitude"])),
    )):
        tape = optimum_tape("step", horizon, amplitude, controls=6, seed=seed + index)
        result = run_production_trace(plant, tape, seed=seed + index, candidates=candidates, cycles=cycles)
        projected_mean = result["learned_mean_vectors"] @ _direction(6)
        response = estimate_step_response(
            projected_mean, onset_epoch=onset, target=amplitude,
            settling_relative_tolerance=float(config["settling_relative_tolerance"]),
            sustained_epochs=int(config["minimum_sustained_settling_epochs"]),
            tau_grid_points=int(config["fit_profile_grid_points"]),
            tau_max=float(config["fit_profile_tau_max_epochs"]),
        )
        ratio, numerator, denominator = _integrated_excess_ratio(result, onset)
        response.update({
            "step_label": label,
            "step_amplitude": amplitude,
            "integrated_excess_error_ratio_mean_over_fixed": ratio,
            "learned_integrated_absolute_excess": numerator,
            "fixed_integrated_absolute_excess": denominator,
            "denominator_identifiable": denominator > 1e-12,
            "independent_detector_acquisition": True,
            "independent_logical_evaluation": True,
            "trace_summary": trace_summary(result),
        })
        rows.append(response)
    primary = rows[0]
    fit_valid = bool(primary["exponential_fit"].get("valid"))
    nonparametric_valid = primary["response_time_63_2_epochs"] is not None
    mechanism_valid = primary["denominator_identifiable"] and (fit_valid or nonparametric_valid)
    performance_pass = bool(
        not smoke and mechanism_valid and primary["settling_time_95_epochs"] is not None
        and primary["integrated_excess_error_ratio_mean_over_fixed"] is not None
        and primary["integrated_excess_error_ratio_mean_over_fixed"] < 1.0
    )
    reasons = []
    if smoke:
        reasons.append("48/96-epoch class run is SMOKE_TEST_ONLY and excluded from scientific conclusions")
    if not mechanism_valid:
        reasons.append("response timescale is not validly identified")
    if not smoke and primary["settling_time_95_epochs"] is None:
        reasons.append("NO_SETTLING_WITHIN_HORIZON")
    if not smoke and (primary["integrated_excess_error_ratio_mean_over_fixed"] is None or
                      primary["integrated_excess_error_ratio_mean_over_fixed"] >= 1.0):
        reasons.append("primary local-step integrated excess ratio is not below one")
    payload = {
        "schema_version": "google-pure-v7-long-step.v1",
        "run_class": "SMOKE_TEST_ONLY" if smoke else "FULL_DEVELOPMENT",
        "controller_code_hash": controller["controller_code_hash"],
        "resolved_config_hash": controller["resolved_config_hash"],
        "objective_mode": CONTROLLER_MODE,
        "protocol_hash": canonical_hash(config),
        "rows": rows,
        "artifact_complete": True,
        "mechanism_valid": mechanism_valid,
        "performance_pass": performance_pass,
        "blocking_reasons": reasons,
        "cost": command_cost(epochs=horizon * 2, candidates=candidates, cycles=cycles),
        "certification_seeds_consumed": False,
        "status": "SMOKE_TEST_ONLY" if smoke else ("PASS" if performance_pass else "STUDY_COMPLETE_NO_PASSING_CONFIGURATION"),
    }
    return write_report("long_step_response_smoke" if smoke else "long_step_response", payload,
                        "Long Step Response (Smoke)" if smoke else "Long Step Response")


def freeze_timescale_sine_protocol() -> dict[str, Any]:
    step = read_artifact("long_step_response")
    if step.get("run_class") != "FULL_DEVELOPMENT" or not step.get("mechanism_valid"):
        raise RuntimeError("a mechanism-valid full long-step response is required before bandwidth labelling")
    primary = step["rows"][0]
    tau = primary["exponential_fit"].get("tau_epochs")
    if tau is None or not np.isfinite(tau) or tau <= 0:
        tau = primary.get("response_time_63_2_epochs")
    if tau is None or not np.isfinite(tau) or tau <= 0:
        raise RuntimeError("finite response timescale is unavailable")
    config = load_config("timescale_matched_sine.yaml")
    burn_in = int(math.ceil(float(config["burn_in_tau_multiple"]) * tau))
    periods = int(config["preferred_complete_periods"])
    rows = []
    for value, label in zip(config["omega_tau_grid"], config["labels"]):
        omega = float(value) / float(tau)
        period = 2.0 * np.pi / omega
        horizon = burn_in + int(math.ceil(periods * period))
        rows.append({"omega_tau": value, "label": label, "omega_radians_per_epoch": omega,
                     "period_epochs": period, "burn_in_epochs": burn_in,
                     "post_burn_in_complete_periods": periods, "horizon_epochs": horizon})
    protocol = {
        "schema_version": "google-pure-v7-timescale-sine-protocol.v1",
        "response_tau_epochs": float(tau),
        "response_tau_source_artifact_hash": canonical_hash(step),
        "resolved_config_hash": step["resolved_config_hash"],
        "amplitude": config["amplitude"],
        "rows": rows,
        "short_48_epoch_runs_classification": "SMOKE_TEST_ONLY",
        "frozen_before_full_sine_run": True,
        "certification_seeds_consumed": False,
    }
    protocol["protocol_hash"] = canonical_hash(protocol)
    protocol["status"] = "FROZEN"
    return write_report("timescale_matched_sine_protocol", protocol, "Timescale-matched Sine Protocol")


def _period_bootstrap_interval(fixed: np.ndarray, learned: np.ndarray, *, period_epochs: float,
                               seed: int, draws: int = 1000) -> list[float]:
    block = max(2, int(round(period_epochs)))
    blocks = min(len(fixed) // block, len(learned) // block)
    if blocks < 3:
        raise InvalidSineDiagnostic("uncertainty requires at least three complete period blocks")
    fixed_blocks = fixed[:blocks*block].reshape(blocks, block)
    learned_blocks = learned[:blocks*block].reshape(blocks, block)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        indices = rng.integers(0, blocks, size=blocks)
        f = fixed_blocks[indices].reshape(-1)
        m = learned_blocks[indices].reshape(-1)
        denominator = float(np.std(m, ddof=1))
        if denominator > 0:
            values.append(float(np.std(f, ddof=1) / denominator))
    if not values:
        raise InvalidSineDiagnostic("suppression uncertainty cannot be estimated")
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def _sine_case(*, horizon: int, burn_in: int, omega: float, omega_tau: float, amplitude: float,
               seed: int, candidates: int, cycles: int, run_class: str) -> dict[str, Any]:
    plant = PureQuadraticPlant(default_spec(6))
    cycles_per_run = omega * horizon / (2.0 * np.pi)
    tape = optimum_tape("sine", horizon, amplitude, cycles_per_run, controls=6, seed=seed)
    result = run_production_trace(plant, tape, seed=seed, candidates=candidates, cycles=cycles)
    direction = _direction(6)
    learned_control = result["learned_mean_vectors"] @ direction
    fit = fit_sine_tracking(np.arange(horizon), learned_control, optimum_amplitude=amplitude,
                            omega_radians_per_epoch=omega, burn_in_epochs=burn_in)
    fixed_eval = result["logical_evaluation"]["fixed_policy"][burn_in:burn_in+fit["analysis_samples"]]
    mean_eval = result["logical_evaluation"]["learned_mean"][burn_in:burn_in+fit["analysis_samples"]]
    fixed_risk = result["logical_risk"]["fixed_policy"][burn_in:]
    mean_risk = result["logical_risk"]["learned_mean"][burn_in:]
    oracle_risk = result["logical_risk"]["oracle_optimum"][burn_in:]
    stability = stability_metrics(fixed_eval, mean_eval)
    interval = _period_bootstrap_interval(fixed_eval, mean_eval, period_epochs=fit["period_epochs"], seed=seed + 9000)
    fixed_excess = fixed_risk - oracle_risk
    mean_excess = mean_risk - oracle_risk
    denominator = float(np.sum(np.abs(fixed_excess)))
    integrated_ratio = float(np.sum(np.abs(mean_excess)) / denominator) if denominator > 1e-12 else None
    mean_reduction = float(1.0 - np.mean(mean_excess) / np.mean(fixed_excess)) if np.mean(fixed_excess) > 1e-12 else None
    floor_fixed = float(np.mean(fixed_eval) * (1.0 - np.mean(fixed_eval)) / 20000.0)
    floor_mean = float(np.mean(mean_eval) * (1.0 - np.mean(mean_eval)) / 20000.0)
    corrected_fixed = float(np.sqrt(max(np.var(fixed_eval, ddof=1) - floor_fixed, 0.0)))
    corrected_mean = float(np.sqrt(max(np.var(mean_eval, ddof=1) - floor_mean, 0.0)))
    corrected_suppression = corrected_fixed / corrected_mean if corrected_mean > 0 else None
    mechanism = bool(denominator > 1e-12 and fit["status"] == "VALID_DIAGNOSTIC" and np.all(np.isfinite(interval)))
    performance = bool(mechanism and stability["stability_suppression_factor_fixed_over_mean"] > 1.0
                       and interval[0] > 1.0 and integrated_ratio is not None and integrated_ratio < 1.0)
    return {
        "run_class": run_class,
        "omega_tau": omega_tau,
        "omega_radians_per_epoch": omega,
        "period_epochs": fit["period_epochs"],
        "sine_fit": fit,
        **stability,
        "suppression_factor_confidence_interval_95": interval,
        "integrated_excess_error_ratio_mean_over_fixed": integrated_ratio,
        "mean_excess_ler_reduction": mean_reduction,
        "fixed_signal_identifiable": denominator > 1e-12,
        "measurement_floor_variance": {"fixed": floor_fixed, "learned_mean": floor_mean},
        "measurement_floor_corrected_std": {"fixed": corrected_fixed, "learned_mean": corrected_mean},
        "measurement_floor_corrected_suppression_factor": corrected_suppression,
        "trace_summary": trace_summary(result),
        "mechanism_valid": mechanism,
        "performance_pass": performance,
        "resolved_config_hash": result["resolved_config_hash"],
    }


def run_timescale_sine(*, smoke: bool, execute: bool = False, seed: int = 7301) -> dict[str, Any]:
    guard_seed(seed)
    controller = require_resolved_controller()
    config = load_config("timescale_matched_sine.yaml")
    if smoke:
        horizon, burn_in, periods = 48, 0, 4.0
        omega = 2.0 * np.pi * periods / horizon
        rows = [_sine_case(horizon=horizon, burn_in=burn_in, omega=omega, omega_tau=float("nan"),
                           amplitude=float(config["amplitude"]), seed=seed, candidates=8, cycles=3000,
                           run_class="SMOKE_TEST_ONLY")]
        # JSON forbids NaN: smoke has no measured dimensionless frequency.
        rows[0]["omega_tau"] = None
        payload = {"schema_version": "google-pure-v7-timescale-sine-smoke.v1", "run_class": "SMOKE_TEST_ONLY",
                   "rows": rows, "artifact_complete": True, "mechanism_valid": rows[0]["mechanism_valid"],
                   "performance_pass": False, "blocking_reasons": ["48-epoch run excluded from scientific conclusions"],
                   "resolved_config_hash": controller["resolved_config_hash"], "certification_seeds_consumed": False,
                   "objective_mode": CONTROLLER_MODE,
                   "status": "SMOKE_TEST_ONLY"}
        return write_report("timescale_matched_sine_smoke", payload, "Timescale-matched Sine Smoke Test")
    if not execute:
        raise RuntimeError("timescale sine sweep is a long user-run experiment; pass --execute after cost review")
    protocol = read_artifact("timescale_matched_sine_protocol")
    if protocol.get("status") != "FROZEN" or protocol["resolved_config_hash"] != controller["resolved_config_hash"]:
        raise RuntimeError("frozen sine protocol/controller mismatch")
    rows = []
    for index, row in enumerate(protocol["rows"]):
        rows.append(_sine_case(horizon=int(row["horizon_epochs"]), burn_in=int(row["burn_in_epochs"]),
                               omega=float(row["omega_radians_per_epoch"]), omega_tau=float(row["omega_tau"]),
                               amplitude=float(protocol["amplitude"]), seed=seed + index, candidates=40, cycles=100000,
                               run_class="FULL_DEVELOPMENT"))
    gains = [item["sine_fit"]["amplitude_gain"] for item in rows]
    cutoff = classify_bandwidth_cutoff([item["omega_tau"] for item in rows], gains)
    primary = next(item for item in rows if np.isclose(item["omega_tau"], config["slow_primary_omega_tau"]))
    mechanism = all(item["mechanism_valid"] for item in rows) and np.isfinite(cutoff["value"])
    performance = mechanism and primary["performance_pass"]
    reasons = [] if performance else (["invalid or null bandwidth cutoff"] if not mechanism else ["slow-sine scientific thresholds failed"])
    payload = {"schema_version": "google-pure-v7-timescale-sine.v1", "run_class": "FULL_DEVELOPMENT",
               "protocol_hash": protocol["protocol_hash"], "resolved_config_hash": controller["resolved_config_hash"],
               "objective_mode": CONTROLLER_MODE,
               "rows": rows, "bandwidth_cutoff": cutoff, "artifact_complete": True,
               "mechanism_valid": mechanism, "performance_pass": performance, "blocking_reasons": reasons,
               "certification_seeds_consumed": False,
               "status": "PASS" if performance else ("INVALID_DIAGNOSTIC" if not mechanism else "STUDY_COMPLETE_NO_PASSING_CONFIGURATION")}
    return write_report("timescale_matched_sine", payload, "Timescale-matched Sine Study")


def run_timescale_strobe(*, execute: bool = False, seed: int = 7401) -> dict[str, Any]:
    guard_seed(seed)
    if not execute:
        raise RuntimeError("timescale strobe is a long user-run experiment; pass --execute after cost review")
    controller = require_resolved_controller()
    sine_protocol = read_artifact("timescale_matched_sine_protocol")
    config = load_config("timescale_matched_strobe.yaml")
    tau = float(sine_protocol["response_tau_epochs"])
    plant = PureQuadraticPlant(default_spec(6))
    rows = []
    for index, ratio in enumerate(config["dwell_tau_ratios"]):
        dwell = max(1, int(round(float(ratio) * tau)))
        horizon = dwell * int(config["transitions"])
        tape = optimum_tape("strobe", horizon, float(config["amplitude"]), controls=6, seed=seed + index)
        result = run_production_trace(plant, tape, seed=seed + index, candidates=40, cycles=100000)
        integrated, _, denominator = _integrated_excess_ratio(result)
        stability = stability_metrics(result["logical_evaluation"]["fixed_policy"], result["logical_evaluation"]["learned_mean"],
                                      identifiable=denominator > 1e-12)
        learned = result["learned_mean_vectors"] @ _direction(6)
        transition_indices = np.arange(dwell, horizon, dwell)
        delays, settled = [], 0
        for transition in transition_indices:
            target = float(config["amplitude"] if (transition // dwell) % 2 else 0.0)
            tolerance = 0.1 * float(config["amplitude"])
            hits = np.flatnonzero(np.abs(learned[transition:min(horizon, transition+dwell)] - target) <= tolerance)
            delays.append(int(hits[0]) if len(hits) else None)
            settled += int(bool(len(hits)))
        rows.append({"dwell_tau_ratio": ratio, "dwell_epochs": dwell,
                     "integrated_excess_error_ratio_mean_over_fixed": integrated,
                     **stability, "transition_delays_epochs": delays,
                     "fraction_transitions_settled_before_next_switch": settled / max(1, len(transition_indices)),
                     "state_specific_steady_residual": {"zero_state": float(np.mean(np.abs(learned[-dwell//2:]))),
                                                         "one_state": float(np.mean(np.abs(float(config["amplitude"])-learned[dwell:dwell+dwell//2])))},
                     "denominator_identifiable": denominator > 1e-12,
                     "trace_summary": trace_summary(result)})
    primary = next(row for row in rows if np.isclose(row["dwell_tau_ratio"], config["primary_slow_dwell_tau_ratio"]))
    mechanism = all(row["denominator_identifiable"] for row in rows)
    performance = mechanism and primary["integrated_excess_error_ratio_mean_over_fixed"] < 1.0
    payload = {"schema_version": "google-pure-v7-timescale-strobe.v1", "pattern": config["pattern"],
               "symmetric_sign_flip_forbidden": True, "zero_denominator_substitution_forbidden": True,
               "cross_family_aggregation_forbidden": True, "response_tau_epochs": tau,
               "resolved_config_hash": controller["resolved_config_hash"], "rows": rows,
               "objective_mode": CONTROLLER_MODE,
               "artifact_complete": True, "mechanism_valid": mechanism, "performance_pass": performance,
               "blocking_reasons": [] if performance else ["primary slow-dwell integrated excess ratio is not below one"],
               "certification_seeds_consumed": False,
               "status": "PASS" if performance else "STUDY_COMPLETE_NO_PASSING_CONFIGURATION"}
    return write_report("timescale_matched_strobe", payload, "Timescale-matched One-sided Strobe")


def run_production_repaired_drift() -> dict[str, Any]:
    controller = require_resolved_controller()
    step, sine, strobe = (read_artifact(name) for name in ("long_step_response", "timescale_matched_sine", "timescale_matched_strobe"))
    artifacts = {"long_step": step, "timescale_sine": sine, "timescale_strobe": strobe}
    hashes = {item["resolved_config_hash"] for item in artifacts.values()}
    if hashes != {controller["resolved_config_hash"]}:
        raise RuntimeError("final repaired-drift studies used different controller hashes")
    if any(item.get("objective_mode") == "legacy_v5_component_clipping_diagnostic_only" for item in artifacts.values()):
        raise RuntimeError("legacy objective is forbidden in final production evidence")
    mechanism = all(item.get("mechanism_valid") is True for item in artifacts.values())
    performance = all(item.get("performance_pass") is True for item in artifacts.values())
    payload = {"schema_version": "google-pure-v7-production-repaired-drift.v1",
               "controller_code_hash": controller["controller_code_hash"],
               "resolved_config_hash": controller["resolved_config_hash"], "objective_mode": CONTROLLER_MODE,
               "baseline_mode": "per_detector_frozen_batch_ema", "replay_mode": "fixed_fifo_1_epoch",
               "units_mode": "latent_normalized_likelihood_bounded_native_application",
               "entropy_mode": "once_per_coordinate",
               "benchmark_protocol_hash": canonical_hash({name: item.get("protocol_hash") for name,item in artifacts.items()}),
               "component_artifact_hashes": {name: canonical_hash(item) for name,item in artifacts.items()},
               "artifact_complete": True, "mechanism_valid": mechanism, "performance_pass": performance,
               "blocking_reasons": [] if performance else ["one or more repaired production drift gates failed"],
               "certification_seeds_consumed": False,
               "status": "PASS" if performance else "STUDY_COMPLETE_NO_PASSING_CONFIGURATION"}
    return write_report("repaired_drift_production_controller", payload, "Production-controller Repaired Drift")
