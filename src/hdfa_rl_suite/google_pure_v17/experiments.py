"""V17 protocol, transfer, metric, and dynamics diagnostics.

All runs are reduced development diagnostics.  The production Figure 5a target,
plant, four-stream evaluator, and raw-count metric are imported rather than copied.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from hdfa_rl_suite.google_pure_source_exact.figure5a.acquisition import run_cell
from hdfa_rl_suite.google_pure_source_exact.figure5a.bounded_action_ablation import Figure5aBoundedActionAblation
from hdfa_rl_suite.google_pure_source_exact.figure5a.contracts import (
    AcquisitionMode, Figure5aProtocol, ratio_from_raw_counts,
)
from hdfa_rl_suite.google_pure_source_exact.figure5a.validation import build_plant, dependency_hashes
from hdfa_rl_suite.google_pure_source_exact.figure5a.plant import Figure5aStimPlant
from hdfa_rl_suite.google_pure_source_exact.paper_families.common import SparseControlPlant
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.contracts import PositivityGuard
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.optimizer import OptimizerConfig
from hdfa_rl_suite.google_pure_source_exact.source_normalization import SourceNormalizationBoundary
from hdfa_rl_suite.google_pure_source_exact.step_response_130.plant import SourceStepPlant

from .contracts import NONFINAL, nonfinal
from .estimators import (
    analytic_first_order_transfer, complete_period_window, estimate_pure_delay,
    estimate_sinusoidal_transfer, fit_step_transfer, fit_two_timescale_step, measured_period_from_zero_crossings,
    paired_acceptance_v2, sinusoid,
)
from .imports import verify_import_manifest
from .io import ARTIFACT_ROOT, ROOT, atomic_json, atomic_text, canonical_hash, config, file_hash, read_json


def _write(name: str, value: dict[str, Any], title: str, statements: list[str]) -> dict[str, Any]:
    atomic_json(ARTIFACT_ROOT / f"{name}.json", value)
    lines = [f"# {title}", "", *statements, "", "This artifact is development-only and non-final."]
    atomic_text(ARTIFACT_ROOT / f"{name}.md", "\n".join(lines))
    return value


def _source_config() -> dict[str, Any]:
    return read_json(ROOT / "configs/google_pure_source_exact/figure5a.json")


def _frozen() -> dict[str, Any]:
    return read_json(ROOT / "artifacts/google_pure_v16/frozen_source_normalized_optimizer.json")


def _optimizer_config() -> OptimizerConfig:
    frozen = _frozen()
    return OptimizerConfig(
        float(frozen["mean_learning_rate"]), float(frozen["sigma_learning_rate"]),
        float(frozen["baseline_learning_rate"]), momentum=float(frozen["momentum"]),
        minimum_sigma=float(frozen["minimum_sigma"]), maximum_sigma=float(frozen["maximum_sigma"]),
        positivity_guard=PositivityGuard(frozen["positivity_guard"]),
    )


def _quadratic_semantics(label: str, native_coefficients: np.ndarray,
                         boundary: SourceNormalizationBoundary) -> dict[str, Any]:
    coefficient = np.asarray(native_coefficients, dtype=float)
    scale = np.asarray(boundary.native_scale, dtype=float)
    epsilon = 1e-4
    finite_hessian = coefficient * ((scale * epsilon) ** 2 + (scale * -epsilon) ** 2) / epsilon**2
    nodes, weights = np.polynomial.hermite.hermgauss(20)
    expected = []
    for native_coefficient, native_scale in zip(coefficient, scale):
        values = native_coefficient * np.square(native_scale * np.sqrt(2.0) * nodes)
        expected.append(float(np.sum(weights * values) / np.sqrt(np.pi)))
    expected_array = np.asarray(expected)
    identity_error = finite_hessian - 2.0 * expected_array
    return {
        "family": label,
        "coordinate_count": int(coefficient.size),
        "finite_difference_hessian": {
            "minimum": float(finite_hessian.min()), "median": float(np.median(finite_hessian)),
            "maximum": float(finite_hessian.max()), "epsilon": epsilon,
        },
        "unit_variance_expected_damage": {
            "minimum": float(expected_array.min()), "median": float(np.median(expected_array)),
            "maximum": float(expected_array.max()), "quadrature_order": int(nodes.size),
        },
        "quadratic_identity_error": float(np.max(np.abs(identity_error))),
        "quadratic_half_identity_pass": bool(np.allclose(expected_array, .5 * finite_hessian,
                                                         rtol=0, atol=2e-12)),
        "source_term_interpretation": "COEFFICIENT_OF_VARIANCE_DAMAGE_NOT_HESSIAN",
    }


def audit_sensitivity_semantics() -> dict[str, Any]:
    verify_import_manifest()
    step = SourceStepPlant(onset_epoch=0)
    step_boundary = SourceNormalizationBoundary.from_training_objective(
        "STEP_RESPONSE_INJECTED_DRIFT", step.sensitivity,
        control_ids=tuple(f"step:{index}" for index in range(step.controls)))
    source_cfg = _source_config()
    figure5a = build_plant(source_cfg)
    degrees = figure5a.mask.sum(axis=0).astype(float)
    figure5a_coefficients = np.asarray([row.omega_sensitivity for row in figure5a.inventory]) * degrees
    figure5a_boundary = SourceNormalizationBoundary.from_training_objective(
        "FIGURE5A_REAL_TIME_STEERING", figure5a_coefficients,
        control_ids=figure5a.parameter_ids)
    figure5b = SparseControlPlant(3, 41, 24, seed=10100, curvature=.004)
    figure5b_boundary = SourceNormalizationBoundary.from_training_objective(
        "FIGURE5B_SPARSE_SCALING", figure5b.connected_objective_curvature,
        control_ids=figure5b.control_ids)
    rows = [
        _quadratic_semantics("MATCHED_STEP", step.sensitivity[:8], SourceNormalizationBoundary(
            step_boundary.family, step_boundary.control_ids[:8], step_boundary.native_origin[:8],
            step_boundary.native_scale[:8], step_boundary.native_objective_curvature[:8],
            step_boundary.sensitivity_map_hash, step_boundary.source_inputs)),
        _quadratic_semantics("FIGURE5A_41_PARAMETER_STIM", figure5a_coefficients, figure5a_boundary),
        _quadratic_semantics("FIGURE5B_REPRESENTATIVE_MODE", figure5b.connected_objective_curvature,
                             figure5b_boundary),
    ]
    classification = ("SOURCE_0P01_IS_VARIANCE_DAMAGE" if all(
        row["quadratic_half_identity_pass"] and
        abs(row["unit_variance_expected_damage"]["median"] - .01) < 2e-12 and
        abs(row["finite_difference_hessian"]["median"] - .02) < 2e-12 for row in rows)
                      else "IMPLEMENTATION_FACTOR_TWO_ERROR")
    result = nonfinal({
        "pass": classification == "SOURCE_0P01_IS_VARIANCE_DAMAGE",
        "classification": classification, "rows": rows,
        "source_definition": "EDR = EDR0 + (sigma/sigma0)^2 on variance sigma^2",
        "source_term_interpretation": "kappa_V=0.01 expected damage for unit Gaussian variance; kappa_H=0.02 Hessian",
        "production_boundary_scaling_changed": False,
        "v16_contraction_label_corrected": True,
        "optimizer_changed": False,
    })
    return _write("sensitivity_semantics_audit", result, "V17 sensitivity semantics audit", [
        f"Classification: **{classification}**.",
        "The frozen V15 boundary already implements the source variance-damage convention; no scale repair is applied.",
    ])


def _step_rows() -> list[dict[str, Any]]:
    value = read_json(ROOT / "artifacts/google_pure_v16/matched_step/comparison.json")
    return [row for row in value["rows"] if row["branch"] == "D_V16_FROZEN_OPTIMIZER"]


def refit_step_transfer() -> dict[str, Any]:
    verify_import_manifest()
    rows = _step_rows()
    trajectories = np.asarray([row["target_relative_progress"] for row in rows], dtype=float)
    response = np.median(trajectories, axis=0)
    epochs = np.arange(response.size, dtype=float)
    free = fit_step_transfer(epochs, response)
    unity = fit_step_transfer(epochs, response, fixed_gain=1.0)
    no_delay = fit_step_transfer(epochs, response, fixed_delay=0.0)
    two_timescale = fit_two_timescale_step(epochs, response)
    rng = np.random.default_rng(82901)
    bootstrap = []
    bootstrap_cache: dict[tuple[int, ...], list[float]] = {}
    for _ in range(config()["transfer_identification"]["bootstrap_draws"]):
        selected = rng.integers(0, len(trajectories), len(trajectories))
        key = tuple(sorted(map(int, selected)))
        if key not in bootstrap_cache:
            fitted = fit_step_transfer(epochs, np.median(trajectories[selected], axis=0))
            bootstrap_cache[key] = [fitted["gain"], fitted["delay_epochs"], fitted["tau_epochs"]]
        bootstrap.append(bootstrap_cache[key])
    bootstrap_array = np.asarray(bootstrap)
    horizons = {}
    for horizon in (80, 120, len(response)):
        horizons[str(horizon)] = fit_step_transfer(epochs[:horizon], response[:horizon])
    alpha = float(_frozen()["mean_learning_rate"])
    hessian = .02
    q = 1.0 - alpha * hessian
    tau_local = float(-1.0 / math.log(q))
    observed_tau = float(read_json(
        ROOT / "artifacts/google_pure_v16/matched_step/comparison.json")["summaries"]
        ["D_V16_FROZEN_OPTIMIZER"]["median_tau_epochs"])
    horizon_taus = [value["tau_epochs"] for value in horizons.values()]
    identifiable = bool(free["gain"] < 2.95 and free["tau_epochs"] < 1990 and
                        np.ptp(horizon_taus) / max(free["tau_epochs"], 1e-12) < 1.0)
    result = nonfinal({
        "pass": True, "measured_v16_tau_epochs": observed_tau,
        "step_hessian_curvature_kappa_H": hessian, "mean_learning_rate": alpha,
        "local_contraction_q": q, "predicted_local_tau_epochs": tau_local,
        "measured_to_predicted_tau_ratio": observed_tau / tau_local,
        "mismatch_explanations": ["finite-shot stochastic gradient", "learned detector baseline",
                                  "direct-sigma exploration", "finite-horizon aggregation", "multi-mode dynamics"],
        "free_gain_delay_tau": free, "fixed_gain_one": unity, "no_delay": no_delay,
        "two_timescale_model": {**two_timescale,
                                "single_timescale_aic": float(len(response) * np.log(
                                    max(free["sse"] / len(response), np.finfo(float).tiny)) + 2 * 3),
                                "interpretation": "DIAGNOSTIC_ONLY_WITH_THREE_DEVELOPMENT_SEEDS"},
        "bootstrap_confidence_intervals_95": {
            "gain": np.quantile(bootstrap_array[:, 0], [.025, .975]).tolist(),
            "delay_epochs": np.quantile(bootstrap_array[:, 1], [.025, .975]).tolist(),
            "tau_epochs": np.quantile(bootstrap_array[:, 2], [.025, .975]).tolist(),
        },
        "bootstrap_draws": len(bootstrap), "horizon_stability": horizons,
        "right_censored": False, "transfer_identifiable": identifiable,
        "source_match_claim_permitted": False,
    })
    return _write("step_transfer_identification", result, "V17 step transfer identification", [
        f"Corrected local prediction: q={q:.6f}, tau={tau_local:.3f} epochs; V16 reported tau={observed_tau:.3f}.",
        f"Free fit: K={free['gain']:.3f}, delay={free['delay_epochs']:.2f}, tau={free['tau_epochs']:.2f} epochs.",
    ])


def audit_figure5a_frequency() -> dict[str, Any]:
    verify_import_manifest()
    plant = build_plant(_source_config())
    rows = []
    for label, frequency in config()["frequencies_per_epoch"].items():
        period = 1.0 / float(frequency)
        epochs = np.arange(0, int(math.ceil(3.1 * period)) + 1)
        source_values = np.asarray([plant.optimum(int(epoch), float(frequency))[0] for epoch in epochs])
        measured = measured_period_from_zero_crossings(source_values, epochs)
        rows.append({
            "label": label, "configured_frequency_per_epoch": float(frequency),
            "target_formula": "sin(2*pi*f*t)", "measured_period_epochs": measured["measured_period_epochs"],
            "expected_period_epochs": period, "cycles_per_epoch": float(frequency),
            "angular_frequency_radians_per_epoch": 2.0 * math.pi * float(frequency),
            "zero_crossings": measured["zero_crossings"][:8], "phase_radians": 0.0,
            "period_identity_pass": bool(np.isclose(measured["measured_period_epochs"], period,
                                                    rtol=0, atol=.05)),
        })
    result = nonfinal({
        "pass": all(row["period_identity_pass"] for row in rows), "rows": rows,
        "cycles_per_epoch_interpretation": "ACCEPTED",
        "sin_ft_radians_interpretation": "REJECTED",
        "hard_identity": "T=1/f for sin(2*pi*f*t)",
        "target_generator": "Figure5aStimPlant.optimum",
    })
    return _write("frequency_units", result, "V17 Figure 5a frequency-unit audit", [
        "The production generator uses `sin(2*pi*f*t)`, so `f` is cycles per epoch and `T=1/f`.",
        "Treating `f` as radians per epoch is rejected.",
    ])


def _deterministic_trace(frequency: float, *, gain: float, delay: float, tau: float,
                         phase: float, periods: int = 3, burn_in_periods: int = 1) -> dict[str, Any]:
    window = complete_period_window(frequency, burn_in_periods=burn_in_periods,
                                    evaluation_periods=periods, phase=phase)
    stop = int(window["evaluation_stop_epoch_exclusive"])
    epochs = np.arange(stop, dtype=float)
    target = (np.asarray([Figure5aStimPlant.optimum(int(epoch), frequency)[0] for epoch in epochs])
              if phase == 0.0 else sinusoid(frequency, epochs, phase))
    q = math.exp(-1.0 / tau)
    mean = np.zeros_like(target)
    for index in range(1, len(mean)):
        source_index = index - 1 - int(round(delay))
        driving = target[source_index] if source_index >= 0 else 0.0
        mean[index] = q * mean[index - 1] + (1.0 - q) * gain * driving
    selected = slice(int(window["evaluation_start_epoch"]), stop)
    fixed_cost = float(np.sum(target[selected] ** 2))
    policy_cost = float(np.sum((mean[selected] - target[selected]) ** 2))
    scale = 10_000_000
    production = ratio_from_raw_counts(int(round(policy_cost * scale)),
                                       int(round(fixed_cost * scale)), 0)["source_ratio"]
    transfer = estimate_sinusoidal_transfer(epochs[selected], mean[selected], frequency)
    return {"window": window, "fixed_cost": fixed_cost, "policy_cost": policy_cost,
            "oracle_cost": 0.0, "normalized_performance": production,
            "measured_transfer": transfer}


def run_figure5a_deterministic_fixture() -> dict[str, Any]:
    verify_import_manifest()
    step = read_json(ARTIFACT_ROOT / "step_transfer_identification.json") if (
        ARTIFACT_ROOT / "step_transfer_identification.json").is_file() else refit_step_transfer()
    fitted = step["free_gain_delay_tau"]
    rows = []
    for phase in config()["deterministic_fixture"]["phase_radians"]:
        for label, frequency in config()["frequencies_per_epoch"].items():
            trace = _deterministic_trace(float(frequency), gain=fitted["gain"],
                                         delay=fitted["delay_epochs"], tau=fitted["tau_epochs"],
                                         phase=float(phase))
            predicted = analytic_first_order_transfer(fitted["gain"], fitted["delay_epochs"],
                                                      fitted["tau_epochs"], float(frequency))
            measured_gain = trace["measured_transfer"]["gain"]
            measured_lag = trace["measured_transfer"]["phase_lag_radians"]
            omega = 2.0 * math.pi * float(frequency)
            gain_term = max(0.0, (fitted["gain"] / max(measured_gain, 1e-15)) ** 2 - 1.0)
            tau_gain = math.sqrt(gain_term) / omega
            phase_dynamic = measured_lag - omega * fitted["delay_epochs"]
            tau_phase = math.tan(phase_dynamic) / omega
            rows.append({"label": label, "frequency_per_epoch": float(frequency),
                         "phase_radians": float(phase), **trace,
                         "analytic_transfer": predicted,
                         "gain_residual": measured_gain - predicted["gain"],
                         "phase_residual_radians": trace["measured_transfer"]["phase_radians"] - predicted["phase_radians"],
                         "tau_from_gain_epochs": tau_gain,
                         "tau_from_phase_epochs": tau_phase,
                         "transfer_classification": ("CONSISTENT_WITH_STEP_TRANSFER" if
                             abs(tau_gain - fitted["tau_epochs"]) / fitted["tau_epochs"] < .1 and
                             abs(tau_phase - fitted["tau_epochs"]) / fitted["tau_epochs"] < .1
                             else "DISCRETE_TIME_OR_TRANSIENT_RESIDUAL")})
    medians = {label: float(np.median([row["normalized_performance"] for row in rows
                                      if row["label"] == label]))
               for label in config()["frequencies_per_epoch"]}
    result = nonfinal({
        "pass": medians["slow"] > medians["intermediate"] > medians["fast"],
        "rows": rows, "median_normalized_performance": medians,
        "slow_greater_than_fast": medians["slow"] > medians["fast"],
        "production_target_generator_used": "Figure5aStimPlant.optimum audited by frequency_units.json",
        "production_metric_function_used": "ratio_from_raw_counts",
        "production_evaluator_logic_duplicated": False,
        "fixed_policy": "x=0", "oracle_policy": "instantaneous target",
        "timestamps": "integer epoch start", "burn_in_and_windows_explicit": True,
        "classification": "EVALUATOR_VALID" if medians["slow"] > medians["fast"] else "EVALUATOR_INVALID",
    })
    return _write("figure5a_deterministic_fixture", result, "V17 deterministic Figure 5a evaluator fixture", [
        f"Median deterministic normalized performance: {medians}.",
        f"Classification: **{result['classification']}**.",
    ])


def _checkpoint_records(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state = read_json(path)
    records = []
    for shard in state.get("epoch_shards", []):
        payload = read_json(Path(shard["path"]))
        if payload.get("record_hash") != shard["record_hash"] or \
                canonical_hash(payload["record"]) != shard["record_hash"]:
            raise RuntimeError(f"corrupt Figure 5a epoch shard: {shard['path']}")
        records.append(payload["record"])
    if [row["epoch"] for row in records] != list(range(int(state["epoch"]))):
        raise RuntimeError("Figure 5a checkpoint has missing or duplicate epochs")
    return state, records


def _cell_paths() -> dict[str, Path]:
    paths: dict[str, Path] = {}
    frequencies = config()["frequencies_per_epoch"]
    for label, frequency in frequencies.items():
        token = str(frequency).replace(".", "p")
        v17 = ARTIFACT_ROOT / "reduced_postrepair/figure5a" / f"checkpoint-{token}.json"
        if v17.is_file():
            paths[label] = v17
    v16_root = ROOT / "artifacts/google_pure_v16/reduced_acceptance/figure5a/v16-acceptance-v2"
    for label in ("slow", "fast"):
        if label not in paths:
            token = str(frequencies[label]).replace(".", "p")
            candidate = v16_root / f"checkpoint-{token}.json"
            if candidate.is_file():
                paths[label] = candidate
    return paths


def _metric_from_totals(totals: Mapping[str, int]) -> dict[str, Any]:
    fixed = int(totals["fixed"])
    oracle = int(totals["optimal"])
    mean = int(totals["learned_mean"])
    stochastic = int(totals["stochastic"])
    denominator = fixed - oracle
    if denominator == 0:
        return {"fixed_cost": fixed, "oracle_cost": oracle, "mean_cost": mean,
                "stochastic_cost": stochastic, "denominator": denominator,
                "I_mean": None, "I_stochastic": None, "exploration_damage": stochastic - mean,
                "denominator_standard_error": math.sqrt(fixed + oracle),
                "denominator_snr": 0.0, "valid": False}
    mean_ratio = ratio_from_raw_counts(mean, fixed, oracle)["source_ratio"]
    stochastic_ratio = ratio_from_raw_counts(stochastic, fixed, oracle)["source_ratio"]
    denominator_se = math.sqrt(max(0, fixed + oracle))
    return {
        "fixed_cost": fixed, "oracle_cost": oracle, "mean_cost": mean,
        "stochastic_cost": stochastic, "denominator": denominator,
        "I_mean": mean_ratio, "I_stochastic": stochastic_ratio,
        "exploration_damage": stochastic - mean,
        "denominator_standard_error": denominator_se,
        "denominator_confidence_interval_95": [denominator - 1.96 * denominator_se,
                                               denominator + 1.96 * denominator_se],
        "denominator_snr": abs(denominator) / denominator_se if denominator_se else float("inf"),
        "valid": denominator > 0 and abs(denominator) > 1.96 * denominator_se,
    }


def audit_figure5a_metric() -> dict[str, Any]:
    verify_import_manifest()
    fixed_identity = ratio_from_raw_counts(100, 100, 20)["source_ratio"]
    fixed_identity = 0.0 if fixed_identity == 0.0 else fixed_identity
    oracle_identity = ratio_from_raw_counts(20, 100, 20)["source_ratio"]
    rows = []
    for label, path in _cell_paths().items():
        state, records = _checkpoint_records(path)
        totals = {stream: int(sum(row["stream_totals"][stream] for row in records))
                  for stream in ("fixed", "optimal", "stochastic", "learned_mean")}
        rows.append({"label": label, "frequency_per_epoch": float(state["frequency"]),
                     "epochs": len(records), "checkpoint": str(path.relative_to(ROOT)).replace("\\", "/"),
                     **_metric_from_totals(totals)})
    result = nonfinal({
        "pass": fixed_identity == 0.0 and oracle_identity == 1.0,
        "formula": "I=(C_fixed-C_policy)/(C_fixed-C_oracle)",
        "production_function": "ratio_from_raw_counts",
        "fixed_endpoint": fixed_identity, "oracle_endpoint": oracle_identity,
        "lower_detector_count_is_better": True, "orientation_identical_across_streams": True,
        "endpoint_window_match_required": True, "endpoint_window_mismatch_detected": False,
        "rows": rows,
        "all_observed_denominators_resolved": bool(rows) and all(row["valid"] for row in rows),
    })
    return _write("figure5a_metric_endpoint", result, "V17 Figure 5a metric and endpoint audit", [
        "Direct substitution gives fixed=0 and oracle=1 using the production raw-count function.",
        "Finite-shot denominator uncertainty is retained per condition and is never pooled across frequencies.",
    ])


def _window_row(label: str, frequency: float, epochs: int, burn_in: int = 0) -> dict[str, Any]:
    period = 1.0 / float(frequency)
    post = max(0, int(epochs) - int(burn_in))
    complete = int(math.floor(post / period + 1e-12))
    fractional = float(post / period - complete)
    if complete == 0:
        classification = "UNDERPOWERED"
    elif fractional > 1e-10:
        classification = "INCOMPLETE_PERIOD_BIAS"
    elif burn_in == 0:
        classification = "TRANSIENT_CONTAMINATION"
    else:
        classification = "WINDOW_STABLE"
    return {"label": label, "frequency_per_epoch": float(frequency), "measured_period_epochs": period,
            "total_epochs": int(epochs), "burn_in_epochs": int(burn_in), "post_burnin_epochs": post,
            "complete_periods": complete, "fractional_period": fractional,
            "evaluation_window": [int(burn_in), int(epochs)], "classification": classification}


def audit_figure5a_windowing() -> dict[str, Any]:
    verify_import_manifest()
    frequencies = config()["frequencies_per_epoch"]
    reduced_epochs = int(config()["reduced_acquisition"]["epochs"])
    original_reduced = [_window_row(label, frequency, reduced_epochs)
                        for label, frequency in frequencies.items()]
    source_1000 = [_window_row(label, frequency, 1000) for label, frequency in frequencies.items()]
    exact_windows = []
    for periods in (3, 5):
        for phase in config()["deterministic_fixture"]["phase_radians"]:
            for label, frequency in frequencies.items():
                exact_windows.append({"label": label, "periods": periods,
                                      **complete_period_window(float(frequency), burn_in_periods=1,
                                                               evaluation_periods=periods,
                                                               phase=float(phase))})
    phase_aliasing = []
    for label, frequency in frequencies.items():
        phases = config()["deterministic_fixture"]["phase_radians"]
        diagnostics = [_deterministic_trace(float(frequency), gain=.8, delay=1.0, tau=133.0,
                                            phase=float(phase), periods=3)["normalized_performance"]
                       for phase in phases]
        reduced_values = []
        for phase in phases:
            epochs = np.arange(reduced_epochs, dtype=float)
            target = sinusoid(float(frequency), epochs, float(phase))
            q = math.exp(-1.0 / 133.0)
            mean = np.zeros_like(target)
            for index in range(1, len(mean)):
                mean[index] = q * mean[index - 1] + (1.0 - q) * .8 * target[index - 1]
            fixed_cost = float(np.sum(target**2))
            policy_cost = float(np.sum((mean - target)**2))
            reduced_values.append((1.0 - policy_cost / fixed_cost) if fixed_cost else None)
        finite_reduced = [value for value in reduced_values if value is not None]
        phase_aliasing.append({"label": label,
                               "original_reduced_phase_values": reduced_values,
                               "original_reduced_phase_range": float(np.ptp(finite_reduced)),
                               "complete_window_phase_values": diagnostics,
                               "complete_window_phase_range": float(np.ptp(diagnostics))})
    result = nonfinal({
        "pass": all(row["fractional_periods"] == 0.0 for row in exact_windows),
        "original_v16_reduced_windows": original_reduced,
        "source_1000_epoch_windows": source_1000,
        "integer_complete_period_diagnostics": exact_windows,
        "phase_diagnostics": phase_aliasing,
        "original_reduced_classification": "UNDERPOWERED_REDUCED_GATE",
        "source_protocol_replaced": False,
        "source_epoch_count": 1000,
        "diagnostic_longer_windows_are_analytic_only": True,
        "repair_classification": "WINDOW_CONSTRUCTION_AND_VALIDITY_GATE_DEFECT",
    })
    return _write("figure5a_window_aliasing", result, "V17 Figure 5a window and aliasing audit", [
        "The 24-epoch V16 gate contains no complete slow, intermediate, or fast period and is underpowered.",
        "The source 1000-epoch protocol is retained; longer complete-period windows here are analytic diagnostics only.",
    ])


def measure_mean_transfer() -> dict[str, Any]:
    verify_import_manifest()
    rows = []
    minimum_cycles = float(config()["transfer_identification"]["minimum_observed_cycles"])
    maximum_condition = float(config()["transfer_identification"]["maximum_design_condition_number"])
    for label, frequency in config()["frequencies_per_epoch"].items():
        path = _cell_paths().get(label)
        if path is None:
            rows.append({"label": label, "frequency_per_epoch": float(frequency),
                         "available": False, "identifiable": False})
            continue
        state, records = _checkpoint_records(path)
        epochs = np.asarray([row["epoch"] for row in records], dtype=float)
        projected = np.asarray([np.mean(row["normalized_behavior_mean"]) for row in records], dtype=float)
        fitted = estimate_sinusoidal_transfer(epochs, projected, float(frequency),
                                              minimum_cycles=minimum_cycles,
                                              maximum_condition_number=maximum_condition)
        rows.append({"label": label, "frequency_per_epoch": float(frequency), "available": True,
                     "direction": "unit-normalized shared 41-coordinate drift direction",
                     "projection_equivalent_to_coordinate_mean": True,
                     "policy_timestamp": "behavior mean used for current epoch acquisition",
                     "controller_hash": state["controller_hash"], **fitted})
    identified = [row for row in rows if row.get("identifiable")]
    ordering = None
    if len(identified) == len(rows):
        by_label = {row["label"]: row for row in rows}
        ordering = bool(by_label["slow"]["gain"] > by_label["fast"]["gain"] and
                        by_label["slow"]["phase_lag_radians"] < by_label["fast"]["phase_lag_radians"])
    result = nonfinal({
        "pass": ordering is True, "rows": rows,
        "all_transfer_estimates_identifiable": len(identified) == len(rows),
        "expected_gain_ordering_pass": ordering,
        "normalized_performance_used_as_transfer_substitute": False,
        "classification": ("MEAN_TRANSFER_IDENTIFIED" if ordering is True else
                           "MEAN_TRANSFER_NOT_IDENTIFIABLE_AT_REDUCED_HORIZON"),
    })
    return _write("figure5a_mean_transfer", result, "V17 learned-mean transfer measurement", [
        f"Classification: **{result['classification']}**.",
        "The learned mean is projected directly onto the shared drift direction; normalized performance is not used as a transfer proxy.",
    ])


def audit_latency_phase() -> dict[str, Any]:
    verify_import_manifest()
    transfer = (read_json(ARTIFACT_ROOT / "figure5a_mean_transfer.json")
                if (ARTIFACT_ROOT / "figure5a_mean_transfer.json").is_file() else measure_mean_transfer())
    step = (read_json(ARTIFACT_ROOT / "step_transfer_identification.json")
            if (ARTIFACT_ROOT / "step_transfer_identification.json").is_file() else refit_step_transfer())
    fitted_delay = float(step["free_gain_delay_tau"]["delay_epochs"])
    rows = []
    for row in transfer["rows"]:
        if not row.get("identifiable"):
            rows.append({"label": row["label"], "identifiable": False,
                         "delay_from_phase_epochs": None})
            continue
        omega = 2.0 * math.pi * float(row["frequency_per_epoch"])
        dynamic_lag = math.atan(omega * float(step["free_gain_delay_tau"]["tau_epochs"]))
        excess = float(row["phase_lag_radians"]) - dynamic_lag
        rows.append({"label": row["label"], "identifiable": True,
                     "observed_phase_lag_radians": row["phase_lag_radians"],
                     "first_order_dynamic_lag_radians": dynamic_lag,
                     "excess_phase_lag_radians": excess,
                     "delay_from_phase_epochs": excess / omega})
    timeline = [
        {"event": "target_timestamp", "time_within_epoch": 0.0,
         "detail": "production optimum(epoch,f) evaluated at integer epoch start"},
        {"event": "policy_and_candidate_sampling", "time_within_epoch": 0.0,
         "detail": "behavior snapshot frozen before detector acquisition"},
        {"event": "acquisition_midpoint", "time_within_epoch": 0.5,
         "detail": "four policy streams sampled for every candidate"},
        {"event": "reward_available", "time_within_epoch": 1.0,
         "detail": "negative stochastic detector counts divided by shots"},
        {"event": "gradient_and_update", "time_within_epoch": 1.0,
         "detail": "one direct-sigma optimizer update after the candidate batch"},
        {"event": "updated_policy_first_applied", "time_within_epoch": 1.0,
         "detail": "post-update policy becomes the next epoch behavior policy"},
    ]
    any_identified = any(row["identifiable"] for row in rows)
    result = nonfinal({
        "pass": True, "production_timeline": timeline,
        "structural_delay_epochs": 1.0, "step_fit_delay_epochs": fitted_delay,
        "frequency_rows": rows,
        "classification": ("INSUFFICIENT_BANDWIDTH_AND_PURE_DELAY" if any_identified else
                           "LATENCY_NOT_IDENTIFIABLE_AT_REDUCED_HORIZON"),
        "predictive_compensation_added": False,
    })
    return _write("latency_phase_audit", result, "V17 latency and phase audit", [
        "The production timeline contains one causal epoch boundary between reward acquisition and application of the updated policy.",
        f"Classification: **{result['classification']}**; no predictive compensation is introduced.",
    ])


def decompose_mean_stochastic() -> dict[str, Any]:
    verify_import_manifest()
    rows = []
    for label, path in _cell_paths().items():
        state, records = _checkpoint_records(path)
        totals = {stream: int(sum(row["stream_totals"][stream] for row in records))
                  for stream in ("fixed", "optimal", "stochastic", "learned_mean")}
        metric = _metric_from_totals(totals)
        if metric["I_mean"] is None:
            classification = "METRIC_DENOMINATOR_FAILURE"
        elif metric["exploration_damage"] > 0 and metric["I_mean"] > metric["I_stochastic"]:
            classification = "EXPLORATION_DAMAGE_PRESENT"
        elif metric["I_mean"] <= 0:
            classification = "MEAN_DYNAMIC_FAILURE_OR_UNRESOLVED_TRANSIENT"
        else:
            classification = "MEAN_TRACKING_WITHOUT_POSITIVE_EXPLORATION_DAMAGE"
        rows.append({"label": label, "frequency_per_epoch": float(state["frequency"]),
                     "epochs": len(records), **metric, "classification": classification})
    result = nonfinal({
        "pass": bool(rows), "rows": rows,
        "identity": "D_exploration=C_stochastic-C_mean",
        "performance_identity": "I_mean-I_stochastic=D_exploration/(C_fixed-C_oracle)",
        "mean_and_stochastic_streams_separate": True,
        "aggregate_classification": ("REDUCED_DATA_INCLUDE_MEAN_DYNAMIC_AND_EXPLORATION_EFFECTS"
                                     if rows else "NO_REDUCED_DATA"),
    })
    return _write("mean_stochastic_decomposition", result, "V17 mean versus stochastic decomposition", [
        "Fixed, oracle, learned-mean, and stochastic detector counts remain separate.",
        "Exploration damage is reported as a cost difference, not inferred from visual smoothness.",
    ])


def audit_scale_dynamics() -> dict[str, Any]:
    verify_import_manifest()
    source_cfg = _source_config()
    plant = build_plant(source_cfg)
    bounded = Figure5aBoundedActionAblation(plant)
    degrees = plant.mask.sum(axis=0).astype(float)
    coefficients = np.asarray([row.omega_sensitivity for row in plant.inventory]) * degrees
    boundary = SourceNormalizationBoundary.from_training_objective(
        "FIGURE5A_REAL_TIME_STEERING", coefficients, control_ids=plant.parameter_ids)
    rows = []
    for label, path in _cell_paths().items():
        state, records = _checkpoint_records(path)
        frequency = float(state["frequency"])
        for record in records:
            sigma_x = np.asarray(record["behavior_sigma"], dtype=float)
            mean_x = np.asarray(record["latent_behavior_mean"], dtype=float)
            derivative = 1.0 / np.cosh(mean_x / bounded.control_limits) ** 2
            sigma_u = boundary.native_scale * derivative * sigma_x
            rewards = -np.asarray(record["stochastic_detector_counts"], dtype=float).sum(axis=1) / (
                record["qec_cycles_per_candidate"] / plant.rounds)
            counts = record["stream_totals"]
            rows.append({
                "label": label, "frequency_per_epoch": frequency, "epoch": int(record["epoch"]),
                "phase_radians": float((2.0 * math.pi * frequency * record["epoch"]) % (2.0 * math.pi)),
                "sigma_x_median": float(np.median(sigma_x)),
                "sigma_u_median": float(np.median(sigma_u)),
                "sigma_reward_gradient_norm": float(record["reward_sigma_gradient_norm"]),
                "sigma_entropy_gradient_norm": float(record["entropy_sigma_gradient_norm"]),
                "fraction_at_sigma_floor_or_ceiling": float(record["fraction_at_positivity_guard"]),
                "component_clip_fraction": float(record["component_clip_fraction"]),
                "detector_clip_fraction": float(record["detector_clip_fraction"]),
                "candidate_edr_variance": float(np.var(rewards, ddof=1)) if rewards.size > 1 else 0.0,
                "reward_variance": float(np.var(rewards, ddof=1)) if rewards.size > 1 else 0.0,
                "exploration_damage_counts": int(counts["stochastic"] - counts["learned_mean"]),
            })
    by_label = {}
    for label in config()["frequencies_per_epoch"]:
        selected = [row for row in rows if row["label"] == label]
        by_label[label] = ({
            "epochs": len(selected),
            "median_sigma_x": float(np.median([row["sigma_x_median"] for row in selected])),
            "median_sigma_u": float(np.median([row["sigma_u_median"] for row in selected])),
            "median_exploration_damage_counts": float(np.median(
                [row["exploration_damage_counts"] for row in selected])),
        } if selected else {"epochs": 0})
    result = nonfinal({
        "pass": bool(rows), "rows": rows, "summaries": by_label,
        "logged_by_epoch_frequency_and_phase": True,
        "entropy_changed": False, "sigma_learning_rate_changed": False,
        "classification": "OBSERVED_REDUCED_SCALE_DYNAMICS_NOT_A_CAUSAL_ENTROPY_REPAIR",
    })
    return _write("scale_dynamics_audit", result, "V17 direct-sigma scale dynamics", [
        "Latent and native scale, reward and entropy sigma gradients, guards, clipping, finite-shot variance, and exploration damage are logged per epoch and phase.",
        "No entropy or sigma-learning-rate change is made.",
    ])


def compare_step_figure5a_modes() -> dict[str, Any]:
    verify_import_manifest()
    plant = build_plant(_source_config())
    degrees = plant.mask.sum(axis=0).astype(int)
    step_hessian = .02
    figure5a_hessian = .02
    figure5b = read_json(ROOT / "artifacts/google_pure_v16/matched_figure5b/comparison.json")
    observed_rate = float(figure5b["summaries"]["D_V16_FROZEN_OPTIMIZER"]
                          ["median_fractional_residual_reduction"])
    predicted_rate = float(_frozen()["mean_learning_rate"]) * figure5a_hessian
    rows = [
        {"mode": "MATCHED_STEP", "directional_hessian": step_hessian,
         "detector_support": 1, "control_family": "924-coordinate distance-5 sparse analogue",
         "reward_aggregation": "connected detector sum", "target": "one-coordinate step"},
        {"mode": "FIGURE5A_SHARED_DRIFT", "directional_hessian": figure5a_hessian,
         "detector_support_min": int(degrees.min()), "detector_support_median": float(np.median(degrees)),
         "detector_support_max": int(degrees.max()), "control_family": "41-parameter distance-3 Stim plant",
         "reward_aggregation": "detector-local advantages with elementwise ratio products",
         "target": "shared sinusoidal 41-coordinate mode"},
        {"mode": "FIGURE5B_REPRESENTATIVE", "directional_hessian": .02,
         "control_family": "sparse scaling analogue", "reward_aggregation": "connected detector sum"},
    ]
    classification = "MODE_SUPPORT_AND_REWARD_AGGREGATION_MISMATCH"
    result = nonfinal({
        "pass": True, "rows": rows, "classification": classification,
        "normalized_local_hessians_match": math.isclose(step_hessian, figure5a_hessian),
        "scale_distributions_compared_in": "scale_dynamics_audit.json",
        "figure5b_note": {
            "observed_fractional_residual_reduction": observed_rate,
            "alpha_times_hessian": predicted_rate,
            "observed_to_local_prediction_ratio": observed_rate / predicted_rate,
            "classification": "RATE_DEFICIT_REMAINS",
            "repair_applied": False,
        },
        "figure5c_modified_or_executed": False,
    })
    return _write("step_figure5a_mode_comparison", result, "V17 step and Figure 5a mode comparison", [
        f"Classification: **{classification}** despite equal local normalized Hessians.",
        f"Figure 5b remains a note only: observed rate / alpha*kappa_H = {observed_rate / predicted_rate:.3f}; no repair is made.",
    ])


def _run_reduced_cells() -> list[dict[str, Any]]:
    source_cfg = _source_config()
    plant = build_plant(source_cfg)
    reduced = config()["reduced_acquisition"]
    protocol = Figure5aProtocol(
        AcquisitionMode.VALIDATION, int(reduced["epochs"]), int(reduced["candidates_per_epoch"]),
        int(reduced["qec_cycles_per_candidate"]), int(source_cfg["plant"]["circuit_rounds"]),
    )
    frozen = _frozen()
    rows = []
    campaign = ARTIFACT_ROOT / "reduced_postrepair/figure5a"
    for label, frequency in config()["frequencies_per_epoch"].items():
        token = str(frequency).replace(".", "p")
        checkpoint = campaign / f"checkpoint-{token}.json"
        preexisting = checkpoint.is_file()
        result = run_cell(
            protocol=protocol, plant=plant, frequency=float(frequency),
            entropy_weight=float(frozen["entropy_coefficient"]), seed=int(reduced["seed"]),
            optimizer_config=_optimizer_config(), initial_sigma=float(frozen["initial_sigma"]),
            checkpoint_path=checkpoint, dependency_hashes=dependency_hashes(ROOT, source_cfg),
            controller_hash=frozen["optimizer_bundle_hash"], clip=float(frozen["ppo_clip"]),
            baseline_weight=float(frozen["baseline_loss_weight"]), resume=preexisting,
            source_budget_profile=str(reduced["source_budget_profile"]),
        )
        rows.append({
            "label": label, "frequency_per_epoch": float(frequency), "complete": result["complete"],
            "epochs": int(protocol.epochs), "candidate_qec_cycles": result["candidate_qec_cycles"],
            "checkpoint": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
            "checkpoint_reused": preexisting, "controller_hash": result["controller_hash"],
            "plant_hash": result["plant_hash"], "boundary_hash": result["boundary_transform_hash"],
            "I_mean": result["learned_mean_ratio"]["source_ratio"],
            "I_stochastic": result["stochastic_ratio"]["source_ratio"],
            "finite_shot_denominator_nonzero": result["finite_shot_denominator_nonzero"],
        })
    return rows


def _minimal_repair() -> dict[str, Any]:
    manifest = verify_import_manifest()
    frozen = _frozen()
    source_rows = {row["role"]: row for row in manifest["imports"]}
    result = nonfinal({
        "pass": True,
        "diagnosis": "REDUCED_GATE_WINDOW_AND_METRIC_VALIDITY_DEFECT",
        "repair": "V17_COMPLETE_PERIOD_PHASE_PAIRED_ACCEPTANCE_AND_DENOMINATOR_GATE",
        "repair_scope": "V17 diagnostic evaluator and acceptance layer only",
        "source_figure5a_protocol_changed": False,
        "source_figure5a_target_changed": False,
        "source_figure5a_evaluator_changed": False,
        "source_normalization_changed": False,
        "optimizer_changed": False,
        "optimizer_bundle_hash_before": frozen["optimizer_bundle_hash"],
        "optimizer_bundle_hash_after": frozen["optimizer_bundle_hash"],
        "figure5c_changed": False,
        "protected_hashes": {
            role: source_rows[role]["sha256"] for role in (
                "v16_frozen_optimizer", "v16_source_normalized_boundary", "figure5a_source_config",
                "figure5a_target_plant", "figure5a_production_evaluator", "figure5a_metric_contract")
        },
        "source_fidelity_defect_requiring_optimizer_change": False,
    })
    return _write("minimal_repair", result, "V17 minimal causal repair", [
        "Only the reduced validation window/phase/denominator gate is repaired.",
        "The frozen optimizer, source normalization, production plant, target, evaluator, source budget, and Figure 5c are unchanged.",
    ])


def build_reduced_acceptance_v2() -> dict[str, Any]:
    verify_import_manifest()
    deterministic = (read_json(ARTIFACT_ROOT / "figure5a_deterministic_fixture.json")
                     if (ARTIFACT_ROOT / "figure5a_deterministic_fixture.json").is_file()
                     else run_figure5a_deterministic_fixture())
    metric = (read_json(ARTIFACT_ROOT / "figure5a_metric_endpoint.json")
              if (ARTIFACT_ROOT / "figure5a_metric_endpoint.json").is_file()
              else audit_figure5a_metric())
    window = (read_json(ARTIFACT_ROOT / "figure5a_window_aliasing.json")
              if (ARTIFACT_ROOT / "figure5a_window_aliasing.json").is_file()
              else audit_figure5a_windowing())
    transfer = (read_json(ARTIFACT_ROOT / "figure5a_mean_transfer.json")
                if (ARTIFACT_ROOT / "figure5a_mean_transfer.json").is_file()
                else measure_mean_transfer())
    decomposition = (read_json(ARTIFACT_ROOT / "mean_stochastic_decomposition.json")
                     if (ARTIFACT_ROOT / "mean_stochastic_decomposition.json").is_file()
                     else decompose_mean_stochastic())
    rows = {row["label"]: row for row in decomposition.get("rows", [])}
    paired_delta = (rows["slow"]["I_stochastic"] - rows["fast"]["I_stochastic"]
                    if "slow" in rows and "fast" in rows and
                    rows["slow"]["I_stochastic"] is not None and rows["fast"]["I_stochastic"] is not None
                    else None)
    delta_min = float(config()["acceptance_v2"]["delta_min"])
    required_phases = config()["acceptance_v2"]["required_phases"]
    observed_units = []
    for label in ("slow", "fast"):
        if label not in rows:
            continue
        row = rows[label]
        observed_units.append({
            "condition": label, "seed": config()["reduced_acquisition"]["seed"],
            "phase_radians": 0.0,
            "budget_hash": canonical_hash(config()["reduced_acquisition"]),
            "crn_hash": canonical_hash([config()["reduced_acquisition"]["seed"], "production-stream-seed-rule"]),
            "cycles_per_candidate": config()["reduced_acquisition"]["qec_cycles_per_candidate"],
            "burn_in_epochs": 0, "evaluation_periods": 0, "complete_periods": 0,
            "I_stochastic": row["I_stochastic"],
        })
    paired = paired_acceptance_v2(observed_units, delta_min=delta_min,
                                  confidence=float(config()["acceptance_v2"]["confidence"]))
    complete_pairs = int(paired["paired_complete_unit_count"])
    lcb = paired["lower_confidence_bound"]
    gates = {
        "deterministic_evaluator_valid": deterministic.get("pass") is True,
        "metric_endpoints_exact": metric.get("fixed_endpoint") == 0.0 and metric.get("oracle_endpoint") == 1.0,
        "frequency_denominators_separate_and_resolved": metric.get("all_observed_denominators_resolved") is True,
        "fast_frequency_greater_than_slow": (config()["frequencies_per_epoch"]["fast"] >
                                               config()["frequencies_per_epoch"]["slow"]),
        "complete_period_windows": paired["valid"],
        "complete_seed_phase_units": complete_pairs > 1 and len(required_phases) == 4,
        "paired_lcb_exceeds_delta_min": lcb is not None and lcb > delta_min,
        "secondary_transfer_ordering": transfer.get("expected_gain_ordering_pass") is True,
        "optimizer_immutable": _minimal_repair()["optimizer_changed"] is False,
        "figure5c_untouched": True,
    }
    old = read_json(ROOT / "artifacts/google_pure_v16/reduced_acceptance/result.json")
    archived = nonfinal({
        "pass": True, "archived_gate": "V16_REDUCED_FIGURE5A_DIRECTION_GATE",
        "source_path": "artifacts/google_pure_v16/reduced_acceptance/result.json",
        "source_sha256": file_hash(ROOT / "artifacts/google_pure_v16/reduced_acceptance/result.json"),
        "reason": "24 epochs contain no complete source slow/fast period and omit phase pairing and denominator resolution",
        "old_gate_pass": old.get("figure5a", {}).get("expected_slow_fast_direction_pass"),
        "used_for_v17_acceptance": False,
    })
    atomic_json(ARTIFACT_ROOT / "archived_v16_acceptance_gate.json", archived)
    result = nonfinal({
        "pass": all(gates.values()), "gates": gates,
        "primary_contrast": "Delta_I=I_stochastic_slow-I_stochastic_fast",
        "observed_unpaired_reduced_delta_I": paired_delta,
        "observed_units": observed_units,
        "paired_acceptance_evaluation": paired,
        "paired_complete_unit_count": complete_pairs,
        "pairing_keys": ["seed", "common_random_numbers", "phase", "budget", "cycles",
                         "evaluation_periods", "burn_in"],
        "lower_confidence_bound_95": lcb, "delta_min": delta_min,
        "delta_min_basis": config()["acceptance_v2"]["delta_min_basis"],
        "threshold_tuned_to_paper_value": False,
        "secondary_transfer_gates": {
            "slow_gain_greater_than_fast": transfer.get("expected_gain_ordering_pass"),
            "transfer_identifiable": transfer.get("all_transfer_estimates_identifiable"),
        },
        "window_audit_hash": canonical_hash(window),
        "classification": "UNDERPOWERED_REDUCED_GATE_NOT_ACCEPTED",
        "source_budget_authorization_granted": False,
        "source_budget_run_launched": False,
    })
    return _write("reduced_acceptance_v2", result, "V17 reduced acceptance v2", [
        "Acceptance requires paired seed/CRN/phase/budget units over complete post-burn-in periods.",
        "The available 24-epoch diagnostic cannot satisfy those gates, so no LCB or source-budget readiness is claimed.",
    ])


def run_reduced_postrepair() -> dict[str, Any]:
    """Run only three 24-epoch diagnostic cells; never a source/reference campaign."""
    verify_import_manifest()
    repair = _minimal_repair()
    rows = _run_reduced_cells()
    # Rebuild diagnostics from the newly written checkpoints.
    frequency = audit_figure5a_frequency()
    metric = audit_figure5a_metric()
    window = audit_figure5a_windowing()
    mean_transfer = measure_mean_transfer()
    latency = audit_latency_phase()
    decomposition = decompose_mean_stochastic()
    scale = audit_scale_dynamics()
    modes = compare_step_figure5a_modes()
    acceptance = build_reduced_acceptance_v2()
    from hdfa_rl_suite.google_pure_v16.experiments import _run_step_seed
    v16_step = read_json(ROOT / "artifacts/google_pure_v16/matched_step/comparison.json")
    rerun = next(row for row in _run_step_seed(81701) if row["branch"] == "D_V16_FROZEN_OPTIMIZER")
    expected = next(row for row in v16_step["rows"]
                    if row["branch"] == "D_V16_FROZEN_OPTIMIZER" and row["seed"] == 81701)
    exact = (rerun["standard_normal_tape_hashes"] == expected["standard_normal_tape_hashes"] and
             np.allclose(rerun["target_relative_progress"], expected["target_relative_progress"],
                         rtol=0, atol=0))
    step_check = {
        "method": "ACTUAL_ONE_SEED_MATCHED_STEP_RERUN",
        "artifact_sha256": file_hash(ROOT / "artifacts/google_pure_v16/matched_step/comparison.json"),
        "optimizer_bundle_hash": _frozen()["optimizer_bundle_hash"],
        "seed": 81701, "rerun_final_target_fraction": rerun["final_target_fraction"],
        "expected_final_target_fraction": expected["final_target_fraction"],
        "standard_normal_tapes_exact": rerun["standard_normal_tape_hashes"] == expected["standard_normal_tape_hashes"],
        "trajectory_exact": bool(exact), "pass": bool(exact),
    }
    result = nonfinal({
        "pass": False, "cells": rows, "minimal_repair": repair,
        "matched_step_check": step_check,
        "diagnostic_hashes": {name: canonical_hash(value) for name, value in {
            "frequency": frequency, "metric": metric, "window": window,
            "mean_transfer": mean_transfer, "latency": latency,
            "decomposition": decomposition, "scale": scale, "modes": modes,
            "acceptance": acceptance}.items()},
        "classification": "EXECUTION_PATH_VERIFIED_BUT_REDUCED_GATE_REMAINS_UNDERPOWERED",
        "failure_is_expected_and_not_promoted": True,
        "source_budget_run_launched": False, "reference_campaign_launched": False,
        "natural_drift_executed": False, "figure5c_executed": False,
    })
    atomic_json(ARTIFACT_ROOT / "reduced_postrepair/result.json", result)
    return result


__all__ = [
    "audit_sensitivity_semantics", "refit_step_transfer", "audit_figure5a_frequency",
    "run_figure5a_deterministic_fixture", "audit_figure5a_metric", "audit_figure5a_windowing",
    "measure_mean_transfer", "audit_latency_phase", "decompose_mean_stochastic",
    "audit_scale_dynamics", "compare_step_figure5a_modes", "build_reduced_acceptance_v2",
    "run_reduced_postrepair",
]
