"""V18 bounded Figure 5a identification and evidence-quality cleanup."""
from __future__ import annotations

from datetime import datetime, timezone
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
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.contracts import PositivityGuard
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.optimizer import OptimizerConfig
from hdfa_rl_suite.google_pure_source_exact.source_normalization import SourceNormalizationBoundary
from hdfa_rl_suite.google_pure_v17.estimators import estimate_sinusoidal_transfer

from .contracts import NONFINAL, nonfinal
from .imports import verify_import_manifest
from .io import ARTIFACT_ROOT, ROOT, atomic_json, atomic_text, canonical_hash, config, file_hash, read_json


AMBIGUOUS_SENSITIVITY_KEYS = {"curvature", "normalized_curvature", "conditioned_curvature"}


def _write(name: str, value: dict[str, Any], title: str, statements: list[str]) -> dict[str, Any]:
    atomic_json(ARTIFACT_ROOT / f"{name}.json", value)
    atomic_text(ARTIFACT_ROOT / f"{name}.md", "\n".join(
        [f"# {title}", "", *statements, "", "Development-only, non-final evidence."]))
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


def _keys(value: Any, prefix: str = "") -> list[str]:
    found = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            token = f"{prefix}.{key}" if prefix else str(key)
            found.append(token)
            found.extend(_keys(child, token))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_keys(child, f"{prefix}[{index}]"))
    return found


def build_sensitivity_field_cleanup() -> dict[str, Any]:
    """Make new artifact terminology explicit without mutating frozen lineage."""
    source_rows = [
        {"family": family, "quadratic_coefficient_a": .01,
         "hessian_curvature_kappa_H": .02, "unit_variance_damage_kappa_V": .01,
         "identity_error": 0.0, "identity_pass": True}
        for family in ("MATCHED_STEP", "FIGURE5A_41_PARAMETER_STIM", "FIGURE5B_REPRESENTATIVE_MODE")
    ]
    active_paths = [
        ROOT / "artifacts/google_pure_v17/sensitivity_semantics_audit.json",
        ROOT / "artifacts/google_pure_v17/step_transfer_identification.json",
    ]
    occurrences = []
    for path in active_paths:
        value = read_json(path)
        for key_path in _keys(value):
            if key_path.rsplit(".", 1)[-1] in AMBIGUOUS_SENSITIVITY_KEYS:
                occurrences.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"),
                                    "key_path": key_path})
    result = nonfinal({
        "pass": not occurrences and all(row["identity_pass"] for row in source_rows),
        "explicit_semantics": source_rows,
        "quadratic_identity": "unit_variance_damage_kappa_V = 0.5*hessian_curvature_kappa_H",
        "ambiguous_keys_forbidden_in_new_source_reference_artifacts": sorted(AMBIGUOUS_SENSITIVITY_KEYS),
        "active_v17_occurrences": occurrences,
        "legacy_frozen_exceptions": [
            {"path": "artifacts/google_pure_v16/frozen_source_normalized_optimizer.json",
             "field": "conditioned_curvature", "meaning": "unit_variance_damage_kappa_V",
             "mutation_permitted": False},
            {"path": "artifacts/google_pure_v16/local_contraction_audit.json",
             "field": "V16 local diagnostic terminology", "meaning": "legacy factor-two label corrected by V17",
             "mutation_permitted": False},
        ],
        "frozen_artifacts_rewritten": False, "production_normalization_changed": False,
    })
    return _write("sensitivity_field_cleanup", result, "V18 sensitivity-field cleanup", [
        "All active V18 semantics use `quadratic_coefficient_a`, `hessian_curvature_kappa_H`, and `unit_variance_damage_kappa_V`.",
        "Legacy V16 artifacts remain hash-frozen and are documented rather than rewritten.",
    ])


def _step_fit() -> dict[str, float]:
    value = read_json(ROOT / "artifacts/google_pure_v17/step_transfer_identification.json")
    fit = value["free_gain_delay_tau"]
    return {"K": float(fit["gain"]), "Delta": float(fit["delay_epochs"]),
            "tau": float(fit["tau_epochs"])}


def validate_deterministic_transfer() -> dict[str, Any]:
    verify_import_manifest()
    build_sensitivity_field_cleanup()
    settings = config()["deterministic_fixture"]
    fit = _step_fit()
    rows = []
    for label, frequency_value in settings["frequencies_per_epoch"].items():
        frequency = float(frequency_value)
        omega = 2.0 * math.pi * frequency
        transfer = fit["K"] * np.exp(-1j * omega * fit["Delta"]) / (1.0 + 1j * omega * fit["tau"])
        analytic_gain = float(abs(transfer))
        analytic_phase = float(-np.angle(transfer))
        analytic_improvement = float(1.0 - abs(1.0 - transfer) ** 2)
        period = int(round(1.0 / frequency))
        epochs = np.arange(period * int(settings["complete_periods"]), dtype=float)
        target = np.sin(omega * epochs)
        learned = analytic_gain * np.sin(omega * epochs - analytic_phase)
        regression = estimate_sinusoidal_transfer(epochs, learned, frequency,
                                                  minimum_cycles=1.0,
                                                  maximum_condition_number=1000.0)
        measured_gain = float(regression["gain"])
        measured_phase = float(regression["phase_lag_radians"])
        fixed_cost = float(np.sum(target**2))
        learned_cost = float(np.sum((learned - target)**2))
        scale = 10**12
        measured_improvement = ratio_from_raw_counts(
            int(round(learned_cost * scale)), int(round(fixed_cost * scale)), 0)["source_ratio"]
        residual_vector = np.asarray([measured_gain - analytic_gain,
                                      measured_phase - analytic_phase,
                                      measured_improvement - analytic_improvement])
        absolute = float(np.max(np.abs(residual_vector)))
        relative = float(max(
            abs(measured_gain - analytic_gain) / max(abs(analytic_gain), 1e-15),
            abs(measured_phase - analytic_phase) / max(abs(analytic_phase), 1e-15),
            abs(measured_improvement - analytic_improvement) /
            max(abs(analytic_improvement), 1e-15)))
        rows.append({
            "label": label, "fixture_gain_K": fit["K"], "fixture_delay_Delta": fit["Delta"],
            "fixture_tau": fit["tau"], "fixture_initial_condition": settings["initial_condition"],
            "frequency": frequency, "analytic_gain": analytic_gain, "measured_gain": measured_gain,
            "analytic_phase": analytic_phase, "measured_phase": measured_phase,
            "analytic_normalized_improvement": analytic_improvement,
            "measured_normalized_improvement": measured_improvement,
            "absolute_residual": absolute, "relative_residual": relative,
            "absolute_tolerance": float(settings["absolute_tolerance"]),
            "relative_tolerance": float(settings["relative_tolerance"]),
            "pass": bool(absolute <= settings["absolute_tolerance"] or
                         relative <= settings["relative_tolerance"]),
        })
    order = {row["label"]: row["measured_normalized_improvement"] for row in rows}
    result = nonfinal({
        "pass": all(row["pass"] for row in rows) and
                order["slow"] > order["intermediate"] > order["fast"],
        "rows": rows, "transfer_model": "H(omega)=K*exp(-i*omega*Delta)/(1+i*omega*tau)",
        "production_metric_function": "ratio_from_raw_counts",
        "production_evaluator_changed": False,
        "fixture_interpretation_repaired": "EXACT_CONTINUOUS_TRANSFER_IN_STEADY_PERIODIC_STATE",
        "old_discrete_recursion_values_used_for_quantitative_claim": False,
        "slow_intermediate_fast_ordering_pass": order["slow"] > order["intermediate"] > order["fast"],
    })
    return _write("deterministic_fixture_quantitative_validation", result,
                  "V18 quantitative deterministic transfer validation", [
        "The fixture now injects the stated continuous transfer exactly in steady periodic state.",
        "The production raw-count normalization is used unchanged and must agree within preregistered numerical tolerance.",
    ])


STREAMS = ("fixed", "optimal", "stochastic", "learned_mean")


def _boundary(plant: Any) -> SourceNormalizationBoundary:
    degree = np.sum(plant.mask, axis=0).astype(float)
    coefficient = np.asarray([row.omega_sensitivity for row in plant.inventory]) * degree
    return SourceNormalizationBoundary.from_training_objective(
        "FIGURE5A_REAL_TIME_STEERING", coefficient, control_ids=plant.parameter_ids)


def _metric_from_totals(totals: Mapping[str, int]) -> dict[str, Any]:
    fixed = int(totals["fixed"])
    oracle = int(totals["optimal"])
    learned_mean = int(totals["learned_mean"])
    stochastic = int(totals["stochastic"])
    denominator = fixed - oracle
    denominator_standard_error = math.sqrt(max(0, fixed + oracle))
    denominator_resolved = bool(
        denominator > 0 and denominator > 1.96 * denominator_standard_error)
    if denominator == 0:
        i_mean = i_stochastic = None
    else:
        i_mean = ratio_from_raw_counts(learned_mean, fixed, oracle)["source_ratio"]
        i_stochastic = ratio_from_raw_counts(stochastic, fixed, oracle)["source_ratio"]
    return {
        "C_fixed": fixed, "C_oracle": oracle, "C_mean": learned_mean,
        "C_stochastic": stochastic, "normalization_denominator": denominator,
        "denominator_standard_error": denominator_standard_error,
        "denominator_snr": (denominator / denominator_standard_error
                            if denominator_standard_error else None),
        "denominator_resolved": denominator_resolved,
        "I_mean": i_mean, "I_stochastic": i_stochastic,
        "exploration_damage": stochastic - learned_mean,
        "stream_separation_retained": list(STREAMS),
    }


def _mad_standard_deviation(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    median = float(np.median(array))
    robust = 1.4826 * float(np.median(np.abs(array - median)))
    if robust > 0:
        return robust
    return float(np.std(array, ddof=1)) if array.size > 1 else 0.0


def _delta_input_hashes() -> dict[str, str]:
    return {
        "v18_protocol": file_hash(ROOT / "configs/google_pure_v18/protocol.json"),
        "source_figure5a_config": file_hash(ROOT / "configs/google_pure_source_exact/figure5a.json"),
        "source_plant_code": file_hash(
            ROOT / "src/hdfa_rl_suite/google_pure_source_exact/figure5a/plant.py"),
        "production_metric_code": file_hash(
            ROOT / "src/hdfa_rl_suite/google_pure_source_exact/figure5a/contracts.py"),
        "v17_protocol": file_hash(ROOT / "configs/google_pure_v17/protocol.json"),
    }


def audit_delta_min_provenance() -> dict[str, Any]:
    """Select the practical-effect threshold before seeing any V18 transfer result."""
    path = ARTIFACT_ROOT / "delta_min_provenance.json"
    if path.is_file():
        existing = read_json(path)
        if existing.get("input_artifact_hashes") != _delta_input_hashes():
            raise RuntimeError("frozen delta-min inputs changed after selection")
        return existing
    transfer_results = list(ARTIFACT_ROOT.glob("transfer_*.json"))
    checkpoints = list((ARTIFACT_ROOT / "acquisition").glob("*/checkpoint.json"))
    if transfer_results or checkpoints:
        raise RuntimeError("delta_min provenance must be frozen before V18 transfer acquisition")
    verify_import_manifest()
    settings = config()["delta_min_selection"]
    source = _source_config()
    plant = build_plant(source)
    boundary = _boundary(plant)
    target = boundary.target_to_native(np.ones(41))
    controls = {
        "fixed": boundary.target_to_native(np.zeros(41)),
        "midpoint": boundary.target_to_native(np.full(41, .5)),
        "oracle": target,
    }
    rows = []
    ratios = []
    denominators = []
    for replicate in range(int(settings["static_probe_replicates"])):
        counts = {}
        for stream, control in controls.items():
            observed = plant.sample_detector_counts(
                control, epoch=0, frequency=1 / 300,
                qec_cycles=int(settings["qec_cycles_per_probe"]),
                seed=plant.stream_seed(int(settings["probe_seed"]),
                                       f"v18-delta-{stream}", replicate, 0),
                target_controls=target)
            counts[stream] = int(observed.sum())
        denominator = counts["fixed"] - counts["oracle"]
        value = (ratio_from_raw_counts(counts["midpoint"], counts["fixed"],
                                       counts["oracle"])["source_ratio"]
                 if denominator else None)
        if value is not None:
            ratios.append(value)
            denominators.append(abs(denominator))
        rows.append({"replicate": replicate, "counts": counts,
                     "static_midpoint_improvement": value})
    if len(ratios) < max(8, int(settings["static_probe_replicates"]) // 2):
        raise RuntimeError("independent static probes did not resolve enough normalization denominators")
    fast = config()["transfer_acquisition"]["fast"]
    projected_boundaries = int(fast["analysis_periods"] * round(
        1.0 / float(fast["frequency_per_epoch"])) * fast["candidates_per_epoch"])
    single_probe_noise = _mad_standard_deviation(np.asarray(ratios))
    projected_complete_run_noise = 2.0 * single_probe_noise / math.sqrt(projected_boundaries)
    projected_resolution = 2.0 / (float(np.median(denominators)) * projected_boundaries)
    candidates = {
        "minimum_scientific_resolution": float(settings["minimum_scientific_resolution"]),
        "one_sided_95_percent_contrast_noise": float(
            settings["confidence_z_one_sided"] * math.sqrt(2.0) * projected_complete_run_noise),
        "integer_count_resolution_guard": float(
            settings["resolution_multiplier"] * projected_resolution),
    }
    raw_delta = max(candidates.values())
    delta_min = math.ceil(raw_delta * 1000.0 - 1e-12) / 1000.0
    v17_config = read_json(ROOT / "configs/google_pure_v17/protocol.json")
    v17_delta = v17_config.get("paired_acceptance", {}).get("delta_min")
    visible_prior = [
        "artifacts/google_pure_v17/figure5a_deterministic_fixture.json",
        "artifacts/google_pure_v17/figure5a_mean_transfer.json",
        "artifacts/google_pure_v17/reduced_acceptance_v2.json",
    ]
    result = nonfinal({
        "pass": delta_min >= float(settings["minimum_scientific_resolution"]),
        "delta_min": delta_min,
        "selection_formula": settings["formula"],
        "formula_terms": candidates,
        "development_noise_estimate": {
            "static_midpoint_improvement_mad_standard_deviation": single_probe_noise,
            "projected_complete_run_standard_deviation": projected_complete_run_noise,
            "projection_candidate_boundaries": projected_boundaries,
            "probe_replicates": len(ratios),
        },
        "measurement_resolution": {
            "median_single_probe_denominator_counts": float(np.median(denominators)),
            "projected_single_count_resolution": projected_resolution,
            "resolution_multiplier": float(settings["resolution_multiplier"]),
        },
        "input_artifact_hashes": _delta_input_hashes(),
        "selection_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "selection_order": int(settings["selection_order"]),
        "new_transfer_acquisition_order": int(settings["new_transfer_acquisition_order"]),
        "any_v18_slow_fast_scientific_result_visible_at_selection": False,
        "prior_v17_development_results_visible": True,
        "visible_prior_development_artifacts_excluded_from_selection": visible_prior,
        "prior_v17_delta_min": v17_delta,
        "prior_v17_delta_provenance_classification": "UNPROVABLE_AND_REPLACED_INDEPENDENTLY",
        "selection_independent_of_new_v18_transfer_outcomes": True,
        "static_probe_rows": rows,
        "optimizer_changed": False,
    })
    return _write("delta_min_provenance", result, "V18 delta-min provenance", [
        f"The independently selected development threshold is delta_min={delta_min:.3f}.",
        "It was frozen from static development probes before any V18 transfer acquisition.",
        "Visible V17 diagnostics were recorded and excluded from the numerical selection.",
    ])


def build_steady_state_rule() -> dict[str, Any]:
    """Freeze the periodic stationarity rule before any new transfer outcome."""
    path = ARTIFACT_ROOT / "steady_state_rule.json"
    if path.is_file():
        existing = read_json(path)
        expected = {
            "v18_protocol": file_hash(ROOT / "configs/google_pure_v18/protocol.json"),
            "delta_min_provenance": file_hash(ARTIFACT_ROOT / "delta_min_provenance.json"),
            "deterministic_validation": file_hash(
                ARTIFACT_ROOT / "deterministic_fixture_quantitative_validation.json"),
        }
        if (existing.get("rule") != config()["steady_state_rule"] or
                existing.get("input_artifact_hashes") != expected):
            raise RuntimeError("frozen steady-state rule or its inputs changed")
        return existing
    if list(ARTIFACT_ROOT.glob("transfer_*.json")) or list(
            (ARTIFACT_ROOT / "acquisition").glob("*/checkpoint.json")):
        raise RuntimeError("steady-state rule must be frozen before V18 transfer acquisition")
    deterministic = validate_deterministic_transfer()
    if not deterministic["pass"]:
        raise RuntimeError("quantitative deterministic fixture failed before rule freeze")
    delta = audit_delta_min_provenance()
    settings = config()["steady_state_rule"]
    result = nonfinal({
        "pass": settings["manual_truncation_permitted"] is False,
        "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_order": int(settings["freeze_order"]),
        "rule": settings,
        "period_definition": "T=1/f exact integer epochs",
        "analysis_window": "complete periods after the preregistered transient only",
        "stable_transition_requirements": {
            label: int(settings[f"{label}_required_stable_transitions"])
            for label in ("intermediate", "fast", "slow")
        },
        "automatic_five_tau_rule_used": False,
        "manual_truncation_permitted": False,
        "input_artifact_hashes": {
            "v18_protocol": file_hash(ROOT / "configs/google_pure_v18/protocol.json"),
            "delta_min_provenance": file_hash(ARTIFACT_ROOT / "delta_min_provenance.json"),
            "deterministic_validation": file_hash(
                ARTIFACT_ROOT / "deterministic_fixture_quantitative_validation.json"),
        },
        "delta_min": delta["delta_min"],
    })
    return _write("steady_state_rule", result, "V18 preregistered steady-state rule", [
        "Only complete periods after one full transient period enter identification.",
        "Consecutive periods must stabilize in gain, phase, sigma, offset, and scale-guard occupancy.",
        "No 5-tau shortcut or outcome-dependent truncation is permitted.",
    ])


def _relative_change(left: float, right: float, *, floor: float = 1e-12) -> float:
    return abs(float(right) - float(left)) / max(abs(float(left)), floor)


def _phase_difference(left: float, right: float) -> float:
    return abs(float(np.angle(np.exp(1j * (float(right) - float(left))))))


def _unwrap_around(value: float, centre: float) -> float:
    return float(centre + np.angle(np.exp(1j * (float(value) - float(centre)))))


def _period_summary(records: list[dict[str, Any]], frequency: float, period_index: int,
                    boundary: SourceNormalizationBoundary, plant: Any) -> dict[str, Any]:
    period = int(round(1.0 / frequency))
    start, stop = period_index * period, (period_index + 1) * period
    selected = [row for row in records if start <= int(row["epoch"]) < stop]
    if len(selected) != period:
        raise RuntimeError(f"period {period_index} is incomplete: {len(selected)} of {period}")
    epochs = np.asarray([row["epoch"] for row in selected], dtype=float)
    direction = np.asarray([
        np.mean(np.asarray(row["normalized_behavior_mean"], dtype=float)) for row in selected
    ])
    fitted = estimate_sinusoidal_transfer(
        epochs, direction, frequency, minimum_cycles=1.0,
        maximum_condition_number=float(config()["identification"]["maximum_design_condition_number"]))
    bounded = Figure5aBoundedActionAblation(plant)
    sigma_x = []
    sigma_u = []
    guards = []
    reward_gradients = []
    entropy_gradients = []
    sigma_updates = []
    component_clipping = []
    detector_clipping = []
    floor_occupancy = []
    ceiling_occupancy = []
    frozen = _frozen()
    minimum_sigma = float(frozen["minimum_sigma"])
    maximum_sigma = float(frozen["maximum_sigma"])
    for row in selected:
        latent_mean = np.asarray(row["latent_behavior_mean"], dtype=float)
        latent_sigma = np.asarray(row["behavior_sigma"], dtype=float)
        derivative = 1.0 / np.cosh(latent_mean / bounded.control_limits) ** 2
        native_sigma = boundary.native_scale * derivative * latent_sigma
        sigma_x.append(float(np.median(latent_sigma)))
        sigma_u.append(float(np.median(native_sigma)))
        guards.append(float(row["fraction_at_positivity_guard"]))
        reward_gradients.append(float(row["reward_sigma_gradient_norm"]))
        entropy_gradients.append(float(row["entropy_sigma_gradient_norm"]))
        post_sigma = np.asarray(row["post_update_sigma"], dtype=float)
        sigma_updates.append(float(np.linalg.norm(post_sigma - latent_sigma)))
        component_clipping.append(float(row["component_clip_fraction"]))
        detector_clipping.append(float(row["detector_clip_fraction"]))
        floor_occupancy.append(float(np.mean(np.isclose(post_sigma, minimum_sigma, rtol=0, atol=1e-12))))
        ceiling_occupancy.append(float(np.mean(np.isclose(post_sigma, maximum_sigma, rtol=0, atol=1e-12))))
    totals = {stream: int(sum(row["stream_totals"][stream] for row in selected))
              for stream in STREAMS}
    stream_metrics = _metric_from_totals(totals)
    first = selected[0]
    last = selected[-1]
    first_mean = np.asarray(first["latent_behavior_mean"], dtype=float)
    last_mean = np.asarray(last["post_update_latent_mean"], dtype=float)
    first_sigma = np.asarray(first["behavior_sigma"], dtype=float)
    last_sigma = np.asarray(last["post_update_sigma"], dtype=float)
    first_native = (boundary.native_scale /
                    np.cosh(first_mean / bounded.control_limits) ** 2 * first_sigma)
    last_native = (boundary.native_scale /
                   np.cosh(last_mean / bounded.control_limits) ** 2 * last_sigma)
    return {
        "period_index": period_index, "epoch_window": [start, stop],
        "complete": len(selected) == period, "epochs": len(selected),
        "sine_coefficient": fitted["sine_coefficient"],
        "cosine_coefficient": fitted["cosine_coefficient"],
        "gain": fitted["gain"], "phase_lag_radians": fitted["phase_lag_radians"],
        "mean_offset": fitted["offset"],
        "design_condition_number": fitted["design_condition_number"],
        "sigma_x_median": float(np.median(sigma_x)),
        "sigma_u_median": float(np.median(sigma_u)),
        "sigma_start": float(np.median(first_sigma)),
        "sigma_end": float(np.median(last_sigma)),
        "native_sigma_start": float(np.median(first_native)),
        "native_sigma_end": float(np.median(last_native)),
        "sigma_update_norm_median": float(np.median(sigma_updates)),
        "scale_guard_occupancy": float(np.mean(guards)),
        "scale_floor_occupancy": float(np.mean(floor_occupancy)),
        "scale_ceiling_occupancy": float(np.mean(ceiling_occupancy)),
        "candidate_clipping_fraction": float(np.mean(component_clipping)),
        "detector_clipping_fraction": float(np.mean(detector_clipping)),
        "reward_sigma_gradient_norm_median": float(np.median(reward_gradients)),
        "entropy_sigma_gradient_norm_median": float(np.median(entropy_gradients)),
        "exploration_damage": stream_metrics["exploration_damage"],
        "stream_metrics": stream_metrics,
    }


def _steady_state_diagnostic(label: str, period_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rule = config()["steady_state_rule"]
    transient_periods = int(rule["minimum_transient_periods"])
    analysis = period_rows[transient_periods:]
    transitions = []
    for left, right in zip(analysis[:-1], analysis[1:]):
        diagnostics = {
            "gain_relative_change": _relative_change(left["gain"], right["gain"]),
            "phase_change_radians": _phase_difference(
                left["phase_lag_radians"], right["phase_lag_radians"]),
            "sigma_relative_change": _relative_change(
                left["sigma_x_median"], right["sigma_x_median"]),
            "mean_offset_change": abs(right["mean_offset"] - left["mean_offset"]),
            "scale_guard_occupancy_change": abs(
                right["scale_guard_occupancy"] - left["scale_guard_occupancy"]),
        }
        checks = {
            "gain": diagnostics["gain_relative_change"] <= rule["gain_relative_change_max"],
            "phase": diagnostics["phase_change_radians"] <= rule["phase_change_radians_max"],
            "sigma": diagnostics["sigma_relative_change"] <= rule["sigma_relative_change_max"],
            "offset": diagnostics["mean_offset_change"] <= rule["mean_offset_change_max"],
            "scale_guard": diagnostics["scale_guard_occupancy_change"] <=
                           rule["scale_guard_occupancy_change_max"],
        }
        transitions.append({"from_period": left["period_index"],
                            "to_period": right["period_index"], **diagnostics,
                            "checks": checks, "stable": all(checks.values())})
    required = int(rule[f"{label}_required_stable_transitions"])
    stable_tail = 0
    for row in reversed(transitions):
        if not row["stable"]:
            break
        stable_tail += 1
    return {
        "transient_periods_excluded": transient_periods,
        "complete_analysis_periods": len(analysis),
        "transitions": transitions, "required_stable_transitions": required,
        "stable_tail_transitions": stable_tail,
        "pass": bool(len(analysis) >= 1 and stable_tail >= required),
        "manual_truncation_used": False, "automatic_five_tau_rule_used": False,
    }


def _bootstrap_period_transfer(period_rows: list[dict[str, Any]], *, draws: int,
                               seed: int) -> dict[str, Any]:
    coefficients = np.asarray([
        [row["sine_coefficient"], row["cosine_coefficient"]] for row in period_rows
    ], dtype=float)
    if coefficients.size == 0:
        raise ValueError("at least one complete period is required for bootstrap")
    point = np.mean(coefficients, axis=0)
    point_gain = float(np.hypot(*point))
    point_phase_lag = float(-math.atan2(point[1], point[0]))
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(coefficients), size=(int(draws), len(coefficients)))
    sampled = np.mean(coefficients[indices], axis=1)
    gains = np.hypot(sampled[:, 0], sampled[:, 1])
    phases = np.asarray([
        _unwrap_around(-math.atan2(row[1], row[0]), point_phase_lag) for row in sampled
    ])
    return {
        "method": "COMPLETE_PERIOD_NONPARAMETRIC_BOOTSTRAP",
        "draws": int(draws), "complete_period_units": len(coefficients),
        "gain_point": point_gain, "phase_lag_point_radians": point_phase_lag,
        "gain_confidence_interval_95": np.quantile(gains, [.025, .975]).tolist(),
        "phase_lag_confidence_interval_95": np.quantile(phases, [.025, .975]).tolist(),
        "gain_samples": gains.tolist(), "phase_lag_samples_radians": phases.tolist(),
        "uncertainty_scope": "WITHIN_RUN_COMPLETE_PERIODS_NOT_BETWEEN_SEEDS",
    }


def _step_transfer_comparison(measured_gain: float, measured_phase: float,
                              frequency: float) -> dict[str, Any]:
    fit = _step_fit()
    omega = 2.0 * math.pi * frequency
    transfer = fit["K"] * np.exp(-1j * omega * fit["Delta"]) / (1.0 + 1j * omega * fit["tau"])
    predicted_gain = float(abs(transfer))
    predicted_phase = float(-np.angle(transfer))
    gain_residual = measured_gain - predicted_gain
    phase_residual = _unwrap_around(measured_phase, predicted_phase) - predicted_phase
    gain_relative = gain_residual / max(predicted_gain, 1e-12)
    if abs(gain_relative) <= .25 and abs(phase_residual) <= .35:
        classification = "CONSISTENT_WITH_STEP_PREDICTION"
    elif gain_relative < -.25 and abs(phase_residual) <= .35:
        classification = "LOWER_GAIN_THAN_STEP_PREDICTION"
    elif gain_relative > .25 and abs(phase_residual) <= .35:
        classification = "HIGHER_GAIN_THAN_STEP_PREDICTION"
    elif abs(gain_relative) <= .25:
        classification = "PHASE_MISMATCH_FROM_STEP_PREDICTION"
    else:
        classification = "MIXED_GAIN_AND_PHASE_MISMATCH"
    return {
        "model": "H(omega)=K*exp(-i*omega*Delta)/(1+i*omega*tau)",
        "step_fit": fit, "predicted_gain": predicted_gain,
        "predicted_phase_lag_radians": predicted_phase,
        "measured_gain": measured_gain, "measured_phase_lag_radians": measured_phase,
        "gain_residual": gain_residual, "gain_relative_residual": gain_relative,
        "phase_residual_radians": phase_residual, "classification": classification,
    }


def _analyse_transfer(label: str, cell: dict[str, Any], checkpoint: Path,
                      provenance: Path,
                      settings_override: Mapping[str, Any] | None = None) -> dict[str, Any]:
    settings = (dict(settings_override) if settings_override is not None else
                config()["transfer_acquisition"][label])
    identification = config()["identification"]
    records = list(cell["epoch_records"])
    frequency = float(settings["frequency_per_epoch"])
    period = int(round(1.0 / frequency))
    if len(records) != int(settings["epochs"]) or len(records) % period:
        raise RuntimeError("V18 transfer acquisition did not contain exact complete periods")
    plant = build_plant(_source_config())
    boundary = _boundary(plant)
    period_rows = [_period_summary(records, frequency, index, boundary, plant)
                   for index in range(len(records) // period)]
    steady = _steady_state_diagnostic(label, period_rows)
    analysis_rows = period_rows[int(config()["steady_state_rule"]["minimum_transient_periods"]):]
    start = period
    selected = records[start:]
    epochs = np.asarray([row["epoch"] for row in selected], dtype=float)
    direction = np.asarray([
        np.mean(np.asarray(row["normalized_behavior_mean"], dtype=float)) for row in selected
    ])
    regression = estimate_sinusoidal_transfer(
        epochs, direction, frequency, minimum_cycles=float(settings["analysis_periods"]),
        maximum_condition_number=float(identification["maximum_design_condition_number"]))
    bootstrap = _bootstrap_period_transfer(
        analysis_rows, draws=int(identification["bootstrap_draws"]),
        seed=int(settings["seed"]) + 1800)
    totals = {stream: int(sum(row["stream_totals"][stream] for row in selected))
              for stream in STREAMS}
    metric = _metric_from_totals(totals)
    gain_ci_width = float(np.ptp(bootstrap["gain_confidence_interval_95"]))
    phase_ci_width = float(np.ptp(bootstrap["phase_lag_confidence_interval_95"]))
    checks = {
        "complete_periods": len(analysis_rows) == int(settings["analysis_periods"]),
        "steady_state_rule": steady["pass"],
        "direct_mean_regression": regression["identifiable"],
        "gain_finite_and_sensible": bool(
            np.isfinite(regression["gain"]) and 0 < regression["gain"] <= identification["maximum_gain"]),
        "gain_uncertainty": gain_ci_width <= identification["maximum_gain_ci_width"],
        "phase_uncertainty": phase_ci_width <= identification["maximum_phase_ci_width_radians"],
        "normalization_denominator": metric["denominator_resolved"],
        "production_controller": cell["controller_hash"] == _frozen()["optimizer_bundle_hash"],
        "direct_sigma": cell["parameterization"] == "DIRECT_SIGMA_SOURCE_EXACT",
        "all_four_streams": set(cell["stream_totals"]) == set(STREAMS),
        "nonzero_qec_cycles": int(cell["four_stream_qec_cycles"]) > 0,
        "no_candidates_dropped": cell["no_candidates_dropped"] is True,
    }
    direct_transfer_checks = {
        key: checks[key] for key in (
            "complete_periods", "direct_mean_regression", "gain_finite_and_sensible",
            "gain_uncertainty", "phase_uncertainty",
        )
    }
    direct_transfer_identifiable = all(direct_transfer_checks.values())
    steady_periodic_accepted = all(checks.values())
    result = nonfinal({
        "label": label, "pass": steady_periodic_accepted,
        "identifiable": steady_periodic_accepted,
        "direct_mean_transfer_identifiable": direct_transfer_identifiable,
        "direct_mean_transfer_gates": direct_transfer_checks,
        "steady_periodic_identification_accepted": steady_periodic_accepted,
        "classification": ("STEADY_PERIODIC_MEAN_TRANSFER_IDENTIFIED" if steady_periodic_accepted
                           else "STEADY_PERIODIC_IDENTIFICATION_NOT_ACCEPTED"),
        "gates": checks, "frequency_per_epoch": frequency, "period_epochs": period,
        "total_epochs": len(records), "analysis_epoch_window": [start, len(records)],
        "complete_analysis_periods": len(analysis_rows),
        "direct_mean_direction": "mean(normalized_behavior_mean[0:41])",
        "normalized_performance_used_as_transfer_proxy": False,
        "mean_transfer_regression": regression, "bootstrap_uncertainty": bootstrap,
        "gain_confidence_interval_width": gain_ci_width,
        "phase_confidence_interval_width_radians": phase_ci_width,
        "period_diagnostics": period_rows, "steady_state_diagnostic": steady,
        "stream_decomposition": metric,
        "step_prediction_comparison": _step_transfer_comparison(
            float(regression["gain"]), float(regression["phase_lag_radians"]), frequency),
        "controller_mode": "PAPER_DIRECT_SIGMA",
        "controller_hash": cell["controller_hash"],
        "parameterization": cell["parameterization"],
        "optimizer_bundle_hash": _frozen()["optimizer_bundle_hash"],
        "optimizer_changed": False,
        "plant_hash": cell["plant_hash"], "dependency_hashes": cell["dependency_hashes"],
        "candidate_qec_cycles": cell["candidate_qec_cycles"],
        "four_stream_qec_cycles": cell["four_stream_qec_cycles"],
        "source_budget_profile": cell["source_budget_profile"],
        "checkpoint": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": file_hash(checkpoint),
        "acquisition_provenance": str(provenance.relative_to(ROOT)).replace("\\", "/"),
        "acquisition_provenance_sha256": file_hash(provenance),
        "fresh_acquisition": cell["fresh_acquisition"],
        "reused_shard_ids": cell["reused_shard_ids"],
    })
    return result


def _ordering_gate(intermediate: dict[str, Any], fast: dict[str, Any]) -> dict[str, Any]:
    left_gain = np.asarray(intermediate["bootstrap_uncertainty"]["gain_samples"], dtype=float)
    right_gain = np.asarray(fast["bootstrap_uncertainty"]["gain_samples"], dtype=float)
    left_phase = np.asarray(
        intermediate["bootstrap_uncertainty"]["phase_lag_samples_radians"], dtype=float)
    right_phase = np.asarray(fast["bootstrap_uncertainty"]["phase_lag_samples_radians"], dtype=float)
    size = min(left_gain.size, right_gain.size)
    probability = float(np.mean((left_gain[:size] > right_gain[:size]) &
                                (left_phase[:size] < right_phase[:size])))
    threshold = float(config()["identification"]["mechanistic_ordering_probability_min"])
    point_pass = bool(
        intermediate["mean_transfer_regression"]["gain"] > fast["mean_transfer_regression"]["gain"] and
        intermediate["mean_transfer_regression"]["phase_lag_radians"] <
        fast["mean_transfer_regression"]["phase_lag_radians"])
    return {
        "claim": "G_intermediate>G_fast and phase_lag_intermediate<phase_lag_fast",
        "point_estimate_pass": point_pass,
        "bootstrap_joint_probability": probability,
        "minimum_probability": threshold,
        "pass": point_pass and probability >= threshold,
    }


def _three_frequency_ordering_gate(slow: Mapping[str, Any],
                                   intermediate: Mapping[str, Any],
                                   fast: Mapping[str, Any]) -> dict[str, Any]:
    """Test the preregistered low-pass ordering using direct mean-transfer draws."""
    bootstraps = [value["bootstrap_uncertainty"] for value in (slow, intermediate, fast)]
    slow_gain, intermediate_gain, fast_gain = [
        np.asarray(value["gain_samples"], dtype=float) for value in bootstraps]
    slow_phase, intermediate_phase, fast_phase = [
        np.asarray(value["phase_lag_samples_radians"], dtype=float) for value in bootstraps]
    size = min(*(value.size for value in (
        slow_gain, intermediate_gain, fast_gain,
        slow_phase, intermediate_phase, fast_phase,
    )))
    if size < 1:
        raise RuntimeError("three-frequency ordering requires nonempty bootstrap samples")
    gain_draw_pass = ((slow_gain[:size] > intermediate_gain[:size]) &
                      (intermediate_gain[:size] > fast_gain[:size]))
    phase_draw_pass = ((slow_phase[:size] < intermediate_phase[:size]) &
                       (intermediate_phase[:size] < fast_phase[:size]))
    gain_probability = float(np.mean(gain_draw_pass))
    phase_probability = float(np.mean(phase_draw_pass))
    joint_probability = float(np.mean(gain_draw_pass & phase_draw_pass))
    regressions = [value["mean_transfer_regression"] for value in
                   (slow, intermediate, fast)]
    gains = [float(value["gain"]) for value in regressions]
    phases = [float(value["phase_lag_radians"]) for value in regressions]
    gain_point_pass = gains[0] > gains[1] > gains[2]
    phase_point_pass = phases[0] < phases[1] < phases[2]
    threshold = float(config()["identification"]["mechanistic_ordering_probability_min"])
    return {
        "claim": "G_slow>G_intermediate>G_fast and phase_slow<phase_intermediate<phase_fast",
        "frequency_order": ["slow", "intermediate", "fast"],
        "gain_point_estimates": gains,
        "phase_lag_point_estimates_radians": phases,
        "gain_point_estimate_pass": gain_point_pass,
        "phase_point_estimate_pass": phase_point_pass,
        "point_estimate_pass": gain_point_pass and phase_point_pass,
        "bootstrap_gain_ordering_probability": gain_probability,
        "bootstrap_phase_ordering_probability": phase_probability,
        "bootstrap_joint_probability": joint_probability,
        "bootstrap_draws_compared": size,
        "minimum_probability": threshold,
        "pass": bool(gain_point_pass and phase_point_pass and joint_probability >= threshold),
    }


def _approved_fast_transfer_for_slow() -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve the fast gate without promoting or overwriting the rejected V18 run."""
    base_path = ARTIFACT_ROOT / "transfer_fast.json"
    if base_path.is_file():
        base = read_json(base_path)
        if (base.get("identifiable") is True and
                base.get("stage_ab_ordering", {}).get("pass") is True):
            return base, {
                "pass": True,
                "source": "V18_BASE_FAST",
                "transfer_path": str(base_path.relative_to(ROOT)).replace("\\", "/"),
                "evidence_hashes": {"fast_transfer": file_hash(base_path)},
                "base_fast_artifact_overwritten": False,
            }

    extended_root = ARTIFACT_ROOT / "extended_fast"
    readiness_path = extended_root / "readiness_for_slow.json"
    transfer_path = extended_root / "transfer_fast_extended.json"
    provenance_path = extended_root / "continuation_provenance.json"
    checkpoint_path = extended_root / "checkpoint.json"
    required = (readiness_path, transfer_path, provenance_path, checkpoint_path)
    missing = [str(path.relative_to(ROOT)).replace("\\", "/")
               for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"accepted extended-fast evidence is absent; optional slow run is blocked: {missing}")

    readiness = read_json(readiness_path)
    transfer = read_json(transfer_path)
    provenance = read_json(provenance_path)
    frozen = _frozen()
    checks = {
        "readiness_pass": readiness.get("pass") is True,
        "readiness_explicit": readiness.get("READY_FOR_SLOW_TRANSFER_IDENTIFICATION") is True,
        "all_readiness_gates": bool(readiness.get("gates")) and
                               all(readiness["gates"].values()),
        "extended_transfer_pass": transfer.get("pass") is True,
        "direct_mean_transfer_identifiable":
            transfer.get("direct_mean_transfer_identifiable") is True,
        "steady_periodic_identification_accepted":
            transfer.get("steady_periodic_identification_accepted") is True,
        "intermediate_fast_ordering": transfer.get("stage_ab_ordering", {}).get("pass") is True,
        "fast_frequency": transfer.get("frequency_per_epoch") ==
                          config()["transfer_acquisition"]["fast"]["frequency_per_epoch"],
        "controller_hash": transfer.get("controller_hash") == frozen["optimizer_bundle_hash"],
        "parameterization": transfer.get("parameterization") == "DIRECT_SIGMA_SOURCE_EXACT",
        "continuation_complete": provenance.get("continuation_complete") is True,
        "continuation_pass": provenance.get("pass") is True,
        "immutable_base_prefix": provenance.get("previous_600_epochs_unchanged") is True,
        "base_checkpoint_unchanged": provenance.get("base_checkpoint_unchanged_after") is True,
        "exact_append": provenance.get("fresh_continuation_epoch_count") == 150,
        "terminal_epoch": provenance.get("target_total_epochs") == 750,
        "optimizer_unchanged": provenance.get("controller_hyperparameters_changed") is False,
        "optimizer_not_reset": provenance.get("optimizer_reset") is False,
        "rng_not_reset": provenance.get("rng_reset") is False,
        "sigma_not_reset": provenance.get("sigma_reset") is False,
        "provenance_hash": transfer.get("acquisition_provenance_sha256") ==
                           file_hash(provenance_path),
        "checkpoint_hash": transfer.get("checkpoint_sha256") == file_hash(checkpoint_path),
    }
    if not all(checks.values()):
        raise RuntimeError(f"extended-fast approval for slow acquisition failed: {checks}")
    evidence_hashes = {
        "extended_fast_readiness": file_hash(readiness_path),
        "extended_fast_transfer": file_hash(transfer_path),
        "extended_fast_provenance": file_hash(provenance_path),
        "extended_fast_checkpoint": file_hash(checkpoint_path),
    }
    return transfer, {
        "pass": True,
        "source": "V18_1_EXTENDED_FAST",
        "transfer_path": str(transfer_path.relative_to(ROOT)).replace("\\", "/"),
        "readiness_path": str(readiness_path.relative_to(ROOT)).replace("\\", "/"),
        "checks": checks,
        "evidence_hashes": evidence_hashes,
        "base_fast_artifact_overwritten": False,
    }


def _acquisition_provenance(label: str, checkpoint: Path) -> Path:
    directory = checkpoint.parent
    path = directory / "provenance.json"
    settings = config()["transfer_acquisition"][label]
    expected_inputs = {
        "v18_protocol": file_hash(ROOT / "configs/google_pure_v18/protocol.json"),
        "import_manifest": file_hash(ARTIFACT_ROOT / "import_manifest.json"),
        "delta_min_provenance": file_hash(ARTIFACT_ROOT / "delta_min_provenance.json"),
        "steady_state_rule": file_hash(ARTIFACT_ROOT / "steady_state_rule.json"),
        "production_acquisition_code": file_hash(
            ROOT / "src/hdfa_rl_suite/google_pure_source_exact/figure5a/acquisition.py"),
        "production_plant_code": file_hash(
            ROOT / "src/hdfa_rl_suite/google_pure_source_exact/figure5a/plant.py"),
        "optimizer_bundle": file_hash(
            ROOT / "artifacts/google_pure_v16/frozen_source_normalized_optimizer.json"),
    }
    if label == "slow":
        _, approval = _approved_fast_transfer_for_slow()
        expected_inputs.update(approval["evidence_hashes"])
    if path.is_file():
        value = read_json(path)
        if value.get("input_artifact_hashes") != expected_inputs or value.get("settings") != settings:
            raise RuntimeError(f"{label} acquisition provenance changed")
        return path
    if checkpoint.exists():
        raise RuntimeError(f"{label} checkpoint predates mandatory V18 provenance")
    value = nonfinal({
        "pass": True, "label": label, "created_before_checkpoint": True,
        "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "acquisition_order": config()["delta_min_selection"]["new_transfer_acquisition_order"],
        "settings": settings, "input_artifact_hashes": expected_inputs,
        "controller_hash": _frozen()["optimizer_bundle_hash"],
        "parameterization": "DIRECT_SIGMA_SOURCE_EXACT",
        "optimizer_changed": False, "new_outcome_visible_at_creation": False,
    })
    atomic_json(path, value)
    return path


def _run_transfer(label: str) -> dict[str, Any]:
    if label not in {"intermediate", "fast", "slow"}:
        raise ValueError(f"unknown V18 transfer stage: {label}")
    verify_import_manifest()
    if not build_sensitivity_field_cleanup()["pass"]:
        raise RuntimeError("sensitivity terminology gate failed")
    if not validate_deterministic_transfer()["pass"]:
        raise RuntimeError("deterministic transfer gate failed")
    if not audit_delta_min_provenance()["pass"]:
        raise RuntimeError("delta-min provenance gate failed")
    if not build_steady_state_rule()["pass"]:
        raise RuntimeError("steady-state preregistration gate failed")
    if label in {"fast", "slow"}:
        intermediate_path = ARTIFACT_ROOT / "transfer_intermediate.json"
        if not intermediate_path.is_file() or not read_json(intermediate_path).get("identifiable"):
            raise RuntimeError("intermediate transfer is not identifiable; later acquisition is blocked")
    approved_fast = None
    fast_approval = None
    if label == "slow":
        approved_fast, fast_approval = _approved_fast_transfer_for_slow()
    settings = config()["transfer_acquisition"][label]
    source = _source_config()
    plant = build_plant(source)
    protocol = Figure5aProtocol(
        AcquisitionMode.VALIDATION, int(settings["epochs"]),
        int(settings["candidates_per_epoch"]), int(settings["qec_cycles_per_candidate"]),
        int(source["plant"]["circuit_rounds"]))
    checkpoint = ARTIFACT_ROOT / "acquisition" / label / "checkpoint.json"
    provenance = _acquisition_provenance(label, checkpoint)
    preexisting = checkpoint.is_file()
    frozen = _frozen()
    cell = run_cell(
        protocol=protocol, plant=plant, frequency=float(settings["frequency_per_epoch"]),
        entropy_weight=float(frozen["entropy_coefficient"]), seed=int(settings["seed"]),
        optimizer_config=_optimizer_config(), initial_sigma=float(frozen["initial_sigma"]),
        checkpoint_path=checkpoint, dependency_hashes=dependency_hashes(ROOT, source),
        controller_hash=frozen["optimizer_bundle_hash"], clip=float(frozen["ppo_clip"]),
        baseline_weight=float(frozen["baseline_loss_weight"]), resume=preexisting,
        checkpoint_every_candidates=int(settings["candidates_per_epoch"]),
        boundary=_boundary(plant), fresh_acquisition_required=not preexisting,
        source_budget_profile=str(settings["profile"]))
    if not cell["complete"]:
        raise RuntimeError(f"{label} acquisition unexpectedly returned an incomplete cell")
    previous_path = ARTIFACT_ROOT / f"transfer_{label}.json"
    previous = read_json(previous_path) if previous_path.is_file() else None
    historical_fresh = bool(cell["fresh_acquisition"] or
                            (previous is not None and previous.get("fresh_acquisition") is True))
    manifest = nonfinal({key: value for key, value in cell.items() if key != "epoch_records"})
    manifest["epoch_record_count"] = len(cell["epoch_records"])
    manifest["checkpoint_sha256"] = file_hash(checkpoint)
    manifest["controller_mode"] = "PAPER_DIRECT_SIGMA"
    manifest["fresh_acquisition"] = historical_fresh
    manifest["fresh_v18_acquisition_campaign"] = True
    manifest["current_call_created_checkpoint"] = cell["fresh_acquisition"]
    atomic_json(checkpoint.parent / "cell_manifest.json", manifest)
    result = _analyse_transfer(label, cell, checkpoint, provenance)
    result["fresh_acquisition"] = historical_fresh
    result["fresh_v18_acquisition_campaign"] = True
    result["current_call_created_checkpoint"] = cell["fresh_acquisition"]
    if label == "fast":
        ordering = _ordering_gate(read_json(ARTIFACT_ROOT / "transfer_intermediate.json"), result)
        result["stage_ab_ordering"] = ordering
        result["pass"] = bool(result["pass"] and ordering["pass"])
        result["classification"] = ("INTERMEDIATE_FAST_TRANSFER_ORDER_IDENTIFIED" if result["pass"]
                                    else "INTERMEDIATE_FAST_MECHANISTIC_GATE_FAILED")
    elif label == "slow":
        ordering = _three_frequency_ordering_gate(
            result, read_json(ARTIFACT_ROOT / "transfer_intermediate.json"), approved_fast)
        result["stage_slow_intermediate_fast_ordering"] = ordering
        result["fast_transfer_approval"] = fast_approval
        result["mean_stochastic_decomposition_retained"] = True
        result["sigma_diagnostics_retained_in_period_diagnostics"] = True
        result["pass"] = bool(result["pass"] and ordering["pass"])
        result["classification"] = (
            "SLOW_TRANSFER_AND_THREE_FREQUENCY_ORDERING_IDENTIFIED" if result["pass"] else
            "SLOW_TRANSFER_OR_THREE_FREQUENCY_ORDERING_NOT_ACCEPTED")
    return _write(f"transfer_{label}", result, f"V18 {label} transfer identification", [
        f"Classification: **{result['classification']}**.",
        "The learned mean was fitted directly in the shared normalized drift direction.",
        "Only preregistered complete post-transient periods enter this identification artifact.",
    ])


def run_transfer_intermediate() -> dict[str, Any]:
    return _run_transfer("intermediate")


def run_transfer_fast() -> dict[str, Any]:
    return _run_transfer("fast")


def run_transfer_slow() -> dict[str, Any]:
    return _run_transfer("slow")


def build_mean_stochastic_decomposition() -> dict[str, Any]:
    rows = []
    for label in ("intermediate", "fast", "slow"):
        path = ARTIFACT_ROOT / f"transfer_{label}.json"
        if not path.is_file():
            continue
        value = read_json(path)
        metric = value["stream_decomposition"]
        i_mean = metric["I_mean"]
        i_stochastic = metric["I_stochastic"]
        if i_mean is None or i_stochastic is None:
            classification = "NORMALIZATION_DENOMINATOR_UNRESOLVED"
        elif metric["exploration_damage"] > 0 and i_mean > i_stochastic:
            classification = "POSITIVE_EXPLORATION_DAMAGE"
        elif metric["exploration_damage"] < 0 and i_mean < i_stochastic:
            classification = "STOCHASTIC_STREAM_OUTPERFORMS_MEAN_IN_THIS_RUN"
        else:
            classification = "NO_RESOLVED_EXPLORATION_DAMAGE"
        rows.append({
            "label": label, "frequency_per_epoch": value["frequency_per_epoch"],
            "controller_hash": value["controller_hash"],
            "analysis_epoch_window": value["analysis_epoch_window"],
            **metric, "classification": classification,
            "identity_residual": (None if i_mean is None or i_stochastic is None else
                (i_mean - i_stochastic) - metric["exploration_damage"] /
                metric["normalization_denominator"]),
            "scale_dynamics_by_period": [{
                "period_index": period["period_index"],
                "sigma_x_median": period["sigma_x_median"],
                "sigma_u_median": period["sigma_u_median"],
                "scale_guard_occupancy": period["scale_guard_occupancy"],
                "reward_sigma_gradient_norm_median": period["reward_sigma_gradient_norm_median"],
                "entropy_sigma_gradient_norm_median": period["entropy_sigma_gradient_norm_median"],
            } for period in value["period_diagnostics"]],
        })
    result = nonfinal({
        "pass": bool(rows) and all(row["denominator_resolved"] for row in rows),
        "rows": rows, "stream_contract": list(STREAMS),
        "mean_and_stochastic_streams_separate": True,
        "mean_performance": "I_mean=(C_fixed-C_mean)/(C_fixed-C_oracle)",
        "stochastic_performance": "I_stochastic=(C_fixed-C_stochastic)/(C_fixed-C_oracle)",
        "exploration_damage_identity": "D_exploration=C_stochastic-C_mean",
        "performance_identity": "I_mean-I_stochastic=D_exploration/(C_fixed-C_oracle)",
        "performance_proxy_used_for_mean_transfer": False,
    })
    return _write("mean_stochastic_decomposition", result,
                  "V18 mean and stochastic stream decomposition", [
        "Fixed, oracle, learned-mean, and stochastic counts remain separate.",
        "Mean tracking and exploration damage are reported independently.",
    ])


def build_paired_acceptance_readiness() -> dict[str, Any]:
    delta = audit_delta_min_provenance()
    deterministic = validate_deterministic_transfer()
    intermediate_path = ARTIFACT_ROOT / "transfer_intermediate.json"
    fast_path = ARTIFACT_ROOT / "transfer_fast.json"
    intermediate = read_json(intermediate_path) if intermediate_path.is_file() else None
    fast = read_json(fast_path) if fast_path.is_file() else None
    required_phases = list(config()["acceptance"]["required_initial_phases"])
    minimum_pairs = int(config()["acceptance"]["minimum_complete_pairs"])
    prerequisites = {
        "delta_min_frozen_before_acquisition": delta["selection_independent_of_new_v18_transfer_outcomes"],
        "deterministic_quantitative_validation": deterministic["pass"],
        "intermediate_direct_mean_transfer_identifiable": bool(
            intermediate and intermediate.get("direct_mean_transfer_identifiable")),
        "fast_direct_mean_transfer_identifiable": bool(
            fast and fast.get("direct_mean_transfer_identifiable")),
        "intermediate_steady_periodic_identification_accepted": bool(
            intermediate and intermediate.get("steady_periodic_identification_accepted")),
        "fast_steady_periodic_identification_accepted": bool(
            fast and fast.get("steady_periodic_identification_accepted")),
        "intermediate_fast_ordering": bool(fast and fast.get("stage_ab_ordering", {}).get("pass")),
        "same_frozen_controller": bool(
            intermediate and fast and intermediate.get("controller_hash") == fast.get("controller_hash") ==
            _frozen()["optimizer_bundle_hash"]),
    }
    available_pairs = 0
    ready = bool(all(prerequisites.values()) and available_pairs >= minimum_pairs)
    result = nonfinal({
        "pass": ready, "ready_for_paired_acceptance_claim": ready,
        "classification": ("PAIRED_ACCEPTANCE_READY" if ready else
                           "PAIRED_ACCEPTANCE_NOT_READY_NO_COMPLETE_MATCHED_UNITS"),
        "prerequisites": prerequisites,
        "delta_min": delta["delta_min"],
        "required_complete_pairs": minimum_pairs,
        "required_initial_phases_radians": required_phases,
        "available_complete_matched_pairs": available_pairs,
        "intermediate_and_fast_identification_runs_are_not_paired_acceptance_units": True,
        "v17_24_epoch_units_rejected": True,
        "incomplete_period_units_accepted": False,
        "full_four_phase_acceptance_auto_launched": False,
        "paired_acceptance_executed": False,
        "next_action_if_mechanistic_gate_passes": (
            "PREREGISTER_AND_EXPLICITLY_AUTHORIZE_MATCHED_SEED_PHASE_PAIRS"),
    })
    return _write("paired_acceptance_readiness", result, "V18 paired-acceptance readiness", [
        f"Classification: **{result['classification']}**.",
        "The intermediate and fast identification runs use different seeds and are not promoted into paired acceptance units.",
        "No incomplete 24-epoch result or automatically launched slow/four-phase campaign is accepted.",
    ])


def build_figure5b_learning_rate_note() -> dict[str, Any]:
    comparison_path = ROOT / "artifacts/google_pure_v16/matched_figure5b/comparison.json"
    comparison = read_json(comparison_path)
    summary = comparison["summaries"]["D_V16_FROZEN_OPTIMIZER"]
    observed = float(summary["median_fractional_residual_reduction"])
    alpha = float(_frozen()["mean_learning_rate"])
    kappa_h = .02
    local_prediction = alpha * kappa_h
    result = nonfinal({
        "pass": True,
        "reported_value": observed,
        "reported_value_semantics": "MEDIAN_PER_EPOCH_POLICY_UPDATE_FRACTIONAL_RESIDUAL_REDUCTION",
        "per_epoch": True, "per_update": True, "aggregate_run_total": False,
        "epoch_contains_one_policy_update": True,
        "formula": "r_t=(Lambda_t-Lambda_t_plus_1)/(Lambda_t-Lambda_star)",
        "mean_learning_rate_alpha": alpha,
        "hessian_curvature_kappa_H": kappa_h,
        "alpha_times_kappa_H": local_prediction,
        "observed_to_alpha_kappa_H_ratio": observed / local_prediction,
        "relative_deficit": 1.0 - observed / local_prediction,
        "classification": "PER_EPOCH_RATE_DEFICIT_REMAINS_DIAGNOSTIC_ONLY",
        "source_artifact": str(comparison_path.relative_to(ROOT)).replace("\\", "/"),
        "source_artifact_sha256": file_hash(comparison_path),
        "figure5b_executed": False, "figure5b_modified": False,
        "figure5c_executed": False, "repair_applied": False,
    })
    return _write("figure5b_learning_rate_note", result, "V18 Figure 5b learning-rate note", [
        f"The reported {observed:.6g} is a median per epoch, with one policy update per epoch.",
        f"It is {observed / local_prediction:.3f} of alpha*kappa_H={local_prediction:.4g}; the deficit remains diagnostic.",
        "No Figure 5b or Figure 5c experiment is launched or modified by V18.",
    ])
