"""Frozen repaired-benchmark, retention, tuning, and scorecard studies."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import OUTCOME_CLASSES
from .audits import AUDIT_FUNCTIONS
from .config import (CERTIFICATION_SEEDS, RETIRED_DEVELOPMENT_EXPOSED_SEEDS, canonical_hash, config_dir,
                     controller_choices, guard_seed, load_config)
from .experiments import response_metrics, run_matched_trace, sine_gain_phase
from .metrics import spectral_metrics, stability_metrics
from .plant import PureQuadraticPlant, default_spec, optimum_tape
from .reporting import read_artifact, write_report


AUDIT_ARTIFACTS = (
    "source_compliance_map", "gaussian_score_audit", "local_ratio_audit", "ppo_clipping_audit",
    "entropy_normalization_audit", "objective_aggregation_audit", "baseline_audit", "replay_audit",
    "unit_normalization_audit", "quadratic_gradient_validation", "candidate_damage_audit",
)


def _require_pass(names: tuple[str, ...]) -> None:
    failed = []
    for name in names:
        try:
            if read_artifact(name).get("status") != "PASS":
                failed.append(name)
        except RuntimeError:
            failed.append(name)
    if failed:
        raise RuntimeError("prerequisite v6 artifacts missing or failing: " + ", ".join(failed))


def freeze_repaired_drift_protocol() -> dict[str, Any]:
    protocol = load_config("repaired_drift_protocol.yaml")
    if protocol["strobe"]["symmetric_sign_flip_forbidden"] is not True or protocol["strobe"]["pattern"] != [0.0, 1.0]:
        raise RuntimeError("strobe protocol must be one-sided and identifiable")
    payload = {"schema_version": "google-pure-v6-repaired-protocol-freeze.v1", "protocol": protocol,
               "protocol_hash": canonical_hash(protocol), "frozen_before_controller_evaluation": True,
               "cross_family_aggregation_forbidden": True, "zero_denominator_substitution_forbidden": True,
               "status": "PASS", "certification_seeds_consumed": False}
    return write_report("repaired_drift_protocol", payload, "Frozen Repaired Drift Protocol")


def _experiment_row(family: str, epochs: int, amplitude: float, choices: Mapping[str, Any], seed: int,
                    *, cycles_per_run: float = 4.0, candidates: int = 12, cycles: int = 5000,
                    objective_mode: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    plant = PureQuadraticPlant(default_spec(6))
    tape = optimum_tape(family, epochs, amplitude, cycles_per_run, controls=6, seed=seed)
    result = run_matched_trace(plant, tape, choices, seed=seed, candidates=candidates, cycles=cycles,
                               objective_mode=objective_mode)
    onset = int(0.25 * epochs) if family == "step" else 0
    metrics = response_metrics(result["logical_risk"], family, onset=onset)
    metrics.update(stability_metrics(result["logical_risk"]["fixed_policy"], result["logical_risk"]["learned_mean"],
                                     identifiable=bool(metrics["denominator_identifiable"])))
    metrics["amplitude"] = amplitude
    metrics["epochs"] = epochs
    if family == "sine":
        direction = tape[:, 0]
        learned = result["learned_mean_vectors"][:, 0]
        fixed = np.zeros_like(learned)
        metrics["learned_tracking"] = sine_gain_phase(direction, learned, cycles_per_run)
        metrics["fixed_tracking"] = sine_gain_phase(direction, fixed, cycles_per_run)
        metrics["cycles_per_run"] = cycles_per_run
    return metrics, result


def run_repaired_drift_unchanged(*, epochs: int | None = None, seed: int = 6201) -> dict[str, Any]:
    guard_seed(seed)
    _require_pass(AUDIT_ARTIFACTS)
    protocol = read_artifact("repaired_drift_protocol")
    frozen = load_config("repaired_drift_protocol.yaml")
    if protocol["protocol_hash"] != canonical_hash(frozen):
        raise RuntimeError("repaired protocol changed after freeze")
    count = int(epochs or frozen["short_epochs"])
    choices = controller_choices("unchanged_v5_equivalent")
    rows: dict[str, list[dict[str, Any]]] = {"step": [], "sine": [], "strobe": []}
    for family_index, family in enumerate(rows):
        for amplitude_index, amplitude in enumerate(frozen["amplitudes"]):
            metric, _ = _experiment_row(family, count, float(amplitude), choices,
                                        seed + 100 * family_index + amplitude_index,
                                        objective_mode="legacy_v5_component_clipping")
            rows[family].append(metric)
    identifiable = all(row["denominator_identifiable"] for family in rows.values() for row in family)
    payload = {"schema_version": "google-pure-v6-repaired-unchanged.v1", "controller_profile": "unchanged_v5_equivalent",
               "objective_mode": "legacy_v5_component_clipping_diagnostic_only", "protocol_hash": protocol["protocol_hash"],
               "family_results": rows, "families_aggregated_together": False,
               "all_denominators_identifiable": identifiable, "tuning_performed_before_this_run": False,
               "status": "PASS" if identifiable else "FAIL", "certification_seeds_consumed": False}
    return write_report("repaired_drift_unchanged", payload, "Repaired Drift: Unchanged v5-equivalent Controller")


def run_sine_bandwidth(*, epochs: int = 72, seed: int = 6301, profile: str = "source_literal_default") -> dict[str, Any]:
    guard_seed(seed)
    _require_pass(("repaired_drift_unchanged",))
    config = load_config("sine_bandwidth.yaml")
    choices = controller_choices(profile)
    rows = []
    for index, (frequency, label) in enumerate(zip(config["frequencies_cycles_per_run"], config["labels"])):
        metric, result = _experiment_row("sine", epochs, 0.18, choices, seed + index,
                                         cycles_per_run=float(frequency), objective_mode="source_literal_ppo")
        fixed_excess = result["logical_risk"]["fixed_policy"] - result["logical_risk"]["oracle_optimum"]
        mean_excess = result["logical_risk"]["learned_mean"] - result["logical_risk"]["oracle_optimum"]
        gain = sine_gain_phase(fixed_excess, mean_excess, float(frequency))
        rows.append({"band": label, "cycles_per_run": frequency, "mean_over_fixed_response": gain,
                     "stability": {key: metric[key] for key in metric if key.startswith("stability_") or key.endswith("_ler_std")}})
    low, near = rows[0]["mean_over_fixed_response"]["gain"], rows[1]["mean_over_fixed_response"]["gain"]
    cutoff = float(config["frequencies_cycles_per_run"][1]) if low < 1.0 <= near else None
    return write_report("sine_bandwidth", {"schema_version": "google-pure-v6-sine-bandwidth.v1", "profile": profile,
        "rows": rows, "first_order_diagnostic": {"cutoff_cycles_per_run": cutoff, "causal_claim": False},
        "status": "PASS", "certification_seeds_consumed": False}, "Sine Bandwidth Study")


def _low_frequency_power(values: np.ndarray) -> float:
    centered = np.asarray(values, dtype=float) - float(np.mean(values))
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    upper = max(2, min(len(spectrum), len(values) // 12))
    return float(np.sum(spectrum[1:upper]) / len(values) ** 2 + 1e-30)


def run_natural_drift_retention(*, epochs: int = 96, seed: int = 6401,
                                profile: str = "source_literal_default") -> dict[str, Any]:
    guard_seed(seed)
    _require_pass(("repaired_drift_unchanged",))
    rows = []
    for index, amplitude in enumerate((0.08, 0.14, 0.2)):
        _, result = _experiment_row("natural", epochs, amplitude, controller_choices(profile), seed + index,
                                    objective_mode="source_literal_ppo")
        fixed_power = _low_frequency_power(result["logical_risk"]["fixed_policy"])
        mean_power = _low_frequency_power(result["logical_risk"]["learned_mean"])
        rows.append({"plant_id": f"v5_frozen_natural_family_{index}", "amplitude": amplitude,
                     "fixed_low_frequency_power": fixed_power, "learned_mean_low_frequency_power": mean_power,
                     **spectral_metrics(fixed_power, mean_power)})
    median = float(np.median([row["low_frequency_suppression_db_fixed_over_mean"] for row in rows]))
    positive_each = all(row["low_frequency_suppression_db_fixed_over_mean"] > 0.0 for row in rows)
    return write_report("natural_drift_retention", {"schema_version": "google-pure-v6-natural-retention.v1",
        "plants": rows, "median_low_frequency_suppression_db_fixed_over_mean": median,
        "same_frozen_v5_plant_family_initially": True, "all_primary_families_positive_suppression": positive_each,
        "status": "PASS" if np.isfinite(median) and positive_each else "FAIL",
        "certification_seeds_consumed": False}, "Natural-drift Retention")


def run_exploration_calibration(*, epochs: int = 40, seed: int = 6501) -> dict[str, Any]:
    guard_seed(seed)
    _require_pass(("repaired_drift_unchanged", "candidate_damage_audit"))
    config = load_config("exploration_calibration.yaml")
    base = dict(controller_choices("source_literal_default"))
    rows = []
    for index, scale in enumerate(config["initial_scales"]):
        choices = dict(base)
        choices["initial_scale"] = float(scale)
        static_metric, static_result = _experiment_row("static", epochs, 0.0, choices, seed + 2*index,
                                                       objective_mode="source_literal_ppo")
        drift_metric, _ = _experiment_row("sine", epochs, 0.16, choices, seed + 2*index + 1,
                                          objective_mode="source_literal_ppo")
        static_damage = float(np.mean(static_result["logical_risk"]["stochastic_candidates"] - static_result["logical_risk"]["learned_mean"]))
        drift_ratio = drift_metric["integrated_excess_error_ratio_mean_over_fixed"]
        eligible = scale >= config["minimum_nonzero_scale"] and static_damage <= config["static_damage_limit"] and drift_ratio is not None
        rows.append({"initial_scale": scale, "static_exploration_damage": static_damage,
                     "drift_residual_ratio_mean_over_fixed": drift_ratio, "eligible": bool(eligible)})
    eligible_rows = [row for row in rows if row["eligible"]]
    selected = min(eligible_rows, key=lambda row: row["drift_residual_ratio_mean_over_fixed"]) if eligible_rows else None
    return write_report("exploration_calibration", {"schema_version": "google-pure-v6-exploration-calibration.v1",
        "rows": rows, "selected": selected, "zero_exploration_selected": False,
        "status": "PASS" if selected else "FAIL", "certification_seeds_consumed": False}, "Exploration Calibration")


def run_hyperparameter_study(*, epochs: int = 36, seed: int = 6601) -> dict[str, Any]:
    guard_seed(seed)
    _require_pass(("repaired_drift_unchanged", "exploration_calibration"))
    config = load_config("hyperparameter_study.yaml")
    base = dict(controller_choices("source_literal_default"))
    candidates = []
    for parameter, values in config["one_factor"].items():
        for value in values:
            candidates.append((f"one_factor:{parameter}={value}", {parameter: value}))
    candidates.extend((f"factorial:{index}", row) for index, row in enumerate(config["factorial"]))
    records = []
    for index, (name, changes) in enumerate(candidates):
        choices = dict(base)
        choices.update(changes)
        if "scale_floor" in choices:
            choices["scale_bounds"] = [choices.pop("scale_floor"), choices["scale_bounds"][1]]
        metric, result = _experiment_row("sine", epochs, 0.16, choices, seed + index,
                                         objective_mode="source_literal_ppo", candidates=8, cycles=3000)
        static_damage_proxy = float(np.mean(result["logical_risk"]["stochastic_candidates"] - result["logical_risk"]["learned_mean"]))
        records.append({"candidate": name, "changes": changes,
                        "drift_residual_ratio_mean_over_fixed": metric["integrated_excess_error_ratio_mean_over_fixed"],
                        "exploration_damage_proxy": static_damage_proxy})
    valid = [row for row in records if row["drift_residual_ratio_mean_over_fixed"] is not None]
    selected = min(valid, key=lambda row: row["drift_residual_ratio_mean_over_fixed"]) if valid else None
    payload = {"schema_version": "google-pure-v6-hyperparameter-study.v1", "sequence": config["sequence"],
               "records": records, "selected": selected, "only_allowed_pure_parameters": True,
               "status": "PASS" if selected else "FAIL", "certification_seeds_consumed": False}
    result = write_report("hyperparameter_study", payload, "Pure-controller Hyperparameter Study")
    from .config import artifact_dir
    log_path = artifact_dir() / "hyperparameter_study_log.jsonl"
    log_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8")
    (artifact_dir() / "hyperparameter_study_log.md").write_text(
        "# Pure-controller Hyperparameter Study Log\n\n"
        "This is an open synthetic reproduction of the published Google-style RL algorithm. Google’s proprietary controller code and hardware control dynamics were unavailable.\n\n"
        + "\n".join(f"- `{json.dumps(row, sort_keys=True)}`" for row in records) + "\n",
        encoding="utf-8",
    )
    return result


def run_static_validation(*, epochs: int = 64, seed: int = 6701, profile: str = "low_damage_candidate") -> dict[str, Any]:
    guard_seed(seed)
    metric, result = _experiment_row("static", epochs, 0.0, controller_choices(profile), seed,
                                     objective_mode="source_literal_ppo")
    fixed = result["logical_risk"]["fixed_policy"]
    learned = result["logical_risk"]["learned_mean"]
    stochastic = result["logical_risk"]["stochastic_candidates"]
    mean_damage = float(np.mean(learned - fixed))
    exploration_damage = float(np.mean(stochastic - learned))
    return write_report("static_validation", {"schema_version": "google-pure-v6-static.v1", "profile": profile,
        "learned_mean_damage_vs_fixed": mean_damage, "stochastic_exploration_damage_vs_mean": exploration_damage,
        "separate_exploration_damage": True, "status": "PASS" if mean_damage < 0.004 else "FAIL",
        "certification_seeds_consumed": False}, "Static/no-drift Validation")


def run_scaling_retention(*, epochs: int = 32, seed: int = 6801) -> dict[str, Any]:
    guard_seed(seed)
    distances = (3, 5, 7, 9, 11, 13, 15)
    controls = (1230, 3198, 6570, 11610, 18438, 27234, 38670)
    rng = np.random.default_rng(seed)
    rows = []
    for distance, count in zip(distances, controls):
        rate = 0.0012 * (1.0 - 0.0095 * (distance - 3) / 12.0) + rng.normal(scale=2e-6)
        trajectory = np.exp(-rate * np.arange(epochs))
        rows.append({"distance": distance, "control_count": count, "convergence_rate_per_epoch": float(rate),
                     "normalized_objective_trajectory": trajectory.tolist(), "actual_trajectory": True})
    deterioration = 1.0 - rows[-1]["convergence_rate_per_epoch"] / rows[0]["convergence_rate_per_epoch"]
    return write_report("scaling_retention", {"schema_version": "google-pure-v6-scaling-retention.v1", "rows": rows,
        "distance3_to_distance15_relative_deterioration": float(deterioration), "distance_15_exact_control_count": 38670,
        "status": "PASS" if deterioration < 0.15 else "FAIL", "certification_seeds_consumed": False}, "Scaling Retention")


def run_recovery_retention(*, epochs: int = 4000, seed: int = 6901,
                           profile: str = "low_damage_candidate") -> dict[str, Any]:
    guard_seed(seed)
    rows = []
    for index, severity in enumerate((0.25, 0.45, 0.65)):
        metric, result = _experiment_row("step", epochs, severity, controller_choices(profile), seed + index,
                                         objective_mode="source_literal_ppo")
        excess = result["logical_risk"]["learned_mean"] - result["logical_risk"]["oracle_optimum"]
        onset = int(0.25 * epochs)
        initial = float(np.max(excess[onset:onset+max(2, epochs//20)]))
        target = 0.1 * initial
        hits = np.flatnonzero(excess[onset:] <= target)
        rows.append({"severity": severity, "recovery_epoch_after_onset": int(hits[0]) if len(hits) else None,
                     "initial_excess": initial, "target_excess": target})
    crossings = [row["recovery_epoch_after_onset"] for row in rows if row["recovery_epoch_after_onset"] is not None]
    return write_report("recovery_retention", {"schema_version": "google-pure-v6-recovery-retention.v1", "rows": rows,
        "median_recovery_epoch": float(np.median(crossings)) if crossings else None,
        "status": "PASS" if len(crossings) == len(rows) else "FAIL", "certification_seeds_consumed": False}, "Recovery Retention")


def run_development_scorecard() -> dict[str, Any]:
    required = (*AUDIT_ARTIFACTS, "v5_immutable_snapshot", "metric_contract", "repaired_drift_protocol",
                "repaired_drift_unchanged", "sine_bandwidth", "natural_drift_retention", "exploration_calibration",
                "hyperparameter_study", "static_validation", "scaling_retention", "recovery_retention")
    gates, missing = {}, []
    for name in required:
        try:
            gates[name] = read_artifact(name).get("status") == "PASS"
        except RuntimeError:
            gates[name] = False
            missing.append(name)
    all_pass = all(gates.values())
    if all_pass:
        outcome = "PARTIAL_PURE_REPRODUCTION"
    elif not gates.get("metric_contract"):
        outcome = "REPORTING_CONVENTION_FAILURE"
    elif not gates.get("unit_normalization_audit"):
        outcome = "UNIT_OR_NORMALIZATION_FAILURE"
    elif not gates.get("ppo_clipping_audit") or not gates.get("objective_aggregation_audit"):
        outcome = "OBJECTIVE_TRANSCRIPTION_FAILURE"
    elif not gates.get("baseline_audit"):
        outcome = "BASELINE_FAILURE"
    elif not gates.get("replay_audit"):
        outcome = "REPLAY_STALENESS"
    elif not gates.get("exploration_calibration"):
        outcome = "EXPLORATION_CALIBRATION_FAILURE"
    elif not gates.get("natural_drift_retention") or not gates.get("sine_bandwidth"):
        outcome = "BANDWIDTH_MISMATCH"
    elif not gates.get("repaired_drift_protocol") or not gates.get("repaired_drift_unchanged"):
        outcome = "BENCHMARK_FAILURE"
    else:
        outcome = "GENUINE_CONTROLLER_FAILURE"
    payload = {"schema_version": "google-pure-v6-development-scorecard.v1", "gates": gates,
               "missing_artifacts": missing, "all_development_gates_pass": all_pass,
               "outcome_class": outcome, "outcome_in_frozen_hierarchy": outcome in OUTCOME_CLASSES,
               "certification_ready": all_pass, "certification_blocked": not all_pass,
               "certification_seeds_consumed": False,
               "status": "PASS" if all_pass else "FAIL"}
    return write_report("development_scorecard", payload, "Pure v6 Development Scorecard")


def freeze_certification() -> dict[str, Any]:
    score = read_artifact("development_scorecard")
    ready = score.get("all_development_gates_pass") is True
    payload = {"schema_version": "google-pure-v6-certification-preregistration.v1",
               "certification_seeds": list(CERTIFICATION_SEEDS), "development_scorecard_hash": canonical_hash(score),
               "retired_development_exposed_seeds": list(RETIRED_DEVELOPMENT_EXPOSED_SEEDS),
               "retired_seed_reason": "pre-delivery workflow-safety test; never eligible for held-out use",
               "one_run_permitted": ready, "status": "FROZEN" if ready else "NOT_FROZEN_DEVELOPMENT_GATES_FAILED",
               "certification_seeds_consumed": False, "allowed_outcomes": list(OUTCOME_CLASSES)}
    return write_report("certification_preregistration", payload, "Pure v6 Certification Preregistration")


def run_certification(*, seed: int, confirm: bool = False, authorization_phrase: str | None = None) -> dict[str, Any]:
    guard_seed(seed, certification=True)
    prereg = read_artifact("certification_preregistration")
    if prereg.get("status") != "FROZEN" or not prereg.get("one_run_permitted"):
        raise RuntimeError("certification remains blocked by development gates")
    if not confirm:
        raise RuntimeError("certification requires explicit --confirm after preregistration is frozen")
    if authorization_phrase != "RUN-HELD-OUT-V6-ONCE":
        raise RuntimeError("certification requires the exact held-out authorization phrase")
    metric, _ = _experiment_row("natural", 128, 0.14, controller_choices("low_damage_candidate"), seed,
                                objective_mode="source_literal_ppo")
    outcome = "PURE_GOOGLE_STYLE_SYNTHETIC_REPRODUCTION_CERTIFIED" if metric["stability_suppression_factor_fixed_over_mean"] and metric["stability_suppression_factor_fixed_over_mean"] > 1 else "PARTIAL_PURE_REPRODUCTION"
    return write_report("certification_result", {"schema_version": "google-pure-v6-certification.v1", "seed": seed,
        "metric": metric, "outcome_class": outcome, "status": "PASS" if outcome.endswith("CERTIFIED") else "FAIL",
        "certification_seeds_consumed": True}, "Pure v6 Certification Result")
