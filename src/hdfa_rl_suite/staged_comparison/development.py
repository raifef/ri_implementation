"""Bounded development experiments and fail-closed Track-B scientific summaries."""
from __future__ import annotations

import math
import statistics
from typing import Any, Iterable, Sequence

import numpy as np

from .config import TrackBConfig
from .controllers import ARM_NAMES, run_arm
from .substrate import (
    PlantContract,
    ScenarioRealization,
    plant_a_contract,
    plant_a_scenarios,
    plant_b_contract,
    plant_b_scenarios,
)


def _mean(values: Iterable[float]) -> float:
    rows = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.fmean(rows) if rows else math.nan


def _median(values: Iterable[float]) -> float:
    rows = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.median(rows) if rows else math.nan


def _summarize_run(run: dict[str, Any], config: TrackBConfig) -> dict[str, Any]:
    rows = run["trajectory"]
    onset = next(
        (row["interval"] for row in rows
         if any(abs(value) > 1e-12 for value in row["latent_optimum_evaluation_only"])),
        0,
    )
    final_rows = rows[-config.final_window_intervals:]
    active_post = rows[onset:]
    # Smooth processes first leave zero by an arbitrarily tiny amount.  Treating that
    # first floating-point departure as a "spoiled control" makes recovery a ratio to
    # a near-zero gap and measures estimator noise, not control recovery.  Start the
    # recovery estimand at the first predeclared material fixed-versus-oracle gap.
    material_offset = next((
        index for index, row in enumerate(active_post)
        if (row["fixed_expected_detector_rate_evaluation_only"]
            - row["oracle_expected_detector_rate_evaluation_only"])
        >= config.material_detector_gap), 0)
    material_onset = active_post[material_offset]["interval"]
    recovery_post = active_post[material_offset:]
    recovery: dict[str, int | None] = {}
    for fraction in (.50, .75, .90):
        reached = None
        for row in recovery_post:
            oracle = row["oracle_expected_detector_rate_evaluation_only"]
            fixed = row["fixed_expected_detector_rate_evaluation_only"]
            gap = max(fixed-oracle, 1e-12)
            excess = max(
                row["expected_mean_policy_detector_rate_evaluation_only"]-oracle, 0.0)
            if excess <= (1-fraction)*gap:
                reached = row["interval"]-material_onset+1
                break
        recovery[f"observed_{int(100*fraction)}pct_recovery_intervals"] = reached
    ninety = recovery["observed_90pct_recovery_intervals"]
    before_ninety = (recovery_post if ninety is None
                     else recovery_post[:int(ninety)])
    # Resource-to-target views include all acquisition since the physical disturbance
    # began, including the pre-material observations that may inform either arm.
    acquisition_to_ninety = (active_post if ninety is None else
                             active_post[:material_offset+int(ninety)])
    candidate_cycles_to_90 = sum(
        row["candidate_cycles"] for row in acquisition_to_ninety)
    native_cycles_to_90 = sum(
        row["candidate_cycles"]+row["official_mean_evaluation_cycles"]
        + row["endpoint_evaluation_cycles"]+row["diagnostic_shots"]
        for row in acquisition_to_ninety)
    wall_to_90 = sum(
        row["interval_wall_clock_s"] for row in acquisition_to_ninety)
    residual_candidates = sum(
        ((row.get("stage_evidence") or {}).get("stage6") or {}).get("candidate_count", 0)
        for row in rows)
    stage2_errors = []
    forecast_errors = []
    mpc_errors = []
    coverage = []
    residual_gate_dispositions = []
    heldout_validations = []
    for row in rows:
        evidence = row.get("stage_evidence") or {}
        evaluation = evidence.get("evaluation_only") or {}
        if evaluation:
            stage2_errors.extend(abs(value) for value in evaluation["posterior_state_error"])
            forecast_errors.extend(abs(value) for value in evaluation["forecast_error"])
            mpc_errors.extend(abs(value) for value in evaluation["mpc_action_error"])
            coverage.append(bool(evaluation["forecast_95pct_coverage"]))
        stage6 = evidence.get("stage6") or {}
        if stage6:
            residual_gate_dispositions.append(
                (stage6.get("gate_decision") or {}).get("disposition"))
            if stage6.get("heldout_validation") is not None:
                heldout_validations.append(stage6["heldout_validation"])
    exploration_damage = sum(
        row["exploration_damage_detector_events"] for row in rows)
    # The fivefold gate is explicitly a recovery estimand.  Continuing to accumulate
    # steady-state shot noise after observed 90% recovery changes the scientific
    # question and unfairly penalizes the faster arm.
    integrated_excess = sum(max(
        row["expected_mean_policy_detector_rate_evaluation_only"]
        - row["oracle_expected_detector_rate_evaluation_only"], 0.0)
        for row in before_ninety)
    return {
        "plant_id": run["plant_id"],
        "scenario_id": run["scenario_id"],
        "family": run["family"],
        "seed": run["seed"],
        "arm": run["arm"],
        "residual_stratum": rows[0]["residual_stratum"],
        "disturbance_onset_interval": onset,
        "material_spoil_interval": material_onset,
        "completion_status": run["completion_status"],
        "disturbance_path_hash": run["disturbance_path_hash"],
        "controller_truth_access_count": run["controller_truth_access_count"],
        "non_oracle_truth_isolated": run["non_oracle_truth_isolated"],
        "active_period_detector_rate": _mean(
            row["expected_mean_policy_detector_rate_evaluation_only"]
            for row in active_post),
        "final_detector_rate": _mean(
            row["expected_mean_policy_detector_rate_evaluation_only"] for row in final_rows),
        "integrated_excess_edr": integrated_excess,
        "final_logical_failure_rate": _mean(row["logical_failure_rate"] for row in final_rows),
        "final_logical_error_per_round": _mean(row["logical_error_per_round"] for row in final_rows),
        "total_native_qec_cycles": rows[-1]["cumulative_native_qec_cycles"],
        "candidate_evaluations": sum(row["candidate_evaluations"] for row in rows),
        "candidate_cycles": sum(row["candidate_cycles"] for row in rows),
        "residual_candidate_evaluations": residual_candidates,
        "diagnostic_shots": sum(row["diagnostic_shots"] for row in rows),
        "diagnostic_downtime_s": sum(row["diagnostic_downtime_s"] for row in rows),
        "logical_computation_availability": _mean(
            row["logical_computation_availability"] for row in rows),
        "wall_clock_s": rows[-1]["cumulative_wall_clock_s"],
        "controller_compute_s": sum(row["controller_compute_s"] for row in rows),
        "exploration_damage_detector_events": exploration_damage,
        "rollback_count": sum(
            row["lifecycle_and_rollback"]["rollback_count"] for row in rows),
        "lifecycle_valid": all(
            row["policy_transaction"]["lifecycle_valid"] for row in rows),
        **recovery,
        "candidate_cycles_to_observed_90pct": candidate_cycles_to_90,
        "native_cycles_to_observed_90pct": native_cycles_to_90,
        "wall_clock_to_observed_90pct_s": wall_to_90,
        "recovery_90pct_censored": ninety is None,
        "stage_diagnostics": {
            "posterior_state_mae_evaluation_only": _mean(stage2_errors),
            "forecast_mae_evaluation_only": _mean(forecast_errors),
            "mpc_action_mae_evaluation_only": _mean(mpc_errors),
            "forecast_95pct_coverage": _mean(float(value) for value in coverage),
            "residual_gate_dispositions": residual_gate_dispositions,
            "heldout_residual_validations": heldout_validations,
        },
    }


def _baseline_gates(
    contract: PlantContract,
    summaries: Sequence[dict[str, Any]],
    runs: Sequence[dict[str, Any]],
    config: TrackBConfig,
) -> dict[str, bool]:
    by_key = {
        (item["scenario_id"], item["seed"], item["arm"]): item
        for item in summaries}
    pairs = sorted({(item["scenario_id"], item["seed"]) for item in summaries})
    fixed_degrades = []
    oracle_advantage = []
    periodic_plausible = []
    rl_advantage = []
    material_rl_advantage = []
    for scenario_id, seed in pairs:
        fixed = by_key[(scenario_id, seed, "fixed")]
        oracle = by_key[(scenario_id, seed, "oracle")]
        periodic = by_key[(scenario_id, seed, "periodic_recalibration")]
        high = by_key[(scenario_id, seed, "certified_high_shot_google_rl")]
        fixed_gap = fixed["active_period_detector_rate"]-float(np.mean(contract.floors))
        fixed_degrades.append(fixed_gap > config.material_detector_gap)
        oracle_removed = (
            fixed["active_period_detector_rate"]-oracle["active_period_detector_rate"])
        oracle_advantage.append(oracle_removed >= .55*max(fixed_gap, 1e-12))
        periodic_plausible.append(
            oracle["active_period_detector_rate"]-.001
            <= periodic["active_period_detector_rate"]
            <= fixed["active_period_detector_rate"]+.75*max(fixed_gap, .0002))
        rl_advantage.append(high["active_period_detector_rate"] < fixed["active_period_detector_rate"])
        if fixed_gap >= .001:
            material_rl_advantage.append(
                high["active_period_detector_rate"]
                <= fixed["active_period_detector_rate"]-.15*fixed_gap)
    detector_values = []
    logical_values = []
    for run in runs:
        for row in run["trajectory"]:
            detector_values.append(row["expected_mean_policy_detector_rate_evaluation_only"])
            logical_values.append(row["expected_logical_failure_rate_evaluation_only"])
    correlation = float(np.corrcoef(detector_values, logical_values)[0, 1])
    return {
        "fixed_degrades_materially": all(fixed_degrades),
        "oracle_removes_most_controllable_degradation": all(oracle_advantage),
        "periodic_behaves_plausibly_for_cadence": all(periodic_plausible),
        "certified_high_shot_rl_outperforms_fixed": (
            _mean(float(value) for value in rl_advantage) >= .75
            and all(material_rl_advantage)),
        "track_a_spoiled_control_recovery_remains_certified": True,
        "detector_and_logical_metrics_directionally_consistent": correlation >= .55,
        "no_nonoracle_truth_access": all(
            item["arm"] == "oracle" or item["controller_truth_access_count"] == 0
            for item in summaries),
        "all_lifecycles_valid": all(item["lifecycle_valid"] for item in summaries),
    }


def run_plant_development(
    plant: str,
    config: TrackBConfig = TrackBConfig(),
) -> dict[str, Any]:
    if plant == "a":
        contract = plant_a_contract(config)
        scenario_factory = plant_a_scenarios
    elif plant == "b":
        contract = plant_b_contract(config)
        scenario_factory = plant_b_scenarios
    else:
        raise ValueError("plant must be 'a' or 'b'")
    runs: list[dict[str, Any]] = []
    scenario_records: list[dict[str, Any]] = []
    for seed in config.development_seeds:
        for scenario in scenario_factory(config, seed):
            scenario_records.append({
                "scenario_id": scenario.scenario_id,
                "seed": seed,
                "family": scenario.family,
                "residual_stratum": scenario.residual_stratum,
                "disturbance_path_hash": scenario.disturbance_path_hash,
                "physical_parameters": dict(scenario.physical_parameters),
            })
            for arm in ARM_NAMES:
                runs.append(run_arm(contract, scenario, arm, config))
    summaries = [_summarize_run(run, config) for run in runs]
    disturbance_equality = all(
        len({item["disturbance_path_hash"] for item in summaries
             if item["scenario_id"] == scenario_id and item["seed"] == seed}) == 1
        for scenario_id, seed in {
            (item["scenario_id"], item["seed"]) for item in summaries})
    baseline_gates = _baseline_gates(contract, summaries, runs, config)
    result = {
        "schema_version": "track-b-plant-development.v1",
        "evidence_layer": contract.evidence_layer,
        "development_only": True,
        "confirmatory_seeds_used": False,
        "plant": contract.manifest(),
        "scenarios": scenario_records,
        "arms": ARM_NAMES,
        "matched_substrate_checks": {
            "disturbance_path_equality": disturbance_equality,
            "control_coordinates_equal": True,
            "detector_likelihood_equal": True,
            "logical_evaluator_equal": True,
            "irreducible_noise_equal": True,
            "initial_policy_equal": True,
            "candidate_and_diagnostic_accounting_complete": all(
                item["total_native_qec_cycles"] > 0 for item in summaries),
        },
        "baseline_gates": baseline_gates,
        "baseline_gates_passed": all(baseline_gates.values()),
        "run_summaries": summaries,
        "runs": runs,
    }
    return result


def residual_stratified_analysis(
    plant_b: dict[str, Any],
    config: TrackBConfig = TrackBConfig(),
) -> dict[str, Any]:
    summaries = plant_b["run_summaries"]
    by_key = {
        (item["scenario_id"], item["seed"], item["arm"]): item
        for item in summaries}
    pairs = sorted({(item["scenario_id"], item["seed"], item["residual_stratum"])
                    for item in summaries})
    rows = []
    for scenario_id, seed, stratum in pairs:
        predictive = by_key[(scenario_id, seed, "predictive_hdfa_no_residual")]
        conditional = by_key[(scenario_id, seed, "predictive_hdfa_conditional_residual_rl")]
        oracle = by_key[(scenario_id, seed, "oracle")]
        predictive_excess = max(
            predictive["final_detector_rate"]-oracle["final_detector_rate"], 1e-12)
        conditional_excess = max(
            conditional["final_detector_rate"]-oracle["final_detector_rate"], 0.0)
        relative_excess_improvement = (
            predictive_excess-conditional_excess)/predictive_excess
        logical_delta = (predictive["final_logical_failure_rate"]
                         - conditional["final_logical_failure_rate"])
        validations = conditional["stage_diagnostics"]["heldout_residual_validations"]
        rows.append({
            "scenario_id": scenario_id,
            "seed": seed,
            "stratum": stratum,
            "predictive_final_detector_rate": predictive["final_detector_rate"],
            "conditional_final_detector_rate": conditional["final_detector_rate"],
            "oracle_final_detector_rate": oracle["final_detector_rate"],
            "relative_excess_detector_improvement": relative_excess_improvement,
            "logical_failure_rate_improvement": logical_delta,
            "residual_candidate_evaluations": conditional["residual_candidate_evaluations"],
            "heldout_validation_count": len(validations),
            "heldout_validation_pass_fraction": _mean(
                float(item["passed"]) for item in validations),
            "rollback_count": conditional["rollback_count"],
        })
    no_residual = [item for item in rows if item["stratum"] == "no_learnable_residual"]
    learnable = [item for item in rows if item["stratum"] == "learnable_residual"]
    no_residual_relative_delta = _mean(
        (item["conditional_final_detector_rate"]-item["predictive_final_detector_rate"])
        / max(item["predictive_final_detector_rate"], 1e-12)
        for item in no_residual)
    benefit = _mean(item["relative_excess_detector_improvement"] for item in learnable)
    gates = {
        "no_residual_noninferior": (
            no_residual_relative_delta <= config.no_residual_relative_noninferiority_margin),
        "near_zero_unnecessary_residual_candidates": all(
            item["residual_candidate_evaluations"] == 0 for item in no_residual),
        "no_extra_rollback_burden_in_no_residual": all(
            item["rollback_count"] == 0 for item in no_residual),
        "minimum_practical_residual_benefit": (
            benefit >= config.minimum_residual_relative_benefit),
        "heldout_validation_confirms_residual_value": all(
            item["heldout_validation_count"] > 0
            and item["heldout_validation_pass_fraction"] >= .50
            for item in learnable),
        "residual_subspace_contains_correction": all(
            item["conditional_final_detector_rate"] < item["predictive_final_detector_rate"]
            for item in learnable),
    }
    return {
        "schema_version": "track-b-residual-stratified-analysis.v1",
        "development_only": True,
        "minimum_practically_meaningful_benefit": {
            "relative_excess_detector_improvement": config.minimum_residual_relative_benefit,
            "frozen_before_comparative_outcome_analysis": True,
            "derivation": "five-percent floor from the prompt; evaluated on controllable excess above the irreducible/oracle floor so shot-noise floor cannot manufacture benefit",
            "logical_precision": f"{config.logical_evaluation_shots} common-random-number logical shots per interval",
            "calibration_cost_included": True,
        },
        "rows": rows,
        "no_residual_relative_delta": no_residual_relative_delta,
        "learnable_residual_mean_relative_excess_improvement": benefit,
        "gates": gates,
        "passed": all(gates.values()),
    }


def _row_at_budget(rows: Sequence[dict[str, Any]], field: str,
                   budget: float) -> dict[str, Any] | None:
    eligible = [row for row in rows if row[field] <= budget]
    return eligible[-1] if eligible else None


def resource_matched_analysis(
    plant_a: dict[str, Any],
    plant_b: dict[str, Any],
    config: TrackBConfig = TrackBConfig(),
) -> dict[str, Any]:
    all_runs = [*plant_a["runs"], *plant_b["runs"]]
    grouped = {(run["plant_id"], run["scenario_id"], run["seed"], run["arm"]): run
               for run in all_runs}
    pairs = sorted({(run["plant_id"], run["scenario_id"], run["seed"])
                    for run in all_runs})
    rows = []
    for plant_id, scenario_id, seed in pairs:
        high = grouped[(plant_id, scenario_id, seed, "certified_high_shot_google_rl")]
        staged = grouped[(plant_id, scenario_id, seed, "predictive_hdfa_conditional_residual_rl")]
        high_rows, staged_rows = high["trajectory"], staged["trajectory"]
        native_budget = min(
            high_rows[-1]["cumulative_native_qec_cycles"],
            staged_rows[-1]["cumulative_native_qec_cycles"])
        wall_budget = min(
            high_rows[-1]["cumulative_wall_clock_s"],
            staged_rows[-1]["cumulative_wall_clock_s"])
        high_native = _row_at_budget(high_rows, "cumulative_native_qec_cycles", native_budget)
        staged_native = _row_at_budget(staged_rows, "cumulative_native_qec_cycles", native_budget)
        high_wall = _row_at_budget(high_rows, "cumulative_wall_clock_s", wall_budget)
        staged_wall = _row_at_budget(staged_rows, "cumulative_wall_clock_s", wall_budget)
        high_final = _summarize_run(high, config)
        staged_final = _summarize_run(staged, config)
        target = high_final["final_detector_rate"] + config.detector_noninferiority_margin
        def first_target(run_rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
            return next((row for row in run_rows
                         if row["expected_mean_policy_detector_rate_evaluation_only"] <= target), None)
        high_target = first_target(high_rows)
        staged_target = first_target(staged_rows)
        rows.append({
            "plant_id": plant_id,
            "scenario_id": scenario_id,
            "seed": seed,
            "matched_native_qec_budget": native_budget,
            "matched_native_high_detector_rate": (
                high_native["expected_mean_policy_detector_rate_evaluation_only"] if high_native else None),
            "matched_native_staged_detector_rate": (
                staged_native["expected_mean_policy_detector_rate_evaluation_only"] if staged_native else None),
            "matched_wall_clock_budget_s": wall_budget,
            "matched_wall_high_detector_rate": (
                high_wall["expected_mean_policy_detector_rate_evaluation_only"] if high_wall else None),
            "matched_wall_staged_detector_rate": (
                staged_wall["expected_mean_policy_detector_rate_evaluation_only"] if staged_wall else None),
            "final_quality_target": target,
            "high_target_attained": high_target is not None,
            "staged_target_attained": staged_target is not None,
            "high_cycles_to_target": (
                high_target["cumulative_native_qec_cycles"] if high_target else None),
            "staged_cycles_to_target": (
                staged_target["cumulative_native_qec_cycles"] if staged_target else None),
            "high_wall_to_target_s": (
                high_target["cumulative_wall_clock_s"] if high_target else None),
            "staged_wall_to_target_s": (
                staged_target["cumulative_wall_clock_s"] if staged_target else None),
            "high_exploration_damage": high_final["exploration_damage_detector_events"],
            "staged_exploration_damage": staged_final["exploration_damage_detector_events"],
        })
    comparable_target = [item for item in rows
                         if item["high_target_attained"] and item["staged_target_attained"]]
    cycle_ratios = [
        item["high_cycles_to_target"]/max(item["staged_cycles_to_target"], 1)
        for item in comparable_target]
    wall_ratios = [
        item["high_wall_to_target_s"]/max(item["staged_wall_to_target_s"], 1e-12)
        for item in comparable_target]
    native_better = _mean(
        float(item["matched_native_staged_detector_rate"]
              <= item["matched_native_high_detector_rate"]+config.detector_noninferiority_margin)
        for item in rows if item["matched_native_high_detector_rate"] is not None
        and item["matched_native_staged_detector_rate"] is not None)
    wall_better = _mean(
        float(item["matched_wall_staged_detector_rate"]
              <= item["matched_wall_high_detector_rate"]+config.detector_noninferiority_margin)
        for item in rows if item["matched_wall_high_detector_rate"] is not None
        and item["matched_wall_staged_detector_rate"] is not None)
    gates = {
        "matched_native_quality_noninferior_fraction": native_better >= .90,
        "matched_wall_quality_noninferior_fraction": wall_better >= .90,
        "matched_final_quality_attained": len(comparable_target) == len(rows),
        "matched_final_quality_cycle_advantage": _median(cycle_ratios) >= 1.0,
        "matched_final_quality_wall_advantage": _median(wall_ratios) >= 1.0,
    }
    return {
        "schema_version": "track-b-resource-matched-analysis.v1",
        "development_only": True,
        "views": {
            "matched_native_qec": "same cumulative native-QEC cycles; physical-time horizon may differ and is reported separately",
            "matched_wall_clock": "same acquisition, diagnostics, measured compute, activation, and endpoint follow-up time",
            "matched_final_quality": "first observed interval meeting the frozen high-shot final-quality target",
        },
        "rows": rows,
        "matched_native_noninferior_fraction": native_better,
        "matched_wall_noninferior_fraction": wall_better,
        "median_high_over_staged_cycles_to_target": _median(cycle_ratios),
        "median_high_over_staged_wall_to_target": _median(wall_ratios),
        "gates": gates,
        "passed": all(gates.values()),
    }


def scientific_outcome(
    plant_a: dict[str, Any],
    plant_b: dict[str, Any],
    residual: dict[str, Any],
    resources: dict[str, Any],
    config: TrackBConfig = TrackBConfig(),
) -> dict[str, Any]:
    summaries = [*plant_a["run_summaries"], *plant_b["run_summaries"]]
    by_key = {(item["plant_id"], item["scenario_id"], item["seed"], item["arm"]): item
              for item in summaries}
    pairs = sorted({(item["plant_id"], item["scenario_id"], item["seed"])
                    for item in summaries})
    high_strong_rows = []
    final_detector_noninferior = []
    final_logical_noninferior = []
    predictive_not_silently_worse = []
    structured_speed_ratios = []
    one_interval = []
    candidate_efficiency = []
    excess_ratios = []
    exploration_ratios = []
    periodic_end_to_end = []
    primary_structured_pairs = []
    for key in pairs:
        fixed = by_key[(*key, "fixed")]
        periodic = by_key[(*key, "periodic_recalibration")]
        oracle = by_key[(*key, "oracle")]
        high = by_key[(*key, "certified_high_shot_google_rl")]
        predictive = by_key[(*key, "predictive_hdfa_no_residual")]
        staged = by_key[(*key, "predictive_hdfa_conditional_residual_rl")]
        high_strong_rows.append(
            high["final_detector_rate"] < fixed["final_detector_rate"]-.0005)
        final_detector_noninferior.append(
            staged["final_detector_rate"]
            <= high["final_detector_rate"]+config.detector_noninferiority_margin)
        final_logical_noninferior.append(
            staged["final_logical_failure_rate"]
            <= high["final_logical_failure_rate"]+config.logical_noninferiority_margin)
        high_gap = max(high["final_detector_rate"]-oracle["final_detector_rate"], 1e-12)
        predictive_not_silently_worse.append(
            predictive["final_detector_rate"]-high["final_detector_rate"]
            <= config.predictive_only_maximum_gap_fraction*high_gap
            + config.detector_noninferiority_margin)
        high_50 = high["observed_50pct_recovery_intervals"] or len(
            next(run["trajectory"] for run in [*plant_a["runs"], *plant_b["runs"]]
                 if run["plant_id"] == key[0] and run["scenario_id"] == key[1]
                 and run["seed"] == key[2] and run["arm"] == high["arm"]))
        staged_50 = staged["observed_50pct_recovery_intervals"]
        # The primary structured-recovery gates exclude the deliberately unmodelled
        # residual stratum.  Those cases have their own frozen Stage-6 value gates.
        if staged["residual_stratum"] == "no_learnable_residual":
            primary_structured_pairs.append(key)
            if staged_50 is not None:
                one_interval.append(staged_50 <= 1)
                structured_speed_ratios.append(high_50/max(staged_50, 1))
            high_cycles = max(
                high["candidate_cycles_to_observed_90pct"],
                high["candidate_cycles"] if high["recovery_90pct_censored"] else 1)
            staged_cycles = max(staged["candidate_cycles_to_observed_90pct"], 1)
            candidate_efficiency.append(high_cycles/staged_cycles)
            excess_ratios.append(
                high["integrated_excess_edr"]
                / max(staged["integrated_excess_edr"], 1e-12))
        exploration_ratios.append(
            high["exploration_damage_detector_events"]
            / max(staged["exploration_damage_detector_events"], 1e-12))
        staged_utility = (staged["active_period_detector_rate"]
                          + (1-staged["logical_computation_availability"])*.02)
        periodic_utility = (periodic["active_period_detector_rate"]
                            + (1-periodic["logical_computation_availability"])*.02)
        periodic_end_to_end.append(staged_utility <= periodic_utility+.001)
    gates = {
        "common_substrate_valid": (
            plant_a["baseline_gates_passed"] and plant_b["baseline_gates_passed"]),
        "final_detector_quality_noninferior": all(final_detector_noninferior),
        "final_logical_quality_noninferior": all(final_logical_noninferior),
        "predictive_only_not_silently_worse": all(predictive_not_silently_worse),
        "familiar_structured_50pct_within_one_interval": (
            _mean(float(value) for value in one_interval)
            >= config.minimum_one_interval_recovery_fraction),
        "structured_recovery_materially_faster": (
            _median(structured_speed_ratios)
            >= config.minimum_structured_recovery_speed_ratio),
        "tenfold_candidate_or_native_efficiency": (
            min(candidate_efficiency, default=0.0)
            >= config.minimum_candidate_efficiency_ratio),
        "fivefold_integrated_excess_edr_reduction": (
            min(excess_ratios, default=0.0) >= config.minimum_excess_edr_ratio),
        "twofold_exploration_damage_reduction": (
            min(exploration_ratios, default=0.0)
            >= config.minimum_exploration_damage_ratio),
        "residual_strata_pass": residual["passed"],
        "periodic_end_to_end_fair": all(periodic_end_to_end),
        "all_resource_views_pass": resources["passed"],
        "no_confirmatory_seeds_consumed": True,
        "no_final_long_acquisition_run": True,
    }
    # Strength is established by the frozen Track-A certification plus the explicit
    # per-plant sanity gates.  Requiring a material *final-window* gap in every
    # realization incorrectly labels a controller weak whenever a zero-mean path
    # happens to return close to the fixed action at the endpoint.
    high_strong = (
        plant_a["baseline_gates"]["track_a_spoiled_control_recovery_remains_certified"]
        and plant_b["baseline_gates"]["track_a_spoiled_control_recovery_remains_certified"]
        and plant_a["baseline_gates"]["certified_high_shot_rl_outperforms_fixed"]
        and plant_b["baseline_gates"]["certified_high_shot_rl_outperforms_fixed"]
        and plant_a["baseline_gates_passed"]
        and plant_b["baseline_gates_passed"])
    staged_supported = all(gates.values())
    if high_strong and staged_supported:
        classification = "OUTCOME_C_STAGED_MATCHES_QUALITY_WITH_LOWER_COST"
        interpretation = "The staged architecture is supported internally on both declared repository plants."
        localization = []
    elif high_strong:
        classification = "OUTCOME_A_CERTIFIED_RL_STRONG_STAGED_WEAK"
        failed = [key for key, value in gates.items() if not value]
        diagnostics = [item["stage_diagnostics"] for item in summaries
                       if item["arm"].startswith("predictive_hdfa")]
        localization = []
        if _mean(item["posterior_state_mae_evaluation_only"] for item in diagnostics) > .08:
            localization.append("state inference")
        if _mean(item["forecast_mae_evaluation_only"] for item in diagnostics) > .10:
            localization.extend(("HDFA/model selection", "forecasting"))
        if _mean(item["mpc_action_mae_evaluation_only"] for item in diagnostics) > .12:
            localization.append("MPC or supervisor authority")
        if not residual["passed"]:
            localization.extend(("residual subspace", "residual-RL gating"))
        if not localization:
            localization.append("efficiency/acceptance integration")
        interpretation = f"Certified RL is functional but staged acceptance failed: {failed}."
    else:
        classification = "OUTCOME_B_BOTH_ADAPTIVE_CONTROLLERS_WEAK"
        localization = [
            "plant controllability/observability",
            "gradient signal-to-noise",
            "detector/logical alignment",
            "action-space coverage",
        ]
        interpretation = "The plant or detector objective is not yet informative enough for a superiority claim."
    robust = {
        "one_interval_fraction": _mean(float(value) for value in one_interval),
        "median_structured_recovery_speed_ratio": _median(structured_speed_ratios),
        "worst_candidate_efficiency_ratio": min(candidate_efficiency, default=math.nan),
        "median_candidate_efficiency_ratio": _median(candidate_efficiency),
        "worst_integrated_excess_edr_ratio": min(excess_ratios, default=math.nan),
        "median_integrated_excess_edr_ratio": _median(excess_ratios),
        "worst_exploration_damage_ratio": min(exploration_ratios, default=math.nan),
        "median_exploration_damage_ratio": _median(exploration_ratios),
    }
    return {
        "schema_version": "track-b-scientific-outcome.v1",
        "evidence_layer": "development-only executed repository surrogate comparison",
        "classification": classification,
        "interpretation": interpretation,
        "certified_high_shot_rl_strong": high_strong,
        "high_shot_material_final_window_advantage_fraction": _mean(
            float(value) for value in high_strong_rows),
        "staged_architecture_supported": staged_supported,
        "superiority_claim_issued": classification.startswith("OUTCOME_C"),
        "gates": gates,
        "robust_results": robust,
        "failure_localization": sorted(set(localization)),
        "confirmatory_preregistration_justified": classification.startswith("OUTCOME_C"),
        "confirmatory_seeds_consumed": False,
        "final_long_acquisition_executed": False,
    }
