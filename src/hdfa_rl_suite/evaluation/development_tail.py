"""Development-only recovery-tail profiling and causal ablations.

Seeds 101--105 and the five v1 scenarios are permanently excluded from confirmation.
This runner retains censoring and safety failures and never emits authoritative evidence.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
import csv
import json
from pathlib import Path
import statistics
from typing import Mapping, Sequence

from hdfa_rl_suite.baselines.controllers import PredictiveHDFARLArm
from hdfa_rl_suite.common import deterministic_hash
from hdfa_rl_suite.product import ProductLoopConfig

from .benchmark import BenchmarkRunner
from .launch import load_launch_definition


DEVELOPMENT_LAUNCH = "experiments/physical_validation/authoritative-comparison-v1.json"
TAIL_SCENARIOS = frozenset({"ou_step", "nested_common"})


@dataclass(frozen=True)
class TailRunSpec:
    scenario_id: str
    seed: int
    variant: str = "repaired_production"


def _factory(variant: str, runner: BenchmarkRunner):
    bootstrap = runner._bootstrap_config()
    cycles = runner.config.candidate_cycles
    if variant == "repaired_production":
        return runner.arm_factories["predictive_hdfa_residual_rl"]
    if variant == "global_reentry_only":
        controller = ProductLoopConfig(regional_max_fraction=.01)
        return lambda seed: PredictiveHDFARLArm(
            seed=seed, residual=True, candidate_count=4,
            candidate_cycles=cycles, bootstrap_config=bootstrap,
            product_config=controller)
    if variant == "no_residual_rl":
        return lambda seed: PredictiveHDFARLArm(
            seed=seed, residual=False, candidate_count=4,
            candidate_cycles=cycles, bootstrap_config=bootstrap)
    if variant == "eight_residual_candidates":
        return lambda seed: PredictiveHDFARLArm(
            seed=seed, residual=True, candidate_count=8,
            candidate_cycles=cycles, bootstrap_config=bootstrap)
    raise ValueError(f"unknown development ablation {variant!r}")


def _run_one(spec: TailRunSpec, launch_path: str) -> dict:
    definition = load_launch_definition(launch_path)
    config = replace(
        definition.config, authoritative=False, logical_shots_per_interval=8)
    scenario = next(item for item in definition.scenarios()
                    if item.scenario_id == spec.scenario_id)
    runner = BenchmarkRunner(config, (scenario,))
    prepared = runner._prepare_matched_state(scenario, spec.seed)
    metric, trajectories, baseline = runner._run_arm(
        scenario, spec.seed, f"development:{spec.variant}",
        _factory(spec.variant, runner), prepared)
    endpoints = {str(round(item.target_fraction, 2)): asdict(item)
                 for item in metric.recovery_endpoints}
    stage_compute: dict[str, float] = {}
    online_compute = host_wall = 0.0
    for row in trajectories:
        if row.timing is None:
            continue
        online_compute += row.timing.online_compute_critical_s
        host_wall += row.timing.total_observed_host_wall_s
        for stage, value in row.timing.stage_compute_s.items():
            stage_compute[stage] = stage_compute.get(stage, 0.0)+value
    worst_values = [value for value in metric.worst_region_recovery_cycles.values()
                    if value is not None]
    return {
        "schema_version": "development-tail-run.v1",
        "scenario_id": spec.scenario_id, "seed": spec.seed,
        "variant": spec.variant,
        "completion_status": metric.completion_status,
        "censoring_reason": metric.censoring_reason,
        "recovery_endpoints": endpoints,
        "worst_region_recovery_cycles": (
            max(worst_values) if worst_values else None),
        "integrated_excess_edr_events": metric.integrated_excess_detector_events,
        "bootstrap_count": metric.bootstrap_count,
        "recovery_count": metric.recovery_count,
        "rollback_count": metric.rollback_count,
        "lifecycle_violation_count": metric.lifecycle_violation_count,
        "physical_rollback_failure_count": metric.physical_rollback_failure_count,
        "exploration_damage": metric.exploration_damage,
        "online_compute_critical_s": online_compute,
        "observed_host_control_wall_s": host_wall,
        "stage_compute_s": stage_compute,
        "baseline_observation_hash": baseline.observation_hash,
        "disturbance_realization_id": metric.disturbance_realization_id,
        "run_hash": deterministic_hash({
            "metric": metric, "trajectories": tuple(
                (row.interval, row.replay_hash) for row in trajectories),
            "variant": spec.variant}),
    }


def _aggregate(rows: Sequence[Mapping[str, object]]) -> tuple[dict, ...]:
    groups: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["variant"]), str(row["scenario_id"])), []).append(row)
    output = []
    for (variant, scenario), values in sorted(groups.items()):
        e90 = [row["recovery_endpoints"]["0.9"] for row in values]
        interval_values = [
            int(item["intervals_after_peak"])
            if item["status"] == "reached" and item["intervals_after_peak"] is not None
            else int(next(row for row in values
                          if row["recovery_endpoints"]["0.9"] is item)
                     .get("completed_intervals", 32))
            for item in e90]
        output.append({
            "variant": variant, "scenario_id": scenario,
            "run_count": len(values),
            "completion_fraction": sum(row["completion_status"] == "completed"
                                       for row in values)/len(values),
            "observed_90pct_fraction": sum(item["status"] == "reached"
                                           for item in e90)/len(e90),
            "median_90pct_intervals_or_censor": statistics.median(interval_values),
            "maximum_90pct_intervals_or_censor": max(interval_values),
            "mean_excess_edr_events": statistics.fmean(
                float(row["integrated_excess_edr_events"]) for row in values),
            "mean_exploration_damage": statistics.fmean(
                float(row["exploration_damage"]) for row in values),
            "mean_online_compute_s": statistics.fmean(
                float(row["online_compute_critical_s"]) for row in values),
            "lifecycle_or_physical_failures": sum(
                int(row["lifecycle_violation_count"])
                + int(row["physical_rollback_failure_count"]) for row in values),
        })
    return tuple(output)


def run_development_tail(output_dir: str | Path, *,
                         launch_path: str = DEVELOPMENT_LAUNCH,
                         workers: int = 1,
                         include_ablations: bool = True) -> tuple[Path, Path, Path]:
    definition = load_launch_definition(launch_path)
    specs = [TailRunSpec(scenario.scenario_id, seed)
             for scenario in definition.scenarios() for seed in definition.config.seeds]
    if include_ablations:
        for variant in ("global_reentry_only", "no_residual_rl",
                        "eight_residual_candidates"):
            specs.extend(TailRunSpec(scenario.scenario_id, seed, variant)
                         for scenario in definition.scenarios()
                         if scenario.scenario_id in TAIL_SCENARIOS
                         for seed in definition.config.seeds)
    rows: list[dict] = []
    if workers <= 1:
        rows = [_run_one(spec, launch_path) for spec in specs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_one, spec, launch_path): spec for spec in specs}
            for future in as_completed(futures):
                rows.append(future.result())
    rows.sort(key=lambda row: (
        str(row["variant"]), str(row["scenario_id"]), int(row["seed"])))
    aggregate = _aggregate(rows)
    payload = {
        "schema_version": "development-recovery-tail.v1",
        "evidence_role": "development-only; seeds/scenarios excluded from confirmation",
        "launch_path": launch_path, "workers": workers,
        "variants": ("repaired_production", "global_reentry_only",
                     "no_residual_rl", "eight_residual_candidates")
                    if include_ablations else ("repaired_production",),
        "rows": rows, "ablation_summary": aggregate,
        "retained_controller": "repaired_production",
        "retention_rule": (
            "No ablation may replace production unless it improves OU/nested tail "
            "without any lifecycle/physical failure or worse EDR/exploration; "
            "held-out confirmation cannot alter this choice."),
    }
    payload["artifact_hash"] = deterministic_hash(payload)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path, csv_path, md_path = (
        target/"development-tail.json", target/"development-tail.csv",
        target/"development-tail.md")
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    columns = (
        "variant", "scenario_id", "run_count", "completion_fraction",
        "observed_90pct_fraction", "median_90pct_intervals_or_censor",
        "maximum_90pct_intervals_or_censor", "mean_excess_edr_events",
        "mean_exploration_damage", "mean_online_compute_s",
        "lifecycle_or_physical_failures")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(aggregate)
    lines = [
        "# Development recovery-tail and ablation report", "",
        "This is development evidence only. Censored runs remain in every summary; "
        "seeds 101--105 and these five scenarios are excluded from confirmation.", "",
        "| Variant | Scenario | Complete | Observed 90% | Median intervals/censor | Max intervals/censor | Safety failures |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['variant']} | {row['scenario_id']} | "
            f"{row['completion_fraction']:.2f} | {row['observed_90pct_fraction']:.2f} | "
            f"{row['median_90pct_intervals_or_censor']} | "
            f"{row['maximum_90pct_intervals_or_censor']} | "
            f"{row['lifecycle_or_physical_failures']} |")
    lines.extend(("", "Retained controller: `repaired_production`. The JSON artifact "
                  "contains 50%, 75%, 90%, worst-region, EDR, re-entry, rollback, "
                  "exploration and stage-timing fields for every run."))
    md_path.write_text("\n".join(lines)+"\n", encoding="utf-8")
    return json_path, csv_path, md_path
