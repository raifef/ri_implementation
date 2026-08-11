"""Short held-out baseline cohort run before authoritative acquisition."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import statistics
from typing import Mapping

from hdfa_rl_suite.baselines.controllers import FullControlRLArm
from hdfa_rl_suite.common import deterministic_hash
from hdfa_rl_suite.evaluation.evidence import (
    EvidenceLayer, EvidenceRecord, REQUIRED_DEVELOPMENT_FIGURES,
    validate_report_payload,
)
from hdfa_rl_suite.logical import RotatedSurfaceCodeEvaluator, SurfaceCodeMemoryConfig
from hdfa_rl_suite.simulator import (
    DriftKind, LatentProcessSpec, ScalableQECDevice, SimulatorConfig,
)

from .common import ValidationCheck, ValidationReport, all_passed, finalize_report


@dataclass(frozen=True)
class DevelopmentCohortConfig:
    seeds: tuple[int, ...] = (701,)
    qubit_count: int = 3
    intervals: int = 4
    evaluation_cycles: int = 4096
    candidate_count: int = 40
    candidate_cycles: int = 2048
    cycle_period_s: float = 1e-5
    periodic_cadence: int = 2
    logical_shots: int = 256


def _scenarios() -> Mapping[str, tuple[LatentProcessSpec, ...]]:
    local = {"drive:q0": 1.0, "drive:q1": .35}
    return {
        "no_disturbance": (LatentProcessSpec(
            "stationary", DriftKind.CONSTANT, {}, amplitude=0.0),),
        "step": (LatentProcessSpec(
            "step", DriftKind.STEP, local, amplitude=.25, step_time_s=0.0),),
        "sinusoid": (LatentProcessSpec(
            "sinusoid", DriftKind.SINUSOID, local, amplitude=.24, period_s=8.0),),
        "rtn": (LatentProcessSpec(
            "rtn", DriftKind.RANDOM_TELEGRAPH, local, amplitude=.22,
            rate_hz=.05, mean_dwell_s=20.0),),
    }


def _bounded_target(device: ScalableQECDevice,
                    target: Mapping[str, float]) -> dict[str, float]:
    current = device.confirmed_policy.controls
    output = dict(current)
    for control, requested in target.items():
        bound = device.limits.controls[control]
        output[control] = max(
            bound.minimum, min(bound.maximum,
                max(current[control]-bound.max_slew,
                    min(current[control]+bound.max_slew, requested))))
    return output


def _figure_manifest(output_dir: Path) -> tuple[dict, ...]:
    names = (*REQUIRED_DEVELOPMENT_FIGURES, "convergence_fit_diagnostics")
    return tuple({
        "figure_id": name,
        "path": str((output_dir / f"{index:02d}_{name}.png").as_posix()),
        "evidence_layer": EvidenceLayer.EXECUTED_REPOSITORY_SIMULATION.value,
    } for index, name in enumerate(names, start=1))


def _plots(rows: tuple[dict, ...], figures: tuple[dict, ...]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = {item["figure_id"]: Path(item["path"]) for item in figures}
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    def selected(scenario, arm, seed=701):
        return sorted((row for row in rows if row["scenario"] == scenario
                       and row["arm"] == arm and row["seed"] == seed),
                      key=lambda item: item["interval"])

    plt.figure(figsize=(7.2, 4.2))
    values = selected("step", "fixed")
    x = [item["interval"] for item in values]
    plt.plot(x, [item["latent_optimum"] for item in values], "o-", label="latent optimum")
    plt.plot(x, [item["mismatch"] for item in values], "s--", label="fixed mismatch")
    plt.xlabel("Control interval"); plt.ylabel("Normalized control"); plt.legend(); plt.tight_layout()
    plt.savefig(paths["latent_optimum_and_fixed_mismatch"], dpi=180); plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=True)
    for axis, scenario in zip(axes.flat, _scenarios()):
        for arm, style in (("fixed", "-"), ("periodic", "--"), ("oracle", ":")):
            values = selected(scenario, arm)
            axis.plot([item["interval"] for item in values],
                      [item["mean_policy_detector_rate"] for item in values],
                      style, marker="o", label=arm)
        axis.set_title(scenario); axis.set_xlabel("Interval"); axis.set_ylabel("EDR")
    axes[0, 0].legend(); fig.tight_layout()
    fig.savefig(paths["fixed_periodic_oracle_trajectories"], dpi=180); plt.close(fig)

    plt.figure(figsize=(7.2, 4.2))
    for scenario in _scenarios():
        values = selected(scenario, "full_rl")
        plt.plot([item["interval"] for item in values],
                 [item["mean_policy_detector_rate"] for item in values],
                 marker="o", label=scenario)
    plt.xlabel("Control interval"); plt.ylabel("Held-out mean-policy EDR"); plt.legend(); plt.tight_layout()
    plt.savefig(paths["rl_mean_policy_trajectory"], dpi=180); plt.close()

    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    values = selected("step", "full_rl")
    x = [item["interval"] for item in values]
    axis.plot(x, [item["aggregate_exploration_detector_rate"] for item in values],
              "o-", label="exploratory aggregate EDR")
    axis.plot(x, [item["mean_policy_detector_rate"] for item in values],
              "s--", label="mean-policy EDR")
    second = axis.twinx()
    second.plot(x, [item["exploration_damage"] for item in values],
                "^:", color="tab:red", label="exploration damage")
    axis.set_xlabel("Control interval"); axis.set_ylabel("EDR"); second.set_ylabel("Damage")
    handles, labels = axis.get_legend_handles_labels(); h2, l2 = second.get_legend_handles_labels()
    axis.legend(handles+h2, labels+l2); fig.tight_layout()
    fig.savefig(paths["exploratory_aggregate_and_damage"], dpi=180); plt.close(fig)

    plt.figure(figsize=(7.2, 4.2))
    for arm in ("fixed", "periodic", "oracle", "full_rl"):
        counts = [sum(row["status"] == "completed" for row in rows
                      if row["arm"] == arm and row["interval"] >= interval)
                  for interval in range(max(row["interval"] for row in rows)+1)]
        plt.step(range(len(counts)), counts, where="post", label=arm)
    plt.xlabel("Control interval"); plt.ylabel("Active trajectories"); plt.legend(); plt.tight_layout()
    plt.savefig(paths["active_risk_set_and_censoring"], dpi=180); plt.close()

    plt.figure(figsize=(7.2, 4.2))
    for arm, marker in (("fixed", "o"), ("periodic", "s"), ("oracle", "^"), ("full_rl", "x")):
        values = [row for row in rows if row["arm"] == arm]
        plt.scatter([row["mean_policy_detector_rate"] for row in values],
                    [row["logical_failure_probability"] for row in values],
                    marker=marker, alpha=.7, label=arm)
    plt.xlabel("Detector-event rate"); plt.ylabel("Stim/PyMatching logical failure probability")
    plt.legend(loc="upper left", frameon=True); plt.tight_layout()
    plt.savefig(paths["logical_versus_detector_relation"], dpi=180); plt.close()

    plt.figure(figsize=(7.2, 4.2))
    arms = ("fixed", "periodic", "oracle", "full_rl")
    qec = [statistics.fmean(row["total_native_qec_cycles"] for row in rows if row["arm"] == arm)
           for arm in arms]
    candidate = [statistics.fmean(row["candidate_cycles"] for row in rows if row["arm"] == arm)
                 for arm in arms]
    plt.bar(arms, qec, label="total native-QEC cycles")
    plt.bar(arms, candidate, label="candidate cycles")
    plt.ylabel("Cycles per interval"); plt.xticks(rotation=15); plt.legend(); plt.tight_layout()
    plt.savefig(paths["cycle_candidate_budget"], dpi=180); plt.close()

    plt.figure(figsize=(7.2, 4.2))
    modes = [row["lifecycle_mode"] for row in rows]
    labels = sorted(set(modes))
    counts = [modes.count(label) for label in labels]
    plt.bar(labels, counts)
    plt.ylabel("Interval records"); plt.xticks(rotation=20); plt.title("Re-entry count: 0")
    plt.tight_layout(); plt.savefig(paths["lifecycle_mode_and_reentry_burden"], dpi=180); plt.close()

    plt.figure(figsize=(7.2, 4.2))
    values = selected("step", "full_rl")
    plt.plot([item["interval"] for item in values],
             [item["mean_policy_detector_rate"] for item in values], "o-")
    plt.xlabel("Control interval"); plt.ylabel("Observed mean-policy EDR")
    plt.text(.5, .92, "Observed endpoints only; no extrapolated convergence fit used",
             transform=plt.gca().transAxes, ha="center")
    plt.tight_layout(); plt.savefig(paths["convergence_fit_diagnostics"], dpi=180); plt.close()


def run_development_cohort(config: DevelopmentCohortConfig = DevelopmentCohortConfig(), *,
                           output_dir: str | Path = "artifacts/validation/development-cohort-figures",
                           generate_figures: bool = False) -> ValidationReport:
    scenarios = _scenarios()
    rows: list[dict] = []
    logical = RotatedSurfaceCodeEvaluator(SurfaceCodeMemoryConfig(
        distance=3, rounds=3, shots=config.logical_shots))
    total_epoch_cycles = config.candidate_count*config.candidate_cycles + config.evaluation_cycles
    for scenario_id, processes in scenarios.items():
        for seed in config.seeds:
            template = ScalableQECDevice(SimulatorConfig(
                qubit_count=config.qubit_count, cycle_period_s=config.cycle_period_s,
                controller_latency_s=0.0, disturbance_resolution_s=.005,
                seed=seed, processes=processes))
            devices = {arm: template.clone() for arm in ("fixed", "periodic", "oracle", "full_rl")}
            rl = FullControlRLArm(
                seed=seed, candidate_count=config.candidate_count,
                candidate_cycles=config.candidate_cycles)
            for interval in range(config.intervals):
                for arm, device in devices.items():
                    candidate_evaluations = candidate_cycles = 0
                    exploration_rate = None
                    exploration_damage = 0.0
                    lifecycle_mode = "fixed_confirmed" if arm == "fixed" else arm
                    if arm == "full_rl":
                        result = rl.run_interval(device, config.evaluation_cycles, interval)
                        batch = result.observation
                        candidate_evaluations = result.candidate_evaluations
                        candidate_cycles = result.candidate_cycles
                        exploration_rate = result.aggregate_exploration_detector_rate
                        exploration_damage = result.exploration_damage
                        lifecycle_mode = result.lifecycle_mode
                    else:
                        padding = total_epoch_cycles-config.evaluation_cycles
                        if arm == "periodic" and (interval+1) % config.periodic_cadence == 0:
                            # The characterization shots occupy the final part of the
                            # pre-evaluation wall-time budget; they are not free time and
                            # do not desynchronise the matched realization.
                            characterization_shots = min(256, padding)
                            if padding > characterization_shots:
                                device.advance_elapsed_time(
                                    (padding-characterization_shots)*config.cycle_period_s)
                            estimate = device.characterize_controls(
                                ("drive:q0", "drive:q1"), shots=characterization_shots)
                            device.apply_policy(
                                _bounded_target(device, estimate.estimates),
                                policy_id=f"development-periodic:{interval}",
                                supervisor_authorization="development-cohort:periodic")
                        else:
                            if padding:
                                device.advance_elapsed_time(padding*config.cycle_period_s)
                        if arm == "oracle":
                            # Evaluate the oracle at the same post-adaptation endpoint as
                            # all arms, so the ideal controller is not made artificially
                            # stale during the declared candidate-evaluation interval.
                            optimum = device.oracle_evaluation_view(
                                "oracle:development-cohort").optimum_policy()
                            device.apply_policy(
                                _bounded_target(device, optimum),
                                policy_id=f"development-oracle:{interval}",
                                supervisor_authorization="development-cohort:oracle")
                        batch = device.acquire(config.evaluation_cycles, retain_records=False)
                    diagnostic = device.oracle_evaluation_view(
                        "evaluation:development-cohort").physical_diagnostic()
                    logical_result = logical.evaluate_device(
                        device, seed=int(deterministic_hash(
                            (scenario_id, seed, arm, interval))[:15], 16))
                    rows.append({
                        "scenario": scenario_id, "seed": seed, "arm": arm,
                        "interval": interval, "timestamp_s": device.now_s,
                        "latent_optimum": diagnostic.latent_optimum["drive:q0"],
                        "applied_control": diagnostic.applied_control["drive:q0"],
                        "mismatch": diagnostic.mismatch["drive:q0"],
                        "expected_detector_rate": diagnostic.expected_global_detector_rate,
                        "mean_policy_detector_rate": batch.detector_rate,
                        "aggregate_exploration_detector_rate": exploration_rate,
                        "exploration_damage": exploration_damage,
                        "logical_failure_probability": logical_result.logical_failure_probability,
                        "logical_error_per_round": logical_result.logical_error_per_round,
                        "physical_state_id": batch.physical_state_id,
                        "disturbance_state_id": batch.disturbance_state_id,
                        "policy_hash": batch.policy_activation.policy_hash,
                        "candidate_evaluations": candidate_evaluations,
                        "cycles_per_candidate": config.candidate_cycles if arm == "full_rl" else 0,
                        "candidate_cycles": candidate_cycles,
                        "mean_policy_evaluation_cycles": batch.cycles,
                        "total_native_qec_cycles": candidate_cycles + batch.cycles,
                        "candidate_budget_class": ("reduced-budget-candidate" if arm == "full_rl"
                                                   else "not_applicable"),
                        "lifecycle_mode": lifecycle_mode,
                        "reentry_count": 0, "status": "completed",
                    })

    def tail(scenario, arm):
        values = [row["mean_policy_detector_rate"] for row in rows
                  if row["scenario"] == scenario and row["arm"] == arm
                  and row["interval"] >= config.intervals-2]
        return statistics.fmean(values)

    floor = tail("no_disturbance", "fixed")
    no_drift = abs(tail("no_disturbance", "fixed")-tail("no_disturbance", "oracle")) <= .005
    ordering = {}
    for scenario in ("step", "sinusoid", "rtn"):
        fixed, periodic, oracle = tail(scenario, "fixed"), tail(scenario, "periodic"), tail(scenario, "oracle")
        ordering[scenario] = {"fixed": fixed, "periodic": periodic, "oracle": oracle,
                              "fixed_worsened": fixed > floor+.002,
                              "ordered": oracle <= periodic+.004 and periodic <= fixed+.004}
    rl_direction = {
        scenario: {"full_rl": tail(scenario, "full_rl"), "fixed": tail(scenario, "fixed")}
        for scenario in ("step", "sinusoid")}
    rl_passed = all(value["full_rl"] <= value["fixed"]+.003 for value in rl_direction.values())
    detector_values = [row["expected_detector_rate"] for row in rows]
    logical_values = [row["logical_failure_probability"] for row in rows]
    mean_x, mean_y = statistics.fmean(detector_values), statistics.fmean(logical_values)
    numerator = sum((x-mean_x)*(y-mean_y) for x, y in zip(detector_values, logical_values))
    denominator = (sum((x-mean_x)**2 for x in detector_values)
                   * sum((y-mean_y)**2 for y in logical_values))**.5
    correlation = numerator/denominator if denominator else 0.0
    figures = _figure_manifest(Path(output_dir))
    report_payload = {
        "evidence_records": [asdict(EvidenceRecord(
            "development_cohort", EvidenceLayer.EXECUTED_REPOSITORY_SIMULATION,
            "Short held-out qualitative baseline cohort", "development_validation",
            "hdfa_rl_suite.validation.development_cohort",
            ("seeds excluded from final acceptance", "not a treatment-effect estimate")))],
        "metrics": [], "recovery_summaries": [], "figures": figures,
    }
    report_issues = validate_report_payload(report_payload, require_figures=True)
    checks = (
        ValidationCheck("development_no_disturbance_no_regression", no_drift,
                        {"fixed": tail("no_disturbance", "fixed"),
                         "oracle": tail("no_disturbance", "oracle")},
                        "fixed and oracle remain equivalent without disturbance",
                        "Stationary controls cannot regress through hidden updates."),
        ValidationCheck("development_family_baseline_ordering",
                        all(value["fixed_worsened"] and value["ordered"]
                            for value in ordering.values()), ordering,
                        "for every disturbance family fixed worsens and oracle <= periodic <= fixed",
                        "A failed family blocks release rather than being silently excluded."),
        ValidationCheck("development_full_rl_direction", rl_passed, rl_direction,
                        "validated full RL is no worse than fixed after step/sinusoid adaptation",
                        "This is qualitative development evidence, not final acceptance."),
        ValidationCheck("development_detector_logical_direction", correlation > 0.1,
                        {"pearson_correlation": correlation},
                        "detector and circuit-logical risk are directionally consistent",
                        "The named logical adapter is evaluated from the same endpoint states."),
        ValidationCheck("development_report_figures", not report_issues,
                        tuple(item.code for item in report_issues),
                        "all required figure/evidence contracts are present",
                        "Figures expose trajectories, exploration, risk sets, logical relation, budgets, and lifecycle."),
    )
    if generate_figures:
        _plots(tuple(rows), figures)
    return finalize_report(ValidationReport(
        "development-cohort.v1", "development_cohort", all_passed(checks),
        checks, tuple(rows),
        {"config": asdict(config), "held_out_from_final_seeds": True,
         "scenarios": tuple(scenarios), "figures": figures,
         "report_contract_issues": tuple(asdict(item) for item in report_issues),
         "recommendation": ("launch-after-fresh-manifest" if all_passed(checks)
                            else "block-authoritative-acquisition"),
         "evidence_layer": EvidenceLayer.EXECUTED_REPOSITORY_SIMULATION.value},
    ))
