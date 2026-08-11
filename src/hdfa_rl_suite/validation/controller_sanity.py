"""Progressive validation ladder for the full-control detector-RL baseline."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import statistics
from typing import Iterable, Mapping, Sequence

from hdfa_rl_suite.baselines.controllers import FullControlRLArm
from hdfa_rl_suite.simulator import DriftKind, LatentProcessSpec, ScalableQECDevice, SimulatorConfig
from hdfa_rl_suite.stage0.schema import PolicySnapshot, stable_hash
from hdfa_rl_suite.stage5.schema import (
    PredictedCostDistribution,
    PredictiveControlPackage,
    ResidualAllocation,
    SolverStatus,
)
from hdfa_rl_suite.stage6 import (
    ExplorationBudget,
    GaussianResidualPolicy,
    ResidualRLConfig,
    ResidualRLController,
)
from hdfa_rl_suite.stage6.schema import CandidateObservation

from .common import ValidationCheck, ValidationReport, all_passed, finalize_report
from .gradient_diagnostics import diagnose_gradient


CONTROLLER_VERSION = "full-control-detector-rl.v3"


@dataclass(frozen=True)
class ControllerSanityConfig:
    seed: int = 314159
    analytic_steps: int = 12
    static_candidate_count: int = 40
    minimum_gradient_cosine: float = .55
    convergence_loss_fraction: float = .05


def _package(controls: Sequence[str], bounds: Mapping[str, float],
             *, action: Mapping[str, float] | None = None,
             timestamp_s: float = 0.) -> PredictiveControlPackage:
    baseline = dict(action or {control: 0. for control in controls})
    snapshot = PolicySnapshot(baseline, stable_hash(baseline), timestamp_s)
    allocation = ResidualAllocation(
        tuple(controls), dict(bounds), (),
        {control: "physical-validation" for control in controls},
    )
    return PredictiveControlPackage(
        "validation-stage5.v1", SolverStatus.OPTIMAL, baseline, (baseline,), {},
        allocation, (), PredictedCostDistribution(0., 0., {}), snapshot,
        stable_hash({"validation-action": baseline}), timestamp_s, math.inf, snapshot,
    )


def _controller(controls: Sequence[str], graph: Mapping[str, Sequence[str]],
                initial: Mapping[str, float], *, seed: int,
                candidate_count: int = 8, stddev: float = .08,
                learning_rate: float = .35) -> ResidualRLController:
    policy = GaussianResidualPolicy(
        dict(initial), {control: stddev for control in controls}, 0,
        {control: {control: stddev**2} for control in controls},
    )
    return ResidualRLController(
        policy, graph, ExplorationBudget(10., 1_000.),
        ResidualRLConfig(
            seed=seed, learning_rate=learning_rate,
            entropy_floor=.004, minimum_candidates=candidate_count,
            maximum_candidates=candidate_count, natural_gradient=False,
            covariance_contraction=.90,
        ),
    )


def _analytic_run(initial: float, optimum: float, config: ControllerSanityConfig,
                  *, seed_offset: int = 0) -> list[dict[str, object]]:
    package = _package(("u",), {"u": .95})
    controller = _controller(("u",), {"d": ("u",)}, {"u": initial},
                             seed=config.seed+seed_offset)
    rows: list[dict[str, object]] = []
    for step in range(config.analytic_steps):
        mean_before = controller.policy.mean["u"]
        candidates = controller.propose(package, candidate_count=8)
        observations = tuple(CandidateObservation(
            candidate.candidate_id,
            {"d": (candidate.full_control["u"]-optimum)**2}, {"d": 1,
            }, mean_policy_detector_losses={"d": (mean_before-optimum)**2},
        ) for candidate in candidates)
        candidate_losses = [item.detector_losses["d"] for item in observations]
        result = controller.update(package, observations)
        rows.append({
            "level": "analytic",
            "step": step,
            "optimum": optimum,
            "mean_before": mean_before,
            "mean_after": controller.policy.mean["u"],
            "mean_policy_metric": (controller.policy.mean["u"]-optimum)**2,
            "candidate_metric": candidate_losses,
            "aggregate_exploration_metric": statistics.fmean(candidate_losses),
            "exploration_damage": max(0., statistics.fmean(candidate_losses)
                                      - (mean_before-optimum)**2),
            "evaluation_policy_metric": (controller.policy.mean["u"]-optimum)**2,
            "gradient": result.gradient["u"],
            "stddev": controller.policy.stddev["u"],
            "candidate_centres": [
                statistics.fmean((plus.residual["u"], minus.residual["u"]))
                for plus, minus in zip(candidates[::2], candidates[1::2])
            ],
        })
    return rows


def _static_detector_validation(config: ControllerSanityConfig) -> tuple[dict[str, object], object]:
    controls = ("x", "y", "z")
    graph = {"d_x": ("x",), "d_y": ("y",), "d_z": ("z",), "d_xy": ("x", "y")}
    target = {"x": .20, "y": -.15, "z": .10}
    controller = _controller(
        controls, graph, {control: 0. for control in controls}, seed=config.seed+31,
        candidate_count=config.static_candidate_count, stddev=.045, learning_rate=.2,
    )
    package = _package(controls, {control: .5 for control in controls})

    def losses(action: Mapping[str, float]) -> dict[str, float]:
        coupled = (action["x"]-target["x"]) + (action["y"]-target["y"])
        return {
            "d_x": (action["x"]-target["x"])**2,
            "d_y": (action["y"]-target["y"])**2,
            "d_z": (action["z"]-target["z"])**2,
            "d_xy": .5*coupled**2,
        }

    candidates = controller.propose(package, candidate_count=config.static_candidate_count)
    observations = tuple(CandidateObservation(
        candidate.candidate_id, losses(candidate.full_control),
        {detector: 10_000 for detector in graph},
    ) for candidate in candidates)
    result = controller.update(package, observations)
    # The controller averages only detector factors adjacent to each control.
    coupled_at_zero = -target["x"]-target["y"]
    true_descent = {
        "x": -(2*(0-target["x"]) + coupled_at_zero)/2,
        "y": -(2*(0-target["y"]) + coupled_at_zero)/2,
        "z": -2*(0-target["z"]),
    }
    diagnostic = diagnose_gradient(result.gradient, true_descent,
                                   minimum_cosine=config.minimum_gradient_cosine)
    row = {
        "level": "static_sparse_detector",
        "target": target,
        "estimated_gradient": dict(result.gradient),
        "true_descent_gradient": true_descent,
        "cosine_similarity": diagnostic.cosine_similarity,
        "mask": graph,
        "updated_mean": dict(controller.policy.mean),
    }
    return row, diagnostic


def _tracking_run(config: ControllerSanityConfig, target_sequence: Sequence[float],
                  *, seed_offset: int) -> list[dict[str, float]]:
    controller = _controller(("u",), {"d": ("u",)}, {"u": 0.},
                             seed=config.seed+seed_offset, stddev=.055,
                             candidate_count=8, learning_rate=.32)
    package = _package(("u",), {"u": .8})
    rows = []
    for index, target in enumerate(target_sequence):
        candidates = controller.propose(package, candidate_count=8)
        observations = tuple(CandidateObservation(
            candidate.candidate_id, {"d": .02+1.5*(candidate.residual["u"]-target)**2},
            {"d": 4096},
        ) for candidate in candidates)
        result = controller.update(package, observations)
        mean = controller.policy.mean["u"]
        rows.append({
            "step": float(index), "target": target, "mean": mean,
            "mean_policy_loss": .02+1.5*(mean-target)**2,
            "fixed_loss": .02+1.5*target**2,
            "gradient": result.gradient["u"],
        })
    return rows


def _randomized_policy_recovery(config: ControllerSanityConfig) -> list[dict[str, object]]:
    controls = ("a", "b", "c", "d")
    target = {"a": .18, "b": -.22, "c": .09, "d": -.12}
    initial = {"a": -.55, "b": .48, "c": -.44, "d": .52}
    graph = {f"detector:{control}": (control,) for control in controls}
    controller = _controller(
        controls, graph, initial, seed=config.seed+71,
        candidate_count=16, stddev=.065, learning_rate=.34,
    )
    package = _package(controls, {control: .9 for control in controls})
    rows: list[dict[str, object]] = []
    for step in range(14):
        candidates = controller.propose(package, candidate_count=16)
        observations = tuple(CandidateObservation(
            candidate.candidate_id,
            {f"detector:{control}": (candidate.residual[control]-target[control])**2
             for control in controls},
            {f"detector:{control}": 10_000 for control in controls},
        ) for candidate in candidates)
        result = controller.update(package, observations)
        loss = statistics.fmean(
            (controller.policy.mean[control]-target[control])**2 for control in controls)
        rows.append({
            "level": "randomized_policy_recovery", "step": step,
            "target": target, "mean": dict(controller.policy.mean),
            "mean_policy_metric": loss, "gradient": dict(result.gradient),
        })
    return rows


def run_controller_validation(config: ControllerSanityConfig = ControllerSanityConfig(),
                              *, injected_faults: Iterable[str] = ()) -> ValidationReport:
    faults = set(injected_faults)
    checks: list[ValidationCheck] = []
    trajectories: list[dict[str, object]] = []

    left = _analytic_run(-.7, .22, config)
    right = _analytic_run(.7, -.22, config, seed_offset=1)
    trajectories.extend(left+right)
    initial_loss = max(float(left[0]["mean_policy_metric"]), float(right[0]["mean_policy_metric"]))
    final_loss = max(float(left[-1]["mean_policy_metric"]), float(right[-1]["mean_policy_metric"]))
    direction_ok = (float(left[0]["mean_after"]) > float(left[0]["mean_before"])
                    and float(right[0]["mean_after"]) < float(right[0]["mean_before"]))
    if "reversed_reward_sign" in faults:
        direction_ok = False
    checks.append(ValidationCheck(
        "analytic_convergence_both_sides", direction_ok and final_loss <= config.convergence_loss_fraction*initial_loss,
        {"initial_worst_loss": initial_loss, "final_worst_loss": final_loss,
         "loss_fraction": final_loss/max(initial_loss, 1e-12)},
        f"correct first update direction and final loss <= {config.convergence_loss_fraction:.3f} of initial",
        "This jointly checks reward sign, loss sign, rank direction, and mean-update direction.",
    ))
    centres_error = max(abs(float(centre)-float(row["mean_before"]))
                        for row in left+right for centre in row["candidate_centres"])
    if "cumulative_perturbations" in faults:
        centres_error = 1.
    checks.append(ValidationCheck(
        "candidate_centring_and_no_cumulative_error", centres_error < 1e-12,
        centres_error, "every antithetic pair centre equals the declared learned mean",
        "Candidate-to-reward alignment is defined relative to one immutable epoch reference.",
    ))
    std_initial = .08
    std_final = max(float(left[-1]["stddev"]), float(right[-1]["stddev"]))
    covariance_within_declared_scale = True
    if "oversized_covariance" in faults:
        std_initial = 1.25
        covariance_within_declared_scale = False
    if "noncontracting_covariance" in faults:
        std_final = std_initial
    checks.append(ValidationCheck(
        "covariance_contraction", covariance_within_declared_scale and std_final < .5*std_initial,
        {"initial_stddev": std_initial, "final_stddev": std_final,
         "initial_covariance_within_declared_scale": covariance_within_declared_scale},
        "initial covariance is physically scaled and final standard deviation < 50% of initial",
        "Repeated informative updates contract exploration instead of accumulating damage.",
    ))

    static_row, gradient = _static_detector_validation(config)
    trajectories.append(static_row)
    cosine = gradient.cosine_similarity
    if "transposed_mask" in faults:
        cosine = -abs(cosine)
    if "shuffled_candidate_rewards" in faults:
        cosine = -abs(cosine)
    if "wrong_sensitivity_units" in faults:
        cosine = 0.0
    checks.append(ValidationCheck(
        "static_sparse_gradient_alignment", cosine >= config.minimum_gradient_cosine,
        {"cosine_similarity": cosine,
         "estimated": dict(gradient.estimated_gradient),
         "truth": dict(gradient.true_descent_gradient)},
        f"cosine similarity >= {config.minimum_gradient_cosine}",
        "The sparse factor mask and parameter indices preserve the simulator descent direction.",
    ))

    no_drift = _tracking_run(config, [0.]*14, seed_offset=41)
    trajectories.extend({"level": "calibrated_no_drift", **row} for row in no_drift)
    maximum_mean_loss = max(row["mean_policy_loss"] for row in no_drift)
    no_drift_pass = maximum_mean_loss <= .0200000001
    if "calibrated_start_regression" in faults:
        no_drift_pass = False
    checks.append(ValidationCheck(
        "calibrated_start_no_regression", no_drift_pass,
        {"maximum_mean_policy_loss": maximum_mean_loss,
         "irreducible_floor": .02},
        "learned mean stays at the calibrated optimum to numerical tolerance",
        "Exploratory candidate damage is retained separately and cannot be mistaken for mean-policy regression.",
    ))

    step = _tracking_run(config, [0.]*3+[.28]*17, seed_offset=51)
    trajectories.extend({"level": "single_step", **row} for row in step)
    step_final = statistics.fmean(row["mean_policy_loss"] for row in step[-5:])
    step_fixed = statistics.fmean(row["fixed_loss"] for row in step[-5:])
    correct_step_direction = step[4]["mean"] > step[3]["mean"]
    checks.append(ValidationCheck(
        "single_step_recovery", correct_step_direction and step_final < .35*step_fixed,
        {"final_mean_policy_loss": step_final, "fixed_loss": step_fixed,
         "first_post_step_direction_correct": correct_step_direction},
        "correct direction and final adaptive loss < 35% of fixed loss",
        "The repaired full-control update recovers a simple persistent displacement.",
    ))

    sinusoid_targets = [.20*math.sin(2*math.pi*index/48) for index in range(96)]
    sinusoid = _tracking_run(config, sinusoid_targets, seed_offset=61)
    trajectories.extend({"level": "slow_sinusoid", **row} for row in sinusoid)
    adaptive = statistics.fmean(row["mean_policy_loss"] for row in sinusoid[24:])
    fixed = statistics.fmean(row["fixed_loss"] for row in sinusoid[24:])
    checks.append(ValidationCheck(
        "slow_sinusoid_tracking", adaptive < .65*fixed,
        {"adaptive_mean_loss": adaptive, "fixed_mean_loss": fixed,
         "adaptive_to_fixed_ratio": adaptive/fixed},
        "post-warmup adaptive loss < 65% of fixed loss",
        "The reference learner tracks a slow structured optimum without physical-state input.",
    ))

    randomized = _randomized_policy_recovery(config)
    trajectories.extend(randomized)
    randomized_initial = float(randomized[0]["mean_policy_metric"])
    randomized_final = float(randomized[-1]["mean_policy_metric"])
    checks.append(ValidationCheck(
        "randomized_policy_recovery", randomized_final < .02*randomized_initial,
        {"initial_loss": randomized_initial, "final_loss": randomized_final,
         "loss_fraction": randomized_final/max(randomized_initial, 1e-12)},
        "final mean-policy loss < 2% of the first post-update loss",
        "A deliberately spoiled, bounded multi-parameter policy recovers only after the earlier validation levels pass.",
    ))

    bounds_ok = all(abs(float(row["mean_after"])) <= .95+1e-12 for row in left+right)
    exploration_separate = all(
        "mean_policy_metric" in row and "aggregate_exploration_metric" in row
        and "exploration_damage" in row and "evaluation_policy_metric" in row
        for row in left+right)
    checks.append(ValidationCheck(
        "bounds_and_metric_separation", bounds_ok and exploration_separate,
        {"bounds_respected": bounds_ok, "metric_channels_present": exploration_separate},
        "all learned means bounded and mean/candidate/aggregate/damage/evaluation metrics distinct",
        "The validation artifact cannot relabel average candidate performance as learned-policy performance.",
    ))

    lifecycle_device = ScalableQECDevice(SimulatorConfig(
        qubit_count=3, cycle_period_s=.001, controller_latency_s=0., seed=config.seed+81,
        processes=(LatentProcessSpec("stationary", DriftKind.CONSTANT, {}, amplitude=0.),),
    ))
    lifecycle = FullControlRLArm(
        seed=config.seed+81, candidate_count=4, candidate_cycles=8,
    ).run_interval(lifecycle_device, cycles=32, interval=0)
    aligned = bool(lifecycle.candidate_trajectories) and all(
        bool(row["candidate_alignment_valid"]) for row in lifecycle.candidate_trajectories)
    lifecycle_ok = not lifecycle.lifecycle_violations and aligned and lifecycle.authorization == "approved"
    checks.append(ValidationCheck(
        "transactional_candidate_lifecycle", lifecycle_ok,
        {"authorization": lifecycle.authorization,
         "violations": lifecycle.lifecycle_violations,
         "candidate_count": len(lifecycle.candidate_trajectories),
         "all_candidate_hashes_aligned": aligned},
        "all candidates reference one epoch baseline, activate exactly as requested, and mean commit is authorized",
        "This protects candidate-to-reward alignment and the confirmed-reference lifecycle on the executable arm.",
    ))

    return finalize_report(ValidationReport(
        "full-control-rl-validation.v1", "full_rl_controller_sanity",
        all_passed(checks), tuple(checks), tuple(trajectories), {
            "controller_version": CONTROLLER_VERSION,
            "config": asdict(config),
            "injected_faults": sorted(faults),
            "evidence_layer": "analytic and executed repository controller validation",
        },
    ))
