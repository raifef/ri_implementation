"""Fair-budget fixed, periodic, greedy, RL, staged, and oracle comparison arms."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
import time
from typing import Protocol

from hdfa_rl_suite.common import IntervalTimingRecorder, OnlineTimingBreakdown
from hdfa_rl_suite.pipeline import build_default_loop
from hdfa_rl_suite.product import HDFAProductController, ProductLoopConfig
from hdfa_rl_suite.simulator import QECObservationBatch, ScalableQECDevice
from hdfa_rl_suite.stage0 import ScalableBootstrapConfig
from hdfa_rl_suite.stage0.schema import BootstrapResult, PolicySnapshot, stable_hash
from hdfa_rl_suite.stage5.schema import SolverStatus, bind_policy_lifecycle
from hdfa_rl_suite.stage6 import (ExplorationBudget, FullControlDetectorRL,
                                  bind_candidate_lifecycle)
from hdfa_rl_suite.stage6.schema import CandidateObservation
from hdfa_rl_suite.stage3.sequential import segment_posterior_means
from hdfa_rl_suite.stage7 import SupervisorConfig, SupervisoryController
from hdfa_rl_suite.stage7.schema import Authorization, StageHealth, SupervisorInput


@dataclass(frozen=True)
class ArmIntervalResult:
    arm: str
    observation: QECObservationBatch
    candidate_evaluations: int = 0
    candidate_cycles: int = 0
    diagnostic_shots: int = 0
    diagnostic_downtime_s: float = 0.0
    exploration_damage: float = 0.0
    rollback_count: int = 0
    policy_hash: str = ""
    telemetry_cycles: int = 0
    auxiliary_detector_events: int = 0
    auxiliary_detector_exposures: int = 0
    auxiliary_logical_failures: int = 0
    bootstrap_qec_cycles: int = 0
    lifecycle_mode: str = "not_applicable"
    authorization: str = "not_applicable"
    lifecycle_violations: tuple[str, ...] = ()
    bootstrap_reason: str | None = None
    bootstrap_count: int = 0
    stage_path: tuple[str, ...] = ()
    replay_hash: str = ""
    bootstrap_evidence: dict | None = None
    candidate_trajectories: tuple[dict, ...] = ()
    stage_evidence: dict | None = None
    mean_policy_detector_rate: float | None = None
    aggregate_exploration_detector_rate: float | None = None
    exploration_excess_detector_events: float = 0.0
    evaluation_policy_cycles: int = 0
    candidate_budget_class: str = "not_applicable"
    physical_rollback_failures: tuple[str, ...] = ()
    rollback_outcomes: tuple[dict, ...] = ()
    reentry_request: dict | None = None
    regional_recovery: dict | None = None
    recovery_count: int = 0
    timing: OnlineTimingBreakdown | None = None

    @property
    def total_qec_cycles(self) -> int:
        return self.observation.cycles + self.candidate_cycles + self.telemetry_cycles + self.bootstrap_qec_cycles


class BenchmarkArm(Protocol):
    name: str
    def run_interval(self, device: ScalableQECDevice, cycles: int, interval: int) -> ArmIntervalResult: ...


def _slew_patch(device: ScalableQECDevice, target: dict[str, float]) -> dict[str, float]:
    current = device.confirmed_policy.controls
    output = {}
    for control, value in target.items():
        bound = device.limits.controls[control]
        output[control] = min(bound.maximum, max(bound.minimum,
            min(current[control] + bound.max_slew, max(current[control] - bound.max_slew, value))))
    return output


def _wilson_interval(events: int, exposures: int, z: float = 1.959963984540054
                     ) -> tuple[float, float]:
    """Exposure-aware interval used only for compact diagnostic evidence."""
    if exposures <= 0:
        return 0.0, 1.0
    rate = events / exposures
    denominator = 1 + z*z/exposures
    centre = (rate + z*z/(2*exposures)) / denominator
    radius = z*math.sqrt(rate*(1-rate)/exposures + z*z/(4*exposures**2))/denominator
    return max(0.0, centre-radius), min(1.0, centre+radius)


class FixedCalibrationArm:
    name = "fixed"

    def run_interval(self, device: ScalableQECDevice, cycles: int, interval: int) -> ArmIntervalResult:
        batch = device.acquire(cycles)
        return ArmIntervalResult(
            self.name, batch, policy_hash=device.confirmed_policy.policy_hash,
            mean_policy_detector_rate=batch.detector_rate,
            evaluation_policy_cycles=batch.cycles,
        )


class PeriodicRecalibrationArm:
    name = "periodic_recalibration"

    def __init__(self, period: int = 8, shots: int = 256) -> None:
        self.period, self.shots = period, shots

    def run_interval(self, device: ScalableQECDevice, cycles: int, interval: int) -> ArmIntervalResult:
        shots, downtime = 0, 0.
        if (interval + 1) % self.period == 0:
            characterization = device.characterize_controls(shots=self.shots)
            target = _slew_patch(device, dict(characterization.estimates))
            device.apply_policy(target, policy_id=f"periodic:{interval}")
            shots, downtime = characterization.shots, characterization.downtime_s
        batch = device.acquire(cycles)
        return ArmIntervalResult(
            self.name, batch, diagnostic_shots=shots,
            diagnostic_downtime_s=downtime, policy_hash=device.confirmed_policy.policy_hash,
            mean_policy_detector_rate=batch.detector_rate,
            evaluation_policy_cycles=batch.cycles,
        )


class GreedyCalibrationArm:
    name = "greedy_calibration"

    def __init__(self, shots: int = 96) -> None:
        self.shots = shots
        self._last: QECObservationBatch | None = None

    def run_interval(self, device: ScalableQECDevice, cycles: int, interval: int) -> ArmIntervalResult:
        if self._last and self._last.detector_counts:
            worst = max(self._last.detector_counts, key=lambda detector: self._last.detector_counts[detector][0] / max(1, self._last.detector_counts[detector][1]))
            controls = device.detector_control_graph[worst]
        else:
            controls = (next(iter(device.limits.controls)),)
        characterization = device.characterize_controls(controls, shots=self.shots)
        target = dict(device.confirmed_policy.controls)
        target.update(characterization.estimates)
        device.apply_policy(_slew_patch(device, target), policy_id=f"greedy:{interval}")
        self._last = device.acquire(cycles)
        return ArmIntervalResult(self.name, self._last, diagnostic_shots=characterization.shots,
            diagnostic_downtime_s=characterization.downtime_s, policy_hash=device.confirmed_policy.policy_hash,
            mean_policy_detector_rate=self._last.detector_rate,
            evaluation_policy_cycles=self._last.cycles)


class OracleControlArm:
    name = "oracle"

    def __init__(self) -> None:
        self._view = None

    def run_interval(self, device: ScalableQECDevice, cycles: int, interval: int) -> ArmIntervalResult:
        self._view = self._view or device.oracle_evaluation_view("oracle:upper-bound-controller")
        target = _slew_patch(device, dict(self._view.optimum_policy()))
        device.apply_policy(target, policy_id=f"oracle:{interval}")
        batch = device.acquire(cycles)
        return ArmIntervalResult(
            self.name, batch, policy_hash=device.confirmed_policy.policy_hash,
            mean_policy_detector_rate=batch.detector_rate,
            evaluation_policy_cycles=batch.cycles,
        )


class PhysicalInferenceArm:
    """State-only, sequential-HDFA, and joint-current-state ablation arms."""

    def __init__(self, mode: str, *, seed: int = 0) -> None:
        if mode not in {"state_only", "sequential_hdfa", "joint_hdfa_reactive"}:
            raise ValueError("unknown inference ablation")
        self.mode, self.seed, self.name = mode, seed, mode
        self._loop = None
        self._timeline = []
        self._history: dict[str, list[float]] = {}

    def run_interval(self, device: ScalableQECDevice, cycles: int, interval: int) -> ArmIntervalResult:
        self._loop = self._loop or build_default_loop(device, seed=self.seed)
        batch = device.acquire(cycles)
        if not self._timeline or self._timeline[-1].policy_hash != batch.policy_activation.policy_hash:
            self._timeline.append(batch.policy_activation)
        telemetry = self._loop.telemetry.process(batch.records, tuple(self._timeline), batch.context)
        target = dict(device.confirmed_policy.controls)
        for region_id, stack in self._loop.regions.items():
            view = telemetry.regional_views.get(region_id)
            if view is None:
                continue
            controls = {control: target[control] for control in stack.controls}
            state = stack.inference.infer(view, controls, previous=stack.previous_state)
            stack.previous_state = state
            variable = next(iter(state.mean), None)
            if variable is None:
                continue
            value = state.mean[variable]
            self._history.setdefault(region_id, []).append(value)
            if self.mode == "sequential_hdfa":
                segments = segment_posterior_means(self._history[region_id], minimum_length=4, jump_threshold=2.)
                value = segments[-1].mean
            elif self.mode == "joint_hdfa_reactive":
                value = stack.dynamics.update(view, controls, device.now_s, state_prior=state).current_state_mean[variable]
            primary = next((control for control in stack.controls if control.startswith("drive:")), stack.controls[0])
            target[primary] = value
        device.apply_policy(_slew_patch(device, target), policy_id=f"{self.mode}:{interval}")
        evaluation = device.acquire(cycles)
        return ArmIntervalResult(self.name, evaluation, policy_hash=device.confirmed_policy.policy_hash,
                                 telemetry_cycles=cycles, auxiliary_detector_events=batch.detector_events,
                                 auxiliary_detector_exposures=batch.detector_exposures,
                                 auxiliary_logical_failures=batch.logical_failures,
                                 mean_policy_detector_rate=evaluation.detector_rate,
                                 evaluation_policy_cycles=evaluation.cycles)


class FullControlRLArm:
    name = "full_control_detector_rl"

    def __init__(self, *, seed: int = 0, candidate_count: int = 40, candidate_cycles: int = 32,
                 damage_budget: float = 10., per_candidate_damage_budget: float = .25) -> None:
        self.seed, self.candidate_count, self.candidate_cycles = seed, candidate_count, candidate_cycles
        self.damage_budget, self.per_candidate_damage_budget = damage_budget, per_candidate_damage_budget
        self._controller: FullControlDetectorRL | None = None
        self._supervisor: SupervisoryController | None = None

    def run_interval(self, device: ScalableQECDevice, cycles: int, interval: int) -> ArmIntervalResult:
        timing_recorder = IntervalTimingRecorder()
        actuation_ack_s = 0.0
        compute_started = time.perf_counter_ns()
        if self._controller is None:
            snapshot = PolicySnapshot(dict(device.confirmed_policy.controls), device.confirmed_policy.policy_hash, device.now_s)
            self._controller = FullControlDetectorRL(device.limits, device.detector_control_graph, snapshot,
                ExplorationBudget(self.per_candidate_damage_budget, self.damage_budget), seed=self.seed,
                candidate_count=self.candidate_count, stddev=.04)
            self._supervisor = SupervisoryController(device.limits, config=SupervisorConfig(
                maximum_exploration_damage=self.damage_budget))
        assert self._supervisor is not None
        mode_decision = self._supervisor.tick(SupervisorInput(
            device.now_s, (StageHealth("full-control-rl", True),), forecast_valid=True,
            residual_learning_safe=True, residual_small=False))
        violations: list[str] = []
        candidates = self._controller.propose()
        package = self._controller.proposed_package
        assert package is not None
        reference = device.confirmed_policy
        package = bind_policy_lifecycle(
            package, policy_id=f"full-rl-epoch:{interval}",
            reference_policy_id=reference.policy_id,
            reference_policy_hash=reference.policy_hash,
            created_from_state_id=stable_hash({
                "controller_state_hash": device.controller_state_hash,
                "timestamp_s": device.now_s,
            }),
            controller_state_hash=device.controller_state_hash,
        )
        package = replace(package, supervisor_authorization=mode_decision.decision_id)
        timing_recorder.add_compute(
            "full_control_rl_proposal_and_supervision",
            compute_started, time.perf_counter_ns())
        observations = []
        candidate_trajectories = []
        candidate_cycles = 0
        auxiliary_events = auxiliary_exposures = auxiliary_logical = 0
        for unbound_candidate in candidates:
            baseline_policy_id = f"{package.policy_id}:candidate-reference:{unbound_candidate.candidate_id}"
            try:
                activation_started = device.now_s
                device.apply_policy(
                    package.action, policy_id=baseline_policy_id,
                    supervisor_authorization=package.supervisor_authorization)
                baseline_activation = device.await_policy_acknowledgement()
                actuation_ack_s += device.now_s-activation_started
            except ValueError as error:
                violations.append(f"candidate reference baseline rejected by device: {error}")
                continue
            compute_started = time.perf_counter_ns()
            candidate = bind_candidate_lifecycle(
                unbound_candidate,
                reference_policy_id=baseline_activation.policy_id,
                reference_policy_hash=baseline_activation.policy_hash,
                created_from_state_id=stable_hash({
                    "controller_state_hash": device.controller_state_hash,
                    "timestamp_s": device.now_s,
                }),
                controller_state_hash=device.controller_state_hash,
            )
            decision = self._supervisor.authorize_residual_candidate(
                package, candidate, device.now_s,
                cumulative_damage=self._controller.core.cumulative_damage)
            timing_recorder.add_compute(
                "full_control_rl_candidate_projection_and_authorization",
                compute_started, time.perf_counter_ns())
            if decision.authorization is not Authorization.APPROVED:
                continue
            candidate = replace(candidate, supervisor_authorization=decision.decision_id)
            try:
                # Each reward is generated from the candidate that the estimator thinks
                # it evaluated.  Returning to the epoch reference prevents the prior
                # candidate and slew clipping from corrupting candidate/reward alignment.
                activation_started = device.now_s
                activation = device.apply_policy(
                    candidate.full_control, policy_id=candidate.policy_id,
                    candidate_id=candidate.candidate_id,
                    perturbation=candidate.residual,
                    reference_policy_id=candidate.reference_policy_id,
                    reference_policy_hash=candidate.reference_policy_hash,
                    created_from_state_id=candidate.created_from_state_id,
                    expected_activation_state_id=candidate.expected_activation_state_id,
                    supervisor_authorization=candidate.supervisor_authorization)
                confirmed_candidate = device.await_policy_acknowledgement()
                actuation_ack_s += device.now_s-activation_started
                if (confirmed_candidate.policy_hash != activation.policy_hash
                        or not confirmed_candidate.activation_acknowledgement):
                    raise ValueError("candidate activation acknowledgement/hash mismatch")
            except ValueError as error:
                violations.append(f"authorized candidate rejected by device: {error}")
                continue
            kernel_started = time.perf_counter_ns()
            batch = device.acquire(self.candidate_cycles, retain_records=False)
            timing_recorder.add_host_kernel(
                (time.perf_counter_ns()-kernel_started)/1e9)
            observations.append(CandidateObservation(candidate.candidate_id,
                {detector: events / max(1, exposure) for detector, (events, exposure) in batch.detector_counts.items()},
                {detector: exposure for detector, (_, exposure) in batch.detector_counts.items()},
                logical_risk=batch.logical_failures / batch.cycles, observed_at_s=device.now_s))
            candidate_trajectories.append({
                "candidate": asdict(candidate), "batch_id": batch.batch_id,
                "reference_policy_hash": package.policy_hash,
                "requested_full_control": dict(candidate.full_control),
                "activated_full_control": dict(batch.policy_activation.controls),
                "activated_policy_hash": batch.policy_activation.policy_hash,
                "candidate_alignment_valid": (
                    stable_hash(candidate.full_control) == batch.policy_activation.policy_hash),
                "detector_counts": dict(batch.detector_counts),
                "detector_events": batch.detector_events,
                "detector_exposures": batch.detector_exposures,
                "logical_failure_proxy": batch.logical_failures,
                "observed_at_s": device.now_s,
            })
            candidate_cycles += batch.cycles
            auxiliary_events += batch.detector_events
            auxiliary_exposures += batch.detector_exposures
            auxiliary_logical += batch.logical_failures
        compute_started = time.perf_counter_ns()
        result = self._controller.update(observations)
        timing_recorder.add_compute(
            "full_control_rl_update", compute_started, time.perf_counter_ns())
        # Commit from the immutable epoch reference, not from the final exploratory
        # candidate.  Awaiting the acknowledgement closes the historic pending-MPC /
        # probe race before the mean is projected and validated.
        commit_reference_id = f"{package.policy_id}:mean-reference"
        activation_started = device.now_s
        device.apply_policy(
            package.action, policy_id=commit_reference_id,
            supervisor_authorization=package.supervisor_authorization)
        commit_reference = device.await_policy_acknowledgement()
        actuation_ack_s += device.now_s-activation_started
        compute_started = time.perf_counter_ns()
        target = _slew_patch(device, dict(self._controller.current_policy.values))
        committed = replace(package, status=SolverStatus.OPTIMAL, action=target, trajectory=(target,),
                            policy_hash=stable_hash(target), activation_time_s=device.now_s,
                            controller_acknowledged_hash=None)
        committed = bind_policy_lifecycle(
            committed, policy_id=f"full-rl-mean:{interval}",
            reference_policy_id=commit_reference.policy_id,
            reference_policy_hash=commit_reference.policy_hash,
            created_from_state_id=stable_hash({
                "controller_state_hash": device.controller_state_hash,
                "timestamp_s": device.now_s,
            }),
            controller_state_hash=device.controller_state_hash,
        )
        commit_decision = self._supervisor.authorize_control(committed, device.now_s)
        timing_recorder.add_compute(
            "full_control_rl_mean_projection_and_authorization",
            compute_started, time.perf_counter_ns())
        if commit_decision.authorization is Authorization.APPROVED:
            committed = replace(committed, supervisor_authorization=commit_decision.decision_id)
            try:
                activation_started = device.now_s
                activation = device.apply_policy(
                    target, policy_id=committed.policy_id,
                    reference_policy_id=committed.reference_policy_id,
                    reference_policy_hash=committed.reference_policy_hash,
                    created_from_state_id=committed.created_from_state_id,
                    expected_activation_state_id=committed.expected_activation_state_id,
                    supervisor_authorization=committed.supervisor_authorization)
                confirmed_mean = device.await_policy_acknowledgement()
                actuation_ack_s += device.now_s-activation_started
                if (confirmed_mean.policy_hash != activation.policy_hash
                        or not confirmed_mean.activation_acknowledgement):
                    raise ValueError("mean activation acknowledgement/hash mismatch")
            except ValueError as error:
                violations.append(f"authorized policy mean rejected by device: {error}")
        else:
            violations.append(f"policy mean authorization failed: {commit_decision.reason}")
        kernel_started = time.perf_counter_ns()
        evaluation = device.acquire(cycles, retain_records=False)
        timing_recorder.add_host_kernel(
            (time.perf_counter_ns()-kernel_started)/1e9)
        self._controller.current_policy = PolicySnapshot(
            dict(device.confirmed_policy.controls), device.confirmed_policy.policy_hash, device.now_s)
        exploration_rate = auxiliary_events / auxiliary_exposures if auxiliary_exposures else None
        exploration_excess = (max(0.0, auxiliary_events - evaluation.detector_rate * auxiliary_exposures)
                              if auxiliary_exposures else 0.0)
        budget_class = ("high-shot-reference" if self.candidate_cycles >= 100_000
                        else "reduced-budget-candidate" if self.candidate_cycles >= 2048
                        else "smoke-test-only")
        timing = timing_recorder.finalize(
            qec_acquisition_s=(candidate_cycles+evaluation.cycles)
            * device.config.cycle_period_s,
            diagnostic_downtime_s=0.0,
            actuation_acknowledgement_s=actuation_ack_s,
            complete=True)
        return ArmIntervalResult(self.name, evaluation, len(observations), candidate_cycles,
                                 exploration_damage=result.exploration_damage,
                                 policy_hash=device.confirmed_policy.policy_hash,
                                 auxiliary_detector_events=auxiliary_events,
                                 auxiliary_detector_exposures=auxiliary_exposures,
                                 auxiliary_logical_failures=auxiliary_logical,
                                 lifecycle_mode=mode_decision.mode.value,
                                 authorization=commit_decision.authorization.value,
                                 lifecycle_violations=tuple(violations),
                                 stage_path=("full_control_detector_rl", "stage7:authorization_lifecycle",
                                             "device:atomic_apply", "telemetry:feedback"),
                                 candidate_trajectories=tuple(candidate_trajectories),
                                 stage_evidence={
                                     "mode_decision": asdict(mode_decision),
                                     "commit_decision": asdict(commit_decision),
                                     "rl_result": asdict(result),
                                     "candidate_budget": {
                                         "cycles_per_candidate": self.candidate_cycles,
                                         "candidate_count": self.candidate_count,
                                         "classification": budget_class,
                                     },
                                 },
                                 mean_policy_detector_rate=evaluation.detector_rate,
                                 aggregate_exploration_detector_rate=exploration_rate,
                                 exploration_excess_detector_events=exploration_excess,
                                 evaluation_policy_cycles=evaluation.cycles,
                                 candidate_budget_class=budget_class,
                                 timing=timing)


class PredictiveHDFARLArm:
    """Full staged arm; set residual=False for the predictive-control ablation."""
    name = "predictive_hdfa_residual_rl"

    def __init__(self, *, seed: int = 0, residual: bool = True, candidate_count: int = 4,
                 candidate_cycles: int = 32,
                 bootstrap_config: ScalableBootstrapConfig = ScalableBootstrapConfig(),
                 product_config: ProductLoopConfig | None = None) -> None:
        self.seed, self.residual, self.candidate_count, self.candidate_cycles = seed, residual, candidate_count, candidate_cycles
        self.bootstrap_config = bootstrap_config
        self.product_config = product_config
        self._product: HDFAProductController | None = None

    def _controller_config(self) -> ProductLoopConfig:
        base = self.product_config or ProductLoopConfig()
        return replace(
            base, enable_residual_rl=self.residual,
            residual_candidate_count=self.candidate_count,
            residual_candidate_cycles=self.candidate_cycles,
            bootstrap=self.bootstrap_config)

    def prepare(self, device: ScalableQECDevice, bootstrap: BootstrapResult) -> None:
        """Attach the benchmark's independently executed, common Stage-0 result.

        The benchmark owns the randomized phase boundary.  Reusing the validated
        result here keeps the product path genuine without rerunning calibration after
        the exogenous disturbance has started.
        """
        if self._product is not None:
            raise RuntimeError("predictive benchmark arm has already been prepared")
        self._product = HDFAProductController(
            device, seed=self.seed, config=self._controller_config())
        self._product.accept_validated_bootstrap(bootstrap)

    def run_interval(self, device: ScalableQECDevice, cycles: int, interval: int) -> ArmIntervalResult:
        if self._product is None:
            self._product = HDFAProductController(
                device, seed=self.seed, config=self._controller_config())
        result = self._product.run_interval(cycles, interval=interval)
        causal_is_new = result.newly_acquired_telemetry_cycles > 0
        auxiliary_events = (result.causal_observation.detector_events if causal_is_new else 0)
        auxiliary_exposures = (result.causal_observation.detector_exposures if causal_is_new else 0)
        auxiliary_logical = (result.causal_observation.logical_failures if causal_is_new else 0)
        auxiliary_events += sum(batch.detector_events for batch in result.candidate_batches)
        auxiliary_exposures += sum(batch.detector_exposures for batch in result.candidate_batches)
        auxiliary_logical += sum(batch.logical_failures for batch in result.candidate_batches)
        decision = result.authorization_log[-1]
        candidate_trajectories = tuple({
            "candidate": asdict(candidate),
            "batch_id": batch.batch_id,
            "detector_counts": dict(batch.detector_counts),
            "detector_events": batch.detector_events,
            "detector_exposures": batch.detector_exposures,
            "logical_failure_proxy": batch.logical_failures,
            "observation": asdict(observation),
        } for candidate, batch, observation in zip(
            result.residual_candidates, result.candidate_batches, result.candidate_observations))
        stage_evidence = {
            "control_replay_hash": result.control.replay_hash,
            "causal_observation": {
                "batch_id": result.causal_observation.batch_id,
                "cycles": result.causal_observation.cycles,
                "detector_events": result.causal_observation.detector_events,
                "detector_exposures": result.causal_observation.detector_exposures,
                "detector_rate": result.causal_observation.detector_rate,
                "detector_rate_ci95": _wilson_interval(
                    result.causal_observation.detector_events,
                    result.causal_observation.detector_exposures),
                "policy_id": result.causal_observation.policy_activation.policy_id,
                "policy_hash": result.causal_observation.policy_activation.policy_hash,
                "controller_state_hash": result.causal_observation.controller_state_hash,
            },
            "feedback_observation": {
                "batch_id": result.feedback_observation.batch_id,
                "cycles": result.feedback_observation.cycles,
                "detector_events": result.feedback_observation.detector_events,
                "detector_exposures": result.feedback_observation.detector_exposures,
                "detector_rate": result.feedback_observation.detector_rate,
                "detector_rate_ci95": _wilson_interval(
                    result.feedback_observation.detector_events,
                    result.feedback_observation.detector_exposures),
                "policy_id": result.feedback_observation.policy_activation.policy_id,
                "policy_hash": result.feedback_observation.policy_activation.policy_hash,
                "controller_state_hash": result.feedback_observation.controller_state_hash,
            },
            "regions": {
                region.region_id: {
                    "state_mean": dict(region.state.mean),
                    "state_covariance": region.state.covariance,
                    "state_validity": region.state.validity.value,
                    "state_invalidity_reasons": region.state.invalidity_reasons,
                    "state_ood_score": region.state.ood_score,
                    "state_model_discrepancy": region.state.model_discrepancy,
                    "posterior_predictive": asdict(region.state.posterior_predictive),
                    "observability": asdict(region.state.observability),
                    "attribution": dict(region.state.attribution),
                    "model_probabilities": dict(region.dynamics.model_evidence.model_probabilities),
                    "unknown_model_probability": region.dynamics.unknown_model_probability,
                    "change_alarm": asdict(region.dynamics.change_alarm),
                    "dynamics_invalidity_reasons": region.dynamics.invalidity_reasons,
                    "forecast_risk": {str(horizon): asdict(risk)
                                      for horizon, risk in region.forecast.risk_by_horizon.items()},
                    "forecast_invalidity_reasons": region.forecast.invalidity_reasons,
                    "forecast_validity_horizon_s": region.forecast.validity_horizon_s,
                    "mpc_status": region.control.status.value,
                    "mpc_action": dict(region.control.action),
                    "mpc_active_constraints": region.control.active_constraints,
                    "mpc_policy_id": region.control.policy_id,
                    "mpc_policy_hash": region.control.policy_hash,
                    "mpc_reference_policy_id": region.control.reference_policy_id,
                    "mpc_reference_policy_hash": region.control.reference_policy_hash,
                } for region in result.control.regions
            },
            "authorizations": [asdict(item) for item in result.authorization_log],
            "familiar_fast_paths": dict(result.control.familiar_fast_paths),
            "familiar_processes": {
                region.region_id: (asdict(region.dynamics.familiar_process)
                                   if region.dynamics.familiar_process else None)
                for region in result.control.regions},
            "residual_result": asdict(result.residual_result) if result.residual_result else None,
            "residual_gate_decision": (
                asdict(result.residual_gate_decision)
                if result.residual_gate_decision else None),
            "residual_candidate_count": len(result.residual_candidates),
            "applied_policy_hashes": result.applied_policy_hashes,
            "pending_reentry": (asdict(self._product.pending_reentry_request)
                                if self._product.pending_reentry_request else None),
            "rollback": {
                "requested": any(item.rollback_required for item in result.authorization_log),
                "application_count": result.rollback_count,
                "legacy_physical_validation_failures": tuple(
                    item for item in result.lifecycle_violations
                    if "rollback failed independent telemetry validation" in item),
            },
            "confirmed_policy": asdict(device.confirmed_policy),
            "latest_policy_transactions": tuple(
                asdict(item) for item in device.policy_transaction_log[-8:]),
            "structured_reentry_request": (
                asdict(result.reentry_request) if result.reentry_request else None),
            "regional_recovery": (
                asdict(result.regional_recovery) if result.regional_recovery else None),
            "rollback_outcomes": tuple(asdict(item) for item in result.rollback_outcomes),
            "physical_rollback_failures": result.physical_rollback_failures,
            "compute_timing": asdict(result.timing) if result.timing else None,
        }
        return ArmIntervalResult(
            self.name if self.residual else "predictive_hdfa_no_residual",
            result.feedback_observation,
            len(result.residual_candidates), result.candidate_cycles,
            result.diagnostic_shots, result.diagnostic_downtime_s,
            result.residual_result.exploration_damage if result.residual_result else 0.0,
            result.rollback_count, device.confirmed_policy.policy_hash,
            result.newly_acquired_telemetry_cycles,
            auxiliary_events, auxiliary_exposures, auxiliary_logical,
            result.bootstrap_qec_cycles,
            result.control.supervisor.mode.value,
            decision.authorization.value,
            result.lifecycle_violations,
            result.bootstrap_reason.value if result.bootstrap_reason else None,
            result.bootstrap_count, result.stage_path, result.replay_hash,
            result.bootstrap.to_dict() if result.bootstrap else None,
            candidate_trajectories, stage_evidence,
            result.feedback_observation.detector_rate,
            (auxiliary_events / auxiliary_exposures if auxiliary_exposures else None),
            max(0.0, auxiliary_events
                - result.feedback_observation.detector_rate * auxiliary_exposures),
            result.feedback_observation.cycles,
            ("paper-scale" if self.candidate_cycles >= 100_000 else
             "validated-reduced-budget" if self.candidate_cycles >= 2048 else
             "smoke-test-only" if self.residual else "not_applicable"),
            result.physical_rollback_failures,
            tuple(asdict(item) for item in result.rollback_outcomes),
            asdict(result.reentry_request) if result.reentry_request else None,
            asdict(result.regional_recovery) if result.regional_recovery else None,
            result.recovery_count, result.timing,
        )
