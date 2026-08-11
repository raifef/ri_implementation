"""End-to-end graph-regional HDFA-RL control loop.

The loop is backend-neutral: it consumes a QEC observation batch and emits one atomic,
supervisor-authorized policy package.  A hardware adapter is responsible for applying the
package and returning an acknowledgement hash.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from concurrent.futures import ThreadPoolExecutor
import math
import time
from typing import Mapping, Sequence

from hdfa_rl_suite.common import deterministic_hash
from hdfa_rl_suite.simulator import QECObservationBatch, ScalableQECDevice
from hdfa_rl_suite.stage0.schema import HardwareLimits, PolicySnapshot, stable_hash
from hdfa_rl_suite.stage1 import TelemetryProcessor
from hdfa_rl_suite.stage1.schema import PolicyActivation, TelemetryBatch
from hdfa_rl_suite.stage2 import InferenceConfig, LatentVariable, PhysicalInferenceEngine, QuadraticLogitObservationModel, StateSchema
from hdfa_rl_suite.stage2.schema import DetectorResponse, InferenceValidity, PhysicalStatePosterior
from hdfa_rl_suite.stage3 import (DynamicsConfig, JointDynamicsEngine,
                                  default_model_bank, extended_structured_model_bank)
from hdfa_rl_suite.stage3.schema import DynamicsPosterior
from hdfa_rl_suite.stage4 import ForecastConfig, ForecastEngine, LatencyModel, ResponseMap
from hdfa_rl_suite.stage4.schema import ForecastBundle
from hdfa_rl_suite.stage5 import MPCConfig, PredictiveController, bind_policy_lifecycle
from hdfa_rl_suite.stage5.schema import PredictedCostDistribution, PredictiveControlPackage, ResidualAllocation, SolverStatus
from hdfa_rl_suite.stage7 import SupervisoryController
from hdfa_rl_suite.stage7.schema import Authorization, StageHealth, SupervisorDecision, SupervisorInput


@dataclass
class RegionalControlStack:
    region_id: str
    controls: tuple[str, ...]
    inference: PhysicalInferenceEngine
    dynamics: JointDynamicsEngine
    forecast: ForecastEngine
    mpc: PredictiveController
    previous_state: PhysicalStatePosterior | None = None


@dataclass(frozen=True)
class RegionLoopResult:
    region_id: str
    state: PhysicalStatePosterior
    dynamics: DynamicsPosterior
    forecast: ForecastBundle
    control: PredictiveControlPackage


@dataclass(frozen=True)
class ControlLoopResult:
    schema_version: str
    telemetry: TelemetryBatch
    regions: tuple[RegionLoopResult, ...]
    proposed_control: PredictiveControlPackage | None
    supervisor: SupervisorDecision
    replay_hash: str
    compute_timings_s: Mapping[str, float] = field(default_factory=dict)
    familiar_fast_paths: Mapping[str, str] = field(default_factory=dict)


class HDFAControlLoop:
    """Sparse local filters/controllers with one global safety and atomicity boundary."""

    def __init__(self, telemetry: TelemetryProcessor, regions: Sequence[RegionalControlStack],
                 limits: HardwareLimits, supervisor: SupervisoryController,
                 latency: LatencyModel, horizons_s: Sequence[float] = (0., .05, .2),
                 *, parallel_regions: bool = False,
                 max_region_workers: int = 4) -> None:
        self.telemetry, self.regions = telemetry, {item.region_id: item for item in regions}
        self.limits, self.supervisor, self.latency = limits, supervisor, latency
        self.horizons = tuple(sorted(dict.fromkeys(horizons_s)))
        self.parallel_regions = bool(parallel_regions)
        self.max_region_workers = max(1, int(max_region_workers))
        if not self.horizons or self.horizons[0] < 0:
            raise ValueError("control horizons must be non-negative")
        self._timeline: list[PolicyActivation] = []

    def _merge(self, packages: Sequence[PredictiveControlPackage], current: PolicySnapshot) -> PredictiveControlPackage | None:
        valid = [package for package in packages if package.status is SolverStatus.OPTIMAL]
        if not valid:
            return None
        action = dict(current.values)
        trajectories: list[dict[str, float]] = [dict(current.values) for _ in range(max(len(item.trajectory) for item in valid))]
        residual_bounds: dict[str, float] = {}
        active: list[str] = []

        def project_global_duty(values: dict[str, float], reference: Mapping[str, float]) -> None:
            duty = sum(value * value for value in values.values()) / max(1, len(values))
            if duty <= self.limits.max_thermal_duty:
                return
            scale = math.sqrt(self.limits.max_thermal_duty / duty)
            for control, value in tuple(values.items()):
                bound = self.limits.controls[control]
                desired = value * scale
                values[control] = min(bound.maximum, max(bound.minimum,
                    min(reference[control] + bound.max_slew,
                        max(reference[control] - bound.max_slew, desired))))
            active.append("global-thermal-duty")

        for control in action:
            owners = [package for package in valid if control in package.action]
            proposals = [package.action[control] for package in owners
                         if abs(package.action[control] - current.values.get(control, 0.)) > 1e-15]
            if proposals:
                action[control] = sum(proposals) / len(proposals)
            bound = self.limits.controls[control]
            action[control] = min(bound.maximum, max(bound.minimum,
                min(current.values[control] + bound.max_slew, max(current.values[control] - bound.max_slew, action[control]))))
            residual_bounds[control] = min((package.residual_allocation.bounds.get(control, math.inf)
                                            for package in owners), default=0.)
            if not math.isfinite(residual_bounds[control]):
                residual_bounds[control] = 0.
        for index, trajectory_step in enumerate(trajectories):
            previous = current.values if index == 0 else trajectories[index - 1]
            for control in trajectory_step:
                owners = [package for package in valid if control in package.action]
                values = [package.trajectory[min(index, len(package.trajectory)-1)].get(control, previous[control])
                          for package in owners]
                if not values:
                    trajectory_step[control] = previous[control]
                    continue
                target = sum(values) / len(values)
                bound = self.limits.controls[control]
                trajectory_step[control] = min(bound.maximum, max(bound.minimum,
                    min(previous[control] + bound.max_slew, max(previous[control] - bound.max_slew, target))))
            project_global_duty(trajectory_step, previous)
        action = dict(trajectories[0])
        for package in valid:
            active.extend(package.active_constraints)
        activation = max(package.activation_time_s for package in valid)
        expiry = min(package.expiry_time_s for package in valid)
        policy_hash = stable_hash({"action": action, "trajectory": trajectories,
                                   "activation": activation, "base": current.policy_hash})
        residual_controls = tuple(control for control, bound in residual_bounds.items() if bound > 0.)
        allocation = ResidualAllocation(residual_controls, residual_bounds, residual_controls,
            {control: "intersection of graph-regional Stage-5 allocations" for control in residual_controls})
        cost = PredictedCostDistribution(
            sum(package.cost_distribution.expected_cost for package in valid),
            max(package.cost_distribution.worst_scenario_cost for package in valid),
            {detector: max((package.cost_distribution.detector_violation_probability.get(detector, 0.) for package in valid), default=0.)
             for detector in sorted({key for package in valid for key in package.cost_distribution.detector_violation_probability})},
            max((package.cost_distribution.cvar_cost for package in valid), default=0.),
            sum(package.cost_distribution.logical_risk for package in valid),
            sum(package.cost_distribution.correlation_risk for package in valid),
        )
        return PredictiveControlPackage("stage5.aggregate.v1", SolverStatus.OPTIMAL, action, tuple(trajectories),
            {control: action[control] - current.values[control] for control in action}, allocation,
            tuple(dict.fromkeys(active)), cost, current, policy_hash, activation, expiry, current, None,
            min((package.robustness_margin for package in valid), default=0.), dict(current.values), ())

    def step(self, batch: QECObservationBatch, *, residual_learning_requested: bool = False,
             residual_health: StageHealth | None = None,
             bootstrap_health: StageHealth | None = None,
             familiar_policy_cache: Mapping[str, Mapping[str, float]] | None = None
             ) -> ControlLoopResult:
        """Run Stages 1--5 and obtain the Stage-7 authorization boundary.

        Stage 6 is executed by :class:`HDFAProductController`, because candidate
        execution requires a device and subsequent causal observations.  The request and
        health arguments make residual learning an explicit supervisor mode instead of an
        action assembled around this loop by a benchmark harness.
        """
        compute: dict[str, float] = {}
        if not self._timeline or self._timeline[-1].policy_hash != batch.policy_activation.policy_hash:
            self._timeline.append(batch.policy_activation)
        started_ns = time.perf_counter_ns()
        telemetry = self.telemetry.process(batch.records, tuple(self._timeline), batch.context)
        compute["stage1_telemetry"] = (time.perf_counter_ns()-started_ns)/1e9
        current = PolicySnapshot(dict(batch.policy_activation.controls), batch.policy_activation.policy_hash,
                                 batch.policy_activation.nominal_activation_s)
        if telemetry.hard_invalid:
            decision = self.supervisor.tick(SupervisorInput(batch.records[-1].device_timestamp_s, (
                StageHealth("stage1", False, True, invalidity_reasons=tuple(flag.code for flag in telemetry.quality_flags)),),
                forecast_valid=False))
            compute["stage7_supervision"] = 0.0
            return ControlLoopResult("loop.v1", telemetry, (), None, decision,
                                     deterministic_hash((telemetry.replay_manifest.manifest_hash, decision.decision_id)),
                                     compute)
        def process_region(item: tuple[str, RegionalControlStack]):
            region_id, stack = item
            view = telemetry.regional_views.get(region_id)
            if view is None:
                return None
            local_compute: dict[str, float] = {}
            controls = {control: current.values.get(control, 0.) for control in stack.controls}
            started_ns = time.perf_counter_ns()
            state = stack.inference.infer(view, controls, method="particle", previous=stack.previous_state)
            local_compute["stage2_inference"] = (time.perf_counter_ns()-started_ns)/1e9
            stack.previous_state = state
            started_ns = time.perf_counter_ns()
            dynamics = stack.dynamics.update(view, controls, batch.records[-1].device_timestamp_s, state_prior=state)
            local_compute["stage3_joint_hdfa"] = (time.perf_counter_ns()-started_ns)/1e9
            started_ns = time.perf_counter_ns()
            forecast = stack.forecast.forecast(dynamics, controls, self.horizons, self.latency,
                                               context_id=batch.context.context_id)
            local_compute["stage4_forecast"] = (time.perf_counter_ns()-started_ns)/1e9
            local_current = PolicySnapshot(controls, current.policy_hash, current.timestamp_s)
            started_ns = time.perf_counter_ns()
            control = stack.mpc.solve_trajectory(forecast, self.horizons, local_current,
                                                 now_s=batch.records[-1].device_timestamp_s)
            local_compute["stage5_mpc"] = (time.perf_counter_ns()-started_ns)/1e9
            started_ns = time.perf_counter_ns()
            familiar = dynamics.familiar_process
            fast_path = None
            if familiar is not None:
                before = control.policy_hash
                cache_key = f"{region_id}|{familiar.regime_id}"
                control = stack.mpc.cached_regime_policy(
                    control, forecast, local_current,
                    (familiar_policy_cache or {}).get(cache_key), cache_key)
                if control.policy_hash == before:
                    control = stack.mpc.familiar_feedforward(
                        control, forecast, local_current,
                        stack.forecast.response_map, familiar)
                if control.policy_hash != before:
                    fast_path = next(
                        (item for item in reversed(control.active_constraints)
                         if item.startswith("validated-regime-policy-cache:")
                         or item.startswith("familiar-fast-path:")),
                        "familiar-fast-path")
            local_compute["stage5_familiar_fast_path"] = (
                time.perf_counter_ns()-started_ns)/1e9
            return RegionLoopResult(region_id, state, dynamics, forecast, control), local_compute, fast_path

        region_items = tuple(sorted(self.regions.items()))
        parallel = self.parallel_regions and len(region_items) > 1
        if parallel:
            with ThreadPoolExecutor(max_workers=min(
                    self.max_region_workers, len(region_items)),
                    thread_name_prefix="hdfa-region") as executor:
                processed = tuple(executor.map(process_region, region_items))
        else:
            processed = tuple(process_region(item) for item in region_items)
        typed = tuple(item for item in processed if item is not None)
        results = [item[0] for item in typed]
        familiar_fast_paths = {
            item[0].region_id: item[2] for item in typed if item[2] is not None}
        for stage in ("stage2_inference", "stage3_joint_hdfa", "stage4_forecast",
                      "stage5_mpc", "stage5_familiar_fast_path"):
            durations = [item[1].get(stage, 0.0) for item in typed]
            # Independent regions occupy one critical-path lane when parallel. The
            # full per-region evidence remains in each RegionLoopResult.
            compute[stage] = (max(durations, default=0.0) if parallel
                              else sum(durations))
        started_ns = time.perf_counter_ns()
        proposed = self._merge([result.control for result in results], current)
        if proposed is not None:
            proposed = bind_policy_lifecycle(
                proposed, policy_id=f"stage5:{batch.batch_id}",
                reference_policy_id=batch.policy_activation.policy_id,
                reference_policy_hash=batch.policy_activation.policy_hash,
                created_from_state_id=batch.physical_state_id,
                controller_state_hash=batch.controller_state_hash,
            )
        compute["stage5_global_merge"] = (time.perf_counter_ns()-started_ns)/1e9
        health = tuple(
            ([bootstrap_health] if bootstrap_health is not None else []) +
            [StageHealth("stage2", all(result.state.validity is InferenceValidity.VALID for result in results),
                         ood_score=max((result.state.ood_score for result in results), default=0.),
                         invalidity_reasons=tuple(reason for result in results for reason in result.state.invalidity_reasons)),
             StageHealth("stage3", not any(result.dynamics.invalidity_reasons for result in results),
                         ood_score=max((result.dynamics.unknown_model_probability for result in results), default=0.)),
             StageHealth("stage4", all(not result.forecast.invalidity_reasons for result in results)),
             StageHealth("stage5", proposed is not None, solver_ok=proposed is not None)] +
            ([residual_health] if residual_health is not None else [])
        )
        forecast_valid = bool(results) and proposed is not None and all(not result.forecast.invalidity_reasons for result in results)
        residual_safe = bool(
            residual_learning_requested and proposed is not None
            and proposed.residual_allocation.projection_controls
            and (residual_health is None or residual_health.valid)
        )
        started_ns = time.perf_counter_ns()
        supervisor = self.supervisor.tick(SupervisorInput(
            batch.records[-1].device_timestamp_s, health,
            broad_ood=any(result.state.ood_score >= 1. for result in results),
            local_change_probability=max((result.dynamics.change_alarm.probability for result in results), default=0.),
            unknown_model_probability=max((result.dynamics.unknown_model_probability for result in results), default=0.),
            observation_nonidentifiable=any(result.state.validity is InferenceValidity.LOW_OBSERVABILITY for result in results),
            diagnostic_decision_relevant=any(result.state.intervention_request is not None for result in results),
            forecast_valid=forecast_valid, residual_small=not residual_safe,
            residual_learning_safe=residual_safe,
            controller_confirmed=True, policy_hash_consistent=True,
        ))
        if proposed and supervisor.authorization is Authorization.APPROVED:
            authorization = self.supervisor.authorize_control(proposed, batch.records[-1].device_timestamp_s)
            if authorization.authorization is not Authorization.APPROVED:
                supervisor = authorization
            else:
                proposed = replace(proposed, supervisor_authorization=authorization.decision_id)
                supervisor = authorization
        compute["stage7_supervision"] = (time.perf_counter_ns()-started_ns)/1e9
        replay_hash = deterministic_hash({
            "telemetry": telemetry.replay_manifest.manifest_hash,
            "regions": [(item.region_id, item.state.mean, item.dynamics.model_evidence.model_probabilities,
                         item.control.policy_hash) for item in results],
            "decision": supervisor.decision_id,
        })
        return ControlLoopResult("loop.v2", telemetry, tuple(results), proposed,
                                 supervisor, replay_hash, compute,
                                 familiar_fast_paths)


def build_default_loop(device: ScalableQECDevice, *, seed: int = 0,
                       horizons_s: Sequence[float] = (0., .05, .2),
                       extended_structured_models: bool = False,
                       parallel_regions: bool = False) -> HDFAControlLoop:
    """Construct scalable one-hop regional stacks from the device detector-control graph."""
    telemetry = TelemetryProcessor(device.circuit.detectors, device.detector_control_graph)
    regions: list[RegionalControlStack] = []
    for index, detector in enumerate(device.circuit.detectors):
        controls = device.detector_control_graph[detector.detector_id]
        primary = next((control for control in controls if control.startswith("drive:")), controls[0])
        variable_id = f"error:{detector.region_id}"
        schema = StateSchema(detector.region_id, (LatentVariable(variable_id, "effective local optimum displacement",
            "normalized", -1., 1., intervention_control=primary, safe_intervention=.025),))
        # Quadratic logit in (x-u) retains sign ambiguity until a deliberate intervention.
        response = DetectorResponse(detector.detector_id, -4.2, {}, {},
            {(variable_id, variable_id): 7.0}, {(primary, primary): 7.0}, {(variable_id, primary): -14.0}, .08)
        observation = QuadraticLogitObservationModel(schema, (response,))
        inference = PhysicalInferenceEngine(schema, observation, InferenceConfig(seed=seed + index, particle_count=192))
        bank = (extended_structured_model_bank(variable_id)
                if extended_structured_models else default_model_bank(variable_id))
        dynamics = JointDynamicsEngine(schema, observation, bank, DynamicsConfig(seed=seed + index, particle_count=256))
        # A regional forecast must carry only the controls owned by that sparse factor.
        # Copying the complete device policy here made every one of the 256 scenarios in
        # every region retain O(P_global) unrelated values, turning an otherwise local
        # pipeline into an avoidable quadratic allocation path.
        local_reference = {control: device.confirmed_policy.controls[control] for control in controls}
        response_map = ResponseMap(local_reference, {(primary, variable_id): -1.}, validity_radius={primary: .25})
        forecast = ForecastEngine(observation, bank, response_map, ForecastConfig(seed=seed + index))
        local_limits = HardwareLimits({control: device.limits.controls[control] for control in controls},
                                      device.limits.max_thermal_duty, device.limits.max_leakage)
        mpc = PredictiveController(local_limits, observation, MPCConfig())
        regions.append(RegionalControlStack(detector.region_id, tuple(controls), inference, dynamics, forecast, mpc))
    latency = LatencyModel(device.config.cycle_period_s, .001, .001,
                           device.config.controller_latency_s, device.config.controller_latency_s * .1)
    supervisor = SupervisoryController(device.limits)
    return HDFAControlLoop(
        telemetry, regions, device.limits, supervisor, latency, horizons_s,
        parallel_regions=parallel_regions)
