"""Authoritative persistent Stage-0--7 product control path.

Unlike the stage-local pipeline, this module owns the device boundary.  Consequently
neither predictive actions nor residual-RL candidates can be applied without Stage-7
authorization, and the post-action observation is retained as the next interval's causal
telemetry rather than discarded by a benchmark wrapper.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from statistics import NormalDist
import time
from typing import Mapping

from hdfa_rl_suite.common import (
    IntervalTimingRecorder, OnlineTimingBreakdown, PolicyLifecycleState,
    deterministic_hash,
)
from hdfa_rl_suite.pipeline import ControlLoopResult, HDFAControlLoop, build_default_loop
from hdfa_rl_suite.simulator import QECObservationBatch, ScalableQECDevice
from hdfa_rl_suite.stage0 import ScalableBootstrapCalibrator, ScalableBootstrapConfig
from hdfa_rl_suite.stage0.schema import BootstrapResult, HealthStatus, PolicySnapshot, stable_hash
from hdfa_rl_suite.stage5.schema import PredictiveControlPackage, bind_policy_lifecycle
from hdfa_rl_suite.stage6 import (
    ExplorationBudget,
    GaussianResidualPolicy,
    ResidualActivationGate,
    ResidualGateConfig,
    ResidualRLConfig,
    ResidualRLController,
    bind_candidate_lifecycle,
)
from hdfa_rl_suite.stage6.schema import (
    CandidateObservation, ResidualCandidate, ResidualGateDecision,
    ResidualGateEvidence, ResidualRLDisposition, ResidualRLResult,
    ShadowValidation,
)
from hdfa_rl_suite.stage7.schema import (
    Authorization, ModelLifecycle, OperatingMode, StageHealth,
    SupervisorDecision, SupervisorInput,
)
from hdfa_rl_suite.recovery import (
    DisturbanceAwareRegionalCalibrator,
    PhysicalRestorationStatus,
    RestorationPrediction,
    RollbackVerification,
    RollbackVerificationStatus,
    RecoveryScope,
    ReentryReason,
    ReentryRequest,
    RegionalRecoveryOutcome,
    RollbackOutcome,
    TransactionRestorationStatus,
    ValidatedPolicySnapshot,
    verify_restoration_evidence,
    wilson_interval,
)


@dataclass(frozen=True)
class ProductLoopConfig:
    enable_residual_rl: bool = True
    residual_candidate_count: int = 4
    residual_candidate_cycles: int = 32
    residual_stddev: float = 0.01
    per_candidate_damage_budget: float = 0.1
    cumulative_damage_budget: float = 1.0
    qec_operability_rate_limit: float = 0.25
    rollback_validation_cycles: int = 64
    maximum_rollback_attempts: int = 3
    maximum_recovery_attempts: int = 3
    recovery_deadline_s: float = 10.0
    rollback_snapshot_max_age_s: float = 30.0
    rollback_context_noninferiority_margin: float = 0.02
    rollback_sequential_max_batches: int = 4
    rollback_irreducible_noise_floor: float = 0.005
    rollback_drift_uncertainty_per_s: float = 0.002
    residual_activation_mode: str = "conditional"
    familiar_policy_minimum_confidence: float = 0.55
    familiar_policy_max_age_s: float = 120.0
    extended_structured_models: bool = False
    parallel_regional_updates: bool = False
    candidate_elimination_z: float | None = None
    regional_unknown_entry_probability: float = 0.35
    regional_ood_entry_score: float = 0.50
    regional_background_unknown_limit: float = 0.15
    regional_background_ood_limit: float = 0.50
    regional_max_fraction: float = 0.40
    bootstrap: ScalableBootstrapConfig = field(default_factory=ScalableBootstrapConfig)

    def __post_init__(self) -> None:
        if self.residual_candidate_count < 0 or self.residual_candidate_count % 2:
            raise ValueError("residual_candidate_count must be a non-negative even integer")
        if self.residual_candidate_cycles <= 0 or self.rollback_validation_cycles <= 0:
            raise ValueError("cycle counts must be positive")
        if self.maximum_rollback_attempts <= 0 or self.maximum_recovery_attempts <= 0:
            raise ValueError("recovery and rollback attempt budgets must be positive")
        if self.rollback_sequential_max_batches <= 0:
            raise ValueError("rollback sequential batch budget must be positive")
        if self.residual_activation_mode not in {"conditional", "always_on"}:
            raise ValueError("residual_activation_mode must be conditional or always_on")
        if not 0 < self.familiar_policy_minimum_confidence <= 1:
            raise ValueError("familiar policy confidence must lie in (0,1]")
        if self.recovery_deadline_s <= 0 or self.rollback_snapshot_max_age_s <= 0:
            raise ValueError("recovery deadlines and snapshot ages must be positive")
        if self.candidate_elimination_z is not None and self.candidate_elimination_z <= 0:
            raise ValueError("candidate-elimination confidence threshold must be positive")
        probabilities = (
            self.regional_unknown_entry_probability,
            self.regional_background_unknown_limit,
            self.regional_max_fraction,
        )
        if any(not 0 < value <= 1 for value in probabilities):
            raise ValueError("regional recovery probabilities must lie in (0,1]")


@dataclass(frozen=True)
class IntegratedIntervalResult:
    schema_version: str
    interval: int
    bootstrap_reason: ReentryReason | None
    bootstrap: BootstrapResult | None
    causal_observation: QECObservationBatch
    control: ControlLoopResult
    residual_candidates: tuple[ResidualCandidate, ...]
    candidate_batches: tuple[QECObservationBatch, ...]
    candidate_observations: tuple[CandidateObservation, ...]
    residual_result: ResidualRLResult | None
    feedback_observation: QECObservationBatch
    authorization_log: tuple[SupervisorDecision, ...]
    stage_path: tuple[str, ...]
    applied_policy_hashes: tuple[str, ...]
    lifecycle_violations: tuple[str, ...]
    rollback_count: int
    bootstrap_count: int
    bootstrap_qec_cycles: int
    candidate_cycles: int
    diagnostic_shots: int
    diagnostic_downtime_s: float
    newly_acquired_telemetry_cycles: int
    replay_hash: str
    reentry_request: ReentryRequest | None = None
    regional_recovery: RegionalRecoveryOutcome | None = None
    rollback_outcomes: tuple[RollbackOutcome, ...] = ()
    physical_rollback_failures: tuple[str, ...] = ()
    recovery_count: int = 0
    timing: OnlineTimingBreakdown | None = None
    residual_gate_decision: ResidualGateDecision | None = None

    @property
    def completed_without_lifecycle_violations(self) -> bool:
        return not self.lifecycle_violations


@dataclass(frozen=True)
class ValidatedRegimePolicy:
    cache_key: str
    region_id: str
    regime_id: str
    model_family: str
    controls: Mapping[str, float]
    validated_at_s: float
    detector_rate: float
    detector_rate_ci99: tuple[float, float]
    confidence: float
    policy_id: str
    policy_hash: str
    validation_batch_id: str

class QECOperabilityError(RuntimeError):
    """Raised when Stage 0 cannot return a validated QEC-operable baseline."""

    def __init__(self, result: BootstrapResult, reason: ReentryReason) -> None:
        super().__init__(f"Stage 0 failed during {reason.value}: {result.health.invalid_reasons}")
        self.result, self.reason = result, reason

    def __reduce__(self):
        """Preserve structured constructor arguments across spawned worker processes."""
        return (type(self), (self.result, self.reason))


class RecoveryCertificationError(RuntimeError):
    """Fail-closed online recovery whose required evidence did not certify safety."""

    def __init__(self, request: ReentryRequest,
                 outcome: RegionalRecoveryOutcome | None, reason: str) -> None:
        super().__init__(
            f"disturbance-aware {request.scope.value} recovery failed: {reason}")
        self.request, self.outcome, self.reason = request, outcome, reason

    def __reduce__(self):
        return (type(self), (self.request, self.outcome, self.reason))


class HDFAProductController:
    """Persistent controller that executes exactly one supervised Stage-0--7 loop."""

    def __init__(self, device: ScalableQECDevice, *, seed: int = 0,
                 config: ProductLoopConfig = ProductLoopConfig()) -> None:
        self.device, self.seed, self.config = device, seed, config
        self._loop: HDFAControlLoop | None = None
        self._bootstrap_result: BootstrapResult | None = None
        self._pending_reentry: ReentryReason | None = ReentryReason.COLD_START
        self._pending_reentry_request: ReentryRequest | None = None
        self._residual: ResidualRLController | None = None
        self._residual_result: ResidualRLResult | None = None
        self._residual_gate = ResidualActivationGate(ResidualGateConfig())
        self._residual_magnitudes: list[float] = []
        self._familiar_policy_cache: dict[str, ValidatedRegimePolicy] = {}
        self._feedback: QECObservationBatch | None = None
        self._bootstrap_count = 0
        self._recovery_count = 0
        self._validated_snapshots: list[ValidatedPolicySnapshot] = []
        self._rejected_rollback_targets: set[str] = set()
        self._rollback_attempts: dict[str, int] = {}
        self._recovery_attempts: dict[str, int] = {}
        self._last_bootstrap_actuation_s = 0.0
        self._global_ood_seen = False
        self._interval = 0

    @property
    def bootstrap_count(self) -> int:
        return self._bootstrap_count

    @property
    def pending_reentry(self) -> ReentryReason | None:
        return self._pending_reentry

    @property
    def pending_reentry_request(self) -> ReentryRequest | None:
        return self._pending_reentry_request

    @property
    def supervisor(self):
        return self._loop.supervisor if self._loop is not None else None

    def request_reentry(self, reason: ReentryReason, *,
                        interval: int | None = None,
                        scope: RecoveryScope = RecoveryScope.GLOBAL,
                        triggering_evidence: Mapping[str, object] | None = None,
                        affected_regions: tuple[str, ...] = (),
                        affected_controls: tuple[str, ...] = (),
                        boundary_detectors: tuple[str, ...] = (),
                        common_mode_probability: float = 1.0,
                        unknown_model_probability: float = 1.0) -> ReentryRequest:
        """Create or coalesce an architecture-sanctioned auditable re-entry request."""
        if not isinstance(reason, ReentryReason):
            raise TypeError("re-entry reason must be an explicit ReentryReason")
        current = self.device.confirmed_policy
        index = self._interval if interval is None else interval
        evidence = dict(triggering_evidence or {"external_request": reason.value})
        request_id = deterministic_hash({
            "cause": reason.value, "interval": index, "scope": scope.value,
            "regions": affected_regions, "controls": affected_controls,
            "reference_policy_id": current.policy_id,
            "reference_policy_hash": current.policy_hash,
            "evidence": evidence,
        })[:24]
        request = ReentryRequest(
            "reentry-request.v1", request_id, reason, self.device.now_s, index,
            evidence, scope, tuple(sorted(set(affected_regions))),
            tuple(sorted(set(affected_controls))),
            tuple(sorted(set(boundary_detectors))),
            max(0.0, min(1.0, common_mode_probability)),
            max(0.0, min(1.0, unknown_model_probability)),
            current.policy_id, current.policy_hash, self.device.controller_state_hash,
            0, self.device.now_s+self.config.recovery_deadline_s,
            ("qec", "sensitivity", "uncertainty", "held_out",
             "hardware_bounds", "rollback_available", "boundary_validation"),
        )
        pending = self._pending_reentry_request
        if pending is not None:
            # Global scope dominates; repeated identical observations are idempotent.
            if pending.request_id == request.request_id:
                return pending
            use_global = (pending.scope is RecoveryScope.GLOBAL
                          or request.scope is RecoveryScope.GLOBAL)
            selected = pending if pending.requested_at_s <= request.requested_at_s else request
            request = replace(
                selected,
                scope=RecoveryScope.GLOBAL if use_global else RecoveryScope.REGIONAL,
                affected_regions=tuple(sorted(set(
                    (*pending.affected_regions, *request.affected_regions)))),
                affected_controls=tuple(sorted(set(
                    (*pending.affected_controls, *request.affected_controls)))),
                boundary_detectors=tuple(sorted(set(
                    (*pending.boundary_detectors, *request.boundary_detectors)))),
                common_mode_probability=max(
                    pending.common_mode_probability, request.common_mode_probability),
                unknown_model_probability=max(
                    pending.unknown_model_probability, request.unknown_model_probability),
                escalation_count=max(
                    pending.escalation_count, request.escalation_count),
                deadline_s=min(pending.deadline_s, request.deadline_s),
                coalesced_request_ids=tuple(dict.fromkeys((
                    *pending.coalesced_request_ids, pending.request_id,
                    *request.coalesced_request_ids, request.request_id))),
            )
        self._pending_reentry = reason
        self._pending_reentry_request = request
        return request

    def _all_regions(self) -> tuple[str, ...]:
        return tuple(detector.region_id for detector in self.device.circuit.detectors)

    def _boundary_detectors(self, regions: tuple[str, ...],
                            controls: tuple[str, ...]) -> tuple[str, ...]:
        selected = set(regions)
        linked = set(controls)
        return tuple(sorted(
            detector.detector_id for detector in self.device.circuit.detectors
            if detector.region_id not in selected
            and not linked.isdisjoint(
                self.device.detector_control_graph[detector.detector_id])))

    def _request_from_control(self, output: ControlLoopResult,
                              interval: int) -> ReentryRequest:
        rows = []
        for region in output.regions:
            rows.append({
                "region_id": region.region_id,
                "ood_score": region.state.ood_score,
                "unknown_model_probability": region.dynamics.unknown_model_probability,
                "local_change_probability": region.dynamics.change_alarm.probability,
                "max_abs_standardized_residual": (
                    region.state.posterior_predictive.max_abs_residual),
                "triggering_detectors": tuple(
                    detector for detector, value in sorted(
                        region.state.posterior_predictive.standardized_residuals.items(),
                        key=lambda item: abs(item[1]), reverse=True)
                    if abs(value) >= 1.0),
            })
        triggered = {
            row["region_id"] for row in rows
            if (row["unknown_model_probability"]
                >= self.config.regional_unknown_entry_probability
                or row["ood_score"] >= self.config.regional_ood_entry_score)
        }
        region_controls = {
            detector.region_id: tuple(
                self.device.detector_control_graph[detector.detector_id])
            for detector in self.device.circuit.detectors
        }
        controls = tuple(sorted({
            control for region in triggered for control in region_controls[region]
        }))
        background = [row for row in rows if row["region_id"] not in triggered]
        connected = True
        if triggered:
            unseen = set(triggered)
            frontier = {next(iter(unseen))}
            seen: set[str] = set()
            while frontier:
                region = frontier.pop()
                if region in seen:
                    continue
                seen.add(region)
                unseen.discard(region)
                neighbours = {
                    other for other in triggered
                    if not set(region_controls[region]).isdisjoint(region_controls[other])
                }
                frontier.update(neighbours-seen)
            connected = not unseen
        fraction = len(triggered)/max(1, len(rows))
        max_unknown = max((float(row["unknown_model_probability"])
                           for row in rows), default=0.0)
        background_quiet = all(
            row["unknown_model_probability"]
            < self.config.regional_background_unknown_limit
            and row["ood_score"] < self.config.regional_background_ood_limit
            for row in background)
        positive_locality = bool(
            triggered and connected and background_quiet
            and (len(triggered) == 1
                 or max_unknown < self.config.regional_unknown_entry_probability)
            and not self._global_ood_seen
            and fraction <= self.config.regional_max_fraction)
        scope = RecoveryScope.REGIONAL if positive_locality else RecoveryScope.GLOBAL
        if scope is RecoveryScope.GLOBAL:
            self._global_ood_seen = True
        affected_regions = tuple(sorted(triggered)) if positive_locality else ()
        affected_controls = controls if positive_locality else ()
        boundaries = (self._boundary_detectors(affected_regions, affected_controls)
                      if positive_locality else ())
        common_probability = min(1.0, fraction + (0.25 if not connected else 0.0))
        evidence = {
            "supervisor_mode": output.supervisor.mode.value,
            "supervisor_reason": output.supervisor.reason,
            "regional_evidence": tuple(rows),
            "triggered_regions": tuple(sorted(triggered)),
            "connected_trigger_component": connected,
            "background_quiet": background_quiet,
            "positive_locality": positive_locality,
            "classification_rule": "regional-locality.v1",
        }
        return self.request_reentry(
            ReentryReason.OOD_RECALIBRATION, interval=interval, scope=scope,
            triggering_evidence=evidence, affected_regions=affected_regions,
            affected_controls=affected_controls, boundary_detectors=boundaries,
            common_mode_probability=common_probability,
            unknown_model_probability=max_unknown)

    def _register_snapshot(self, batch: QECObservationBatch, *,
                           evidence_phase: str,
                           scope: RecoveryScope = RecoveryScope.GLOBAL,
                           regions: tuple[str, ...] | None = None) -> None:
        ci95 = wilson_interval(batch.detector_events, batch.detector_exposures)
        ci99 = wilson_interval(
            batch.detector_events, batch.detector_exposures,
            z=2.5758293035489004)
        if (not math.isfinite(batch.detector_rate)
                or ci99[1] > self.config.qec_operability_rate_limit
                or not batch.policy_activation.activation_acknowledgement):
            return
        snapshot = ValidatedPolicySnapshot(
            "validated-policy-snapshot.v1", batch.policy_activation.policy_id,
            batch.policy_activation.policy_hash,
            dict(batch.policy_activation.controls), self.device.now_s,
            scope, tuple(regions or self._all_regions()), batch.detector_rate,
            ci95, batch.detector_exposures, batch.batch_id,
            batch.controller_state_hash, evidence_phase,
            self.config.qec_operability_rate_limit, True)
        self._validated_snapshots.append(snapshot)

    def _register_familiar_policies(self, output: ControlLoopResult,
                                    feedback: QECObservationBatch) -> None:
        """Cache only current detector-validated policies, never model proposals."""
        for region in output.regions:
            familiar = region.dynamics.familiar_process
            if (familiar is None
                    or familiar.confidence < self.config.familiar_policy_minimum_confidence
                    or familiar.invalidity_reasons):
                continue
            detector_ids = tuple(
                detector.detector_id for detector in self.device.circuit.detectors
                if detector.region_id == region.region_id)
            events = sum(feedback.detector_counts.get(detector, (0, 0))[0]
                         for detector in detector_ids)
            exposures = sum(feedback.detector_counts.get(detector, (0, 0))[1]
                            for detector in detector_ids)
            ci99 = wilson_interval(events, exposures, z=2.5758293035489004)
            if not exposures or ci99[1] > self.config.qec_operability_rate_limit:
                continue
            controls = tuple(self._loop.regions[region.region_id].controls)
            key = f"{region.region_id}|{familiar.regime_id}"
            policy = self.device.confirmed_policy
            self._familiar_policy_cache[key] = ValidatedRegimePolicy(
                key, region.region_id, familiar.regime_id, familiar.family,
                {control: policy.controls[control] for control in controls},
                self.device.now_s, events/exposures, ci99, familiar.confidence,
                policy.policy_id, policy.policy_hash, feedback.batch_id)

    def _active_familiar_cache(self) -> Mapping[str, Mapping[str, float]]:
        return {
            key: dict(item.controls) for key, item in self._familiar_policy_cache.items()
            if self.device.now_s-item.validated_at_s
            <= self.config.familiar_policy_max_age_s}

    def _register_bootstrap_snapshot(self, result: BootstrapResult,
                                     evidence_phase: str) -> None:
        estimate = result.calibration_estimates.get("final_validation")
        if estimate is None:
            return
        rate = float(estimate.model_scores.get(
            "detector_rate", self.device.config.base_detector_probability))
        upper99 = float(estimate.model_scores.get("upper_99", 1.0))
        se = max(0.0, (upper99-rate)/2.326)
        ci95 = (max(0.0, rate-1.959963984540054*se),
                min(1.0, rate+1.959963984540054*se))
        activation = self.device.confirmed_policy
        snapshot = ValidatedPolicySnapshot(
            "validated-policy-snapshot.v1", activation.policy_id,
            activation.policy_hash, dict(activation.controls), self.device.now_s,
            RecoveryScope.GLOBAL, self._all_regions(), rate, ci95,
            self.config.bootstrap.validation_cycles*len(self.device.circuit.detectors),
            f"bootstrap:{result.replay_hash}", self.device.controller_state_hash,
            evidence_phase, self.config.qec_operability_rate_limit, True)
        self._validated_snapshots.append(snapshot)

    def _select_rollback_snapshot(self, request: ReentryRequest
                                  ) -> ValidatedPolicySnapshot | None:
        compatible = []
        affected = set(request.affected_regions)
        for snapshot in self._validated_snapshots:
            if (not snapshot.independently_validated
                    or snapshot.policy_id in self._rejected_rollback_targets):
                continue
            if request.scope is RecoveryScope.GLOBAL and snapshot.scope is not RecoveryScope.GLOBAL:
                continue
            if (request.scope is RecoveryScope.REGIONAL
                    and snapshot.scope is RecoveryScope.REGIONAL
                    and not affected <= set(snapshot.regions)):
                continue
            age = self.device.now_s-snapshot.validated_at_s
            if age <= self.config.rollback_snapshot_max_age_s:
                compatible.append(snapshot)
        if compatible:
            return max(compatible, key=lambda item: item.validated_at_s)
        # A broad event retains one explicit global known-safe fallback even if old;
        # its age remains visible and physical validation is still mandatory.
        if request.scope is not RecoveryScope.GLOBAL:
            return None
        fallbacks = [
            item for item in self._validated_snapshots
            if item.scope is RecoveryScope.GLOBAL
            and item.policy_id not in self._rejected_rollback_targets
        ]
        return min(fallbacks, key=lambda item: item.validated_at_s) if fallbacks else None

    def _bootstrap_accounting(self, result: BootstrapResult,
                              elapsed_s: float) -> tuple[int, int, float]:
        shots = qec_cycles = 0
        for estimate in result.calibration_estimates.values():
            diagnostics = estimate.diagnostics
            shots += int(diagnostics.get("active_design_shots", 0))
            shots += int(diagnostics.get("held_out_shots", 0))
            qec_cycles += int(diagnostics.get("qec_cycles", 0))
        diagnostic_downtime = shots*self.device.config.cycle_period_s
        qec_time = qec_cycles*self.device.config.cycle_period_s
        self._last_bootstrap_actuation_s = max(
            0.0, elapsed_s-diagnostic_downtime-qec_time)
        return shots, qec_cycles, diagnostic_downtime

    def _run_bootstrap(self) -> tuple[ReentryReason, BootstrapResult, int, int, float]:
        reason = self._pending_reentry or ReentryReason.COLD_START
        started = self.device.now_s
        result = ScalableBootstrapCalibrator(self.device, self.config.bootstrap).run()
        elapsed = self.device.now_s - started
        self._bootstrap_count += 1
        self._bootstrap_result = result
        self._feedback = None
        self._residual = None
        self._residual_result = None
        self._residual_gate.reset_after_recalibration()
        self._residual_magnitudes.clear()
        self._familiar_policy_cache.clear()
        if result.health.status is not HealthStatus.PASSED:
            self._pending_reentry = reason
            raise QECOperabilityError(result, reason)
        self._loop = build_default_loop(
            self.device, seed=self.seed, horizons_s=self._forecast_horizons(),
            extended_structured_models=self.config.extended_structured_models,
            parallel_regions=self.config.parallel_regional_updates)
        self._loop.supervisor.set_model_lifecycle(
            f"bootstrap:{result.replay_hash}", ModelLifecycle.PROMOTED, held_out_passed=True)
        self._loop.supervisor.set_model_lifecycle("joint-dynamics.v2", ModelLifecycle.SHADOW)
        self._pending_reentry = None
        self._pending_reentry_request = None
        self._feedback = None
        self._register_bootstrap_snapshot(
            result, "stationary-stage0" if reason is ReentryReason.COLD_START
            else "online-disturbance-aware-global-recovery")
        shots, qec_cycles, downtime = self._bootstrap_accounting(result, elapsed)
        return reason, result, shots, qec_cycles, downtime

    def accept_validated_bootstrap(self, result: BootstrapResult) -> None:
        """Attach a previously executed Stage-0 result without silently recalibrating."""
        if result.health.status is not HealthStatus.PASSED:
            raise QECOperabilityError(result, ReentryReason.COLD_START)
        if result.qec_circuit.circuit_hash != self.device.circuit.circuit_hash:
            raise ValueError("bootstrap circuit does not match the controlled device")
        if result.baseline_policy.policy_hash != self.device.confirmed_policy.policy_hash:
            raise ValueError("bootstrap baseline is not the device's confirmed policy")
        self._bootstrap_result = result
        self._bootstrap_count = 1
        self._loop = build_default_loop(
            self.device, seed=self.seed, horizons_s=self._forecast_horizons(),
            extended_structured_models=self.config.extended_structured_models,
            parallel_regions=self.config.parallel_regional_updates)
        self._loop.supervisor.set_model_lifecycle(
            f"bootstrap:{result.replay_hash}", ModelLifecycle.PROMOTED, held_out_passed=True)
        self._loop.supervisor.set_model_lifecycle("joint-dynamics.v2", ModelLifecycle.SHADOW)
        self._pending_reentry = None
        self._pending_reentry_request = None
        self._register_bootstrap_snapshot(result, "stationary-stage0-import")

    def _forecast_horizons(self) -> tuple[float, ...]:
        """Cover the complete declared residual experiment before authorizing it.

        Candidate cycles and both reference/candidate activations consume physical
        time.  Forecast authority therefore extends to the mean-policy commit rather
        than silently using the former 0.2 s package after a multi-second campaign.
        The Stage-4 ten-second demonstrated-validity cap remains unchanged and will
        still fail closed for a configuration that exceeds it.
        """
        latency = self.device.config.controller_latency_s
        campaign = self.config.residual_candidate_count * (
            self.config.residual_candidate_cycles*self.device.config.cycle_period_s
            + 2*latency)
        mean_commit = 2*latency
        safety_margin = max(.05, 2*self.device.config.cycle_period_s)
        return (0.0, .05, max(.2, campaign+mean_commit+safety_margin))

    def _bootstrap_health(self) -> StageHealth:
        assert self._bootstrap_result is not None
        return StageHealth(
            "stage0", self._bootstrap_result.health.status is HealthStatus.PASSED,
            invalidity_reasons=self._bootstrap_result.health.invalid_reasons,
            policy_hash=self._bootstrap_result.baseline_policy.policy_hash,
            model_version=self._bootstrap_result.schema_version,
            timestamp_s=self._bootstrap_result.baseline_policy.timestamp_s,
        )

    def _residual_health(self) -> StageHealth:
        result = self._residual_result
        return StageHealth(
            "stage6", result is None or not result.fallback_requested,
            residual_bias=result.residual_bias if result else 0.0,
            exploration_damage=result.cumulative_damage if result else 0.0,
            invalidity_reasons=result.invalidity_reasons if result else (),
            model_version=f"residual-policy.v{result.policy_version}" if result else "residual-policy.v0",
            timestamp_s=self.device.now_s,
        )

    def _residual_gate_evidence(self, output: ControlLoopResult,
                                causal: QECObservationBatch) -> ResidualGateEvidence:
        magnitudes = [region.state.posterior_predictive.max_abs_residual
                      for region in output.regions]
        magnitude = max(magnitudes, default=0.0)
        self._residual_magnitudes.append(magnitude)
        self._residual_magnitudes = self._residual_magnitudes[-8:]
        noise_floor = 1.0  # standardized posterior-predictive residual units
        persistence = 0
        for value in reversed(self._residual_magnitudes):
            if value <= 1.5*noise_floor:
                break
            persistence += 1
        recent = self._residual_magnitudes[-max(2, persistence):]
        spread = (math.sqrt(sum((value-sum(recent)/len(recent))**2
                                for value in recent)/len(recent))
                  if recent else math.inf)
        repeatability = max(0.0, min(1.0, 1-spread/max(magnitude, noise_floor)))
        identifiable = bool(output.regions) and all(
            not region.state.observability.unresolved_variable_ids
            and region.state.intervention_request is None
            for region in output.regions)
        sensitivities = [
            abs(value) for region in output.regions
            for value in region.state.attribution.values()]
        sensitivity = max(sensitivities, default=1.0 if output.proposed_control else 0.0)
        variances = [
            abs(value) for region in output.regions
            for risk in region.forecast.risk_by_horizon.values()
            for value in risk.state_variance.values()]
        forecast_uncertainty = min(1.0, math.sqrt(max(variances, default=0.0)))
        stability = 1-max((
            risk.worst_region_probability for region in output.regions
            for risk in region.forecast.risk_by_horizon.values()), default=0.0)
        previous_shadow = (self._residual_result.shadow_validation
                           if self._residual_result else None)
        probability = (previous_shadow.probability_positive_value
                       if previous_shadow else .5)
        gain = previous_shadow.estimated_gain if previous_shadow else 0.0
        snr = self._residual_result.gradient_snr if self._residual_result else 0.0
        lifecycle_healthy = not any(
            item.rollback_required or item.authorization is Authorization.REJECTED
            for item in (output.supervisor,))
        return ResidualGateEvidence(
            persistence, repeatability, sensitivity, identifiable,
            forecast_uncertainty, stability, magnitude, noise_floor,
            probability, gain, snr, lifecycle_healthy,
            predictive_only_adequate=magnitude <= noise_floor,
            ood=any(region.state.ood_score >= 1.0 for region in output.regions),
            rollback_unresolved=self._pending_reentry is ReentryReason.FAILED_ROLLBACK,
            predictive_better=(previous_shadow is not None
                               and previous_shadow.estimated_gain < 0),
        )

    def _restoration_prediction(self, output: ControlLoopResult,
                                snapshot: ValidatedPolicySnapshot,
                                target: Mapping[str, float],
                                conditioned_at_s: float) -> RestorationPrediction:
        detector_rates: dict[str, float] = {}
        detector_stddev: dict[str, float] = {}
        regional_upper: dict[str, float] = {}
        model_terms: list[float] = []
        elapsed = max(0.0, self.device.now_s-conditioned_at_s)
        drift_uncertainty = self.config.rollback_drift_uncertainty_per_s*elapsed
        first_view = next(iter(output.telemetry.regional_views.values()), None)
        context_id = first_view.context.context_id if first_view is not None else "default"
        assert self._loop is not None
        for region in output.regions:
            stack = self._loop.regions[region.region_id]
            samples = region.state.samples or ()
            for detector in stack.inference.model.responses:
                values = []
                weights = []
                if samples:
                    for sample in samples:
                        values.append(stack.inference.model.probability(
                            detector, sample.state, target, context_id))
                        weights.append(sample.weight)
                else:
                    values = [stack.inference.model.probability(
                        detector, region.state.mean, target, context_id)]
                    weights = [1.0]
                total = sum(weights) or 1.0
                mean = sum(value*weight for value, weight in zip(values, weights))/total
                variance = sum(weight*(value-mean)**2
                               for value, weight in zip(values, weights))/total
                discrepancy = max(
                    region.state.model_discrepancy,
                    region.dynamics.unknown_model_probability*.02)
                stddev = math.sqrt(max(0.0, variance)+discrepancy**2
                                   + drift_uncertainty**2)
                detector_rates[detector] = mean
                detector_stddev[detector] = stddev
                model_terms.append(discrepancy)
            local = [detector for detector in stack.inference.model.responses
                     if detector in detector_rates]
            if local:
                regional_upper[region.region_id] = min(
                    self.config.qec_operability_rate_limit,
                    max(detector_rates[item]+2.5758293035489004*
                        detector_stddev[item] for item in local)
                    + self.config.rollback_irreducible_noise_floor
                    + self.config.rollback_context_noninferiority_margin)
        mean_rate = (sum(detector_rates.values())/len(detector_rates)
                     if detector_rates else math.nan)
        epistemic = max(detector_stddev.values(), default=1.0)
        expected_upper = (min(
            self.config.qec_operability_rate_limit,
            mean_rate+2.5758293035489004*epistemic
            + self.config.rollback_irreducible_noise_floor
            + self.config.rollback_context_noninferiority_margin)
            if math.isfinite(mean_rate) else 0.0)
        invalid = (() if detector_rates else
                   ("no current causal posterior was available",))
        return RestorationPrediction(
            "restoration-prediction.v1", conditioned_at_s,
            snapshot.policy_id, snapshot.policy_hash, elapsed,
            detector_rates, detector_stddev, regional_upper,
            mean_rate, expected_upper, self.config.qec_operability_rate_limit,
            max(model_terms, default=1.0), drift_uncertainty,
            self.config.rollback_irreducible_noise_floor, True, invalid)

    def _ensure_residual(self, package: PredictiveControlPackage) -> ResidualRLController:
        controls = tuple(package.residual_allocation.projection_controls)
        if self._residual is not None and set(self._residual.policy.mean) == set(controls):
            return self._residual
        old_mean = self._residual.policy.mean if self._residual is not None else {}
        mean = {control: max(-package.residual_allocation.bounds[control],
                            min(package.residual_allocation.bounds[control], old_mean.get(control, 0.0)))
                for control in controls}
        stddev = {control: min(self.config.residual_stddev,
                               package.residual_allocation.bounds[control]) for control in controls}
        policy = GaussianResidualPolicy(mean, stddev, 0,
            {control: {control: stddev[control] ** 2} for control in controls})
        rl_config = ResidualRLConfig(
            seed=self.seed,
            minimum_candidates=2,
            maximum_candidates=max(2, self.config.residual_candidate_count),
            # Residual coordinates are already small and uncertainty-scaled; covariance
            # preconditioning prevents one noisy microbatch from saturating the issued
            # subspace and spuriously forcing Stage-0 re-entry.
            natural_gradient=True,
        )
        self._residual = ResidualRLController(
            policy, self.device.detector_control_graph,
            ExplorationBudget(self.config.per_candidate_damage_budget,
                              self.config.cumulative_damage_budget),
            rl_config,
        )
        assert self._loop is not None
        model_id = f"residual-policy.v{self._residual.policy.version}"
        if model_id not in self._loop.supervisor.model_lifecycle:
            self._loop.supervisor.set_model_lifecycle(model_id, ModelLifecycle.CANDIDATE)
        return self._residual

    def _combined_package(self, package: PredictiveControlPackage,
                          residual_mean: Mapping[str, float]) -> PredictiveControlPackage:
        action = dict(package.action)
        for control in package.residual_allocation.projection_controls:
            residual = max(-package.residual_allocation.bounds[control],
                           min(package.residual_allocation.bounds[control], residual_mean.get(control, 0.0)))
            bound = self.device.limits.controls[control]
            action[control] = max(bound.minimum, min(bound.maximum, action[control] + residual))
        duty = sum(value * value for value in action.values()) / max(1, len(action))
        if duty > self.device.limits.max_thermal_duty:
            scale = math.sqrt(self.device.limits.max_thermal_duty / duty)
            action = {control: value * scale for control, value in action.items()}
        policy_hash = stable_hash({"baseline": package.policy_hash, "residual": dict(residual_mean), "action": action})
        return replace(package, action=action, trajectory=(action,), policy_hash=policy_hash,
                       active_constraints=package.active_constraints + ("stage6-residual-subspace",),
                       controller_acknowledged_hash=None)

    def _apply_bound_package(self, package: PredictiveControlPackage
                             ) -> tuple[str, float, str]:
        started = self.device.now_s
        activation = self.device.apply_policy(
            package.action, policy_id=package.policy_id,
            reference_policy_id=package.reference_policy_id,
            reference_policy_hash=package.reference_policy_hash,
            created_from_state_id=package.created_from_state_id,
            expected_activation_state_id=package.expected_activation_state_id,
            supervisor_authorization=package.supervisor_authorization,
        )
        confirmed = self.device.await_policy_acknowledgement()
        if (confirmed.policy_hash != activation.policy_hash
                or confirmed.lifecycle_state is not PolicyLifecycleState.CONFIRMED
                or not confirmed.activation_acknowledgement):
            raise ValueError("policy activation acknowledgement did not confirm the full hash")
        return confirmed.policy_hash, self.device.now_s-started, confirmed.activation_acknowledgement

    def _run_regional_recovery(self, request: ReentryRequest
                               ) -> tuple[RegionalRecoveryOutcome | None,
                                          tuple[SupervisorDecision, ...]]:
        assert self._loop is not None and self._bootstrap_result is not None
        decisions = [self._loop.supervisor.begin_local_recovery(
            self.device.now_s, request.request_id)]
        attempts = self._recovery_attempts.get(request.request_id, 0)+1
        self._recovery_attempts[request.request_id] = attempts
        if (attempts > self.config.maximum_recovery_attempts
                or self.device.now_s > request.deadline_s):
            self._pending_reentry = None
            self._pending_reentry_request = None
            escalated = self.request_reentry(
                request.cause, interval=request.requested_interval,
                scope=RecoveryScope.GLOBAL,
                triggering_evidence={
                    "regional_request": request.request_id,
                    "reason": "regional recovery attempt/deadline budget exhausted",
                },
                common_mode_probability=1.0,
                unknown_model_probability=request.unknown_model_probability)
            self._pending_reentry_request = replace(
                escalated, escalation_count=request.escalation_count+1)
            return None, tuple(decisions)
        outcome = DisturbanceAwareRegionalCalibrator(
            self.device, request, self.config.bootstrap,
            rollback_available=self._select_rollback_snapshot(request) is not None).run()
        self._recovery_count += 1
        if not outcome.passed:
            self._pending_reentry = None
            self._pending_reentry_request = None
            escalated = self.request_reentry(
                request.cause, interval=request.requested_interval,
                scope=RecoveryScope.GLOBAL,
                triggering_evidence={
                    "regional_request": request.request_id,
                    "regional_recovery_hash": outcome.replay_hash,
                    "failed_gates": outcome.invalid_reasons,
                    "reason": "regional or boundary evidence did not certify locality",
                }, common_mode_probability=1.0,
                unknown_model_probability=request.unknown_model_probability)
            self._pending_reentry_request = replace(
                escalated, escalation_count=request.escalation_count+1)
            return outcome, tuple(decisions)

        current = self.device.confirmed_policy
        registry = dict(self._bootstrap_result.parameter_registry)
        for control in request.affected_controls:
            record = registry[control]
            local_jacobian = outcome.sensitivity_jacobian.get(control, {})
            registry[control] = replace(
                record, current_value=current.controls[control],
                covariance=outcome.characterized_variances.get(
                    control, record.covariance),
                sensitivity_scale=max((abs(value) for value in local_jacobian.values()),
                                      default=record.sensitivity_scale or 0.0),
                local_jacobian=dict(local_jacobian),
                validity_until_s=self.device.now_s+120.0,
                model_version="regional-recovery.v1")
        self._bootstrap_result = replace(
            self._bootstrap_result,
            baseline_policy=PolicySnapshot(
                dict(current.controls), current.policy_hash, self.device.now_s),
            parameter_registry=registry)
        # Refresh only affected regional models. Unaffected posterior/dynamics stacks and
        # control values remain warm and named by the prior confirmed policy.
        fresh = build_default_loop(
            self.device, seed=self.seed, horizons_s=self._forecast_horizons(),
            extended_structured_models=self.config.extended_structured_models,
            parallel_regions=self.config.parallel_regional_updates)
        for region in request.affected_regions:
            if region in fresh.regions:
                self._loop.regions[region] = fresh.regions[region]
        if self._residual is not None:
            for control in request.affected_controls:
                if control in self._residual.policy.mean:
                    self._residual.policy.mean[control] = 0.0
                    stddev = min(
                        self.config.residual_stddev,
                        self.device.limits.controls[control].trust_radius)
                    self._residual.policy.stddev[control] = stddev
                    if self._residual.policy.covariance is not None:
                        self._residual.policy.covariance[control] = {
                            control: stddev*stddev}
            self._residual.policy.version += 1
            self._residual_result = None
        self._pending_reentry = None
        self._pending_reentry_request = None
        self._register_snapshot(
            outcome.qec_validation_batch,
            evidence_phase=outcome.evidence_phase,
            scope=RecoveryScope.REGIONAL,
            regions=request.affected_regions)
        decisions.append(self._loop.supervisor.complete_recovery(
            self.device.now_s, request_id=request.request_id, regional=True,
            evidence_hash=outcome.replay_hash))
        return outcome, tuple(decisions)

    def _run_global_recovery(self, request: ReentryRequest
                             ) -> tuple[RegionalRecoveryOutcome,
                                        tuple[SupervisorDecision, ...]]:
        """Run active-disturbance global recovery without mislabelling it Stage 0."""
        assert self._loop is not None and self._bootstrap_result is not None
        decisions = [self._loop.supervisor.begin_global_recovery(
            self.device.now_s, request.request_id)]
        attempts = self._recovery_attempts.get(request.request_id, 0)+1
        self._recovery_attempts[request.request_id] = attempts
        if (attempts > self.config.maximum_recovery_attempts
                or self.device.now_s > request.deadline_s):
            self._pending_reentry = request.cause
            self._pending_reentry_request = request
            raise RecoveryCertificationError(
                request, None, "recovery attempt/deadline budget exhausted")
        outcome = DisturbanceAwareRegionalCalibrator(
            self.device, request, self.config.bootstrap,
            rollback_available=self._select_rollback_snapshot(request) is not None).run()
        self._recovery_count += 1
        if not outcome.passed:
            self._pending_reentry = request.cause
            self._pending_reentry_request = replace(
                request, escalation_count=request.escalation_count+1)
            raise RecoveryCertificationError(
                request, outcome, f"required gates {outcome.invalid_reasons}")

        current = self.device.confirmed_policy
        registry = dict(self._bootstrap_result.parameter_registry)
        for control in self.device.limits.controls:
            record = registry[control]
            local_jacobian = outcome.sensitivity_jacobian.get(control, {})
            registry[control] = replace(
                record, current_value=current.controls[control],
                covariance=outcome.characterized_variances.get(
                    control, record.covariance),
                sensitivity_scale=max((abs(value) for value in local_jacobian.values()),
                                      default=record.sensitivity_scale or 0.0),
                local_jacobian=dict(local_jacobian),
                validity_until_s=self.device.now_s+120.0,
                model_version="global-disturbance-aware-recovery.v1")
        self._bootstrap_result = replace(
            self._bootstrap_result,
            baseline_policy=PolicySnapshot(
                dict(current.controls), current.policy_hash, self.device.now_s),
            parameter_registry=registry)
        fresh = build_default_loop(
            self.device, seed=self.seed, horizons_s=self._forecast_horizons(),
            extended_structured_models=self.config.extended_structured_models,
            parallel_regions=self.config.parallel_regional_updates)
        self._loop.telemetry = fresh.telemetry
        self._loop.regions = fresh.regions
        self._residual = None
        self._residual_result = None
        self._residual_gate.reset_after_recalibration()
        self._residual_magnitudes.clear()
        self._familiar_policy_cache.clear()
        self._feedback = None
        self._pending_reentry = None
        self._pending_reentry_request = None
        self._global_ood_seen = False
        self._register_snapshot(
            outcome.qec_validation_batch,
            evidence_phase=outcome.evidence_phase,
            scope=RecoveryScope.GLOBAL, regions=self._all_regions())
        decisions.append(self._loop.supervisor.complete_recovery(
            self.device.now_s, request_id=request.request_id, regional=False,
            evidence_hash=outcome.replay_hash))
        return outcome, tuple(decisions)

    def _rollback(self, request: ReentryRequest,
                  pre_rollback_observation: QECObservationBatch,
                  control_output: ControlLoopResult,
                  authorizations: list[SupervisorDecision], applied: list[str],
                  violations: list[str], physical_failures: list[str]
                  ) -> tuple[QECObservationBatch, RollbackOutcome]:
        assert self._loop is not None
        validation_cycles = max(
            self.config.rollback_validation_cycles,
            self.config.bootstrap.validation_cycles)
        snapshot = self._select_rollback_snapshot(request)
        attempt = self._rollback_attempts.get(request.request_id, 0)+1
        self._rollback_attempts[request.request_id] = attempt
        scope_controls = (request.affected_controls
                          if request.scope is RecoveryScope.REGIONAL
                          else tuple(self.device.limits.controls))
        if snapshot is None or attempt > self.config.maximum_rollback_attempts:
            reason = ("no scope-compatible independently validated rollback snapshot"
                      if snapshot is None else "rollback attempt budget exhausted")
            violations.append(f"rollback transaction failed: {reason}")
            validation = self.device.acquire(validation_cycles)
            outcome = RollbackOutcome(
                "rollback-outcome.v1", request.request_id,
                snapshot.policy_id if snapshot else None,
                snapshot.policy_hash if snapshot else None, request.scope,
                request.affected_regions, tuple(scope_controls),
                TransactionRestorationStatus.FAILED,
                PhysicalRestorationStatus.NOT_EVALUATED,
                (), (), (), None, self.device.confirmed_policy.policy_hash,
                None, self.config.qec_operability_rate_limit,
                validation.detector_rate, None, validation.batch_id,
                validation.cycles, attempt, 0.0, reason,
                verification_status=RollbackVerificationStatus.TRANSACTION_FAILED)
            self.request_reentry(
                ReentryReason.FAILED_ROLLBACK, interval=self._interval,
                scope=RecoveryScope.GLOBAL,
                triggering_evidence={"rollback_outcome": deterministic_hash(outcome)},
                common_mode_probability=1.0, unknown_model_probability=1.0)
            return validation, outcome

        initial = dict(self.device.confirmed_policy.controls)
        target = dict(initial)
        for control in scope_controls:
            target[control] = snapshot.controls[control]
        maximum_steps = max((
            math.ceil(abs(target[control]-initial[control]) /
                      max(self.device.limits.controls[control].max_slew, 1e-12))
            for control in scope_controls), default=1)
        ids: list[str] = []
        hashes: list[str] = []
        acknowledgements: list[str] = []
        transaction_status = TransactionRestorationStatus.CONFIRMED
        transaction_reason = "all bounded steps atomically acknowledged"
        elapsed = 0.0
        authorization_id = next(
            (decision.decision_id for decision in reversed(authorizations)
             if decision.rollback_required),
            f"rollback:{request.request_id}")
        for step_index in range(1, max(1, maximum_steps)+1):
            fraction = step_index/max(1, maximum_steps)
            step = dict(initial)
            for control in scope_controls:
                step[control] = initial[control] + fraction*(target[control]-initial[control])
            reference = self.device.confirmed_policy
            policy_id = f"rollback:{request.request_id}:{attempt}:{step_index}"
            try:
                started = self.device.now_s
                activation = self.device.apply_policy(
                    step, policy_id=policy_id,
                    reference_policy_id=reference.policy_id,
                    reference_policy_hash=reference.policy_hash,
                    created_from_state_id=stable_hash({
                        "request": request.request_id,
                        "controller_state_hash": self.device.controller_state_hash,
                        "step": step_index,
                    }), supervisor_authorization=authorization_id)
                confirmed = self.device.await_policy_acknowledgement()
                elapsed += self.device.now_s-started
                if (confirmed.policy_hash != activation.policy_hash
                        or not confirmed.activation_acknowledgement
                        or confirmed.lifecycle_state is not PolicyLifecycleState.CONFIRMED):
                    raise ValueError("rollback acknowledgement/hash mismatch")
                ids.append(confirmed.policy_id)
                hashes.append(confirmed.policy_hash)
                acknowledgements.append(confirmed.activation_acknowledgement)
                applied.append(confirmed.policy_hash)
            except (ValueError, RuntimeError) as error:
                transaction_status = TransactionRestorationStatus.FAILED
                transaction_reason = str(error)
                violations.append(f"rollback transaction failed: {error}")
                break
        expected_hash = stable_hash(target)
        observed_hash = self.device.confirmed_policy.policy_hash
        if observed_hash != expected_hash:
            transaction_status = TransactionRestorationStatus.FAILED
            transaction_reason = "final active hash differs from the selected rollback target composition"
            violations.append(f"rollback transaction failed: {transaction_reason}")

        evidence_batches: list[QECObservationBatch] = []
        validation = self.device.acquire(validation_cycles)
        evidence_batches.append(validation)
        observed_ci99 = wilson_interval(
            validation.detector_events, validation.detector_exposures,
            z=2.5758293035489004)
        physical_status = PhysicalRestorationStatus.NOT_EVALUATED
        expected_interval = None
        prediction = None
        verification_status = RollbackVerificationStatus.TRANSACTION_FAILED
        sequential_evidence = ()
        regional_observed_ci99: Mapping[str, tuple[float, float]] = {}
        if transaction_status is TransactionRestorationStatus.CONFIRMED:
            conditioned_at = (pre_rollback_observation.records[-1].device_timestamp_s
                              if pre_rollback_observation.records else self.device.now_s)
            prediction = self._restoration_prediction(
                control_output, snapshot, target, conditioned_at)
            expected_interval = (0.0, prediction.upper_noninferiority_bound)
            regional_detectors = {
                region: tuple(detector.detector_id
                              for detector in self.device.circuit.detectors
                              if detector.region_id == region)
                for region in (request.affected_regions or self._all_regions())}
            alpha = .01/max(1, self.config.rollback_sequential_max_batches)
            sequential_z = NormalDist().inv_cdf(1-alpha/2)
            verification = verify_restoration_evidence(
                prediction, evidence_batches,
                regional_detectors=regional_detectors,
                decisive_z=sequential_z)
            if prediction.validity_reasons:
                verification = RollbackVerification(
                    RollbackVerificationStatus.PHYSICAL_RESTORATION_INCONCLUSIVE,
                    verification.observed_rate, verification.observed_ci99,
                    verification.expected_upper,
                    verification.regional_observed_ci99,
                    verification.evidence,
                    "current-state restoration prediction is invalid: "
                    + "; ".join(prediction.validity_reasons))
            while (verification.status is
                   RollbackVerificationStatus.PHYSICAL_RESTORATION_INCONCLUSIVE
                   and len(evidence_batches) < self.config.rollback_sequential_max_batches
                   and not prediction.validity_reasons):
                validation = self.device.acquire(
                    validation_cycles)
                evidence_batches.append(validation)
                verification = verify_restoration_evidence(
                    prediction, evidence_batches,
                    regional_detectors=regional_detectors,
                    decisive_z=sequential_z)
            verification_status = verification.status
            sequential_evidence = verification.evidence
            regional_observed_ci99 = verification.regional_observed_ci99
            observed_ci99 = verification.observed_ci99
            decision = self._loop.supervisor.verify_rollback_evidence(
                verification, self.device.now_s)
            authorizations.append(decision)
            if verification_status is RollbackVerificationStatus.PHYSICAL_RESTORATION_VERIFIED:
                physical_status = PhysicalRestorationStatus.VALIDATED
            elif verification_status is RollbackVerificationStatus.PHYSICAL_RESTORATION_FAILED:
                physical_status = PhysicalRestorationStatus.FAILED
                message = (
                    "rollback transaction confirmed but independent physical "
                    "safety/restoration validation failed")
                physical_failures.append(message)
                self._rejected_rollback_targets.add(snapshot.policy_id)
                self.request_reentry(
                    ReentryReason.FAILED_ROLLBACK, interval=self._interval,
                    scope=RecoveryScope.GLOBAL,
                    triggering_evidence={
                        "target_policy_id": snapshot.policy_id,
                        "observed_ci99": observed_ci99,
                        "expected_interval": expected_interval,
                    }, common_mode_probability=1.0,
                    unknown_model_probability=1.0)
            else:
                # Uncertainty is not falsification.  Hold the restored policy and enter
                # explicit recalibration without recording a lifecycle/physical failure.
                self.request_reentry(
                    ReentryReason.OOD_RECALIBRATION, interval=self._interval,
                    scope=RecoveryScope.GLOBAL,
                    triggering_evidence={
                        "target_policy_id": snapshot.policy_id,
                        "verification_status": verification_status.value,
                        "sequential_batches": len(evidence_batches),
                    }, common_mode_probability=1.0,
                    unknown_model_probability=1.0)
        else:
            self.request_reentry(
                ReentryReason.FAILED_ROLLBACK, interval=self._interval,
                scope=RecoveryScope.GLOBAL,
                triggering_evidence={"transaction_reason": transaction_reason},
                common_mode_probability=1.0, unknown_model_probability=1.0)
        reason = (transaction_reason if transaction_status is TransactionRestorationStatus.FAILED
                  else "transaction confirmed; physical restoration validated"
                  if physical_status is PhysicalRestorationStatus.VALIDATED
                  else "transaction confirmed; physical restoration failed"
                  if physical_status is PhysicalRestorationStatus.FAILED
                  else "transaction confirmed; physical restoration inconclusive")
        total_events = sum(item.detector_events for item in evidence_batches)
        total_exposures = sum(item.detector_exposures for item in evidence_batches)
        return validation, RollbackOutcome(
            "rollback-outcome.v2", request.request_id, snapshot.policy_id,
            snapshot.policy_hash, request.scope, request.affected_regions,
            tuple(scope_controls), transaction_status, physical_status,
            tuple(ids), tuple(hashes), tuple(acknowledgements), expected_hash,
            observed_hash, expected_interval, self.config.qec_operability_rate_limit,
            total_events/max(1, total_exposures), observed_ci99,
            deterministic_hash(tuple(item.batch_id for item in evidence_batches)),
            sum(item.cycles for item in evidence_batches), attempt, elapsed, reason,
            verification_status=verification_status,
            restoration_prediction=prediction,
            sequential_evidence=sequential_evidence,
            regional_observed_ci99=regional_observed_ci99)

    def run_interval(self, cycles: int, *, interval: int | None = None) -> IntegratedIntervalResult:
        if cycles <= 0:
            raise ValueError("cycles must be positive")
        timing_recorder = IntervalTimingRecorder()
        index = self._interval if interval is None else interval
        if (self._pending_reentry is None and self._bootstrap_result is not None
                and any(self.device.now_s > record.validity_until_s
                        for record in self._bootstrap_result.parameter_registry.values())):
            self.request_reentry(
                ReentryReason.OOD_RECALIBRATION, interval=index,
                scope=RecoveryScope.GLOBAL,
                triggering_evidence={"expired_parameter_registry": True},
                common_mode_probability=1.0,
                unknown_model_probability=1.0)
        bootstrap_reason = None
        bootstrap = None
        handled_reentry_request = self._pending_reentry_request
        regional_recovery: RegionalRecoveryOutcome | None = None
        recovery_decisions: tuple[SupervisorDecision, ...] = ()
        diagnostic_shots = 0
        bootstrap_qec_cycles = 0
        diagnostic_downtime = 0.0
        actuation_acknowledgement_s = 0.0
        online_recovery_causes = {
            ReentryReason.OOD_RECALIBRATION, ReentryReason.FAILED_ROLLBACK}
        if (self._pending_reentry_request is not None
                and self._pending_reentry_request.cause in online_recovery_causes
                and self._loop is not None and self._bootstrap_result is not None):
            request = self._pending_reentry_request
            if request.scope is RecoveryScope.REGIONAL:
                regional_recovery, recovery_decisions = self._run_regional_recovery(request)
            else:
                regional_recovery, recovery_decisions = self._run_global_recovery(request)
            if regional_recovery is not None:
                diagnostic_shots += regional_recovery.diagnostic_shots
                bootstrap_qec_cycles += regional_recovery.qec_cycles
                diagnostic_downtime += regional_recovery.diagnostic_downtime_s
                actuation_acknowledgement_s += regional_recovery.actuation_acknowledgement_s
        if self._pending_reentry is not None or self._loop is None or self._bootstrap_result is None:
            (bootstrap_reason, bootstrap, bootstrap_shots,
             bootstrap_cycles, bootstrap_downtime) = self._run_bootstrap()
            diagnostic_shots += bootstrap_shots
            bootstrap_qec_cycles += bootstrap_cycles
            diagnostic_downtime += bootstrap_downtime
            actuation_acknowledgement_s += self._last_bootstrap_actuation_s
        assert self._loop is not None and self._bootstrap_result is not None

        causal_was_cached = self._feedback is not None
        if self._feedback is not None:
            causal = self._feedback
        else:
            kernel_started = time.perf_counter_ns()
            causal = self.device.acquire(cycles)
            timing_recorder.add_host_kernel(
                (time.perf_counter_ns()-kernel_started)/1e9)
        output = self._loop.step(
            causal,
            residual_learning_requested=self.config.enable_residual_rl,
            residual_health=self._residual_health(),
            bootstrap_health=self._bootstrap_health(),
            familiar_policy_cache=self._active_familiar_cache(),
        )
        for stage, duration in output.compute_timings_s.items():
            timing_recorder.add_compute_duration(stage, duration)
        authorizations: list[SupervisorDecision] = [
            *recovery_decisions, output.supervisor]
        candidates: list[ResidualCandidate] = []
        candidate_batches: list[QECObservationBatch] = []
        observations: list[CandidateObservation] = []
        applied: list[str] = []
        violations: list[str] = []
        physical_failures: list[str] = []
        rollback_outcomes: list[RollbackOutcome] = []
        rollback_count = 0
        package = output.proposed_control

        if package is not None and output.supervisor.authorization is Authorization.APPROVED:
            try:
                policy_hash, ack_s, _ = self._apply_bound_package(package)
                applied.append(policy_hash)
                actuation_acknowledgement_s += ack_s
            except ValueError as error:
                violations.append(f"authorized Stage-5 action rejected by device: {error}")
                failure = self._loop.supervisor.tick(SupervisorInput(
                    self.device.now_s, (self._bootstrap_health(),), hard_invariant_failed=True))
                authorizations.append(failure)
        residual_result: ResidualRLResult | None = None
        residual_gate_decision: ResidualGateDecision | None = None
        shadow_validation_cycles = 0
        if self.config.enable_residual_rl and package is not None:
            gate_evidence = self._residual_gate_evidence(output, causal)
            if self.config.residual_activation_mode == "always_on":
                residual_gate_decision = ResidualGateDecision(
                    ResidualRLDisposition.ACTIVE, True,
                    ("development ablation: predeclared always-on residual learner",),
                    gate_evidence)
            else:
                residual_gate_decision = self._residual_gate.evaluate(gate_evidence)
            if (residual_gate_decision.disposition is ResidualRLDisposition.DEACTIVATED
                    and self._residual is not None):
                prior_model = f"residual-policy.v{self._residual.policy.version}"
                self._residual.deactivate()
                if (self._loop.supervisor.model_lifecycle.get(prior_model)
                        in {ModelLifecycle.CANDIDATE, ModelLifecycle.SHADOW,
                            ModelLifecycle.VALIDATED, ModelLifecycle.PROMOTED}):
                    self._loop.supervisor.set_model_lifecycle(
                        prior_model, ModelLifecycle.QUARANTINED)
        if (self.config.enable_residual_rl and package is not None
                and output.supervisor.mode is OperatingMode.RESIDUAL_LEARNING
                and output.supervisor.authorization is Authorization.APPROVED
                and residual_gate_decision is not None
                and residual_gate_decision.eligible):
            compute_started = time.perf_counter_ns()
            controller = self._ensure_residual(package)
            proposed_candidates = controller.propose(
                package, candidate_count=min(
                    self.config.residual_candidate_count,
                    controller.suggest_candidate_count(
                        self._residual_result.gradient_snr
                        if self._residual_result else None)))
            timing_recorder.add_compute(
                "stage6_residual_proposal", compute_started,
                time.perf_counter_ns())
            reserved_damage = self._residual_result.cumulative_damage if self._residual_result else 0.0
            for unbound_candidate in proposed_candidates:
                baseline_policy_id = f"{package.policy_id}:candidate-reference:{unbound_candidate.candidate_id}"
                try:
                    # The reference is made controller-confirmed before projection and
                    # candidate validation.  This is the permanent fix for the historic
                    # pending-MPC/intervention-probe reference race.
                    self.device.apply_policy(
                        package.action, policy_id=baseline_policy_id,
                        supervisor_authorization=package.supervisor_authorization)
                    ack_started = self.device.now_s
                    baseline_activation = self.device.await_policy_acknowledgement()
                    actuation_acknowledgement_s += self.device.now_s-ack_started
                except ValueError as error:
                    violations.append(f"candidate reference baseline rejected by device: {error}")
                    continue
                creation_state_id = stable_hash({
                    "controller_state_hash": self.device.controller_state_hash,
                    "timestamp_s": self.device.now_s,
                    "reference_policy_id": baseline_activation.policy_id,
                })
                candidate = bind_candidate_lifecycle(
                    unbound_candidate, reference_policy_id=baseline_activation.policy_id,
                    reference_policy_hash=baseline_activation.policy_hash,
                    created_from_state_id=creation_state_id,
                    controller_state_hash=self.device.controller_state_hash,
                )
                decision = self._loop.supervisor.authorize_residual_candidate(
                    package, candidate, self.device.now_s, cumulative_damage=reserved_damage)
                authorizations.append(decision)
                if decision.authorization is not Authorization.APPROVED:
                    continue
                candidate = replace(candidate, supervisor_authorization=decision.decision_id)
                try:
                    activation_started = self.device.now_s
                    activation = self.device.apply_policy(
                        candidate.full_control, policy_id=candidate.policy_id,
                        candidate_id=candidate.candidate_id, perturbation=candidate.residual,
                        reference_policy_id=candidate.reference_policy_id,
                        reference_policy_hash=candidate.reference_policy_hash,
                        created_from_state_id=candidate.created_from_state_id,
                        expected_activation_state_id=candidate.expected_activation_state_id,
                        supervisor_authorization=candidate.supervisor_authorization)
                    confirmed_candidate = self.device.await_policy_acknowledgement()
                    actuation_acknowledgement_s += self.device.now_s-activation_started
                    if (confirmed_candidate.policy_hash != activation.policy_hash
                            or not confirmed_candidate.activation_acknowledgement):
                        raise ValueError("candidate activation acknowledgement/hash mismatch")
                    applied.append(confirmed_candidate.policy_hash)
                except ValueError as error:
                    violations.append(f"authorized residual candidate rejected by device: {error}")
                    continue
                # Candidate scoring consumes aggregate detector counts only.  Omitting
                # raw per-cycle records activates the numerically equivalent simulator
                # kernel without changing samples, budgets, or lifecycle semantics.
                kernel_started = time.perf_counter_ns()
                batch = self.device.acquire(
                    self.config.residual_candidate_cycles, retain_records=False)
                timing_recorder.add_host_kernel(
                    (time.perf_counter_ns()-kernel_started)/1e9)
                candidate_batches.append(batch)
                observations.append(CandidateObservation(
                    candidate.candidate_id,
                    {detector: events / max(1, exposure)
                     for detector, (events, exposure) in batch.detector_counts.items()},
                    {detector: exposure for detector, (_, exposure) in batch.detector_counts.items()},
                    logical_risk=batch.logical_failures / max(1, batch.cycles),
                    regime_id="online", context_id=batch.context.context_id,
                    model_version="joint.v2", observed_at_s=self.device.now_s,
                ))
                candidates.append(candidate)
                reserved_damage += candidate.predicted_damage
                if (self.config.candidate_elimination_z is not None
                        and len(observations) >= controller.config.minimum_candidates
                        and len(observations) % 2 == 0):
                    left_batch, right_batch = candidate_batches[-2:]
                    left_rate = left_batch.detector_rate
                    right_rate = right_batch.detector_rate
                    standard_error = math.sqrt(
                        left_rate*(1-left_rate)/max(1, left_batch.detector_exposures)
                        + right_rate*(1-right_rate)/max(1, right_batch.detector_exposures))
                    ranking_z = abs(left_rate-right_rate)/max(standard_error, 1e-12)
                    if ranking_z >= self.config.candidate_elimination_z:
                        # A complete antithetic pair is always retained.  The stop
                        # changes the number of directions, never the evidence per
                        # candidate, and only fires after predeclared ranking confidence.
                        break
            if observations:
                compute_started = time.perf_counter_ns()
                previous_model = f"residual-policy.v{controller.policy.version}"
                if self._loop.supervisor.model_lifecycle.get(previous_model) is ModelLifecycle.CANDIDATE:
                    self._loop.supervisor.set_model_lifecycle(previous_model, ModelLifecycle.SHADOW)
                conditional = self.config.residual_activation_mode == "conditional"
                residual_result = controller.update(
                    package, observations, current_regime="online",
                    current_context=causal.context.context_id,
                    current_model_version="joint.v2", commit=not conditional,
                    gate_decision=residual_gate_decision)
                current_model = f"residual-policy.v{residual_result.policy_version}"
                if current_model not in self._loop.supervisor.model_lifecycle:
                    self._loop.supervisor.set_model_lifecycle(current_model, ModelLifecycle.CANDIDATE)
                combined = self._combined_package(package, residual_result.policy_mean)
                current = self.device.confirmed_policy
                combined = bind_policy_lifecycle(
                    combined, policy_id=f"residual-mean:{index}",
                    reference_policy_id=current.policy_id,
                    reference_policy_hash=current.policy_hash,
                    created_from_state_id=(candidate_batches[-1].physical_state_id
                                           if candidate_batches else causal.physical_state_id),
                    controller_state_hash=self.device.controller_state_hash,
                    reference_controls=current.controls,
                    reference_timestamp_s=self.device.now_s,
                )
                decision = self._loop.supervisor.authorize_control(combined, self.device.now_s)
                authorizations.append(decision)
                promote = not conditional
                if conditional and decision.authorization is Authorization.APPROVED:
                    # Independently score the live predictive baseline and proposed
                    # residual mean.  Candidate microbatches are never reused as the
                    # promotion outcome.
                    try:
                        base_activation = self.device.apply_policy(
                            package.action, policy_id=f"shadow-baseline:{index}",
                            supervisor_authorization=package.supervisor_authorization)
                        self.device.await_policy_acknowledgement()
                        baseline_batch = self.device.acquire(
                            self.config.residual_candidate_cycles,
                            retain_records=False)
                        shadow_validation_cycles += baseline_batch.cycles
                        combined = bind_policy_lifecycle(
                            combined, policy_id=f"shadow-residual:{index}",
                            reference_policy_id=base_activation.policy_id,
                            reference_policy_hash=base_activation.policy_hash,
                            created_from_state_id=baseline_batch.physical_state_id,
                            controller_state_hash=self.device.controller_state_hash,
                            reference_controls=base_activation.controls,
                            reference_timestamp_s=self.device.now_s)
                        shadow_decision = self._loop.supervisor.authorize_control(
                            combined, self.device.now_s)
                        authorizations.append(shadow_decision)
                        if shadow_decision.authorization is not Authorization.APPROVED:
                            raise ValueError(shadow_decision.reason)
                        combined = replace(
                            combined,
                            supervisor_authorization=shadow_decision.decision_id)
                        self._apply_bound_package(combined)
                        shadow_batch = self.device.acquire(
                            self.config.residual_candidate_cycles,
                            retain_records=False)
                        shadow_validation_cycles += shadow_batch.cycles
                        baseline_rate = baseline_batch.detector_rate
                        shadow_rate = shadow_batch.detector_rate
                        gain = baseline_rate-shadow_rate
                        se = math.sqrt(
                            max(baseline_rate*(1-baseline_rate), 1e-12)
                            / max(1, baseline_batch.detector_exposures)
                            + max(shadow_rate*(1-shadow_rate), 1e-12)
                            / max(1, shadow_batch.detector_exposures))
                        probability = NormalDist().cdf(gain/max(se, 1e-12))
                        promote = self._residual_gate.record_shadow_outcome(
                            gain=gain,
                            probability_positive_value=probability,
                            gradient_snr=residual_result.gradient_snr)
                        shadow = ShadowValidation(
                            baseline_rate, shadow_rate,
                            baseline_batch.detector_exposures,
                            shadow_batch.detector_exposures, gain, se,
                            probability, promote,
                            ("positive independent value and gradient evidence"
                             if promote else
                             "shadow update did not establish positive value"))
                        residual_result = replace(
                            residual_result, shadow_validation=shadow)
                    except (ValueError, RuntimeError) as error:
                        promote = False
                        residual_result = replace(
                            residual_result,
                            invalidity_reasons=residual_result.invalidity_reasons
                            + (f"shadow validation failed: {error}",))
                if conditional:
                    if promote:
                        controller.commit_shadow(residual_result)
                        residual_result = replace(residual_result, committed=True)
                        self._loop.supervisor.set_model_lifecycle(
                            current_model, ModelLifecycle.SHADOW)
                        self._loop.supervisor.set_model_lifecycle(
                            current_model, ModelLifecycle.VALIDATED,
                            held_out_passed=True)
                        self._loop.supervisor.set_model_lifecycle(
                            current_model, ModelLifecycle.PROMOTED,
                            held_out_passed=True)
                    else:
                        if self._residual_gate.deactivated:
                            controller.deactivate()
                            if self._loop.supervisor.model_lifecycle.get(current_model) in {
                                    ModelLifecycle.CANDIDATE, ModelLifecycle.SHADOW}:
                                self._loop.supervisor.set_model_lifecycle(
                                    current_model, ModelLifecycle.QUARANTINED)
                        # Restore the currently authorised Stage-5 predictive policy;
                        # never a stale historical controller state.
                        try:
                            self.device.apply_policy(
                                package.action, policy_id=f"shadow-abstain:{index}",
                                supervisor_authorization=package.supervisor_authorization)
                            self.device.await_policy_acknowledgement()
                        except ValueError as error:
                            violations.append(
                                f"current predictive policy restoration failed: {error}")
                self._residual_result = residual_result
                timing_recorder.add_compute(
                    "stage6_residual_update_and_stage7_authorization",
                    compute_started, time.perf_counter_ns())
                if (not conditional and decision.authorization is Authorization.APPROVED):
                    combined = replace(combined, supervisor_authorization=decision.decision_id)
                    try:
                        policy_hash, ack_s, _ = self._apply_bound_package(combined)
                        applied.append(policy_hash)
                        actuation_acknowledgement_s += ack_s
                    except ValueError as error:
                        violations.append(f"authorized residual mean rejected by device: {error}")
                if residual_result.fallback_requested:
                    self._loop.supervisor.set_model_lifecycle(current_model, ModelLifecycle.QUARANTINED)
                    # Persistent residuals invalidate model authority; Stage 0 is reserved
                    # for the subsequent explicit OOD/recalibration transition.
                    self.request_reentry(ReentryReason.OOD_RECALIBRATION)

        if output.supervisor.mode is OperatingMode.UNKNOWN_EVENT:
            self._request_from_control(output, index)
        rollback_required = any(decision.rollback_required for decision in authorizations)
        if rollback_required:
            request = self._pending_reentry_request
            if request is None:
                request = self.request_reentry(
                    ReentryReason.FAILED_ROLLBACK, interval=index,
                    scope=RecoveryScope.GLOBAL,
                    triggering_evidence={
                        "authorization_reasons": tuple(
                            decision.reason for decision in authorizations
                            if decision.rollback_required),
                    }, common_mode_probability=1.0,
                    unknown_model_probability=1.0)
            feedback, rollback_outcome = self._rollback(
                request, candidate_batches[-1] if candidate_batches else causal,
                output,
                authorizations, applied, violations, physical_failures)
            rollback_outcomes.append(rollback_outcome)
            rollback_count = 1
            actuation_acknowledgement_s += rollback_outcome.bounded_downtime_s
        else:
            kernel_started = time.perf_counter_ns()
            feedback = self.device.acquire(cycles)
            timing_recorder.add_host_kernel(
                (time.perf_counter_ns()-kernel_started)/1e9)

        if not math.isfinite(feedback.detector_rate):
            violations.append("feedback telemetry has no detector exposure")
            self.request_reentry(
                ReentryReason.QEC_OPERABILITY_LOST, interval=index,
                scope=RecoveryScope.GLOBAL,
                triggering_evidence={"feedback_batch_id": feedback.batch_id,
                                     "detector_exposures": feedback.detector_exposures},
                common_mode_probability=1.0,
                unknown_model_probability=1.0)
        elif feedback.detector_rate > self.config.qec_operability_rate_limit:
            self.request_reentry(
                ReentryReason.QEC_OPERABILITY_LOST, interval=index,
                scope=RecoveryScope.GLOBAL,
                triggering_evidence={
                    "feedback_batch_id": feedback.batch_id,
                    "detector_rate": feedback.detector_rate,
                    "absolute_limit": self.config.qec_operability_rate_limit,
                }, common_mode_probability=1.0,
                unknown_model_probability=1.0)
        self._feedback = feedback
        if (not rollback_required and not violations and not physical_failures
                and feedback.policy_activation.policy_hash
                == self.device.confirmed_policy.policy_hash):
            self._register_snapshot(
                feedback, evidence_phase="online-independent-feedback")
            self._register_familiar_policies(output, feedback)

        stage0_label = bootstrap_reason.value if bootstrap_reason else "validated_cache"
        recovery_path = (() if regional_recovery is None else (
            f"recovery:online_disturbance_aware_{regional_recovery.request.scope.value}",))
        stage_path = (
            f"stage0:{stage0_label}", *recovery_path,
            "stage1:telemetry", "stage2:physical_inference", "stage3:joint_dynamics_hdfa",
            "stage4:forecast", "stage5:mpc", "stage6:residual_rl",
            "stage7:authorization_lifecycle", "device:atomic_apply", "stage1:feedback",
        )
        replay_hash = deterministic_hash({
            "interval": index,
            "bootstrap": bootstrap.replay_hash if bootstrap else self._bootstrap_result.replay_hash,
            "causal": causal.batch_id,
            "loop": output.replay_hash,
            "candidates": [candidate.candidate_id for candidate in candidates],
            "observations": [(item.candidate_id, item.detector_losses) for item in observations],
            "candidate_batches": [item.batch_id for item in candidate_batches],
            "feedback": feedback.batch_id,
            "authorizations": [item.decision_id for item in authorizations],
            "applied": applied,
            "violations": violations,
            "physical_rollback_failures": physical_failures,
            "reentry_request": deterministic_hash(
                self._pending_reentry_request) if self._pending_reentry_request else None,
            "regional_recovery": (
                regional_recovery.replay_hash if regional_recovery else None),
            "rollback_outcomes": [deterministic_hash(item) for item in rollback_outcomes],
        })
        candidate_cycle_count = len(observations)*self.config.residual_candidate_cycles
        newly_causal_cycles = 0 if causal_was_cached else causal.cycles
        qec_acquisition_cycles = (
            bootstrap_qec_cycles + newly_causal_cycles
            + candidate_cycle_count + shadow_validation_cycles + feedback.cycles)
        timing = timing_recorder.finalize(
            qec_acquisition_s=(
                qec_acquisition_cycles*self.device.config.cycle_period_s),
            diagnostic_downtime_s=diagnostic_downtime,
            actuation_acknowledgement_s=actuation_acknowledgement_s,
            complete=True)
        interval_request = handled_reentry_request or self._pending_reentry_request
        result = IntegratedIntervalResult(
            schema_version="product-loop.v2", interval=index,
            bootstrap_reason=bootstrap_reason, bootstrap=bootstrap,
            causal_observation=causal, control=output,
            residual_candidates=tuple(candidates),
            candidate_batches=tuple(candidate_batches),
            candidate_observations=tuple(observations),
            residual_result=residual_result, feedback_observation=feedback,
            authorization_log=tuple(authorizations), stage_path=stage_path,
            applied_policy_hashes=tuple(applied),
            lifecycle_violations=tuple(violations),
            rollback_count=rollback_count, bootstrap_count=self._bootstrap_count,
            bootstrap_qec_cycles=bootstrap_qec_cycles,
            candidate_cycles=candidate_cycle_count,
            diagnostic_shots=diagnostic_shots,
            diagnostic_downtime_s=diagnostic_downtime,
            newly_acquired_telemetry_cycles=newly_causal_cycles,
            replay_hash=replay_hash, reentry_request=interval_request,
            regional_recovery=regional_recovery,
            rollback_outcomes=tuple(rollback_outcomes),
            physical_rollback_failures=tuple(physical_failures),
            recovery_count=self._recovery_count, timing=timing,
            residual_gate_decision=residual_gate_decision,
        )
        self._interval = index + 1
        return result
