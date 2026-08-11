"""Structured disturbance-aware re-entry and rollback evidence contracts.

Initial Stage 0 remains the stationary bootstrap.  The scoped calibrator below is an
explicit online recovery phase: it may run only after a disturbance is armed, charges
all characterization/QEC/actuation cost, and cannot weaken Stage-0 safety gates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from statistics import NormalDist
from typing import Mapping, Sequence

from hdfa_rl_suite.common import deterministic_hash
from hdfa_rl_suite.simulator import QECObservationBatch, ScalableQECDevice
from hdfa_rl_suite.stage0 import ScalableBootstrapConfig
from hdfa_rl_suite.stage0.schema import stable_hash


class ReentryReason(str, Enum):
    COLD_START = "cold_start"
    QEC_OPERABILITY_LOST = "qec_operability_lost"
    MAJOR_HARDWARE_RECONFIGURATION = "major_hardware_reconfiguration"
    FAILED_ROLLBACK = "failed_rollback"
    OOD_RECALIBRATION = "ood_recalibration"


class RecoveryScope(str, Enum):
    REGIONAL = "regional"
    GLOBAL = "global"


class TransactionRestorationStatus(str, Enum):
    NOT_REQUESTED = "not_requested"
    CONFIRMED = "confirmed"
    FAILED = "failed"


class PhysicalRestorationStatus(str, Enum):
    NOT_EVALUATED = "not_evaluated"
    VALIDATED = "validated"
    FAILED = "failed"


class RollbackVerificationStatus(str, Enum):
    """Joint transaction/physical result; uncertainty is not a failure."""

    TRANSACTION_FAILED = "transaction_failed"
    PHYSICAL_RESTORATION_FAILED = "physical_restoration_failed"
    PHYSICAL_RESTORATION_INCONCLUSIVE = "physical_restoration_inconclusive"
    PHYSICAL_RESTORATION_VERIFIED = "physical_restoration_verified"


@dataclass(frozen=True)
class RestorationPrediction:
    """Detector-level prediction at the *current* inferred state and restored policy.

    The historical snapshot identifies the controls to restore.  It does not supply the
    expected detector performance: that is recomputed from the causal Stage-2/3 belief.
    ``upper_noninferiority_bound`` includes declared model/discretisation uncertainty
    and an irreducible noise floor, but never simulator latent truth.
    """

    schema_version: str
    conditioned_at_s: float
    target_policy_id: str
    target_policy_hash: str
    elapsed_since_conditioning_s: float
    detector_expected_rates: Mapping[str, float]
    detector_rate_stddev: Mapping[str, float]
    regional_upper_bounds: Mapping[str, float]
    expected_global_rate: float
    upper_noninferiority_bound: float
    absolute_qec_operability_ceiling: float
    model_uncertainty: float
    drift_uncertainty: float
    irreducible_noise_floor: float
    finite_shot_uncertainty_is_observation_only: bool = True
    validity_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class RestorationEvidenceBatch:
    batch_id: str
    cycles: int
    detector_events: int
    detector_exposures: int
    detector_counts: Mapping[str, tuple[int, int]]
    cumulative_rate: float
    cumulative_ci99: tuple[float, float]
    status: RollbackVerificationStatus
    reason: str


@dataclass(frozen=True)
class RollbackVerification:
    status: RollbackVerificationStatus
    observed_rate: float | None
    observed_ci99: tuple[float, float] | None
    expected_upper: float | None
    regional_observed_ci99: Mapping[str, tuple[float, float]]
    evidence: tuple[RestorationEvidenceBatch, ...]
    reason: str


@dataclass(frozen=True)
class ReentryRequest:
    schema_version: str
    request_id: str
    cause: ReentryReason
    requested_at_s: float
    requested_interval: int
    triggering_evidence: Mapping[str, object]
    scope: RecoveryScope
    affected_regions: tuple[str, ...]
    affected_controls: tuple[str, ...]
    boundary_detectors: tuple[str, ...]
    common_mode_probability: float
    unknown_model_probability: float
    reference_policy_id: str
    reference_policy_hash: str
    controller_state_id: str
    escalation_count: int
    deadline_s: float
    required_gates: tuple[str, ...]
    coalesced_request_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.request_id or not self.reference_policy_id or not self.reference_policy_hash:
            raise ValueError("re-entry requests require request and policy identities")
        if self.scope is RecoveryScope.REGIONAL and (
                not self.affected_regions or not self.affected_controls):
            raise ValueError("regional re-entry requires positively identified scope")
        if self.deadline_s < self.requested_at_s:
            raise ValueError("re-entry deadline cannot precede the request")
        if not 0 <= self.common_mode_probability <= 1:
            raise ValueError("common-mode probability must lie in [0,1]")
        if not 0 <= self.unknown_model_probability <= 1:
            raise ValueError("unknown-model probability must lie in [0,1]")


@dataclass(frozen=True)
class ValidatedPolicySnapshot:
    schema_version: str
    policy_id: str
    policy_hash: str
    controls: Mapping[str, float]
    validated_at_s: float
    scope: RecoveryScope
    regions: tuple[str, ...]
    detector_rate: float
    detector_rate_ci95: tuple[float, float]
    detector_exposures: int
    validation_batch_id: str
    controller_state_hash: str
    evidence_phase: str
    qec_operability_ceiling: float
    independently_validated: bool = True


@dataclass(frozen=True)
class RollbackOutcome:
    schema_version: str
    request_id: str
    target_policy_id: str | None
    target_policy_hash: str | None
    target_scope: RecoveryScope
    affected_regions: tuple[str, ...]
    affected_controls: tuple[str, ...]
    transaction_status: TransactionRestorationStatus
    physical_status: PhysicalRestorationStatus
    applied_policy_ids: tuple[str, ...]
    applied_policy_hashes: tuple[str, ...]
    acknowledgement_ids: tuple[str, ...]
    expected_final_hash: str | None
    observed_final_hash: str | None
    expected_detector_rate_interval: tuple[float, float] | None
    absolute_qec_operability_ceiling: float
    observed_detector_rate: float | None
    observed_detector_rate_ci99: tuple[float, float] | None
    validation_batch_id: str | None
    validation_cycles: int
    attempt: int
    bounded_downtime_s: float
    reason: str
    verification_status: RollbackVerificationStatus | None = None
    restoration_prediction: RestorationPrediction | None = None
    sequential_evidence: tuple[RestorationEvidenceBatch, ...] = ()
    regional_observed_ci99: Mapping[str, tuple[float, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class RegionalRecoveryOutcome:
    schema_version: str
    request: ReentryRequest
    passed: bool
    escalated_to_global: bool
    gate_results: Mapping[str, bool]
    invalid_reasons: tuple[str, ...]
    affected_detectors: tuple[str, ...]
    boundary_detectors: tuple[str, ...]
    frozen_controls: Mapping[str, float]
    characterized_estimates: Mapping[str, float]
    characterized_variances: Mapping[str, float]
    sensitivity_jacobian: Mapping[str, Mapping[str, float]]
    qec_validation_batch: QECObservationBatch
    qec_detector_rate_ci99: tuple[float, float]
    regional_detector_rate_ci: Mapping[str, tuple[float, float]]
    final_policy_id: str
    final_policy_hash: str
    final_acknowledgement_id: str
    diagnostic_shots: int
    qec_cycles: int
    diagnostic_downtime_s: float
    actuation_acknowledgement_s: float
    evidence_phase: str = "online-disturbance-aware-regional-recovery"

    @property
    def replay_hash(self) -> str:
        return deterministic_hash({
            "request": self.request,
            "passed": self.passed,
            "gates": self.gate_results,
            "invalid": self.invalid_reasons,
            "qec_batch": self.qec_validation_batch.batch_id,
            "policy": (self.final_policy_id, self.final_policy_hash),
        })


def wilson_interval(events: int, exposures: int, *, z: float = 1.959963984540054
                    ) -> tuple[float, float]:
    if exposures <= 0:
        return 0.0, 1.0
    rate = events/exposures
    denominator = 1 + z*z/exposures
    centre = (rate + z*z/(2*exposures))/denominator
    radius = z*math.sqrt(rate*(1-rate)/exposures + z*z/(4*exposures**2))/denominator
    return max(0.0, centre-radius), min(1.0, centre+radius)


def verify_restoration_evidence(
        prediction: RestorationPrediction,
        batches: Sequence[QECObservationBatch],
        *, regional_detectors: Mapping[str, Sequence[str]] | None = None,
        decisive_z: float = 2.5758293035489004,
        equivalence_slack: float = 1e-12) -> RollbackVerification:
    """Classify sequential rollback evidence without turning overlap into failure.

    Failure requires the entire confidence interval to be worse than a current-state
    non-inferiority or operability bound.  Verification requires the entire interval to
    be inside every applicable bound.  All overlap/boundary cases are inconclusive and
    therefore justify another predeclared batch while the safe restored policy remains
    active.
    """
    if not batches:
        return RollbackVerification(
            RollbackVerificationStatus.PHYSICAL_RESTORATION_INCONCLUSIVE,
            None, None, prediction.upper_noninferiority_bound, {}, (),
            "no finite-shot restoration evidence has been acquired")
    events = sum(batch.detector_events for batch in batches)
    exposures = sum(batch.detector_exposures for batch in batches)
    rate = events / exposures if exposures else None
    interval = wilson_interval(events, exposures, z=decisive_z)
    global_bound = min(prediction.upper_noninferiority_bound,
                       prediction.absolute_qec_operability_ceiling)
    region_intervals: dict[str, tuple[float, float]] = {}
    regional_failed = False
    regional_verified = True
    for region, detector_ids in (regional_detectors or {}).items():
        region_events = region_exposures = 0
        for batch in batches:
            for detector in detector_ids:
                count = batch.detector_counts.get(detector, (0, 0))
                region_events += count[0]
                region_exposures += count[1]
        region_interval = wilson_interval(region_events, region_exposures, z=decisive_z)
        region_intervals[region] = region_interval
        bound = min(prediction.regional_upper_bounds.get(region, global_bound),
                    prediction.absolute_qec_operability_ceiling)
        regional_failed = regional_failed or region_interval[0] > bound + equivalence_slack
        regional_verified = regional_verified and region_interval[1] <= bound + equivalence_slack
    global_failed = interval[0] > global_bound + equivalence_slack
    global_verified = interval[1] <= global_bound + equivalence_slack
    if global_failed or regional_failed:
        status = RollbackVerificationStatus.PHYSICAL_RESTORATION_FAILED
        reason = ("credible finite-shot evidence places restored performance above "
                  "a current-state physical-suitability bound")
    elif global_verified and regional_verified:
        status = RollbackVerificationStatus.PHYSICAL_RESTORATION_VERIFIED
        reason = ("current-state physical restoration and regional QEC operability "
                  "are verified at the declared confidence level")
    else:
        status = RollbackVerificationStatus.PHYSICAL_RESTORATION_INCONCLUSIVE
        reason = ("finite-shot interval overlaps a restoration boundary; additional "
                  "sequential evidence is required")
    rows: list[RestorationEvidenceBatch] = []
    cumulative: list[QECObservationBatch] = []
    for batch in batches:
        cumulative.append(batch)
        n_events = sum(item.detector_events for item in cumulative)
        n_exposures = sum(item.detector_exposures for item in cumulative)
        ci = wilson_interval(n_events, n_exposures, z=decisive_z)
        interim = (RollbackVerificationStatus.PHYSICAL_RESTORATION_FAILED
                   if ci[0] > global_bound + equivalence_slack
                   else RollbackVerificationStatus.PHYSICAL_RESTORATION_VERIFIED
                   if ci[1] <= global_bound + equivalence_slack
                   else RollbackVerificationStatus.PHYSICAL_RESTORATION_INCONCLUSIVE)
        rows.append(RestorationEvidenceBatch(
            batch.batch_id, batch.cycles, batch.detector_events,
            batch.detector_exposures, dict(batch.detector_counts),
            n_events/max(1, n_exposures), ci, interim,
            "sequential cumulative boundary classification"))
    return RollbackVerification(status, rate, interval,
                                prediction.upper_noninferiority_bound,
                                region_intervals, tuple(rows), reason)


class DisturbanceAwareScopedCalibrator:
    """Run online scoped characterization with global/boundary sentinels.

    This is never stationary Stage 0.  Global scope means that every control and
    detector is actively re-characterized under the current disturbance context.
    """

    def __init__(self, device: ScalableQECDevice, request: ReentryRequest,
                 config: ScalableBootstrapConfig, *, rollback_available: bool) -> None:
        self.device = device
        self.request = request
        self.config = config
        self.rollback_available = rollback_available
        self._sequence = 0

    def _apply_and_ack(self, controls: Mapping[str, float], label: str):
        reference = self.device.confirmed_policy
        started = self.device.now_s
        activation = self.device.apply_policy(
            controls, policy_id=f"regional-recovery:{self.request.request_id}:{label}:{self._sequence}",
            reference_policy_id=reference.policy_id,
            reference_policy_hash=reference.policy_hash,
            created_from_state_id=stable_hash({
                "request": self.request.request_id,
                "controller_state_hash": self.device.controller_state_hash,
                "timestamp_s": self.device.now_s,
            }),
            supervisor_authorization=f"regional-recovery:{self.request.request_id}")
        self._sequence += 1
        confirmed = self.device.await_policy_acknowledgement()
        if (confirmed.policy_hash != activation.policy_hash
                or not confirmed.activation_acknowledgement):
            raise RuntimeError("regional recovery activation was not atomically acknowledged")
        return confirmed, self.device.now_s-started

    def _slew_to(self, desired: Mapping[str, float], label: str):
        start = dict(self.device.confirmed_policy.controls)
        steps = max((math.ceil(abs(desired[control]-start[control]) /
                               max(self.device.limits.controls[control].max_slew, 1e-12))
                     for control in desired), default=1)
        total_ack = 0.0
        final = self.device.confirmed_policy
        for index in range(1, max(1, steps)+1):
            fraction = index/max(1, steps)
            patch = dict(self.device.confirmed_policy.controls)
            for control, target in desired.items():
                patch[control] = start[control] + fraction*(target-start[control])
            final, elapsed = self._apply_and_ack(patch, f"{label}:{index}")
            total_ack += elapsed
        return final, total_ack

    def run(self) -> RegionalRecoveryOutcome:
        started = self.device.now_s
        global_scope = self.request.scope is RecoveryScope.GLOBAL
        affected_controls = (tuple(self.device.limits.controls) if global_scope
                             else tuple(self.request.affected_controls))
        affected_detectors = tuple(
            detector.detector_id for detector in self.device.circuit.detectors
            if global_scope or detector.region_id in self.request.affected_regions)
        validation_detectors = tuple(dict.fromkeys(
            (*affected_detectors, *self.request.boundary_detectors)))
        unaffected = {
            control: value for control, value in self.device.confirmed_policy.controls.items()
            if control not in affected_controls
        }
        gate_results: dict[str, bool] = {}
        invalid: list[str] = []
        actuation_s = 0.0

        characterization = self.device.characterize_controls(
            affected_controls, shots=self.config.characterization_shots)
        target = dict(characterization.estimates)
        uncertainty_ok = all(
            math.sqrt(characterization.variances[control])
            <= self.config.target_posterior_stddev
            for control in affected_controls)
        gate_results["uncertainty"] = uncertainty_ok
        if not uncertainty_ok:
            invalid.append("uncertainty")
        hardware_ok = all(
            self.device.limits.controls[control].validate(value)
            for control, value in target.items())
        gate_results["hardware_bounds"] = hardware_ok
        if not hardware_ok:
            invalid.append("hardware_bounds")

        final, elapsed = self._slew_to(target, "characterized-target")
        actuation_s += elapsed
        held = self.device.characterize_controls(
            affected_controls, shots=self.config.characterization_shots)
        comparisons = max(1, len(self.device.limits.controls))
        z_limit = NormalDist().inv_cdf(
            1-self.config.block_predictive_familywise_alpha/(2*comparisons))
        heldout_z = max((
            abs(held.estimates[control]-target[control]) /
            math.sqrt(held.variances[control]+characterization.variances[control])
            for control in affected_controls), default=0.0)
        heldout_ok = heldout_z <= z_limit
        gate_results["held_out"] = heldout_ok
        if not heldout_ok:
            invalid.append("held_out")

        sensitivity_cycles = max(64, self.config.validation_cycles//4)
        jacobian: dict[str, dict[str, float]] = {}
        sensitivity_ok = True
        sensitivity_qec_cycles = 0
        sensitivity_tests = max(
            1, len(affected_controls)*len(self.device.circuit.detectors))
        sentinel_z_limit = NormalDist().inv_cdf(
            1-self.config.sensitivity_interference_alpha/(2*sensitivity_tests))
        recovery_policy = dict(final.controls)
        for control in affected_controls:
            bound = self.device.limits.controls[control]
            epsilon = min(
                bound.max_slew/2,
                bound.trust_radius*self.config.sensitivity_fraction,
                recovery_policy[control]-bound.minimum,
                bound.maximum-recovery_policy[control])
            if epsilon <= 0:
                sensitivity_ok = False
                jacobian[control] = {}
                continue
            plus, minus = dict(recovery_policy), dict(recovery_policy)
            plus[control] += epsilon
            minus[control] -= epsilon
            _, elapsed = self._apply_and_ack(plus, f"sensitivity:{control}:plus")
            actuation_s += elapsed
            positive = self.device.acquire(sensitivity_cycles, retain_records=False)
            _, elapsed = self._apply_and_ack(minus, f"sensitivity:{control}:minus")
            actuation_s += elapsed
            negative = self.device.acquire(sensitivity_cycles, retain_records=False)
            _, elapsed = self._apply_and_ack(recovery_policy, f"sensitivity:{control}:restore")
            actuation_s += elapsed
            sensitivity_qec_cycles += 2*sensitivity_cycles
            local: dict[str, float] = {}
            for detector in self.device.detector_control_graph:
                pos_count = positive.detector_counts[detector]
                neg_count = negative.detector_counts[detector]
                pos_rate = pos_count[0]/pos_count[1] if pos_count[1] else math.nan
                neg_rate = neg_count[0]/neg_count[1] if neg_count[1] else math.nan
                slope = ((pos_rate-neg_rate)/(2*epsilon)
                         if math.isfinite(pos_rate) and math.isfinite(neg_rate) else math.nan)
                local[detector] = slope
                se = math.sqrt(
                    max(pos_rate*(1-pos_rate), 1e-12)/max(1, pos_count[1])
                    + max(neg_rate*(1-neg_rate), 1e-12)/max(1, neg_count[1]))
                z_score = (abs(pos_rate-neg_rate)/se
                           if math.isfinite(se) and se > 0 else math.inf)
                if detector not in validation_detectors and z_score > sentinel_z_limit:
                    sensitivity_ok = False
            jacobian[control] = local
        gate_results["sensitivity"] = sensitivity_ok
        if not sensitivity_ok:
            invalid.append("sensitivity")

        validation = self.device.acquire(
            self.config.validation_cycles, retain_records=False)
        global_ci = wilson_interval(
            validation.detector_events, validation.detector_exposures, z=2.5758293035489004)
        qec_ok = global_ci[1] <= self.config.qec_detector_rate_limit
        regional_ci: dict[str, tuple[float, float]] = {}
        local_z = NormalDist().inv_cdf(
            1-self.config.block_predictive_familywise_alpha /
            (2*max(1, len(validation_detectors))))
        for detector in validation_detectors:
            events, exposures = validation.detector_counts[detector]
            interval = wilson_interval(events, exposures, z=local_z)
            regional_ci[detector] = interval
            qec_ok = qec_ok and interval[1] <= self.config.qec_detector_rate_limit
        gate_results["qec"] = qec_ok
        if not qec_ok:
            invalid.append("qec")

        unaffected_ok = all(
            self.device.confirmed_policy.controls[control] == value
            for control, value in unaffected.items())
        gate_results["unaffected_policy_frozen"] = unaffected_ok
        if not unaffected_ok:
            invalid.append("unaffected_policy_frozen")
        gate_results["rollback_available"] = self.rollback_available
        if not self.rollback_available:
            invalid.append("rollback_available")
        gate_results["boundary_validation"] = (
            global_scope or (bool(self.request.boundary_detectors) and all(
                detector in regional_ci for detector in self.request.boundary_detectors)))
        if not gate_results["boundary_validation"]:
            invalid.append("boundary_validation")

        passed = all(gate_results.values())
        confirmed = self.device.confirmed_policy
        return RegionalRecoveryOutcome(
            "regional-recovery.v1", self.request, passed, not passed,
            gate_results, tuple(dict.fromkeys(invalid)), affected_detectors,
            self.request.boundary_detectors, unaffected,
            dict(characterization.estimates), dict(characterization.variances),
            jacobian, validation, global_ci, regional_ci,
            confirmed.policy_id, confirmed.policy_hash,
            confirmed.activation_acknowledgement,
            characterization.shots+held.shots,
            sensitivity_qec_cycles+validation.cycles,
            characterization.downtime_s+held.downtime_s,
            actuation_s,
            ("online-disturbance-aware-global-recovery" if global_scope
             else "online-disturbance-aware-regional-recovery"),
        )


# Compatibility name retained for existing imports and serialized diagnostics.
DisturbanceAwareRegionalCalibrator = DisturbanceAwareScopedCalibrator
