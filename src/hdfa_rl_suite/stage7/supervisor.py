"""Deterministic runtime assurance for the staged calibration architecture."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from hdfa_rl_suite.stage0.schema import HardwareLimits, stable_hash
from hdfa_rl_suite.stage5.schema import PredictiveControlPackage, SolverStatus
from hdfa_rl_suite.stage6.schema import ResidualCandidate
from hdfa_rl_suite.recovery import RollbackVerification, RollbackVerificationStatus

from .schema import (Authorization, BudgetState, DecoderWorkloadAdvisory, DiagnosticOption,
                     DiagnosticPlan, ModelLifecycle, OperatingMode, SupervisorDecision,
                     SupervisorInput, TransitionRecord)


@dataclass(frozen=True)
class SupervisorConfig:
    local_change_entry_probability: float = .95
    broad_ood_threshold: float = .8
    unknown_model_threshold: float = .35
    minimum_dwell_s: float = .1
    maximum_exploration_damage: float = 1.0
    local_change_exit_probability: float = .40
    unknown_model_exit_probability: float = .15
    maximum_mode_dwell_s: float = 30.0
    maximum_diagnostic_downtime_s: float = 10.0
    maximum_diagnostics: int = 8
    logical_risk_limit: float = .25
    correlation_alarm_limit: float = .5


class SupervisoryController:
    """Authorise actions; no learned/inference component can bypass these invariant checks."""

    _exit = {
        OperatingMode.BOOTSTRAP: "validated QEC-operable baseline", OperatingMode.NOMINAL_PREDICTIVE: "forecast invalidity, residual growth, or new event",
        OperatingMode.RESIDUAL_LEARNING: "residual converges, budget exhausts, or forecast invalidates", OperatingMode.LOCAL_RECOVERY: "local posterior/validation recovers",
        OperatingMode.UNKNOWN_EVENT: "common-mode diagnosis or validated fallback", OperatingMode.DIAGNOSTIC: "predeclared decision rule evaluated",
        OperatingMode.DEGRADED: "validated baseline or controlled recovery", OperatingMode.FAIL_SAFE: "manual or watchdog-confirmed safe restoration",
    }

    def __init__(self, limits: HardwareLimits, diagnostics: Sequence[DiagnosticOption] = (), config: SupervisorConfig = SupervisorConfig()) -> None:
        self.limits, self.diagnostics, self.config = limits, tuple(diagnostics), config
        self.mode, self._last_transition_s = OperatingMode.BOOTSTRAP, float("-inf")
        self.audit_log: list[TransitionRecord] = []
        self.decision_log: list[SupervisorDecision] = []
        self.model_lifecycle: dict[str, ModelLifecycle] = {}
        self._diagnostic_downtime_s = 0.0
        self._diagnostics_used = 0
        self._exploration_damage = 0.0

    _permitted_modes = {
        OperatingMode.BOOTSTRAP: frozenset({"bootstrap", "rollback"}),
        OperatingMode.NOMINAL_PREDICTIVE: frozenset({"predictive", "rollback"}),
        OperatingMode.RESIDUAL_LEARNING: frozenset({"predictive", "residual", "rollback"}),
        OperatingMode.LOCAL_RECOVERY: frozenset({"predictive", "diagnostic", "rollback"}),
        OperatingMode.UNKNOWN_EVENT: frozenset({"baseline", "diagnostic", "rollback"}),
        OperatingMode.DIAGNOSTIC: frozenset({"diagnostic", "rollback"}),
        OperatingMode.DEGRADED: frozenset({"baseline", "rollback"}),
        OperatingMode.FAIL_SAFE: frozenset({"rollback"}),
    }

    def _budgets(self) -> BudgetState:
        return BudgetState(self._exploration_damage, self.config.maximum_exploration_damage,
                           self._diagnostic_downtime_s, self.config.maximum_diagnostic_downtime_s,
                           self._diagnostics_used, self.config.maximum_diagnostics)

    def _decision(self, authorization: Authorization, reason: str, transition: TransitionRecord | None,
                  diagnostic: DiagnosticPlan | None, rollback: bool,
                  advisory: DecoderWorkloadAdvisory | None = None) -> SupervisorDecision:
        payload = {"sequence": len(self.decision_log), "mode": self.mode.value, "authorization": authorization.value,
                   "reason": reason, "transition": transition.timestamp_s if transition else None}
        decision = SupervisorDecision(self.mode, authorization, reason, transition, diagnostic, rollback,
                                      dict(self.model_lifecycle), self._budgets(), advisory, stable_hash(payload))
        self.decision_log.append(decision)
        return decision

    def _transition(self, next_mode: OperatingMode, timestamp_s: float, reasons: Sequence[str], *, force: bool = False) -> TransitionRecord | None:
        if next_mode is self.mode or (not force and timestamp_s - self._last_transition_s < self.config.minimum_dwell_s):
            return None
        record = TransitionRecord(timestamp_s, self.mode, next_mode, tuple(reasons), self._exit[next_mode],
                                  {"forced": force}, {})
        self.mode, self._last_transition_s = next_mode, timestamp_s
        self.audit_log.append(record)
        return record

    def _select_diagnostic(self) -> DiagnosticPlan | None:
        candidates = [item for item in self.diagnostics if item.required_for_decision]
        if not candidates:
            return None
        option = max(candidates, key=lambda item: item.value)
        return DiagnosticPlan(option.diagnostic_id, option.value, option.interruption_s, "selected by expected decision-relevant regret reduction per interruption/risk cost")

    def tick(self, state: SupervisorInput) -> SupervisorDecision:
        data_bad = any(item.hard_invalid_data for item in state.health)
        broad_ood = state.broad_ood or state.unknown_model_probability >= self.config.unknown_model_threshold or any(item.ood_score >= self.config.broad_ood_threshold for item in state.health)
        excessive_damage = any(item.exploration_damage >= self.config.maximum_exploration_damage for item in state.health)
        self._exploration_damage = max(self._exploration_damage, max((item.exploration_damage for item in state.health), default=0.))
        if (state.hard_invariant_failed or not state.controller_confirmed or not state.policy_hash_consistent
                or state.human_veto or state.logical_risk >= self.config.logical_risk_limit
                or state.correlation_alarm >= self.config.correlation_alarm_limit):
            transition = self._transition(OperatingMode.FAIL_SAFE, state.timestamp_s, ("hard invariant or controller consistency failure",), force=True)
            advisory = DecoderWorkloadAdvisory("pause-or-increase-protection", "hard safety/logical-risk invariant failed")
            return self._decision(Authorization.ROLLBACK, "hard invariant failure", transition, None, True, advisory)
        if data_bad:
            transition = self._transition(OperatingMode.DEGRADED, state.timestamp_s, ("hard-invalid telemetry blocks updates",), force=True)
            return self._decision(Authorization.REJECTED, "telemetry invalid", transition, None, False)
        if state.bootstrap_required:
            transition = self._transition(OperatingMode.BOOTSTRAP, state.timestamp_s, ("baseline or calibration graph invalid",))
            return self._decision(Authorization.DELAYED, "bootstrap required", transition, None, False)
        recovery_modes = {OperatingMode.LOCAL_RECOVERY, OperatingMode.UNKNOWN_EVENT,
                          OperatingMode.DIAGNOSTIC, OperatingMode.DEGRADED}
        if (self.mode in recovery_modes and state.timestamp_s - self._last_transition_s > self.config.maximum_mode_dwell_s):
            transition = self._transition(OperatingMode.FAIL_SAFE, state.timestamp_s,
                                          ("maximum recovery-mode dwell exceeded",), force=True)
            return self._decision(Authorization.ROLLBACK, "recovery deadline exceeded", transition, None, True)
        if self.mode is OperatingMode.UNKNOWN_EVENT and (broad_ood or state.unknown_model_probability > self.config.unknown_model_exit_probability):
            return self._decision(Authorization.ROLLBACK, "unknown-event hysteresis holds fallback", None, None, True,
                                  DecoderWorkloadAdvisory("use-validated-decoder-fallback", "model authority remains revoked"))
        if self.mode is OperatingMode.LOCAL_RECOVERY and state.local_change_probability > self.config.local_change_exit_probability:
            return self._decision(Authorization.DELAYED, "local-recovery hysteresis holds identification", None, None, False)
        if broad_ood:
            transition = self._transition(OperatingMode.UNKNOWN_EVENT, state.timestamp_s, ("unknown model or broad OOD evidence",), force=True)
            return self._decision(Authorization.ROLLBACK, "forecast authority revoked for unknown event", transition, None, True,
                                  DecoderWorkloadAdvisory("use-validated-decoder-fallback", "broad/common-mode event"))
        if state.local_change_probability >= self.config.local_change_entry_probability:
            transition = self._transition(OperatingMode.LOCAL_RECOVERY, state.timestamp_s, ("posterior local change alarm",), force=True)
            return self._decision(Authorization.DELAYED, "local identification/recovery required", transition, None, False)
        if state.observation_nonidentifiable and state.diagnostic_decision_relevant:
            plan = self._select_diagnostic()
            if plan:
                if (self._diagnostics_used >= self.config.maximum_diagnostics
                        or self._diagnostic_downtime_s + plan.downtime_s > self.config.maximum_diagnostic_downtime_s):
                    transition = self._transition(OperatingMode.DEGRADED, state.timestamp_s, ("diagnostic budget exhausted",), force=True)
                    return self._decision(Authorization.REJECTED, "diagnostic budget exhausted", transition, None, False)
                self._diagnostics_used += 1
                self._diagnostic_downtime_s += plan.downtime_s
                transition = self._transition(OperatingMode.DIAGNOSTIC, state.timestamp_s, ("native QEC observations are decision-relevantly nonidentifiable",), force=True)
                return self._decision(Authorization.DELAYED, "explicit diagnostic downtime authorised", transition, plan, False)
        if state.forecast_valid and state.residual_small and not excessive_damage:
            transition = self._transition(OperatingMode.NOMINAL_PREDICTIVE, state.timestamp_s, ("calibrated forecast and small residual",))
            if self.mode is not OperatingMode.NOMINAL_PREDICTIVE:
                return self._decision(
                    Authorization.DELAYED,
                    f"minimum dwell retains {self.mode.value}; predictive authority not yet restored",
                    transition, None, False)
            return self._decision(Authorization.APPROVED, "predictive baseline authorised", transition, None, False)
        if state.forecast_valid and state.residual_learning_safe and not excessive_damage:
            transition = self._transition(OperatingMode.RESIDUAL_LEARNING, state.timestamp_s, ("safe residual learning is justified",))
            if self.mode is not OperatingMode.RESIDUAL_LEARNING:
                return self._decision(
                    Authorization.DELAYED,
                    f"minimum dwell retains {self.mode.value}; residual authority not yet restored",
                    transition, None, False)
            return self._decision(Authorization.APPROVED, "predictive baseline plus bounded residual learning authorised", transition, None, False)
        transition = self._transition(OperatingMode.DEGRADED, state.timestamp_s, ("no valid predictive operating pathway",), force=True)
        return self._decision(Authorization.REJECTED, "use validated fixed/RL fallback", transition, None, False)

    def authorize_control(self, package: PredictiveControlPackage, now_s: float) -> SupervisorDecision:
        violations = []
        if package.status is not SolverStatus.OPTIMAL:
            violations.append("solver did not return a certified optimal action")
        if now_s > package.expiry_time_s:
            violations.append("control package is stale")
        if not package.rollback_snapshot.policy_hash:
            violations.append("rollback snapshot missing")
        if package.controller_acknowledged_hash is not None and package.controller_acknowledged_hash != package.policy_hash:
            violations.append("controller acknowledgement hash mismatch")
        if not package.policy_id:
            violations.append("transactional policy_id missing")
        if not package.reference_policy_id or not package.reference_policy_hash:
            violations.append("transactional reference policy missing")
        if package.reference_policy_hash != package.baseline_policy.policy_hash:
            violations.append("proposal reference differs from the policy used for projection")
        if not package.created_from_state_id or not package.expected_activation_state_id:
            violations.append("creation/expected activation state identifiers missing")
        if not (package.projection_certificate and package.bounds_certificate
                and package.slew_certificate):
            violations.append("projection/bounds/slew certificates missing")
        if not package.controller_state_hash:
            violations.append("controller state hash missing")
        if package.policy_hash != stable_hash(dict(package.action)):
            violations.append("policy hash does not identify the action that will be active")
        if not package.action or any(set(step) != set(package.action) for step in package.trajectory):
            violations.append("policy patch is incomplete/non-atomic")
        for control, value in package.action.items():
            bound = self.limits.controls.get(control)
            if bound and not bound.validate(value):
                violations.append(f"hard bound:{control}")
        for index, step in enumerate(package.trajectory):
            duty = sum(value * value for value in step.values()) / max(1, len(step))
            if duty > self.limits.max_thermal_duty + 1e-12:
                violations.append(f"thermal duty:trajectory:{index}")
        if "predictive" not in self._permitted_modes[self.mode]:
            violations.append(f"mode {self.mode.value} does not permit predictive action")
        if violations:
            return self._decision(Authorization.ROLLBACK, "; ".join(violations), None, None, True)
        return self._decision(Authorization.APPROVED, "all hard invariants satisfied", None, None, False)

    def authorize_residual_candidate(self, package: PredictiveControlPackage,
                                     candidate: ResidualCandidate, now_s: float,
                                     *, cumulative_damage: float = 0.0) -> SupervisorDecision:
        """Authorize one Stage-6 candidate inside the Stage-5 residual subspace.

        This is deliberately a separate hard boundary from ``authorize_control``:
        candidate policies are empirical interventions and must never inherit authority
        merely because their predictive baseline was accepted.
        """
        violations: list[str] = []
        allocation = package.residual_allocation
        if self.mode is not OperatingMode.RESIDUAL_LEARNING:
            violations.append(f"mode {self.mode.value} does not permit residual action")
        if package.status is not SolverStatus.OPTIMAL or now_s > package.expiry_time_s:
            violations.append("residual baseline is invalid or stale")
        if not (package.reference_policy_id and package.reference_policy_hash
                and package.created_from_state_id and package.expected_activation_state_id):
            violations.append("residual baseline lacks transactional reference semantics")
        if not (candidate.reference_policy_id and candidate.reference_policy_hash
                and candidate.created_from_state_id and candidate.expected_activation_state_id):
            violations.append("residual candidate lacks transactional reference semantics")
        if (candidate.reference_policy_hash != package.policy_hash
                or not candidate.reference_policy_id.startswith(package.policy_id)):
            violations.append("residual candidate was projected from the wrong active baseline")
        if not (candidate.projection_certificate and candidate.bounds_certificate
                and candidate.slew_certificate and candidate.controller_state_hash):
            violations.append("residual candidate certificates/state hash missing")
        if set(candidate.full_control) != set(package.action):
            violations.append("candidate policy patch is incomplete/non-atomic")
        if not set(candidate.residual) <= set(allocation.projection_controls):
            violations.append("candidate escapes the Stage-5 residual projection")
        for control, residual in candidate.residual.items():
            bound = allocation.bounds.get(control)
            if bound is None or abs(residual) > bound + 1e-12:
                violations.append(f"residual bound:{control}")
            expected = package.action.get(control, 0.0) + residual
            if abs(candidate.full_control.get(control, expected) - expected) > 1e-12:
                violations.append(f"candidate composition:{control}")
        for control, value in candidate.full_control.items():
            hardware_bound = self.limits.controls.get(control)
            if hardware_bound is None or not hardware_bound.validate(value):
                violations.append(f"hard bound:{control}")
        duty = sum(value * value for value in candidate.full_control.values()) / max(1, len(candidate.full_control))
        if duty > self.limits.max_thermal_duty + 1e-12:
            violations.append("thermal duty:candidate")
        if cumulative_damage + candidate.predicted_damage > self.config.maximum_exploration_damage + 1e-12:
            violations.append("supervisory exploration budget")
        if violations:
            return self._decision(Authorization.REJECTED, "; ".join(violations), None, None, False)
        return self._decision(Authorization.APPROVED, "bounded residual candidate authorised", None, None, False)

    def verify_rollback(self, restored_metric: float, expected_lower: float, expected_upper: float, timestamp_s: float) -> SupervisorDecision:
        if expected_lower <= restored_metric <= expected_upper:
            return self._decision(Authorization.APPROVED, "rollback telemetry validated", None, None, False)
        transition = self._transition(OperatingMode.UNKNOWN_EVENT, timestamp_s, ("rollback did not restore expected telemetry",), force=True)
        return self._decision(Authorization.ROLLBACK, "rollback invalid; prior policy is stale", transition, None, True)

    def begin_local_recovery(self, timestamp_s: float, request_id: str) -> SupervisorDecision:
        transition = self._transition(
            OperatingMode.LOCAL_RECOVERY, timestamp_s,
            (f"structured regional recovery request {request_id}",), force=True)
        return self._decision(
            Authorization.DELAYED, "regional authority revoked pending held-out boundary validation",
            transition, None, False)

    def begin_global_recovery(self, timestamp_s: float, request_id: str) -> SupervisorDecision:
        transition = self._transition(
            OperatingMode.UNKNOWN_EVENT, timestamp_s,
            (f"structured global recovery request {request_id}",), force=True)
        return self._decision(
            Authorization.DELAYED,
            "global authority revoked pending disturbance-aware held-out validation",
            transition, None, False)

    def complete_recovery(self, timestamp_s: float, *, request_id: str,
                          regional: bool, evidence_hash: str) -> SupervisorDecision:
        transition = self._transition(
            OperatingMode.NOMINAL_PREDICTIVE, timestamp_s,
            (f"{'regional' if regional else 'global'} recovery independently validated",),
            force=True)
        if transition is not None:
            transition = TransitionRecord(
                transition.timestamp_s, transition.previous_mode, transition.next_mode,
                transition.rationale, transition.expected_exit_condition,
                {"request_id": request_id, "evidence_hash": evidence_hash,
                 "regional": regional}, transition.stage_versions)
            self.audit_log[-1] = transition
        return self._decision(
            Authorization.APPROVED, "validated recovery restored predictive baseline authority",
            transition, None, False)

    def verify_rollback_interval(self, observed_interval: tuple[float, float],
                                 expected_interval: tuple[float, float],
                                 absolute_ceiling: float,
                                 timestamp_s: float) -> SupervisorDecision:
        observed_lower, observed_upper = observed_interval
        expected_lower, expected_upper = expected_interval
        valid = (
            0 <= observed_lower <= observed_upper <= 1
            and 0 <= expected_lower <= expected_upper <= 1
            and observed_upper <= expected_upper
            and observed_upper <= absolute_ceiling)
        if valid:
            return self._decision(
                Authorization.APPROVED,
                "rollback transaction confirmed and uncertainty-aware physical telemetry validated",
                None, None, False)
        transition = self._transition(
            OperatingMode.UNKNOWN_EVENT, timestamp_s,
            ("rollback transaction confirmed but physical safety/restoration validation failed",),
            force=True)
        return self._decision(
            Authorization.ROLLBACK,
            "physical rollback validation failed; target quarantined and model authority revoked",
            transition, None, True)

    def verify_rollback_evidence(self, verification: RollbackVerification,
                                 timestamp_s: float) -> SupervisorDecision:
        """Authorize a verified restoration and distinguish uncertainty from failure."""
        if verification.status is RollbackVerificationStatus.PHYSICAL_RESTORATION_VERIFIED:
            return self._decision(
                Authorization.APPROVED, verification.reason, None, None, False)
        if verification.status is RollbackVerificationStatus.PHYSICAL_RESTORATION_INCONCLUSIVE:
            transition = self._transition(
                OperatingMode.DEGRADED, timestamp_s,
                ("rollback restoration evidence remains inconclusive",), force=True)
            return self._decision(
                Authorization.DELAYED,
                "safe restored policy held while sequential restoration evidence remains inconclusive",
                transition, None, False)
        transition = self._transition(
            OperatingMode.UNKNOWN_EVENT, timestamp_s,
            ("current-state physical restoration was credibly falsified",), force=True)
        return self._decision(
            Authorization.ROLLBACK,
            "physical restoration failed; target quarantined and model authority revoked",
            transition, None, True)

    def set_model_lifecycle(self, model_id: str, lifecycle: ModelLifecycle, *, held_out_passed: bool = False) -> None:
        if lifecycle in {ModelLifecycle.VALIDATED, ModelLifecycle.PROMOTED} and not held_out_passed:
            raise ValueError("promotion/validation requires held-out predictive and closed-loop evidence")
        current = self.model_lifecycle.get(model_id)
        allowed = {
            None: {ModelLifecycle.CANDIDATE, ModelLifecycle.SHADOW, ModelLifecycle.PROMOTED},
            ModelLifecycle.CANDIDATE: {ModelLifecycle.SHADOW, ModelLifecycle.QUARANTINED, ModelLifecycle.ARCHIVED},
            ModelLifecycle.SHADOW: {ModelLifecycle.VALIDATED, ModelLifecycle.QUARANTINED, ModelLifecycle.ARCHIVED},
            ModelLifecycle.VALIDATED: {ModelLifecycle.PROMOTED, ModelLifecycle.QUARANTINED, ModelLifecycle.ARCHIVED},
            ModelLifecycle.PROMOTED: {ModelLifecycle.QUARANTINED, ModelLifecycle.ARCHIVED},
            ModelLifecycle.QUARANTINED: {ModelLifecycle.SHADOW, ModelLifecycle.ARCHIVED},
            ModelLifecycle.ARCHIVED: set(),
        }
        # Preserve direct externally validated promotion for backward compatibility and explicit imports.
        if lifecycle not in allowed[current] and not (lifecycle is ModelLifecycle.PROMOTED and held_out_passed):
            raise ValueError(f"invalid lifecycle transition {current} -> {lifecycle}")
        self.model_lifecycle[model_id] = lifecycle
