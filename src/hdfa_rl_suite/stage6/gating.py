"""Predeclared activation, shadow-validation, and deactivation rules for Stage 6."""
from __future__ import annotations

from dataclasses import dataclass

from .schema import (
    ResidualGateDecision, ResidualGateEvidence, ResidualRLDisposition,
)


@dataclass(frozen=True)
class ResidualGateConfig:
    minimum_persistence_intervals: int = 2
    minimum_repeatability: float = .55
    minimum_control_sensitivity: float = 1e-4
    maximum_forecast_uncertainty: float = .35
    minimum_stability_probability: float = .80
    minimum_probability_positive_value: float = .80
    minimum_gradient_snr: float = 1.0
    minimum_gain: float = 0.0
    deactivation_regressions: int = 2


class ResidualActivationGate:
    """Stateful, observable-only gate; it may explicitly decline to learn."""

    def __init__(self, config: ResidualGateConfig = ResidualGateConfig()) -> None:
        self.config = config
        self._consecutive_regressions = 0
        self._deactivated = False
        self._authority_observed = False

    def evaluate(self, evidence: ResidualGateEvidence) -> ResidualGateDecision:
        c = self.config
        hard = []
        if not evidence.lifecycle_healthy:
            hard.append("lifecycle is unhealthy")
        if evidence.rollback_unresolved:
            hard.append("rollback restoration is unresolved")
        if evidence.ood:
            hard.append("out-of-distribution evidence revokes residual authority")
        if not evidence.identifiable:
            hard.append("residual is not control-identifiable")
        if evidence.control_sensitivity < c.minimum_control_sensitivity:
            hard.append("control sensitivity is below the declared floor")
        if evidence.forecast_uncertainty > c.maximum_forecast_uncertainty:
            hard.append("forecast uncertainty is excessive")
        if evidence.stability_probability < c.minimum_stability_probability:
            hard.append("local closed-loop stability evidence is insufficient")
        if evidence.predictive_better:
            hard.append("held-out predictive-only control is better")
        if self._deactivated:
            hard.append("residual learner was automatically deactivated")
        if hard:
            disposition = (ResidualRLDisposition.DEACTIVATED
                           if self._authority_observed or self._deactivated
                           else ResidualRLDisposition.ABSTAIN)
            self._authority_observed = False
            return ResidualGateDecision(
                disposition, False, tuple(hard), evidence,
                self._consecutive_regressions)

        noise_only = evidence.residual_magnitude <= evidence.noise_floor
        nonpersistent = evidence.persistence_intervals < c.minimum_persistence_intervals
        if evidence.predictive_only_adequate or noise_only or nonpersistent:
            reasons = []
            if evidence.predictive_only_adequate:
                reasons.append("predictive-only control already meets the declared adequacy target")
            if noise_only:
                reasons.append("residual magnitude is indistinguishable from the noise floor")
            if nonpersistent:
                reasons.append("residual has not persisted for the predeclared duration")
            disposition = (ResidualRLDisposition.DEACTIVATED
                           if self._authority_observed else ResidualRLDisposition.ABSTAIN)
            self._authority_observed = False
            return ResidualGateDecision(
                disposition, False, tuple(reasons), evidence,
                self._consecutive_regressions)

        value_ready = (
            evidence.repeatability >= c.minimum_repeatability
            and evidence.probability_positive_value >= c.minimum_probability_positive_value
            and evidence.expected_heldout_gain > c.minimum_gain
            and evidence.gradient_snr >= c.minimum_gradient_snr)
        if value_ready:
            self._authority_observed = True
            return ResidualGateDecision(
                ResidualRLDisposition.ACTIVE, True,
                ("persistent, repeatable, identifiable residual has positive held-out value",),
                evidence, self._consecutive_regressions)
        return ResidualGateDecision(
            ResidualRLDisposition.SHADOW, True,
            ("eligibility prerequisites pass; positive value requires shadow evidence",),
            evidence, self._consecutive_regressions)

    def record_shadow_outcome(self, *, gain: float,
                              probability_positive_value: float,
                              gradient_snr: float) -> bool:
        """Return whether promotion remains allowed after independent evaluation."""
        regressed = (gain <= self.config.minimum_gain
                     or probability_positive_value < self.config.minimum_probability_positive_value
                     or gradient_snr < self.config.minimum_gradient_snr)
        self._consecutive_regressions = (
            self._consecutive_regressions + 1 if regressed else 0)
        if self._consecutive_regressions >= self.config.deactivation_regressions:
            self._deactivated = True
        if not regressed and not self._deactivated:
            self._authority_observed = True
        return not regressed and not self._deactivated

    def reset_after_recalibration(self) -> None:
        self._consecutive_regressions = 0
        self._deactivated = False
        self._authority_observed = False

    @property
    def deactivated(self) -> bool:
        return self._deactivated
