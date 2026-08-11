"""Versioned residual-RL records with causal candidate and replay provenance."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Mapping

from hdfa_rl_suite.stage0.schema import stable_hash


class ResidualRLDisposition(str, Enum):
    ABSTAIN = "abstain"
    SHADOW = "shadow"
    ACTIVE = "active"
    DEACTIVATED = "deactivated"


@dataclass(frozen=True)
class ResidualGateEvidence:
    persistence_intervals: int
    repeatability: float
    control_sensitivity: float
    identifiable: bool
    forecast_uncertainty: float
    stability_probability: float
    residual_magnitude: float
    noise_floor: float
    probability_positive_value: float
    expected_heldout_gain: float
    gradient_snr: float
    lifecycle_healthy: bool
    predictive_only_adequate: bool
    ood: bool = False
    rollback_unresolved: bool = False
    predictive_better: bool = False


@dataclass(frozen=True)
class ResidualGateDecision:
    disposition: ResidualRLDisposition
    eligible: bool
    reasons: tuple[str, ...]
    evidence: ResidualGateEvidence
    consecutive_regressions: int = 0
    schema_version: str = "residual-gate.v1"


@dataclass(frozen=True)
class ShadowValidation:
    baseline_rate: float
    candidate_rate: float
    baseline_exposures: int
    candidate_exposures: int
    estimated_gain: float
    gain_standard_error: float
    probability_positive_value: float
    passed: bool
    reason: str


@dataclass(frozen=True)
class ResidualCandidate:
    candidate_id: str
    pair_id: str
    sign: int
    residual: Mapping[str, float]
    full_control: Mapping[str, float]
    predicted_damage: float
    policy_version: int
    mean_residual: Mapping[str, float] = field(default_factory=dict)
    exploration_offset: Mapping[str, float] = field(default_factory=dict)
    policy_id: str = ""
    reference_policy_id: str = ""
    reference_policy_hash: str = ""
    created_from_state_id: str = ""
    expected_activation_state_id: str = ""
    projection_certificate: str = ""
    bounds_certificate: str = ""
    slew_certificate: str = ""
    supervisor_authorization: str = ""
    activation_acknowledgement: str = ""
    controller_state_hash: str = ""


def bind_candidate_lifecycle(candidate: ResidualCandidate, *,
                             reference_policy_id: str,
                             reference_policy_hash: str,
                             created_from_state_id: str,
                             controller_state_hash: str,
                             policy_id: str | None = None) -> ResidualCandidate:
    transaction_policy_id = policy_id or (
        f"{reference_policy_id}:candidate:{candidate.candidate_id}")
    policy_hash = stable_hash(dict(candidate.full_control))
    expected = stable_hash({
        "candidate_id": candidate.candidate_id,
        "policy_hash": policy_hash,
        "reference_policy_id": reference_policy_id,
        "reference_policy_hash": reference_policy_hash,
        "created_from_state_id": created_from_state_id,
    })
    base = {"candidate_id": candidate.candidate_id, "policy_hash": policy_hash,
            "reference_policy_id": reference_policy_id}
    return replace(
        candidate, policy_id=transaction_policy_id,
        reference_policy_id=reference_policy_id,
        reference_policy_hash=reference_policy_hash,
        created_from_state_id=created_from_state_id,
        expected_activation_state_id=expected,
        projection_certificate=stable_hash({**base, "type": "projection"}),
        bounds_certificate=stable_hash({**base, "type": "bounds"}),
        slew_certificate=stable_hash({**base, "type": "slew"}),
        controller_state_hash=controller_state_hash,
    )


@dataclass(frozen=True)
class CandidateObservation:
    candidate_id: str
    detector_losses: Mapping[str, float]
    exposures: Mapping[str, int]
    logical_risk: float = 0.0
    leakage_risk: float = 0.0
    correlation_penalty: float = 0.0
    regime_id: str = "unknown"
    context_id: str = "default"
    model_version: str = "unknown"
    observed_at_s: float = 0.0
    behaviour_probability: float = 1.0
    current_probability: float = 1.0
    mean_policy_detector_losses: Mapping[str, float] = field(default_factory=dict)

    @property
    def total_damage(self) -> float:
        return self.logical_risk + self.leakage_risk + self.correlation_penalty


@dataclass(frozen=True)
class EmpiricalResponseEvidence:
    control_id: str
    detector_id: str
    directional_response: float
    pair_id: str


@dataclass(frozen=True)
class ReplayItem:
    candidate: ResidualCandidate
    observation: CandidateObservation
    baseline_hash: str


@dataclass(frozen=True)
class ResidualRLResult:
    policy_mean: Mapping[str, float]
    policy_stddev: Mapping[str, float]
    policy_version: int
    gradient: Mapping[str, float]
    exploration_damage: float
    cumulative_damage: float
    response_evidence: tuple[EmpiricalResponseEvidence, ...]
    fallback_requested: bool
    fallback_reason: str | None
    replay_size: int
    policy_covariance: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    residual_bias: float = 0.0
    gradient_snr: float = 0.0
    invalidity_reasons: tuple[str, ...] = ()
    gate_decision: ResidualGateDecision | None = None
    shadow_validation: ShadowValidation | None = None
    committed: bool = True
