"""Safety-bearing control-package contracts for Stage 5."""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Mapping

from hdfa_rl_suite.stage0.schema import PolicySnapshot, stable_hash


class SolverStatus(str, Enum):
    OPTIMAL = "optimal"
    FALLBACK = "fallback"
    INFEASIBLE = "infeasible"
    EXPIRED_FORECAST = "expired_forecast"


@dataclass(frozen=True)
class ResidualAllocation:
    projection_controls: tuple[str, ...]
    bounds: Mapping[str, float]
    modelled_controls: tuple[str, ...]
    rationale: Mapping[str, str]


@dataclass(frozen=True)
class PredictedCostDistribution:
    expected_cost: float
    worst_scenario_cost: float
    detector_violation_probability: Mapping[str, float]
    cvar_cost: float = 0.0
    logical_risk: float = 0.0
    correlation_risk: float = 0.0


@dataclass(frozen=True)
class SharedResourceConstraint:
    constraint_id: str
    coefficients: Mapping[str, float]
    maximum: float
    unit: str = "normalized"


@dataclass(frozen=True)
class InfeasibilityCertificate:
    reason: str
    violated_constraints: tuple[str, ...]
    fallback_policy_hash: str


@dataclass(frozen=True)
class PredictiveControlPackage:
    schema_version: str
    status: SolverStatus
    action: Mapping[str, float]
    trajectory: tuple[Mapping[str, float], ...]
    feedforward_component: Mapping[str, float]
    residual_allocation: ResidualAllocation
    active_constraints: tuple[str, ...]
    cost_distribution: PredictedCostDistribution
    baseline_policy: PolicySnapshot
    policy_hash: str
    activation_time_s: float
    expiry_time_s: float
    rollback_snapshot: PolicySnapshot
    infeasibility: InfeasibilityCertificate | None = None
    robustness_margin: float = 0.0
    fallback_action: Mapping[str, float] | None = None
    invalidity_reasons: tuple[str, ...] = ()
    controller_acknowledged_hash: str | None = None
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


def bind_policy_lifecycle(package: PredictiveControlPackage, *, policy_id: str,
                          reference_policy_id: str, reference_policy_hash: str,
                          created_from_state_id: str,
                          controller_state_hash: str,
                          reference_controls: Mapping[str, float] | None = None,
                          reference_timestamp_s: float | None = None) -> PredictiveControlPackage:
    """Bind a proposal to the exact confirmed policy and causal device state."""
    action_hash = stable_hash(dict(package.action))
    expected_state = stable_hash({
        "created_from_state_id": created_from_state_id,
        "reference_policy_id": reference_policy_id,
        "reference_policy_hash": reference_policy_hash,
        "policy_hash": action_hash,
        "activation_time_s": package.activation_time_s,
    })
    certificate_base = {
        "policy_hash": action_hash,
        "reference_policy_id": reference_policy_id,
        "reference_policy_hash": reference_policy_hash,
    }
    baseline = package.baseline_policy
    if reference_controls is not None:
        baseline = PolicySnapshot(dict(reference_controls), reference_policy_hash,
                                  package.activation_time_s if reference_timestamp_s is None
                                  else reference_timestamp_s)
    return replace(
        package, policy_hash=action_hash, policy_id=policy_id,
        baseline_policy=baseline, rollback_snapshot=baseline,
        reference_policy_id=reference_policy_id,
        reference_policy_hash=reference_policy_hash,
        created_from_state_id=created_from_state_id,
        expected_activation_state_id=expected_state,
        projection_certificate=stable_hash({**certificate_base, "type": "projection"}),
        bounds_certificate=stable_hash({**certificate_base, "type": "bounds"}),
        slew_certificate=stable_hash({**certificate_base, "type": "slew"}),
        controller_state_hash=controller_state_hash,
    )
