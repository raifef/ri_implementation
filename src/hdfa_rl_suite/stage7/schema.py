"""Auditable Stage-7 supervisory state-machine contracts."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class OperatingMode(str, Enum):
    BOOTSTRAP = "bootstrap"
    NOMINAL_PREDICTIVE = "nominal_predictive"
    RESIDUAL_LEARNING = "residual_learning"
    LOCAL_RECOVERY = "local_recovery"
    UNKNOWN_EVENT = "unknown_event"
    DIAGNOSTIC = "diagnostic"
    DEGRADED = "degraded"
    FAIL_SAFE = "fail_safe"


class Authorization(str, Enum):
    APPROVED = "approved"
    CLIPPED = "clipped"
    DELAYED = "delayed"
    REJECTED = "rejected"
    ROLLBACK = "rollback"


class ModelLifecycle(str, Enum):
    CANDIDATE = "candidate"
    SHADOW = "shadow"
    VALIDATED = "validated"
    PROMOTED = "promoted"
    QUARANTINED = "quarantined"
    ARCHIVED = "archived"


@dataclass(frozen=True)
class StageHealth:
    stage: str
    valid: bool = True
    hard_invalid_data: bool = False
    ood_score: float = 0.0
    latency_s: float = 0.0
    residual_bias: float = 0.0
    solver_ok: bool = True
    exploration_damage: float = 0.0
    invalidity_reasons: tuple[str, ...] = ()
    policy_hash: str | None = None
    model_version: str | None = None
    timestamp_s: float | None = None


@dataclass(frozen=True)
class DiagnosticOption:
    diagnostic_id: str
    expected_regret_reduction: float
    interruption_s: float
    risk_cost: float
    required_for_decision: bool = True

    @property
    def value(self) -> float:
        return self.expected_regret_reduction / max(self.interruption_s + self.risk_cost, 1e-12)


@dataclass(frozen=True)
class DiagnosticPlan:
    diagnostic_id: str
    value: float
    downtime_s: float
    rationale: str


@dataclass(frozen=True)
class SupervisorInput:
    timestamp_s: float
    health: tuple[StageHealth, ...]
    hard_invariant_failed: bool = False
    bootstrap_required: bool = False
    broad_ood: bool = False
    local_change_probability: float = 0.0
    unknown_model_probability: float = 0.0
    observation_nonidentifiable: bool = False
    diagnostic_decision_relevant: bool = False
    forecast_valid: bool = False
    residual_learning_safe: bool = False
    residual_small: bool = True
    controller_confirmed: bool = True
    policy_hash_consistent: bool = True
    logical_risk: float = 0.0
    correlation_alarm: float = 0.0
    human_veto: bool = False


@dataclass(frozen=True)
class TransitionRecord:
    timestamp_s: float
    previous_mode: OperatingMode
    next_mode: OperatingMode
    rationale: tuple[str, ...]
    expected_exit_condition: str
    evidence: Mapping[str, float | str | bool] = field(default_factory=dict)
    stage_versions: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BudgetState:
    exploration_used: float
    exploration_limit: float
    diagnostic_downtime_used_s: float
    diagnostic_downtime_limit_s: float
    diagnostics_used: int
    diagnostics_limit: int


@dataclass(frozen=True)
class DecoderWorkloadAdvisory:
    action: str
    rationale: str
    requires_independent_validation: bool = True


@dataclass(frozen=True)
class SupervisorDecision:
    mode: OperatingMode
    authorization: Authorization
    reason: str
    transition: TransitionRecord | None
    diagnostic: DiagnosticPlan | None
    rollback_required: bool
    model_lifecycle: Mapping[str, ModelLifecycle] = field(default_factory=dict)
    budgets: BudgetState | None = None
    advisory: DecoderWorkloadAdvisory | None = None
    decision_id: str = ""
