"""Stage-3 joint state/regime/model posterior contracts."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

from hdfa_rl_suite.stage2.schema import PosteriorSample


class DynamicsModelKind(str, Enum):
    CONSTANT = "constant"
    RANDOM_WALK = "random_walk"
    ORNSTEIN_UHLENBECK = "ornstein_uhlenbeck"
    OSCILLATOR = "oscillator"
    RANDOM_TELEGRAPH = "random_telegraph"
    SEMI_MARKOV_TELEGRAPH = "semi_markov_telegraph"
    STEP = "step"
    UNKNOWN = "unknown_heavy_tailed"
    ADDITIVE_COMPOSITE = "additive_composite"


@dataclass(frozen=True)
class DynamicsModelSpec:
    model_id: str
    kind: DynamicsModelKind
    variable_id: str
    prior_probability: float
    parameters: Mapping[str, float] = field(default_factory=dict)
    complexity_penalty: float = 0.0
    components: tuple["DynamicsModelSpec", ...] = ()


@dataclass(frozen=True)
class DynamicsParticle:
    model_id: str
    state: Mapping[str, float]
    auxiliary_state: Mapping[str, float]
    regime: str
    changepoint: bool
    weight: float
    component_state: Mapping[str, float] = field(default_factory=dict)
    parameter_state: Mapping[str, float] = field(default_factory=dict)
    lineage_id: str = ""


@dataclass(frozen=True)
class ChangeAlarm:
    probability: float
    onset_time_s: float
    severity: str
    affected_region: str
    onset_interval_s: tuple[float, float] | None = None


@dataclass(frozen=True)
class ModelEvidence:
    model_probabilities: Mapping[str, float]
    unknown_probability: float
    predictive_log_score: float


@dataclass(frozen=True)
class HierarchicalComponent:
    component_id: str
    model_id: str
    variable_id: str
    timescale_s: float
    responsibility: float
    parent_component_id: str | None = None


@dataclass(frozen=True)
class RecurringRegimeMatch:
    regime_id: str
    probability: float
    expected_control_signature: Mapping[str, float]
    previously_validated: bool


@dataclass(frozen=True)
class FamiliarProcessState:
    """Persistent causal summary used for warm prediction and safe policy lookup."""

    family: str
    confidence: float
    regime_id: str
    state_value: float
    state_velocity: float
    amplitude_mean: float | None = None
    period_mean_s: float | None = None
    phase_mean_rad: float | None = None
    mean_dwell_s: float | None = None
    transition_probability: float | None = None
    common_mode_component: float = 0.0
    local_switching_component: float = 0.0
    smooth_component: float = 0.0
    abrupt_step_component: float = 0.0
    phase_coherent: bool = False
    warm_started: bool = False
    immediate_feedforward_safe: bool = False
    invalidity_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class DynamicsPosterior:
    schema_version: str
    region_id: str
    timestamp_s: float
    particles: tuple[DynamicsParticle, ...]
    current_state_mean: Mapping[str, float]
    model_evidence: ModelEvidence
    change_alarm: ChangeAlarm
    hierarchy: tuple[HierarchicalComponent, ...]
    model_parameters: Mapping[str, Mapping[str, float]]
    unknown_model_probability: float
    invalidity_reasons: tuple[str, ...]
    online_approximation_note: str
    recurring_regime: RecurringRegimeMatch | None = None
    offline_divergence: float | None = None
    familiar_process: FamiliarProcessState | None = None

    @property
    def state_samples(self) -> tuple[PosteriorSample, ...]:
        return tuple(PosteriorSample(item.state, item.weight) for item in self.particles)
