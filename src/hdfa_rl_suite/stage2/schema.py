"""Likelihood-preserving data contracts for latent physical-state inference."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class InferenceValidity(str, Enum):
    VALID = "valid"
    LOW_OBSERVABILITY = "low_observability"
    OOD = "out_of_distribution"
    INSUFFICIENT_DATA = "insufficient_data"
    MODEL_MISMATCH = "model_mismatch"


@dataclass(frozen=True)
class LatentVariable:
    variable_id: str
    physical_interpretation: str
    unit: str
    lower: float
    upper: float
    nominal: float = 0.0
    intervention_control: str | None = None
    safe_intervention: float | None = None


@dataclass(frozen=True)
class StateSchema:
    region_id: str
    variables: tuple[LatentVariable, ...]
    shared_variables: tuple[str, ...] = ()
    version: str = "stage2.v1"


@dataclass(frozen=True)
class DetectorResponse:
    detector_id: str
    intercept: float
    state_linear: Mapping[str, float] = field(default_factory=dict)
    control_linear: Mapping[str, float] = field(default_factory=dict)
    state_quadratic: Mapping[tuple[str, str], float] = field(default_factory=dict)
    control_quadratic: Mapping[tuple[str, str], float] = field(default_factory=dict)
    state_control: Mapping[tuple[str, str], float] = field(default_factory=dict)
    discrepancy_scale: float = 0.0
    context_intercepts: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PosteriorSample:
    state: Mapping[str, float]
    weight: float


@dataclass(frozen=True)
class ObservabilityReport:
    fisher_information: tuple[tuple[float, ...], ...]
    eigenvalues: tuple[float, ...]
    eigenvectors: tuple[tuple[float, ...], ...]
    rank: int
    condition_number: float
    unresolved_variable_ids: tuple[str, ...]
    detector_contributions: Mapping[str, float]


@dataclass(frozen=True)
class InterventionDesignRequest:
    variable_id: str
    control_id: str
    positive_patch: Mapping[str, float]
    negative_patch: Mapping[str, float]
    expected_information_gain: float
    rationale: str


@dataclass(frozen=True)
class PosteriorPredictiveCheck:
    detector_expected_rates: Mapping[str, float]
    standardized_residuals: Mapping[str, float]
    max_abs_residual: float
    correlation_warning: bool
    pair_standardized_residuals: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PhysicalStatePosterior:
    schema_version: str
    region_id: str
    method: str
    mean: Mapping[str, float]
    covariance: tuple[tuple[float, ...], ...]
    samples: tuple[PosteriorSample, ...]
    observability: ObservabilityReport
    posterior_predictive: PosteriorPredictiveCheck
    attribution: Mapping[str, float]
    model_discrepancy: float
    validity: InferenceValidity
    invalidity_reasons: tuple[str, ...]
    observation_model_version: str
    intervention_request: InterventionDesignRequest | None = None
    validity_radius: Mapping[str, float] = field(default_factory=dict)
    validity_horizon_s: float = 0.0
    ood_score: float = 0.0
    shared_component_mean: Mapping[str, float] = field(default_factory=dict)
