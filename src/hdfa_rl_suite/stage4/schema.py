"""Stage-4 forecast data contracts preserving sample/model multimodality."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class LatencyModel:
    acquisition_s: float
    inference_s: float
    optimisation_s: float
    upload_s: float
    jitter_s: float = 0.0

    @property
    def mean_s(self) -> float:
        return self.acquisition_s + self.inference_s + self.optimisation_s + self.upload_s

    def quantile_s(self, probability: float) -> float:
        # Normal approximation is intentionally conservative at the supported 50/90/99% levels.
        z = .0 if probability <= .5 else (1.282 if probability <= .9 else 2.326)
        return max(0., self.mean_s + z * self.jitter_s)


@dataclass(frozen=True)
class ResponseMap:
    """Local state-to-optimum map, valid only inside the stated trust region."""
    reference_controls: Mapping[str, float]
    correction_gain: Mapping[tuple[str, str], float]
    detector_threshold: float = .10
    validity_radius: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ForecastScenario:
    horizon_s: float
    activation_offset_s: float
    model_id: str
    state: Mapping[str, float]
    optimum_controls: Mapping[str, float]
    detector_probabilities: Mapping[str, float]
    weight: float
    logical_risk: float = 0.0
    correlation_risk: float = 0.0
    leakage_risk: float = 0.0
    context_id: str = "default"


@dataclass(frozen=True)
class ForecastRisk:
    detector_threshold_probability: Mapping[str, float]
    model_disagreement: float
    state_variance: Mapping[str, float]
    optimum_variance: Mapping[str, float]
    unknown_model_probability: float
    logical_risk_mean: float = 0.0
    logical_risk_cvar: float = 0.0
    worst_region_probability: float = 0.0
    correlation_risk_mean: float = 0.0


@dataclass(frozen=True)
class ForecastCalibration:
    count: int
    mean_log_score: float | None
    mean_brier_score: float | None
    interval_coverage: float | None
    mean_crps: float | None = None
    mean_energy_score: float | None = None


@dataclass(frozen=True)
class ForecastBundle:
    schema_version: str
    region_id: str
    issued_at_s: float
    latency: LatencyModel
    scenarios_by_horizon: Mapping[float, tuple[ForecastScenario, ...]]
    risk_by_horizon: Mapping[float, ForecastRisk]
    validity_horizon_s: float
    calibration: ForecastCalibration
    invalidity_reasons: tuple[str, ...]
    state_quantiles_by_horizon: Mapping[float, Mapping[str, tuple[float, float, float]]] = field(default_factory=dict)
    detector_count_moments_by_horizon: Mapping[float, Mapping[str, tuple[float, float]]] = field(default_factory=dict)
    scenario_reduction_error_bound: float = 0.0
    context_id: str = "default"

    def scenarios(self, horizon_s: float) -> tuple[ForecastScenario, ...]:
        return self.scenarios_by_horizon[horizon_s]
