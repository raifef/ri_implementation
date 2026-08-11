"""Versioned Stage-1 data contracts; all event statistics retain exact exposures."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Mapping

from hdfa_rl_suite.common import PolicyLifecycleState
from hdfa_rl_suite.stage0.schema import DetectorDefinition, stable_hash

SCHEMA_VERSION = "stage1.v1"


class QualitySeverity(str, Enum):
    WARNING = "warning"
    SOFT_INVALID = "soft_invalid"
    HARD_INVALID = "hard_invalid"


@dataclass(frozen=True)
class CircuitContext:
    circuit_hash: str
    context_id: str
    logical_basis: str
    code_distance: int
    schedule_id: str
    reset_mode: str
    decoder_version: str = "none"


@dataclass(frozen=True)
class ClockCalibration:
    """Affine device-to-reference clock map with a conservative uncertainty bound."""
    offset_s: float = 0.0
    scale: float = 1.0
    uncertainty_s: float = 0.0
    version: str = "clock.v1"

    def reference_time(self, device_timestamp_s: float) -> float:
        return self.offset_s + self.scale * device_timestamp_s


@dataclass(frozen=True)
class RawMeasurementRecord:
    record_id: str
    sequence: int
    shot: int
    cycle: int
    device_timestamp_s: float
    measurements: tuple[int | None, ...]
    circuit_hash: str
    channel_ids: tuple[str, ...]


@dataclass(frozen=True)
class PolicyActivation:
    policy_id: str
    policy_hash: str
    requested_at_s: float
    acknowledged_at_s: float | None
    activation_uncertainty_s: float
    controls: Mapping[str, float]
    perturbation: Mapping[str, float] = field(default_factory=dict)
    candidate_id: str | None = None
    antithetic_pair_id: str | None = None
    transaction_id: str = ""
    reference_policy_id: str = ""
    reference_policy_hash: str = ""
    created_from_state_id: str = ""
    expected_activation_state_id: str = ""
    projection_certificate: str = ""
    bounds_certificate: str = ""
    slew_certificate: str = ""
    supervisor_authorization: str = ""
    activation_acknowledgement: str = ""
    lifecycle_state: PolicyLifecycleState = PolicyLifecycleState.CONFIRMED

    @property
    def nominal_activation_s(self) -> float:
        return self.acknowledged_at_s if self.acknowledged_at_s is not None else self.requested_at_s

    def ambiguity_interval(self) -> tuple[float, float]:
        value = self.nominal_activation_s
        return value - self.activation_uncertainty_s, value + self.activation_uncertainty_s


@dataclass(frozen=True)
class QualityFlag:
    code: str
    severity: QualitySeverity
    message: str
    record_id: str | None = None


@dataclass(frozen=True)
class AlignedDetectorEvent:
    record_id: str
    shot: int
    cycle: int
    detector_id: str
    value: int | None
    exposure: bool
    timestamp_s: float
    policy_id: str | None
    policy_hash: str | None
    candidate_id: str | None
    context_id: str
    region_id: str
    source_sequence: int
    ambiguous_policy: bool = False
    active_controls: Mapping[str, float] = field(default_factory=dict)
    perturbation: Mapping[str, float] = field(default_factory=dict)
    activation_interval_s: tuple[float, float] | None = None


@dataclass(frozen=True)
class CountFactor:
    window_size: int
    detector_id: str
    events: int
    exposures: int
    alpha: float
    beta: float
    start_timestamp_s: float | None
    end_timestamp_s: float | None
    policy_hash: str | None = None
    context_id: str | None = None
    active_controls: Mapping[str, float] = field(default_factory=dict)
    perturbation: Mapping[str, float] = field(default_factory=dict)

    @property
    def rate(self) -> float | None:
        return self.events / self.exposures if self.exposures else None


@dataclass(frozen=True)
class PairCount:
    detector_a: str
    detector_b: str
    n00: int
    n01: int
    n10: int
    n11: int
    policy_hash: str | None = None
    context_id: str | None = None


@dataclass(frozen=True)
class TelemetryRegionView:
    region_id: str
    detector_ids: tuple[str, ...]
    control_ids: tuple[str, ...]
    events: tuple[AlignedDetectorEvent, ...]
    count_factors: tuple[CountFactor, ...]
    pair_counts: tuple[PairCount, ...]
    context: CircuitContext


@dataclass(frozen=True)
class ReplayManifest:
    raw_record_ids: tuple[str, ...]
    raw_hash: str
    detector_definition_hash: str
    policy_timeline_hash: str
    context_hash: str
    transformation_version: str = SCHEMA_VERSION

    @property
    def manifest_hash(self) -> str:
        return stable_hash(asdict(self))


@dataclass(frozen=True)
class TelemetryBatch:
    schema_version: str
    event_tensor: Mapping[tuple[int, int, str], int | None]
    exposure_mask: Mapping[tuple[int, int, str], bool]
    events: tuple[AlignedDetectorEvent, ...]
    count_factors: tuple[CountFactor, ...]
    pair_counts: tuple[PairCount, ...]
    regional_views: Mapping[str, TelemetryRegionView]
    quality_flags: tuple[QualityFlag, ...]
    replay_manifest: ReplayManifest

    @property
    def hard_invalid(self) -> bool:
        return any(flag.severity is QualitySeverity.HARD_INVALID for flag in self.quality_flags)

    def to_dict(self) -> dict[str, Any]:
        """Serialize tuple-indexed tensors without losing their explicit coordinates."""
        return {
            "schema_version": self.schema_version,
            "event_tensor": [{"shot": shot, "cycle": cycle, "detector_id": detector, "value": value}
                             for (shot, cycle, detector), value in self.event_tensor.items()],
            "exposure_mask": [{"shot": shot, "cycle": cycle, "detector_id": detector, "exposure": value}
                              for (shot, cycle, detector), value in self.exposure_mask.items()],
            "events": [_jsonable(item) for item in self.events],
            "count_factors": [_jsonable(item) for item in self.count_factors],
            "pair_counts": [_jsonable(item) for item in self.pair_counts],
            "regional_views": {key: _jsonable(value) for key, value in self.regional_views.items()},
            "quality_flags": [_jsonable(item) for item in self.quality_flags],
            "replay_manifest": _jsonable(self.replay_manifest),
        }


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    return value
