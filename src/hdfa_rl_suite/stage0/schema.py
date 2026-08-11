"""Versioned Stage-0 records.  These records intentionally use only canonical SI units."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from hashlib import sha256
import json
from typing import Any, Mapping

SCHEMA_VERSION = "stage0.v1"


def stable_hash(value: Any) -> str:
    """Hash a JSON-compatible object deterministically for provenance and replay."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return sha256(encoded).hexdigest()


class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    STALE = "stale"
    BLOCKED = "blocked"


class HealthStatus(str, Enum):
    PASSED = "passed"
    DEGRADED = "degraded"
    FAILED = "failed"


@dataclass(frozen=True)
class ControlBound:
    minimum: float
    maximum: float
    max_slew: float
    unit: str
    trust_radius: float

    def validate(self, value: float) -> bool:
        return self.minimum <= value <= self.maximum


@dataclass(frozen=True)
class DeviceTopology:
    device_id: str
    qubits: tuple[str, ...]
    couplers: tuple[tuple[str, str], ...]
    resonators: Mapping[str, str]
    control_channels: Mapping[str, str]
    sample_period_s: float = 1e-9
    controller_latency_s: float = 1e-4


@dataclass(frozen=True)
class HardwareLimits:
    controls: Mapping[str, ControlBound]
    max_thermal_duty: float = 0.5
    max_leakage: float = 0.05


@dataclass(frozen=True)
class DetectorDefinition:
    detector_id: str
    measurement_indices: tuple[int, ...]
    reference_parity: int
    affected_gates: tuple[str, ...]
    region_id: str


@dataclass(frozen=True)
class TargetQECCircuit:
    circuit_id: str
    circuit_hash: str
    gates: tuple[str, ...]
    detectors: tuple[DetectorDefinition, ...]
    code_distance: int = 3


@dataclass(frozen=True)
class ParameterRecord:
    parameter_id: str
    physical_name: str
    channel: str
    unit: str
    current_value: float
    bound: ControlBound
    owning_node: str
    affected_gates: tuple[str, ...]
    affected_detectors: tuple[str, ...]
    region_id: str
    covariance: float
    sensitivity_scale: float | None = None
    local_jacobian: Mapping[str, float] = field(default_factory=dict)
    validity_until_s: float = float("inf")
    model_version: str = SCHEMA_VERSION


@dataclass(frozen=True)
class CalibrationEstimate:
    values: Mapping[str, float]
    variances: Mapping[str, float]
    model_scores: Mapping[str, float]
    held_out_score: float
    confidence: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CalibrationNode:
    node_id: str
    family: str
    owned_parameters: tuple[str, ...]
    prerequisites: tuple[str, ...]
    invalidates: tuple[str, ...]
    resources: tuple[str, ...]
    max_attempts: int = 2
    validity_duration_s: float = 3600.0
    minimum_confidence: float = 0.90


@dataclass(frozen=True)
class CalibrationEvent:
    sequence: int
    node_id: str
    event_type: str
    timestamp_s: float
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class DeclaredCompromise:
    affected_claim: str
    rationale: str
    expected_cost: str
    removal_plan: str


@dataclass(frozen=True)
class PolicySnapshot:
    values: Mapping[str, float]
    policy_hash: str
    timestamp_s: float


@dataclass(frozen=True)
class StageHealthPacket:
    status: HealthStatus
    invalid_reasons: tuple[str, ...]
    unresolved_nodes: tuple[str, ...]
    ambiguities: Mapping[str, Any]
    rollback_available: bool


@dataclass(frozen=True)
class BootstrapResult:
    schema_version: str
    baseline_policy: PolicySnapshot
    qec_circuit: TargetQECCircuit
    parameter_registry: Mapping[str, ParameterRecord]
    calibration_dag: Mapping[str, NodeStatus]
    detector_control_graph: Mapping[str, tuple[str, ...]]
    sensitivity_scales: Mapping[str, float]
    rollback_snapshot: PolicySnapshot
    health: StageHealthPacket
    event_log: tuple[CalibrationEvent, ...]
    compromises: tuple[DeclaredCompromise, ...] = ()
    calibration_nodes: Mapping[str, CalibrationNode] = field(default_factory=dict)
    calibration_estimates: Mapping[str, CalibrationEstimate] = field(default_factory=dict)
    resource_batches: tuple[tuple[str, ...], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def replay_hash(self) -> str:
        return stable_hash(self.to_dict())
