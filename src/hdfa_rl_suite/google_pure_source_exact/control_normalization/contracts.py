"""Immutable contracts for empirical detector-sensitivity calibration.

The public method fixes the equation and the simultaneous control-type sweep.
Numerical sweep grids, shot budgets, fit diagnostics, and acceptance tolerances
are not public and are therefore explicit preregistered fields here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "google-pure-source-exact-control-normalization.v1"
PAPER_NORMALIZATION_METHOD = "EMPIRICAL_DETECTOR_SENSITIVITY_FIG_S3"
NON_PAPER_NORMALIZATION_ABLATION = "NON_PAPER_NORMALIZATION_ABLATION"


class SourceIdentifiability(StrEnum):
    SOURCE_LITERAL = "SOURCE_LITERAL"
    SOURCE_DERIVED = "SOURCE_DERIVED"
    SOURCE_REFERENCED_PRIMARY_METHOD = "SOURCE_REFERENCED_PRIMARY_METHOD"
    SOURCE_UNSPECIFIED_PREREGISTERED = "SOURCE_UNSPECIFIED_PREREGISTERED"
    NOT_PUBLICLY_IDENTIFIABLE = "NOT_PUBLICLY_IDENTIFIABLE"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_hash(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _require_finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class ControlTypeSpec:
    """One paper-level control type perturbed across all registered gates."""

    control_type: str
    gate_ids: tuple[str, ...]
    native_unit: str
    reference_value_native: float
    sweep_sigmas_native: tuple[float, ...]
    fit_interval_native: tuple[float, float]
    stim_error_channel: str
    synthetic_probability_gain: float
    source_identifiability: SourceIdentifiability = SourceIdentifiability.SOURCE_DERIVED

    def __post_init__(self) -> None:
        if not self.control_type or not self.native_unit:
            raise ValueError("control type and native unit are required")
        if not self.gate_ids or len(set(self.gate_ids)) != len(self.gate_ids):
            raise ValueError("gate_ids must be a non-empty unique registry")
        sigmas = tuple(_require_finite("sweep sigma", value) for value in self.sweep_sigmas_native)
        if len(sigmas) < 4 or sigmas[0] != 0.0 or any(value < 0 for value in sigmas):
            raise ValueError("sweep requires zero plus at least three non-negative sigma values")
        if any(right <= left for left, right in zip(sigmas, sigmas[1:])):
            raise ValueError("sweep sigmas must be strictly increasing")
        lower, upper = map(float, self.fit_interval_native)
        if lower < 0 or not lower < upper or upper > sigmas[-1]:
            raise ValueError("fit interval must lie inside the sweep range")
        channels = {
            "after_clifford_depolarization",
            "before_round_data_depolarization",
            "before_measure_flip_probability",
            "after_reset_flip_probability",
        }
        if self.stim_error_channel not in channels:
            raise ValueError(f"unsupported Stim error channel: {self.stim_error_channel}")
        if _require_finite("synthetic_probability_gain", self.synthetic_probability_gain) <= 0:
            raise ValueError("synthetic probability gain must be positive")
        object.__setattr__(self, "gate_ids", tuple(str(item) for item in self.gate_ids))
        object.__setattr__(self, "sweep_sigmas_native", sigmas)
        object.__setattr__(self, "fit_interval_native", (lower, upper))
        object.__setattr__(self, "reference_value_native", float(self.reference_value_native))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ControlTypeSpec":
        payload = dict(value)
        payload["gate_ids"] = tuple(payload["gate_ids"])
        payload["sweep_sigmas_native"] = tuple(payload["sweep_sigmas_native"])
        payload["fit_interval_native"] = tuple(payload["fit_interval_native"])
        payload["source_identifiability"] = SourceIdentifiability(payload["source_identifiability"])
        return cls(**payload)


@dataclass(frozen=True)
class FrozenReference:
    reference_policy_hash: str
    circuit_hash: str
    detector_set_hash: str
    detector_ids: tuple[str, ...]
    parameter_registry_hash: str

    def __post_init__(self) -> None:
        values = (
            self.reference_policy_hash,
            self.circuit_hash,
            self.detector_set_hash,
            self.parameter_registry_hash,
        )
        if any(not str(value) for value in values) or not self.detector_ids:
            raise ValueError("frozen reference hashes and detector ids are required")
        if canonical_hash(list(self.detector_ids)) != self.detector_set_hash:
            raise ValueError("detector_set_hash does not match detector_ids")
        object.__setattr__(self, "detector_ids", tuple(str(item) for item in self.detector_ids))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrozenReference":
        payload = dict(value)
        payload["detector_ids"] = tuple(payload["detector_ids"])
        return cls(**payload)


@dataclass(frozen=True)
class SweepProtocol:
    candidates_per_sigma: int
    shots_per_candidate: int
    qec_rounds_per_shot: int
    perturbation_seed: int
    detector_seed: int
    edr_unit: str = "percentage_point"

    def __post_init__(self) -> None:
        if min(self.candidates_per_sigma, self.shots_per_candidate, self.qec_rounds_per_shot) <= 0:
            raise ValueError("candidate, shot, and QEC-round budgets must be positive")
        if self.edr_unit != "percentage_point":
            raise ValueError("the canonical fit uses detector-rate percentage points")

    @property
    def qec_cycles_per_sigma(self) -> int:
        return self.candidates_per_sigma * self.shots_per_candidate * self.qec_rounds_per_shot

    @property
    def protocol_hash(self) -> str:
        return canonical_hash(asdict(self))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SweepProtocol":
        return cls(**dict(value))


@dataclass(frozen=True)
class FitRules:
    confidence_z: float = 1.959963984540054
    minimum_r_squared: float = 0.94
    maximum_monotonicity_z: float = 3.0
    maximum_quartic_z: float = 3.0
    maximum_reduced_chi_squared: float = 4.0
    minimum_positive_coefficient_z: float = 3.0
    stability_relative_tolerance: float = 0.25
    isotropy_relative_tolerance: float = 0.35
    prediction_relative_tolerance: float = 0.35

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if _require_finite(name, value) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.minimum_r_squared > 1:
            raise ValueError("minimum_r_squared cannot exceed one")

    @property
    def rules_hash(self) -> str:
        return canonical_hash(asdict(self))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FitRules":
        return cls(**dict(value))


@dataclass(frozen=True)
class SweepPoint:
    control_type: str
    sigma_native: float
    detector_events: int
    detector_opportunities: int
    candidates: int
    shots_per_candidate: int
    qec_cycles: int
    perturbation_seed: int
    detector_seed: int
    candidate_detector_events: tuple[int, ...] = ()
    candidate_detector_opportunities: int = 0

    def __post_init__(self) -> None:
        if self.sigma_native < 0 or self.detector_events < 0:
            raise ValueError("sigma and event count must be non-negative")
        if self.detector_opportunities <= 0 or self.detector_events > self.detector_opportunities:
            raise ValueError("invalid detector event count")
        if min(self.candidates, self.shots_per_candidate, self.qec_cycles) <= 0:
            raise ValueError("measurement budgets must be positive")
        candidate_events = tuple(int(value) for value in self.candidate_detector_events)
        if candidate_events:
            if len(candidate_events) != self.candidates or self.candidate_detector_opportunities <= 0:
                raise ValueError("candidate-level detector counts must align with candidates")
            if any(not 0 <= value <= self.candidate_detector_opportunities for value in candidate_events):
                raise ValueError("invalid candidate-level detector count")
            if sum(candidate_events) != self.detector_events:
                raise ValueError("candidate detector counts do not sum to aggregate events")
            if self.candidates * self.candidate_detector_opportunities != self.detector_opportunities:
                raise ValueError("candidate opportunities do not sum to aggregate opportunities")
        object.__setattr__(self, "candidate_detector_events", candidate_events)

    @property
    def edr_fraction(self) -> float:
        return self.detector_events / self.detector_opportunities

    @property
    def edr_percentage_points(self) -> float:
        return 100.0 * self.edr_fraction

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SweepPoint":
        payload = dict(value)
        payload["candidate_detector_events"] = tuple(payload.get("candidate_detector_events", ()))
        payload.setdefault("candidate_detector_opportunities", 0)
        return cls(**payload)


@dataclass(frozen=True)
class SweepResult:
    schema_version: str
    control_type: str
    native_unit: str
    reference: FrozenReference
    protocol: SweepProtocol
    fit_interval_native: tuple[float, float]
    points: tuple[SweepPoint, ...]
    simultaneous_gate_ids: tuple[str, ...]
    plant_hash: str
    shard_index: int = 0
    shard_count: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported sweep schema")
        if not self.points or any(point.control_type != self.control_type for point in self.points):
            raise ValueError("sweep points do not match control type")
        sigmas = [point.sigma_native for point in self.points]
        if len(sigmas) != len(set(sigmas)):
            raise ValueError("duplicate sigma measurements are forbidden")
        if not 0 <= self.shard_index < self.shard_count:
            raise ValueError("invalid shard identity")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SweepResult":
        payload = dict(value)
        payload["reference"] = FrozenReference.from_dict(payload["reference"])
        payload["protocol"] = SweepProtocol.from_dict(payload["protocol"])
        payload["fit_interval_native"] = tuple(payload["fit_interval_native"])
        payload["points"] = tuple(SweepPoint.from_dict(item) for item in payload["points"])
        payload["simultaneous_gate_ids"] = tuple(payload["simultaneous_gate_ids"])
        return cls(**payload)


@dataclass(frozen=True)
class SensitivityFit:
    control_type: str
    native_unit: str
    edr0_percentage_points: float
    quadratic_coefficient_per_native_squared: float
    sigma0_native: float
    sigma0_confidence_interval_95: tuple[float, float]
    coefficient_confidence_interval_95: tuple[float, float]
    parameter_covariance: tuple[tuple[float, float], tuple[float, float]]
    fit_interval_native: tuple[float, float]
    fit_point_count: int
    r_squared: float
    reduced_chi_squared: float
    quartic_z_score: float
    monotonicity_max_z: float
    fit_rules_hash: str
    detector_set_hash: str
    circuit_hash: str
    reference_policy_hash: str
    shot_budget: int
    qec_cycle_budget: int
    uncertainty_method: str
    passed: bool
    blocking_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.sigma0_native <= 0 or self.quadratic_coefficient_per_native_squared <= 0:
            raise ValueError("accepted sensitivity must be positive")
        if self.passed and self.blocking_reasons:
            raise ValueError("a passing fit cannot have blocking reasons")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SensitivityFit":
        payload = dict(value)
        for name in (
            "sigma0_confidence_interval_95",
            "coefficient_confidence_interval_95",
            "fit_interval_native",
            "blocking_reasons",
        ):
            payload[name] = tuple(payload[name])
        payload["parameter_covariance"] = tuple(tuple(row) for row in payload["parameter_covariance"])
        return cls(**payload)


@dataclass(frozen=True)
class CalibrationBundle:
    schema_version: str
    normalization_method: str
    reference: FrozenReference
    control_specs: tuple[ControlTypeSpec, ...]
    fits: tuple[SensitivityFit, ...]
    fit_rules: FitRules
    config_hash: str
    plant_hash: str
    source_contract_hash: str
    source_identifiability: Mapping[str, str]
    artifact_complete: bool
    mathematical_contract_pass: bool
    protocol_contract_pass: bool
    source_structure_match: bool
    quantitative_match: bool
    paper_comparable: bool
    blocking_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported calibration schema")
        if self.normalization_method != PAPER_NORMALIZATION_METHOD:
            raise ValueError("source-exact bundle must use empirical detector sensitivity")
        names = [item.control_type for item in self.control_specs]
        fit_names = [item.control_type for item in self.fits]
        if len(names) != len(set(names)) or set(names) != set(fit_names):
            raise ValueError("control specifications and sensitivity fits must match exactly")
        if self.paper_comparable and self.blocking_reasons:
            raise ValueError("paper-comparable bundle cannot have blockers")
        if self.artifact_complete and not all(item.passed for item in self.fits):
            raise ValueError("complete bundle cannot contain a rejected fit")
        object.__setattr__(self, "source_identifiability", dict(self.source_identifiability))

    @property
    def bundle_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate_context(self, *, circuit_hash: str, detector_set_hash: str,
                         reference_policy_hash: str | None = None) -> None:
        failures = []
        if circuit_hash != self.reference.circuit_hash:
            failures.append("circuit hash changed")
        if detector_set_hash != self.reference.detector_set_hash:
            failures.append("detector set hash changed")
        if reference_policy_hash is not None and reference_policy_hash != self.reference.reference_policy_hash:
            failures.append("reference policy hash changed")
        if failures:
            raise StaleCalibrationError("; ".join(failures))

    def fit_by_type(self) -> dict[str, SensitivityFit]:
        return {item.control_type: item for item in self.fits}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CalibrationBundle":
        payload = dict(value)
        payload["reference"] = FrozenReference.from_dict(payload["reference"])
        payload["control_specs"] = tuple(ControlTypeSpec.from_dict(item) for item in payload["control_specs"])
        payload["fits"] = tuple(SensitivityFit.from_dict(item) for item in payload["fits"])
        payload["fit_rules"] = FitRules.from_dict(payload["fit_rules"])
        payload["blocking_reasons"] = tuple(payload.get("blocking_reasons", ()))
        return cls(**payload)


@dataclass(frozen=True)
class IterationRecord:
    iteration_id: str
    source_commit: str
    code_hash: str
    config_hash: str
    plant_hash: str
    protocol_hash: str
    analysis_hash: str
    seed_registry_hash: str
    changes_from_previous_iteration: tuple[str, ...]
    failed_gates: tuple[str, ...]
    numerical_results: Mapping[str, Any]
    next_diagnosis: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StaleCalibrationError(RuntimeError):
    """Raised when a fitted scale is used with a changed circuit or detector set."""


def build_source_contract() -> dict[str, Any]:
    fields = [
        {
            "field": "control_type_grouping",
            "value": "perturb all gates depending on one control-parameter type",
            "status": SourceIdentifiability.SOURCE_LITERAL,
            "source": "Supplementary Information II.C, Fig. S3",
        },
        {
            "field": "perturbation_distribution",
            "value": "independent Gaussian perturbations N(0, sigma) across registered gates of the selected type",
            "status": SourceIdentifiability.SOURCE_LITERAL,
            "source": "Supplementary Information II.C, Fig. S3",
        },
        {
            "field": "sweep_axis",
            "value": "variance sigma^2",
            "status": SourceIdentifiability.SOURCE_LITERAL,
            "source": "Supplementary Information II.C, Fig. S3",
        },
        {
            "field": "response_model",
            "value": "EDR = EDR0 + (sigma/sigma0)^2",
            "status": SourceIdentifiability.SOURCE_LITERAL,
            "source": "Supplementary Information II.C, Fig. S3",
        },
        {
            "field": "downstream_coordinates",
            "value": "rescale all controls by fitted sensitivity coefficients and initialize a spherical Gaussian",
            "status": SourceIdentifiability.SOURCE_LITERAL,
            "source": "Supplementary Information II.C",
        },
        {
            "field": "fig5a_detector_evaluation",
            "value": "distance-3 surface-code circuit simulated in Stim; EDR counted from sampled detector events",
            "status": SourceIdentifiability.SOURCE_LITERAL,
            "source": "Supplementary Information VI.A",
        },
        {
            "field": "sweep_grid_and_shot_budget",
            "value": "profile configuration",
            "status": SourceIdentifiability.SOURCE_UNSPECIFIED_PREREGISTERED,
            "source": "not numerically reported in the public paper",
        },
        {
            "field": "fit_interval_and_acceptance_thresholds",
            "value": "frozen FitRules and per-type fit intervals",
            "status": SourceIdentifiability.SOURCE_UNSPECIFIED_PREREGISTERED,
            "source": "not numerically reported in the public paper",
        },
        {
            "field": "synthetic_gate_sensitivity_distribution",
            "value": "fixed config gains used only to construct Stim circuit noise; never used as normalization output",
            "status": SourceIdentifiability.SOURCE_UNSPECIFIED_PREREGISTERED,
            "source": "Supplement VI states random Omega_i but does not publish the draw distribution",
        },
        {
            "field": "proprietary_controller_code",
            "value": None,
            "status": SourceIdentifiability.NOT_PUBLICLY_IDENTIFIABLE,
            "source": "Nature code-availability statement",
        },
    ]
    payload = {
        "schema_version": "google-pure-source-exact-source-contract.v1",
        "paper_doi": "10.1038/s41586-026-10759-2",
        "fields": fields,
        "clean_room": True,
        "old_version_artifacts_are_read_only": True,
    }
    payload["source_contract_hash"] = canonical_hash(payload)
    return payload
