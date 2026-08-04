"""Typed schemas shared by the v3 dataset and analysis layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class ReproductionStatus(StrEnum):
    EXACTLY_REPRODUCED = "EXACTLY_REPRODUCED"
    REPRODUCED_WITH_DOCUMENTED_APPROXIMATION = "REPRODUCED_WITH_DOCUMENTED_APPROXIMATION"
    NOT_REPRODUCIBLE_FROM_RELEASED_DATA = "NOT_REPRODUCIBLE_FROM_RELEASED_DATA"
    ANALYSIS_DEFINITION_AMBIGUOUS = "ANALYSIS_DEFINITION_AMBIGUOUS"


class SurrogateValidationOutcome(StrEnum):
    VALIDATED = "EMPIRICAL_SURROGATE_VALIDATED"
    PARTIALLY_VALIDATED = "EMPIRICAL_SURROGATE_PARTIALLY_VALIDATED"
    REJECTED = "EMPIRICAL_SURROGATE_REJECTED"


@dataclass(frozen=True)
class CircuitShape:
    measurements: int
    detectors: int
    observables: int
    sweep_bits: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    data_dir: str
    code_family: str
    distance: int
    condition: str
    subgrid: str
    basis: str
    rounds: int
    shots: int
    qubit_coords: tuple[tuple[int, int], ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["qubit_coords"] = [list(x) for x in self.qubit_coords]
        return value


@dataclass(frozen=True)
class LogicalEstimate:
    logical_error_per_cycle: float
    intercept: float
    slope: float
    standard_error: float
    method: str
    points: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    code: str
    message: str
    severity: str = "ERROR"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


OFFICIAL_RELEASE = {
    "record_id": "18896801",
    "doi": "10.5281/zenodo.18896801",
    "version": "2.0.0",
    "title": 'Data for "Reinforcement Learning Control of Quantum Error Correction"',
    "creator": "Google Quantum AI",
    "archive_name": "google_reinforcement_learning_qec.zip",
    "archive_bytes": 7_786_791_716,
    "archive_md5": "ca54323082fcd0e3671d5b90ce45d85c",
    "archive_sha256": "39563ad104bcbec2e36907373b25d176cf7f2a2e3852d8390623223dadf96e76",
    "record_url": "https://zenodo.org/records/18896801",
}


CERTIFICATION_SEEDS = tuple(range(8101, 8113))

