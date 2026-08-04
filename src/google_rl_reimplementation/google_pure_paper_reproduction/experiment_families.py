"""Experiment-family identities and non-conflation rules."""
from __future__ import annotations

from enum import StrEnum
from typing import Any, Iterable, Mapping


class ExperimentFamily(StrEnum):
    PUBLIC_ENDPOINT_DATA_REPRODUCTION = "PUBLIC_ENDPOINT_DATA_REPRODUCTION"
    FIGURE5A_REAL_TIME_STEERING = "FIGURE5A_REAL_TIME_STEERING"
    FIGURE5B_SPARSE_SCALING = "FIGURE5B_SPARSE_SCALING"
    FIGURE5C_CONVERGENCE_LAW = "FIGURE5C_CONVERGENCE_LAW"
    NATURAL_DRIFT_SPECTRAL_SUPPRESSION = "NATURAL_DRIFT_SPECTRAL_SUPPRESSION"
    RANDOMIZED_RECOVERY_AFTER_SPOIL = "RANDOMIZED_RECOVERY_AFTER_SPOIL"
    STEP_RESPONSE_INJECTED_DRIFT = "STEP_RESPONSE_INJECTED_DRIFT"
    PUBLIC_TABLE_REPRODUCTION = "PUBLIC_TABLE_REPRODUCTION"


class RunMode(StrEnum):
    SMOKE = "smoke"
    VALIDATION = "validation"
    REFERENCE = "reference"
    PAPER_SCALE = "paper-scale"


class EvidenceClass(StrEnum):
    PUBLIC_EXACT = "PUBLIC_DATA_DIRECTLY_REPRODUCIBLE"
    SYNTHETIC = "PUBLIC_SIMULATION_ANALOGUE_REPRODUCIBLE"
    VISUAL = "VISUAL_ONLY_TARGET"
    UNIDENTIFIABLE = "NOT_IDENTIFIABLE_FROM_PUBLIC_INFORMATION"
    NOT_IMPLEMENTED = "NOT_YET_IMPLEMENTED"
    MISMATCHED = "IMPLEMENTED_BUT_CURRENTLY_MISMATCHED"


FINAL_MODES = {RunMode.REFERENCE.value, RunMode.PAPER_SCALE.value}
PUBLIC_FAMILIES = {
    ExperimentFamily.PUBLIC_ENDPOINT_DATA_REPRODUCTION.value,
    ExperimentFamily.PUBLIC_TABLE_REPRODUCTION.value,
}
SYNTHETIC_FAMILIES = {item.value for item in ExperimentFamily} - PUBLIC_FAMILIES
CERTIFICATION_SEEDS = set(range(12101, 12113))
RETIRED_SEEDS = {10101}


def require_family(value: str | ExperimentFamily) -> str:
    return ExperimentFamily(value).value


def guard_seed(seed: int) -> None:
    if int(seed) in CERTIFICATION_SEEDS | RETIRED_SEEDS:
        raise RuntimeError(f"seed {seed} is reserved or retired and cannot be consumed")


def evidence_class_for(family: str | ExperimentFamily) -> str:
    return EvidenceClass.PUBLIC_EXACT.value if require_family(family) in PUBLIC_FAMILIES else EvidenceClass.SYNTHETIC.value


def assert_claim_compatible(claim: Mapping[str, Any], family: str | ExperimentFamily) -> None:
    expected = require_family(family)
    allowed = set(claim.get("experiment_families", ()))
    if expected not in allowed:
        raise RuntimeError(f"claim {claim.get('claim_id')} cannot be combined with {expected}")
    direct = expected in PUBLIC_FAMILIES
    if bool(claim.get("public_data_direct")) != direct and len(allowed) == 1:
        raise RuntimeError("public-data and synthetic claim semantics are conflated")


def assert_merge_compatible(records: Iterable[Mapping[str, Any]]) -> None:
    rows = list(records)
    if not rows:
        raise RuntimeError("cannot merge zero records")
    keys = ("experiment_family", "protocol_hash", "controller_hash", "plant_hash", "graph_hash", "mode")
    for key in keys:
        values = {row["provenance"].get(key) for row in rows}
        if len(values) != 1:
            raise RuntimeError(f"incompatible shard merge: mixed {key} values {sorted(map(str, values))}")


def final_evidence_allowed(*, mode: str, complete: bool, scientifically_valid: bool) -> bool:
    return mode in FINAL_MODES and complete and scientifically_valid

