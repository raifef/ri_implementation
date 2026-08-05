"""Experiment-family, scientific-classification, and provenance contracts."""
from __future__ import annotations

from enum import Enum
from typing import Any, Iterable, Mapping

from google_rl_reimplementation.google_pure_v7.config import canonical_hash

from .common import write_artifact


class ExperimentFamily(str, Enum):
    CONTROL_ONLY = "CONTROL_ONLY"
    CONTROL_PLUS_FIXED_DECODER = "CONTROL_PLUS_FIXED_DECODER"
    CONTROL_PLUS_DECODER_STEERING = "CONTROL_PLUS_DECODER_STEERING"
    NATURAL_DRIFT_SPECTRAL_SUPPRESSION_V10 = "NATURAL_DRIFT_SPECTRAL_SUPPRESSION_V10"
    STEP_RESPONSE_INJECTED_DRIFT_V10 = "STEP_RESPONSE_INJECTED_DRIFT_V10"


EVIDENCE_CLASSES = frozenset(
    {
        "analytic_scaling_model",
        "stochastic_qec_simulation",
        "decoder_coupled_simulation",
        "public_hardware_result",
    }
)
PROVENANCE_FIELDS = frozenset(
    {
        "experiment_family",
        "controller_hash",
        "decoder_hash",
        "plant_hash",
        "graph_hash",
        "protocol_hash",
        "seed",
        "drift_tape_hash",
        "mode",
        "qec_cycle_budget",
        "candidate_budget",
        "observable_definition",
        "analysis_contract",
    }
)


def validate_provenance(record: Mapping[str, Any]) -> None:
    missing = sorted(PROVENANCE_FIELDS - set(record))
    if missing:
        raise ValueError(f"result provenance is incomplete: {missing}")
    if record["experiment_family"] not in {item.value for item in ExperimentFamily}:
        raise ValueError("unknown experiment family")


def validate_evidence_class(label: str, *, decoder_executed: bool) -> None:
    if label not in EVIDENCE_CLASSES:
        raise ValueError("unknown evidence class")
    if label == "decoder_coupled_simulation" and not decoder_executed:
        raise ValueError("analytic or control-only data cannot be labeled decoder-coupled")


def enforce_family_separation(records: Iterable[Mapping[str, Any]]) -> None:
    families = {str(row["experiment_family"]) for row in records}
    decoder_families = {
        ExperimentFamily.CONTROL_PLUS_FIXED_DECODER.value,
        ExperimentFamily.CONTROL_PLUS_DECODER_STEERING.value,
    }
    if ExperimentFamily.CONTROL_ONLY.value in families and families & decoder_families:
        raise ValueError("control-only and decoder-assisted values cannot be merged into one estimand")


def evidence_envelope(
    *,
    complete: bool,
    mechanism_valid: bool,
    claim_supported: bool,
    paper_comparable: bool,
    blocking_reasons: Iterable[str] = (),
) -> dict[str, Any]:
    blockers = list(dict.fromkeys(map(str, blocking_reasons)))
    if claim_supported and (not complete or not mechanism_valid or blockers):
        raise ValueError("a supported claim requires complete, valid, unblocked evidence")
    if paper_comparable and not claim_supported:
        raise ValueError("paper comparability requires a supported claim")
    return {
        "artifact_complete": bool(complete),
        "mechanism_valid": bool(mechanism_valid),
        "claim_supported": bool(claim_supported),
        "paper_comparable": bool(paper_comparable),
        "blocking_reasons": blockers,
    }


def corrected_fault_contract() -> dict[str, Any]:
    payload = {
        "schema_version": "google-pure-v10-corrected-fault-contract.v1",
        "entropy": [
            "ENTROPY_IMPLEMENTATION_PASS",
            "ENTROPY_SWEEP_NOT_OPERATIONAL",
            "ENTROPY_REWARD_BALANCE_NOT_SOURCE_IDENTIFIABLE",
        ],
        "exploration": [
            "CURRENT_OPERATIONAL_POLICY_SCALE_TOO_LARGE",
            "POLICY_SCALE_FAILS_TO_CONTRACT",
            "MINIMUM_SCALE_FLOOR_EFFECT_NOT_ESTABLISHED",
        ],
        "temporal": [
            "TEMPORAL_IMPLEMENTATION_PASS",
            "TEMPORAL_EVALUATION_PROTOCOL_FAILURE",
            "PHASE_AND_WINDOW_ALIASING",
        ],
        "step_response": [
            "STEP_RESPONSE_TOO_SLOW_OR_UNDERRESOLVED",
            "PPO_CLIPPING_CAUSAL_ROLE_UNESTABLISHED",
            "LEARNING_RATE_CAUSAL_ROLE_UNESTABLISHED",
        ],
        "behavior_changed": False,
        **evidence_envelope(complete=True, mechanism_valid=True, claim_supported=True, paper_comparable=False),
    }
    return write_artifact("corrected_fault_contract", payload, "Corrected Scientific Fault Contract")


def hash_analysis_contract(value: Mapping[str, Any]) -> str:
    return canonical_hash(dict(value))
