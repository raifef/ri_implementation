"""Closed-loop QEC execution with detector-only physical-controller feedback."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np

from google_rl_reimplementation.google_pure_v7.config import canonical_hash

from ..common import load_config, write_artifact
from ..contracts import ExperimentFamily, evidence_envelope, validate_provenance
from .decoder_steering import FrozenDecoderSteeringPolicy
from .interface import CodeConfig, Decoder
from .mwpm import MWPMDecoder


@dataclass(frozen=True)
class DetectorBatch:
    detector_events: np.ndarray
    observables: np.ndarray
    circuit_hash: str
    shots: int


class PhysicalController(Protocol):
    @property
    def controller_hash(self) -> str: ...
    def action(self, step: int) -> Mapping[str, float]: ...
    def observe_detector_events(self, detector_events: np.ndarray) -> None: ...


class StaticPhysicalController:
    """Deterministic control fixture that receives detector events but never logical truth."""

    def __init__(self, physical_error_probability: float) -> None:
        self.probability = float(physical_error_probability)
        self.detector_feedback: list[float] = []

    @property
    def controller_hash(self) -> str:
        return canonical_hash({"type": "static_physical_control", "probability": self.probability})

    def action(self, step: int) -> Mapping[str, float]:
        return {"physical_error_probability": self.probability, "step": float(step)}

    def observe_detector_events(self, detector_events: np.ndarray) -> None:
        self.detector_feedback.append(float(np.mean(np.asarray(detector_events, dtype=float))))


class StimSurfaceCodeSource:
    def __init__(self, code_config: CodeConfig, shots: int) -> None:
        if shots < 1:
            raise ValueError("shots must be positive")
        self.code_config = code_config
        self.shots = int(shots)
        self._sampler: Any | None = None
        self._circuit_hash: str | None = None

    def reset(self, seed: int) -> None:
        try:
            import stim
        except ImportError as error:
            raise RuntimeError("STIM_REQUIRED_FOR_DECODER_COUPLED_RUN") from error
        circuit = stim.Circuit.generated(
            self.code_config.circuit_family,
            distance=self.code_config.distance,
            rounds=self.code_config.rounds,
            after_clifford_depolarization=self.code_config.physical_error_probability,
        )
        self._sampler = circuit.compile_detector_sampler(seed=int(seed))
        self._circuit_hash = canonical_hash({"circuit": str(circuit), "config": self.code_config.to_dict()})

    def sample(self, control_action: Mapping[str, float]) -> DetectorBatch:
        if self._sampler is None or self._circuit_hash is None:
            raise RuntimeError("detector source must be reset before use")
        probability = float(control_action["physical_error_probability"])
        if not np.isclose(probability, self.code_config.physical_error_probability):
            raise RuntimeError("CONTROL_ACTION_AND_FROZEN_CIRCUIT_STATE_MISMATCH")
        events, observables = self._sampler.sample(self.shots, separate_observables=True)
        return DetectorBatch(np.asarray(events), np.asarray(observables), self._circuit_hash, self.shots)


def run_closed_loop(
    *,
    family: ExperimentFamily,
    controller: PhysicalController,
    source: StimSurfaceCodeSource,
    decoder: Decoder | None,
    steps: int,
    seed: int,
    steering_policy: FrozenDecoderSteeringPolicy | None = None,
) -> dict[str, Any]:
    if family not in {
        ExperimentFamily.CONTROL_ONLY,
        ExperimentFamily.CONTROL_PLUS_FIXED_DECODER,
        ExperimentFamily.CONTROL_PLUS_DECODER_STEERING,
    }:
        raise ValueError("closed-loop runner received a non-decoder experiment family")
    if family == ExperimentFamily.CONTROL_ONLY and decoder is not None:
        raise ValueError("control-only runs cannot execute a decoder")
    if family != ExperimentFamily.CONTROL_ONLY and decoder is None:
        raise ValueError("decoder-assisted runs require an explicit decoder")
    if family == ExperimentFamily.CONTROL_PLUS_DECODER_STEERING and steering_policy is None:
        raise ValueError("decoder-steering runs require a frozen steering policy")
    source.reset(seed)
    if decoder is not None:
        decoder.reset(source.code_config, seed + 1)
    event_rates = []
    logical_failures = []
    steering_actions = []
    circuit_hashes = []
    for step in range(steps):
        action = controller.action(step)
        batch = source.sample(action)
        circuit_hashes.append(batch.circuit_hash)
        event_rates.append(float(np.mean(batch.detector_events)))
        controller.observe_detector_events(batch.detector_events)
        if decoder is not None:
            decoded = decoder.decode(batch.detector_events, batch.observables)
            logical_failures.append(int(decoded.logical_failures or 0))
            if steering_policy is not None:
                steering = steering_policy.action(step, decoder.metrics())
                if steering is not None:
                    decoder.update_parameters(steering)
                    steering_actions.append({"step": step, "action": dict(steering)})
    if len(set(circuit_hashes)) != 1:
        raise RuntimeError("active circuit changed within a closed-loop run")
    decoder_metrics = dict(decoder.metrics()) if decoder is not None else None
    return {
        "experiment_family": family.value,
        "sequence": [
            "physical_control_policy",
            "qec_circuit_and_detector_generation",
            "detector_events",
            "decoder" if decoder is not None else "decoder_not_executed",
            "logical_prediction_and_metrics" if decoder is not None else "control_metrics_only",
            "optional_decoder_steering" if steering_policy is not None else "no_decoder_steering",
        ],
        "controller_reward_input": "detector_events_only",
        "hidden_logical_outcome_used_by_physical_controller": False,
        "control_metrics": {"mean_detector_event_rate": float(np.mean(event_rates)), "per_step_detector_event_rate": event_rates},
        "decoder_metrics": decoder_metrics,
        "logical_failures_per_step": logical_failures,
        "steering_actions": steering_actions,
        "steering_policy_hash": steering_policy.policy_hash if steering_policy else None,
        "circuit_hash": circuit_hashes[0],
        "steps": steps,
        "shots_per_step": source.shots,
        "control_and_decoder_contributions_reported_separately": True,
    }


def _configured() -> tuple[CodeConfig, dict[str, Any]]:
    config = load_config("decoder.json")
    code = CodeConfig(
        circuit_family=str(config["code_family"]),
        distance=int(config["distance"]),
        rounds=int(config["rounds"]),
        physical_error_probability=float(config["physical_error_probability"]),
    )
    return code, config


def validate_decoder() -> dict[str, Any]:
    code, config = _configured()
    decoder = MWPMDecoder()
    source = StimSurfaceCodeSource(code, int(config["shots"]))
    controller = StaticPhysicalController(code.physical_error_probability)
    result = run_closed_loop(
        family=ExperimentFamily.CONTROL_PLUS_FIXED_DECODER,
        controller=controller,
        source=source,
        decoder=decoder,
        steps=3,
        seed=21201,
    )
    contract = {
        "schema_version": "google-pure-v10-decoder-contract.v1",
        "interface": ["reset", "decode", "update_parameters", "metrics"],
        "reference_backend": "pymatching_mwpm",
        "deterministic_test_fixture": "deterministic_test_fixture",
        "neural_backend": "neural_decoder_untrained_stub",
        "silent_fallback_permitted": False,
        "controller_reward": "detector based",
        "experiment_families": [
            ExperimentFamily.CONTROL_ONLY.value,
            ExperimentFamily.CONTROL_PLUS_FIXED_DECODER.value,
            ExperimentFamily.CONTROL_PLUS_DECODER_STEERING.value,
        ],
        "code_config": code.to_dict(),
        "decoder_hash": result["decoder_metrics"]["decoder_hash"],
        **evidence_envelope(complete=True, mechanism_valid=True, claim_supported=True, paper_comparable=False),
    }
    contract = write_artifact("decoder/decoder_contract", contract, "Decoder Contract")
    payload = {
        "schema_version": "google-pure-v10-decoder-validation.v1",
        "contract_hash": contract["artifact_hash"],
        "closed_loop": result,
        "mwpm_executed": True,
        "silent_fallback_used": result["decoder_metrics"]["silent_fallback_used"],
        "neural_decoder_trained": False,
        "control_only_claims_contaminated": False,
        **evidence_envelope(complete=True, mechanism_valid=True, claim_supported=True, paper_comparable=False),
    }
    return write_artifact("decoder/decoder_validation", payload, "Decoder Validation")


def run_control_only() -> dict[str, Any]:
    code, config = _configured()
    result = run_closed_loop(
        family=ExperimentFamily.CONTROL_ONLY,
        controller=StaticPhysicalController(code.physical_error_probability),
        source=StimSurfaceCodeSource(code, int(config["shots"])),
        decoder=None,
        steps=3,
        seed=21211,
    )
    payload = {
        "schema_version": "google-pure-v10-control-only.v1",
        "result": result,
        "decoder_executed": False,
        "evidence_class": "stochastic_qec_simulation",
        **evidence_envelope(complete=True, mechanism_valid=True, claim_supported=True, paper_comparable=False),
    }
    return write_artifact("decoder/control_only_results", payload, "Control-only QEC Simulation")


def run_control_plus_decoder() -> dict[str, Any]:
    code, config = _configured()
    decoder = MWPMDecoder()
    controller = StaticPhysicalController(code.physical_error_probability)
    result = run_closed_loop(
        family=ExperimentFamily.CONTROL_PLUS_FIXED_DECODER,
        controller=controller,
        source=StimSurfaceCodeSource(code, int(config["shots"])),
        decoder=decoder,
        steps=4,
        seed=21221,
    )
    provenance = {
        "experiment_family": ExperimentFamily.CONTROL_PLUS_FIXED_DECODER.value,
        "controller_hash": controller.controller_hash,
        "decoder_hash": result["decoder_metrics"]["decoder_hash"],
        "plant_hash": result["circuit_hash"],
        "graph_hash": result["circuit_hash"],
        "protocol_hash": canonical_hash({"steps": 4, "shots": config["shots"], "code": code.to_dict()}),
        "seed": 21221,
        "drift_tape_hash": canonical_hash([code.physical_error_probability] * 4),
        "mode": "smoke",
        "qec_cycle_budget": 4 * int(config["shots"]) * code.rounds,
        "candidate_budget": 0,
        "observable_definition": "decoder logical failures compared with sampled Stim observables",
        "analysis_contract": "fixed MWPM decoder, no decoder steering",
    }
    validate_provenance(provenance)
    payload = {
        "schema_version": "google-pure-v10-closed-loop-results.v1",
        **provenance,
        "result": result,
        "evidence_class": "decoder_coupled_simulation",
        "control_only_metrics_merged": False,
        **evidence_envelope(complete=True, mechanism_valid=True, claim_supported=True, paper_comparable=False),
    }
    return write_artifact("decoder/closed_loop_results", payload, "Control plus Fixed-decoder Results")


def run_decoder_steering() -> dict[str, Any]:
    code, config = _configured()
    policy = FrozenDecoderSteeringPolicy(
        update_cadence=2,
        backend="pymatching_mwpm",
        source_defined=bool(config["decoder_steering_source_defined"]),
    )
    result = run_closed_loop(
        family=ExperimentFamily.CONTROL_PLUS_DECODER_STEERING,
        controller=StaticPhysicalController(code.physical_error_probability),
        source=StimSurfaceCodeSource(code, int(config["shots"])),
        decoder=MWPMDecoder(),
        steps=4,
        seed=21231,
        steering_policy=policy,
    )
    blockers = [] if policy.source_defined else ["DECODER_STEERING_NOT_SOURCE_DEFINED"]
    payload = {
        "schema_version": "google-pure-v10-decoder-steering.v1",
        "result": result,
        "source_defined": policy.source_defined,
        "decoder_training_data": policy.training_data_hash,
        "control_metrics": result["control_metrics"],
        "decoder_metrics": result["decoder_metrics"],
        **evidence_envelope(complete=True, mechanism_valid=True, claim_supported=False, paper_comparable=False, blocking_reasons=blockers),
    }
    return write_artifact("decoder/decoder_steering_results", payload, "Decoder-steering Results")
