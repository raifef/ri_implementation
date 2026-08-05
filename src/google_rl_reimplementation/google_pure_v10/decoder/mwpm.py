"""PyMatching MWPM decoder with explicit availability and no toy fallback."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from google_rl_reimplementation.google_pure_v7.config import canonical_hash

from .decoder_state import DecoderState
from .interface import CodeConfig, DecodeResult, Decoder


class DecoderUnavailableError(RuntimeError):
    """Raised when the requested reference decoder cannot be constructed."""


class MWPMDecoder(Decoder):
    backend = "pymatching_mwpm"

    def __init__(self) -> None:
        self._matching: Any | None = None
        self._state: DecoderState | None = None
        self._decode_calls = 0
        self._shots = 0
        self._failures = 0

    def reset(self, code_config: CodeConfig, seed: int) -> None:
        try:
            import pymatching
            import stim
        except ImportError as error:
            raise DecoderUnavailableError("MWPM_REQUIRES_STIM_AND_PYMATCHING") from error
        try:
            circuit = stim.Circuit.generated(
                code_config.circuit_family,
                distance=code_config.distance,
                rounds=code_config.rounds,
                after_clifford_depolarization=code_config.physical_error_probability,
            )
            detector_error_model = circuit.detector_error_model(decompose_errors=True)
            self._matching = pymatching.Matching.from_detector_error_model(detector_error_model)
        except Exception as error:
            raise DecoderUnavailableError("MWPM_CONSTRUCTION_FAILED_WITHOUT_FALLBACK") from error
        self._state = DecoderState(
            backend=self.backend,
            code_config=code_config,
            seed=int(seed),
            parameter_version=0,
            parameters={"edge_weight_scale": 1.0},
            update_cadence=None,
            training_data_hash=None,
        )
        self._decode_calls = self._shots = self._failures = 0

    def decode(self, detector_events: np.ndarray, observables: np.ndarray | None = None) -> DecodeResult:
        if self._matching is None or self._state is None:
            raise RuntimeError("MWPM decoder must be reset before use")
        events = np.asarray(detector_events, dtype=np.uint8)
        if events.ndim == 1:
            events = events[None, :]
        if events.ndim != 2:
            raise ValueError("detector events must be one- or two-dimensional")
        predictions = np.asarray([np.atleast_1d(self._matching.decode(row)) for row in events], dtype=np.uint8)
        failures = None
        if observables is not None:
            truth = np.asarray(observables, dtype=np.uint8).reshape(len(events), -1)
            if truth.shape != predictions.shape:
                raise ValueError(f"observable shape mismatch: {truth.shape} != {predictions.shape}")
            failures = int(np.count_nonzero(np.any(predictions != truth, axis=1)))
            self._failures += failures
        self._decode_calls += 1
        self._shots += len(events)
        return DecodeResult(
            predicted_observables=predictions,
            backend=self.backend,
            logical_failures=failures,
            shots=len(events),
            metadata={"decoder_hash": self._state.decoder_hash, "silent_fallback_used": False},
        )

    def update_parameters(self, steering_action: Mapping[str, float]) -> None:
        if self._state is None:
            raise RuntimeError("MWPM decoder must be reset before steering")
        if set(steering_action) != {"edge_weight_scale"}:
            raise ValueError("MWPM steering supports only an explicitly declared edge_weight_scale")
        value = float(steering_action["edge_weight_scale"])
        if not np.isclose(value, 1.0):
            raise RuntimeError("NONIDENTITY_MWPM_STEERING_NOT_IMPLEMENTED_FAIL_CLOSED")
        self._state = self._state.update({"edge_weight_scale": value})

    def metrics(self) -> Mapping[str, Any]:
        return {
            "backend": self.backend,
            "reference_backend": True,
            "configured": self._state is not None,
            "decoder_hash": self._state.decoder_hash if self._state else None,
            "decode_calls": self._decode_calls,
            "shots": self._shots,
            "logical_failures": self._failures,
            "logical_error_rate": self._failures / self._shots if self._shots else None,
            "silent_fallback_used": False,
            "parameters": dict(self._state.parameters) if self._state else None,
        }
