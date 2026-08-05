"""Stable decoder API plus a deterministic test fixture."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class CodeConfig:
    circuit_family: str
    distance: int
    rounds: int
    physical_error_probability: float

    def __post_init__(self) -> None:
        if self.distance < 3 or self.distance % 2 == 0 or self.rounds < 1:
            raise ValueError("surface-code distance must be odd and rounds positive")
        if not 0 < self.physical_error_probability < 0.5:
            raise ValueError("physical error probability must lie in (0,0.5)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecodeResult:
    predicted_observables: np.ndarray
    backend: str
    logical_failures: int | None
    shots: int
    metadata: Mapping[str, Any]


class Decoder(ABC):
    @abstractmethod
    def reset(self, code_config: CodeConfig, seed: int) -> None:
        """Reset all decoder state for one frozen code configuration."""

    @abstractmethod
    def decode(self, detector_events: np.ndarray, observables: np.ndarray | None = None) -> DecodeResult:
        """Decode one or more shots without changing physical-controller rewards."""

    @abstractmethod
    def update_parameters(self, steering_action: Mapping[str, float]) -> None:
        """Apply an explicitly supported decoder-only steering action."""

    @abstractmethod
    def metrics(self) -> Mapping[str, Any]:
        """Return decoder-only metrics and provenance."""


class DeterministicParityDecoder(Decoder):
    """Small deterministic fixture; never a reference-run fallback."""

    backend = "deterministic_test_fixture"

    def __init__(self) -> None:
        self._threshold = 1
        self._decode_calls = 0
        self._shots = 0
        self._failures = 0
        self._configured = False

    def reset(self, code_config: CodeConfig, seed: int) -> None:
        del seed
        self._config = code_config
        self._threshold = 1
        self._decode_calls = self._shots = self._failures = 0
        self._configured = True

    def decode(self, detector_events: np.ndarray, observables: np.ndarray | None = None) -> DecodeResult:
        if not self._configured:
            raise RuntimeError("decoder must be reset before use")
        events = np.asarray(detector_events, dtype=np.uint8)
        if events.ndim == 1:
            events = events[None, :]
        prediction = (np.sum(events, axis=1) % 2 >= self._threshold).astype(np.uint8)[:, None]
        failures = None
        if observables is not None:
            truth = np.asarray(observables, dtype=np.uint8).reshape(len(events), -1)
            if truth.shape != prediction.shape:
                raise ValueError("fixture observable shape mismatch")
            failures = int(np.count_nonzero(np.any(prediction != truth, axis=1)))
            self._failures += failures
        self._decode_calls += 1
        self._shots += len(events)
        return DecodeResult(prediction, self.backend, failures, len(events), {"threshold": self._threshold})

    def update_parameters(self, steering_action: Mapping[str, float]) -> None:
        if set(steering_action) != {"parity_threshold"}:
            raise ValueError("fixture supports only parity_threshold steering")
        value = int(steering_action["parity_threshold"])
        if value not in {0, 1}:
            raise ValueError("fixture parity threshold must be zero or one")
        self._threshold = value

    def metrics(self) -> Mapping[str, Any]:
        return {
            "backend": self.backend,
            "reference_backend": False,
            "decode_calls": self._decode_calls,
            "shots": self._shots,
            "logical_failures": self._failures,
            "logical_error_rate": self._failures / self._shots if self._shots else None,
            "parameters": {"parity_threshold": self._threshold},
        }
