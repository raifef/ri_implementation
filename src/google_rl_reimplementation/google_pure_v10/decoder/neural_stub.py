"""Neural-decoder interface stub that never claims a trained model."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .interface import CodeConfig, DecodeResult, Decoder


class NeuralDecoderStub(Decoder):
    backend = "neural_decoder_untrained_stub"

    def __init__(self) -> None:
        self._config: CodeConfig | None = None

    def reset(self, code_config: CodeConfig, seed: int) -> None:
        del seed
        self._config = code_config

    def decode(self, detector_events: np.ndarray, observables: np.ndarray | None = None) -> DecodeResult:
        del detector_events, observables
        raise RuntimeError("NEURAL_DECODER_MODEL_NOT_TRAINED")

    def update_parameters(self, steering_action: Mapping[str, float]) -> None:
        del steering_action
        raise RuntimeError("NEURAL_DECODER_MODEL_NOT_TRAINED")

    def metrics(self) -> Mapping[str, Any]:
        return {
            "backend": self.backend,
            "configured": self._config is not None,
            "trained": False,
            "reference_backend": False,
            "claim_supported": False,
            "blocking_reasons": ["NEURAL_DECODER_MODEL_NOT_TRAINED"],
        }
