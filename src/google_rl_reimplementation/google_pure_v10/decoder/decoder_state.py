"""Immutable decoder configuration and steering provenance."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from google_rl_reimplementation.google_pure_v7.config import canonical_hash

from .interface import CodeConfig


@dataclass(frozen=True)
class DecoderState:
    backend: str
    code_config: CodeConfig
    seed: int
    parameter_version: int
    parameters: Mapping[str, float]
    update_cadence: int | None
    training_data_hash: str | None

    @property
    def decoder_hash(self) -> str:
        return canonical_hash(
            {
                "backend": self.backend,
                "code_config": self.code_config.to_dict(),
                "seed": self.seed,
                "parameter_version": self.parameter_version,
                "parameters": dict(self.parameters),
                "update_cadence": self.update_cadence,
                "training_data_hash": self.training_data_hash,
            }
        )

    def update(self, parameters: Mapping[str, float]) -> "DecoderState":
        return replace(self, parameter_version=self.parameter_version + 1, parameters=dict(parameters))
