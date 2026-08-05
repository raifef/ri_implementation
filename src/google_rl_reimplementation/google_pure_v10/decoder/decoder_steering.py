"""Explicitly frozen decoder-only steering policy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from google_rl_reimplementation.google_pure_v7.config import canonical_hash


@dataclass(frozen=True)
class FrozenDecoderSteeringPolicy:
    update_cadence: int
    backend: str
    source_defined: bool
    training_data_hash: str | None = None

    def __post_init__(self) -> None:
        if self.update_cadence < 1:
            raise ValueError("decoder update cadence must be positive")

    @property
    def policy_hash(self) -> str:
        return canonical_hash(
            {
                "update_cadence": self.update_cadence,
                "backend": self.backend,
                "source_defined": self.source_defined,
                "training_data_hash": self.training_data_hash,
            }
        )

    def action(self, step: int, metrics: Mapping[str, Any]) -> Mapping[str, float] | None:
        if step % self.update_cadence:
            return None
        if self.backend == "deterministic_test_fixture":
            failures = int(metrics.get("logical_failures", 0))
            return {"parity_threshold": float(1 if failures == 0 else 0)}
        if self.backend == "pymatching_mwpm":
            return {"edge_weight_scale": 1.0}
        raise ValueError("unsupported decoder-steering backend")
