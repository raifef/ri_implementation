"""Hardware/simulator boundary for bootstrap calibration.

Backends must return recorded observations only.  Fitting, acceptance and DAG decisions
remain pure Stage-0 logic so a saved observation log can be replayed offline.
"""
from __future__ import annotations

from typing import Any, Mapping, Protocol


class CalibrationBackend(Protocol):
    @property
    def now_s(self) -> float: ...

    def execute(self, family: str, parameters: Mapping[str, float], shots: int,
                held_out: bool = False) -> dict[str, Any]: ...
