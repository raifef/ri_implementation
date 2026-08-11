"""Paper-matched sparse local quadratic synthetic calibration plant."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PurePlantSpec:
    plant_id: str
    detector_count: int = 12
    control_count: int = 24
    curvature: float = 0.30
    detector_floor: float = 0.055
    logical_floor: float = 0.005
    logical_gain: float = 1.45
    draw_seed: int = 7001


class PureQuadraticPlant:
    """The controller sees counts and a graph, never the optimum or logical score."""

    def __init__(self, spec: PurePlantSpec) -> None:
        self.spec = spec
        if spec.control_count != 2 * spec.detector_count:
            raise ValueError("the frozen development plant uses two controls per detector")
        rng = np.random.default_rng(spec.draw_seed)
        self.mask = np.zeros((spec.detector_count, spec.control_count), dtype=bool)
        for detector in range(spec.detector_count):
            self.mask[detector, 2 * detector : 2 * detector + 2] = True
        self.curvature = spec.curvature * rng.uniform(0.92, 1.08, spec.detector_count)
        self.floors = spec.detector_floor * rng.uniform(0.96, 1.04, spec.detector_count)
        category_scale = np.array([1.0, 0.8, 1.2], dtype=float)
        self.native_sensitivity = np.resize(category_scale, spec.control_count)
        self.base_optimum = np.zeros(spec.control_count, dtype=float)
        self._indices = tuple(np.flatnonzero(row) for row in self.mask)

    def controller_view(self) -> dict[str, np.ndarray | int | float]:
        return {
            "detector_count": self.spec.detector_count,
            "control_count": self.spec.control_count,
            "mask": self.mask.copy(),
            "native_sensitivity": self.native_sensitivity.copy(),
            "normalized_bound": 1.0,
        }

    def detector_rates(self, actions: np.ndarray, optimum: np.ndarray) -> np.ndarray:
        values = np.atleast_2d(np.asarray(actions, dtype=float))
        target = np.asarray(optimum, dtype=float)
        if values.shape[1] != self.spec.control_count or target.shape != (self.spec.control_count,):
            raise ValueError("plant action/optimum shape mismatch")
        rates = np.empty((len(values), self.spec.detector_count), dtype=float)
        for detector, indices in enumerate(self._indices):
            error = values[:, indices] - target[indices][None, :]
            rates[:, detector] = self.floors[detector] + self.curvature[detector] * np.mean(
                error * error, axis=1
            )
        return np.clip(rates, 1e-7, 0.45)

    def logical_risk(self, actions: np.ndarray, optimum: np.ndarray) -> np.ndarray:
        rates = self.detector_rates(actions, optimum)
        excess = np.maximum(rates - self.floors[None, :], 0.0).mean(axis=1)
        return np.clip(self.spec.logical_floor + self.spec.logical_gain * excess, 1e-8, 0.5)

    def acquire_counts(
        self,
        actions: np.ndarray,
        optimum: np.ndarray,
        *,
        effective_cycles: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return rng.binomial(effective_cycles, self.detector_rates(actions, optimum)).astype(np.int64)
