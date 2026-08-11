"""Unit-explicit synthetic quadratic calibration plant used by pure v6."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .units import CoordinateContract


@dataclass(frozen=True)
class PlantSpec:
    mask: np.ndarray
    base_optimum_normalized: np.ndarray
    detector_floor: np.ndarray
    normalized_curvature: np.ndarray
    coordinates: CoordinateContract

    @property
    def control_count(self) -> int:
        return int(self.mask.shape[1])

    @property
    def detector_count(self) -> int:
        return int(self.mask.shape[0])


def default_spec(control_count: int = 6) -> PlantSpec:
    if control_count < 3:
        raise ValueError("at least three controls are required")
    detector_count = max(3, control_count - 1)
    mask = np.zeros((detector_count, control_count), dtype=bool)
    for detector in range(detector_count):
        mask[detector, detector % control_count] = True
        mask[detector, (detector + 1) % control_count] = True
        if detector % 3 == 0:
            mask[detector, (detector + 2) % control_count] = True
    sensitivity = np.resize(np.asarray([1.0, 0.8, 1.2], dtype=float), control_count)
    coordinates = CoordinateContract(
        native_offset=np.zeros(control_count),
        native_per_normalized=sensitivity,
        normalized_bounds=(-1.0, 1.0),
        native_units=tuple(np.resize(np.asarray(["rad", "relative_amplitude", "MHz"]), control_count)),
        sensitivity_version="v6-default-sensitivity-1",
    )
    return PlantSpec(
        mask=mask,
        base_optimum_normalized=np.zeros(control_count),
        detector_floor=np.linspace(0.018, 0.024, detector_count),
        normalized_curvature=np.linspace(0.018, 0.032, detector_count),
        coordinates=coordinates,
    )


class PureQuadraticPlant:
    """Detector probabilities are evaluated in native coordinates exactly once."""

    def __init__(self, spec: PlantSpec) -> None:
        self.spec = spec
        self.mask = np.asarray(spec.mask, dtype=bool)
        degree = self.mask.sum(axis=1)
        if np.any(degree == 0):
            raise ValueError("every detector must touch a control")
        scale_squared = spec.coordinates.native_per_normalized ** 2
        self.native_component_curvature = (
            spec.normalized_curvature[:, None] * self.mask / degree[:, None] / scale_squared[None, :]
        )

    @property
    def base_optimum_native(self) -> np.ndarray:
        return self.spec.coordinates.to_native(self.spec.base_optimum_normalized)

    def detector_rates_native(self, applied_native: np.ndarray, optimum_native: np.ndarray) -> np.ndarray:
        actions = np.atleast_2d(np.asarray(applied_native, dtype=float))
        optimum = np.asarray(optimum_native, dtype=float)
        if actions.shape[1] != self.spec.control_count or optimum.shape != (self.spec.control_count,):
            raise ValueError("native plant coordinate shape mismatch")
        delta2 = (actions - optimum[None, :]) ** 2
        rates = self.spec.detector_floor[None, :] + delta2 @ self.native_component_curvature.T
        return np.clip(rates, 1e-9, 1.0 - 1e-9)

    def detector_rates_normalized(self, applied_normalized: np.ndarray, optimum_normalized: np.ndarray) -> np.ndarray:
        return self.detector_rates_native(
            self.spec.coordinates.to_native(applied_normalized),
            self.spec.coordinates.to_native(optimum_normalized),
        )

    def logical_risk_native(self, applied_native: np.ndarray, optimum_native: np.ndarray) -> np.ndarray:
        rates = self.detector_rates_native(applied_native, optimum_native)
        return 0.0025 + 0.42 * np.mean(rates, axis=1)

    def acquire_counts(self, applied_native: np.ndarray, optimum_native: np.ndarray, *, cycles: int,
                       rng: np.random.Generator) -> np.ndarray:
        if cycles <= 0:
            raise ValueError("effective cycles must be positive")
        return rng.binomial(cycles, self.detector_rates_native(applied_native, optimum_native))


def optimum_tape(kind: str, epochs: int, amplitude: float, cycles_per_run: float = 4.0,
                 *, controls: int = 6, seed: int = 6201) -> np.ndarray:
    if epochs < 4 or amplitude < 0:
        raise ValueError("invalid optimum tape")
    t = np.arange(epochs, dtype=float)
    direction = np.linspace(1.0, 0.45, controls)
    direction /= np.linalg.norm(direction)
    if kind == "static":
        scalar = np.zeros(epochs)
    elif kind == "step":
        scalar = np.where(t >= int(0.25 * epochs), amplitude, 0.0)
    elif kind == "sine":
        scalar = amplitude * np.sin(2.0 * np.pi * cycles_per_run * t / epochs)
    elif kind == "strobe":
        scalar = amplitude * (np.floor(t / max(1, epochs // 12)).astype(int) % 2)
    elif kind == "natural":
        rng = np.random.default_rng(seed)
        scalar = np.zeros(epochs)
        innovations = rng.normal(scale=amplitude * 0.12, size=epochs)
        for index in range(1, epochs):
            scalar[index] = 0.96 * scalar[index - 1] + innovations[index]
        scalar += amplitude * 0.45 * np.sin(2.0 * np.pi * t / max(epochs, 1))
    else:
        raise ValueError(f"unknown disturbance family: {kind}")
    return scalar[:, None] * direction[None, :]
