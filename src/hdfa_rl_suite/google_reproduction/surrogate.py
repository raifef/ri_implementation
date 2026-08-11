"""Frozen local quadratic plant following the paper's declared simulation form."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Any

import numpy as np

from .config import load_surrogate_config


def surface_code_gate_count(distance: int) -> int:
    if distance < 3 or distance % 2 == 0:
        raise ValueError("surface-code distance must be odd and at least three")
    return 6 * distance * distance - 4 * distance - 1


def surface_code_parameter_count(distance: int, controls_per_gate: int = 30) -> int:
    if controls_per_gate <= 0:
        raise ValueError("controls per gate must be positive")
    return surface_code_gate_count(distance) * controls_per_gate


@dataclass(frozen=True)
class SurrogateEvaluation:
    detector_rates: np.ndarray
    logical_risk: np.ndarray


class PaperAnchoredSurrogate:
    """Sparse synthetic plant, not a Willow hardware or pulse-level model."""

    def __init__(
        self,
        *,
        distance: int = 3,
        controls_per_gate: int | None = None,
        configuration: Mapping[str, Any] | None = None,
    ) -> None:
        cfg = configuration or load_surrogate_config()
        layout = cfg["layout"]
        plant = cfg["plant"]
        self.distance = distance
        self.controls_per_gate = int(
            controls_per_gate if controls_per_gate is not None else (
                layout["development_controls_per_gate"] if distance == layout["development_distance"]
                else layout["controls_per_gate_large_code"]
            )
        )
        self.gate_count = surface_code_gate_count(distance)
        self.control_count = surface_code_parameter_count(distance, self.controls_per_gate)
        self.detector_count = self.gate_count
        rng = np.random.default_rng(int(plant["surrogate_seed"]) + distance * 100 + self.controls_per_gate)
        self.sensitivity = rng.uniform(*plant["sensitivity_range_native_per_normalized"], self.control_count)
        self.floors = rng.uniform(*plant["irreducible_detector_floor_range"], self.detector_count)
        self.weights = rng.uniform(*plant["quadratic_weight_range"], self.detector_count)
        residual = float(plant["initial_residual_normalized"])
        self.initial_mean_normalized = residual * np.sin(np.arange(self.control_count) * 1.61803398875 + 0.3)
        self.optimum_normalized = np.zeros(self.control_count)
        self.maximum_detector_probability = float(plant["maximum_detector_probability"])
        self.logical_floor = float(plant["logical_floor"])
        self.logical_excess_scale = float(plant["logical_excess_scale"])
        self.factor_radius = int(layout["factor_radius"])
        self.control_ids = tuple(f"g{index // self.controls_per_gate}:p{index % self.controls_per_gate}" for index in range(self.control_count))
        self.detector_ids = tuple(f"detector:{index}" for index in range(self.detector_count))
        self.factor_indices = tuple(self._factor_for_detector(index) for index in range(self.detector_count))

    def _factor_for_detector(self, detector: int) -> np.ndarray:
        gates = [(detector + offset) % self.gate_count for offset in range(-self.factor_radius, self.factor_radius + 1)]
        return np.asarray(
            [gate * self.controls_per_gate + parameter for gate in gates for parameter in range(self.controls_per_gate)],
            dtype=int,
        )

    def dense_mask(self) -> np.ndarray:
        if self.detector_count * self.control_count > 5_000_000:
            raise MemoryError("large-code validation must use sparse factors, not a dense mask")
        mask = np.zeros((self.detector_count, self.control_count), dtype=bool)
        for detector, indices in enumerate(self.factor_indices):
            mask[detector, indices] = True
        return mask

    @property
    def initial_mean_native(self) -> np.ndarray:
        return self.initial_mean_normalized * self.sensitivity

    def validate_sensitivity_calibration(self, candidate: np.ndarray) -> None:
        """Fail closed if a controller uses anything but the frozen calibration vector."""
        value = np.asarray(candidate, dtype=float)
        if value.shape != self.sensitivity.shape or not np.allclose(value, self.sensitivity, rtol=1e-12, atol=1e-14):
            raise ValueError("controller sensitivity scale does not match the frozen surrogate calibration")

    def optimum_at(
        self,
        epoch: int,
        *,
        drift_frequency_per_epoch: float = 0.0,
        drift_amplitude: float = 0.0,
        step_epoch: int | None = None,
        step_amplitude: float = 0.0,
    ) -> np.ndarray:
        optimum = self.optimum_normalized.copy()
        active = max(1, int(round(0.2 * self.control_count)))
        if drift_frequency_per_epoch:
            optimum[:active] += drift_amplitude * np.sin(2 * np.pi * drift_frequency_per_epoch * epoch)
        if step_epoch is not None and epoch >= step_epoch:
            optimum[:active] += step_amplitude
        return optimum

    def evaluate_native(self, actions_native: np.ndarray, optimum_normalized: np.ndarray | None = None) -> SurrogateEvaluation:
        actions = np.atleast_2d(np.asarray(actions_native, dtype=float)) / self.sensitivity[None, :]
        if actions.shape[1] != self.control_count:
            raise ValueError("action/control dimension mismatch")
        optimum = self.optimum_normalized if optimum_normalized is None else np.asarray(optimum_normalized, dtype=float)
        excess = np.empty((actions.shape[0], self.detector_count), dtype=float)
        for detector, indices in enumerate(self.factor_indices):
            mismatch = actions[:, indices] - optimum[indices]
            excess[:, detector] = self.weights[detector] * np.mean(mismatch * mismatch, axis=1)
        rates = np.clip(self.floors[None, :] + excess, 0.0, self.maximum_detector_probability)
        logical = self.logical_floor + self.logical_excess_scale * excess.mean(axis=1)
        return SurrogateEvaluation(rates, logical)

    def acquire_counts(
        self,
        actions_native: np.ndarray,
        cycles: int,
        rng: np.random.Generator,
        optimum_normalized: np.ndarray | None = None,
    ) -> tuple[np.ndarray, SurrogateEvaluation]:
        if cycles <= 0:
            raise ValueError("cycles must be positive")
        evaluation = self.evaluate_native(actions_native, optimum_normalized)
        return rng.binomial(cycles, evaluation.detector_rates), evaluation
