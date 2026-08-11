"""Normalized/native coordinates driven only by fitted detector sensitivity."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np

from .contracts import (
    NON_PAPER_NORMALIZATION_ABLATION,
    PAPER_NORMALIZATION_METHOD,
    CalibrationBundle,
)


CoordinateSpace = Literal["native", "normalized"]


@dataclass(frozen=True)
class CoordinateVector:
    values: np.ndarray
    parameter_ids: tuple[str, ...]
    space: CoordinateSpace

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.shape != (len(self.parameter_ids),) or not np.all(np.isfinite(values)):
            raise ValueError("coordinate values must be a finite vector aligned to parameter_ids")
        if self.space not in {"native", "normalized"}:
            raise ValueError("coordinate space must be native or normalized")
        copy = values.copy()
        copy.setflags(write=False)
        object.__setattr__(self, "values", copy)
        object.__setattr__(self, "parameter_ids", tuple(self.parameter_ids))


@dataclass(frozen=True)
class NativePolicyBatch:
    latent_normalized: np.ndarray
    applied_native: np.ndarray
    parameter_ids: tuple[str, ...]
    normalization_method: str
    sensitivity_application_count: int
    calibration_bundle_hash: str


class EmpiricalCoordinateSystem:
    """Apply each empirically measured sigma0 once at the hardware boundary."""

    def __init__(self, bundle: CalibrationBundle) -> None:
        if bundle.normalization_method != PAPER_NORMALIZATION_METHOD or not bundle.artifact_complete:
            raise ValueError("an accepted empirical calibration bundle is required")
        self.bundle = bundle
        fits = bundle.fit_by_type()
        parameter_ids: list[str] = []
        type_names: list[str] = []
        references: list[float] = []
        scales: list[float] = []
        type_slices: dict[str, np.ndarray] = {}
        for spec in bundle.control_specs:
            start = len(parameter_ids)
            parameter_ids.extend(spec.gate_ids)
            type_names.extend([spec.control_type] * len(spec.gate_ids))
            references.extend([spec.reference_value_native] * len(spec.gate_ids))
            scales.extend([fits[spec.control_type].sigma0_native] * len(spec.gate_ids))
            type_slices[spec.control_type] = np.arange(start, len(parameter_ids), dtype=int)
        if len(parameter_ids) != len(set(parameter_ids)):
            raise ValueError("parameter ids must be globally unique")
        self.parameter_ids = tuple(parameter_ids)
        self.parameter_types = tuple(type_names)
        self.reference_native = np.asarray(references, dtype=float)
        self.native_per_normalized = np.asarray(scales, dtype=float)
        self._type_indices = type_slices

    def _require(self, vector: CoordinateVector, space: CoordinateSpace) -> None:
        if vector.space != space:
            raise ValueError(
                f"expected {space} coordinates, got {vector.space}; sensitivity cannot be applied twice")
        if vector.parameter_ids != self.parameter_ids:
            raise ValueError("parameter registry mismatch")

    def to_normalized(self, native: CoordinateVector) -> CoordinateVector:
        self._require(native, "native")
        return CoordinateVector(
            (native.values - self.reference_native) / self.native_per_normalized,
            self.parameter_ids,
            "normalized",
        )

    def to_native(self, normalized: CoordinateVector) -> CoordinateVector:
        self._require(normalized, "normalized")
        return CoordinateVector(
            self.reference_native + self.native_per_normalized * normalized.values,
            self.parameter_ids,
            "native",
        )

    def covariance_to_native(self, normalized_covariance: np.ndarray) -> np.ndarray:
        covariance = np.asarray(normalized_covariance, dtype=float)
        shape = (len(self.parameter_ids), len(self.parameter_ids))
        if covariance.shape != shape:
            raise ValueError("covariance shape mismatch")
        scale = self.native_per_normalized
        return scale[:, None] * covariance * scale[None, :]

    def covariance_to_normalized(self, native_covariance: np.ndarray) -> np.ndarray:
        covariance = np.asarray(native_covariance, dtype=float)
        shape = (len(self.parameter_ids), len(self.parameter_ids))
        if covariance.shape != shape:
            raise ValueError("covariance shape mismatch")
        scale = self.native_per_normalized
        return covariance / scale[:, None] / scale[None, :]

    def normalized_edr_hessian(self) -> np.ndarray:
        """H such that 0.5 Tr(H I) is one percentage point per type."""
        diagonal = np.zeros(len(self.parameter_ids), dtype=float)
        for indices in self._type_indices.values():
            diagonal[indices] = 2.0 / len(indices)
        return np.diag(diagonal)

    def native_edr_hessian(self) -> np.ndarray:
        scale = self.native_per_normalized
        return self.normalized_edr_hessian() / scale[:, None] / scale[None, :]

    def predict_candidate_damage_percentage_points(self, normalized_mean: np.ndarray,
                                                   normalized_covariance: np.ndarray) -> float:
        mean = np.asarray(normalized_mean, dtype=float)
        covariance = np.asarray(normalized_covariance, dtype=float)
        dimension = len(self.parameter_ids)
        if mean.shape != (dimension,) or covariance.shape != (dimension, dimension):
            raise ValueError("candidate moment shape mismatch")
        hessian = self.normalized_edr_hessian()
        return float(0.5 * np.trace(hessian @ covariance) + 0.5 * mean @ hessian @ mean)

    def type_sigma_native(self, normalized_sigma_by_type: Mapping[str, float]) -> dict[str, float]:
        fits = self.bundle.fit_by_type()
        unknown = set(normalized_sigma_by_type) - set(fits)
        if unknown:
            raise ValueError(f"unknown control types: {sorted(unknown)}")
        return {
            name: float(value) * fits[name].sigma0_native
            for name, value in normalized_sigma_by_type.items()
        }


class EmpiricallyNormalizedGaussianPolicy:
    """Minimal controller boundary proving sensitivity is consumed once."""

    def __init__(self, coordinates: EmpiricalCoordinateSystem, *, seed: int) -> None:
        self.coordinates = coordinates
        self._rng = np.random.default_rng(int(seed))

    def sample(self, candidate_count: int, normalized_mean: np.ndarray,
               normalized_standard_deviation: np.ndarray) -> NativePolicyBatch:
        dimension = len(self.coordinates.parameter_ids)
        mean = np.asarray(normalized_mean, dtype=float)
        scale = np.asarray(normalized_standard_deviation, dtype=float)
        if candidate_count <= 0 or mean.shape != (dimension,) or scale.shape != (dimension,):
            raise ValueError("invalid Gaussian policy shape or candidate count")
        if np.any(scale <= 0) or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
            raise ValueError("Gaussian parameters must be finite with positive scale")
        latent = self._rng.normal(mean, scale, size=(candidate_count, dimension))
        native = (
            self.coordinates.reference_native[None, :]
            + self.coordinates.native_per_normalized[None, :] * latent
        )
        return NativePolicyBatch(
            latent_normalized=latent,
            applied_native=native,
            parameter_ids=self.coordinates.parameter_ids,
            normalization_method=PAPER_NORMALIZATION_METHOD,
            sensitivity_application_count=1,
            calibration_bundle_hash=self.coordinates.bundle.bundle_hash,
        )


def legacy_algebraic_ablation_label() -> str:
    """The only allowed label for any old/native heuristic scale branch."""
    return NON_PAPER_NORMALIZATION_ABLATION
