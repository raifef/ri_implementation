"""Canonical normalized/native coordinate transform."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CoordinateContract:
    native_offset: np.ndarray
    native_per_normalized: np.ndarray
    normalized_bounds: tuple[float, float]
    native_units: tuple[str, ...] = ()
    sensitivity_version: str = "sensitivity-v6.1"

    def __post_init__(self) -> None:
        offset = np.asarray(self.native_offset, dtype=float)
        scale = np.asarray(self.native_per_normalized, dtype=float)
        if offset.ndim != 1 or scale.shape != offset.shape or np.any(scale <= 0):
            raise ValueError("coordinate offset/scale must be aligned and scale positive")
        if not self.normalized_bounds[0] < self.normalized_bounds[1]:
            raise ValueError("invalid normalized bounds")
        units = self.native_units or tuple("native_unit" for _ in offset)
        if len(units) != len(offset):
            raise ValueError("one native unit label is required per control")
        object.__setattr__(self, "native_offset", offset.copy())
        object.__setattr__(self, "native_per_normalized", scale.copy())
        object.__setattr__(self, "native_units", tuple(str(unit) for unit in units))

    @property
    def native_sensitivity(self) -> np.ndarray:
        """Named alias documenting that this scale is applied exactly once."""
        return self.native_per_normalized

    def to_native(self, normalized: np.ndarray) -> np.ndarray:
        return self.native_offset + self.native_per_normalized * np.asarray(normalized, dtype=float)

    def to_normalized(self, native: np.ndarray) -> np.ndarray:
        return (np.asarray(native, dtype=float) - self.native_offset) / self.native_per_normalized

    def apply_bounds(self, latent_normalized: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(latent_normalized, dtype=float), *self.normalized_bounds)
