"""Explicit legacy bounded-action ablation for Figure 5a.

The canonical source-coordinate plant executes Gaussian controls directly and
therefore must not construct, validate, or hash a tanh action domain.  This
module retains that old transform only for named noncanonical comparisons.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .plant import Figure5aStimPlant


@dataclass(frozen=True)
class Figure5aBoundedActionAblation:
    """Phase-independent scaled-tanh domain used only by legacy ablations."""

    plant: Figure5aStimPlant
    maximum_probability: float | None = None
    action_probability_margin_fraction: float = 1e-6

    def __post_init__(self) -> None:
        maximum_probability = (None if self.maximum_probability is None
                               else float(self.maximum_probability))
        margin = float(self.action_probability_margin_fraction)
        if (maximum_probability is not None
                and not 0.0 < maximum_probability <=
                float(np.min(self.plant.probability_ceilings))):
            raise ValueError(
                "bounded-action maximum probability must fit every Stim channel")
        if not 0.0 < margin < 1.0:
            raise ValueError("action probability margin fraction must lie in (0,1)")
        irreducible = np.asarray(
            [item.irreducible_error for item in self.plant.inventory], dtype=float)
        omega = np.asarray(
            [item.omega_sensitivity for item in self.plant.inventory], dtype=float)
        probability_ceiling = (
            self.plant.probability_ceilings if maximum_probability is None
            else np.full(self.plant.control_count, maximum_probability)) * (1.0 - margin)
        maximum_mismatch = np.sqrt((probability_ceiling - irreducible) / omega)
        control_limits = maximum_mismatch - 1.0
        if not np.all(np.isfinite(control_limits)) or np.any(control_limits <= 1.0):
            raise ValueError(
                "bounded-action ablation has no symmetric domain containing the full optimum range")
        maximum_mismatch.setflags(write=False)
        control_limits.setflags(write=False)
        object.__setattr__(self, "maximum_mismatch", maximum_mismatch)
        object.__setattr__(self, "control_limits", control_limits)

    def normalized_control_limits(
        self, native_scale: np.ndarray | None = None,
    ) -> np.ndarray:
        scale = (np.ones(self.plant.control_count, dtype=float) if native_scale is None
                 else np.asarray(native_scale, dtype=float))
        if (scale.shape != (self.plant.control_count,) or np.any(scale <= 0)
                or not np.all(np.isfinite(scale))):
            raise ValueError("native scale must be a positive 41-coordinate vector")
        native_absolute_limit = self.maximum_mismatch - scale
        normalized_limit = native_absolute_limit / scale
        if np.any(normalized_limit <= 1.0):
            raise ValueError(
                "empirical normalization leaves no bounded ablation domain containing the full optimum")
        return normalized_limit

    def apply_control_transform(
        self, latent_controls: np.ndarray, *, native_scale: np.ndarray | None = None,
    ) -> np.ndarray:
        latent = np.asarray(latent_controls, dtype=float)
        if (latent.shape[-1:] != (self.plant.control_count,)
                or not np.all(np.isfinite(latent))):
            raise ValueError("latent Figure 5a controls must end in 41 finite coordinates")
        limits = self.normalized_control_limits(native_scale)
        return limits * np.tanh(latent / limits)

    def latent_controls_for(
        self, applied_controls: np.ndarray, *, native_scale: np.ndarray | None = None,
    ) -> np.ndarray:
        applied = np.asarray(applied_controls, dtype=float)
        if (applied.shape[-1:] != (self.plant.control_count,)
                or not np.all(np.isfinite(applied))):
            raise ValueError("applied Figure 5a controls must end in 41 finite coordinates")
        limits = self.normalized_control_limits(native_scale)
        ratio = applied / limits
        if np.any(np.abs(ratio) >= 1.0):
            raise ValueError("applied controls must lie strictly inside the bounded ablation domain")
        return limits * np.arctanh(ratio)
