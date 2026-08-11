"""Optimizers whose state and updates are attached to direct sigma."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from .contracts import PositivityGuard


class GradientClippingMode(StrEnum):
    """Preregistered gradient-stabilization nuisance variants.

    The source reports a clipping magnitude but does not identify the clipping
    geometry.  These modes are therefore explicit experimental variants, not
    source-derived defaults.
    """

    NONE = "none"
    PER_COMPONENT = "per_component"
    GLOBAL_L2 = "global_l2"


@dataclass(frozen=True)
class OptimizerConfig:
    mean_learning_rate: float
    sigma_learning_rate: float
    baseline_learning_rate: float
    momentum: float = 0.0
    minimum_sigma: float = 1e-6
    maximum_sigma: float = 10.0
    positivity_guard: PositivityGuard = PositivityGuard.PROJECTED_GRADIENT
    maximum_backtracks: int = 30
    gradient_clipping_mode: GradientClippingMode = GradientClippingMode.NONE
    gradient_clip_threshold: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "positivity_guard", PositivityGuard(self.positivity_guard))
        object.__setattr__(self, "gradient_clipping_mode",
                           GradientClippingMode(self.gradient_clipping_mode))
        if min(self.mean_learning_rate, self.sigma_learning_rate, self.baseline_learning_rate) <= 0:
            raise ValueError("learning rates must be positive")
        if not 0 <= self.momentum < 1 or not 0 < self.minimum_sigma < self.maximum_sigma:
            raise ValueError("invalid momentum or sigma bounds")
        if self.gradient_clipping_mode == GradientClippingMode.NONE:
            if self.gradient_clip_threshold is not None:
                raise ValueError("gradient_clip_threshold must be null when clipping is disabled")
        elif self.gradient_clip_threshold is None or not np.isfinite(self.gradient_clip_threshold) \
                or self.gradient_clip_threshold <= 0:
            raise ValueError("enabled gradient clipping requires a finite positive threshold")


def _gradient_diagnostics(gradients: tuple[np.ndarray, ...]) -> dict[str, float]:
    names = ("mean", "sigma", "baseline")
    return {
        f"{name}_gradient_l2_norm": float(np.linalg.norm(value))
        for name, value in zip(names, gradients, strict=True)
    }


def _clip_gradients(
    gradients: tuple[np.ndarray, np.ndarray, np.ndarray],
    config: OptimizerConfig,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, Any]]:
    """Clip one joint optimizer gradient before momentum is accumulated."""
    values = tuple(np.asarray(value, dtype=float) for value in gradients)
    if any(not np.all(np.isfinite(value)) for value in values):
        raise ValueError("optimizer gradients must be finite")
    flattened = np.concatenate([value.reshape(-1) for value in values])
    pre_norm = float(np.linalg.norm(flattened))
    threshold = config.gradient_clip_threshold
    clipped_component_count = 0
    scale = 1.0
    if config.gradient_clipping_mode == GradientClippingMode.NONE:
        clipped = tuple(value.copy() for value in values)
    elif config.gradient_clipping_mode == GradientClippingMode.PER_COMPONENT:
        assert threshold is not None
        clipped_component_count = int(np.count_nonzero(np.abs(flattened) > threshold))
        clipped = tuple(np.clip(value, -threshold, threshold) for value in values)
    elif config.gradient_clipping_mode == GradientClippingMode.GLOBAL_L2:
        assert threshold is not None
        scale = min(1.0, threshold / pre_norm) if pre_norm > 0 else 1.0
        clipped_component_count = int(flattened.size if scale < 1.0 else 0)
        clipped = tuple(value * scale for value in values)
    else:  # pragma: no cover - guarded by OptimizerConfig
        raise ValueError(f"unsupported gradient clipping mode: {config.gradient_clipping_mode}")
    post_flattened = np.concatenate([value.reshape(-1) for value in clipped])
    diagnostics: dict[str, Any] = {
        "gradient_clipping_mode": config.gradient_clipping_mode.value,
        "gradient_clip_threshold": threshold,
        "gradient_global_l2_norm_before_clipping": pre_norm,
        "gradient_global_l2_norm_after_clipping": float(np.linalg.norm(post_flattened)),
        "gradient_global_clip_scale": float(scale),
        "gradient_component_count": int(flattened.size),
        "gradient_clipped_component_count": clipped_component_count,
        "gradient_clipped_component_fraction": float(clipped_component_count / flattened.size),
    }
    diagnostics.update({f"raw_{key}": value for key, value in _gradient_diagnostics(values).items()})
    diagnostics.update({f"applied_{key}": value for key, value in _gradient_diagnostics(clipped).items()})
    return clipped, diagnostics


class DirectSigmaOptimizer:
    def __init__(self, dimension: int, detector_count: int, config: OptimizerConfig) -> None:
        if min(dimension, detector_count) <= 0:
            raise ValueError("optimizer dimensions must be positive")
        self.config = config
        self.mean_velocity = np.zeros(dimension)
        self.sigma_velocity = np.zeros(dimension)
        self.baseline_velocity = np.zeros(detector_count)
        self.steps = 0

    def step(self, mean: np.ndarray, sigma: np.ndarray, baseline: np.ndarray,
             grad_mean: np.ndarray, grad_sigma: np.ndarray, grad_baseline: np.ndarray,
             *, mean_bounds: tuple[float, float] | None = None) -> dict[str, Any]:
        cfg = self.config
        gradients = (np.asarray(grad_mean, dtype=float), np.asarray(grad_sigma, dtype=float),
                     np.asarray(grad_baseline, dtype=float))
        expected_shapes = (self.mean_velocity.shape, self.sigma_velocity.shape,
                           self.baseline_velocity.shape)
        if tuple(value.shape for value in gradients) != expected_shapes:
            raise ValueError(f"optimizer gradient shapes must be {expected_shapes}")
        clipped, clipping_diagnostics = _clip_gradients(gradients, cfg)
        self.mean_velocity = cfg.momentum * self.mean_velocity + clipped[0]
        self.sigma_velocity = cfg.momentum * self.sigma_velocity + clipped[1]
        self.baseline_velocity = cfg.momentum * self.baseline_velocity + clipped[2]
        proposed_mean = np.asarray(mean) - cfg.mean_learning_rate * self.mean_velocity
        if mean_bounds is not None:
            proposed_mean = np.clip(proposed_mean, *mean_bounds)
        raw_sigma = np.asarray(sigma) - cfg.sigma_learning_rate * self.sigma_velocity
        backtracks = 0
        if cfg.positivity_guard == PositivityGuard.PROJECTED_GRADIENT:
            proposed_sigma = np.maximum(raw_sigma, cfg.minimum_sigma)
        elif cfg.positivity_guard == PositivityGuard.BOUNDED_OPTIMIZER:
            proposed_sigma = np.clip(raw_sigma, cfg.minimum_sigma, cfg.maximum_sigma)
        elif cfg.positivity_guard == PositivityGuard.BACKTRACKING_STEP:
            learning_rate = cfg.sigma_learning_rate
            proposed_sigma = raw_sigma
            while np.any(proposed_sigma < cfg.minimum_sigma) and backtracks < cfg.maximum_backtracks:
                learning_rate *= 0.5
                proposed_sigma = np.asarray(sigma) - learning_rate * self.sigma_velocity
                backtracks += 1
            proposed_sigma = np.maximum(proposed_sigma, cfg.minimum_sigma)
        else:
            raise ValueError(f"unsupported positivity guard: {cfg.positivity_guard}")
        proposed_sigma = np.minimum(proposed_sigma, cfg.maximum_sigma)
        mean[:] = proposed_mean
        sigma[:] = proposed_sigma
        baseline[:] = np.asarray(baseline) - cfg.baseline_learning_rate * self.baseline_velocity
        self.steps += 1
        return {"fraction_at_positivity_guard": float(np.mean(proposed_sigma <= cfg.minimum_sigma)),
                "backtracks": backtracks, "optimized_scale_variable": "sigma",
                **clipping_diagnostics}

    def state_dict(self) -> dict[str, Any]:
        return {"schema_version": "direct-sigma-optimizer.v1", "optimized_scale_variable": "sigma",
                "config": {**asdict(self.config), "positivity_guard": self.config.positivity_guard.value,
                           "gradient_clipping_mode": self.config.gradient_clipping_mode.value},
                "mean_velocity": self.mean_velocity.tolist(), "sigma_velocity": self.sigma_velocity.tolist(),
                "baseline_velocity": self.baseline_velocity.tolist(), "steps": self.steps}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "DirectSigmaOptimizer":
        if state.get("schema_version") != "direct-sigma-optimizer.v1" or \
                state.get("optimized_scale_variable") != "sigma":
            raise ValueError("optimizer checkpoint is not direct-sigma state")
        config_value = dict(state["config"])
        config_value["positivity_guard"] = PositivityGuard(config_value["positivity_guard"])
        mean_velocity = np.asarray(state["mean_velocity"], dtype=float)
        baseline_velocity = np.asarray(state["baseline_velocity"], dtype=float)
        result = cls(len(mean_velocity), len(baseline_velocity), OptimizerConfig(**config_value))
        result.mean_velocity[:] = mean_velocity
        result.sigma_velocity[:] = np.asarray(state["sigma_velocity"], dtype=float)
        result.baseline_velocity[:] = baseline_velocity
        result.steps = int(state["steps"])
        return result
