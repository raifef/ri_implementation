"""Optimizers whose state and updates are attached to direct sigma."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .contracts import PositivityGuard


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

    def __post_init__(self) -> None:
        if min(self.mean_learning_rate, self.sigma_learning_rate, self.baseline_learning_rate) <= 0:
            raise ValueError("learning rates must be positive")
        if not 0 <= self.momentum < 1 or not 0 < self.minimum_sigma < self.maximum_sigma:
            raise ValueError("invalid momentum or sigma bounds")


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
        self.mean_velocity = cfg.momentum * self.mean_velocity + np.asarray(grad_mean)
        self.sigma_velocity = cfg.momentum * self.sigma_velocity + np.asarray(grad_sigma)
        self.baseline_velocity = cfg.momentum * self.baseline_velocity + np.asarray(grad_baseline)
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
                "backtracks": backtracks, "optimized_scale_variable": "sigma"}

    def state_dict(self) -> dict[str, Any]:
        return {"schema_version": "direct-sigma-optimizer.v1", "optimized_scale_variable": "sigma",
                "config": {**asdict(self.config), "positivity_guard": self.config.positivity_guard.value},
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
