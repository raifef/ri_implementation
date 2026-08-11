"""Per-detector collection-time baseline with frozen batch advantages."""
from __future__ import annotations

import numpy as np


class DetectorBaseline:
    def __init__(self, detector_count: int, *, coefficient: float) -> None:
        if detector_count <= 0 or not 0.0 <= coefficient <= 1.0:
            raise ValueError("invalid detector baseline configuration")
        self.value = np.zeros(detector_count, dtype=float)
        self.coefficient = float(coefficient)

    def snapshot(self) -> np.ndarray:
        return self.value.copy()

    def advantages(self, rewards: np.ndarray, frozen: np.ndarray) -> np.ndarray:
        rewards = np.asarray(rewards, dtype=float)
        base = np.asarray(frozen, dtype=float)
        if rewards.ndim != 2 or rewards.shape[1:] != base.shape:
            raise ValueError("reward/baseline shape mismatch")
        return rewards - base[None, :]

    def update(self, rewards: np.ndarray) -> np.ndarray:
        target = np.asarray(rewards, dtype=float).mean(axis=0)
        self.value = (1.0 - self.coefficient) * self.value + self.coefficient * target
        return self.snapshot()

    def reset(self) -> None:
        self.value.fill(0.0)
