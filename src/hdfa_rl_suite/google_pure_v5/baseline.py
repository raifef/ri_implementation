"""Per-detector baseline obtained by gradient descent on Supplement Eq. 19."""
from __future__ import annotations

import numpy as np

from .reward import detector_advantages


class DetectorBaseline:
    def __init__(self, detector_count: int, *, learning_rate: float) -> None:
        if detector_count <= 0 or learning_rate < 0:
            raise ValueError("invalid detector baseline configuration")
        self.value = np.zeros(detector_count, dtype=float)
        self.learning_rate = float(learning_rate)

    def snapshot(self) -> np.ndarray:
        return self.value.copy()

    def advantages(self, rewards: np.ndarray, *, frozen: np.ndarray | None = None) -> np.ndarray:
        return detector_advantages(rewards, self.value if frozen is None else frozen)

    def loss(self, rewards: np.ndarray, *, value: np.ndarray | None = None) -> float:
        base = self.value if value is None else np.asarray(value, dtype=float)
        residual = np.asarray(rewards, dtype=float) - base[None, :]
        return float(np.mean(np.sum(residual * residual, axis=1)))

    def gradient(self, rewards: np.ndarray, *, value: np.ndarray | None = None) -> np.ndarray:
        base = self.value if value is None else np.asarray(value, dtype=float)
        return 2.0 * (base - np.asarray(rewards, dtype=float).mean(axis=0))

    def update(self, rewards: np.ndarray) -> np.ndarray:
        self.value -= self.learning_rate * self.gradient(rewards)
        return self.snapshot()

    def reset(self) -> None:
        self.value.fill(0.0)
