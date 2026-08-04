"""Source-specified detector-local reward and advantage definitions."""
from __future__ import annotations

import numpy as np


def detector_rewards(detector_counts: np.ndarray, effective_cycles: int) -> np.ndarray:
    counts = np.asarray(detector_counts, dtype=float)
    if counts.ndim != 2 or effective_cycles <= 0:
        raise ValueError("detector counts must be a candidate-by-detector matrix")
    if np.any(counts < 0) or np.any(counts > effective_cycles):
        raise ValueError("detector counts outside the binomial range")
    return -counts / float(effective_cycles)


def detector_advantages(rewards: np.ndarray, frozen_baseline: np.ndarray) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=float)
    baseline = np.asarray(frozen_baseline, dtype=float)
    if rewards.ndim != 2 or rewards.shape[1:] != baseline.shape:
        raise ValueError("reward and detector baseline shapes are inconsistent")
    return rewards - baseline[None, :]
