"""Source-minimal FIFO replay with complete collection provenance."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .policy import CandidateBatch


@dataclass(frozen=True)
class ReplayItem:
    batch: CandidateBatch
    rewards: np.ndarray
    frozen_advantages: np.ndarray
    collection_baseline: np.ndarray

    @property
    def age_metadata(self) -> dict[str, int]:
        return {"epoch": self.batch.epoch, "environment_time": self.batch.environment_time}


class FifoReplay:
    def __init__(self, capacity_epochs: int) -> None:
        if capacity_epochs < 0:
            raise ValueError("replay capacity cannot be negative")
        self.capacity_epochs = int(capacity_epochs)
        self._items: list[ReplayItem] = []

    def items(self) -> tuple[ReplayItem, ...]:
        return tuple(self._items)

    def append(self, item: ReplayItem) -> None:
        if self.capacity_epochs:
            self._items.append(item)
            self._items = self._items[-self.capacity_epochs :]

    def reset(self) -> None:
        self._items.clear()
