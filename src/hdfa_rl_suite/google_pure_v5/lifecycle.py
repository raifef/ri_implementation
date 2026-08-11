"""Candidate provenance, single-use, and policy-version lifecycle checks."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .policy import CandidateBatch, action_hash


@dataclass(frozen=True)
class DetectorEvidence:
    candidate_id: str
    action_hash: str
    detector_counts: np.ndarray
    effective_cycles: int


class PolicyLifecycle:
    def __init__(self) -> None:
        self.policy_version = 0
        self.epoch = 0
        self._consumed: set[tuple[int, tuple[str, ...]]] = set()

    def validate(self, batch: CandidateBatch, evidence: tuple[DetectorEvidence, ...]) -> np.ndarray:
        key = (batch.policy_version, batch.candidate_ids)
        if batch.policy_version != self.policy_version or batch.epoch != self.epoch:
            raise ValueError("stale policy batch")
        if key in self._consumed:
            raise ValueError("already-consumed policy batch")
        by_id = {row.candidate_id: row for row in evidence}
        if len(by_id) != len(evidence) or set(by_id) != set(batch.candidate_ids):
            raise ValueError("candidate evidence labels are missing, duplicate, or unknown")
        counts = []
        cycles: int | None = None
        for index, candidate_id in enumerate(batch.candidate_ids):
            row = by_id[candidate_id]
            if row.action_hash != batch.action_hashes[index]:
                raise ValueError("candidate-to-reward action provenance mismatch")
            if row.action_hash != action_hash(batch.normalized_actions[index]):
                raise ValueError("candidate action mutated after collection")
            if row.effective_cycles <= 0:
                raise ValueError("effective cycle count must be positive")
            if cycles is None:
                cycles = row.effective_cycles
            elif cycles != row.effective_cycles:
                raise ValueError("mixed effective-cycle counts in one epoch")
            counts.append(np.asarray(row.detector_counts, dtype=float))
        result = np.asarray(counts)
        if result.ndim != 2:
            raise ValueError("detector evidence is not rectangular")
        self._consumed.add(key)
        return result

    def advance(self) -> None:
        self.policy_version += 1
        self.epoch += 1

    def reset(self) -> None:
        self.policy_version = 0
        self.epoch = 0
        self._consumed.clear()
