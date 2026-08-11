"""Candidate/evidence provenance and one-use policy lifecycle."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .policy import CandidateBatch, action_hash


@dataclass(frozen=True)
class DetectorEvidence:
    candidate_id: str
    applied_action_hash: str
    detector_counts: np.ndarray
    effective_cycles: int
    environment_time: int


class PolicyLifecycle:
    def __init__(self) -> None:
        self.policy_version = 0
        self.epoch = 0
        self.environment_time = 0
        self._consumed: set[tuple[int, tuple[str, ...]]] = set()

    def validate(self, batch: CandidateBatch, evidence: tuple[DetectorEvidence, ...]) -> tuple[np.ndarray, int]:
        key = (batch.policy_version, batch.candidate_ids)
        if batch.policy_version != self.policy_version or batch.epoch != self.epoch:
            raise ValueError("stale policy batch")
        if batch.environment_time != self.environment_time or key in self._consumed:
            raise ValueError("environment-time mismatch or consumed batch")
        by_id = {row.candidate_id: row for row in evidence}
        if len(by_id) != len(evidence) or set(by_id) != set(batch.candidate_ids):
            raise ValueError("candidate evidence labels invalid")
        rows, cycles = [], None
        for index, candidate_id in enumerate(batch.candidate_ids):
            item = by_id[candidate_id]
            if item.applied_action_hash != batch.applied_action_hashes[index]:
                raise ValueError("candidate-to-reward applied-action provenance mismatch")
            if item.applied_action_hash != action_hash(batch.applied_native_actions[index]):
                raise ValueError("applied candidate mutated")
            if item.environment_time != batch.environment_time:
                raise ValueError("environment-time leakage")
            cycles = item.effective_cycles if cycles is None else cycles
            if item.effective_cycles != cycles or cycles <= 0:
                raise ValueError("mixed or invalid effective cycles")
            rows.append(np.asarray(item.detector_counts, dtype=float))
        self._consumed.add(key)
        return np.asarray(rows), int(cycles)

    def advance(self) -> None:
        self.policy_version += 1
        self.epoch += 1
        self.environment_time += 1
