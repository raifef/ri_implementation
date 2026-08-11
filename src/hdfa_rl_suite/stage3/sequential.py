"""Sequential HDFA-style segmentation baseline; never use as the deployed joint path."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class SequentialSegment:
    start: int
    end: int
    mean: float


def segment_posterior_means(values: Sequence[float], minimum_length: int = 8, jump_threshold: float = 3.0) -> tuple[SequentialSegment, ...]:
    """A transparent diagnostic baseline that intentionally lacks detector-likelihood feedback."""
    if not values:
        return ()
    segments, start = [], 0
    for index in range(minimum_length, len(values)):
        left = values[start:index]
        if len(left) < minimum_length:
            continue
        mean = sum(left) / len(left)
        variance = sum((value - mean) ** 2 for value in left) / max(1, len(left) - 1)
        if abs(values[index] - mean) > jump_threshold * max(variance ** .5, 1e-9):
            segments.append(SequentialSegment(start, index, mean))
            start = index
    tail = values[start:]
    segments.append(SequentialSegment(start, len(values), sum(tail) / len(tail)))
    return tuple(segments)
