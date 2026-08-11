"""Small deterministic statistics shared by the certification ladder."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    denominator = float(np.linalg.norm(a)*np.linalg.norm(b))
    if denominator <= 1e-15:
        return 1.0 if float(np.linalg.norm(a-b)) <= 1e-15 else 0.0
    return float(np.dot(a, b)/denominator)


def ranking_accuracy(reference: np.ndarray, observed: np.ndarray) -> float:
    expected = np.asarray(reference, dtype=float)
    measured = np.asarray(observed, dtype=float)
    concordant = total = 0
    for first in range(len(expected)):
        for second in range(first+1, len(expected)):
            sign = np.sign(expected[first]-expected[second])
            if sign == 0:
                continue
            concordant += int(sign == np.sign(measured[first]-measured[second]))
            total += 1
    return concordant/max(1, total)


def recovery_endpoints(fractions: Sequence[float], cycles_per_epoch: int,
                       candidate_count: int) -> list[dict[str, Any]]:
    output = []
    for target in (.50, .75, .90):
        reached = next((index for index, value in enumerate(fractions, 1)
                        if value >= target), None)
        output.append({
            "target_fraction": target,
            "status": "reached" if reached is not None else "censored",
            "epochs": reached,
            "native_qec_cycles": reached*cycles_per_epoch if reached is not None else None,
            "candidate_evaluations": reached*candidate_count if reached is not None else None,
            "censoring_reason": None if reached is not None else "threshold_not_observed_within_declared_horizon",
        })
    return output


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True)+"\n",
                    encoding="utf-8")
