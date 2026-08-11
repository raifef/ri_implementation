"""Gradient-direction diagnostics for sparse detector-control objectives."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


@dataclass(frozen=True)
class GradientDiagnostic:
    estimated_gradient: Mapping[str, float]
    true_descent_gradient: Mapping[str, float]
    cosine_similarity: float
    estimated_norm: float
    true_norm: float
    positive_alignment: bool


def cosine_similarity(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = tuple(sorted(set(left) | set(right)))
    numerator = sum(float(left.get(key, 0.))*float(right.get(key, 0.)) for key in keys)
    left_norm = math.sqrt(sum(float(left.get(key, 0.))**2 for key in keys))
    right_norm = math.sqrt(sum(float(right.get(key, 0.))**2 for key in keys))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator/(left_norm*right_norm)


def diagnose_gradient(estimated: Mapping[str, float], true_descent: Mapping[str, float],
                      *, minimum_cosine: float = .5) -> GradientDiagnostic:
    cosine = cosine_similarity(estimated, true_descent)
    estimated_norm = math.sqrt(sum(float(value)**2 for value in estimated.values()))
    true_norm = math.sqrt(sum(float(value)**2 for value in true_descent.values()))
    return GradientDiagnostic(
        dict(estimated), dict(true_descent), cosine, estimated_norm, true_norm,
        cosine >= minimum_cosine and estimated_norm > 0 and true_norm > 0,
    )
