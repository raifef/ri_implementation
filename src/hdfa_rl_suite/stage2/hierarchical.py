"""Sparse many-region coordination for explicitly shared operational factors."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping, Sequence

from .schema import PhysicalStatePosterior, PosteriorSample


@dataclass(frozen=True)
class SharedFactorBelief:
    variable_id: str
    mean: float
    variance: float
    contributing_regions: tuple[str, ...]
    disagreement_chi2: float
    valid: bool


@dataclass(frozen=True)
class HierarchicalInferenceResult:
    regional_posteriors: Mapping[str, PhysicalStatePosterior]
    shared_factors: Mapping[str, SharedFactorBelief]
    invalidity_reasons: tuple[str, ...]


class HierarchicalInferenceCoordinator:
    """Fuse shared factors by precision while retaining every local posterior/sample set."""

    def __init__(self, shared_variable_ids: Sequence[str], *, disagreement_threshold: float = 16.) -> None:
        self.shared_variable_ids = tuple(shared_variable_ids)
        self.disagreement_threshold = disagreement_threshold

    def combine(self, posteriors: Sequence[PhysicalStatePosterior]) -> HierarchicalInferenceResult:
        by_region = {posterior.region_id: posterior for posterior in posteriors}
        shared: dict[str, SharedFactorBelief] = {}
        invalid: list[str] = []
        for variable in self.shared_variable_ids:
            entries = []
            for posterior in posteriors:
                if variable not in posterior.mean:
                    continue
                names = tuple(posterior.mean)
                index = names.index(variable)
                variance = max(posterior.covariance[index][index], 1e-9)
                entries.append((posterior.region_id, posterior.mean[variable], variance))
            if not entries:
                continue
            precision = sum(1 / variance for _, _, variance in entries)
            mean = sum(value / variance for _, value, variance in entries) / precision
            variance = 1 / precision
            disagreement = sum((value - mean) ** 2 / local_variance for _, value, local_variance in entries)
            valid = disagreement <= self.disagreement_threshold
            shared[variable] = SharedFactorBelief(variable, mean, variance,
                tuple(region for region, _, _ in entries), disagreement, valid)
            if not valid:
                invalid.append(f"shared factor {variable} is incoherent across regions")
                continue
            for region, _, _ in entries:
                posterior = by_region[region]
                names = tuple(posterior.mean)
                index = names.index(variable)
                updated_mean = dict(posterior.mean)
                updated_mean[variable] = mean
                covariance = [list(row) for row in posterior.covariance]
                covariance[index][index] = variance
                samples = tuple(PosteriorSample({**sample.state, variable: sample.state.get(variable, mean)
                    + (mean - posterior.mean[variable])}, sample.weight) for sample in posterior.samples)
                by_region[region] = replace(posterior, mean=updated_mean,
                    covariance=tuple(tuple(row) for row in covariance), samples=samples,
                    shared_component_mean={**posterior.shared_component_mean, variable: mean})
        return HierarchicalInferenceResult(by_region, shared, tuple(invalid))
