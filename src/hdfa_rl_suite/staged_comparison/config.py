"""Frozen development configuration and scientific thresholds for Track B."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


TRACK_B_SCHEMA = "staged-vs-certified-rl-config.v1"
DEVELOPMENT_SEEDS = (6101, 6102, 6103)
PROTECTED_CONFIRMATORY_V3_SEEDS = tuple(range(5001, 5025))
FUTURE_TRACK_B_CONFIRMATORY_SEEDS = tuple(range(7101, 7113))


@dataclass(frozen=True)
class TrackBConfig:
    schema_version: str = TRACK_B_SCHEMA
    development_seeds: tuple[int, ...] = DEVELOPMENT_SEEDS
    plant_a_intervals: int = 84
    plant_b_intervals: int = 96
    onset_interval: int = 8
    endpoint_evaluation_cycles: int = 20_000
    logical_evaluation_shots: int = 100_000
    stage2_probe_cycles: int = 24_576
    stage2_probe_normalized: float = 0.12
    residual_candidate_count: int = 4
    residual_candidate_cycles: int = 32_768
    periodic_cadence_intervals: int = 12
    periodic_characterization_shots: int = 512
    final_window_intervals: int = 12
    material_detector_gap: float = 0.00015
    detector_noninferiority_margin: float = 0.0015
    logical_noninferiority_margin: float = 0.00020
    predictive_only_maximum_gap_fraction: float = 0.35
    minimum_residual_relative_benefit: float = 0.05
    no_residual_relative_noninferiority_margin: float = 0.02
    minimum_one_interval_recovery_fraction: float = 0.90
    minimum_structured_recovery_speed_ratio: float = 2.0
    minimum_candidate_efficiency_ratio: float = 10.0
    minimum_excess_edr_ratio: float = 5.0
    minimum_exploration_damage_ratio: float = 2.0
    maximum_detector_probability: float = 0.35

    def __post_init__(self) -> None:
        if self.schema_version != TRACK_B_SCHEMA:
            raise ValueError("unsupported Track-B configuration schema")
        if not self.development_seeds:
            raise ValueError("development seeds cannot be empty")
        protected = set(PROTECTED_CONFIRMATORY_V3_SEEDS) | set(FUTURE_TRACK_B_CONFIRMATORY_SEEDS)
        if protected.intersection(self.development_seeds):
            raise ValueError("development seeds overlap protected confirmatory seeds")
        counts = (
            self.plant_a_intervals, self.plant_b_intervals, self.onset_interval,
            self.endpoint_evaluation_cycles, self.logical_evaluation_shots,
            self.stage2_probe_cycles, self.residual_candidate_count,
            self.residual_candidate_cycles, self.periodic_cadence_intervals,
            self.periodic_characterization_shots, self.final_window_intervals,
        )
        if min(counts) <= 0:
            raise ValueError("Track-B counts and intervals must be positive")
        if self.residual_candidate_count < 4 or self.residual_candidate_count % 2:
            raise ValueError("residual candidate count must be even and at least four")
        if not 0 < self.minimum_residual_relative_benefit <= 1:
            raise ValueError("minimum residual benefit must be a relative fraction")
        if self.material_detector_gap <= 0:
            raise ValueError("material detector gap must be positive")
        if self.minimum_residual_relative_benefit < 0.05:
            raise ValueError("Track B requires at least a five-percent residual benefit")
        if not 0 < self.minimum_one_interval_recovery_fraction <= 1:
            raise ValueError("one-interval recovery fraction must be in (0,1]")
        if min(
            self.minimum_structured_recovery_speed_ratio,
            self.minimum_candidate_efficiency_ratio,
            self.minimum_excess_edr_ratio,
            self.minimum_exploration_damage_ratio,
        ) <= 1:
            raise ValueError("comparative improvement ratios must exceed one")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
