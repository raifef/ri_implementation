"""Executable controller, temporal, decomposition, and selection contracts for v9."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Mapping

import numpy as np

from google_rl_reimplementation.google_pure_v8.contracts import (
    cost_decomposition,
    normalized_edr_improvement,
)

from .common import guard_seed_registry, protocol_hash


SOURCE_CLASSES = frozenset(
    {
        "SOURCE_LITERAL",
        "SOURCE_ANCHORED",
        "SOURCE_UNSPECIFIED_BUT_PREREGISTERED",
        "NON_SOURCE_EXTENSION",
    }
)
PURE_BASELINE_CLASSES = SOURCE_CLASSES - {"NON_SOURCE_EXTENSION"}
INITIAL_SCALE_GRID = (0.01, 0.02, 0.04, 0.06, 0.09, 0.12, 0.14)
ENTROPY_GRID = (0.0, 0.0004, 0.001, 0.01, 0.02, 0.1)
SCALE_LEARNING_RATE_GRID = (0.001, 0.002, 0.005, 0.01, 0.02)
REQUIRED_FREQUENCIES = (1 / 300, 1 / 150, 1 / 60)
REQUIRED_PHASES = (0.0, 2 * math.pi / 3, 4 * math.pi / 3)


@dataclass(frozen=True)
class ControllerConfig:
    initial_scale: float
    minimum_scale: float
    maximum_scale: float
    scale_learning_rate: float = 0.002
    entropy_coefficient: float = 0.0004
    mean_learning_rate: float = 0.02
    replay_capacity_epochs: int = 1
    update_passes: int = 1
    ppo_clip: float = 0.2
    baseline_coefficient: float = 0.08
    optimizer: str = "plain_sgd_ascent"
    scale_parameterization: str = "log_scale"

    def __post_init__(self) -> None:
        values = (self.initial_scale, self.minimum_scale, self.maximum_scale)
        if not all(np.isfinite(values)) or not (0 < self.minimum_scale < self.initial_scale < self.maximum_scale):
            raise ValueError("minimum, initial, and maximum scales must be independent and strictly ordered")
        if self.mean_learning_rate <= 0 or self.scale_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if self.entropy_coefficient < 0 or self.replay_capacity_epochs < 0 or self.update_passes < 1:
            raise ValueError("invalid update lifecycle")
        if self.scale_parameterization != "log_scale" or self.optimizer != "plain_sgd_ascent":
            raise ValueError("v9 preserves the Gaussian log-scale SGD architecture")

    def to_agent_choices(self) -> dict[str, Any]:
        return {
            "initial_scale": float(self.initial_scale),
            "scale_bounds": [float(self.minimum_scale), float(self.maximum_scale)],
            "normalized_bounds": [-1.0, 1.0],
            "mean_learning_rate": float(self.mean_learning_rate),
            "scale_learning_rate": float(self.scale_learning_rate),
            "baseline_coefficient": float(self.baseline_coefficient),
            "replay_capacity_epochs": int(self.replay_capacity_epochs),
            "ppo_clip": float(self.ppo_clip),
            "entropy_coefficient": float(self.entropy_coefficient),
            "update_passes": int(self.update_passes),
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TemporalProtocol:
    frequencies: tuple[float, ...]
    phases: tuple[float, ...]
    burn_in_periods: int
    primary_periods: int
    extension_periods: int
    seeds: tuple[int, ...]
    mode: str
    window_tolerance: float = 0.15

    def __post_init__(self) -> None:
        if not self.frequencies or any(not np.isfinite(value) or value <= 0 for value in self.frequencies):
            raise ValueError("frequencies must be positive cycles per epoch")
        if not self.phases or len(set(self.phases)) != len(self.phases):
            raise ValueError("phase averaging requires a non-empty unique phase registry")
        if self.burn_in_periods < 1 or self.primary_periods < 1 or self.extension_periods != 1:
            raise ValueError("protocol requires burn-in and one-period window perturbations")
        if self.mode in {"validation", "reference"}:
            if self.primary_periods < 5:
                raise ValueError("validation requires at least five complete post-burn-in periods")
            if len(self.phases) < 3 or not all(any(np.isclose(x, y) for x in self.phases) for y in REQUIRED_PHASES):
                raise ValueError("validation requires the frozen three-phase registry")
            if not all(any(np.isclose(x, y) for x in self.frequencies) for y in REQUIRED_FREQUENCIES):
                raise ValueError("validation requires the frozen slow, near-critical, and fast frequencies")
        guard_seed_registry(self.seeds)
        if not 0 < self.window_tolerance < 1:
            raise ValueError("window tolerance must lie in (0,1)")

    @property
    def acquisition_periods(self) -> int:
        return self.primary_periods + self.extension_periods

    def epochs_for(self, frequency: float) -> dict[str, int]:
        period = int(round(1.0 / float(frequency)))
        if not np.isclose(period * frequency, 1.0, rtol=0, atol=1e-10):
            raise ValueError("configured frequencies must have an integer epoch period")
        burn = self.burn_in_periods * period
        return {
            "period_epochs": period,
            "burn_in_epochs": burn,
            "primary_analysis_epochs": self.primary_periods * period,
            "acquisition_epochs": burn + self.acquisition_periods * period,
        }

    def plan(self, *, controller_count: int, candidates: int, cycles: int) -> dict[str, Any]:
        per_frequency = {str(value): self.epochs_for(value) for value in self.frequencies}
        total_epochs = sum(row["acquisition_epochs"] for row in per_frequency.values())
        runs = controller_count * len(self.frequencies) * len(self.phases) * len(self.seeds)
        qec_cycles = controller_count * len(self.phases) * len(self.seeds) * total_epochs * candidates * cycles
        payload = {
            "mode": self.mode,
            "runs": runs,
            "frequencies_cycles_per_epoch": list(self.frequencies),
            "phases_radians": list(self.phases),
            "seeds": list(self.seeds),
            "periods": self.primary_periods,
            "burn_in_periods": self.burn_in_periods,
            "window_extension_periods": self.extension_periods,
            "per_frequency": per_frequency,
            "candidates": candidates,
            "cycles_per_candidate": cycles,
            "estimated_qec_cycles": qec_cycles,
            "estimated_runtime": "seconds for smoke; explicit user-run acquisition for validation/reference",
            "estimated_storage_bytes": int(runs * max(row["acquisition_epochs"] for row in per_frequency.values()) * 1024),
        }
        payload["protocol_hash"] = protocol_hash(payload)
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "frequencies": list(self.frequencies),
            "phases": list(self.phases),
            "burn_in_periods": self.burn_in_periods,
            "primary_periods": self.primary_periods,
            "extension_periods": self.extension_periods,
            "seeds": list(self.seeds),
            "mode": self.mode,
            "window_tolerance": self.window_tolerance,
        }


def scale_floor_classification(fraction_at_floor: float) -> str:
    fraction = float(fraction_at_floor)
    if not 0 <= fraction <= 1:
        raise ValueError("floor fraction must lie in [0,1]")
    return "MINIMUM_SCALE_FLOOR_EFFECT_NOT_ESTABLISHED" if fraction == 0 else "MINIMUM_SCALE_REACHED_DIAGNOSTIC_ONLY"


def validate_source_choices(choices: Mapping[str, str], *, pure_baseline: bool = True) -> None:
    required = {
        "initial_scale",
        "minimum_scale",
        "maximum_scale",
        "mean_learning_rate",
        "scale_learning_rate",
        "entropy_coefficient",
        "replay_capacity_epochs",
        "update_passes",
        "optimizer",
    }
    if set(choices) != required:
        raise ValueError("every changed or retained controller field needs exactly one source classification")
    permitted = PURE_BASELINE_CLASSES if pure_baseline else SOURCE_CLASSES
    if any(value not in permitted for value in choices.values()):
        raise ValueError("controller contains an impermissible source classification")


def five_policy_decomposition(costs: Mapping[str, float], *, resolution: float = 0.0) -> dict[str, float | bool]:
    required = {"fixed", "oracle", "oracle_with_scale", "learned_mean", "sampled_candidates"}
    if set(costs) != required:
        raise ValueError("every cell must contain the five frozen policy classes")
    fixed = float(costs["fixed"])
    oracle = float(costs["oracle"])
    decomposition = cost_decomposition(fixed, float(costs["sampled_candidates"]), float(costs["learned_mean"]), oracle)
    result: dict[str, float | bool] = {
        "C_fixed": fixed,
        "C_oracle": oracle,
        "C_oracle_with_scale": float(costs["oracle_with_scale"]),
        "C_mean": float(costs["learned_mean"]),
        "C_candidate": float(costs["sampled_candidates"]),
        "D_fixed": decomposition["d_fixed"],
        "D_tracking": decomposition["d_tracking"],
        "D_exploration": decomposition["d_exploration"],
        "I_fixed": 0.0,
        "I_oracle": 1.0,
        "I_oracle_with_scale": normalized_edr_improvement(fixed, float(costs["oracle_with_scale"]), oracle, resolution=resolution),
        "I_mean": normalized_edr_improvement(fixed, float(costs["learned_mean"]), oracle, resolution=resolution),
        "I_candidate": normalized_edr_improvement(fixed, float(costs["sampled_candidates"]), oracle, resolution=resolution),
        "denominator_positive_and_resolved": bool(fixed - oracle > max(0.0, resolution)),
        "decomposition_identity_pass": bool(np.isclose(decomposition["direct_improvement"], decomposition["decomposition_improvement"])),
    }
    return result


def entropy_operationality(rows: Iterable[Mapping[str, float]], *, relative_threshold: float = 0.05) -> dict[str, Any]:
    records = tuple(rows)
    if len(records) < 2:
        raise ValueError("entropy operationality requires at least two coefficient settings")
    metrics = ("mean_scale", "native_candidate_displacement", "D_exploration", "I_candidate")
    changed: dict[str, bool] = {}
    spans: dict[str, float] = {}
    for name in metrics:
        values = np.asarray([float(row[name]) for row in records], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"non-finite entropy diagnostic: {name}")
        span = float(np.ptp(values))
        reference = max(float(np.max(np.abs(values))), 1e-12)
        spans[name] = span
        changed[name] = span / reference >= relative_threshold
    count = sum(changed.values())
    return {
        "classification": "ENTROPY_AXIS_OPERATIONAL" if count >= 2 else "ENTROPY_COEFFICIENT_EFFECT_TOO_SMALL_FOR_PROTOCOL",
        "operational": count >= 2,
        "materially_changed_metrics": [name for name, value in changed.items() if value],
        "metric_spans": spans,
        "required_changed_metric_count": 2,
    }

def window_stability(primary: Mapping[str, float], minus: Mapping[str, float], plus: Mapping[str, float], *, tolerance: float) -> dict[str, Any]:
    fields = ("I_mean", "I_candidate", "D_tracking", "D_exploration")
    changes = {
        name: max(abs(float(primary[name]) - float(minus[name])), abs(float(primary[name]) - float(plus[name])))
        for name in fields
    }
    maximum = max(changes.values())
    return {
        "changes": changes,
        "maximum_absolute_change": maximum,
        "tolerance": float(tolerance),
        "stable": maximum <= tolerance,
        "classification": "PASS" if maximum <= tolerance else "TEMPORAL_RESULT_NOT_WINDOW_STABLE",
    }


def controller_selection_gates(summary: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "I_mean_ci_lower",
        "I_candidate_phase_average",
        "D_fixed",
        "D_exploration",
        "tracking_gain_ci_lower",
        "phase_identifiable",
        "window_stable",
        "clipping_fraction",
        "entropy_operational",
        "held_out_protocol_frozen",
        "plant_hash_unchanged",
        "phase_count",
        "mode",
    }
    missing = sorted(required - set(summary))
    if missing:
        raise ValueError(f"selection summary is incomplete: {missing}")
    gates = {
        "learned_mean_positive_with_uncertainty": float(summary["I_mean_ci_lower"]) > 0,
        "sampled_candidates_positive_phase_average": float(summary["I_candidate_phase_average"]) > 0,
        "exploration_below_fixed_degradation": float(summary["D_exploration"]) < float(summary["D_fixed"]),
        "tracking_gain_materially_positive": float(summary["tracking_gain_ci_lower"]) > 0,
        "phase_estimate_identifiable": bool(summary["phase_identifiable"]),
        "window_stable": bool(summary["window_stable"]),
        "clipping_guard": float(summary["clipping_fraction"]) <= 0.01,
        "entropy_axis_operational": bool(summary["entropy_operational"]),
        "held_out_protocol_frozen": bool(summary["held_out_protocol_frozen"]),
        "plant_frozen": bool(summary["plant_hash_unchanged"]),
        "phase_averaging_complete": int(summary["phase_count"]) >= 3,
        "held_out_evidence_mode": str(summary["mode"]) in {"validation", "reference"},
    }
    return {
        "gates": gates,
        "eligible": all(gates.values()),
        "blocking_reasons": [name for name, passed in gates.items() if not passed],
    }
