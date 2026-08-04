"""Strict configuration and native-QEC-cycle accounting for the v2 reference."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REFERENCE_SCHEMA = "google-public-reference-config.v2"
SURROGATE_SCHEMA = "google-paper-anchored-surrogate.v2"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_json_yaml(path: str | Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("configuration root must be a mapping")
    return payload


@dataclass(frozen=True)
class Sampling:
    candidates_per_epoch: int
    shots_per_candidate: int
    qec_cycles_per_shot: int
    effective_cycles_per_candidate: int
    candidate_design: str
    mean_evaluation_shots: int
    fixed_policy_re_evaluated: bool
    mean_evaluation_period_epochs: int

    def __post_init__(self) -> None:
        if self.candidates_per_epoch != 40:
            raise ValueError("paper-scale reference requires exactly 40 candidates")
        if self.shots_per_candidate != 4000 or self.qec_cycles_per_shot != 25:
            raise ValueError("paper-scale reference requires 4,000 shots of 25 cycles")
        if self.effective_cycles_per_candidate != 100_000:
            raise ValueError("paper-scale reference requires 100,000 effective cycles per candidate")
        if self.effective_cycles_per_candidate != self.shots_per_candidate * self.qec_cycles_per_shot:
            raise ValueError("shot/cycle accounting is inconsistent")
        if self.candidate_design != "independent_diagonal_gaussian":
            raise ValueError("the public protocol uses independently sampled policies")
        if not self.fixed_policy_re_evaluated or min(
            self.mean_evaluation_shots, self.mean_evaluation_period_epochs
        ) <= 0:
            raise ValueError("independent learned-mean/fixed evaluation is required")


@dataclass(frozen=True)
class AgentChoices:
    initial_stddev_normalized: float
    mean_learning_rate: float
    log_stddev_learning_rate: float
    baseline_learning_rate: float
    entropy_weight: float
    entropy_scale_mode: str
    minimum_stddev_normalized: float
    maximum_stddev_normalized: float
    ppo_clip: float
    gradient_clip: float
    optimizer_steps: int
    replay_capacity_epochs: int
    replay_max_regime_age_epochs: int
    absolute_bound_normalized: float

    def __post_init__(self) -> None:
        if not 0 < self.minimum_stddev_normalized <= self.initial_stddev_normalized <= self.maximum_stddev_normalized:
            raise ValueError("policy standard-deviation bounds are inconsistent")
        if min(self.mean_learning_rate, self.baseline_learning_rate, self.gradient_clip) <= 0:
            raise ValueError("learning rates and gradient clip must be positive")
        if self.log_stddev_learning_rate < 0 or self.entropy_weight < 0:
            raise ValueError("scale learning and entropy weight must be non-negative")
        if self.entropy_scale_mode not in {"absolute", "mean_absolute_advantage"}:
            raise ValueError("unsupported entropy scaling mode")
        if not 0 < self.ppo_clip < 1 or self.optimizer_steps < 1:
            raise ValueError("PPO choices are invalid")
        if min(self.replay_capacity_epochs, self.replay_max_regime_age_epochs) < 0:
            raise ValueError("replay limits must be non-negative")
        if self.absolute_bound_normalized <= 0:
            raise ValueError("control bound must be positive")


@dataclass(frozen=True)
class ReferenceConfig:
    schema_version: str
    name: str
    evidence_layer: str
    sampling: Sampling
    agent: AgentChoices
    development_seeds: tuple[int, ...]
    untouched_certification_seeds: tuple[int, ...]
    protected_prior_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        groups = [set(self.development_seeds), set(self.untouched_certification_seeds), set(self.protected_prior_seeds)]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("development, certification, and protected seed sets must be disjoint")

    def cost(self, epochs: int, *, policies_per_evaluation: int = 2) -> dict[str, int]:
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        evaluations = (epochs + self.sampling.mean_evaluation_period_epochs - 1) // self.sampling.mean_evaluation_period_epochs
        candidate_cycles = epochs * self.sampling.candidates_per_epoch * self.sampling.effective_cycles_per_candidate
        diagnostic_cycles = evaluations * policies_per_evaluation * self.sampling.mean_evaluation_shots * self.sampling.qec_cycles_per_shot
        return {
            "epochs": epochs,
            "candidate_count": epochs * self.sampling.candidates_per_epoch,
            "candidate_native_qec_cycles": candidate_cycles,
            "diagnostic_native_qec_cycles": diagnostic_cycles,
            "total_native_qec_cycles": candidate_cycles + diagnostic_cycles,
            "ideal_candidate_acquisition_seconds": epochs * self.sampling.candidates_per_epoch * 0.1,
        }


def load_reference_config(path: str | Path | None = None) -> ReferenceConfig:
    path = path or repository_root() / "configs/google_rl/paper_scale_reference_v2.yaml"
    value = _read_json_yaml(path)
    if value.get("schema_version") != REFERENCE_SCHEMA:
        raise ValueError("unsupported reference schema")
    return ReferenceConfig(
        schema_version=str(value["schema_version"]),
        name=str(value["name"]),
        evidence_layer=str(value["evidence_layer"]),
        sampling=Sampling(**value["sampling"]),
        agent=AgentChoices(**value["agent"]),
        development_seeds=tuple(int(x) for x in value["development_seeds"]),
        untouched_certification_seeds=tuple(int(x) for x in value["untouched_certification_seeds"]),
        protected_prior_seeds=tuple(int(x) for x in value["protected_prior_seeds"]),
    )


def load_surrogate_config(path: str | Path | None = None) -> Mapping[str, Any]:
    path = path or repository_root() / "configs/google_rl/paper_anchored_surrogate_v2.yaml"
    value = _read_json_yaml(path)
    if value.get("schema_version") != SURROGATE_SCHEMA:
        raise ValueError("unsupported surrogate schema")
    return value


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
