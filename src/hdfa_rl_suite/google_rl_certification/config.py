"""Versioned Google-RL certification configuration and cost accounting."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


CONFIG_SCHEMA = "google-rl-config.v1"


@dataclass(frozen=True)
class SamplingConfig:
    candidates_per_epoch: int
    shots_per_candidate: int
    qec_cycles_per_shot: int
    effective_cycles_per_candidate: int
    candidate_design: str
    mean_evaluation_shots: int
    mean_evaluation_period_epochs: int

    def __post_init__(self) -> None:
        if self.candidates_per_epoch < 4 or self.candidates_per_epoch % 2:
            raise ValueError("candidate count must be even and at least four")
        if min(self.shots_per_candidate, self.qec_cycles_per_shot,
               self.effective_cycles_per_candidate, self.mean_evaluation_shots,
               self.mean_evaluation_period_epochs) <= 0:
            raise ValueError("sampling counts must be positive")
        if self.effective_cycles_per_candidate != (
                self.shots_per_candidate * self.qec_cycles_per_shot):
            raise ValueError("effective cycles must equal shots times QEC cycles per shot")
        if self.candidate_design not in {
                "independent_gaussian", "complete_antithetic_pairs"}:
            raise ValueError("unsupported candidate design")


@dataclass(frozen=True)
class PolicyConfig:
    initial_stddev_normalized: float
    mean_learning_rate: float
    log_stddev_learning_rate: float
    baseline_learning_rate: float
    entropy_weight: float
    minimum_stddev_normalized: float
    maximum_stddev_normalized: float
    ppo_clip: float
    optimizer: str
    optimizer_steps: int
    gradient_clip: float
    replay_epochs: int

    def __post_init__(self) -> None:
        if not 0 < self.minimum_stddev_normalized <= self.initial_stddev_normalized <= self.maximum_stddev_normalized:
            raise ValueError("standard-deviation limits are inconsistent")
        if self.mean_learning_rate <= 0 or self.log_stddev_learning_rate < 0:
            raise ValueError("learning rates must be positive/non-negative")
        if not 0 < self.baseline_learning_rate <= 1:
            raise ValueError("baseline learning rate must lie in (0, 1]")
        if self.entropy_weight < 0 or self.gradient_clip <= 0:
            raise ValueError("entropy/gradient limits are invalid")
        if not 0 < self.ppo_clip < 1 or self.optimizer_steps <= 0:
            raise ValueError("PPO clipping and optimizer steps are invalid")
        if self.optimizer not in {"sgd"}:
            raise ValueError("only the declared dependency-free SGD approximation is supported")
        if self.replay_epochs <= 0:
            raise ValueError("replay epoch count must be positive")


@dataclass(frozen=True)
class SafetyConfig:
    absolute_bound_normalized: float
    candidate_slew_normalized: float
    mean_slew_normalized: float

    def __post_init__(self) -> None:
        if min(self.absolute_bound_normalized, self.candidate_slew_normalized,
               self.mean_slew_normalized) <= 0:
            raise ValueError("safety bounds must be positive")
        if self.candidate_slew_normalized > 2 * self.absolute_bound_normalized:
            raise ValueError("candidate slew exceeds the entire bounded range")


@dataclass(frozen=True)
class GoogleRLConfig:
    schema_version: str
    name: str
    evidence_label: str
    sampling: SamplingConfig
    policy: PolicyConfig
    safety: SafetyConfig

    @property
    def native_qec_cycles_per_epoch(self) -> int:
        return (self.sampling.candidates_per_epoch
                * self.sampling.effective_cycles_per_candidate)

    @property
    def mean_evaluation_qec_cycles(self) -> int:
        return (self.sampling.mean_evaluation_shots
                * self.sampling.qec_cycles_per_shot)

    def estimated_cost(self, epochs: int, *, evaluated_policies: int = 1) -> Mapping[str, int]:
        evaluations = (epochs + self.sampling.mean_evaluation_period_epochs - 1) // (
            self.sampling.mean_evaluation_period_epochs)
        candidate_cycles = epochs * self.native_qec_cycles_per_epoch
        evaluation_cycles = evaluations * evaluated_policies * self.mean_evaluation_qec_cycles
        return {
            "epochs": epochs,
            "candidate_evaluations": epochs * self.sampling.candidates_per_epoch,
            "candidate_qec_cycles": candidate_cycles,
            "mean_policy_evaluation_qec_cycles": evaluation_cycles,
            "total_native_qec_cycles": candidate_cycles + evaluation_cycles,
        }


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def load_config(path: str | Path) -> GoogleRLConfig:
    """Load the JSON subset of YAML used for dependency-free versioned configs."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    root = _mapping(payload, "configuration")
    if root.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unsupported Google-RL configuration schema")
    return GoogleRLConfig(
        str(root["schema_version"]), str(root["name"]), str(root["evidence_label"]),
        SamplingConfig(**_mapping(root["sampling"], "sampling")),
        PolicyConfig(**_mapping(root["policy"], "policy")),
        SafetyConfig(**_mapping(root["safety"], "safety")),
    )


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def named_config(name: str) -> GoogleRLConfig:
    filenames = {
        "high_shot_reference": "high_shot_reference.yaml",
        "reduced_budget_candidate": "reduced_budget_candidate.yaml",
    }
    try:
        filename = filenames[name]
    except KeyError as error:
        raise ValueError(f"unknown Google-RL configuration {name!r}") from error
    return load_config(repository_root() / "configs" / "google_rl" / filename)
