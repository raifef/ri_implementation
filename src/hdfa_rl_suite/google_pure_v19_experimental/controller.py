"""Controller identity for the explicitly non-source public-analogue branch."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .io import canonical_hash


CONTROLLER_MODE = "PUBLIC_ANALOGUE_DIRECT_SIGMA_EXPERIMENTAL_V19"
PARAMETERIZATION = "DIRECT_SIGMA_PUBLIC_ANALOGUE_EXPERIMENTAL"
SCALE_OBJECTIVE = "MEAN_JOINT_GAUSSIAN_ENTROPY_OVER_ACTIVE_COORDINATES"


@dataclass(frozen=True)
class PublicAnalogueControllerSpec:
    inherited_entropy_coefficient: float
    active_dimensions: int
    frozen_parent_controller_hash: str
    mean_learning_rate: float
    sigma_learning_rate: float
    baseline_learning_rate: float
    ppo_clip: float
    baseline_loss_weight: float
    minimum_sigma: float
    maximum_sigma: float
    initial_sigma: float

    def __post_init__(self) -> None:
        if self.inherited_entropy_coefficient <= 0 or self.active_dimensions <= 0:
            raise ValueError("public-analogue scale objective requires positive beta and dimension")
        if self.frozen_parent_controller_hash == self.controller_hash:
            raise ValueError("experimental controller identity must differ from its frozen parent")

    @property
    def effective_entropy_coefficient(self) -> float:
        return self.inherited_entropy_coefficient / self.active_dimensions

    @property
    def identity_payload(self) -> dict[str, Any]:
        return {
            "controller_mode": CONTROLLER_MODE,
            "parameterization": PARAMETERIZATION,
            "scale_objective": SCALE_OBJECTIVE,
            "source_exact": False,
            "source_scale_hyperparameters_identifiable": False,
            "inherited_entropy_coefficient": self.inherited_entropy_coefficient,
            "active_dimensions": self.active_dimensions,
            "effective_entropy_coefficient": self.effective_entropy_coefficient,
            "frozen_parent_controller_hash": self.frozen_parent_controller_hash,
            "mean_learning_rate": self.mean_learning_rate,
            "sigma_learning_rate": self.sigma_learning_rate,
            "baseline_learning_rate": self.baseline_learning_rate,
            "ppo_clip": self.ppo_clip,
            "baseline_loss_weight": self.baseline_loss_weight,
            "minimum_sigma": self.minimum_sigma,
            "maximum_sigma": self.maximum_sigma,
            "initial_sigma": self.initial_sigma,
        }

    @property
    def controller_hash(self) -> str:
        return canonical_hash({
            "controller_mode": CONTROLLER_MODE,
            "parameterization": PARAMETERIZATION,
            "scale_objective": SCALE_OBJECTIVE,
            "source_exact": False,
            "inherited_entropy_coefficient": self.inherited_entropy_coefficient,
            "active_dimensions": self.active_dimensions,
            "effective_entropy_coefficient": self.effective_entropy_coefficient,
            "frozen_parent_controller_hash": self.frozen_parent_controller_hash,
            "optimizer": {
                "mean_learning_rate": self.mean_learning_rate,
                "sigma_learning_rate": self.sigma_learning_rate,
                "baseline_learning_rate": self.baseline_learning_rate,
                "ppo_clip": self.ppo_clip,
                "baseline_loss_weight": self.baseline_loss_weight,
                "minimum_sigma": self.minimum_sigma,
                "maximum_sigma": self.maximum_sigma,
                "initial_sigma": self.initial_sigma,
            },
        })
