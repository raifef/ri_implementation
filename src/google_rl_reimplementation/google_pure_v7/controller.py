"""One explicit source-mapped production controller configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from google_rl_reimplementation.google_pure_v6.reference_agent import PureGoogleV6Agent

from .config import canonical_hash, repository_root, sha256_file
from .reporting import read_artifact, write_report


CONTROLLER_MODE = "source_mapped_v7_production_ppo"

RESOLVED_PARAMETERS: dict[str, Any] = {
    "initial_scale": 0.14,
    "minimum_scale": 0.04,
    "maximum_scale": 0.25,
    "entropy_coefficient": 0.0004,
    "mean_learning_rate": 0.02,
    "scale_learning_rate": 0.002,
    "replay_capacity_epochs": 1,
    "baseline_coefficient": 0.08,
    "ppo_clip": 0.2,
    "update_passes": 1,
    "optimizer": "plain_sgd_ascent",
}


def _controller_code_hash() -> str:
    root = repository_root()
    paths = [
        root / "src/google_rl_reimplementation/google_pure_v6/policy.py",
        root / "src/google_rl_reimplementation/google_pure_v6/factor_graph.py",
        root / "src/google_rl_reimplementation/google_pure_v6/baseline.py",
        root / "src/google_rl_reimplementation/google_pure_v6/replay.py",
        root / "src/google_rl_reimplementation/google_pure_v6/update.py",
        root / "src/google_rl_reimplementation/google_pure_v6/reference_agent.py",
        Path(__file__),
    ]
    return canonical_hash({path.relative_to(root).as_posix(): sha256_file(path) for path in paths})


def resolve_production_controller() -> dict[str, Any]:
    root = repository_root()
    base_path = root / "configs/google_pure_v6/source_unspecified_choices.yaml"
    base_hash = sha256_file(base_path)
    resolved_hash = canonical_hash(RESOLVED_PARAMETERS)
    payload = {
        "schema_version": "google-pure-v7-resolved-controller.v1",
        "controller_mode": CONTROLLER_MODE,
        "controller_code_hash": _controller_code_hash(),
        "base_config_hash": base_hash,
        "resolved_config_hash": resolved_hash,
        "parameters": dict(RESOLVED_PARAMETERS),
        "selection_provenance": {
            "initial_scale": "v6 exploration study selected 0.14 under its then-current gates",
            "mean_learning_rate": "v6 one-factor study selected 0.02 under its then-current gates",
            "combined_configuration": "preregistered v7 development candidate; not promoted as scientifically passing",
            "hard_gate_status": "PENDING_V7_SCIENTIFIC_TESTS",
        },
        "all_parameters_explicit": True,
        "legacy_v5_defaults_used": False,
        "objective_mode": CONTROLLER_MODE,
        "certification_seeds_consumed": False,
        "status": "RESOLVED_FOR_DEVELOPMENT",
    }
    return write_report("resolved_production_controller", payload, "Resolved Production Controller")


def require_resolved_controller() -> dict[str, Any]:
    artifact = read_artifact("resolved_production_controller")
    required = {"controller_mode", "controller_code_hash", "base_config_hash", "resolved_config_hash", "parameters"}
    if not required.issubset(artifact) or artifact["controller_mode"] != CONTROLLER_MODE:
        raise RuntimeError("production controller is unresolved")
    if artifact["resolved_config_hash"] != canonical_hash(artifact["parameters"]):
        raise RuntimeError("resolved controller configuration hash mismatch")
    if set(artifact["parameters"]) != set(RESOLVED_PARAMETERS):
        raise RuntimeError("resolved controller contains implicit or missing parameters")
    return artifact


def agent_choices(controller: dict[str, Any] | None = None) -> dict[str, Any]:
    parameters = (controller or require_resolved_controller())["parameters"]
    return {
        "initial_scale": parameters["initial_scale"],
        "scale_bounds": [parameters["minimum_scale"], parameters["maximum_scale"]],
        "normalized_bounds": [-1.0, 1.0],
        "mean_learning_rate": parameters["mean_learning_rate"],
        "scale_learning_rate": parameters["scale_learning_rate"],
        "baseline_coefficient": parameters["baseline_coefficient"],
        "replay_capacity_epochs": parameters["replay_capacity_epochs"],
        "ppo_clip": parameters["ppo_clip"],
        "entropy_coefficient": parameters["entropy_coefficient"],
        "update_passes": parameters["update_passes"],
    }


def build_production_agent(mask: np.ndarray, initial_mean: np.ndarray, coordinates: Any, *, seed: int) -> PureGoogleV6Agent:
    controller = require_resolved_controller()
    return PureGoogleV6Agent(mask, initial_mean, coordinates, agent_choices(controller), seed=seed,
                             objective_mode="source_literal_ppo")
