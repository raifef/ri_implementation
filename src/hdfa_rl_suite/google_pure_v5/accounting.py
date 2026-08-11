"""Exact separation of acquisition and evaluation costs."""
from __future__ import annotations

from typing import Any, Mapping


def acquisition_accounting(
    epochs: int,
    config: Mapping[str, Any],
    *,
    mean_evaluations: int = 0,
    fixed_evaluations: int = 0,
    logical_evaluations: int = 0,
) -> dict[str, int]:
    candidates = int(config["candidates_per_epoch"])
    shots = int(config["shots_per_candidate"])
    cycles_per_shot = int(config["qec_cycles_per_shot"])
    effective = shots * cycles_per_shot
    if candidates != 40 or shots != 4_000 or cycles_per_shot != 25 or effective != 100_000:
        raise ValueError("reference acquisition must be 40 x 4,000 x 25 = 100,000")
    return {
        "epochs": int(epochs),
        "complete_policy_candidates": int(epochs) * candidates,
        "candidate_acquisition_cycles": int(epochs) * candidates * effective,
        "mean_policy_diagnostic_cycles": int(mean_evaluations) * int(config["mean_policy_diagnostic_cycles"]),
        "fixed_policy_diagnostic_cycles": int(fixed_evaluations) * int(config["fixed_policy_diagnostic_cycles"]),
        "logical_evaluation_cycles": int(logical_evaluations) * int(config["logical_evaluation_cycles"]),
        "effective_cycles_per_candidate": effective,
    }


def command_estimate(
    command: str,
    *,
    epochs: int,
    experiments: int,
    paper_config: Mapping[str, Any],
    certification: bool = False,
    scaling: bool = False,
) -> dict[str, Any]:
    acc = acquisition_accounting(max(epochs, 0) * max(experiments, 0), paper_config)
    return {
        "command": command,
        "estimated_wall_time": "seconds to minutes for deterministic development; paper-scale horizon is user-selected",
        "candidate_count": acc["complete_policy_candidates"],
        "qec_cycle_cost": acc["candidate_acquisition_cycles"],
        "memory_estimate": "<1 GiB" if not scaling else "approximately 80 MiB peak at distance 15",
        "disk_estimate": "<25 MiB",
        "certification_seeds_touched": bool(certification),
    }
