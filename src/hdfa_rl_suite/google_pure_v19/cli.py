"""Dedicated V19 diagnostic commands; none launches a production acquisition."""
from __future__ import annotations

import json
from typing import Any, Callable

from .diagnostics import (
    audit_entropy_reward_aggregation,
    audit_exploration_damage,
    audit_phase_sigma_gradients,
    build_import_manifest,
    build_report,
    build_status,
    classify_root_cause,
    decompose_exploration_damage,
    derive_sigma_equilibrium,
    run_all,
    run_frozen_sigma_sweep,
    run_minimal_repair_validation,
)
from .io import ARTIFACT_ROOT


def _compact(value: dict[str, Any]) -> dict[str, Any]:
    omitted = {"state_rows", "damage_per_coordinate", "phase_bins", "coordinates",
               "cumulative_damage_rank_curve", "rows"}
    return {key: item for key, item in value.items() if key not in omitted}


def _run(command: str, function: Callable[[], dict[str, Any]]) -> int:
    result = function()
    print(json.dumps({
        "command": command, "result": _compact(result),
        "output_root": str(ARTIFACT_ROOT.resolve()),
        "production_acquisition_launched": False,
        "source_budget_auto_launched": False,
        "heldout_auto_launched": False,
        "reference_auto_launched": False,
        "natural_drift_auto_launched": False,
        "figure5c_auto_launched": False,
        "paired_acceptance_auto_launched": False,
    }, indent=2, sort_keys=True))
    return 0 if result.get("pass", result.get("execution_complete", True)) is not False else 2


def import_manifest_main() -> int:
    return _run("import-manifest", build_import_manifest)


def exploration_damage_main() -> int:
    return _run("audit-exploration-damage", audit_exploration_damage)


def decomposition_main() -> int:
    return _run("decompose-exploration-damage", decompose_exploration_damage)


def aggregation_main() -> int:
    return _run("audit-entropy-reward-aggregation", audit_entropy_reward_aggregation)


def equilibrium_main() -> int:
    return _run("derive-sigma-equilibrium", derive_sigma_equilibrium)


def phase_main() -> int:
    return _run("audit-phase-sigma-gradients", audit_phase_sigma_gradients)


def sweep_main() -> int:
    return _run("run-frozen-sigma-sweep", run_frozen_sigma_sweep)


def root_cause_main() -> int:
    return _run("classify-root-cause", classify_root_cause)


def repair_validation_main() -> int:
    return _run("run-minimal-repair-validation", run_minimal_repair_validation)


def status_main() -> int:
    return _run("status", build_status)


def report_main() -> int:
    return _run("report", build_report)


def run_all_main() -> int:
    return _run("run-all-bounded-diagnostics", run_all)
