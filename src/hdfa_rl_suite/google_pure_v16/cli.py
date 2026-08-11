"""Dedicated V16 CLIs; none launches long, held-out, or source-budget work."""
from __future__ import annotations

import json
from typing import Any, Callable

from .coordinate import (
    audit_coordinate_transform,
    audit_entropy_covariance,
    audit_gradient_covariance,
    audit_ppo_covariance,
    build_hypothesis_artifact,
    run_covariance_fixture,
)
from .experiments import run_matched_figure5b, run_matched_step, run_reduced_acceptance
from .imports import build_import_manifest, verify_import_manifest
from .io import ARTIFACT_ROOT
from .optimizer_audits import (
    audit_baseline_reward_scaling,
    audit_direct_sigma,
    audit_local_contraction,
    audit_native_exploration,
    audit_optimizer_sources,
    calibrate_optimizer,
    freeze_optimizer,
    run_source_entropy_anchors,
)
from .reporting import build_report, build_status


def _prepare() -> None:
    if not (ARTIFACT_ROOT / "import_manifest.json").is_file():
        build_import_manifest()
    verify_import_manifest()


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    omitted = {"rows", "artifacts", "step", "recovery", "figure5a", "figure5b",
               "mean_learning_rate_rows", "sigma_learning_rate_rows", "initial_sigma_rows"}
    compact = {key: value for key, value in result.items() if key not in omitted}
    if len(compact) != len(result):
        compact["full_artifact_root"] = str(ARTIFACT_ROOT.resolve())
    return compact


def _run(command: str, function: Callable[[], dict[str, Any]]) -> int:
    _prepare()
    result = function()
    print(json.dumps({
        "command": command, "result": _compact(result),
        "output_root": str(ARTIFACT_ROOT.resolve()),
        "source_budget_auto_launched": False,
        "heldout_auto_launched": False,
        "long_run_auto_launched": False,
    }, indent=2, sort_keys=True))
    return 0 if result.get("pass", True) is not False else 2


def audit_coordinate_transform_main() -> int:
    def run() -> dict[str, Any]:
        build_hypothesis_artifact()
        return audit_coordinate_transform()
    return _run("audit-coordinate-transform", run)


def run_covariance_fixture_main() -> int: return _run("run-covariance-fixture", run_covariance_fixture)
def audit_gradient_covariance_main() -> int: return _run("audit-gradient-covariance", audit_gradient_covariance)
def audit_ppo_covariance_main() -> int: return _run("audit-ppo-covariance", audit_ppo_covariance)
def audit_entropy_covariance_main() -> int: return _run("audit-entropy-covariance", audit_entropy_covariance)
def audit_optimizer_sources_main() -> int: return _run("audit-optimizer-sources", audit_optimizer_sources)
def audit_native_exploration_main() -> int: return _run("audit-native-exploration", audit_native_exploration)
def audit_direct_sigma_main() -> int: return _run("audit-direct-sigma", audit_direct_sigma)
def run_source_entropy_anchors_main() -> int: return _run("run-source-entropy-anchors", run_source_entropy_anchors)
def calibrate_optimizer_main() -> int: return _run("calibrate-optimizer", calibrate_optimizer)
def freeze_optimizer_main() -> int: return _run("freeze-optimizer", freeze_optimizer)
def run_matched_step_main() -> int: return _run("run-matched-step", run_matched_step)
def run_matched_figure5b_main() -> int: return _run("run-matched-figure5b", run_matched_figure5b)
def audit_local_contraction_main() -> int:
    def run() -> dict[str, Any]:
        audit_baseline_reward_scaling()
        return audit_local_contraction()
    return _run("audit-local-contraction", run)
def run_reduced_acceptance_main() -> int: return _run("run-reduced-acceptance", run_reduced_acceptance)
def status_main() -> int: return _run("status", build_status)
def report_main() -> int: return _run("report", build_report)
