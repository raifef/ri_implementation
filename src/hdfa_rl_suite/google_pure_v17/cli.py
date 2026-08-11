"""Dedicated V17 CLIs; none auto-launches source, held-out, or long work."""
from __future__ import annotations

import json
from typing import Any, Callable

from .experiments import (
    audit_figure5a_frequency, audit_figure5a_metric, audit_figure5a_windowing,
    audit_latency_phase, audit_scale_dynamics, audit_sensitivity_semantics,
    build_reduced_acceptance_v2, compare_step_figure5a_modes,
    decompose_mean_stochastic, measure_mean_transfer, refit_step_transfer,
    run_figure5a_deterministic_fixture, run_reduced_postrepair,
)
from .imports import verify_import_manifest
from .io import ARTIFACT_ROOT
from .reporting import build_report, build_status


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    omitted = {"rows", "cells", "diagnostic_hashes", "integer_complete_period_diagnostics"}
    compact = {key: value for key, value in result.items() if key not in omitted}
    if len(compact) != len(result):
        compact["full_artifact_root"] = str(ARTIFACT_ROOT.resolve())
    return compact


def _run(command: str, function: Callable[[], dict[str, Any]]) -> int:
    verify_import_manifest()
    result = function()
    print(json.dumps({"command": command, "result": _compact(result),
                      "output_root": str(ARTIFACT_ROOT.resolve()),
                      "source_budget_auto_launched": False, "heldout_auto_launched": False,
                      "long_run_auto_launched": False, "figure5c_executed": False},
                     indent=2, sort_keys=True))
    return 0 if result.get("pass", True) is not False else 2


def audit_sensitivity_semantics_main() -> int: return _run("audit-sensitivity-semantics", audit_sensitivity_semantics)
def refit_step_transfer_main() -> int: return _run("refit-step-transfer", refit_step_transfer)
def audit_figure5a_frequency_main() -> int: return _run("audit-figure5a-frequency", audit_figure5a_frequency)
def run_figure5a_deterministic_fixture_main() -> int: return _run("run-figure5a-deterministic-fixture", run_figure5a_deterministic_fixture)
def audit_figure5a_metric_main() -> int: return _run("audit-figure5a-metric", audit_figure5a_metric)
def audit_figure5a_windowing_main() -> int: return _run("audit-figure5a-windowing", audit_figure5a_windowing)
def measure_mean_transfer_main() -> int: return _run("measure-mean-transfer", measure_mean_transfer)
def audit_latency_phase_main() -> int: return _run("audit-latency-phase", audit_latency_phase)
def decompose_mean_stochastic_main() -> int: return _run("decompose-mean-stochastic", decompose_mean_stochastic)
def audit_scale_dynamics_main() -> int: return _run("audit-scale-dynamics", audit_scale_dynamics)
def compare_step_figure5a_modes_main() -> int: return _run("compare-step-figure5a-modes", compare_step_figure5a_modes)
def build_reduced_acceptance_v2_main() -> int: return _run("build-reduced-acceptance-v2", build_reduced_acceptance_v2)
def run_reduced_postrepair_main() -> int: return _run("run-reduced-postrepair", run_reduced_postrepair)
def status_main() -> int: return _run("status", build_status)
def report_main() -> int: return _run("report", build_report)
