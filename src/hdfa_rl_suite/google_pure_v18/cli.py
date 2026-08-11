"""Dedicated V18 commands with no implicit long or confirmatory campaign."""
from __future__ import annotations

import json
from typing import Any, Callable

from .experiments import (
    audit_delta_min_provenance, build_figure5b_learning_rate_note,
    build_mean_stochastic_decomposition, build_paired_acceptance_readiness,
    build_sensitivity_field_cleanup, build_steady_state_rule, run_transfer_fast,
    run_transfer_intermediate, run_transfer_slow, validate_deterministic_transfer,
)
from .imports import verify_import_manifest
from .io import ARTIFACT_ROOT
from .reporting import build_report, build_status


def _compact(value: dict[str, Any]) -> dict[str, Any]:
    omitted = {"rows", "static_probe_rows", "period_diagnostics", "gain_samples",
               "phase_lag_samples_radians", "artifact_inventory", "status"}
    compact = {key: item for key, item in value.items() if key not in omitted}
    bootstrap = compact.get("bootstrap_uncertainty")
    if isinstance(bootstrap, dict):
        compact["bootstrap_uncertainty"] = {
            key: item for key, item in bootstrap.items() if key not in omitted}
    return compact


def _run(command: str, function: Callable[[], dict[str, Any]]) -> int:
    result = function()
    print(json.dumps({
        "command": command, "result": _compact(result),
        "output_root": str(ARTIFACT_ROOT.resolve()),
        "figure5c_auto_launched": False, "natural_drift_auto_launched": False,
        "heldout_auto_launched": False, "reference_auto_launched": False,
        "source_budget_auto_launched": False,
        "full_slow_fast_acceptance_auto_launched": False,
    }, indent=2, sort_keys=True))
    return 0 if result.get("pass", True) is not False else 2


def import_manifest_main() -> int: return _run("import-manifest", verify_import_manifest)
def sensitivity_cleanup_main() -> int: return _run("sensitivity-cleanup", build_sensitivity_field_cleanup)
def deterministic_transfer_main() -> int: return _run("deterministic-transfer", validate_deterministic_transfer)
def delta_min_provenance_main() -> int: return _run("delta-min-provenance", audit_delta_min_provenance)
def steady_state_rule_main() -> int: return _run("steady-state-rule", build_steady_state_rule)
def transfer_intermediate_main() -> int: return _run("transfer-intermediate", run_transfer_intermediate)
def transfer_fast_main() -> int: return _run("transfer-fast", run_transfer_fast)
def transfer_slow_main() -> int: return _run("transfer-slow", run_transfer_slow)
def mean_stochastic_main() -> int: return _run("mean-stochastic", build_mean_stochastic_decomposition)
def acceptance_readiness_main() -> int: return _run("acceptance-readiness", build_paired_acceptance_readiness)
def figure5b_note_main() -> int: return _run("figure5b-note", build_figure5b_learning_rate_note)
def status_main() -> int: return _run("status", build_status)
def report_main() -> int: return _run("report", build_report)


def extended_fast_validation_main() -> int:
    from .extended_fast import run_extended_fast_validation

    result = run_extended_fast_validation()
    print(json.dumps({
        "command": "run-extended-fast-validation", "result": result,
        "output_root": str((ARTIFACT_ROOT / "extended_fast").resolve()),
        "slow_auto_launched": False, "paired_acceptance_auto_launched": False,
        "source_budget_auto_launched": False, "heldout_auto_launched": False,
        "reference_auto_launched": False, "natural_drift_auto_launched": False,
        "figure5c_auto_launched": False,
    }, indent=2, sort_keys=True))
    return 0 if result.get("execution_complete") is True else 2
