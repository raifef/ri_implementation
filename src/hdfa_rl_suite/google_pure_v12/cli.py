"""Dedicated V12 command-line entry points."""
from __future__ import annotations

import json
from typing import Any, Callable

from .directional import (
    _cases,
    audit_directional_gradient,
    audit_directional_sensitivity,
    audit_factor_graph_direction,
    audit_gradient_snr,
    audit_units,
    audit_update_efficiency,
    compare_protocols,
    run_directional_comparison,
)
from .imports import build_import_manifest, validate_import_manifest
from .io import ARTIFACT_ROOT, canonical_hash
from .lineage import audit_figure5b_lineage, audit_figure5c_lineage, validate_figure5c_derivative
from .reporting import build_report, build_status
from .spectral import analyse_natural_drift_uncertainty, validate_natural_drift_sign


def _imports() -> None:
    if not (ARTIFACT_ROOT / "immutable_import_manifest.json").exists():
        build_import_manifest()
    validate_import_manifest()


def _plan(command: str) -> dict[str, Any]:
    cases = _cases()
    identity = json.loads((ARTIFACT_ROOT.parent / "google_pure_source_exact/direct_sigma_integration/controller_identity.json").read_text(encoding="utf-8"))
    return {"command": command, "controller_mode": identity["controller_mode"],
            "controller_hash": identity["controller_hash"], "controller_code_hash": identity["controller_code_hash"],
            "plant_hashes": {name: case.plant_hash for name, case in cases.items()},
            "graph_hashes": {name: case.graph_hash for name, case in cases.items()},
            "protocol_hash": canonical_hash({name: {"seed": case.seed, "epochs": case.epochs,
                "candidates": case.candidates, "cycles": case.cycles} for name, case in cases.items()}),
            "seeds": {name: case.seed for name, case in cases.items()},
            "candidate_counts": {name: case.candidates for name, case in cases.items()},
            "cycle_budgets": {name: case.cycles for name, case in cases.items()},
            "expected_runtime": "under 1 minute" if command != "run-directional-comparison" else "measured about 14 minutes on the development machine; validation only",
            "output_root": str(ARTIFACT_ROOT.resolve()),
            "output_paths": [str((ARTIFACT_ROOT / name).resolve()) for name in (
                "audits", "lineage", "spectral", "directional_comparison", "status.json", "FINAL_REPORT.md")],
            "paper_scale_auto_launch": False}


def _run(command: str, function: Callable[[], dict[str, Any]]) -> int:
    _imports()
    result = function()
    compact = {key: value for key, value in result.items()
               if key not in {"rows", "trajectory_table", "conditions"}}
    if len(compact) != len(result):
        compact["full_result_path"] = str(ARTIFACT_ROOT.resolve())
    print(json.dumps({"execution_plan": _plan(command), "result": compact}, indent=2, sort_keys=True))
    return 0


def audit_directional_sensitivity_main() -> int: return _run("audit-directional-sensitivity", audit_directional_sensitivity)
def audit_factor_graph_direction_main() -> int: return _run("audit-factor-graph-direction", audit_factor_graph_direction)
def audit_directional_gradient_main() -> int: return _run("audit-directional-gradient", audit_directional_gradient)
def audit_gradient_snr_main() -> int: return _run("audit-gradient-snr", audit_gradient_snr)
def audit_update_efficiency_main() -> int: return _run("audit-update-efficiency", audit_update_efficiency)
def audit_step_units_main() -> int: return _run("audit-step-units", lambda: audit_units("step"))
def audit_spoil_units_main() -> int: return _run("audit-spoil-units", lambda: audit_units("spoil"))
def compare_fig5a_step_main() -> int: return _run("compare-fig5a-step", compare_protocols)
def audit_figure5b_lineage_main() -> int: return _run("audit-figure5b-lineage", audit_figure5b_lineage)
def audit_figure5c_lineage_main() -> int: return _run("audit-figure5c-lineage", audit_figure5c_lineage)
def validate_figure5c_derivative_main() -> int: return _run("validate-figure5c-derivative", validate_figure5c_derivative)
def validate_natural_drift_sign_main() -> int: return _run("validate-natural-drift-sign", validate_natural_drift_sign)
def analyse_natural_drift_uncertainty_main() -> int: return _run("analyse-natural-drift-uncertainty", analyse_natural_drift_uncertainty)
def run_directional_comparison_main() -> int: return _run("run-directional-comparison", run_directional_comparison)
def status_main() -> int: return _run("status", build_status)
def report_main() -> int: return _run("report", build_report)
