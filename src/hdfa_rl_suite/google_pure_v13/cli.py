"""Dedicated V13 command-line entry points; long work always requires an explicit flag."""
from __future__ import annotations

import argparse
import json
from typing import Any, Callable

from .diagnostics import (audit_ppo_lifecycle, report_effective_sample_size,
                          report_epoch_semantics, test_detector_logical_alignment)
from .findings import write_v12_findings_contract
from .imports import build_import_manifest, verify_import_manifest
from .io import ARTIFACT_ROOT
from .natural import analyse_natural_drift, plan_natural_drift_power, run_natural_drift
from .reporting import build_report, build_status
from .runtime import (compare_normalization_branches, run_step_validation,
                      verify_candidate_lineage, verify_state_chain)
from .scaling import (analyse_figure5c, audit_figure5b_contract,
                      audit_figure5b_convergence, run_figure5b_validation,
                      validate_figure5c_fit)
from .sensitivity import calibrate_edr_sensitivity, validate_sensitivity_map
from .step import fit_step_response


def _prepare() -> None:
    if not (ARTIFACT_ROOT / "import_manifest.json").is_file():
        build_import_manifest()
    verify_import_manifest()
    write_v12_findings_contract()


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    omitted = {"rows", "diagnostics", "conditions", "runs", "contract", "v12_findings_frozen"}
    value = {key: item for key, item in result.items() if key not in omitted}
    if len(value) != len(result):
        value["full_artifact_root"] = str(ARTIFACT_ROOT.resolve())
    return value


def _run(command: str, function: Callable[[], dict[str, Any]]) -> int:
    _prepare()
    result = function()
    print(json.dumps({"command": command, "result": _compact(result),
                      "paper_scale_auto_launch": False,
                      "output_root": str(ARTIFACT_ROOT.resolve())}, indent=2, sort_keys=True))
    return 0


def _epochs() -> int | None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int)
    return parser.parse_args().epochs


def calibrate_edr_sensitivity_main() -> int: return _run("calibrate-edr-sensitivity", calibrate_edr_sensitivity)
def validate_sensitivity_map_main() -> int: return _run("validate-sensitivity-map", validate_sensitivity_map)
def compare_normalization_branches_main() -> int:
    epochs = _epochs()
    return _run("compare-normalization-branches", lambda: compare_normalization_branches(epochs_override=epochs))
def run_step_validation_main() -> int:
    epochs = _epochs()
    return _run("run-step-validation", lambda: run_step_validation(epochs_override=epochs))
def fit_step_response_main() -> int: return _run("fit-step-response", fit_step_response)
def verify_state_chain_main() -> int: return _run("verify-state-chain", verify_state_chain)
def verify_candidate_lineage_main() -> int: return _run("verify-candidate-lineage", verify_candidate_lineage)
def audit_figure5b_contract_main() -> int: return _run("audit-figure5b-contract", audit_figure5b_contract)
def audit_figure5b_convergence_main() -> int: return _run("audit-figure5b-convergence", audit_figure5b_convergence)
def run_figure5b_validation_main() -> int:
    epochs = _epochs()
    return _run("run-figure5b-validation", lambda: run_figure5b_validation(epochs_override=epochs))
def analyse_figure5c_main() -> int: return _run("analyse-figure5c", analyse_figure5c)
def validate_figure5c_fit_main() -> int: return _run("validate-figure5c-fit", validate_figure5c_fit)
def plan_natural_drift_power_main() -> int: return _run("plan-natural-drift-power", plan_natural_drift_power)
def run_natural_drift_main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-long", action="store_true")
    parser.add_argument("--maximum-runs", type=int)
    args = parser.parse_args()
    return _run("run-natural-drift", lambda: run_natural_drift(
        execute_long=args.execute_long, maximum_runs=args.maximum_runs))
def analyse_natural_drift_main() -> int: return _run("analyse-natural-drift", analyse_natural_drift)
def test_detector_logical_alignment_main() -> int: return _run("test-detector-logical-alignment", test_detector_logical_alignment)
def report_effective_sample_size_main() -> int: return _run("report-effective-sample-size", report_effective_sample_size)
def audit_ppo_lifecycle_main() -> int: return _run("audit-ppo-lifecycle", audit_ppo_lifecycle)
def report_epoch_semantics_main() -> int: return _run("report-epoch-semantics", report_epoch_semantics)
def status_main() -> int: return _run("status", build_status)
def report_main() -> int: return _run("report", build_report)

