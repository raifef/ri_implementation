"""Command-line entry points for the bounded V20 workflow."""
from __future__ import annotations

import json
from typing import Any, Callable

from .analysis import audit_transfer_geometry, decompose_fast_mean_cost
from .data import build_import_manifest
from .gradient_diagnostics import (
    audit_acquisition_bias,
    audit_dynamic_sigma,
    audit_fast_gradient_statistics,
    compute_reference_gradients,
    run_candidate_shot_factorial,
    run_fixed_budget_comparison,
    run_scale_information_frontier,
)
from .io import ARTIFACT_ROOT
from .population import classify_root_cause, run_population_gradient_fast
from .repair import run_minimal_repair_validation
from .reporting import build_report, status


def _run(function: Callable[[], dict[str, Any]]) -> int:
    result = function()
    compact = {
        "pass": result.get("pass"),
        "classification": result.get("classification") or
            result.get("primary_classification") or result.get("primary_root_cause"),
        "execution_complete": result.get("execution_complete", True),
        "output_root": str(ARTIFACT_ROOT.resolve()),
        "paper_equivalence_claim_permitted": False,
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0 if result.get("pass") is True else 2


def decompose_main() -> int: return _run(decompose_fast_mean_cost)
def geometry_main() -> int: return _run(audit_transfer_geometry)
def gradients_main() -> int: return _run(audit_fast_gradient_statistics)
def reference_main() -> int: return _run(compute_reference_gradients)
def factorial_main() -> int: return _run(run_candidate_shot_factorial)
def fixed_budget_main() -> int: return _run(run_fixed_budget_comparison)
def scale_main() -> int: return _run(run_scale_information_frontier)
def sigma_main() -> int: return _run(audit_dynamic_sigma)
def acquisition_main() -> int: return _run(audit_acquisition_bias)
def population_main() -> int: return _run(run_population_gradient_fast)
def root_cause_main() -> int: return _run(classify_root_cause)
def repair_main() -> int: return _run(run_minimal_repair_validation)
def status_main() -> int: return _run(status)
def report_main() -> int: return _run(build_report)


def run_all() -> dict[str, Any]:
    build_import_manifest()
    decompose_fast_mean_cost()
    audit_transfer_geometry()
    audit_fast_gradient_statistics()
    compute_reference_gradients()
    run_candidate_shot_factorial()
    run_fixed_budget_comparison()
    run_scale_information_frontier()
    audit_dynamic_sigma()
    audit_acquisition_bias()
    run_population_gradient_fast()
    classify_root_cause()
    run_minimal_repair_validation()
    return build_report()


def run_all_main() -> int: return _run(run_all)
