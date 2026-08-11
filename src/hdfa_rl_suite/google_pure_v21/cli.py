"""CLI entry points for V21."""
from __future__ import annotations

import json
from typing import Any, Callable

from .benchmark import (
    audit_frame_coverage,
    benchmark_candidate_designs,
    reconcile_exploration_damage,
    run_design_scale_pareto,
    run_frozen_promotion_gate,
)
from .conclusion import build_report, classify, decide_minimal_repair, status
from .diagnostics import (
    audit_projection_reference_retention,
    classify_candidate_source_fidelity,
    decompose_gradient_variance,
    document_candidate_estimators,
)
from .io import ARTIFACT_ROOT
from .lineage import build_import_manifest
from .online import run_generalization_audit, run_short_fast_rollouts


def _run(function: Callable[[], dict[str, Any]]) -> int:
    result = function()
    print(json.dumps({
        "pass": result.get("pass"),
        "classification": result.get("classification") or
            result.get("primary_classification"),
        "repair_adopted": result.get("repair_adopted"),
        "output_root": str(ARTIFACT_ROOT.resolve()),
        "paper_equivalence_claim_permitted": False,
    }, indent=2, sort_keys=True))
    return 0 if result.get("pass") is True else 2


def projection_main() -> int: return _run(audit_projection_reference_retention)
def variance_main() -> int: return _run(decompose_gradient_variance)
def fidelity_main() -> int: return _run(classify_candidate_source_fidelity)


def benchmark_main() -> int:
    document_candidate_estimators()
    return _run(benchmark_candidate_designs)


def coverage_main() -> int: return _run(audit_frame_coverage)
def damage_main() -> int: return _run(reconcile_exploration_damage)
def pareto_main() -> int: return _run(run_design_scale_pareto)
def promotion_main() -> int: return _run(run_frozen_promotion_gate)
def online_main() -> int: return _run(run_short_fast_rollouts)
def generalization_main() -> int: return _run(run_generalization_audit)
def classify_main() -> int: return _run(classify)
def status_main() -> int: return _run(status)
def report_main() -> int: return _run(build_report)


def run_all() -> dict[str, Any]:
    build_import_manifest()
    audit_projection_reference_retention()
    decompose_gradient_variance()
    classify_candidate_source_fidelity()
    document_candidate_estimators()
    audit_frame_coverage()
    benchmark_candidate_designs()
    reconcile_exploration_damage()
    run_design_scale_pareto()
    run_frozen_promotion_gate()
    run_short_fast_rollouts()
    run_generalization_audit()
    classify()
    decide_minimal_repair()
    return build_report()


def run_all_main() -> int: return _run(run_all)
