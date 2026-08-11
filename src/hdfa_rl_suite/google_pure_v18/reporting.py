"""Truthful V18 status and compact scientific handoff."""
from __future__ import annotations

from typing import Any

from .contracts import nonfinal
from .experiments import (
    build_figure5b_learning_rate_note, build_mean_stochastic_decomposition,
    build_paired_acceptance_readiness, build_sensitivity_field_cleanup,
    build_steady_state_rule, validate_deterministic_transfer,
)
from .imports import verify_import_manifest
from .io import ARTIFACT_ROOT, ROOT, atomic_json, atomic_text, file_hash, read_json


def _optional(name: str) -> dict[str, Any] | None:
    path = ARTIFACT_ROOT / f"{name}.json"
    return read_json(path) if path.is_file() else None


def build_status() -> dict[str, Any]:
    imports = verify_import_manifest()
    sensitivity = build_sensitivity_field_cleanup()
    deterministic = validate_deterministic_transfer()
    steady = build_steady_state_rule()
    figure5b = build_figure5b_learning_rate_note()
    intermediate = _optional("transfer_intermediate")
    fast = _optional("transfer_fast")
    slow = _optional("transfer_slow")
    if intermediate or fast or slow:
        decomposition = build_mean_stochastic_decomposition()
    else:
        decomposition = None
    readiness = build_paired_acceptance_readiness()
    gates = {
        "imports_frozen": imports["all_imports_valid"],
        "sensitivity_terms_explicit": sensitivity["pass"],
        "deterministic_transfer_quantitative": deterministic["pass"],
        "steady_state_rule_preregistered": steady["pass"],
        "intermediate_direct_mean_transfer_identifiable": bool(
            intermediate and intermediate.get("direct_mean_transfer_identifiable")),
        "fast_direct_mean_transfer_identifiable": bool(
            fast and fast.get("direct_mean_transfer_identifiable")),
        "intermediate_steady_periodic_identification_accepted": bool(
            intermediate and intermediate.get("steady_periodic_identification_accepted")),
        "fast_steady_periodic_identification_accepted": bool(
            fast and fast.get("steady_periodic_identification_accepted")),
        "intermediate_fast_mechanistic_ordering": bool(
            fast and fast.get("stage_ab_ordering", {}).get("pass")),
        "stream_decomposition_available": bool(decomposition and decomposition.get("pass")),
        "figure5b_semantics_resolved": figure5b["pass"],
    }
    mechanistic_complete = all(gates.values())
    result = nonfinal({
        "pass": mechanistic_complete,
        "implementation_complete": True,
        "mechanistic_identification_complete": mechanistic_complete,
        "classification": ("V18_QUICK_IDENTIFICATION_COMPLETE_NONFINAL" if mechanistic_complete else
                           "V18_QUICK_IDENTIFICATION_INCOMPLETE_OR_GATE_FAILED"),
        "gates": gates,
        "paired_acceptance_ready": readiness["ready_for_paired_acceptance_claim"],
        "paired_acceptance_classification": readiness["classification"],
        "optional_slow_identification_present": slow is not None,
        "optional_slow_identification_pass": bool(slow and slow.get("pass")),
        "optimizer_changed": False,
        "controller_hash": (intermediate or {}).get("controller_hash"),
        "forbidden_runs_auto_launched": [],
        "forbidden_auto_runs": [
            "figure5c", "natural_drift", "heldout", "reference", "source_budget",
            "full_four_phase_slow_fast_acceptance",
        ],
        "artifact_root": str(ARTIFACT_ROOT.resolve()),
    })
    atomic_json(ARTIFACT_ROOT / "status.json", result)
    return result


def build_report() -> dict[str, Any]:
    status = build_status()
    names = [
        "import_manifest", "sensitivity_field_cleanup",
        "deterministic_fixture_quantitative_validation", "delta_min_provenance",
        "steady_state_rule", "transfer_intermediate", "transfer_fast", "transfer_slow",
        "mean_stochastic_decomposition", "paired_acceptance_readiness",
        "figure5b_learning_rate_note", "status",
    ]
    inventory = []
    for name in names:
        path = ARTIFACT_ROOT / f"{name}.json"
        inventory.append({
            "name": name, "present": path.is_file(),
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": file_hash(path) if path.is_file() else None,
        })
    intermediate = _optional("transfer_intermediate")
    fast = _optional("transfer_fast")
    readiness = _optional("paired_acceptance_readiness")
    result = nonfinal({
        "pass": status["pass"], "status": status, "artifact_inventory": inventory,
        "headline": {
            "intermediate": (None if intermediate is None else {
                "direct_mean_transfer_identifiable": intermediate["direct_mean_transfer_identifiable"],
                "steady_periodic_identification_accepted":
                    intermediate["steady_periodic_identification_accepted"],
                "gain": intermediate["mean_transfer_regression"]["gain"],
                "phase_lag_radians": intermediate["mean_transfer_regression"]["phase_lag_radians"],
                "I_mean": intermediate["stream_decomposition"]["I_mean"],
                "I_stochastic": intermediate["stream_decomposition"]["I_stochastic"],
            }),
            "fast": (None if fast is None else {
                "direct_mean_transfer_identifiable": fast["direct_mean_transfer_identifiable"],
                "steady_periodic_identification_accepted":
                    fast["steady_periodic_identification_accepted"],
                "gain": fast["mean_transfer_regression"]["gain"],
                "phase_lag_radians": fast["mean_transfer_regression"]["phase_lag_radians"],
                "I_mean": fast["stream_decomposition"]["I_mean"],
                "I_stochastic": fast["stream_decomposition"]["I_stochastic"],
                "ordering": fast.get("stage_ab_ordering"),
            }),
        },
        "acceptance_readiness": readiness,
        "claim_boundary": (
            "V18 is a development-only mechanistic identification campaign. It does not establish "
            "paper equivalence or paired slow-versus-fast acceptance."),
        "automatic_campaigns_not_run": [
            "Figure 5c", "natural drift", "held-out", "reference", "source budget",
            "full slow/four-phase statistical acceptance",
        ],
    })
    atomic_json(ARTIFACT_ROOT / "report.json", result)
    ordering = ((fast or {}).get("stage_ab_ordering") or {})
    lines = [
        "# V18 quick Figure 5a identification report", "",
        f"Classification: **{status['classification']}**.", "",
        "## Mechanistic gates", "",
        *[f"- {name}: **{'PASS' if value else 'FAIL/ABSENT'}**"
          for name, value in status["gates"].items()], "",
        "## Transfer result", "",
        ("- Intermediate: not acquired." if intermediate is None else
         f"- Intermediate: gain={intermediate['mean_transfer_regression']['gain']:.4f}, "
         f"phase lag={intermediate['mean_transfer_regression']['phase_lag_radians']:.4f} rad, "
         f"direct-transfer-identifiable={intermediate['direct_mean_transfer_identifiable']}, "
         f"steady-periodic-accepted={intermediate['steady_periodic_identification_accepted']}."),
        ("- Fast: not acquired." if fast is None else
         f"- Fast: gain={fast['mean_transfer_regression']['gain']:.4f}, "
         f"phase lag={fast['mean_transfer_regression']['phase_lag_radians']:.4f} rad, "
         f"direct-transfer-identifiable={fast['direct_mean_transfer_identifiable']}, "
         f"steady-periodic-accepted={fast['steady_periodic_identification_accepted']}."),
        ("- Intermediate/fast ordering: unavailable." if not ordering else
         f"- Intermediate/fast joint ordering probability={ordering['bootstrap_joint_probability']:.3f}; "
         f"pass={ordering['pass']}."), "",
        "## Evidence boundary", "",
        f"- Paired acceptance: **{readiness['classification'] if readiness else 'NOT ASSESSED'}**.",
        "- This is development-only, non-final evidence and does not support a paper-equivalence claim.",
        "- No Figure 5c, natural-drift, held-out, reference, source-budget, or full four-phase campaign was launched.",
    ]
    atomic_text(ARTIFACT_ROOT / "report.md", "\n".join(lines))
    return result
