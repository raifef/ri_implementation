"""A-to-Z fault ledger with evidence-backed, terminal closure semantics."""
from __future__ import annotations

from typing import Any

from .contracts import (ALLOWED_STATUSES, HARDWARE_ONLY_UNTESTED, ISSUES,
                        RESOLVED, RULED_OUT, SOURCE_NON_IDENTIFIABLE,
                        TERMINAL_STATUSES, V15_SCHEMA, nonfinal)
from .io import ARTIFACT_ROOT, atomic_json


EVIDENCE = {
    "A": ("sensitivity/source_definition_audit.json", RESOLVED),
    "B": ("sensitivity/source_definition_audit.json", RESOLVED),
    "C": ("sensitivity/detector_degree_audit.json", RESOLVED),
    "D": ("sensitivity/multi_point_calibration.json", RESOLVED),
    "E": ("scaling/hessian_spectrum.json", RULED_OUT),
    "F": ("sensitivity/uncertainty_propagation.json", RESOLVED),
    "G": ("sensitivity/calibration_firewall.json", RESOLVED),
    "H": ("scaling/boundary_map.json", RESOLVED),
    "I": ("scaling/curvature_distribution.json", RESOLVED),
    "J": ("scaling/hessian_spectrum.json", RULED_OUT),
    "K": ("scaling/slow_mode_projection.json", RESOLVED),
    "L": ("scaling/gradient_normalization.json", RULED_OUT),
    "M": ("scaling/information_ablation.json", RESOLVED),
    "N": ("scaling/effective_sample_size.json", RESOLVED),
    "O": ("dynamics/scale_floor.json", RULED_OUT),
    "P": ("dynamics/mean_scale_conditioning.json", RESOLVED),
    "Q": ("dynamics/residual_decay.json", RESOLVED),
    "R": ("fidelity/objective_alignment.json", SOURCE_NON_IDENTIFIABLE),
    "S": ("fidelity/figure5a_latency.json", HARDWARE_ONLY_UNTESTED),
    "T": ("fidelity/step_response_fit.json", SOURCE_NON_IDENTIFIABLE),
    "U": ("fidelity/natural_drift_power.json", RESOLVED),
    "V": ("fidelity/ppo_lifecycle.json", RESOLVED),
    "W": ("fidelity/provenance.json", RESOLVED),
    "X": ("decoder/offline_steering.json", SOURCE_NON_IDENTIFIABLE),
    "Y": ("gate/reference_gate_status.json", RESOLVED),
    "Z": ("source_gap_register.json", HARDWARE_ONLY_UNTESTED),
}


def build_fault_ledger() -> dict[str, Any]:
    if set(EVIDENCE) != set(ISSUES):
        raise RuntimeError("V15 ledger must contain exactly issues A through Z")
    rows = []
    for issue in sorted(ISSUES):
        relative, status = EVIDENCE[issue]
        if status not in ALLOWED_STATUSES:
            raise RuntimeError(f"invalid status for {issue}: {status}")
        path = ARTIFACT_ROOT / relative
        rows.append({
            "issue": issue,
            "title": ISSUES[issue],
            "status": status,
            "terminal": status in TERMINAL_STATUSES,
            "evidence_path": relative,
            "evidence_present": path.is_file(),
        })
    result = nonfinal({
        "issue_count": len(rows),
        "issues": rows,
        "all_terminal": all(row["terminal"] for row in rows),
        "all_evidence_present": all(row["evidence_present"] for row in rows),
        "closure_complete": all(row["terminal"] and row["evidence_present"] for row in rows),
        "closure_does_not_imply_reference_gate_pass": True,
    })
    atomic_json(ARTIFACT_ROOT / "fault_ledger.json", result)
    return result
