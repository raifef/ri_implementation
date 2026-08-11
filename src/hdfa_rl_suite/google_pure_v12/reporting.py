"""Derived V12 status and non-promotional final report."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import CURRENT_CLASSIFICATIONS, NONFINAL_FIELDS, fail_closed_status
from .imports import validate_import_manifest
from .io import ARTIFACT_ROOT, atomic_json, atomic_text, read_json


REQUIRED_ARTIFACTS = {
    "directional_sensitivity": "audits/directional_sensitivity.json",
    "factor_graph_direction": "audits/factor_graph_direction.json",
    "directional_gradient": "audits/directional_gradient.json",
    "gradient_snr": "audits/gradient_snr.json",
    "update_efficiency": "audits/update_efficiency.json",
    "step_units": "audits/step_units.json",
    "spoil_units": "audits/spoil_units.json",
    "protocol_diff": "audits/figure5a_step_protocol_diff.json",
    "figure5b_lineage": "lineage/figure5b_lineage.json",
    "figure5c_lineage": "lineage/figure5c_lineage.json",
    "figure5c_derivative_fixture": "lineage/figure5c_derivative_fixture.json",
    "natural_sign": "spectral/sign_validation.json",
    "natural_uncertainty": "spectral/natural_drift_uncertainty.json",
    "directional_comparison": "directional_comparison/comparison.json",
}


def build_status() -> dict[str, Any]:
    import_check = validate_import_manifest()
    present = {name: (ARTIFACT_ROOT / relative).is_file()
               for name, relative in REQUIRED_ARTIFACTS.items()}
    classifications = dict(CURRENT_CLASSIFICATIONS)
    gates = {"immutable_imports_valid": import_check["pass"], **present}
    if present["directional_comparison"]:
        comparison = read_json(ARTIFACT_ROOT / REQUIRED_ARTIFACTS["directional_comparison"])
        if comparison.get("gates", {}).get("step_reaches_50_percent_in_all_validation_seeds") and comparison.get("gates", {}).get("step_response_time_identifiable_in_all_validation_seeds"):
            classifications["STEP_RESPONSE_INJECTED_DRIFT"] = "PARTIAL"
        if comparison.get("gates", {}).get("recovery_directional_motion_in_all_validation_seeds"):
            classifications["RANDOMIZED_RECOVERY_AFTER_SPOIL"] = "PARTIAL"
        gates["directional_comparison_scientific_gates"] = bool(comparison.get("development_validation_pass"))
    else:
        gates["directional_comparison_scientific_gates"] = False
    if present["figure5b_lineage"]:
        figure5b = read_json(ARTIFACT_ROOT / REQUIRED_ARTIFACTS["figure5b_lineage"])
        classifications["FIGURE5B_SPARSE_SCALING"] = figure5b["classification"]
        gates["figure5b_visible_trajectory"] = bool(figure5b["visibly_evolving_gate"])
    else:
        gates["figure5b_visible_trajectory"] = False
    if present["figure5c_lineage"]:
        figure5c = read_json(ARTIFACT_ROOT / REQUIRED_ARTIFACTS["figure5c_lineage"])
        classifications["FIGURE5C_CONVERGENCE_LAW"] = figure5c["classification"]
        gates["figure5c_real_derivative_identifiable"] = figure5c["identifiable_fit_condition_count"] == figure5c["condition_count"]
    else:
        gates["figure5c_real_derivative_identifiable"] = False
    if present["natural_uncertainty"]:
        natural = read_json(ARTIFACT_ROOT / REQUIRED_ARTIFACTS["natural_uncertainty"])
        gates["natural_drift_direction_identifiable"] = bool(natural["direction_identifiable"])
    else:
        gates["natural_drift_direction_identifiable"] = False
    status = fail_closed_status(gates=gates, classifications=classifications)
    status.update({"artifact_presence": present,
                   "scientific_conclusion": "DIRECTIONAL_REMEDIATION_DEVELOPMENT_ONLY_NOT_FINAL",
                   "long_paper_scale_run_authorized": False,
                   "blocking_reasons": [
                       "proprietary hardware plant and exact experimental traces are unavailable",
                       "Figure5b does not meet the trajectory visibility gate",
                       "Figure5c real trajectories do not enter the preregistered local-fit window",
                       "natural-drift run-level uncertainty does not establish the paper anchor",
                   ]})
    atomic_json(ARTIFACT_ROOT / "status.json", status)
    return status


def build_report() -> dict[str, Any]:
    status = build_status()
    report_path = ARTIFACT_ROOT / "FINAL_REPORT.md"
    lines = [
        "# Google pure-RL V12 directional-learning and lineage repair", "",
        "## Outcome", "",
        "V12 identifies a causal protocol mismatch: the slow Figure 5a path consumes detector-connected control sensitivity, while the 924-coordinate step and spoil analogues passed nominally normalized actions directly into raw-EDR curvature. The optimizer transmits its gradient correctly; the gradient reaching it is attenuated by the missing normalized/native boundary map.", "",
        "The minimal amendment applies `u = u0 + s*x` once at the plant boundary. Its reference curvature is frozen from the median Figure 5a sensitivity multiplied by detector degree. It does not change the direct-sigma policy, PPO clipping, baseline, entropy term, architecture, or controller observations.", "",
        "## Imported evidence classifications", "",
        "| Family | V12 status |", "|---|---|",
    ]
    lines.extend(f"| {family} | {classification} |" for family, classification in status["classifications"].items())
    lines.extend([
        "", "## Lineage findings", "",
        "Figure 5b raw physical/logical trajectories vary, but their floor-normalized progress remains below the frozen visibility gate. This localizes the failure to acquisition dynamics rather than merge or plotting.", "",
        "Figure 5c raw finite differences are nonzero. The imported traces never enter the preregistered local-fit window, so V12 records the slope as unidentifiable instead of substituting zero. A synthetic exponential fixture verifies the derivative implementation.", "",
        "Natural drift now uses the fixed convention `10 log10(P_learned/P_fixed)`, where negative means suppression, and resamples independent runs rather than frequency bins.", "",
        "## Evidence boundary", "",
        "All V12 outputs remain non-final public-analogue development evidence. No paper-equivalence claim is permitted, and no paper-scale run is launched automatically.", "",
    ])
    atomic_text(report_path, "\n".join(lines))
    result = {"report": str(report_path.resolve()), "status": str((ARTIFACT_ROOT / "status.json").resolve()),
              "classifications": status["classifications"], **NONFINAL_FIELDS}
    atomic_json(ARTIFACT_ROOT / "report_manifest.json", result)
    return result
