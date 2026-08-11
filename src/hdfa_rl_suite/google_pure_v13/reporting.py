"""Fail-closed V13 status and scientific report."""
from __future__ import annotations

from typing import Any

from .contracts import NONFINAL, V12_FINDINGS, V13_SCHEMA
from .imports import verify_import_manifest
from .io import ARTIFACT_ROOT, atomic_json, atomic_text, read_json


ARTIFACTS = {
    "sensitivity_validation": "sensitivity_calibration/validation.json",
    "normalization_comparison": "sensitivity_calibration/comparison.json",
    "step_runs": "step_validation/runs.json",
    "step_fit": "step_validation/fit.json",
    "state_chain": "provenance/state_chain_validation.json",
    "candidate_lineage": "provenance/candidate_lineage_validation.json",
    "figure5b_contract": "figure5b/source_contract.json",
    "figure5b_validation": "figure5b/validation.json",
    "figure5b_convergence": "figure5b/convergence_audit.json",
    "figure5c_analysis": "figure5c/analysis.json",
    "figure5c_fixture": "figure5c/fit_fixture.json",
    "natural_power": "natural_drift/power_plan.json",
    "natural_analysis": "natural_drift/analysis.json",
    "detector_logical_alignment": "diagnostics/detector_logical_alignment.json",
    "effective_sample_size": "diagnostics/effective_sample_size.json",
    "ppo_lifecycle": "diagnostics/ppo_lifecycle.json",
    "epoch_semantics": "diagnostics/epoch_semantics.json",
}


def _load(name: str) -> dict[str, Any] | None:
    path = ARTIFACT_ROOT / ARTIFACTS[name]
    return read_json(path) if path.is_file() else None


def build_status() -> dict[str, Any]:
    imports = verify_import_manifest()
    presence = {name: (ARTIFACT_ROOT / path).is_file() for name, path in ARTIFACTS.items()}
    sensitivity = _load("sensitivity_validation")
    step = _load("step_fit")
    state = _load("state_chain")
    candidate = _load("candidate_lineage")
    figure5b = _load("figure5b_validation")
    figure5c = _load("figure5c_analysis")
    natural = _load("natural_analysis")
    alignment = _load("detector_logical_alignment")
    ppo = _load("ppo_lifecycle")
    gates = {
        "immutable_imports_valid": bool(imports["pass"]),
        "independent_sensitivity_calibration_valid": bool(sensitivity and sensitivity.get("pass")),
        "step_response_fit_valid": bool(step and step.get("fit_valid")),
        "policy_state_chain_complete": bool(state and state.get("pass")),
        "candidate_reward_lineage_complete": bool(candidate and candidate.get("pass")),
        "figure5b_acquisition_valid": bool(figure5b and figure5b.get("acquisition_valid")),
        "figure5c_fit_valid": bool(figure5c and figure5c.get("fit_valid")),
        "natural_drift_power_plan_complete": bool(natural and natural.get("power_plan_complete")),
        "natural_drift_direction_identifiable": bool(natural and natural.get("direction_identifiable")),
        "detector_logical_alignment_identified": bool(alignment and alignment.get("pass")),
        "ppo_lifecycle_valid": bool(ppo and ppo.get("pass")),
        "proprietary_google_plant_available": False,
        "experimental_source_traces_available": False,
    }
    classifications = {
        "SENSITIVITY_BOUNDARY": "SOURCE_LITERAL_KAPPA_PUBLIC_ANALOGUE_CALIBRATED" if gates["independent_sensitivity_calibration_valid"] else "UNRESOLVED",
        "STEP_RESPONSE": "SOURCE_NORMALIZED_SYNTHETIC_VALIDATION" if gates["step_response_fit_valid"] else "PENDING_OR_INVALID",
        "RANDOMIZED_RECOVERY": "SOURCE_NORMALIZED_SYNTHETIC_VALIDATION" if presence["normalization_comparison"] else "PENDING",
        "FIGURE5B": "NORMALIZED_SYNTHETIC_CONVERGENCE_DIAGNOSTIC" if presence["figure5b_validation"] else "PENDING",
        "FIGURE5C": figure5c.get("classification", "PENDING") if figure5c else "PENDING",
        "NATURAL_DRIFT": natural.get("classification", "PENDING") if natural else "PENDING",
        "DETECTOR_LOGICAL_ALIGNMENT": alignment.get("classification", "PENDING") if alignment else "PENDING",
        "FINAL_GOOGLE_BASELINE": "NOT_READY",
    }
    blocking = [name for name, passed in gates.items() if not passed]
    status = {"schema_version": V13_SCHEMA, "v12_findings_frozen": V12_FINDINGS,
              "artifact_presence": presence, "gates": gates, "classifications": classifications,
              "all_scientific_gates_pass": all(gates.values()),
              "blocking_reasons": blocking,
              "long_runs_auto_launched": False,
              "scientific_conclusion": "V13_DEVELOPMENT_VALIDATION_ONLY_NOT_FINAL",
              **NONFINAL}
    atomic_json(ARTIFACT_ROOT / "status.json", status)
    return status


def build_report() -> dict[str, Any]:
    status = build_status()
    path = ARTIFACT_ROOT / "FINAL_REPORT.md"
    lines = [
        "# Google pure-RL V13 source-faithful sensitivity and scaling repair", "",
        "## Outcome", "",
        "V13 replaces the V12 outcome-derived directional curvature with the source-literal normalization law: one normalized variance unit corresponds to one EDR percentage point. Independent symmetric public-analogue calibration supplies the native scale for each coordinate, and a typed plant boundary applies `u = u0 + s*x` exactly once.", "",
        "State continuity and candidate lineage are recorded from sampling through detector rewards and the optimizer. Figure 5b retains physical and logical error trajectories on the required panel quantities; Figure 5c is downstream and remains unidentifiable unless acquisition enters its frozen local regime.", "",
        "Natural-drift uncertainty uses complete paired runs. The pilot power calculation calls for 48 runs; the long acquisition is explicitly gated and is not launched by reporting or status commands.", "",
        "## Current classifications", "", "| Item | Classification |", "|---|---|",
    ]
    lines.extend(f"| {name} | {value} |" for name, value in status["classifications"].items())
    lines.extend(["", "## Blocking gates", ""])
    lines.extend(f"- `{name}`" for name in status["blocking_reasons"])
    lines.extend(["", "## Evidence boundary", "",
                  "The proprietary plant and experimental traces are unavailable. All generated results remain public-analogue development evidence; paper equivalence and downstream comparison are not permitted."])
    atomic_text(path, "\n".join(lines))
    result = {"schema_version": V13_SCHEMA, "report": str(path.resolve()),
              "status": str((ARTIFACT_ROOT / "status.json").resolve()),
              "classifications": status["classifications"], **NONFINAL}
    atomic_json(ARTIFACT_ROOT / "report_manifest.json", result)
    return result

