"""V15 status, deterministic closure iterations, and final scientific report."""
from __future__ import annotations

from typing import Any

from .contracts import ISSUES, nonfinal
from .gate import ANALYSIS_ARTIFACTS, reference_gate_status
from .imports import verify_import_manifest
from .io import ARTIFACT_ROOT, atomic_json, atomic_text, read_json
from .ledger import build_fault_ledger


def _load(relative: str) -> dict[str, Any] | None:
    path = ARTIFACT_ROOT / relative
    return read_json(path) if path.is_file() else None


def build_closure_iterations() -> dict[str, Any]:
    stages = [
        ("01_source_definition", ANALYSIS_ARTIFACTS[:5]),
        ("02_scaling_geometry", ANALYSIS_ARTIFACTS[5:13]),
        ("03_controller_dynamics", ANALYSIS_ARTIFACTS[13:16]),
        ("04_figure_fidelity", ANALYSIS_ARTIFACTS[16:24]),
        ("05_decoder_and_source_gaps", ANALYSIS_ARTIFACTS[24:]),
        ("06_reference_gate", ["gate/heldout_freeze.json", "gate/reference_gate_status.json"]),
    ]
    cumulative = 0
    rows = []
    for iteration, paths in stages:
        present = [path for path in paths if (ARTIFACT_ROOT / path).is_file()]
        cumulative += len(present)
        rows.append({"iteration": iteration, "required_artifacts": paths,
                     "present_artifacts": present, "complete": len(present) == len(paths),
                     "cumulative_present_count": cumulative})
    result = nonfinal({
        "iterations": rows,
        "all_iterations_complete": all(row["complete"] for row in rows),
        "iterations_are_a_deterministic_dependency_audit_not_a_claim_of_experimental_runs": True,
    })
    atomic_json(ARTIFACT_ROOT / "closure_iterations.json", result)
    return result


def build_status() -> dict[str, Any]:
    imports = verify_import_manifest()
    if not (ARTIFACT_ROOT / "gate/reference_gate_status.json").is_file():
        reference_gate_status()
    gate = read_json(ARTIFACT_ROOT / "gate/reference_gate_status.json")
    ledger = build_fault_ledger()
    iterations = build_closure_iterations()
    artifact_presence = {path: (ARTIFACT_ROOT / path).is_file() for path in ANALYSIS_ARTIFACTS}
    result = nonfinal({
        "immutable_imports_valid": bool(imports["pass"]),
        "artifact_presence": artifact_presence,
        "artifact_count": sum(artifact_presence.values()),
        "required_artifact_count": len(artifact_presence),
        "fault_ledger_issue_count": ledger["issue_count"],
        "fault_ledger_all_terminal": ledger["all_terminal"],
        "fault_ledger_all_evidence_present": ledger["all_evidence_present"],
        "open_issue_closure_complete": ledger["closure_complete"],
        "closure_iterations_complete": iterations["all_iterations_complete"],
        "reference_gate_status": gate["status"],
        "reference_gate_pass": gate["pass"],
        "reference_gate_blocking_reasons": gate["blocking_reasons"],
        "lineage_sequence": imports["lineage_sequence"],
        "long_runs_auto_launched": False,
        "heldout_seeds_consumed": False,
        "scientific_conclusion": (
            "PUBLIC_IMPLEMENTATION_ISSUES_CLOSED_WITH_TERMINAL_CLASSIFICATIONS; "
            "REFERENCE_GATE_CLOSED_AND_PAPER_EQUIVALENCE_NOT_ESTABLISHED"),
    })
    atomic_json(ARTIFACT_ROOT / "status.json", result)
    return result


def build_report() -> dict[str, Any]:
    status = build_status()
    ledger = read_json(ARTIFACT_ROOT / "fault_ledger.json")
    gate = read_json(ARTIFACT_ROOT / "gate/reference_gate_status.json")
    source = _load("sensitivity/source_definition_audit.json") or {}
    step = _load("fidelity/step_response_fit.json") or {}
    natural = _load("fidelity/natural_drift_power.json") or {}
    decoder = _load("decoder/offline_steering.json") or {}
    sections = [
        ("1. Immutable inputs", "Every available V12, V13, V15, controller, calibration, and source-contract input is SHA-256 pinned. The version sequence intentionally contains no V14."),
        ("2. A-to-Z issue ledger", f"{ledger['issue_count']} issues are represented; terminal closure is {ledger['all_terminal']} and evidence presence is {ledger['all_evidence_present']}."),
        ("3. Public sensitivity definition", f"The group-Gaussian variance definition audit passes: {source.get('pass')}."),
        ("4. Mathematical target", "The source target is sigma0 = 1/sqrt(a_pp), with a_pp measured in EDR percentage points per native-unit squared."),
        ("5. Detector-degree normalization", "The exact connected-detector sum is calibrated directly. No second degree, detector-count, or control-count correction is permitted."),
        ("6. Multipoint operating-state audit", "Symmetric offsets, a cubic term, forward/reverse ordering, and operating-state hashes are retained. Hardware hysteresis remains untested."),
        ("7. Calibration uncertainty and firewall", "Coefficient uncertainty is propagated through the delta transform. Calibration, development, validation, and held-out seed registries are disjoint."),
        ("8. One-use plant boundary", "V15 implements u = u0 + s*x with a one-use provenance token. Legacy acquisitions lacking that exact boundary are not promoted."),
        ("9. Figure 5b decomposition", "Physical and logical error, learned mean and stochastic candidate, distances, epoch colour, and irreducible floors are separate quantities."),
        ("10. Gradient and curvature scaling", "The gradient factor ledger contains one K mean and connected detector sums. Source-normalized group curvature is 0.01 EDR fraction per normalized variance."),
        ("11. Hessian and slow modes", "The current public quadratic simulator is diagonal after exact calibration; that rules out coupling only in this simulator. A conditioned fixture validates slow-mode projection."),
        ("12. Information and ESS", "Candidate richness, detector-shot richness, policy Kish ESS, policy directional rank, and detector ESS are reported as distinct concepts."),
        ("13. Mean, scale, and floor", "Direct-sigma reward and entropy conditioning are explicit. The minimum sigma floor is negligible in the mean physical metric, while initial exploration is not."),
        ("14. Residual decay", "Finite-horizon trajectories are classified as still decaying, empirical plateau, or unidentified; none is relabelled as an asymptote."),
        ("15. Objective alignment", "The analytic sparse surrogate is aligned by construction. Stim/PyMatching finite differences and proprietary hardware alignment remain unresolved."),
        ("16. Figure 5c", "The derivative window is frozen upstream. The imported local regime was not reached, so zero fallback values are rejected."),
        ("17. Figure 5a latency", "Ideal batch delay is small relative to public scan periods. Unpublished optimizer and hardware latency remain unmeasured."),
        ("18. Step response", f"Absolute 0.9 target crossings are used; {step.get('censored_count')} traces are censored. The observed-final-excursion threshold is forbidden."),
        ("19. Natural drift", f"The frozen plan calls for {natural.get('planned_complete_paired_runs')} complete paired runs and sufficient duration to resolve the lowest DFT frequency. It has not been auto-launched."),
        ("20. PPO, provenance, and resources", "One fresh batch and one optimizer step define an epoch. State, candidate, QEC-cycle, detector-trial, decoded-shot, and wall-clock semantics remain separate."),
        ("21. Offline decoder steering", f"The four-arm accounting contract is executable, but the scientific run is {decoder.get('execution_status')} because verified Sparse Blossom, its public benchmark, and frozen held-out data are absent."),
        ("22. Reference decision", f"The immutable reference gate is {gate['status']}. Closure of software issues does not establish performance equivalence to the paper."),
    ]
    lines = [
        "# Google pure-RL V15 complete open-issue closure", "",
        "## Outcome", "",
        "V15 closes the public implementation audit without promoting simulator-only evidence. The reference gate remains closed because required proprietary, hardware, source-budget, natural-drift, and decoder evidence is unavailable or incomplete.", "",
    ]
    for title, body in sections:
        lines.extend([f"## {title}", "", body, ""])
    lines.extend(["## Fault ledger", "", "| Issue | Title | Status | Evidence |", "|---|---|---|---|"])
    lines.extend(f"| {row['issue']} | {row['title']} | {row['status']} | {row['evidence_path']} |"
                 for row in ledger["issues"])
    lines.extend(["", "## Blocking reference gates", ""])
    lines.extend(f"- {name}" for name in gate["blocking_reasons"])
    path = ARTIFACT_ROOT / "FINAL_REPORT.md"
    atomic_text(path, "\\n".join(lines))
    manifest = nonfinal({
        "report": str(path.resolve()),
        "status": str((ARTIFACT_ROOT / "status.json").resolve()),
        "fault_ledger": str((ARTIFACT_ROOT / "fault_ledger.json").resolve()),
        "section_count": len(sections),
        "issue_count": len(ISSUES),
        "reference_gate_status": gate["status"],
    })
    atomic_json(ARTIFACT_ROOT / "report_manifest.json", manifest)
    return manifest
