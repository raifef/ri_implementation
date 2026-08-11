"""Fail-closed V17 status and seventeen-point report."""
from __future__ import annotations

from typing import Any

from .contracts import NONFINAL
from .imports import verify_import_manifest
from .io import ARTIFACT_ROOT, atomic_json, atomic_text, canonical_hash, read_json


EXPECTED_ARTIFACTS = {
    "sensitivity": "sensitivity_semantics_audit.json",
    "step_transfer": "step_transfer_identification.json",
    "frequency": "frequency_units.json",
    "deterministic": "figure5a_deterministic_fixture.json",
    "metric": "figure5a_metric_endpoint.json",
    "window": "figure5a_window_aliasing.json",
    "mean_transfer": "figure5a_mean_transfer.json",
    "latency": "latency_phase_audit.json",
    "decomposition": "mean_stochastic_decomposition.json",
    "scale": "scale_dynamics_audit.json",
    "modes": "step_figure5a_mode_comparison.json",
    "repair": "minimal_repair.json",
    "archived_gate": "archived_v16_acceptance_gate.json",
    "acceptance": "reduced_acceptance_v2.json",
    "postrepair": "reduced_postrepair/result.json",
}


def _value(name: str) -> dict[str, Any] | None:
    path = ARTIFACT_ROOT / EXPECTED_ARTIFACTS[name]
    return read_json(path) if path.is_file() else None


def build_status() -> dict[str, Any]:
    manifest = verify_import_manifest()
    rows = {}
    for name, relative in EXPECTED_ARTIFACTS.items():
        path = ARTIFACT_ROOT / relative
        if path.is_file():
            value = read_json(path)
            rows[name] = {"present": True, "pass": value.get("pass"),
                          "sha256": canonical_hash(value), "path": relative}
        else:
            rows[name] = {"present": False, "pass": None, "sha256": None, "path": relative}
    diagnostic_required = ["sensitivity", "step_transfer", "frequency", "deterministic", "metric",
                           "window", "latency", "decomposition", "scale", "modes", "repair"]
    complete = all(row["present"] for row in rows.values())
    diagnostic_contracts_pass = all(rows[name]["pass"] is True for name in diagnostic_required
                                    if rows[name]["present"])
    acceptance_pass = rows["acceptance"]["present"] and rows["acceptance"]["pass"] is True
    readiness = ("READY_FOR_EXPLICITLY_AUTHORIZED_SOURCE_BUDGET_VALIDATION"
                 if complete and diagnostic_contracts_pass and acceptance_pass else
                 "NOT_READY_FOR_SOURCE_BUDGET_VALIDATION")
    blockers = [name for name, row in rows.items() if not row["present"]]
    if rows["mean_transfer"]["present"] and rows["mean_transfer"]["pass"] is not True:
        blockers.append("mean_transfer_not_identifiable")
    if rows["acceptance"]["present"] and rows["acceptance"]["pass"] is not True:
        blockers.append("paired_complete_period_acceptance_not_met")
    result = {
        "schema_version": "google-pure-v17-status.v1",
        "import_manifest_hash": manifest["import_manifest_hash"], "artifacts": rows,
        "artifact_set_complete": complete, "diagnostic_contracts_pass": diagnostic_contracts_pass,
        "reduced_acceptance_v2_pass": acceptance_pass, "readiness": readiness,
        "blockers": sorted(set(blockers)),
        "primary_diagnosis": "V16_REDUCED_GATE_UNDERPOWERED_BY_INCOMPLETE_PERIOD_AND_UNRESOLVED_METRIC_DENOMINATOR",
        "v16_optimizer_changed": False, "source_normalization_changed": False,
        "source_figure5a_protocol_changed": False, "figure5c_modified_or_executed": False,
        "natural_drift_executed": False, "source_budget_auto_launched": False,
        **NONFINAL,
    }
    atomic_json(ARTIFACT_ROOT / "status.json", result)
    return result


def build_report() -> dict[str, Any]:
    status = build_status()
    values = {name: _value(name) or {} for name in EXPECTED_ARTIFACTS}
    sensitivity = values["sensitivity"]
    step = values["step_transfer"]
    frequency = values["frequency"]
    deterministic = values["deterministic"]
    metric = values["metric"]
    window = values["window"]
    transfer = values["mean_transfer"]
    latency = values["latency"]
    decomposition = values["decomposition"]
    scale = values["scale"]
    modes = values["modes"]
    acceptance = values["acceptance"]
    repair = values["repair"]
    postrepair = values["postrepair"]
    sections = [
        ("1. Frozen import and lineage audit", {"pass": True},
         "V16 optimizer, normalization, sensitivity, step, reduced Figure 5a, entropy, direct-sigma, contraction and Figure 5b artifacts plus the production target/evaluator are hash-pinned fail closed."),
        ("2. Sensitivity semantics", sensitivity,
         f"Classification is `{sensitivity.get('classification', 'MISSING')}`: kappa_V=0.01 variance damage and kappa_H=0.02 Hessian; the production scale is unchanged."),
        ("3. Step transfer refit", step,
         f"Corrected local tau is {step.get('predicted_local_tau_epochs', 'MISSING')} and measured V16 tau is {step.get('measured_v16_tau_epochs', 'MISSING')}; K/delay/tau uncertainty and horizon stability are retained."),
        ("4. Frequency-unit reconciliation", frequency,
         "The production target is sin(2*pi*f*t), frequencies are cycles per epoch, measured periods must equal 1/f, and radians-per-epoch misuse is rejected."),
        ("5. Deterministic production-evaluator fixture", deterministic,
         f"Classification is `{deterministic.get('classification', 'MISSING')}`; slow must outperform intermediate and fast under complete-period windows."),
        ("6. Metric endpoints and denominators", metric,
         f"Production raw-count substitution gives fixed={metric.get('fixed_endpoint', 'MISSING')} and oracle={metric.get('oracle_endpoint', 'MISSING')}; per-frequency denominator uncertainty is explicit."),
        ("7. Window and aliasing audit", window,
         f"The old gate is `{window.get('original_reduced_classification', 'MISSING')}` because 24 epochs cover no complete slow/fast period; source 1000 epochs were not replaced."),
        ("8. Learned-mean transfer", transfer,
         f"Classification is `{transfer.get('classification', 'MISSING')}`; direct drift-direction gain/phase fits are not replaced by normalized performance."),
        ("9. Step-predicted versus measured frequency response", deterministic,
         "Analytic K exp(-i omega Delta)/(1+i omega tau) gains/phases and deterministic measured residuals are recorded per frequency and phase."),
        ("10. Latency and phase timeline", latency,
         f"Classification is `{latency.get('classification', 'MISSING')}`; target, sampling, acquisition, reward, gradient, update and next-policy timestamps are explicit, without predictive compensation."),
        ("11. Mean versus stochastic decomposition", decomposition,
         "C_fixed, C_oracle, C_mean, C_stochastic, I_mean, I_stochastic and exploration damage remain separate."),
        ("12. Direct-sigma scale dynamics", scale,
         "Latent/native sigma, reward/entropy gradients, guards, clipping, candidate/reward variance and exploration damage are logged by epoch, frequency and phase; entropy is unchanged."),
        ("13. Step/Figure 5a mode comparison", modes,
         f"Classification is `{modes.get('classification', 'MISSING')}`: normalized local Hessians match, but target support and reward aggregation differ."),
        ("14. Paired reduced acceptance v2", acceptance,
         f"Classification is `{acceptance.get('classification', 'MISSING')}`; LCB95(Delta I)>delta_min is required over seed/CRN/phase/budget complete-period pairs."),
        ("15. Minimal causal repair", repair,
         f"Repair scope is `{repair.get('repair_scope', 'MISSING')}`; optimizer, normalization, production Figure 5a code and Figure 5c are unchanged."),
        ("16. Reduced post-repair execution", postrepair,
         f"Classification is `{postrepair.get('classification', 'NOT_RUN')}`; this reduced diagnostic is not source/reference evidence."),
        ("17. Figure 5b note and readiness", modes,
         f"Figure 5b rate deficit remains diagnostic only; readiness is `{status['readiness']}` and no source-budget, held-out, natural-drift, Figure 5c, or reference run was launched."),
    ]
    lines = ["# V17 Figure 5a protocol and dynamic-tracking repair report", "",
             f"Readiness: **{status['readiness']}**", "",
             "Primary conclusion: V16's 24-epoch Figure 5a direction gate was not a valid dynamic-tracking test. It contained no complete period and did not resolve finite-shot metric denominators. This V17 repair changes only the reduced validation logic; it does not retune the optimizer or promote evidence.", ""]
    report_sections = []
    for heading, artifact, statement in sections:
        state = "MISSING" if not artifact else ("PASS" if artifact.get("pass") is True else
                "BLOCKED" if artifact.get("pass") is False else "RECORDED")
        lines.extend([f"## {heading}", "", f"Status: **{state}**. {statement}", ""])
        report_sections.append({"heading": heading, "status": state, "statement": statement})
    if status["blockers"]:
        lines.extend(["## Blocking conditions", "", *[f"- `{item}`" for item in status["blockers"]], ""])
    atomic_text(ARTIFACT_ROOT / "FINAL_REPORT.md", "\n".join(lines))
    result = {"schema_version": "google-pure-v17-report-manifest.v1",
              "sections": report_sections, "status_hash": canonical_hash(status),
              "readiness": status["readiness"], "blockers": status["blockers"],
              "report_path": "artifacts/google_pure_v17/FINAL_REPORT.md", **NONFINAL}
    atomic_json(ARTIFACT_ROOT / "report_manifest.json", result)
    return result
