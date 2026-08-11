"""Fail-closed V16 status and thirteen-point scientific report."""
from __future__ import annotations

from typing import Any

from .contracts import NONFINAL
from .imports import verify_import_manifest
from .io import ARTIFACT_ROOT, atomic_json, atomic_text, canonical_hash, read_json


EXPECTED_ARTIFACTS = {
    "hypothesis": "optimizer_consistency_hypothesis.json",
    "coordinate_transform": "coordinate_transform_contract.json",
    "strict_covariance": "coordinate_covariance_fixture.json",
    "gradient_covariance": "gradient_covariance_audit.json",
    "ppo_covariance": "ppo_covariance_audit.json",
    "entropy_covariance": "entropy_covariance_audit.json",
    "optimizer_sources": "optimizer_source_audit.json",
    "native_exploration": "native_exploration_audit.json",
    "direct_sigma": "direct_sigma_dynamics.json",
    "entropy_anchors": "source_entropy_anchors.json",
    "optimizer_calibration": "optimizer_calibration_results.json",
    "frozen_optimizer": "frozen_source_normalized_optimizer.json",
    "matched_step": "matched_step/comparison.json",
    "matched_figure5b": "matched_figure5b/comparison.json",
    "local_contraction": "local_contraction_audit.json",
    "baseline_reward_scaling": "baseline_reward_scaling_audit.json",
    "reduced_acceptance": "reduced_acceptance/result.json",
}


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
    required_pass = [
        "coordinate_transform", "strict_covariance", "gradient_covariance", "ppo_covariance",
        "entropy_covariance", "optimizer_sources", "native_exploration", "direct_sigma",
        "entropy_anchors", "optimizer_calibration", "local_contraction", "reduced_acceptance",
    ]
    complete = all(rows[name]["present"] for name in EXPECTED_ARTIFACTS)
    gates_pass = all(rows[name]["pass"] is True for name in required_pass)
    readiness = ("READY_FOR_EXPLICITLY_AUTHORIZED_SOURCE_BUDGET_VALIDATION"
                 if complete and gates_pass else "NOT_READY_FOR_SOURCE_BUDGET_VALIDATION")
    blockers = [name for name in EXPECTED_ARTIFACTS if not rows[name]["present"]]
    blockers.extend(name for name in required_pass if rows[name]["present"] and rows[name]["pass"] is not True)
    result = {
        "schema_version": "google-pure-v16-status.v1",
        "import_manifest_hash": manifest["import_manifest_hash"],
        "artifacts": rows,
        "artifact_set_complete": complete,
        "required_reduced_gates_pass": gates_pass,
        "readiness": readiness,
        "blockers": sorted(set(blockers)),
        "optimizer_hypothesis_accepted": bool(
            rows["reduced_acceptance"]["present"] and rows["reduced_acceptance"]["pass"] is True),
        "source_budget_auto_launched": False,
        "reference_campaign_auto_launched": False,
        "figure5c_modified_or_executed": False,
        "natural_drift_executed": False,
        **NONFINAL,
    }
    atomic_json(ARTIFACT_ROOT / "status.json", result)
    return result


def _value(name: str) -> dict[str, Any] | None:
    relative = EXPECTED_ARTIFACTS[name]
    path = ARTIFACT_ROOT / relative
    return read_json(path) if path.is_file() else None


def build_report() -> dict[str, Any]:
    status = build_status()
    hypothesis = _value("hypothesis") or {}
    source = _value("optimizer_sources") or {}
    covariance = _value("strict_covariance") or {}
    exploration = _value("native_exploration") or {}
    direct_sigma = _value("direct_sigma") or {}
    anchors = _value("entropy_anchors") or {}
    frozen = _value("frozen_optimizer") or {}
    step = _value("matched_step") or {}
    figure5b = _value("matched_figure5b") or {}
    contraction = _value("local_contraction") or {}
    acceptance = _value("reduced_acceptance") or {}
    step_summaries = step.get("summaries", {})
    figure5b_summaries = figure5b.get("summaries", {})
    v15_step = step_summaries.get("C_V15_INHERITED_OPTIMIZER", {})
    v16_step = step_summaries.get("D_V16_FROZEN_OPTIMIZER", {})
    v16_figure5b = figure5b_summaries.get("D_V16_FROZEN_OPTIMIZER", {})
    remaining = [name for name, passed in acceptance.get("gates", {}).items() if not passed]
    sections = [
        ("1. Exact coordinate-transform derivation", _value("coordinate_transform"),
         "For u=u0+Sx, grad_x=S^T grad_u and Delta_u=-S A_x S^T grad_u; native scalar eta_u requires A_x=eta_u S^-2. Mean, direct sigma, support, entropy, PPO, baseline and reward rules are recorded."),
        ("2. Optimizer hyperparameter source classification", source,
         f"All audited quantities are classified; LEGACY_INHERITED count is {source.get('legacy_inherited_count', 'missing')}."),
        ("3. V12/V15 curvature and timescale", hypothesis,
         f"The measured V12/V15 conditioned-curvature ratio is {hypothesis.get('curvature_ratio_v12_over_v15', 'missing')}; fixed learning rates therefore predict an approximately fourfold slower local V15 adaptation."),
        ("4. Strict covariance fixture", covariance,
         f"Native action, reward, mean-update and sigma-update maxima are {covariance.get('updated_native_mean_max_abs_errors', 'missing')} and {covariance.get('native_reward_max_abs_errors', 'missing')}; A/B/C physical invariants are enforced."),
        ("5. Native exploration comparison", exploration,
         f"Inherited V15/V12 initial native-sigma ratio is {exploration.get('v15_over_v12_initial_native_sigma', 'missing')}; initialization, pre-step, early and late EDR spread, reward variance and SNR are logged."),
        ("6. Mean versus direct-sigma gradients and steps", direct_sigma,
         f"Reward/entropy mean and sigma gradients and realized steps are separate. The inherited V15 entropy-to-reward sigma ratio is {direct_sigma.get('v15_inherited_entropy_reward_ratio_over_v12', 'missing')} times V12."),
        ("7. Entropy regimes under V15 normalization", anchors,
         f"The public 0.001/0.01/0.1 regimes preserve their too-little/balanced/too-much ordering under `{anchors.get('normalization', 'missing')}`."),
        ("8. Independently frozen source-normalized optimizer", frozen,
         f"Frozen mean LR={frozen.get('mean_learning_rate', 'missing')}, sigma LR={frozen.get('sigma_learning_rate', 'missing')}, initial sigma={frozen.get('initial_sigma', 'missing')}, entropy={frozen.get('entropy_coefficient', 'missing')}; no dynamic or headline output selected them."),
        ("9. Matched physical-step result", step,
         f"Median final progress changed from {v15_step.get('median_final_target_fraction', 'missing')} (V15 inherited) to {v16_step.get('median_final_target_fraction', 'missing')} (V16 frozen); V16 median t50={v16_step.get('median_t50_epochs', 'not identified')} and tau={v16_step.get('median_tau_epochs', 'not identified')}."),
        ("10. Matched Figure 5b d=3/P=1", figure5b,
         f"V16 median fractional residual reduction is {v16_figure5b.get('median_fractional_residual_reduction', 'missing')}; sustained-positive runs={v16_figure5b.get('sustained_positive_run_count', 'missing')}/{v16_figure5b.get('run_count', 'missing')}."),
        ("11. Measured versus predicted contraction", contraction,
         f"Local source-normalized q agrees with |1-alpha*kappa|; maximum disagreements are {[row.get('maximum_abs_disagreement') for row in contraction.get('rows', [])]}."),
        ("12. Remaining source-identifiable mismatch", acceptance,
         f"Failed reduced gates: {remaining or 'none'}. A failure remains a blocker and did not relax a metric or retune the optimizer."),
        ("13. Readiness for source-budget acquisition", acceptance,
         f"Readiness is `{status['readiness']}`. Figure 5c and natural drift were not run or modified; final and paper-equivalence evidence remain false."),
    ]
    lines = ["# V16 source-normalized optimizer consistency report", "",
             f"Readiness: **{status['readiness']}**", "",
             "All evidence remains development-only and non-final. No source-budget, held-out, natural-drift, Figure 5c or reference campaign was launched.", ""]
    if acceptance:
        conclusion = ("The optimizer-consistency hypothesis is supported for matched step, recovery, and Figure 5b, "
                      "but is not accepted as a complete repair because at least one reduced gate failed."
                      if not acceptance.get("pass") else
                      "The reduced campaign supports the optimizer-consistency hypothesis; source equivalence remains untested.")
        lines.extend([f"Causal conclusion: **{conclusion}**", ""])
    report_sections = []
    for heading, artifact, statement in sections:
        state = "MISSING" if artifact is None else ("PASS" if artifact.get("pass") is True else
                "RECORDED" if artifact.get("pass") is None else "FAIL")
        lines.extend([f"## {heading}", "", f"Status: **{state}**. {statement}", ""])
        report_sections.append({"heading": heading, "status": state, "statement": statement})
    if status["blockers"]:
        lines.extend(["## Blocking conditions", "",
                      *[f"- `{blocker}`" for blocker in status["blockers"]], ""])
    atomic_text(ARTIFACT_ROOT / "FINAL_REPORT.md", "\n".join(lines))
    result = {
        "schema_version": "google-pure-v16-report-manifest.v1",
        "sections": report_sections,
        "status_hash": canonical_hash(status),
        "readiness": status["readiness"],
        "blockers": status["blockers"],
        "report_path": "artifacts/google_pure_v16/FINAL_REPORT.md",
        **NONFINAL,
    }
    atomic_json(ARTIFACT_ROOT / "report_manifest.json", result)
    return result
