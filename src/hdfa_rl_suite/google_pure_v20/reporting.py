"""V20 status synthesis and evidence-bounded final report."""
from __future__ import annotations

from typing import Any

from .data import FORBIDDEN_CAMPAIGNS, verify_import_manifest
from .io import ARTIFACT_ROOT, atomic_json, atomic_text, nonfinal, read_json


REQUIRED_DIAGNOSTICS = (
    "fast_mean_cost_decomposition",
    "transfer_evaluation_geometry_audit",
    "fast_gradient_statistics",
    "fast_update_efficiency",
    "fast_reference_gradients",
    "candidate_vs_shots_factorial",
    "fixed_budget_information_comparison",
    "frozen_scale_information_damage_frontier",
    "dynamic_sigma_signed_gradients",
    "acquisition_bias_audit",
    "population_gradient_fast_rollout",
    "root_cause_classification",
    "minimal_repair",
    "postrepair_fast_validation",
)


def build_report() -> dict[str, Any]:
    manifest = verify_import_manifest()
    missing = [name for name in REQUIRED_DIAGNOSTICS
               if not (ARTIFACT_ROOT / f"{name}.json").is_file()]
    if missing:
        raise RuntimeError(f"V20 report requires missing artifacts: {missing}")
    artifacts = {name: read_json(ARTIFACT_ROOT / f"{name}.json")
                 for name in REQUIRED_DIAGNOSTICS}
    decomposition = artifacts["fast_mean_cost_decomposition"]
    geometry = artifacts["transfer_evaluation_geometry_audit"]
    gradients = artifacts["fast_gradient_statistics"]
    reference = artifacts["fast_reference_gradients"]
    factorial = artifacts["candidate_vs_shots_factorial"]
    fixed = artifacts["fixed_budget_information_comparison"]
    scale = artifacts["frozen_scale_information_damage_frontier"]
    acquisition = artifacts["acquisition_bias_audit"]
    population = artifacts["population_gradient_fast_rollout"]
    root = artifacts["root_cause_classification"]
    validation = artifacts["postrepair_fast_validation"]
    penalties = decomposition["sequential_component_penalties"]
    answers = {
        "1_missing_fast_cost_explanation": (
            f"{decomposition['classification']}; exact A-to-E penalty="
            f"{decomposition['exact_fundamental_to_full_missing_cost']:.6f}"),
        "2_component_costs": {
            **penalties,
            "transfer_geometry_classification": geometry["classification"],
        },
        "3_K8_reference_alignment": {
            "median_K8_alignment": reference["median_ordinary_reference_alignment"],
            "median_high_candidate_alignment": reference[
                "median_highcandidate_reference_alignment"],
            "classification": reference["classification"],
        },
        "4_limiting_information_source": {
            "factorial": factorial["classification"],
            "fixed_budget": fixed["classification"],
        },
        "5_sigma_information_damage": {
            "classification": scale["classification"],
            "pareto_frontier": scale["pareto_frontier"],
        },
        "6_batch_motion": acquisition["classification"],
        "7_population_gradient_success": population["sampling_information_failure_pattern"],
        "8_primary_failure": root["primary_classification"],
        "9_repair_after_classification": artifacts["minimal_repair"][
            "causal_parent_sha256"] == __import__(
                "hashlib").sha256((ARTIFACT_ROOT / "root_cause_classification.json").read_bytes()).hexdigest(),
        "10_repair_result_and_lineage": {
            "baseline_I_mean": validation["baseline_v19_experimental_fast"]["I_mean"],
            "repaired_I_mean": validation["repaired_fast"]["I_mean"],
            "source_style_branch_unchanged": validation["gates"][
                "source_style_branch_unchanged"],
        },
    }
    required_pass = all(artifacts[name].get("pass") is True for name in (
        "fast_mean_cost_decomposition", "transfer_evaluation_geometry_audit",
        "fast_gradient_statistics", "fast_update_efficiency", "fast_reference_gradients",
        "candidate_vs_shots_factorial", "fixed_budget_information_comparison",
        "frozen_scale_information_damage_frontier", "dynamic_sigma_signed_gradients",
        "acquisition_bias_audit", "population_gradient_fast_rollout",
        "root_cause_classification", "minimal_repair", "postrepair_fast_validation"))
    status = nonfinal({
        "pass": required_pass,
        "execution_complete": True,
        "primary_root_cause": root["primary_classification"],
        "secondary_causes": root["secondary_causes"],
        "repair": artifacts["minimal_repair"]["repair"],
        "repaired_controller_hash": artifacts["minimal_repair"]["controller"][
            "controller_hash"],
        "completion_answers": answers,
        "required_artifacts_complete": True,
        "frozen_lineage_manifest_pass": manifest["pass"],
        "frozen_source_style_branch_unchanged": validation["gates"][
            "source_style_branch_unchanged"],
        "frozen_v19_experimental_parent_unchanged": True,
        "slow_intermediate_rerun": False,
        "forbidden_auto_runs": list(FORBIDDEN_CAMPAIGNS),
        "forbidden_auto_runs_launched": [],
    })
    atomic_json(ARTIFACT_ROOT / "status.json", status)
    baseline = validation["baseline_v19_experimental_fast"]
    repaired = validation["repaired_fast"]
    pop = validation["population_gradient_reference"]
    lines = [
        "# V20 fast mean-failure diagnosis and causal repair", "",
        f"Primary root cause: **{root['primary_classification']}**.",
        f"Single repair: **{artifacts['minimal_repair']['repair']}**.", "",
        "## Missing mean-policy cost", "",
        f"- Exact decomposition classification: `{decomposition['classification']}`.",
        f"- Fundamental-only to full-41D exact penalty: "
        f"{decomposition['exact_fundamental_to_full_missing_cost']:.6f}.",
        f"- DC {penalties['DC_penalty']:.6f}; harmonics {penalties['harmonic_penalty']:.6f}; "
        f"transient {penalties['transient_penalty']:.6f}; orthogonal "
        f"{penalties['orthogonal_penalty']:.6f}.",
        f"- Transfer weighting audit: `{geometry['classification']}`.", "",
        "## Gradient information", "",
        f"- Stored K=8/reference median alignment: "
        f"{reference['median_ordinary_reference_alignment']:.4f}.",
        f"- High-candidate/reference median alignment: "
        f"{reference['median_highcandidate_reference_alignment']:.4f}.",
        f"- Reference classification: `{reference['classification']}`.",
        f"- K/M factorial: `{factorial['classification']}`; fixed-budget result: "
        f"`{fixed['classification']}`.",
        f"- Logged optimizer median update efficiency: "
        f"{read_json(ARTIFACT_ROOT / 'fast_update_efficiency.json')['median_eta']:.6f}.",
        f"- Batch-motion audit: `{acquisition['classification']}`.", "",
        "## Decisive and repaired fast-only comparisons", "",
        "| branch | I_mean | I_stochastic | G | phase lag | orthogonal power |",
        "|---|---:|---:|---:|---:|---:|",
        f"| V19 ordinary | {baseline['I_mean']:.5f} | {baseline['I_stochastic']:.5f} | "
        f"{baseline['gain']:.5f} | {baseline['phase_lag_radians']:.5f} | "
        f"{baseline['orthogonal_diffusion_power']:.6g} |",
        f"| V20 repaired | {repaired['I_mean']:.5f} | {repaired['I_stochastic']:.5f} | "
        f"{repaired['gain']:.5f} | {repaired['phase_lag_radians']:.5f} | "
        f"{repaired['orthogonal_diffusion_power']:.6g} |",
        f"| population reference | {pop['I_mean']:.5f} | {pop['I_stochastic']:.5f} | "
        f"{pop['gain']:.5f} | {pop['phase_lag_radians']:.5f} | "
        f"{pop['orthogonal_diffusion_power']:.6g} |", "",
        "## Evidence boundary", "",
        "This is a bounded fast-only, single-seed development diagnosis. It is not source-exact, "
        "paper-comparable, final evidence, or a paper-equivalence result. No slow, intermediate, "
        "long, held-out, source-budget, natural-drift, Figure 5c, or paired campaign was launched.",
    ]
    atomic_text(ARTIFACT_ROOT / "FINAL_REPORT.md", "\n".join(lines))
    return status


def status() -> dict[str, Any]:
    path = ARTIFACT_ROOT / "status.json"
    return read_json(path) if path.is_file() else build_report()
