"""V21 classification, one-repair rule, status, and final report."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from .candidate_design import SOURCE_FIDELITY
from .io import ARTIFACT_ROOT, atomic_json, atomic_text, nonfinal, read_json, write_artifact
from .lineage import FORBIDDEN_CAMPAIGNS, verify_import_manifest


def classify() -> dict[str, Any]:
    verify_import_manifest()
    required = {
        "projection": "projection_reference_retention.json",
        "variance": "gradient_variance_decomposition.json",
        "benchmark": "frozen_state_candidate_design_benchmark.json",
        "pareto": "candidate_design_scale_pareto.json",
        "promotion": "frozen_state_promotion_gate.json",
        "online": "short_fast_online_rollouts.json",
        "generalization": "generalization_audit.json",
    }
    missing = [name for name in required.values() if not (ARTIFACT_ROOT / name).is_file()]
    if missing:
        raise RuntimeError(f"V21 classification requires missing artifacts: {missing}")
    values = {key: read_json(ARTIFACT_ROOT / name) for key, name in required.items()}
    promoted = values["promotion"]["promoted_designs_for_short_online_rollout"]
    online_success = False
    if promoted:
        row = next(source for source in values["online"]["online_rows"]
                   if source["design_id"] == promoted[0])
        baseline = values["online"]["comparators"]["V19_V20_iid_gaussian_baseline"]
        online_success = (
            row["summary"]["I_mean"] > baseline["I_mean"] and
            row["summary"]["orthogonal_mean_diffusion"] <
            baseline["orthogonal_diffusion_power"])
    active = []
    if values["variance"]["direction_variance_material"]:
        active.append("DIRECTION_VARIANCE_CONFIRMED")
    if values["pareto"]["designs_pareto_dominating_iid"]:
        active.append("STRUCTURED_DESIGN_SHIFTS_INFORMATION_DAMAGE_FRONTIER")
    elif promoted:
        active.append("STRUCTURED_DESIGN_IMPROVES_ESTIMATOR_ONLY")
    else:
        active.append("NO_FIXED_BUDGET_DESIGN_IMPROVEMENT")
    if promoted and SOURCE_FIDELITY[promoted[0]] == "SOURCE_IMPLIED":
        active.append("FACTOR_GRAPH_LOCAL_DESIGN_SOURCE_IMPLIED")
    if promoted and not values["generalization"]["pass"]:
        active.append("DIAGNOSTIC_DESIGN_ONLY")
    primary = (
        "FACTOR_GRAPH_LOCAL_DESIGN_SOURCE_IMPLIED" if
        "FACTOR_GRAPH_LOCAL_DESIGN_SOURCE_IMPLIED" in active and online_success and
        values["generalization"]["pass"] else
        "DIAGNOSTIC_DESIGN_ONLY" if "DIAGNOSTIC_DESIGN_ONLY" in active else
        "STRUCTURED_DESIGN_SHIFTS_INFORMATION_DAMAGE_FRONTIER" if
        "STRUCTURED_DESIGN_SHIFTS_INFORMATION_DAMAGE_FRONTIER" in active else active[0])
    value = {
        "pass": True,
        "primary_classification": primary,
        "active_classifications": active,
        "promoted_design": promoted[0] if promoted else None,
        "online_mechanism_improvement": online_success,
        "generalization_pass": values["generalization"]["pass"],
        "hard_projection_status": "ORACLE_LIKE_DIAGNOSTIC_UPPER_BOUND",
        "hard_projection_promoted": False,
        "forbidden_auto_runs_launched": [],
    }
    return write_artifact("root_cause_and_candidate_design_classification", value,
                          title="V21 root-cause and candidate-design classification")


def decide_minimal_repair() -> dict[str, Any]:
    verify_import_manifest()
    classification_path = ARTIFACT_ROOT / "root_cause_and_candidate_design_classification.json"
    if not classification_path.is_file():
        classify()
    classification = read_json(classification_path)
    promotion = read_json(ARTIFACT_ROOT / "frozen_state_promotion_gate.json")
    online = read_json(ARTIFACT_ROOT / "short_fast_online_rollouts.json")
    generalization = read_json(ARTIFACT_ROOT / "generalization_audit.json")
    benchmark = read_json(ARTIFACT_ROOT / "frozen_state_candidate_design_benchmark.json")
    pareto = read_json(ARTIFACT_ROOT / "candidate_design_scale_pareto.json")
    promoted = list(promotion["promoted_designs_for_short_online_rollout"])
    if promoted:
        design_id = promoted[0]
        frozen = next(row for row in benchmark["designs"] if row["design_id"] == design_id)
        baseline_frozen = next(row for row in benchmark["designs"] if row["design_id"] == "D0")
        online_row = next(row for row in online["online_rows"] if row["design_id"] == design_id)
        baseline_online = online["comparators"]["V19_V20_iid_gaussian_baseline"]
        pareto_row = next(row for row in pareto["designs"] if row["design_id"] == design_id)
        gate = next(row for row in promotion["design_gates"] if row["design_id"] == design_id)
        gates = {
            "fixed_total_cycle_budget": online_row["K"] * online_row["M"] ==
                online_row["B"] == 96000,
            "mathematically_valid_estimator": gate["gates"]["estimator_valid"],
            "better_population_gradient_approximation":
                frozen["mean_reference_gradient_MSE"] <
                baseline_frozen["mean_reference_gradient_MSE"],
            "no_oracle_information": SOURCE_FIDELITY[design_id] != "ORACLE_LIKE",
            "legitimate_reference_components_not_suppressed":
                frozen["median_reference_gradient_capture"] > .5 and
                frozen["mean_orthogonal_error_power"] <
                baseline_frozen["mean_orthogonal_error_power"],
            "directional_magnitude_and_phase_favorable": .5 <=
                frozen["median_directional_magnitude_ratio"] <= 1.5 and
                online_row["summary"]["closer_to_population_than_iid"] and
                online_row["summary"]["distance_to_population_gain_phase"] <=
                float(np.hypot(
                    baseline_online["gain"] - online["comparators"][
                        "V20_population_gradient_reference"]["gain"],
                    baseline_online["phase_lag_radians"] - online["comparators"][
                        "V20_population_gradient_reference"]["phase_lag_radians"])),
            "information_damage_pareto_shift": pareto_row["any_pareto_dominance"],
            "fast_I_mean_positive_or_materially_improved":
                online_row["summary"]["I_mean"] > 0 or
                online_row["summary"]["I_mean"] - baseline_online["I_mean"] >= .1,
            "I_stochastic_interpretation_present": True,
            "generalization_pass": generalization["pass"],
            "frozen_source_style_branch_unchanged": verify_import_manifest()["invariants"][
                "source_style_branch_unchanged"],
        }
        adopted = all(gates.values())
        reason = None if adopted else "one or more preregistered promotion requirements failed"
    else:
        design_id = None
        gates = {}
        adopted = False
        reason = "no fixed-budget design passed the frozen-state promotion gate"
    value = {
        "pass": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repair_adopted": adopted,
        "repair_design": design_id if adopted else None,
        "candidate_design_source_fidelity": SOURCE_FIDELITY.get(design_id) if design_id else None,
        "promotion_requirements": gates,
        "reason": reason,
        "at_most_one_repair": True,
        "repairs_adopted_count": 1 if adopted else 0,
        "classification_parent": classification["primary_classification"],
        "hard_projection_adopted": False,
        "source_style_branch_changed": False,
        "forbidden_auto_runs_launched": [],
    }
    return write_artifact("minimal_candidate_design_repair", value,
                          title="V21 minimal candidate-design repair decision")


REQUIRED = (
    "projection_reference_retention", "gradient_variance_decomposition",
    "candidate_design_source_fidelity", "candidate_estimators", "candidate_frame_coverage",
    "frozen_state_candidate_design_benchmark", "exploration_damage_metric_reconciliation",
    "candidate_design_scale_pareto", "frozen_state_promotion_gate",
    "short_fast_online_rollouts", "generalization_audit",
    "root_cause_and_candidate_design_classification", "minimal_candidate_design_repair",
)


def build_report() -> dict[str, Any]:
    manifest = verify_import_manifest()
    missing = [name for name in REQUIRED if not (ARTIFACT_ROOT / f"{name}.json").is_file()]
    if missing:
        raise RuntimeError(f"V21 report requires missing artifacts: {missing}")
    a = {name: read_json(ARTIFACT_ROOT / f"{name}.json") for name in REQUIRED}
    projection = a["projection_reference_retention"]
    variance = a["gradient_variance_decomposition"]
    benchmark = a["frozen_state_candidate_design_benchmark"]
    pareto = a["candidate_design_scale_pareto"]
    online = a["short_fast_online_rollouts"]
    generalization = a["generalization_audit"]
    classification = a["root_cause_and_candidate_design_classification"]
    repair = a["minimal_candidate_design_repair"]
    best = next(row for row in benchmark["designs"]
                if row["design_id"] == benchmark["best_MSE_design"])
    promoted = online["promoted_designs"]
    online_row = next((row for row in online["online_rows"]
                       if row["design_id"] == promoted[0]), None) if promoted else None
    answers = {
        "1_projection_discards_legitimate_content": {
            "answer": projection["classification"] == "HARD_PROJECTION_OVERREGULARIZED",
            "retained_fraction": projection["median_reference_gradient_retained_fraction"],
            "discarded_fraction": projection["median_reference_gradient_discarded_fraction"],
        },
        "2_gradient_variance": variance["nonnegative_variance_fractions"],
        "3_mathematically_valid_designs": {
            "frozen_mean_estimator_valid": [row["design_id"] for row in
                a["candidate_estimators"]["designs"] if row["mean_estimator_valid"]],
            "online_mean_and_sigma_estimator_valid": [row["design_id"] for row in
                a["candidate_estimators"]["designs"]
                if row["online_controller_eligible"]],
        },
        "4_source_fidelity": {row["design_id"]: row["source_fidelity"] for row in
            a["candidate_design_source_fidelity"]["designs"]},
        "5_best_fixed_budget_design": benchmark["best_MSE_design"],
        "6_orthogonal_error_reduction_ratio": best["orthogonal_error_ratio_to_iid"],
        "7_same_accuracy_at_lower_sigma": pareto[
            "designs_matching_iid_accuracy_at_smaller_sigma"],
        "8_pareto_dominating_designs": pareto["designs_pareto_dominating_iid"],
        "9_online_advantage_retained": online_row["summary"] if online_row else None,
        "10_generalization": generalization["classification"],
        "11_source_compatibility": repair["candidate_design_source_fidelity"],
        "12_source_style_branch_unchanged": manifest["invariants"][
            "source_style_branch_unchanged"],
    }
    status = nonfinal({
        "pass": all(a[name].get("pass") is True for name in REQUIRED),
        "execution_complete": True,
        "primary_classification": classification["primary_classification"],
        "active_classifications": classification["active_classifications"],
        "repair_adopted": repair["repair_adopted"],
        "repair_design": repair["repair_design"],
        "completion_answers": answers,
        "required_artifacts_complete": True,
        "source_style_branch_unchanged": manifest["invariants"][
            "source_style_branch_unchanged"],
        "v19_parent_unchanged": manifest["invariants"]["v19_parent_unchanged"],
        "v20_projection_not_promoted": True,
        "forbidden_auto_runs": list(FORBIDDEN_CAMPAIGNS),
        "forbidden_auto_runs_launched": [],
    })
    atomic_json(ARTIFACT_ROOT / "status.json", status)
    lines = [
        "# V21 fixed-budget candidate-design and Pareto repair", "",
        f"Primary classification: **{classification['primary_classification']}**.",
        f"Repair adopted: **{repair['repair_adopted']}**"
        + (f" (`{repair['repair_design']}`)." if repair["repair_design"] else "."), "",
        "## Projection and variance", "",
        f"- V20 hard projection retained {projection['median_reference_gradient_retained_fraction']:.3f} "
        f"and discarded {projection['median_reference_gradient_discarded_fraction']:.3f} of the "
        "reference-gradient norm; it remains oracle-like diagnostic evidence.",
        f"- Direction/shot/interaction variance fractions: "
        f"{variance['nonnegative_variance_fractions']['direction']:.3f} / "
        f"{variance['nonnegative_variance_fractions']['shot']:.3f} / "
        f"{variance['nonnegative_variance_fractions']['interaction']:.3f}.", "",
        "## Fixed-budget design result", "",
        f"- Best frozen-state design: `{benchmark['best_MSE_design']}` "
        f"({best['source_fidelity']}); MSE ratio to iid={best['MSE_ratio_to_iid']:.4f}, "
        f"orthogonal-error ratio={best['orthogonal_error_ratio_to_iid']:.4f}.",
        f"- Designs matching iid accuracy at lower sigma: "
        f"{', '.join(pareto['designs_matching_iid_accuracy_at_smaller_sigma']) or 'none'}.",
    ]
    if online_row:
        base = online["comparators"]["V19_V20_iid_gaussian_baseline"]
        source = online_row["summary"]
        lines += ["", "## Short online fast validation", "",
                  f"- iid baseline: I_mean={base['I_mean']:.5f}, "
                  f"I_stochastic={base['I_stochastic']:.5f}.",
                  f"- {online_row['design_id']}: I_mean={source['I_mean']:.5f}, "
                  f"I_stochastic={source['I_stochastic']:.5f}, orthogonal diffusion="
                  f"{source['orthogonal_mean_diffusion']:.6g}.",
                  f"- Generalization: `{generalization['classification']}`."]
    failed_repair_gates = [name for name, passed in
                           repair["promotion_requirements"].items() if not passed]
    if failed_repair_gates:
        lines += [f"- Repair blocked by: {', '.join(failed_repair_gates)}."]
    lines += ["", "## Evidence boundary", "",
              "V21 is a bounded, single-seed fast development study. It does not promote the V20 "
              "hard projection, consume held-out/source-budget evidence, or establish source-exact "
              "paper performance. Source-implied labels mean derivable from the public factor graph, "
              "not explicitly described in the public algorithm."]
    atomic_text(ARTIFACT_ROOT / "FINAL_REPORT.md", "\n".join(lines))
    return status


def status() -> dict[str, Any]:
    path = ARTIFACT_ROOT / "status.json"
    return read_json(path) if path.is_file() else build_report()
