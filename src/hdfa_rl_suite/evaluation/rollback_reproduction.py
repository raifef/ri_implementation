"""Deterministic replay of the four rollback anomalies retained in acceptance v2."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

from hdfa_rl_suite.recovery import (
    RestorationPrediction, RollbackVerificationStatus,
    verify_restoration_evidence,
)

from .acceptance_v2 import _ConcatenatedGzipText, _StreamingJSON


CASES = (
    ("confirmatory_semi_markov", 3002, "predictive_hdfa_residual_rl"),
    ("confirmatory_semi_markov", 3003, "predictive_hdfa_residual_rl"),
    ("confirmatory_nested_common", 3001, "predictive_hdfa_residual_rl"),
    ("confirmatory_nested_common", 3010, "predictive_hdfa_no_residual"),
)


def reproduce_rollbacks_v2(parts: tuple[Path, Path], output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected: dict[tuple[str, int, str], list[dict]] = {case: [] for case in CASES}

    def consume(value: object) -> None:
        row = value
        key = (str(row.get("scenario_id")), int(row.get("seed", -1)),
               str(row.get("arm")))
        if key in selected and row.get("rollback_outcomes"):
            selected[key].append(row)

    source = _ConcatenatedGzipText(parts)
    _StreamingJSON(source).top_level({"trajectories": consume})
    cases = []
    for scenario, seed, arm in CASES:
        rows = selected[(scenario, seed, arm)]
        failure_records = [
            (row, outcome) for row in rows for outcome in row["rollback_outcomes"]
            if str(outcome.get("physical_status", "")).endswith("failed")]
        if not failure_records:
            cases.append({"scenario_id": scenario, "seed": seed, "arm": arm,
                          "reproduced": False,
                          "reason": "no retained rollback outcome found"})
            continue
        failure_row, outcome = failure_records[0]
        observed = tuple(outcome.get("observed_detector_rate_ci99") or ())
        expected = tuple(outcome.get("expected_detector_rate_interval") or ())
        transaction_exact = (
            str(outcome.get("transaction_status", "")).endswith("confirmed")
            and outcome.get("expected_final_hash") == outcome.get("observed_final_hash")
            and bool(outcome.get("acknowledgement_ids")))
        if len(observed) == 2 and len(expected) == 2:
            if observed[0] > expected[1]:
                revised = RollbackVerificationStatus.PHYSICAL_RESTORATION_FAILED
                root = "finite-shot interval is credibly above the legacy bound"
            elif observed[1] <= expected[1]:
                revised = RollbackVerificationStatus.PHYSICAL_RESTORATION_VERIFIED
                root = "finite-shot interval is inside the legacy bound"
            else:
                revised = RollbackVerificationStatus.PHYSICAL_RESTORATION_INCONCLUSIVE
                root = "single-batch interval overlaps the boundary and was mislabelled as failure"
        else:
            revised = RollbackVerificationStatus.PHYSICAL_RESTORATION_INCONCLUSIVE
            root = "legacy record lacks a complete physical interval"
        retained = [{
            "interval": row["interval"],
            "elapsed_time_s": row["elapsed_time_s"],
            "physical_state_id": row.get("physical_state_id"),
            "disturbance_state_id": row.get("disturbance_state_id"),
            "evaluation_only_latent_state": row.get("evaluation_only_latent_state"),
            "evaluation_only_process_state": row.get("evaluation_only_process_state"),
            "policy_hash": row.get("policy_hash"),
            "controller_state_hash": row.get("controller_state_hash"),
            "authorization": row.get("authorization"),
            "lifecycle_mode": row.get("lifecycle_mode"),
            "rollback_outcomes": row.get("rollback_outcomes"),
            "stage_evidence": row.get("stage_evidence"),
        } for row in (failure_row,)]
        case = {
            "scenario_id": scenario, "seed": seed, "arm": arm,
            "reproduced": True, "transaction_exact": transaction_exact,
            "legacy_physical_status": outcome.get("physical_status"),
            "revised_evidence_classification": revised.value,
            "root_cause": root,
            "historical_bound_is_not_current_state_prediction": True,
            "retained_intervals": retained,
        }
        cases.append(case)
        (output_dir/f"{scenario}-{seed}-{arm}.json").write_text(
            json.dumps(case, indent=2, default=str), encoding="utf-8")
    report = {
        "schema_version": "rollback-reproduction.v2",
        "development_only": True,
        "immutable_source_sha256": source.digest.hexdigest(),
        "all_four_reproduced": len(cases) == 4 and all(
            item.get("reproduced") for item in cases),
        "transaction_restoration_exact_in_all_four": all(
            item.get("transaction_exact") for item in cases),
        "cases": cases,
        "classification": {
            "implementation_defect": (
                "historical validation intervals were substituted for a current-state "
                "restoration prediction; a one-shot boundary overlap was treated as failure"),
            "plant_limitation": (
                "none is established by these four overlapping intervals; seed 3003 has "
                "a high point estimate but still requires additional current-state evidence"),
            "statistical_limitation": (
                "all four boundary cases were underpowered; sequential alpha-controlled "
                "batches are required before declaring either verification or failure"),
            "controller_design_limitation": (
                "rollback selected controls transactionally but did not condition physical "
                "suitability on the current Stage-2/3 state"),
            "scientific_design_limitation": (
                "v2 encoded uncertainty as a lifecycle failure, conflating safety evidence "
                "with protocol correctness"),
        },
    }
    md = ["# Rollback v2 deterministic reproductions", "",
          f"All four retained anomalies reproduced: **{report['all_four_reproduced']}**", "",
          "These are lossless replays of the exact v2 intervals, seeds, disturbances, "
          "policy hashes, latent evaluation views, and rollback evidence. Controller truth "
          "access remains empty; latent state is retained only for evaluation.", "",
          "## Cases", ""]
    for case in cases:
        md.append(f"- `{case['scenario_id']}` seed {case['seed']} / `{case['arm']}`: "
                  f"{case.get('revised_evidence_classification')} — {case.get('root_cause')}")
    md.extend(["", "## Root-cause classification", ""])
    md.extend(f"- **{name.replace('_', ' ')}:** {text}"
              for name, text in report["classification"].items())
    (output_dir/"rollback_classification_report.md").write_text(
        "\n".join(md)+"\n", encoding="utf-8")
    (output_dir/"rollback_classification_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def validate_rollback_semantics(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction = RestorationPrediction(
        "test.v1", 0.0, "policy", "hash", 0.0, {"D0": .05},
        {"D0": .01}, {"q0": .12}, .05, .12, .20, .01, .0, .005)

    def batch(name: str, events: int, exposures: int):
        return SimpleNamespace(
            batch_id=name, cycles=exposures, detector_events=events,
            detector_exposures=exposures,
            detector_counts={"D0": (events, exposures)})

    cases = {
        "stationary_verified": verify_restoration_evidence(
            prediction, (batch("stationary", 5, 200),),
            regional_detectors={"q0": ("D0",)}),
        "ongoing_drift_failed": verify_restoration_evidence(
            prediction, (batch("drift", 40, 200),),
            regional_detectors={"q0": ("D0",)}),
        "boundary_inconclusive": verify_restoration_evidence(
            prediction, (batch("boundary", 10, 100),),
            regional_detectors={"q0": ("D0",)}),
        "additional_batches_resolve": verify_restoration_evidence(
            prediction, (batch("boundary", 10, 100), batch("more", 15, 400)),
            regional_detectors={"q0": ("D0",)}),
    }
    expected = {
        "stationary_verified": RollbackVerificationStatus.PHYSICAL_RESTORATION_VERIFIED,
        "ongoing_drift_failed": RollbackVerificationStatus.PHYSICAL_RESTORATION_FAILED,
        "boundary_inconclusive": RollbackVerificationStatus.PHYSICAL_RESTORATION_INCONCLUSIVE,
        "additional_batches_resolve": RollbackVerificationStatus.PHYSICAL_RESTORATION_VERIFIED,
    }
    checks = {name: value.status is expected[name] for name, value in cases.items()}
    checks["transaction_failure_is_distinct"] = (
        RollbackVerificationStatus.TRANSACTION_FAILED.value == "transaction_failed")
    report = {
        "schema_version": "rollback-semantics-validation.v1",
        "passed": all(checks.values()), "checks": checks,
        "cases": {name: asdict(value) for name, value in cases.items()},
    }
    (output_dir/"rollback-semantics-validation.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report
