"""Console entry points for the fail-closed public-paper reproduction workflow."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Callable

from .audit import write_forensic_audit
from .config import load_reference_config, repository_root, sha256_file
from .experiments import (
    run_drift_stability,
    run_finetuning,
    run_randomized_recovery,
    run_scaling,
    run_source_choice_sensitivity,
    run_steering_phase,
    run_step_response,
)
from .reporting import (
    artifact_directory,
    initial_gate_artifacts,
    source_tree_hash,
    write_amendment_log,
    write_anchor_registry,
    write_development_scorecard,
    write_json,
    write_markdown,
    write_public_data_reproduction,
    write_surrogate_contract,
)
from .validation import validate_surrogate


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def anchors_main() -> None:
    registry = write_anchor_registry()
    public_data = write_public_data_reproduction()
    write_development_scorecard()
    initial_gate_artifacts()
    write_amendment_log()
    _print({"anchors": len(registry["anchors"]), "public_data_status": public_data["status"]})


def audit_main() -> None:
    _print(write_forensic_audit())


def validate_surrogate_main() -> None:
    payload = validate_surrogate()
    write_json("surrogate_validation", payload)
    write_markdown(
        "surrogate_validation",
        "Surrogate validation",
        [f"Status: `{payload['status']}`.", "", *[f"- {name}: `{value}`" for name, value in payload["checks"].items()]],
    )
    _print(payload)


def _development_parser(name: str, default_seed: int, default_epochs: int | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=name)
    parser.add_argument("--seed", type=int, default=default_seed)
    if default_epochs is not None:
        parser.add_argument("--epochs", type=int, default=default_epochs)
    parser.add_argument("--execute", action="store_true", help="execute after printing paper-native cost")
    return parser.parse_args()


def _run_development(
    command: str,
    anchor: str,
    function: Callable[..., dict[str, Any]],
    *,
    default_seed: int,
    default_epochs: int | None,
    include_sensitivity: bool = False,
) -> None:
    args = _development_parser(command, default_seed, default_epochs)
    config = load_reference_config()
    epochs = getattr(args, "epochs", 1)
    accounted_epochs = epochs
    if anchor in {"drift_stability", "step_response"}:
        accounted_epochs += 150
    if anchor == "steering_phase":
        accounted_epochs = (epochs + 150) * 12
    if include_sensitivity:
        accounted_epochs += 6 * 160
    cost = config.cost(max(1, accounted_epochs))
    preview = {
        "command": command,
        "seed": args.seed,
        "estimated_host_runtime": "seconds to minutes on the analytic sufficient-statistic surrogate",
        **cost,
        "execution_requested": args.execute,
    }
    _print(preview)
    if not args.execute:
        print("Preview only. Re-run with --execute to spend this declared native-QEC-cycle budget.")
        return
    kwargs = {"seed": args.seed}
    if default_epochs is not None:
        kwargs["epochs"] = args.epochs
    result = function(**kwargs)
    sensitivity = run_source_choice_sensitivity(args.seed) if include_sensitivity else None
    scorecard = write_development_scorecard(anchor, result, sensitivity=sensitivity)
    _print({"anchor": anchor, "summary": result.get("summary", result.get("status")), "all_gates_pass": scorecard["all_required_development_gates_pass"]})


def finetuning_main() -> None:
    _run_development("google-rl-v2-run-finetuning-development", "fine_tuning", run_finetuning, default_seed=7901, default_epochs=240, include_sensitivity=True)


def drift_main() -> None:
    _run_development("google-rl-v2-run-drift-stability-development", "drift_stability", run_drift_stability, default_seed=7901, default_epochs=600)


def step_main() -> None:
    _run_development("google-rl-v2-run-step-response-development", "step_response", run_step_response, default_seed=7902, default_epochs=520)


def recovery_main() -> None:
    _run_development("google-rl-v2-run-randomized-recovery-development", "randomized_recovery", run_randomized_recovery, default_seed=7903, default_epochs=1400)


def steering_main() -> None:
    _run_development("google-rl-v2-run-steering-phase-development", "steering_phase", run_steering_phase, default_seed=7901, default_epochs=360)


def scaling_main() -> None:
    _run_development("google-rl-v2-run-scaling-development", "scaling", run_scaling, default_seed=7902, default_epochs=None)


def _pending_source_choices() -> list[str]:
    path = repository_root() / "configs/google_rl/source_unspecified_choices.yaml"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [choice["name"] for choice in payload["choices"] if "pending" in str(choice["sensitivity_result"]).lower()]


def freeze_main() -> None:
    scorecard_path = artifact_directory() / "development_anchor_scorecard.json"
    scorecard = json.loads(scorecard_path.read_text(encoding="utf-8")) if scorecard_path.exists() else write_development_scorecard()
    blockers = []
    if not scorecard.get("all_required_development_gates_pass"):
        blockers.append("not all independent development anchors passed")
    pending = _pending_source_choices()
    if pending:
        blockers.append("source-choice sensitivities pending: " + ", ".join(pending))
    validation_path = artifact_directory() / "surrogate_validation.json"
    if not validation_path.exists() or json.loads(validation_path.read_text(encoding="utf-8")).get("status") != "PASS":
        blockers.append("surrogate validation has not passed")
    source_hash, source_files = source_tree_hash()
    payload = {
        "schema_version": "google-public-certification-preregistration.v2",
        "status": "FROZEN_READY" if not blockers else "DRAFT_BLOCKED_PENDING_DEVELOPMENT_GATES",
        "frozen": not blockers,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat() if not blockers else None,
        "source_aggregate_sha256": source_hash,
        "source_files": source_files,
        "anchor_seed_assignment": {
            "fine_tuning": 8101,
            "drift_stability": 8102,
            "step_response": 8103,
            "randomized_recovery": 8104,
            "steering_phase": 8105,
            "scaling": 8106,
        },
        "unused_reserve_certification_seeds": [8107, 8108, 8109, 8110, 8111, 8112],
        "certification_seeds_consumed": False,
        "single_run_limit": 1,
        "allowed_outcomes": [
            "GOOGLE_STYLE_OPEN_REPRODUCTION_CERTIFIED",
            "PARTIAL_REPRODUCTION_ONLY",
            "REPRODUCTION_FAILED_ALGORITHM",
            "REPRODUCTION_FAILED_SURROGATE",
            "REPRODUCTION_NOT_EVALUABLE_MISSING_PUBLIC_DETAIL",
        ],
        "blocked_by": blockers,
        "no_post_opening_amendments": True,
    }
    write_json("certification_preregistration", payload)
    write_markdown(
        "certification_preregistration",
        "Certification preregistration",
        [
            f"Status: `{payload['status']}`.",
            "",
            f"Source aggregate: `{source_hash}`.",
            "",
            *(["Blockers:", "", *[f"- {item}" for item in blockers]] if blockers else ["The single-use held-out run is frozen and ready. No amendments are allowed after opening."]),
        ],
    )
    _print(payload)
    if blockers:
        raise SystemExit(2)


def certification_main() -> None:
    parser = argparse.ArgumentParser(prog="google-rl-v2-run-certification")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-single-use", action="store_true")
    args = parser.parse_args()
    config = load_reference_config()
    total_epochs = 240 + (600 + 150) + (520 + 150) + 1400 + (360 + 150) * 12 + 1
    _print({
        "estimated_runtime": "minutes on analytic sufficient statistics; hardware projection is much larger",
        **config.cost(total_epochs),
        "certification_is_single_use": True,
        "execution_requested": args.execute,
    })
    if not (args.execute and args.acknowledge_single_use):
        print("Preview only. Both --execute and --acknowledge-single-use are required.")
        return
    prereg_path = artifact_directory() / "certification_preregistration.json"
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    if prereg.get("status") != "FROZEN_READY" or not prereg.get("frozen"):
        raise SystemExit("certification is not frozen and ready")
    current_hash, _ = source_tree_hash()
    if current_hash != prereg["source_aggregate_sha256"]:
        raise SystemExit("source/config hash changed after freeze")
    lock = artifact_directory() / "certification_single_use.lock"
    try:
        with lock.open("x", encoding="utf-8") as stream:
            stream.write(datetime.now(timezone.utc).isoformat() + "\n")
    except FileExistsError as error:
        raise SystemExit("single-use certification was already started; rerun is forbidden") from error
    results = {
        "fine_tuning": run_finetuning(8101, certification=True),
        "drift_stability": run_drift_stability(8102, certification=True),
        "step_response": run_step_response(8103, certification=True),
        "randomized_recovery": run_randomized_recovery(8104, certification=True),
        "steering_phase": run_steering_phase(8105, certification=True),
        "scaling": run_scaling(8106, certification=True),
    }
    passed = [name for name, result in results.items() if result.get("summary", {}).get("status", result.get("status")) == "PASS"]
    outcome = "GOOGLE_STYLE_OPEN_REPRODUCTION_CERTIFIED" if len(passed) == len(results) else "PARTIAL_REPRODUCTION_ONLY"
    payload = {
        "schema_version": "google-public-final-certification.v2",
        "status": "COMPLETE",
        "outcome": outcome,
        "certification_executed": True,
        "certification_run_count": 1,
        "certification_seeds_consumed": True,
        "passed_anchors": passed,
        "failed_anchors": [name for name in results if name not in passed],
        "results": {name: result.get("summary", {"status": result.get("status")}) for name, result in results.items()},
        "source_aggregate_sha256": current_hash,
        "claim_boundary": "Open algorithm on the declared surrogate; never Willow hardware equivalence.",
    }
    write_json("final_certification", payload)
    write_markdown(
        "final_certification",
        "Final certification",
        [f"Outcome: `{outcome}`.", "", f"Passed anchors: {', '.join(passed) or 'none'}.", "", "This outcome applies only to the declared surrogate evidence layer."],
    )
    _print(payload)


def reduced_budget_main() -> None:
    final_path = artifact_directory() / "final_certification.json"
    final = json.loads(final_path.read_text(encoding="utf-8")) if final_path.exists() else {}
    if final.get("outcome") != "GOOGLE_STYLE_OPEN_REPRODUCTION_CERTIFIED":
        payload = {
            "schema_version": "google-public-reduced-budget-equivalence.v2",
            "status": "BLOCKED_REFERENCE_NOT_CERTIFIED",
            "evaluated": False,
            "reason": "Paper-scale reference is not fully certified; reduced acquisition remains scientifically unvalidated.",
        }
        write_json("reduced_budget_equivalence", payload)
        write_markdown("reduced_budget_equivalence", "Reduced-budget equivalence", [f"Status: `{payload['status']}`.", "", payload["reason"]])
        _print(payload)
        raise SystemExit(2)
    raise SystemExit("reference is certified, but a new reduced-budget preregistration must be frozen before spending held-out environments")
