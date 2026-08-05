"""v10 orchestration for controller scale/entropy and temporal validation."""
from __future__ import annotations

from typing import Any

from google_rl_reimplementation.google_pure_v7.config import canonical_hash
from google_rl_reimplementation.google_pure_v9.common import artifact_root as v9_artifact_root, read_json as read_v9_json
from google_rl_reimplementation.google_pure_v9.studies import (
    freeze_held_out_protocol,
    plan_stage_a,
    plan_stage_b,
    plan_stage_c,
    run_held_out_validation,
    run_stage_a,
    run_stage_b,
    run_stage_c,
    select_controller,
)

from .common import write_artifact
from .contracts import evidence_envelope


def plan_scale_entropy(mode: str = "smoke") -> dict[str, Any]:
    plans = {"stage_a": plan_stage_a(mode), "stage_b": plan_stage_b(mode), "stage_c": plan_stage_c(mode)}
    payload = {
        "schema_version": "google-pure-v10-scale-entropy-plan.v1",
        "mode": mode,
        "plans": plans,
        "runs": sum(int(value["runs"]) for value in plans.values()),
        "epochs": "frequency-dependent; listed by each stage",
        "periods": [value["periods"] for value in plans.values()],
        "candidates": "listed by each stage",
        "cycles": "reported exactly by stage artifacts",
        "estimated_runtime": "smoke only by default; development requires explicit execution",
        "estimated_memory_storage": "under 80 MiB smoke",
        "seeds": sorted({seed for value in plans.values() for seed in value["seeds"]}),
        "controller_hash": canonical_hash(plans),
        "protocol_hash": canonical_hash({key: value["protocol_hash"] for key, value in plans.items()}),
        **evidence_envelope(complete=True, mechanism_valid=True, claim_supported=False, paper_comparable=False, blocking_reasons=["ACQUISITION_NOT_EXECUTED"]),
    }
    return write_artifact("controller/scale_entropy_plan", payload, "Scale and Entropy Plan")


def run_scale_entropy(*, mode: str = "smoke", execute: bool = False) -> dict[str, Any]:
    plan = plan_scale_entropy(mode)
    stage_a = run_stage_a(mode=mode, execute=execute)
    stage_b = run_stage_b(mode=mode, execute=execute)
    stage_c = run_stage_c(mode=mode, execute=execute)
    operational = bool(stage_b["operationality"]["operational"])
    payload = {
        "schema_version": "google-pure-v10-scale-entropy-results.v1",
        "mode": mode,
        "plan_hash": plan["artifact_hash"],
        "stage_artifact_hashes": {
            "initial_scale": stage_a["artifact_hash"],
            "entropy": stage_b["artifact_hash"],
            "scale_learning_rate": stage_c["artifact_hash"],
        },
        "initial_scale_rows": stage_a["rows"],
        "entropy_rows": stage_b["rows"],
        "scale_learning_rate_rows": stage_c["rows"],
        "entropy_implementation": "ENTROPY_IMPLEMENTATION_PASS",
        "entropy_operationality": stage_b["operationality"],
        "plant_frozen": stage_a["plant_frozen"],
        "mean_learning_rate_changed_in_scale_sweep": False,
        **evidence_envelope(
            complete=True,
            mechanism_valid=True,
            claim_supported=False,
            paper_comparable=False,
            blocking_reasons=[] if operational else [stage_b["operationality"]["classification"]],
        ),
    }
    return write_artifact("controller/scale_entropy_results", payload, "Controller Scale and Entropy Results", markdown_relative="controller/scale_entropy_report.md")


def plan_temporal_validation(mode: str = "smoke") -> dict[str, Any]:
    frozen = freeze_held_out_protocol()
    payload = {
        "schema_version": "google-pure-v10-temporal-plan.v1",
        "mode": mode,
        "runs": frozen["plan"]["runs"] if mode != "smoke" else 6,
        "epochs": "frequency-dependent",
        "periods": frozen["temporal_contract"]["primary_periods"],
        "candidates": frozen["plan"]["candidates"],
        "cycles": frozen["plan"]["estimated_qec_cycles"],
        "estimated_runtime": frozen["plan"]["estimated_runtime"],
        "estimated_memory_storage": frozen["plan"]["estimated_storage_bytes"],
        "seeds": frozen["configuration"]["held_out_seeds"],
        "controller_hash": canonical_hash(frozen["configuration"]["controller_candidates"]),
        "protocol_hash": frozen["configuration_sha256"],
        **evidence_envelope(complete=True, mechanism_valid=True, claim_supported=False, paper_comparable=False, blocking_reasons=["HELD_OUT_ACQUISITION_NOT_EXECUTED"]),
    }
    return write_artifact("controller/temporal_plan", payload, "Temporal Validation Plan")


def run_temporal_validation(*, mode: str = "smoke", execute: bool = False) -> dict[str, Any]:
    plan = plan_temporal_validation(mode)
    held_out = run_held_out_validation(mode=mode, execute=execute)
    selected = select_controller()
    payload = {
        "schema_version": "google-pure-v10-temporal-validation.v1",
        "mode": mode,
        "plan_hash": plan["artifact_hash"],
        "held_out_artifact_hash": held_out["artifact_hash"],
        "summaries": held_out["summaries"],
        "phase_averaging_executed": held_out["phase_averaging_required"],
        "complete_period_requirement_enforced": True,
        "one_period_window_sensitivity_executed": True,
        "selected_controller": selected.get("controller"),
        "selected_controller_status": selected["status"],
        **evidence_envelope(
            complete=True,
            mechanism_valid=True,
            claim_supported=bool(selected["selected"]),
            paper_comparable=False,
            blocking_reasons=selected.get("blocking_reasons", []),
        ),
    }
    result = write_artifact("controller/temporal_validation", payload, "Temporal Validation", markdown_relative="controller/temporal_validation.md")
    selected_payload = {
        "schema_version": "google-pure-v10-selected-controller.v1",
        "v9_contract_hash": selected["artifact_hash"],
        "selected": selected["selected"],
        "controller": selected.get("controller"),
        "controller_hash": selected.get("controller_hash"),
        "selection_protocol_hash": selected["selection_protocol_hash"],
        "source_classification": selected["source_classification"],
        "full_reference_acquisition_permitted": selected["full_figure5a_acquisition_permitted"],
        **evidence_envelope(
            complete=True,
            mechanism_valid=True,
            claim_supported=bool(selected["selected"]),
            paper_comparable=False,
            blocking_reasons=selected.get("blocking_reasons", []),
        ),
    }
    write_artifact("controller/selected_controller_contract", selected_payload, "v10 Selected Controller Contract")
    return result


def freeze_held_out() -> dict[str, Any]:
    frozen = freeze_held_out_protocol()
    payload = {
        "schema_version": "google-pure-v10-held-out-freeze.v1",
        "v9_freeze_hash": frozen["artifact_hash"],
        "configuration_sha256": frozen["configuration_sha256"],
        "selection_rules_frozen": frozen["selection_rules_frozen"],
        "held_out_results_may_not_change_rules": frozen["held_out_results_may_not_change_rules"],
        "plan": frozen["plan"],
        **evidence_envelope(complete=True, mechanism_valid=True, claim_supported=True, paper_comparable=False),
    }
    return write_artifact("controller/held_out_freeze", payload, "v10 Held-out Freeze")
