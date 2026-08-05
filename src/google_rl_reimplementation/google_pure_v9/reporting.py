"""Fail-closed v9 synthesis, status, and next-command reporting."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import artifact_root, read_json, write_artifact


REQUIRED = (
    "v8_import_manifest.json",
    "corrected_v8_fault_classification.json",
    "stage_a_initial_scale/results.json",
    "stage_b_entropy/results.json",
    "stage_c_scale_learning_rate/results.json",
    "stage_d_held_out_validation/results.json",
    "selected_controller_contract.json",
)


def _required(name: str) -> dict[str, Any]:
    path = artifact_root() / name
    if not path.is_file():
        raise RuntimeError(f"missing required v9 artifact: {name}")
    return read_json(path)


def report_root_cause_update() -> dict[str, Any]:
    corrected = _required("corrected_v8_fault_classification.json")
    stage_a = _required("stage_a_initial_scale/results.json")
    stage_b = _required("stage_b_entropy/results.json")
    stage_c = _required("stage_c_scale_learning_rate/results.json")
    held_out = _required("stage_d_held_out_validation/results.json")
    selected = _required("selected_controller_contract.json")
    selected_row = selected.get("controller")
    entropy = stage_b["operationality"]
    blockers = list(selected.get("blocking_reasons", []))
    if held_out.get("mode") == "smoke":
        blockers.append("REFERENCE_HELD_OUT_ACQUISITION_NOT_EXECUTED")
    payload = {
        "schema_version": "google-pure-v9-root-cause-update.v1",
        "corrected_fault_classification_hash": corrected["artifact_hash"],
        "exploration_classification": corrected["exploration"],
        "entropy_classification": entropy["classification"],
        "entropy_operational": entropy["operational"],
        "temporal_classification": corrected["temporal"],
        "initial_scale_feasibility_quantified": bool(stage_a["rows"]),
        "scale_learning_rate_effects_quantified": bool(stage_c["rows"]),
        "held_out_mode": held_out["mode"],
        "phase_averaging_executed": held_out["phase_averaging_required"],
        "window_sensitivity_executed": all("window_sensitivity" in row for row in held_out["cells"]),
        "source_compatible_controller_identified": selected["selected"],
        "selected_controller": selected_row,
        "full_figure5a_acquisition_permitted": selected["full_figure5a_acquisition_permitted"],
        "plant_remained_frozen": all(not row["plant_modified"] for row in stage_a["cells"] + stage_b["cells"] + stage_c["cells"] + held_out["cells"]),
        "remaining_plausible_causes": corrected["unresolved_causes"] if not selected["selected"] else ["PUBLIC_INFORMATION_LIMITS_REMAIN"],
        "artifact_complete": True,
        "claim_supported": selected["selected"],
        "blocking_reasons": sorted(set(blockers)),
    }
    return write_artifact("root_cause_update", payload, "v9 Root-cause Update")


def write_next_commands() -> dict[str, Any]:
    commands = [
        "google-rl-v9-plan-stage-a --mode development",
        "google-rl-v9-run-stage-a --mode development --execute",
        "google-rl-v9-plan-stage-b --mode development",
        "google-rl-v9-run-stage-b --mode development --execute",
        "google-rl-v9-plan-stage-c --mode development",
        "google-rl-v9-run-stage-c --mode development --execute",
        "google-rl-v9-freeze-held-out-protocol",
        "google-rl-v9-run-held-out-validation --mode validation --execute",
        "google-rl-v9-select-controller",
        "google-rl-v9-report-root-cause-update",
    ]
    payload = {
        "schema_version": "google-pure-v9-next-commands.v1",
        "commands": commands,
        "reference_scale_runs_automatic": False,
        "certification_seeds_consumed": False,
        "blocking_reasons": ["RUN_DEVELOPMENT_AND_HELD_OUT_COMMANDS_EXPLICITLY"],
    }
    return write_artifact("next_commands", payload, "Exact v9 Next Commands", markdown_relative="next_commands.md")


def status() -> dict[str, Any]:
    states = {name: (artifact_root() / name).is_file() for name in REQUIRED}
    root_update = (artifact_root() / "root_cause_update.json").is_file()
    next_commands = (artifact_root() / "next_commands.md").is_file()
    selected = read_json(artifact_root() / "selected_controller_contract.json") if states["selected_controller_contract.json"] else {}
    payload = {
        "schema_version": "google-pure-v9-status.v1",
        "artifacts": states,
        "root_cause_update_present": root_update,
        "next_commands_present": next_commands,
        "implementation_complete": all(states.values()) and root_update and next_commands,
        "reference_evidence_complete": bool(selected.get("selected", False)),
        "full_figure5a_acquisition_permitted": bool(selected.get("full_figure5a_acquisition_permitted", False)),
        "certification_seeds_consumed": False,
        "blocking_reasons": [] if selected.get("selected") else ["NO_HELD_OUT_PASSING_CONTROLLER"],
    }
    return write_artifact("status", payload, "v9 Status")
