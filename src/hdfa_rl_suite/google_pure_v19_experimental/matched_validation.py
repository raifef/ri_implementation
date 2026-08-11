"""Matched-resolution follow-up to the preserved failed 3,000-cycle pilot."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hdfa_rl_suite.google_pure_source_exact.figure5a.contracts import (
    AcquisitionMode,
    Figure5aProtocol,
)
from hdfa_rl_suite.google_pure_source_exact.figure5a.validation import (
    build_plant,
    dependency_hashes,
)

from .acquisition import run_experimental_cell
from .controller import CONTROLLER_MODE, PARAMETERIZATION, SCALE_OBJECTIVE
from .dynamic_validation import (
    ACQUISITION_ORDER,
    FREQUENCY_ORDER,
    _analyse_cell,
    _boundary,
    _controller_spec,
    _frozen_source_paths,
    _hashes,
    _optimizer_config,
    _ordering,
    _source_config,
    _verify_frozen_source_branch,
    _write_report,
)
from .io import (
    ROOT,
    atomic_json,
    file_hash,
    nonfinal,
    read_json,
)


MATCHED_CONFIG_PATH = (
    ROOT / "configs/google_pure_v19/public_analogue_matched_dynamic_validation.json")
MATCHED_ARTIFACT_ROOT = (
    ROOT / "artifacts/google_pure_v19/experimental_public_analogue_matched")
PILOT_ROOT = ROOT / "artifacts/google_pure_v19/experimental_public_analogue"


def _settings() -> dict[str, Any]:
    value = read_json(MATCHED_CONFIG_PATH)
    frequencies = value.get("frequencies", {})
    checks = {
        "schema": value.get("schema_version") ==
            "google-pure-v19-public-analogue-matched-dynamic-validation.v1",
        "labels": list(frequencies) == list(FREQUENCY_ORDER),
        "frequencies": [float(frequencies[label]["frequency_per_epoch"])
                        for label in FREQUENCY_ORDER] == [0.001, 1/300, 1/150],
        "matched_cycles": all(int(row["qec_cycles_per_candidate"]) == 12000
                              for row in frequencies.values()),
        "bounded_candidates": all(int(row["candidates_per_epoch"]) == 8
                                  for row in frequencies.values()),
        "known_horizons": [int(frequencies[label]["epochs"]) for label in FREQUENCY_ORDER] ==
                          [3000, 900, 750],
        "no_heldout": value.get("heldout_seeds") == [],
        "no_auto_campaigns": value.get("automatic_campaigns_permitted") == [],
        "mean_unchanged": value.get("mean_hyperparameters_changed") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"invalid matched public-analogue settings: {checks}")
    for label, row in frequencies.items():
        period = round(1.0 / float(row["frequency_per_epoch"]))
        expected = period * (int(row["transient_periods"]) + int(row["analysis_periods"]))
        if int(row["epochs"]) != expected:
            raise RuntimeError(f"{label} matched horizon does not contain exact periods")
    return value


def build_matched_preflight() -> dict[str, Any]:
    settings = _settings()
    controller = _controller_spec()
    pilot_status_path = PILOT_ROOT / "status.json"
    pilot_report_path = PILOT_ROOT / "REPORT.md"
    if not pilot_status_path.is_file() or not pilot_report_path.is_file():
        raise RuntimeError("matched validation requires the preserved completed pilot")
    pilot = read_json(pilot_status_path)
    source_hashes = _hashes(_frozen_source_paths())
    gates = {
        "pilot_execution_complete": pilot.get("execution_complete") is True,
        "pilot_failure_preserved": pilot.get("pass") is False,
        "pilot_gain_order_passed": pilot.get("ordering", {}).get(
            "gain_point_ordering_pass") is True,
        "pilot_positive_I_gate_failed": pilot.get(
            "sampled_policy_I_positive_all_frequencies") is False,
        "pilot_frozen_branch_unchanged": pilot.get("frozen_source_branch_unchanged") is True,
        "matched_resolution_has_prior_basis": settings["resolution_basis"].startswith(
            "matched to the V18"),
        "matched_horizon_has_prior_basis": settings["horizon_basis"].startswith(
            "previously established V18"),
        "controller_distinct_from_parent": controller.controller_hash !=
            controller.frozen_parent_controller_hash,
        "no_heldout": settings["heldout_seeds"] == [],
        "no_auto_campaigns": settings["automatic_campaigns_permitted"] == [],
    }
    if not all(gates.values()):
        raise RuntimeError(f"matched public-analogue preflight failed: {gates}")
    value = nonfinal({
        "pass": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "controller": controller.identity_payload,
        "controller_hash": controller.controller_hash,
        "frozen_parent_controller_hash": controller.frozen_parent_controller_hash,
        "frozen_source_branch_hashes": source_hashes,
        "pilot_evidence": {
            "status_path": str(pilot_status_path.relative_to(ROOT)).replace("\\", "/"),
            "status_sha256": file_hash(pilot_status_path),
            "report_sha256": file_hash(pilot_report_path),
            "pilot_pass": False,
        },
        "experimental_code_hashes": {
            "controller": file_hash(Path(__file__).with_name("controller.py")),
            "acquisition": file_hash(Path(__file__).with_name("acquisition.py")),
            "matched_validation": file_hash(Path(__file__)),
            "protocol": file_hash(MATCHED_CONFIG_PATH),
        },
        "gates": gates,
        "acquisition_order": list(ACQUISITION_ORDER),
        "forbidden_auto_runs": settings["forbidden_auto_runs"],
        "forbidden_auto_runs_launched": [],
    })
    path = MATCHED_ARTIFACT_ROOT / "preflight_manifest.json"
    if path.is_file():
        previous = read_json(path)
        immutable = (
            "controller_hash", "frozen_parent_controller_hash", "frozen_source_branch_hashes",
            "pilot_evidence", "experimental_code_hashes", "acquisition_order")
        if any(previous.get(key) != value.get(key) for key in immutable):
            raise RuntimeError("matched public-analogue preflight changed after creation")
        return previous
    atomic_json(path, value)
    return value


def run_matched_three_frequency_validation() -> dict[str, Any]:
    status_path = MATCHED_ARTIFACT_ROOT / "status.json"
    if status_path.is_file():
        return read_json(status_path)
    settings = _settings()
    preflight = build_matched_preflight()
    controller = _controller_spec()
    source = _source_config()
    plant = build_plant(source)
    boundary = _boundary(plant)
    dependencies = {
        **dependency_hashes(ROOT, source),
        "experimental_controller_code": file_hash(Path(__file__).with_name("controller.py")),
        "experimental_acquisition_code": file_hash(Path(__file__).with_name("acquisition.py")),
        "experimental_protocol": file_hash(MATCHED_CONFIG_PATH),
    }
    analysed = {}
    for label in ACQUISITION_ORDER:
        row = settings["frequencies"][label]
        protocol = Figure5aProtocol(
            AcquisitionMode.VALIDATION, int(row["epochs"]),
            int(row["candidates_per_epoch"]), int(row["qec_cycles_per_candidate"]),
            int(source["plant"]["circuit_rounds"]))
        checkpoint = MATCHED_ARTIFACT_ROOT / "acquisition" / label / "checkpoint.json"
        cell = run_experimental_cell(
            protocol=protocol, plant=plant,
            frequency=float(row["frequency_per_epoch"]), seed=int(row["seed"]),
            optimizer_config=_optimizer_config(controller), controller=controller,
            checkpoint_path=checkpoint, dependency_hashes=dependencies, boundary=boundary,
            resume=checkpoint.is_file())
        analysed[label] = _analyse_cell(
            label, cell, row, controller, protocol_settings=settings,
            artifact_root=MATCHED_ARTIFACT_ROOT)
    ordering = _ordering(analysed, protocol_settings=settings)
    frozen_check = _verify_frozen_source_branch(preflight)
    rows = [analysed[label] for label in FREQUENCY_ORDER]
    sampled_positive = all(
        row["stream_decomposition"]["sampled_policy_I_positive"] for row in rows)
    result = nonfinal({
        "pass": bool(
            all(row["pass"] for row in rows) and ordering["pass"] and sampled_positive and
            frozen_check["pass"]),
        "execution_complete": True,
        "controller_mode": CONTROLLER_MODE,
        "controller_hash": controller.controller_hash,
        "frozen_parent_controller_hash": controller.frozen_parent_controller_hash,
        "parameterization": PARAMETERIZATION,
        "scale_objective": SCALE_OBJECTIVE,
        "rows": rows,
        "ordering": ordering,
        "sampled_policy_I_positive_all_frequencies": sampled_positive,
        "frozen_source_branch_unchanged": frozen_check["pass"],
        "frozen_source_branch_hash_audit": frozen_check,
        "mean_stochastic_decomposition_retained": True,
        "sigma_diagnostics_retained": True,
        "mean_hyperparameters_changed": False,
        "only_scale_objective_changed": True,
        "between_seed_uncertainty_available": False,
        "campaign_scope": "MATCHED_RESOLUTION_SINGLE_SEED_THREE_FREQUENCY_DEVELOPMENT_VALIDATION",
        "prior_failed_pilot_preserved": True,
        "source_budget_fraction_per_epoch": 0.024,
        "forbidden_auto_runs": settings["forbidden_auto_runs"],
        "forbidden_auto_runs_launched": [],
    })
    atomic_json(MATCHED_ARTIFACT_ROOT / "three_frequency_validation.json", result)
    atomic_json(status_path, result)
    _write_report(result, artifact_root=MATCHED_ARTIFACT_ROOT)
    return result
