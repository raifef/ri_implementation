from __future__ import annotations

import ast
from copy import deepcopy
import inspect

import pytest

from hdfa_rl_suite.google_pure_source_exact.figure5a.contracts import (
    AcquisitionMode, Figure5aProtocol,
)
from hdfa_rl_suite.google_pure_v18.extended_fast import (
    build_continuation_state, classify_new_sigma_transition, extended_config,
    forbidden_auto_run_contract, repair_preacquisition_checkpoint_provenance,
    run_extended_fast_validation,
    validate_appended_state, verify_extension_preflight,
)
from hdfa_rl_suite.google_pure_v18.experiments import _approved_fast_transfer_for_slow
from hdfa_rl_suite.google_pure_v18.io import ROOT, canonical_hash, read_json


def _base_state() -> dict:
    optimizer = {
        "steps": 600,
        "config": {
            "mean_learning_rate": .32, "sigma_learning_rate": .08,
            "baseline_learning_rate": .08, "momentum": 0.0,
            "minimum_sigma": .002, "maximum_sigma": .8,
            "positivity_guard": "projected_gradient",
        },
        "mean_velocity": [1.0], "sigma_velocity": [2.0], "baseline_velocity": [3.0],
    }
    policy = {
        "schema_version": "direct-sigma-checkpoint.v1",
        "parameterization": "DIRECT_SIGMA_SOURCE_EXACT",
        "mean": [.4, -.2], "sigma": [.72, .8], "baseline": [.1],
        "policy_version": 600, "optimizer_state": optimizer,
        "rng_state": {"bit_generator": "PCG64", "state": {"state": 17, "inc": 19}},
    }
    protocol = Figure5aProtocol(AcquisitionMode.VALIDATION, 600, 8, 12000, 3)
    return {
        "schema_version": "figure5a-cell-checkpoint.v2",
        "protocol": {
            "mode": "validation", "epochs": 600, "candidates_per_epoch": 8,
            "qec_cycles_per_candidate": 12000, "circuit_rounds": 3,
        },
        "protocol_hash": protocol.protocol_hash, "plant_hash": "plant", "frequency": 1 / 150,
        "entropy_weight": .01, "seed": 83902, "dependency_hashes": {"x": "y"},
        "controller_hash": "controller", "v15_boundary": {"boundary": "frozen"},
        "epoch": 600, "active_batch": None, "policy": policy,
        "epoch_shards": [{"epoch": index, "path": f"base-{index}.json",
                          "record_hash": f"hash-{index}"} for index in range(600)],
        "candidate_boundaries_completed": 4800,
    }


def _extension_protocol() -> Figure5aProtocol:
    return Figure5aProtocol(AcquisitionMode.VALIDATION, 750, 8, 12000, 3)


def test_continuation_clones_terminal_runtime_without_any_reset():
    base = _base_state()
    original = deepcopy(base)
    extended = build_continuation_state(base, _extension_protocol())
    assert base == original
    assert extended["epoch"] == 600
    assert extended["protocol"]["epochs"] == 750
    assert extended["policy"] == base["policy"]
    assert extended["policy"]["optimizer_state"] == base["policy"]["optimizer_state"]
    assert extended["policy"]["sigma"] == base["policy"]["sigma"]
    assert extended["policy"]["rng_state"] == base["policy"]["rng_state"]
    assert extended["continuation_parent"]["policy_state_hash"] == canonical_hash(base["policy"])
    assert extended["continuation_parent"]["sigma_hash"] == canonical_hash(base["policy"]["sigma"])


def test_platform_sensitive_hash_repair_is_limited_to_untouched_initial_state():
    state = build_continuation_state(_base_state(), _extension_protocol())
    repaired = repair_preacquisition_checkpoint_provenance(
        {"continuation_checkpoint_initial_sha256": "prewrite-lf-hash"},
        state, "observed-windows-file-hash")
    assert repaired["continuation_checkpoint_initial_content_hash"] == canonical_hash(state)
    assert repaired["continuation_checkpoint_initial_sha256"] == "observed-windows-file-hash"
    assert repaired["checkpoint_created_before_acquisition"] is True
    assert repaired["platform_sensitive_initial_hash_repaired"] is True
    assert repaired["superseded_prewrite_hash_prediction"] == "prewrite-lf-hash"
    started = deepcopy(state)
    started["epoch"] = 601
    with pytest.raises(RuntimeError, match="forbidden after continuation acquisition begins"):
        repair_preacquisition_checkpoint_provenance({}, started, "observed")


def test_exact_150_epoch_append_preserves_600_epoch_prefix_and_has_no_duplicates():
    base = _base_state()
    extended = build_continuation_state(base, _extension_protocol())
    extended["epoch_shards"].extend(
        {"epoch": index, "path": f"extended-{index}.json", "record_hash": f"hash-{index}"}
        for index in range(600, 750))
    extended["epoch"] = 750
    extended["candidate_boundaries_completed"] = 6000
    extended["policy"]["policy_version"] = 750
    extended["policy"]["optimizer_state"]["steps"] = 750
    checks = validate_appended_state(base, extended)
    assert checks["exact_new_epoch_count"] is True
    assert checks["immutable_base_shard_prefix"] is True
    assert checks["no_duplicate_epoch_rows"] is True
    assert extended["epoch_shards"][:600] == base["epoch_shards"]


def test_duplicate_epoch_is_rejected_by_append_validator():
    base = _base_state()
    extended = build_continuation_state(base, _extension_protocol())
    extended["epoch_shards"].extend(
        {"epoch": index, "path": f"extended-{index}.json", "record_hash": f"hash-{index}"}
        for index in range(600, 750))
    extended["epoch_shards"][-1]["epoch"] = 748
    extended["epoch"] = 750
    extended["candidate_boundaries_completed"] = 6000
    extended["policy"]["policy_version"] = 750
    extended["policy"]["optimizer_state"]["steps"] = 750
    with pytest.raises(RuntimeError, match="invalid V18.1 appended checkpoint"):
        validate_appended_state(base, extended)


def test_frozen_tail_contains_period_three_to_four_and_threshold_is_unchanged():
    settings = extended_config()
    original_rule = read_json(ROOT / "artifacts/google_pure_v18/steady_state_rule.json")
    assert settings["frozen_stability"]["tail_transition_period_indices"] == [[2, 3], [3, 4]]
    assert settings["frozen_stability"]["sigma_relative_change_max"] == .15
    assert original_rule["rule"]["sigma_relative_change_max"] == .15
    assert original_rule["rule"]["fast_required_stable_transitions"] == 2


@pytest.mark.parametrize(
    ("current", "expected"),
    [
        ({"stable": True, "checks": {"sigma": True, "gain": True}}, "FAST_SIGMA_STABILIZED"),
        ({"stable": False, "checks": {"sigma": False, "gain": True}},
         "FAST_SIGMA_CONTINUES_SETTLING"),
        ({"stable": False, "checks": {"sigma": False, "gain": False}},
         "FAST_SIGMA_OSCILLATORY_OR_UNSTABLE"),
    ],
)
def test_new_sigma_transition_classification(current, expected):
    previous = {
        "from_period": 2, "to_period": 3, "sigma_start": .68, "sigma_end": .76,
        "sigma_relative_change": .118, "stable": True,
        "checks": {"sigma": True, "gain": True},
    }
    current = {
        "from_period": 3, "to_period": 4, "sigma_start": .76, "sigma_end": .80,
        "sigma_relative_change": .053, **current,
    }
    assert classify_new_sigma_transition([previous, current]) == expected


def test_completed_slow_lineage_closes_the_one_time_extension_preflight():
    with pytest.raises(RuntimeError, match="slow_not_previously_launched.*False"):
        verify_extension_preflight(validate_shards=False)
    readiness = read_json(ROOT / "artifacts/google_pure_v18/extended_fast/readiness_for_slow.json")
    provenance = read_json(
        ROOT / "artifacts/google_pure_v18/extended_fast/continuation_provenance.json")
    slow = read_json(ROOT / "artifacts/google_pure_v18/transfer_slow.json")
    assert readiness["pass"] is True
    assert readiness["READY_FOR_SLOW_TRANSFER_IDENTIFICATION"] is True
    assert provenance["continuation_complete"] is True
    assert provenance["previous_600_epochs_unchanged"] is True
    assert provenance["base_checkpoint_unchanged_after"] is True
    assert slow["fast_transfer_approval"]["source"] == "V18_1_EXTENDED_FAST"
    assert slow["stage_slow_intermediate_fast_ordering"]["pass"] is True


def test_slow_gate_consumes_hash_checked_extended_fast_approval():
    transfer, approval = _approved_fast_transfer_for_slow()
    assert approval["pass"] is True
    assert approval["source"] == "V18_1_EXTENDED_FAST"
    assert approval["base_fast_artifact_overwritten"] is False
    assert all(approval["checks"].values())
    assert transfer["steady_periodic_identification_accepted"] is True
    assert transfer["stage_ab_ordering"]["pass"] is True


def test_extension_command_has_no_forbidden_acquisition_calls():
    tree = ast.parse(inspect.getsource(run_extended_fast_validation))
    called = {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "run_cell" in called
    assert called.isdisjoint({
        "run_transfer_slow", "paired_acceptance_v2", "run_natural_drift",
        "run_figure5c", "run_reference", "run_source_budget",
    })
    contract = forbidden_auto_run_contract()
    assert contract["only_acquisition"] == "FAST_EPOCHS_600_TO_749"
    assert contract["slow_acquisition_launched"] is False
    assert contract["paired_acceptance_executed"] is False
    assert set(contract["forbidden_auto_runs"]) == {
        "slow", "paired_acceptance", "source_budget", "heldout", "reference",
        "natural_drift", "figure5c",
    }


def test_controller_hyperparameters_and_cli_are_exactly_the_small_patch():
    controller = extended_config()["controller"]
    assert controller == {
        "controller_hash": "6b24d03aeb0f16ed8c9ed855755ebdb6b5e7cc8a558b4dfd9a646dcd6bfe5aa2",
        "parameterization": "DIRECT_SIGMA_SOURCE_EXACT",
        "mean_learning_rate": .32, "sigma_learning_rate": .08,
        "initial_sigma": .15, "entropy_coefficient": .01, "ppo_clip": .2,
        "baseline_loss_weight": .2, "optimizer_changes_permitted": False,
    }
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    command = "hdfa-google-v18-run-extended-fast-validation"
    assert pyproject.count(command) == 1
