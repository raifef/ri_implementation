"""V18.1 one-period continuation of the frozen V18 fast identification run.

Nothing in this module runs at import time.  The sole public execution function
copies the terminal 600-epoch state into a separately provenanced checkpoint and
allows the production acquisition loop to append epochs 600 through 749.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from hdfa_rl_suite.google_pure_source_exact.figure5a.acquisition import run_cell
from hdfa_rl_suite.google_pure_source_exact.figure5a.contracts import AcquisitionMode, Figure5aProtocol
from hdfa_rl_suite.google_pure_source_exact.figure5a.validation import build_plant, dependency_hashes

from .contracts import NONFINAL, nonfinal
from .experiments import (
    STREAMS, _analyse_transfer, _boundary, _frozen, _optimizer_config,
    _ordering_gate, _source_config,
)
from .imports import verify_import_manifest
from .io import ARTIFACT_ROOT, ROOT, atomic_json, atomic_text, canonical_hash, file_hash, read_json


EXTENDED_ROOT = ARTIFACT_ROOT / "extended_fast"
CONFIG_PATH = ROOT / "configs/google_pure_v18/extended_fast.json"
BASE_CHECKPOINT = ARTIFACT_ROOT / "acquisition/fast/checkpoint.json"
BASE_TRANSFER = ARTIFACT_ROOT / "transfer_fast.json"
INTERMEDIATE_TRANSFER = ARTIFACT_ROOT / "transfer_intermediate.json"
EXTENDED_CHECKPOINT = EXTENDED_ROOT / "checkpoint.json"
PROVENANCE_PATH = EXTENDED_ROOT / "continuation_provenance.json"
PROFILE = "V18_EXTENDED_TRANSFER_IDENTIFICATION_ONLY"


def extended_config() -> dict[str, Any]:
    value = read_json(CONFIG_PATH)
    identities = {
        "profile": value.get("profile") == PROFILE,
        "frequency": value.get("frequency_per_epoch") == 1 / 150,
        "period": value.get("period_epochs") == 150,
        "base_epochs": value.get("base_epochs") == 600,
        "extra_epochs": value.get("extra_epochs") == 150,
        "total_epochs": value.get("total_epochs") == 750,
        "period_arithmetic": value.get("total_epochs") - value.get("base_epochs") ==
                             value.get("extra_epochs"),
        "analysis_periods": value.get("extended_analysis_periods") == 4,
        "optimizer_immutable": value.get("controller", {}).get(
            "optimizer_changes_permitted") is False,
        "sigma_threshold": value.get("frozen_stability", {}).get(
            "sigma_relative_change_max") == .15,
        "tail_transitions": value.get("frozen_stability", {}).get(
            "tail_transition_period_indices") == [[2, 3], [3, 4]],
    }
    if not all(identities.values()):
        raise RuntimeError(f"invalid V18.1 extension contract: {identities}")
    return value


def _policy_identity(state: Mapping[str, Any]) -> dict[str, Any]:
    policy = state["policy"]
    return {
        "policy_state_hash": canonical_hash(policy),
        "optimizer_state_hash": canonical_hash(policy["optimizer_state"]),
        "sigma_hash": canonical_hash(policy["sigma"]),
        "rng_state_hash": canonical_hash(policy["rng_state"]),
        "mean_hash": canonical_hash(policy["mean"]),
        "baseline_hash": canonical_hash(policy["baseline"]),
        "policy_version": int(policy["policy_version"]),
        "optimizer_steps": int(policy["optimizer_state"]["steps"]),
        "sigma_median": float(np.median(np.asarray(policy["sigma"], dtype=float))),
    }


def _frozen_input_hashes(settings: Mapping[str, Any]) -> dict[str, str]:
    hashes = {}
    for role, declaration in settings["frozen_inputs"].items():
        path = ROOT / declaration["path"]
        if not path.is_file():
            raise RuntimeError(f"missing frozen V18.1 input: {declaration['path']}")
        observed = file_hash(path)
        if observed != declaration["sha256"]:
            raise RuntimeError(
                f"frozen V18.1 input changed: {role}: expected {declaration['sha256']}, observed {observed}")
        hashes[role] = observed
    return hashes


def _validate_shards(state: Mapping[str, Any], *, expected_epochs: int,
                     validate_payloads: bool) -> None:
    rows = list(state["epoch_shards"])
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != list(range(expected_epochs)) or len(set(epochs)) != expected_epochs:
        raise RuntimeError("checkpoint contains missing, duplicate, or reordered epochs")
    if validate_payloads:
        for row in rows:
            path = Path(row["path"])
            if not path.is_file():
                raise RuntimeError(f"missing epoch shard: {path}")
            payload = read_json(path)
            if (payload.get("record_hash") != row["record_hash"] or
                    canonical_hash(payload.get("record")) != row["record_hash"] or
                    int(payload["record"]["epoch"]) != int(row["epoch"])):
                raise RuntimeError(f"corrupt or mislabelled epoch shard: {path}")


def _validate_base_state(state: Mapping[str, Any], settings: Mapping[str, Any],
                         *, validate_payloads: bool = True) -> None:
    protocol = state.get("protocol", {})
    controller = settings["controller"]
    frozen = _frozen()
    optimizer_config = state.get("policy", {}).get("optimizer_state", {}).get("config", {})
    checks = {
        "terminal_epoch": state.get("epoch") == settings["base_epochs"],
        "inactive_batch": state.get("active_batch") is None,
        "candidate_boundaries": state.get("candidate_boundaries_completed") ==
                                settings["base_epochs"] * settings["candidates_per_epoch"],
        "protocol_epochs": protocol.get("epochs") == settings["base_epochs"],
        "protocol_candidates": protocol.get("candidates_per_epoch") == settings["candidates_per_epoch"],
        "protocol_cycles": protocol.get("qec_cycles_per_candidate") ==
                           settings["qec_cycles_per_candidate"],
        "protocol_rounds": protocol.get("circuit_rounds") == settings["circuit_rounds"],
        "frequency": state.get("frequency") == settings["frequency_per_epoch"],
        "seed": state.get("seed") == settings["seed"],
        "controller_hash": state.get("controller_hash") == controller["controller_hash"],
        "entropy": state.get("entropy_weight") == controller["entropy_coefficient"],
        "policy_parameterization": state.get("policy", {}).get("parameterization") ==
                                   controller["parameterization"],
        "policy_version": state.get("policy", {}).get("policy_version") == settings["base_epochs"],
        "optimizer_steps": state.get("policy", {}).get("optimizer_state", {}).get("steps") ==
                           settings["base_epochs"],
        "mean_lr": optimizer_config.get("mean_learning_rate") == controller["mean_learning_rate"] ==
                   frozen["mean_learning_rate"],
        "sigma_lr": optimizer_config.get("sigma_learning_rate") == controller["sigma_learning_rate"] ==
                    frozen["sigma_learning_rate"],
        "initial_sigma_contract": controller["initial_sigma"] == frozen["initial_sigma"],
        "frozen_entropy_contract": controller["entropy_coefficient"] == frozen["entropy_coefficient"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"base V18 fast checkpoint is not extendable: {checks}")
    _validate_shards(state, expected_epochs=int(settings["base_epochs"]),
                     validate_payloads=validate_payloads)


def build_continuation_state(base_state: Mapping[str, Any],
                             protocol: Figure5aProtocol) -> dict[str, Any]:
    """Pure state transition used by synthetic tests and the production preflight."""
    state = deepcopy(dict(base_state))
    if int(state["epoch"]) != 600 or state.get("active_batch") is not None:
        raise ValueError("continuation requires the terminal, inactive 600-epoch state")
    if protocol.epochs != 750 or protocol.candidates_per_epoch != 8:
        raise ValueError("V18.1 continuation protocol must be exactly 750 epochs x 8 candidates")
    parent_identity = _policy_identity(state)
    state["protocol"] = asdict(protocol)
    state["protocol_hash"] = protocol.protocol_hash
    state["continuation_parent"] = {
        "base_epoch": 600,
        "base_candidate_boundaries": 4800,
        "base_epoch_shard_manifest_hash": canonical_hash(state["epoch_shards"]),
        **parent_identity,
    }
    return state


def validate_appended_state(base_state: Mapping[str, Any], extended_state: Mapping[str, Any],
                            *, base_epochs: int = 600, total_epochs: int = 750,
                            candidates_per_epoch: int = 8,
                            validate_payloads: bool = False) -> dict[str, Any]:
    """Fail closed unless the extension is an immutable-prefix, exact-length append."""
    base_shards = list(base_state["epoch_shards"])
    extended_shards = list(extended_state["epoch_shards"])
    checks = {
        "terminal_epoch": int(extended_state["epoch"]) == total_epochs,
        "inactive_batch": extended_state.get("active_batch") is None,
        "exact_new_epoch_count": len(extended_shards) - len(base_shards) == total_epochs - base_epochs,
        "immutable_base_shard_prefix": extended_shards[:base_epochs] == base_shards,
        "no_duplicate_epoch_rows": [int(row["epoch"]) for row in extended_shards] ==
                                   list(range(total_epochs)),
        "candidate_boundary_count": int(extended_state["candidate_boundaries_completed"]) ==
                                    total_epochs * candidates_per_epoch,
        "policy_updates_appended": int(extended_state["policy"]["policy_version"]) == total_epochs,
        "optimizer_steps_appended": int(
            extended_state["policy"]["optimizer_state"]["steps"]) == total_epochs,
        "parent_policy_identity_retained": extended_state.get("continuation_parent", {}).get(
            "policy_state_hash") == canonical_hash(base_state["policy"]),
        "parent_sigma_identity_retained": extended_state.get("continuation_parent", {}).get(
            "sigma_hash") == canonical_hash(base_state["policy"]["sigma"]),
        "parent_rng_identity_retained": extended_state.get("continuation_parent", {}).get(
            "rng_state_hash") == canonical_hash(base_state["policy"]["rng_state"]),
        "parent_optimizer_identity_retained": extended_state.get("continuation_parent", {}).get(
            "optimizer_state_hash") == canonical_hash(base_state["policy"]["optimizer_state"]),
    }
    if validate_payloads and all(checks[key] for key in (
            "terminal_epoch", "exact_new_epoch_count", "immutable_base_shard_prefix")):
        _validate_shards(extended_state, expected_epochs=total_epochs, validate_payloads=True)
        extension_root = (EXTENDED_CHECKPOINT.parent / f"{EXTENDED_CHECKPOINT.stem}_epochs").resolve()
        checks["new_shards_owned_by_extension"] = all(
            Path(row["path"]).resolve().parent == extension_root
            for row in extended_shards[base_epochs:])
    if not all(checks.values()):
        raise RuntimeError(f"invalid V18.1 appended checkpoint: {checks}")
    return checks


def verify_extension_preflight(*, validate_shards: bool = True) -> dict[str, Any]:
    settings = extended_config()
    verify_import_manifest()
    frozen_hashes = _frozen_input_hashes(settings)
    base_state = read_json(BASE_CHECKPOINT)
    _validate_base_state(base_state, settings, validate_payloads=validate_shards)
    base_transfer = read_json(BASE_TRANSFER)
    intermediate = read_json(INTERMEDIATE_TRANSFER)
    steady = read_json(ARTIFACT_ROOT / "steady_state_rule.json")
    base_transitions = {
        (int(row["from_period"]), int(row["to_period"])): row
        for row in base_transfer["steady_state_diagnostic"]["transitions"]
    }
    checks = {
        "base_checkpoint_hash_matches_transfer": base_transfer.get("checkpoint_sha256") ==
                                                 frozen_hashes["base_checkpoint"],
        "base_fast_direct_transfer_identifiable": base_transfer.get(
            "direct_mean_transfer_identifiable") is True,
        "base_fast_steady_gate_failed": base_transfer.get(
            "steady_periodic_identification_accepted") is False,
        "intermediate_steady_gate_passed": intermediate.get(
            "steady_periodic_identification_accepted") is True,
        "existing_ordering_passed": base_transfer.get("stage_ab_ordering", {}).get("pass") is True,
        "observed_period_1_to_2_sigma_change": bool(np.isclose(
            base_transitions[(1, 2)]["sigma_relative_change"], .21910942340564843,
            rtol=0, atol=1e-12)),
        "observed_period_2_to_3_sigma_change": bool(np.isclose(
            base_transitions[(2, 3)]["sigma_relative_change"], .11813399757315973,
            rtol=0, atol=1e-12)),
        "frozen_sigma_threshold": steady["rule"]["sigma_relative_change_max"] ==
                                  settings["frozen_stability"]["sigma_relative_change_max"] == .15,
        "frozen_tail_requirement": steady["rule"]["fast_required_stable_transitions"] == 2,
        "slow_not_previously_launched": not (
            ARTIFACT_ROOT / "acquisition/slow/checkpoint.json").exists(),
        "paired_acceptance_not_executed": read_json(
            ARTIFACT_ROOT / "paired_acceptance_readiness.json")["paired_acceptance_executed"] is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"V18.1 continuation preflight failed: {checks}")
    return {
        "pass": True, "checks": checks, "frozen_input_hashes": frozen_hashes,
        "base_policy_identity": _policy_identity(base_state),
        "base_epoch_shard_manifest_hash": canonical_hash(base_state["epoch_shards"]),
    }


def _extension_protocol(settings: Mapping[str, Any]) -> Figure5aProtocol:
    return Figure5aProtocol(
        AcquisitionMode.VALIDATION, int(settings["total_epochs"]),
        int(settings["candidates_per_epoch"]), int(settings["qec_cycles_per_candidate"]),
        int(settings["circuit_rounds"]))


def _initial_provenance(settings: Mapping[str, Any], preflight: Mapping[str, Any],
                        state: Mapping[str, Any]) -> dict[str, Any]:
    return nonfinal({
        "pass": True, "profile": PROFILE, "continuation_complete": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_checkpoint_path": str(BASE_CHECKPOINT.relative_to(ROOT)).replace("\\", "/"),
        "continuation_checkpoint_path": str(EXTENDED_CHECKPOINT.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_before_sha256": preflight["frozen_input_hashes"]["base_checkpoint"],
        "continuation_checkpoint_initial_content_hash": canonical_hash(state),
        "continuation_checkpoint_initial_sha256": None,
        "checkpoint_created_before_acquisition": False,
        "checkpoint_after_sha256": None,
        "base_checkpoint_unchanged_after": None,
        "base_epochs": settings["base_epochs"], "extra_epochs": settings["extra_epochs"],
        "target_total_epochs": settings["total_epochs"],
        "fresh_continuation_epoch_count": None,
        "inherited_base_epoch_shard_count": settings["base_epochs"],
        "base_epoch_shard_manifest_hash": preflight["base_epoch_shard_manifest_hash"],
        "base_policy_identity": preflight["base_policy_identity"],
        "continuation_initial_policy_identity": _policy_identity(state),
        "optimizer_reset": False, "sigma_reset": False, "rng_reset": False,
        "controller_hyperparameters_changed": False,
        "random_stream_lineage": "SAME_POLICY_RNG_STATE_AND_BASE_SEED_WITH_NEW_EPOCH_INDICES_600_TO_749",
        "frozen_input_hashes": preflight["frozen_input_hashes"],
        "extension_config_sha256": file_hash(CONFIG_PATH),
        "source_budget_profile": PROFILE,
        "paired_acceptance_executed": False,
        "slow_acquisition_launched": False,
        "forbidden_auto_runs_launched": [],
    })


def repair_preacquisition_checkpoint_provenance(
        provenance: Mapping[str, Any], state: Mapping[str, Any],
        observed_checkpoint_sha256: str) -> dict[str, Any]:
    """Repair only an untouched 600-epoch initialization record.

    The initial V18.1 implementation predicted a byte hash before writing the
    JSON file.  Windows newline translation made that prediction platform
    dependent.  Canonical content identity is platform independent; the actual
    file SHA is observed only after the atomic write and before acquisition.
    """
    if (int(state.get("epoch", -1)) != 600 or state.get("active_batch") is not None or
            len(state.get("epoch_shards", [])) != 600 or
            int(state.get("candidate_boundaries_completed", -1)) != 4800 or
            int(state.get("policy", {}).get("policy_version", -1)) != 600 or
            int(state.get("policy", {}).get("optimizer_state", {}).get("steps", -1)) != 600):
        raise RuntimeError("checkpoint-hash repair is forbidden after continuation acquisition begins")
    repaired = deepcopy(dict(provenance))
    prior_prediction = repaired.get("continuation_checkpoint_initial_sha256")
    repaired.update({
        "continuation_checkpoint_initial_content_hash": canonical_hash(state),
        "continuation_checkpoint_initial_sha256": str(observed_checkpoint_sha256),
        "checkpoint_created_before_acquisition": True,
        "platform_sensitive_initial_hash_repaired": bool(
            prior_prediction not in (None, observed_checkpoint_sha256)),
        "superseded_prewrite_hash_prediction": (
            prior_prediction if prior_prediction not in (None, observed_checkpoint_sha256) else None),
        "hash_repair_reason": (
            "PREWRITE_TEXT_NEWLINE_TRANSLATION_REPLACED_BY_CANONICAL_CONTENT_HASH_AND_OBSERVED_FILE_SHA256"
            if prior_prediction not in (None, observed_checkpoint_sha256) else None),
    })
    return repaired


def prepare_continuation_checkpoint() -> tuple[dict[str, Any], dict[str, Any]]:
    settings = extended_config()
    preflight = verify_extension_preflight(validate_shards=True)
    base_state = read_json(BASE_CHECKPOINT)
    if EXTENDED_CHECKPOINT.is_file():
        if not PROVENANCE_PATH.is_file():
            raise RuntimeError("extension checkpoint exists without continuation provenance")
        provenance = read_json(PROVENANCE_PATH)
        if (provenance.get("frozen_input_hashes") != preflight["frozen_input_hashes"] or
                provenance.get("extension_config_sha256") != file_hash(CONFIG_PATH)):
            raise RuntimeError("extension provenance no longer matches frozen inputs")
        state = read_json(EXTENDED_CHECKPOINT)
        if (state.get("continuation_parent", {}).get("base_epoch_shard_manifest_hash") !=
                preflight["base_epoch_shard_manifest_hash"] or
                state.get("protocol_hash") != _extension_protocol(settings).protocol_hash):
            raise RuntimeError("extension checkpoint lost its frozen parent or protocol identity")
        if list(state["epoch_shards"])[:settings["base_epochs"]] != list(base_state["epoch_shards"]):
            raise RuntimeError("extension checkpoint changed the inherited 600-epoch prefix")
        epoch = int(state["epoch"])
        if not settings["base_epochs"] <= epoch <= settings["total_epochs"]:
            raise RuntimeError("extension checkpoint epoch is outside the authorized window")
        active = state.get("active_batch")
        next_candidate = 0 if active is None else int(active["next_candidate"])
        progress_checks = {
            "completed_epoch_shards": len(state["epoch_shards"]) == epoch,
            "completed_epoch_order": [int(row["epoch"]) for row in state["epoch_shards"]] ==
                                     list(range(epoch)),
            "candidate_boundary_progress": int(state["candidate_boundaries_completed"]) ==
                                           epoch * settings["candidates_per_epoch"] + next_candidate,
            "policy_version": int(state["policy"]["policy_version"]) == epoch,
            "optimizer_steps": int(state["policy"]["optimizer_state"]["steps"]) == epoch,
            "candidate_cursor": 0 <= next_candidate <= settings["candidates_per_epoch"],
        }
        if not all(progress_checks.values()):
            raise RuntimeError(f"extension checkpoint is not exactly resumable: {progress_checks}")
        observed_sha256 = file_hash(EXTENDED_CHECKPOINT)
        expected_content_hash = provenance.get("continuation_checkpoint_initial_content_hash")
        expected_file_hash = provenance.get("continuation_checkpoint_initial_sha256")
        if epoch == settings["base_epochs"] and active is None and (
                expected_content_hash != canonical_hash(state) or
                expected_file_hash != observed_sha256 or
                provenance.get("checkpoint_created_before_acquisition") is not True):
            provenance = repair_preacquisition_checkpoint_provenance(
                provenance, state, observed_sha256)
            atomic_json(PROVENANCE_PATH, provenance)
        return state, provenance
    if PROVENANCE_PATH.is_file():
        raise RuntimeError("continuation provenance exists without its extension checkpoint")
    state = build_continuation_state(base_state, _extension_protocol(settings))
    provenance = _initial_provenance(settings, preflight, state)
    atomic_json(PROVENANCE_PATH, provenance)
    atomic_json(EXTENDED_CHECKPOINT, state)
    observed = read_json(EXTENDED_CHECKPOINT)
    if canonical_hash(observed) != provenance["continuation_checkpoint_initial_content_hash"]:
        raise RuntimeError("serialized continuation checkpoint changed canonical content")
    provenance = repair_preacquisition_checkpoint_provenance(
        provenance, observed, file_hash(EXTENDED_CHECKPOINT))
    atomic_json(PROVENANCE_PATH, provenance)
    return state, provenance


def _transition_diagnostics(transfer: Mapping[str, Any]) -> list[dict[str, Any]]:
    periods = {int(row["period_index"]): row for row in transfer["period_diagnostics"]}
    transitions = []
    for row in transfer["steady_state_diagnostic"]["transitions"]:
        left = periods[int(row["from_period"])]
        right = periods[int(row["to_period"])]
        transitions.append({
            "from_period": row["from_period"], "to_period": row["to_period"],
            "sigma_start": left["sigma_x_median"], "sigma_end": right["sigma_x_median"],
            "sigma_relative_change": row["sigma_relative_change"],
            "period_average_sigma": .5 * (left["sigma_x_median"] + right["sigma_x_median"]),
            "native_sigma": {
                "start": left["sigma_u_median"], "end": right["sigma_u_median"],
                "average": .5 * (left["sigma_u_median"] + right["sigma_u_median"]),
            },
            "reward_sigma_gradient_norm": {
                "start": left["reward_sigma_gradient_norm_median"],
                "end": right["reward_sigma_gradient_norm_median"],
            },
            "entropy_sigma_gradient_norm": {
                "start": left["entropy_sigma_gradient_norm_median"],
                "end": right["entropy_sigma_gradient_norm_median"],
            },
            "sigma_update_norm": {
                "start": left["sigma_update_norm_median"],
                "end": right["sigma_update_norm_median"],
            },
            "scale_floor_occupancy": {
                "start": left["scale_floor_occupancy"], "end": right["scale_floor_occupancy"],
            },
            "scale_ceiling_occupancy": {
                "start": left["scale_ceiling_occupancy"], "end": right["scale_ceiling_occupancy"],
            },
            "candidate_clipping_fraction": {
                "start": left["candidate_clipping_fraction"],
                "end": right["candidate_clipping_fraction"],
            },
            "exploration_damage": {
                "start": left["exploration_damage"], "end": right["exploration_damage"],
                "change": right["exploration_damage"] - left["exploration_damage"],
            },
            "gain_relative_change": row["gain_relative_change"],
            "phase_change_radians": row["phase_change_radians"],
            "checks": row["checks"], "stable": row["stable"],
        })
    return transitions


def classify_new_sigma_transition(transitions: list[Mapping[str, Any]]) -> str:
    lookup = {(int(row["from_period"]), int(row["to_period"])): row for row in transitions}
    previous = lookup[(2, 3)]
    current = lookup[(3, 4)]
    if current["stable"] and current["checks"]["sigma"]:
        return "FAST_SIGMA_STABILIZED"
    previous_direction = math.copysign(1.0, previous["sigma_end"] - previous["sigma_start"])
    current_direction = math.copysign(1.0, current["sigma_end"] - current["sigma_start"])
    other_checks = all(value for key, value in current["checks"].items() if key != "sigma")
    if (current_direction == previous_direction and
            current["sigma_relative_change"] < previous["sigma_relative_change"] and other_checks):
        return "FAST_SIGMA_CONTINUES_SETTLING"
    return "FAST_SIGMA_OSCILLATORY_OR_UNSTABLE"


def _mean_transfer_diagnostics(transfer: Mapping[str, Any]) -> dict[str, Any]:
    regression = transfer["mean_transfer_regression"]
    bootstrap = transfer["bootstrap_uncertainty"]
    transitions = transfer["steady_state_diagnostic"]["transitions"]
    return {
        "gain": regression["gain"], "phase_lag_radians": regression["phase_lag_radians"],
        "gain_confidence_interval_95": bootstrap["gain_confidence_interval_95"],
        "phase_lag_confidence_interval_95": bootstrap["phase_lag_confidence_interval_95"],
        "period_to_period_gain_change": [{
            "from_period": row["from_period"], "to_period": row["to_period"],
            "relative_change": row["gain_relative_change"],
        } for row in transitions],
        "period_to_period_phase_change": [{
            "from_period": row["from_period"], "to_period": row["to_period"],
            "absolute_change_radians": row["phase_change_radians"],
        } for row in transitions],
        "complete_periods_only": True,
        "normalized_performance_used_as_transfer_proxy": False,
    }


def _extended_decomposition(transfer: Mapping[str, Any]) -> dict[str, Any]:
    current = transfer["stream_decomposition"]
    previous = read_json(BASE_TRANSFER)["stream_decomposition"]
    fields = ("C_fixed", "C_oracle", "C_mean", "C_stochastic", "I_mean",
              "I_stochastic", "exploration_damage")
    changes = {key: current[key] - previous[key] for key in fields}
    previous_epochs = read_json(BASE_TRANSFER)["analysis_epoch_window"][1] - read_json(
        BASE_TRANSFER)["analysis_epoch_window"][0]
    current_epochs = transfer["analysis_epoch_window"][1] - transfer["analysis_epoch_window"][0]
    return nonfinal({
        "pass": current["denominator_resolved"], "profile": PROFILE,
        "extended_fast": {key: current[key] for key in fields},
        "previous_fast": {key: previous[key] for key in fields},
        "change_from_previous_fast": changes,
        "analysis_epochs": {"previous": previous_epochs, "extended": current_epochs},
        "per_epoch_count_rates": {
            "previous": {key: previous[key] / previous_epochs for key in fields[:4]},
            "extended": {key: current[key] / current_epochs for key in fields[:4]},
            "change": {key: current[key] / current_epochs - previous[key] / previous_epochs
                       for key in fields[:4]},
        },
        "mean_and_stochastic_streams_separate": True,
        "diagnostic_only": True, "optimizer_changed": False,
    })


def _step_fit_note() -> dict[str, Any]:
    step = read_json(ROOT / "artifacts/google_pure_v17/step_transfer_identification.json")
    return {
        "constrained_source_style_step_tau_approx_epochs": step["measured_v16_tau_epochs"],
        "current_grid_refit_fixed_gain_one_tau_epochs": step["fixed_gain_one"]["tau_epochs"],
        "free_gain_transfer_fit": {
            "K": step["free_gain_delay_tau"]["gain"],
            "tau_epochs": step["free_gain_delay_tau"]["tau_epochs"],
            "delay_epochs": step["free_gain_delay_tau"]["delay_epochs"],
        },
        "source_comparable_step_fit_convention": "UNRESOLVED",
        "tau_approximately_133_is_paper_equivalent": False,
        "step_experiment_rerun": False,
    }


def _write_outputs(transfer: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    transitions = _transition_diagnostics(transfer)
    requested_tail = [[row["from_period"], row["to_period"]] for row in transitions[-2:]]
    if requested_tail != [[2, 3], [3, 4]]:
        raise RuntimeError(f"extended stability gate used the wrong tail: {requested_tail}")
    sigma_classification = classify_new_sigma_transition(transitions)
    sigma = nonfinal({
        "pass": transfer["steady_periodic_identification_accepted"], "profile": PROFILE,
        "classification": sigma_classification,
        "frozen_sigma_relative_change_max": .15,
        "frozen_required_stable_tail_transitions": 2,
        "evaluated_tail_transitions": transitions[-2:],
        "all_post_transient_transitions": transitions,
        "sigma_learning_rate_changed": False, "entropy_changed": False,
        "optimizer_changed": False,
    })
    transfer["profile"] = PROFILE
    transfer["mean_transfer_diagnostics"] = _mean_transfer_diagnostics(transfer)
    transfer["stage_ab_ordering"] = _ordering_gate(read_json(INTERMEDIATE_TRANSFER), transfer)
    transfer["pass"] = bool(
        transfer["steady_periodic_identification_accepted"] and
        transfer["direct_mean_transfer_identifiable"] and transfer["stage_ab_ordering"]["pass"])
    transfer["classification"] = (
        "EXTENDED_FAST_STEADY_PERIODIC_IDENTIFICATION_ACCEPTED" if transfer["pass"] else
        "EXTENDED_FAST_STEADY_PERIODIC_IDENTIFICATION_NOT_ACCEPTED")
    transfer["fresh_continuation_epochs"] = 150
    transfer["inherited_base_epoch_shards"] = 600
    transfer["optimizer_changed"] = False
    transfer["entropy_changed"] = False
    transfer["sigma_learning_rate_changed"] = False
    decomposition = _extended_decomposition(transfer)
    ready = bool(transfer["pass"] and sigma["pass"])
    readiness = nonfinal({
        "pass": ready, "READY_FOR_SLOW_TRANSFER_IDENTIFICATION": ready,
        "classification": ("READY_FOR_SLOW_TRANSFER_IDENTIFICATION" if ready else
                           "NOT_READY_FOR_SLOW_TRANSFER_IDENTIFICATION"),
        "gates": {
            "direct_fast_mean_transfer_identifiable": transfer["direct_mean_transfer_identifiable"],
            "fast_steady_periodic_gate": transfer["steady_periodic_identification_accepted"],
            "intermediate_fast_ordering": transfer["stage_ab_ordering"]["pass"],
            "exact_150_epoch_append": provenance["fresh_continuation_epoch_count"] == 150,
            "immutable_600_epoch_prefix": provenance["previous_600_epochs_unchanged"],
        },
        "sigma_classification": sigma_classification,
        "slow_acquisition_launched": False,
        "paired_acceptance_executed": False,
        "source_budget_auto_launched": False,
        "heldout_auto_launched": False, "reference_auto_launched": False,
        "natural_drift_auto_launched": False, "figure5c_executed": False,
    })
    step_note = _step_fit_note()
    status = nonfinal({
        "pass": True, "execution_complete": True, "profile": PROFILE,
        "READY_FOR_SLOW_TRANSFER_IDENTIFICATION": ready,
        "extended_fast_transfer_accepted": transfer["pass"],
        "sigma_classification": sigma_classification,
        "paired_acceptance_executed": False,
        "original_v18_artifacts_overwritten": False,
        "step_fit_terminology": step_note,
        "forbidden_auto_runs_launched": [],
    })
    atomic_json(EXTENDED_ROOT / "transfer_fast_extended.json", transfer)
    atomic_json(EXTENDED_ROOT / "sigma_stability_extended.json", sigma)
    atomic_json(EXTENDED_ROOT / "mean_stochastic_decomposition_extended.json", decomposition)
    atomic_json(EXTENDED_ROOT / "readiness_for_slow.json", readiness)
    atomic_json(EXTENDED_ROOT / "status.json", status)
    report_lines = [
        "# V18.1 extended fast validation", "",
        f"Profile: `{PROFILE}`.", "",
        f"Extended fast classification: **{transfer['classification']}**.",
        f"Sigma classification: **{sigma_classification}**.",
        f"Ready for slow transfer identification: **{str(ready).lower()}**.", "",
        "## Mean transfer", "",
        f"- Gain: {transfer['mean_transfer_diagnostics']['gain']:.6f}.",
        f"- Phase lag: {transfer['mean_transfer_diagnostics']['phase_lag_radians']:.6f} rad.",
        f"- Intermediate-greater-than-fast gain and lower phase lag: {transfer['stage_ab_ordering']['pass']}.",
        "- Only four complete post-transient fast periods were used.", "",
        "## Sigma and exploration", "",
        f"- Frozen sigma relative-change threshold: {sigma['frozen_sigma_relative_change_max']:.2f}.",
        "- The decision uses period 2→3 and period 3→4; no threshold, sigma LR, or entropy was changed.",
        f"- I_mean: {decomposition['extended_fast']['I_mean']:.6f}.",
        f"- I_stochastic: {decomposition['extended_fast']['I_stochastic']:.6f}.",
        f"- Exploration damage counts: {decomposition['extended_fast']['exploration_damage']}.", "",
        "## Step-fit terminology", "",
        f"- The constrained/source-style step summary is approximately {step_note['constrained_source_style_step_tau_approx_epochs']:.0f} epochs.",
        f"- The free-gain transfer fit has K={step_note['free_gain_transfer_fit']['K']:.3f} and tau={step_note['free_gain_transfer_fit']['tau_epochs']:.1f} epochs.",
        f"- The current diagnostic grid refit with gain fixed to one gives tau={step_note['current_grid_refit_fixed_gain_one_tau_epochs']:.1f} epochs.",
        "- The source-comparable step-fit convention remains unresolved; tau≈133 is not labelled paper-equivalent.", "",
        "## Evidence boundary", "",
        "This is development-only transfer identification. It is not paired acceptance, source/reference evidence, or a paper-equivalence result.",
        "No slow, paired-acceptance, source-budget, held-out, reference, natural-drift, or Figure 5c run was launched.",
    ]
    atomic_text(EXTENDED_ROOT / "REPORT.md", "\n".join(report_lines))
    return status


def run_extended_fast_validation() -> dict[str, Any]:
    """Execute exactly one new 150-epoch fast period, resumably and fail closed."""
    settings = extended_config()
    preflight = verify_extension_preflight(validate_shards=True)
    prepare_continuation_checkpoint()
    provenance = read_json(PROVENANCE_PATH)
    base_state = read_json(BASE_CHECKPOINT)
    state_before = read_json(EXTENDED_CHECKPOINT)
    if int(state_before["epoch"]) < settings["base_epochs"] or int(
            state_before["epoch"]) > settings["total_epochs"]:
        raise RuntimeError("extension checkpoint epoch is outside the authorized continuation window")
    source = _source_config()
    plant = build_plant(source)
    frozen = _frozen()
    cell = run_cell(
        protocol=_extension_protocol(settings), plant=plant,
        frequency=float(settings["frequency_per_epoch"]),
        entropy_weight=float(frozen["entropy_coefficient"]), seed=int(settings["seed"]),
        optimizer_config=_optimizer_config(), initial_sigma=float(frozen["initial_sigma"]),
        checkpoint_path=EXTENDED_CHECKPOINT, dependency_hashes=dependency_hashes(ROOT, source),
        controller_hash=frozen["optimizer_bundle_hash"], clip=float(frozen["ppo_clip"]),
        baseline_weight=float(frozen["baseline_loss_weight"]), resume=True,
        checkpoint_every_candidates=int(settings["candidates_per_epoch"]),
        boundary=_boundary(plant), fresh_acquisition_required=False,
        source_budget_profile=PROFILE)
    if not cell["complete"]:
        raise RuntimeError("extended fast acquisition returned incomplete without an external interruption")
    extended_state = read_json(EXTENDED_CHECKPOINT)
    append_checks = validate_appended_state(
        base_state, extended_state, base_epochs=int(settings["base_epochs"]),
        total_epochs=int(settings["total_epochs"]),
        candidates_per_epoch=int(settings["candidates_per_epoch"]), validate_payloads=True)
    if file_hash(BASE_CHECKPOINT) != preflight["frozen_input_hashes"]["base_checkpoint"]:
        raise RuntimeError("original V18 fast checkpoint changed during continuation")
    provenance.update({
        "continuation_complete": True,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_after_sha256": file_hash(EXTENDED_CHECKPOINT),
        "base_checkpoint_unchanged_after": True,
        "fresh_continuation_epoch_count": int(settings["extra_epochs"]),
        "previous_600_epochs_unchanged": append_checks["immutable_base_shard_prefix"],
        "no_duplicate_epochs": append_checks["no_duplicate_epoch_rows"],
        "new_epoch_window": [600, 750],
        "new_epoch_shard_manifest_hash": canonical_hash(extended_state["epoch_shards"][600:]),
        "continuation_final_policy_identity": _policy_identity(extended_state),
        "append_checks": append_checks,
    })
    atomic_json(PROVENANCE_PATH, provenance)
    analysis_settings = {
        "label": "fast", "frequency_per_epoch": settings["frequency_per_epoch"],
        "epochs": settings["total_epochs"], "analysis_periods": settings["extended_analysis_periods"],
        "candidates_per_epoch": settings["candidates_per_epoch"],
        "qec_cycles_per_candidate": settings["qec_cycles_per_candidate"],
        "seed": settings["seed"], "profile": PROFILE,
    }
    transfer = _analyse_transfer(
        "fast", cell, EXTENDED_CHECKPOINT, PROVENANCE_PATH,
        settings_override=analysis_settings)
    transfer["fresh_acquisition"] = False
    transfer["fresh_continuation_data"] = True
    transfer.pop("reused_shard_ids", None)
    transfer["inherited_base_epoch_shard_count"] = 600
    transfer["inherited_base_epoch_shard_manifest_hash"] = provenance[
        "base_epoch_shard_manifest_hash"]
    return _write_outputs(transfer, provenance)


def forbidden_auto_run_contract() -> dict[str, Any]:
    """Pure contract exposed for smoke tests without starting acquisition."""
    settings = extended_config()
    return {
        "only_acquisition": "FAST_EPOCHS_600_TO_749",
        "forbidden_auto_runs": settings["forbidden_auto_runs"],
        "paired_acceptance_executed": False,
        "slow_acquisition_launched": False,
        **NONFINAL,
    }
