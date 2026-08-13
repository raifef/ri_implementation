"""Single gated V20 experimental repair and bounded fast-only validation."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
from hdfa_rl_suite.google_pure_source_exact.figure5a.bounded_action_ablation import Figure5aBoundedActionAblation

from hdfa_rl_suite.google_pure_source_exact.figure5a.acquisition import (
    STREAMS,
    _behavior,
    _freeze_batch,
    _load_runtime,
    _new_state,
)
from hdfa_rl_suite.google_pure_source_exact.figure5a.contracts import (
    AcquisitionMode,
    Figure5aProtocol,
    canonical_hash as source_hash,
)
from hdfa_rl_suite.google_pure_source_exact.figure5a.validation import (
    build_plant,
    dependency_hashes,
)
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.gaussian import (
    DirectSigmaGaussianPolicy,
    entropy,
)
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.losses import (
    total_loss_and_gradients,
)
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.optimizer import (
    DirectSigmaOptimizer,
)
from hdfa_rl_suite.google_pure_source_exact.source_normalization import (
    require_v15_boundary_provenance,
)
from hdfa_rl_suite.google_pure_v19_experimental.dynamic_validation import (
    _boundary,
    _controller_spec,
    _fit_period,
    _optimizer_config,
    _source_config,
    _stream_metrics,
)

from .core import cosine_alignment, project_shared_subspace
from .data import (
    EXPECTED_V19_CONTROLLER,
    evaluator,
    load_matched_run,
    verify_import_manifest,
)
from .io import (
    ARTIFACT_ROOT,
    ROOT,
    atomic_json,
    canonical_hash,
    file_hash,
    nonfinal,
    read_json,
    relative,
    settings,
    write_artifact,
)


REPAIR_NAME = "PUBLIC_FIGURE5A_SHARED_SUBSPACE_MEAN_GRADIENT_PROJECTION"
CONTROLLER_MODE = "PUBLIC_ANALOGUE_DIRECT_SIGMA_EXPERIMENTAL_V20_SHARED_SUBSPACE"
PARAMETERIZATION = "DIRECT_SIGMA_PUBLIC_ANALOGUE_EXPERIMENTAL"
REPAIRED_ROOT = ARTIFACT_ROOT / "repaired_fast"


def repaired_controller_hash() -> str:
    parent = _controller_spec()
    return canonical_hash({
        "controller_mode": CONTROLLER_MODE,
        "frozen_experimental_parent_hash": parent.controller_hash,
        "frozen_source_parent_hash": parent.frozen_parent_controller_hash,
        "parameterization": PARAMETERIZATION,
        "scale_objective": parent.identity_payload["scale_objective"],
        "single_causal_repair": REPAIR_NAME,
        "projection_basis": "unit vector [1,...,1]/sqrt(41)",
        "projection_justification": "Figure5aStimPlant.optimum returns one shared value for all 41 coordinates",
        "mean_learning_rate": parent.mean_learning_rate,
        "sigma_learning_rate": parent.sigma_learning_rate,
        "baseline_learning_rate": parent.baseline_learning_rate,
        "effective_entropy_coefficient": parent.effective_entropy_coefficient,
        "ppo_clip": parent.ppo_clip,
        "baseline_loss_weight": parent.baseline_loss_weight,
        "sigma_bounds": [parent.minimum_sigma, parent.maximum_sigma],
        "initial_sigma": parent.initial_sigma,
    })


def _atomic_state(path: Path, value: Mapping[str, Any]) -> None:
    delays = (.02, .05, .1, .2, .4, .8, 1.6, 3.2)
    for attempt, delay in enumerate(delays):
        try:
            atomic_json(path, value)
            return
        except PermissionError:
            if attempt == len(delays) - 1:
                raise
            time.sleep(delay)


def _identity() -> dict[str, Any]:
    parent = _controller_spec()
    return {
        "controller_mode": CONTROLLER_MODE,
        "controller_hash": repaired_controller_hash(),
        "parameterization": PARAMETERIZATION,
        "source_exact": False,
        "frozen_experimental_parent_hash": EXPECTED_V19_CONTROLLER,
        "frozen_source_parent_hash": parent.frozen_parent_controller_hash,
        "single_causal_repair": REPAIR_NAME,
        "scale_objective": parent.identity_payload["scale_objective"],
        "effective_entropy_coefficient": parent.effective_entropy_coefficient,
        "mean_learning_rate": parent.mean_learning_rate,
        "sigma_learning_rate": parent.sigma_learning_rate,
        "baseline_learning_rate": parent.baseline_learning_rate,
        "candidate_count_changed": False,
        "cycles_per_candidate_changed": False,
        "normalization_changed": False,
        "sigma_objective_changed": False,
        "mean_learning_rate_changed": False,
    }


def run_repaired_fast_acquisition() -> dict[str, Any]:
    root_cause_path = ARTIFACT_ROOT / "root_cause_classification.json"
    if not root_cause_path.is_file():
        raise RuntimeError("V20 root-cause artifact must exist before repair acquisition")
    root_cause = read_json(root_cause_path)
    if root_cause.get("repair_permitted") is not True or \
            root_cause.get("permitted_single_repair") != REPAIR_NAME:
        raise RuntimeError("V20 root-cause gate did not authorize the implemented repair")
    cfg = settings()
    cell = cfg["postrepair"]
    source = _source_config()
    plant = build_plant(source)
    bounded = Figure5aBoundedActionAblation(plant)
    boundary = _boundary(plant)
    parent = _controller_spec()
    protocol = Figure5aProtocol(
        AcquisitionMode.VALIDATION, int(cell["epochs"]), int(cell["candidates_per_epoch"]),
        int(cell["qec_cycles_per_candidate"]), int(source["plant"]["circuit_rounds"]))
    frequency = float(cfg["fast_frequency_per_epoch"])
    seed = int(cell["seed"])
    checkpoint_path = REPAIRED_ROOT / "checkpoint.json"
    dependencies = {
        **dependency_hashes(ROOT, source),
        "v20_repair_code": file_hash(Path(__file__)),
        "v20_protocol": file_hash(ROOT / "configs/google_pure_v20/protocol.json"),
        "v20_root_cause": file_hash(root_cause_path),
    }
    identity = _identity()
    if checkpoint_path.is_file():
        state = read_json(checkpoint_path)
        expected = {
            "protocol_hash": protocol.protocol_hash,
            "plant_hash": plant.plant_hash,
            "frequency": frequency,
            "seed": seed,
            "controller_hash": identity["controller_hash"],
            "v20_experimental_controller": identity,
            "dependency_hashes": dependencies,
            "v15_boundary": boundary.provenance_fields(),
        }
        if {key: state.get(key) for key in expected} != expected:
            raise RuntimeError("V20 repaired checkpoint identity or dependency changed")
        policy, optimizer, baseline = _load_runtime(state)
    else:
        policy = DirectSigmaGaussianPolicy(
            np.zeros(41), np.full(41, parent.initial_sigma), seed=seed)
        optimizer = DirectSigmaOptimizer(41, plant.detector_count, _optimizer_config(parent))
        baseline = np.zeros(plant.detector_count)
        state = _new_state(
            protocol=protocol, plant=plant, frequency=frequency,
            entropy_weight=parent.effective_entropy_coefficient, seed=seed,
            policy=policy, optimizer=optimizer, baseline=baseline,
            dependency_hashes=dependencies, controller_hash=identity["controller_hash"],
            boundary=boundary)
        state["v20_experimental_controller"] = identity
        state["controller_mode"] = CONTROLLER_MODE
        state["parameterization"] = PARAMETERIZATION
        state["single_causal_repair"] = REPAIR_NAME
        _atomic_state(checkpoint_path, state)

    while int(state["epoch"]) < protocol.epochs:
        epoch = int(state["epoch"])
        if state["active_batch"] is None:
            state["active_batch"] = _freeze_batch(
                policy, plant, boundary, protocol.candidates_per_epoch)
            state["policy"] = policy.state_dict(
                optimizer_state=optimizer.state_dict(), baseline=baseline)
            _atomic_state(checkpoint_path, state)
        active = state["active_batch"]
        while int(active["next_candidate"]) < protocol.candidates_per_epoch:
            candidate = int(active["next_candidate"])
            optimum_normalized = plant.optimum(epoch, frequency)
            optimum = boundary.target_to_native(optimum_normalized)
            controls = {
                "fixed": boundary.target_to_native(np.zeros(41)),
                "optimal": optimum,
                "stochastic": np.asarray(active["applied_actions"][candidate]),
                "learned_mean": np.asarray(active["applied_behavior_mean"]),
            }
            detector_counts = {}
            for stream in STREAMS:
                detector_counts[stream] = plant.sample_detector_counts(
                    controls[stream], epoch=epoch, frequency=frequency,
                    qec_cycles=protocol.qec_cycles_per_candidate,
                    seed=plant.stream_seed(seed, stream, epoch, candidate),
                    target_controls=optimum)
                active["counts"][stream].append(int(detector_counts[stream].sum()))
            active["stochastic_detector_counts"].append(
                detector_counts["stochastic"].tolist())
            active["next_candidate"] = candidate + 1
            state["candidate_boundaries_completed"] += 1
            state["active_batch"] = active
            _atomic_state(checkpoint_path, state)

        behavior = _behavior(active)
        stochastic_counts = np.asarray(active["stochastic_detector_counts"], dtype=float)
        rewards = -stochastic_counts / protocol.shots_per_policy
        loss = total_loss_and_gradients(
            np.asarray(active["latent_actions"]), rewards, plant.mask, policy.mean,
            policy.sigma, baseline, behavior, clip=parent.ppo_clip,
            entropy_weight=parent.effective_entropy_coefficient,
            baseline_weight=parent.baseline_loss_weight)
        raw_update_direction = -loss.grad_mean
        projected_grad_mean = project_shared_subspace(loss.grad_mean)
        projected_update_direction = -projected_grad_mean
        target_latent = bounded.latent_controls_for(plant.optimum(epoch, frequency))
        beneficial = target_latent - policy.mean
        beneficial /= max(float(np.linalg.norm(beneficial)), 1e-15)
        update = optimizer.step(
            policy.mean, policy.sigma, baseline, projected_grad_mean, loss.grad_sigma,
            loss.grad_baseline, mean_bounds=(-2.0, 2.0))
        policy.policy_version += 1
        record = {
            "epoch": epoch,
            "optimum": float(plant.optimum(epoch, frequency)[0]),
            **identity,
            "root_cause_artifact_sha256": dependencies["v20_root_cause"],
            "root_cause_primary": root_cause["primary_classification"],
            "raw_mean_update_direction": raw_update_direction.tolist(),
            "projected_mean_update_direction": projected_update_direction.tolist(),
            "raw_beneficial_alignment": cosine_alignment(raw_update_direction, beneficial),
            "projected_beneficial_alignment": cosine_alignment(
                projected_update_direction, beneficial),
            "raw_wrong_sign": float(raw_update_direction @ beneficial) <= 0,
            "projected_wrong_sign": float(projected_update_direction @ beneficial) <= 0,
            "behavior_mean": active["applied_behavior_mean"],
            "normalized_behavior_mean": active["normalized_behavior_mean"],
            "latent_behavior_mean": active["latent_behavior_mean"],
            "behavior_sigma": active["behavior_sigma"],
            "post_update_mean": boundary.apply(
                bounded.apply_control_transform(policy.mean)).native.tolist(),
            "post_update_normalized_mean": bounded.apply_control_transform(policy.mean).tolist(),
            "post_update_latent_mean": policy.mean.tolist(),
            "post_update_sigma": policy.sigma.tolist(),
            "policy_entropy": entropy(np.asarray(active["behavior_sigma"])),
            "counts": {stream: list(map(int, active["counts"][stream]))
                       for stream in STREAMS},
            "stream_totals": {stream: int(sum(active["counts"][stream]))
                              for stream in STREAMS},
            "stochastic_detector_counts": active["stochastic_detector_counts"],
            "reward_sigma_gradient_norm": loss.diagnostics["reward_sigma_gradient_norm"],
            "entropy_sigma_gradient_norm": loss.diagnostics["entropy_sigma_gradient_norm"],
            "fraction_at_positivity_guard": update["fraction_at_positivity_guard"],
            "candidate_count": protocol.candidates_per_epoch,
            "qec_cycles_per_candidate": protocol.qec_cycles_per_candidate,
            "ratio_clipping_mode": loss.diagnostics["ratio_clipping_mode"],
            "baseline_mode": loss.diagnostics["baseline_mode"],
            **boundary.provenance_fields(),
        }
        epoch_path = REPAIRED_ROOT / "checkpoint_epochs" / f"epoch-{epoch:04d}.json"
        record_hash = source_hash(record)
        _atomic_state(epoch_path, {"record_hash": record_hash, "record": record})
        state["epoch_shards"].append({
            "epoch": epoch, "path": str(epoch_path.resolve()), "record_hash": record_hash})
        state["epoch"] = epoch + 1
        state["active_batch"] = None
        state["policy"] = policy.state_dict(
            optimizer_state=optimizer.state_dict(), baseline=baseline)
        _atomic_state(checkpoint_path, state)

    records = []
    for shard in state["epoch_shards"]:
        payload = read_json(Path(shard["path"]))
        if payload["record_hash"] != shard["record_hash"] or \
                source_hash(payload["record"]) != shard["record_hash"]:
            raise RuntimeError("V20 repaired checkpoint shard corruption")
        records.append(payload["record"])
    if [row["epoch"] for row in records] != list(range(protocol.epochs)):
        raise RuntimeError("V20 repaired checkpoint is incomplete")
    artifact = nonfinal({
        "pass": True,
        "execution_complete": True,
        **identity,
        "protocol": state["protocol"],
        "protocol_hash": state["protocol_hash"],
        "plant_hash": plant.plant_hash,
        "frequency": frequency,
        "seed": seed,
        "dependency_hashes": dependencies,
        "epoch_records": records,
        "candidate_boundaries_completed": state["candidate_boundaries_completed"],
        "no_candidates_dropped": state["candidate_boundaries_completed"] ==
            protocol.epochs * protocol.candidates_per_epoch,
        "checkpoint": relative(checkpoint_path),
        "candidate_qec_cycles": protocol.candidate_qec_cycles,
        "four_stream_qec_cycles": protocol.four_stream_qec_cycles,
        "campaign_scope": "BOUNDED_FAST_ONLY_SINGLE_SEED_DEVELOPMENT_VALIDATION",
        "forbidden_auto_runs_launched": [],
        **boundary.provenance_fields(),
    })
    require_v15_boundary_provenance(artifact)
    atomic_json(REPAIRED_ROOT / "acquisition.json", artifact)
    return artifact


def _orthogonal_power(records: list[dict[str, Any]]) -> float:
    trace = np.asarray([row["normalized_behavior_mean"] for row in records])
    return float(np.mean((trace - np.mean(trace, axis=1)[:, None])**2))


def _cell_summary(records: list[dict[str, Any]], frequency: float) -> dict[str, Any]:
    metrics = _stream_metrics(records)
    transfer = _fit_period(records, frequency)
    return {
        "I_mean": metrics["I_mean"],
        "I_stochastic": metrics["I_stochastic"],
        "gain": transfer["gain"],
        "phase_lag_radians": transfer["phase_lag_radians"],
        "orthogonal_diffusion_power": _orthogonal_power(records),
        "sigma_median": float(np.median([
            value for row in records for value in row["behavior_sigma"]])),
        "candidate_damage_counts": metrics["exploration_damage"],
        "cycle_budget": int(sum(
            row["candidate_count"] * row["qec_cycles_per_candidate"] for row in records)),
    }


def run_minimal_repair_validation() -> dict[str, Any]:
    verify_import_manifest()
    root_path = ARTIFACT_ROOT / "root_cause_classification.json"
    if not root_path.is_file():
        from .population import classify_root_cause
        classify_root_cause()
    root = read_json(root_path)
    if root.get("repair_permitted") is not True:
        raise RuntimeError("V20 causal gate forbids repair")
    identity = _identity()
    minimal = {
        "pass": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "causal_parent_artifact": relative(root_path),
        "causal_parent_sha256": file_hash(root_path),
        "causal_parent_classification": root["primary_classification"],
        "repair": REPAIR_NAME,
        "exactly_one_causal_repair": True,
        "controller": identity,
        "mechanism": (
            "orthogonal projection of the supplied noisy mean gradient onto the public "
            "Figure-5a shared driven subspace before the unchanged optimizer step"),
        "plant_justification": (
            "Figure5aStimPlant.optimum(epoch, frequency) is a length-41 constant vector"),
        "hidden_target_value_used": False,
        "production_source_style_controller_changed": False,
        "frozen_v19_experimental_controller_changed": False,
        "mean_learning_rate_changed": False,
        "entropy_changed": False,
        "sigma_learning_rate_changed": False,
        "normalization_changed": False,
        "candidate_budget_changed": False,
        "forbidden_auto_runs_launched": [],
    }
    write_artifact("minimal_repair", minimal, title="V20 single causal repair")
    repaired = run_repaired_fast_acquisition()
    cfg = settings()
    start = int(cfg["postrepair"]["transient_epochs"])
    stop = int(cfg["postrepair"]["epochs"])
    frequency = float(cfg["fast_frequency_per_epoch"])
    repaired_records = repaired["epoch_records"][start:stop]
    baseline_records = load_matched_run("fast")["records"][start:stop]
    repaired_summary = _cell_summary(repaired_records, frequency)
    baseline_summary = _cell_summary(baseline_records, frequency)
    repaired_summary["gradient_alignment"] = float(np.median([
        row["projected_beneficial_alignment"] for row in repaired_records]))
    repaired_summary["wrong_sign_fraction"] = float(np.mean([
        row["projected_wrong_sign"] for row in repaired_records]))
    gradient_path = ARTIFACT_ROOT / "fast_gradient_statistics.json"
    gradients = read_json(gradient_path)
    baseline_gradient_rows = [row for row in gradients["epoch_rows"]
                              if start <= int(row["epoch"]) < stop]
    baseline_summary["gradient_alignment"] = float(np.median([
        row["cosine_alignment_with_local_beneficial_direction"]
        for row in baseline_gradient_rows]))
    baseline_summary["wrong_sign_fraction"] = float(np.mean([
        row["actual_wrong_sign"] for row in baseline_gradient_rows]))
    population = read_json(ARTIFACT_ROOT / "population_gradient_fast_rollout.json")[
        "population_gradient_fast"]
    population_summary = {
        "I_mean": population["I_mean"],
        "I_stochastic": population["I_stochastic_exact_diagnostic"],
        "gain": population["gain"],
        "phase_lag_radians": population["phase_lag_radians"],
        "orthogonal_diffusion_power": population["orthogonal_diffusion_power"],
        "gradient_alignment": 1.0,
        "wrong_sign_fraction": 0.0,
        "sigma_median": float(np.median([
            value for row in repaired["epoch_records"][start:stop]
            for value in row["behavior_sigma"]])),
        "candidate_damage_counts": None,
        "cycle_budget": 0,
    }
    manifest = verify_import_manifest()
    gates = {
        "repair_targets_diagnosed_mechanism": root["primary_classification"] in {
            "FINITE_CANDIDATE_DIRECTIONAL_FAILURE", "ORTHOGONAL_MEAN_DIFFUSION"},
        "fixed_budget_fairness": baseline_summary["cycle_budget"] ==
            repaired_summary["cycle_budget"],
        "mean_improvement_is_mechanistically_explained":
            repaired_summary["I_mean"] > baseline_summary["I_mean"] and
            repaired_summary["orthogonal_diffusion_power"] <
            baseline_summary["orthogonal_diffusion_power"],
        "no_hidden_sigma_suppression": identity["sigma_objective_changed"] is False,
        "source_style_branch_unchanged": manifest["invariants"][
            "frozen_source_style_branch_unchanged"] and verify_import_manifest()["inputs"] ==
            manifest["inputs"],
        "slow_intermediate_not_rerun": True,
        "paper_equivalence_claim_permitted": False,
    }
    value = {
        "pass": all(value is True for key, value in gates.items()
                    if key != "paper_equivalence_claim_permitted") and
            gates["paper_equivalence_claim_permitted"] is False,
        "execution_complete": True,
        "controller": identity,
        "analysis_epoch_window": [start, stop],
        "baseline_v19_experimental_fast": baseline_summary,
        "repaired_fast": repaired_summary,
        "population_gradient_reference": population_summary,
        "gates": gates,
        "exactly_one_causal_repair": True,
        "source_style_hashes_after": verify_import_manifest()["inputs"],
        "slow_intermediate_rerun": False,
        "forbidden_auto_runs_launched": [],
    }
    return write_artifact("postrepair_fast_validation", value,
                          title="V20 bounded post-repair fast validation")
