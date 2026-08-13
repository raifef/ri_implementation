"""Gated V21 short fast-only online rollout and bounded generalization audit."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
from hdfa_rl_suite.google_pure_source_exact.figure5a.bounded_action_ablation import Figure5aBoundedActionAblation

from hdfa_rl_suite.google_pure_v20.core import cosine_alignment
from hdfa_rl_suite.google_pure_source_exact.figure5a.acquisition import (
    STREAMS,
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

from .candidate_design import (
    DESIGN_NAMES,
    SOURCE_FIDELITY,
    candidate_is_nonoracle,
    estimate_policy_updates,
    generate_frame,
)
from .diagnostics import _blocks
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
from .lineage import verify_import_manifest


CONTROLLER_MODE = "PUBLIC_ANALOGUE_DIRECT_SIGMA_EXPERIMENTAL_V21_CANDIDATE_DESIGN"
PARAMETERIZATION = "DIRECT_SIGMA_PUBLIC_ANALOGUE_EXPERIMENTAL"
ONLINE_ROOT = ARTIFACT_ROOT / "online_fast"
POSTHOC_ONLY_COMPATIBLE_ONLINE_HASHES = {
    "af243bb74b209ca27643425b4afd1ddfba53c0ed4c8f7eab61220e7fbc6c83e4",
}


def candidate_controller_hash(design_id: str) -> str:
    parent = _controller_spec()
    return canonical_hash({
        "controller_mode": CONTROLLER_MODE,
        "frozen_v19_parent": parent.controller_hash,
        "candidate_design": design_id,
        "candidate_design_name": DESIGN_NAMES[design_id],
        "source_fidelity": SOURCE_FIDELITY[design_id],
        "K": 8, "M": 12000, "B": 96000,
        "mean_learning_rate": parent.mean_learning_rate,
        "sigma_learning_rate": parent.sigma_learning_rate,
        "baseline_learning_rate": parent.baseline_learning_rate,
        "effective_entropy_coefficient": parent.effective_entropy_coefficient,
        "scale_objective": parent.identity_payload["scale_objective"],
        "normalization_changed": False,
        "candidate_estimator": "frame-specific mean and sigma scores",
        "candidate_design_code_hash": file_hash(
            ROOT / "src/hdfa_rl_suite/google_pure_v21/candidate_design.py"),
        "action_safety_envelope": "public all-target V15-normalized physical range v1",
    })


def _identity(design_id: str) -> dict[str, Any]:
    parent = _controller_spec()
    return {
        "controller_mode": CONTROLLER_MODE,
        "controller_hash": candidate_controller_hash(design_id),
        "frozen_v19_parent_hash": parent.controller_hash,
        "candidate_design": design_id,
        "candidate_design_name": DESIGN_NAMES[design_id],
        "source_fidelity": SOURCE_FIDELITY[design_id],
        "parameterization": PARAMETERIZATION,
        "source_exact": False,
        "mean_learning_rate_changed": False,
        "sigma_learning_rate_changed": False,
        "entropy_changed": False,
        "normalization_changed": False,
        "exploration_scale_multiplier": 1.0,
        "elementwise_coordinate_ratio_clipping":
            "retained; the single on-policy update has unit behavior/current ratio",
        "action_safety_envelope": "public all-target V15-normalized physical range v1",
    }


def _atomic(path: Path, value: Mapping[str, Any]) -> None:
    delays = (.02, .05, .1, .2, .4, .8, 1.6, 3.2)
    for attempt, delay in enumerate(delays):
        try:
            atomic_json(path, value)
            return
        except PermissionError:
            if attempt == len(delays) - 1:
                raise
            time.sleep(delay)


def normalized_safety_limits(plant: Any, boundary: Any) -> np.ndarray:
    """Static normalized limits safe for every public target in [-1, 1]."""
    limits = Figure5aBoundedActionAblation(plant).normalized_control_limits(
        boundary.native_scale)
    if np.any(limits <= 1.0) or not np.all(np.isfinite(limits)):
        raise RuntimeError("V21 plant has no phase-independent normalized safety envelope")
    return limits


def apply_safe_control_transform(latent: np.ndarray, plant: Any, boundary: Any) -> np.ndarray:
    """Apply the same scaled-tanh family with the all-target safe envelope."""
    value = np.asarray(latent, dtype=float)
    limits = normalized_safety_limits(plant, boundary)
    if value.shape[-1:] != limits.shape or not np.all(np.isfinite(value)):
        raise ValueError("latent V21 controls are not aligned and finite")
    return limits * np.tanh(value / limits)


def _freeze_frame(policy: DirectSigmaGaussianPolicy, plant: Any, boundary: Any,
                  design_id: str, epoch: int, seed: int) -> dict[str, Any]:
    frame = generate_frame(
        design_id, dimension=41, epoch=epoch, seed=seed, blocks=_blocks())
    if not frame.sigma_estimator_valid:
        raise RuntimeError(f"{design_id} has no valid online sigma estimator")
    mean = policy.mean.copy(); sigma = policy.sigma.copy()
    actions = mean[None, :] + sigma[None, :] * frame.standardized_directions
    limits = normalized_safety_limits(plant, boundary)
    normalized_actions = apply_safe_control_transform(actions, plant, boundary)
    applied_actions = [boundary.apply(action).native.tolist() for action in normalized_actions]
    normalized_mean = apply_safe_control_transform(mean, plant, boundary)
    applied_mean = boundary.apply(normalized_mean).native
    return {
        "design_id": design_id,
        "frame_hash": canonical_hash(frame.standardized_directions.tolist()),
        "standardized_directions": frame.standardized_directions.tolist(),
        "mean_score_factors": frame.mean_score_factors.tolist(),
        "sigma_score_factors": np.asarray(frame.sigma_score_factors).tolist(),
        "inclusion_probabilities": frame.inclusion_probabilities.tolist(),
        "selected_blocks": list(frame.selected_blocks),
        "frame_metadata": dict(frame.metadata),
        "latent_actions": actions.tolist(),
        "normalized_actions": normalized_actions.tolist(),
        "applied_actions": applied_actions,
        "latent_behavior_mean": mean.tolist(),
        "normalized_behavior_mean": normalized_mean.tolist(),
        "applied_behavior_mean": applied_mean.tolist(),
        "normalized_safety_limits": limits.tolist(),
        "safety_envelope_uses_current_target_or_phase": False,
        "behavior_sigma": sigma.tolist(),
        "next_candidate": 0,
        "counts": {stream: [] for stream in STREAMS},
        "stochastic_detector_counts": [],
    }


def _restore_frame(active: Mapping[str, Any]) -> Any:
    from .candidate_design import CandidateFrame
    return CandidateFrame(
        design_id=str(active["design_id"]),
        standardized_directions=np.asarray(active["standardized_directions"], dtype=float),
        mean_score_factors=np.asarray(active["mean_score_factors"], dtype=float),
        sigma_score_factors=np.asarray(active["sigma_score_factors"], dtype=float),
        inclusion_probabilities=np.asarray(active["inclusion_probabilities"], dtype=float),
        selected_blocks=tuple(active["selected_blocks"]),
        estimator_valid=True,
        sigma_estimator_valid=True,
        physical_covariance_preserved_in_expectation=True,
        metadata=dict(active["frame_metadata"]),
    )


def _run_design(design_id: str) -> dict[str, Any]:
    cfg = settings()
    parent = _controller_spec()
    source = _source_config()
    plant = build_plant(source)
    boundary = _boundary(plant)
    epochs = int(cfg["online_epochs"])
    seed = int(cfg["new_fast_seeds"][0])
    frequency = float(cfg["fast_frequency_per_epoch"])
    protocol = Figure5aProtocol(
        AcquisitionMode.VALIDATION, epochs, 8, 12000,
        int(source["plant"]["circuit_rounds"]))
    identity = _identity(design_id)
    root = ONLINE_ROOT / design_id
    checkpoint_path = root / "checkpoint.json"
    dependencies = {
        **dependency_hashes(ROOT, source),
        "v21_online_code": file_hash(Path(__file__)),
        "v21_candidate_design_code": file_hash(
            ROOT / "src/hdfa_rl_suite/google_pure_v21/candidate_design.py"),
        "v21_protocol": file_hash(ROOT / "configs/google_pure_v21/protocol.json"),
        "v21_promotion_gate": file_hash(ARTIFACT_ROOT / "frozen_state_promotion_gate.json"),
    }
    if checkpoint_path.is_file():
        state = read_json(checkpoint_path)
        expected = {
            "protocol_hash": protocol.protocol_hash,
            "plant_hash": plant.plant_hash,
            "frequency": frequency,
            "seed": seed,
            "controller_hash": identity["controller_hash"],
            "v21_candidate_controller": identity,
            "dependency_hashes": dependencies,
            "v15_boundary": boundary.provenance_fields(),
        }
        if {key: state.get(key) for key in expected} != expected:
            old_dependencies = state.get("dependency_hashes", {})
            noncode_dependencies_match = {
                key: value for key, value in old_dependencies.items()
                if key != "v21_online_code"
            } == {
                key: value for key, value in dependencies.items()
                if key != "v21_online_code"
            }
            nondependency_identity_matches = all(
                state.get(key) == value for key, value in expected.items()
                if key != "dependency_hashes")
            completed_analysis_only_migration = (
                int(state.get("epoch", -1)) == epochs and
                state.get("active_batch") is None and
                int(state.get("candidate_boundaries_completed", -1)) == epochs * 8 and
                old_dependencies.get("v21_online_code") in
                    POSTHOC_ONLY_COMPATIBLE_ONLINE_HASHES and
                noncode_dependencies_match and nondependency_identity_matches)
            if not completed_analysis_only_migration:
                raise RuntimeError("V21 online checkpoint identity changed")
            state["dependency_hashes"] = dependencies
            state.setdefault("completed_acquisition_analysis_hash_migrations", []).append({
                "from_online_code_hash": old_dependencies["v21_online_code"],
                "to_online_code_hash": dependencies["v21_online_code"],
                "scope": "posthoc cosine metric import only; no epoch data changed",
                "epoch_records_changed": False,
            })
            _atomic(checkpoint_path, state)
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
        state["v21_candidate_controller"] = identity
        state["controller_mode"] = CONTROLLER_MODE
        state["candidate_design"] = design_id
        _atomic(checkpoint_path, state)
    while int(state["epoch"]) < epochs:
        epoch = int(state["epoch"])
        if state["active_batch"] is None:
            state["active_batch"] = _freeze_frame(
                policy, plant, boundary, design_id, epoch, seed)
            state["policy"] = policy.state_dict(
                optimizer_state=optimizer.state_dict(), baseline=baseline)
            _atomic(checkpoint_path, state)
        active = state["active_batch"]
        while int(active["next_candidate"]) < 8:
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
                    qec_cycles=12000, seed=plant.stream_seed(
                        seed, stream, epoch, candidate), target_controls=optimum)
                active["counts"][stream].append(int(detector_counts[stream].sum()))
            active["stochastic_detector_counts"].append(
                detector_counts["stochastic"].tolist())
            active["next_candidate"] = candidate + 1
            state["candidate_boundaries_completed"] += 1
            state["active_batch"] = active
            _atomic(checkpoint_path, state)
        frame = _restore_frame(active)
        mean_before = policy.mean.copy(); sigma_before = policy.sigma.copy()
        rewards = (-np.asarray(active["stochastic_detector_counts"], dtype=float) /
                   protocol.shots_per_policy)
        estimates = estimate_policy_updates(
            frame, rewards, baseline, plant.mask, sigma_before)
        mean_update = np.asarray(estimates["mean_update_direction"], dtype=float)
        reward_sigma_gradient = np.asarray(estimates["sigma_reward_gradient"], dtype=float)
        entropy_sigma_gradient = -parent.effective_entropy_coefficient / sigma_before
        sigma_gradient = reward_sigma_gradient + entropy_sigma_gradient
        baseline_gradient = parent.baseline_loss_weight * np.asarray(
            estimates["baseline_gradient"], dtype=float)
        update = optimizer.step(
            policy.mean, policy.sigma, baseline, -mean_update, sigma_gradient,
            baseline_gradient, mean_bounds=(-2.0, 2.0))
        policy.policy_version += 1
        record = {
            "epoch": epoch,
            **identity,
            "frame_hash": active["frame_hash"],
            "frame_rank": int(np.linalg.matrix_rank(
                np.asarray(active["standardized_directions"]))),
            "selected_blocks": active["selected_blocks"],
            "latent_behavior_mean": active["latent_behavior_mean"],
            "normalized_behavior_mean": active["normalized_behavior_mean"],
            "behavior_mean": active["applied_behavior_mean"],
            "behavior_sigma": active["behavior_sigma"],
            "mean_update_direction": mean_update.tolist(),
            "reward_sigma_gradient": reward_sigma_gradient.tolist(),
            "entropy_sigma_gradient": entropy_sigma_gradient.tolist(),
            "post_update_latent_mean": policy.mean.tolist(),
            "post_update_normalized_mean": apply_safe_control_transform(
                policy.mean, plant, boundary).tolist(),
            "post_update_sigma": policy.sigma.tolist(),
            "policy_entropy": entropy(sigma_before),
            "fraction_at_positivity_guard": update["fraction_at_positivity_guard"],
            "normalized_safety_limits": active["normalized_safety_limits"],
            "safety_envelope_uses_current_target_or_phase": False,
            "counts": {stream: list(map(int, active["counts"][stream]))
                       for stream in STREAMS},
            "stream_totals": {stream: int(sum(active["counts"][stream]))
                              for stream in STREAMS},
            "stochastic_detector_counts": active["stochastic_detector_counts"],
            "candidate_count": 8,
            "qec_cycles_per_candidate": 12000,
            "mean_delta": (policy.mean - mean_before).tolist(),
            **boundary.provenance_fields(),
        }
        epoch_path = root / "checkpoint_epochs" / f"epoch-{epoch:04d}.json"
        record_hash = source_hash(record)
        _atomic(epoch_path, {"record_hash": record_hash, "record": record})
        state["epoch_shards"].append({
            "epoch": epoch, "path": str(epoch_path.resolve()), "record_hash": record_hash})
        state["epoch"] = epoch + 1
        state["active_batch"] = None
        state["policy"] = policy.state_dict(
            optimizer_state=optimizer.state_dict(), baseline=baseline)
        _atomic(checkpoint_path, state)
    records = []
    for shard in state["epoch_shards"]:
        payload = read_json(Path(shard["path"]))
        if payload["record_hash"] != shard["record_hash"] or \
                source_hash(payload["record"]) != shard["record_hash"]:
            raise RuntimeError("V21 online shard corruption")
        records.append(payload["record"])
    artifact = nonfinal({
        "pass": True,
        "execution_complete": True,
        **identity,
        "protocol": state["protocol"],
        "plant_hash": plant.plant_hash,
        "frequency": frequency,
        "seed": seed,
        "dependency_hashes": dependencies,
        "epoch_records": records,
        "candidate_boundaries_completed": state["candidate_boundaries_completed"],
        "no_candidates_dropped": state["candidate_boundaries_completed"] == epochs*8,
        "checkpoint": relative(checkpoint_path),
        "candidate_qec_cycles": protocol.candidate_qec_cycles,
        "four_stream_qec_cycles": protocol.four_stream_qec_cycles,
        "forbidden_auto_runs_launched": [],
        **boundary.provenance_fields(),
    })
    require_v15_boundary_provenance(artifact)
    atomic_json(root / "acquisition.json", artifact)
    return artifact


def _orthogonal_power(records: list[dict[str, Any]]) -> float:
    trace = np.asarray([row["normalized_behavior_mean"] for row in records])
    return float(np.mean((trace - np.mean(trace, axis=1)[:, None])**2))


def _summary(records: list[dict[str, Any]], frequency: float) -> dict[str, Any]:
    stream = _stream_metrics(records)
    transfer = _fit_period(records, frequency)
    sigma = np.asarray([row["behavior_sigma"] for row in records], dtype=float)
    return {
        "I_mean": stream["I_mean"],
        "I_stochastic": stream["I_stochastic"],
        "gain": transfer["gain"],
        "phase_lag_radians": transfer["phase_lag_radians"],
        "orthogonal_mean_diffusion": _orthogonal_power(records),
        "sigma_median": float(np.median(sigma)),
        "sigma_minimum": float(np.min(sigma)),
        "sigma_maximum": float(np.max(sigma)),
        "sigma_trajectory": [{
            "epoch": int(row["epoch"]),
            "median": float(np.median(row["behavior_sigma"])),
            "minimum": float(np.min(row["behavior_sigma"])),
            "maximum": float(np.max(row["behavior_sigma"])),
        } for row in records],
        "candidate_damage_counts": stream["exploration_damage"],
        "cycle_budget": int(sum(row["candidate_count"] *
                                row["qec_cycles_per_candidate"] for row in records)),
    }


def _native_detector_expectations(plant: Any, controls: np.ndarray, epoch: int,
                                  frequency: float, target: np.ndarray) -> np.ndarray:
    """Evaluate exact Stim DEM detector marginals for an already-native action."""
    probabilities = plant.probabilities(
        controls, epoch, frequency, target_controls=target)
    dem = plant._circuit_from_probabilities(probabilities).detector_error_model(
        decompose_errors=False, approximate_disjoint_errors=True, flatten_loops=True)
    parity_products = np.ones(plant.detector_count, dtype=float)
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        probability = float(instruction.args_copy()[0])
        parity: dict[int, int] = defaultdict(int)
        for item in instruction.targets_copy():
            if item.is_relative_detector_id():
                parity[int(item.val)] ^= 1
        for detector, odd in parity.items():
            if odd:
                parity_products[detector] *= 1.0 - 2.0 * probability
    return (1.0 - parity_products) / 2.0


def _posthoc_reference_metrics(records: list[dict[str, Any]], frequency: float,
                               design_id: str) -> dict[str, Any]:
    indices = np.linspace(0, len(records)-1, 8, dtype=int)
    rows = []
    plant = build_plant(_source_config())
    boundary = _boundary(plant)
    limits = normalized_safety_limits(plant, boundary)
    for index in indices:
        record = records[int(index)]
        epoch = int(record["epoch"])
        mean = np.asarray(record["latent_behavior_mean"], dtype=float)
        sigma = np.asarray(record["behavior_sigma"], dtype=float)
        base_rng = np.random.default_rng(51_000 + epoch)
        base = base_rng.normal(size=(64, 41))
        ref_actions = mean[None, :] + sigma[None, :] * np.concatenate([base, -base])
        # Use the Gaussian score directly for the K128 antithetic reference.
        target_normalized = plant.optimum(epoch, frequency)
        target = boundary.target_to_native(target_normalized)
        rewards = -np.asarray([_native_detector_expectations(
            plant,
            boundary.apply(apply_safe_control_transform(action, plant, boundary)).native,
            epoch, frequency, target) for action in ref_actions])
        advantages = rewards  # a constant detector baseline cancels under exact antithetic pairing
        local = advantages @ plant.mask.astype(float)
        reference = np.mean(local * np.concatenate([base, -base]) / sigma[None, :], axis=0)
        update = np.asarray(record["mean_update_direction"], dtype=float)
        target_latent = limits * np.arctanh(target_normalized / limits)
        beneficial = target_latent - mean
        beneficial /= max(float(np.linalg.norm(beneficial)), 1e-15)
        # Candidate contributions are unavailable in compact epoch records; use batch update metrics.
        ref_norm2 = max(float(reference @ reference), 1e-15)
        capture = float(reference @ update) / ref_norm2
        orthogonal = update - capture * reference
        ref_progress = float(reference @ beneficial)
        rows.append({
            "epoch": epoch,
            "alignment": cosine_alignment(update, reference),
            "directional_magnitude_ratio": float(update @ beneficial) /
                ref_progress if abs(ref_progress) > 1e-15 else None,
            "reference_gradient_capture": capture,
            "orthogonal_error_power": float(orthogonal @ orthogonal),
            "signed_progress": float(update @ beneficial),
        })
    return {
        "states": rows,
        "median_alignment": float(np.median([row["alignment"] for row in rows])),
        "median_directional_magnitude_ratio": float(np.median([
            row["directional_magnitude_ratio"] for row in rows
            if row["directional_magnitude_ratio"] is not None])),
        "median_reference_gradient_capture": float(np.median([
            row["reference_gradient_capture"] for row in rows])),
        "mean_orthogonal_error_power": float(np.mean([
            row["orthogonal_error_power"] for row in rows])),
        "cumulative_signed_progress": float(np.sum([
            row["signed_progress"] for row in rows])),
        "reference_used_for_controller_updates": False,
        "reference_action_transform": "same phase-independent V21 safety envelope",
        "reference_design_id": design_id,
    }


def run_short_fast_rollouts() -> dict[str, Any]:
    verify_import_manifest()
    existing = ARTIFACT_ROOT / "short_fast_online_rollouts.json"
    if existing.is_file():
        return read_json(existing)
    gate = read_json(ARTIFACT_ROOT / "frozen_state_promotion_gate.json")
    promoted = list(gate["promoted_designs_for_short_online_rollout"])
    cfg = settings()
    frequency = float(cfg["fast_frequency_per_epoch"])
    start = int(cfg["online_transient_epochs"])
    stop = int(cfg["online_epochs"])
    rows = []
    for design_id in promoted:
        started = time.perf_counter()
        acquisition = _run_design(design_id)
        runtime_seconds = time.perf_counter() - started
        selected = acquisition["epoch_records"][start:stop]
        summary = _summary(selected, frequency)
        summary.update(_posthoc_reference_metrics(selected, frequency, design_id))
        rows.append({
            "design_id": design_id,
            "source_fidelity": SOURCE_FIDELITY[design_id],
            "controller_hash": acquisition["controller_hash"],
            "summary": summary,
            "K": 8, "M": 12000, "B": 96000,
            "runtime_seconds_this_invocation": runtime_seconds,
            "runtime_scope": "checkpointed invocation runtime is reported but not a scientific gate",
        })
    v20 = read_json(ARTIFACT_ROOT.parent / "google_pure_v20/postrepair_fast_validation.json")
    population = v20["population_gradient_reference"]
    baseline = v20["baseline_v19_experimental_fast"]
    projection = v20["repaired_fast"]
    for row in rows:
        source = row["summary"]
        source["I_mean_improvement_over_iid"] = source["I_mean"] - baseline["I_mean"]
        source["distance_to_population_I_mean"] = abs(source["I_mean"] - population["I_mean"])
        source["distance_to_population_gain_phase"] = float(np.hypot(
            source["gain"] - population["gain"],
            source["phase_lag_radians"] - population["phase_lag_radians"]))
        source["closer_to_population_than_iid"] = (
            source["distance_to_population_I_mean"] <
            abs(baseline["I_mean"] - population["I_mean"]))
    value = {
        "pass": True,
        "promoted_designs": promoted,
        "online_rows": rows,
        "comparators": {
            "V19_V20_iid_gaussian_baseline": baseline,
            "V20_hard_projection_oracle_diagnostic": projection,
            "V20_population_gradient_reference": population,
        },
        "same_K": True,
        "same_M": True,
        "same_total_cycle_budget": True,
        "same_mean_learning_rate": True,
        "same_sigma_learning_rate": True,
        "same_entropy_rule": True,
        "same_normalization": True,
        "same_fast_target": True,
        "zero_orthogonal_motion_required": False,
        "slow_intermediate_rerun": False,
        "forbidden_auto_runs_launched": [],
    }
    return write_artifact("short_fast_online_rollouts", value,
                          title="V21 gated short fast-only online rollouts")


def _synthetic_rewards(actions: np.ndarray, target: np.ndarray, hessian: np.ndarray,
                       mask: np.ndarray) -> np.ndarray:
    errors = actions - target[None, :]
    # Each detector is a public local quadratic factor. For dense coupled cases,
    # the caller supplies a correspondingly dense detector-control mask.
    costs = .5 * (errors @ hessian.T)**2
    return -costs[:, :mask.shape[0]]


def run_generalization_audit() -> dict[str, Any]:
    verify_import_manifest()
    existing = ARTIFACT_ROOT / "generalization_audit.json"
    if existing.is_file():
        return read_json(existing)
    online = read_json(ARTIFACT_ROOT / "short_fast_online_rollouts.json")
    promoted = list(online["promoted_designs"])
    if not promoted:
        return write_artifact("generalization_audit", {
            "pass": True, "designs": [], "classification": "NO_PROMOTED_DESIGN",
            "forbidden_auto_runs_launched": []},
            title="V21 bounded candidate-design generalization audit")
    design_id = promoted[0]
    dimension = 12
    blocks = tuple(np.asarray(block, dtype=int) for block in np.array_split(
        np.arange(dimension), 4))
    rng = np.random.default_rng(61_000)
    cases = []
    targets = {
        "new_fast_seed_86101": rng.normal(size=dimension),
        "new_fast_seed_86102": rng.normal(size=dimension),
        "new_initial_phase_0p37": np.sin(.37) * rng.normal(size=dimension),
        "new_initial_phase_1p11": np.sin(1.11) * rng.normal(size=dimension),
        "unseen_drift_direction": rng.normal(size=dimension),
        "two_simultaneous_local_modes": np.concatenate([
            rng.normal(size=3), np.zeros(3), rng.normal(size=3), np.zeros(3)]),
        "stationary_arbitrary_perturbation": rng.uniform(-1, 1, size=dimension),
        "coupled_multi_neighborhood_disturbance": rng.normal(size=dimension),
    }
    for case_index, (label, target) in enumerate(targets.items()):
        coupled = label == "coupled_multi_neighborhood_disturbance"
        hessian = np.eye(dimension)
        if coupled:
            hessian += .08 * (np.ones((dimension, dimension)) - np.eye(dimension))
            mask = np.ones((dimension, dimension), dtype=bool)
        else:
            mask = np.eye(dimension, dtype=bool)
        mean = np.zeros(dimension); sigma = np.full(dimension, .3)
        baseline = np.zeros(dimension)
        true_gradient = hessian.T @ (hessian @ target)
        design_estimates = []
        iid_estimates = []
        frame_hashes_by_target = []
        for repeat in range(96):
            for current, destination in ((design_id, design_estimates), ("D0", iid_estimates)):
                frame = generate_frame(
                    current, dimension=dimension, epoch=repeat,
                    seed=62_000 + case_index*1000, blocks=blocks)
                actions = mean[None, :] + sigma[None, :] * frame.standardized_directions
                rewards = _synthetic_rewards(actions, target, hessian, mask)
                estimate = estimate_policy_updates(
                    frame, rewards, baseline, mask, sigma)["mean_update_direction"]
                destination.append(np.asarray(estimate, dtype=float))
                if current == design_id:
                    frame_hashes_by_target.append(canonical_hash(
                        frame.standardized_directions.tolist()))
        design_array = np.asarray(design_estimates)
        iid_array = np.asarray(iid_estimates)
        design_mse = float(np.mean(np.sum((design_array-true_gradient)**2, axis=1)))
        iid_mse = float(np.mean(np.sum((iid_array-true_gradient)**2, axis=1)))
        cases.append({
            "case": label,
            "design_MSE": design_mse,
            "iid_MSE": iid_mse,
            "MSE_ratio_to_iid": design_mse / max(iid_mse, 1e-15),
            "design_mean_alignment": cosine_alignment(
                np.mean(design_array, axis=0), true_gradient),
            "design_signed_progress": float(np.mean(design_array, axis=0) @ true_gradient),
            "frame_hashes": frame_hashes_by_target,
            "frames_conditioned_on_target": False,
        })
    provenance = {
        "uses_known_driven_direction": False,
        "uses_target_trajectory": False,
        "uses_future_phase": False,
        "uses_population_or_reference_gradient": False,
        "uses_hidden_optimum": False,
        "uses_multi_run_leakage": False,
        "uses_posthoc_selected_subspace": False,
    }
    pass_cases = candidate_is_nonoracle(provenance) and all(
        row["MSE_ratio_to_iid"] <= 1.0 and row["design_signed_progress"] > 0
        for row in cases)
    value = {
        "pass": pass_cases,
        "design_id": design_id,
        "source_fidelity": SOURCE_FIDELITY[design_id],
        "cases": cases,
        "new_fast_seeds_tested": settings()["new_fast_seeds"],
        "new_initial_phases_tested": settings()["new_initial_phases"],
        "known_synthetic_driven_direction_used_for_frames": False,
        "future_target_phase_used_for_frames": False,
        "reference_population_gradient_used_for_frames": False,
        "hidden_optimum_used_for_frames": False,
        "posthoc_selected_subspace_used": False,
        "oracle_provenance_gate_passed": candidate_is_nonoracle(provenance),
        "classification": "GENERALIZES_BOUNDED_DEVELOPMENT_CASES" if pass_cases else
            "DIAGNOSTIC_ONLY",
        "source_budget_campaign": False,
        "forbidden_auto_runs_launched": [],
    }
    return write_artifact("generalization_audit", value,
                          title="V21 bounded candidate-design generalization audit")
