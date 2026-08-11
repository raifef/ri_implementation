"""Candidate-boundary resumable four-stream Figure 5a acquisition."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.gaussian import (
    BehaviorSnapshot,
    DirectSigmaGaussianPolicy,
    entropy,
)
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.losses import total_loss_and_gradients
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.optimizer import (
    DirectSigmaOptimizer,
    OptimizerConfig,
)

from .contracts import Figure5aProtocol, atomic_json, canonical_hash, ratio_from_raw_counts
from .plant import Figure5aStimPlant
from hdfa_rl_suite.google_pure_source_exact.source_normalization import (
    SourceNormalizationBoundary,
    require_v15_boundary_provenance,
)


STREAMS = ("fixed", "optimal", "stochastic", "learned_mean")


def _new_state(*, protocol: Figure5aProtocol, plant: Figure5aStimPlant, frequency: float,
               entropy_weight: float, seed: int, policy: DirectSigmaGaussianPolicy,
               optimizer: DirectSigmaOptimizer, baseline: np.ndarray,
               dependency_hashes: Mapping[str, str], controller_hash: str,
               boundary: SourceNormalizationBoundary) -> dict[str, Any]:
    return {
        "schema_version": "figure5a-cell-checkpoint.v2", "protocol": asdict(protocol),
        "protocol_hash": protocol.protocol_hash, "plant_hash": plant.plant_hash,
        "frequency": float(frequency), "entropy_weight": float(entropy_weight), "seed": int(seed),
        "dependency_hashes": dict(dependency_hashes), "controller_hash": controller_hash,
        "v15_boundary": boundary.provenance_fields(),
        "epoch": 0, "active_batch": None,
        "policy": policy.state_dict(optimizer_state=optimizer.state_dict(), baseline=baseline),
        "epoch_shards": [], "candidate_boundaries_completed": 0,
    }


def _load_runtime(state: dict[str, Any]) -> tuple[DirectSigmaGaussianPolicy, DirectSigmaOptimizer, np.ndarray]:
    policy = DirectSigmaGaussianPolicy.from_state_dict(state["policy"])
    optimizer = DirectSigmaOptimizer.from_state_dict(state["policy"]["optimizer_state"])
    baseline = np.asarray(state["policy"]["baseline"], dtype=float)
    return policy, optimizer, baseline


def _freeze_batch(policy: DirectSigmaGaussianPolicy, plant: Figure5aStimPlant,
                  boundary: SourceNormalizationBoundary, count: int) -> dict[str, Any]:
    batch = policy.sample(count)
    normalized_actions = plant.apply_control_transform(batch.actions)
    normalized_mean = plant.apply_control_transform(batch.behavior.mean)
    applied_actions = boundary.apply(normalized_actions).native
    applied_mean = boundary.apply(normalized_mean).native
    return {"latent_actions": batch.actions.tolist(), "normalized_actions": normalized_actions.tolist(),
            "applied_actions": applied_actions.tolist(),
            "standardized_noise": batch.standardized_noise.tolist(),
            "latent_behavior_mean": batch.behavior.mean.tolist(),
            "normalized_behavior_mean": normalized_mean.tolist(),
            "applied_behavior_mean": applied_mean.tolist(),
            "behavior_sigma": batch.behavior.sigma.tolist(),
            "behavior_component_log_probability": batch.behavior.component_log_probability.tolist(),
            "policy_version": batch.behavior.policy_version, "next_candidate": 0,
            "counts": {stream: [] for stream in STREAMS}, "stochastic_detector_counts": []}


def _behavior(active: Mapping[str, Any]) -> BehaviorSnapshot:
    return BehaviorSnapshot(np.asarray(active["latent_behavior_mean"]), np.asarray(active["behavior_sigma"]),
                            np.asarray(active["behavior_component_log_probability"]),
                            int(active["policy_version"]))


def run_cell(*, protocol: Figure5aProtocol, plant: Figure5aStimPlant, frequency: float,
             entropy_weight: float, seed: int, optimizer_config: OptimizerConfig,
             initial_sigma: float, checkpoint_path: Path, dependency_hashes: Mapping[str, str],
             controller_hash: str, clip: float = 0.2, baseline_weight: float = 0.2,
             resume: bool = False, max_candidate_boundaries: int | None = None,
             boundary_callback: Callable[[dict[str, Any]], None] | None = None,
             checkpoint_every_candidates: int = 1,
             boundary: SourceNormalizationBoundary | None = None,
             fresh_acquisition_required: bool = False,
             source_budget_profile: str = "UNSPECIFIED_DEVELOPMENT") -> dict[str, Any]:
    """Execute or resume a cell; checkpoint atomically after every candidate boundary."""
    if frequency <= 0 or entropy_weight <= 0 or initial_sigma <= 0:
        raise ValueError("frequency, entropy weight, and initial sigma must be positive")
    if checkpoint_every_candidates < 1:
        raise ValueError("checkpoint_every_candidates must be at least one")
    degree = np.sum(plant.mask, axis=0).astype(float)
    curvature = np.asarray([item.omega_sensitivity for item in plant.inventory]) * degree
    boundary = boundary or SourceNormalizationBoundary.from_training_objective(
        "FIGURE5A_REAL_TIME_STEERING", curvature, control_ids=plant.parameter_ids)
    checkpoint_preexisted = checkpoint_path.exists()
    if checkpoint_preexisted and fresh_acquisition_required:
        raise RuntimeError("fresh V15 acquisition forbids reuse of a lower-level checkpoint")
    if checkpoint_path.exists():
        if not resume:
            raise RuntimeError(f"checkpoint exists; pass resume=True: {checkpoint_path}")
        state = __import__("json").loads(checkpoint_path.read_text(encoding="utf-8"))
        identity = (state["protocol_hash"], state["plant_hash"], state["frequency"],
                    state["entropy_weight"], state["seed"], state["dependency_hashes"],
                    state.get("controller_hash"), state.get("v15_boundary"))
        expected = (protocol.protocol_hash, plant.plant_hash, float(frequency), float(entropy_weight),
                    int(seed), dict(dependency_hashes), controller_hash,
                    boundary.provenance_fields())
        if identity != expected:
            raise RuntimeError("resume rejected: checkpoint identity changed")
        policy, optimizer, baseline = _load_runtime(state)
    else:
        policy = DirectSigmaGaussianPolicy(np.zeros(41), np.full(41, initial_sigma), seed=seed)
        optimizer = DirectSigmaOptimizer(41, plant.detector_count, optimizer_config)
        baseline = np.zeros(plant.detector_count)
        state = _new_state(protocol=protocol, plant=plant, frequency=frequency,
                           entropy_weight=entropy_weight, seed=seed, policy=policy,
                           optimizer=optimizer, baseline=baseline, dependency_hashes=dependency_hashes,
                           controller_hash=controller_hash, boundary=boundary)
        atomic_json(checkpoint_path, state)
    completed_this_call = 0
    while int(state["epoch"]) < protocol.epochs:
        epoch = int(state["epoch"])
        if state["active_batch"] is None:
            state["active_batch"] = _freeze_batch(policy, plant, boundary, protocol.candidates_per_epoch)
            state["policy"] = policy.state_dict(optimizer_state=optimizer.state_dict(), baseline=baseline)
            atomic_json(checkpoint_path, state)
        active = state["active_batch"]
        while int(active["next_candidate"]) < protocol.candidates_per_epoch:
            candidate = int(active["next_candidate"])
            optimum_normalized = plant.optimum(epoch, frequency)
            optimum = boundary.target_to_native(optimum_normalized)
            controls = {
                "fixed": boundary.target_to_native(np.zeros(41)), "optimal": optimum,
                "stochastic": np.asarray(active["applied_actions"][candidate]),
                "learned_mean": np.asarray(active["applied_behavior_mean"]),
            }
            detector_counts: dict[str, np.ndarray] = {}
            for stream in STREAMS:
                detector_counts[stream] = plant.sample_detector_counts(
                    controls[stream], epoch=epoch, frequency=frequency,
                    qec_cycles=protocol.qec_cycles_per_candidate,
                    seed=plant.stream_seed(seed, stream, epoch, candidate),
                    target_controls=optimum)
                active["counts"][stream].append(int(detector_counts[stream].sum()))
            active["stochastic_detector_counts"].append(detector_counts["stochastic"].tolist())
            active["next_candidate"] = candidate + 1
            state["candidate_boundaries_completed"] += 1
            state["active_batch"] = active
            flush_boundary = (active["next_candidate"] % checkpoint_every_candidates == 0 or
                              active["next_candidate"] == protocol.candidates_per_epoch)
            if flush_boundary:
                atomic_json(checkpoint_path, state)
            if boundary_callback is not None:
                boundary_callback(state)
            completed_this_call += 1
            if max_candidate_boundaries is not None and completed_this_call >= max_candidate_boundaries:
                if not flush_boundary:
                    atomic_json(checkpoint_path, state)
                return {"complete": False, "checkpoint_path": str(checkpoint_path.resolve()),
                        "candidate_boundaries_completed": state["candidate_boundaries_completed"],
                        "epoch": state["epoch"], "next_candidate": active["next_candidate"]}
        behavior = _behavior(active)
        stochastic_counts = np.asarray(active["stochastic_detector_counts"], dtype=float)
        rewards = -stochastic_counts / protocol.shots_per_policy
        loss = total_loss_and_gradients(
            np.asarray(active["latent_actions"]), rewards, plant.mask, policy.mean, policy.sigma,
            baseline, behavior, clip=float(clip), entropy_weight=float(entropy_weight),
            baseline_weight=float(baseline_weight))
        update = optimizer.step(policy.mean, policy.sigma, baseline, loss.grad_mean,
                                loss.grad_sigma, loss.grad_baseline, mean_bounds=(-2.0, 2.0))
        policy.policy_version += 1
        record = {
            "epoch": epoch, "optimum": float(optimum_normalized[0]),
            "controller_mode": "PAPER_DIRECT_SIGMA",
            "parameterization": "direct_sigma",
            "ratio_clipping_mode": loss.diagnostics["ratio_clipping_mode"],
            "baseline_mode": loss.diagnostics["baseline_mode"],
            "coordinate_ratios_clipped_before_sparse_product":
                loss.diagnostics["coordinate_ratios_clipped_before_sparse_product"],
            "component_clip_fraction": loss.diagnostics["component_clip_fraction"],
            "detector_clip_fraction": loss.diagnostics["detector_clip_fraction"],
            "behavior_mean": active["applied_behavior_mean"],
            "normalized_behavior_mean": active["normalized_behavior_mean"],
            "latent_behavior_mean": active["latent_behavior_mean"],
            "behavior_sigma": active["behavior_sigma"],
            "post_update_mean": boundary.apply(plant.apply_control_transform(policy.mean)).native.tolist(),
            "post_update_normalized_mean": plant.apply_control_transform(policy.mean).tolist(),
            "post_update_latent_mean": policy.mean.tolist(),
            "post_update_sigma": policy.sigma.tolist(),
            "action_execution": "plant_derived_per_coordinate_scaled_tanh",
            "plant_boundary_execution": "v15_source_normalized_to_native_once",
            "likelihood_space": "latent_gaussian",
            "action_transform_uses_hidden_optimum": False,
            "control_limits": plant.control_limits.tolist(),
            "maximum_abs_latent_applied_delta": float(np.max(np.abs(
                np.asarray(active["latent_actions"]) - np.asarray(active["applied_actions"])))),
            "policy_entropy": entropy(np.asarray(active["behavior_sigma"])),
            "counts": {stream: list(map(int, active["counts"][stream])) for stream in STREAMS},
            "stream_totals": {stream: int(sum(active["counts"][stream])) for stream in STREAMS},
            "stochastic_detector_counts": active["stochastic_detector_counts"],
            "reward_sigma_gradient_norm": loss.diagnostics["reward_sigma_gradient_norm"],
            "entropy_sigma_gradient_norm": loss.diagnostics["entropy_sigma_gradient_norm"],
            "fraction_at_positivity_guard": update["fraction_at_positivity_guard"],
            "candidate_count": protocol.candidates_per_epoch,
            "qec_cycles_per_candidate": protocol.qec_cycles_per_candidate,
            **boundary.provenance_fields(),
        }
        epoch_directory = checkpoint_path.parent / f"{checkpoint_path.stem}_epochs"
        epoch_path = epoch_directory / f"epoch-{epoch:04d}.json"
        record_hash = canonical_hash(record)
        atomic_json(epoch_path, {"record_hash": record_hash, "record": record})
        state["epoch_shards"].append({"epoch": epoch, "path": str(epoch_path.resolve()),
                                      "record_hash": record_hash})
        state["epoch"] = epoch + 1
        state["active_batch"] = None
        state["policy"] = policy.state_dict(optimizer_state=optimizer.state_dict(), baseline=baseline)
        atomic_json(checkpoint_path, state)
    epoch_records = []
    for shard in state["epoch_shards"]:
        payload = __import__("json").loads(Path(shard["path"]).read_text(encoding="utf-8"))
        if payload["record_hash"] != shard["record_hash"] or canonical_hash(payload["record"]) != shard["record_hash"]:
            raise RuntimeError("epoch shard corruption detected")
        epoch_records.append(payload["record"])
    if [record["epoch"] for record in epoch_records] != list(range(protocol.epochs)):
        raise RuntimeError("missing or duplicate epoch shard")
    totals = {stream: int(sum(record["stream_totals"][stream] for record in epoch_records))
              for stream in STREAMS}
    finite_shot_denominator_nonzero = totals["optimal"] != totals["fixed"]
    if finite_shot_denominator_nonzero:
        ratios = ratio_from_raw_counts(totals["stochastic"], totals["fixed"], totals["optimal"])
        learned_ratios = ratio_from_raw_counts(totals["learned_mean"], totals["fixed"], totals["optimal"])
    else:
        if protocol.mode.value == "reference":
            raise RuntimeError("reference cell has a zero finite-shot fixed/optimal denominator")
        ratios = learned_ratios = {"source_ratio": None, "positive_cost_ratio": None}
    artifact = {
        "schema_version": "figure5a-cell.v2", "complete": True,
        "protocol": state["protocol"], "protocol_hash": state["protocol_hash"],
        "plant_hash": plant.plant_hash, "dependency_hashes": dict(dependency_hashes),
        "controller_hash": controller_hash,
        "frequency": float(frequency), "entropy_weight": float(entropy_weight), "seed": int(seed),
        "parameterization": "DIRECT_SIGMA_SOURCE_EXACT", "control_count": 41,
        "action_execution": "plant_derived_per_coordinate_scaled_tanh",
        "plant_boundary_execution": "v15_source_normalized_to_native_once",
        "likelihood_space": "latent_gaussian",
        "action_transform_invertible": True,
        "action_transform_uses_hidden_optimum": False,
        "control_limits": plant.control_limits.tolist(),
        "detector_count": plant.detector_count, "stream_totals": totals,
        "stochastic_ratio": ratios, "learned_mean_ratio": learned_ratios,
        "finite_shot_denominator_nonzero": finite_shot_denominator_nonzero,
        "epoch_records": epoch_records,
        "candidate_qec_cycles": protocol.candidate_qec_cycles,
        "four_stream_qec_cycles": protocol.four_stream_qec_cycles,
        "candidate_boundaries_completed": state["candidate_boundaries_completed"],
        "checkpoint_every_candidates": int(checkpoint_every_candidates),
        "no_candidates_dropped": state["candidate_boundaries_completed"] == protocol.epochs * protocol.candidates_per_epoch,
        "epoch_shards": state["epoch_shards"],
        "fresh_acquisition": not checkpoint_preexisted,
        "reused_shard_ids": [],
        "source_budget_profile": str(source_budget_profile),
        "boundary_trace": boundary.trace(np.eye(1, 41, 0).ravel()),
        **boundary.provenance_fields(),
        "artifact_hash": canonical_hash({key: value for key, value in state.items() if key != "policy"}),
    }
    require_v15_boundary_provenance(artifact)
    return artifact


def substitution_identity(counts: Mapping[str, int]) -> dict[str, float]:
    fixed = ratio_from_raw_counts(counts["fixed"], counts["fixed"], counts["optimal"])
    optimal = ratio_from_raw_counts(counts["optimal"], counts["fixed"], counts["optimal"])
    if fixed["source_ratio"] != 0.0 or optimal["source_ratio"] != 1.0:
        raise AssertionError("raw-count substitution identities failed")
    return {"fixed_substitution": fixed["source_ratio"], "optimal_substitution": optimal["source_ratio"]}
