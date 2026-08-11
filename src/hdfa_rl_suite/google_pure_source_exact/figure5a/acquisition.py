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

from .contracts import (DIAGNOSTIC_STREAM_ACQUISITION_CONTRACT, Figure5aProtocol,
                        atomic_json, canonical_hash, ratio_from_raw_counts)
from .plant import Figure5aStimPlant


STREAMS = ("fixed", "optimal", "stochastic", "learned_mean")
DIAGNOSTIC_STREAMS = ("fixed", "optimal", "learned_mean")
STOCHASTIC_STREAM = "stochastic"
COORDINATE_CONTRACT = "SOURCE_GAUSSIAN_P_EQUALS_APPLIED_PLANT_P_V1"
FIGURE5A_IMPLEMENTATION_VERSION = "figure5a-source-coordinate-aggregated-diagnostics.v2"
CHECKPOINT_SCHEMA_VERSION = "figure5a-cell-checkpoint.v5"
ARTIFACT_SCHEMA_VERSION = "figure5a-cell.v5"
DIAGNOSTIC_ACQUISITION_MODE = DIAGNOSTIC_STREAM_ACQUISITION_CONTRACT


def _new_state(*, protocol: Figure5aProtocol, plant: Figure5aStimPlant, frequency: float,
               entropy_weight: float, seed: int, policy: DirectSigmaGaussianPolicy,
               optimizer: DirectSigmaOptimizer, baseline: np.ndarray,
               dependency_hashes: Mapping[str, str], controller_hash: str) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION, "protocol": asdict(protocol),
        "protocol_hash": protocol.protocol_hash, "plant_hash": plant.plant_hash,
        "frequency": float(frequency), "entropy_weight": float(entropy_weight), "seed": int(seed),
        "dependency_hashes": dict(dependency_hashes), "controller_hash": controller_hash,
        "coordinate_contract": COORDINATE_CONTRACT,
        "epoch": 0, "active_batch": None,
        "policy": policy.state_dict(optimizer_state=optimizer.state_dict(), baseline=baseline),
        "epoch_shards": [], "candidate_boundaries_completed": 0,
    }


def _load_runtime(state: dict[str, Any]) -> tuple[DirectSigmaGaussianPolicy, DirectSigmaOptimizer, np.ndarray]:
    policy = DirectSigmaGaussianPolicy.from_state_dict(state["policy"])
    optimizer = DirectSigmaOptimizer.from_state_dict(state["policy"]["optimizer_state"])
    baseline = np.asarray(state["policy"]["baseline"], dtype=float)
    return policy, optimizer, baseline


def _freeze_batch(policy: DirectSigmaGaussianPolicy, count: int) -> dict[str, Any]:
    """Freeze the Gaussian batch that is applied directly as source p."""
    batch = policy.sample(count)
    return {"gaussian_actions": batch.actions.tolist(),
            "applied_actions": batch.actions.tolist(),
            "standardized_noise": batch.standardized_noise.tolist(),
            "gaussian_behavior_mean": batch.behavior.mean.tolist(),
            "applied_behavior_mean": batch.behavior.mean.tolist(),
            "behavior_sigma": batch.behavior.sigma.tolist(),
            "behavior_component_log_probability": batch.behavior.component_log_probability.tolist(),
            "policy_version": batch.behavior.policy_version, "next_candidate": 0,
            "counts": {stream: [] for stream in STREAMS}, "stochastic_detector_counts": []}


def _behavior(active: Mapping[str, Any]) -> BehaviorSnapshot:
    return BehaviorSnapshot(np.asarray(active["gaussian_behavior_mean"]), np.asarray(active["behavior_sigma"]),
                            np.asarray(active["behavior_component_log_probability"]),
                            int(active["policy_version"]))


def source_controls_for_epoch(
    plant: Figure5aStimPlant, *, epoch: int, frequency: float,
    stochastic: np.ndarray, learned_mean: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return the literal Supplement-VI.A controls used by acquisition."""
    optimum = plant.optimum(epoch, frequency)
    stochastic = np.asarray(stochastic, dtype=float)
    learned_mean = np.asarray(learned_mean, dtype=float)
    if stochastic.shape != (plant.control_count,) or learned_mean.shape != (plant.control_count,):
        raise ValueError("Figure 5a policy controls must be 41-vectors")
    controls = {
        "fixed": np.zeros(plant.control_count, dtype=float),
        "optimal": optimum.copy(),
        "stochastic": stochastic.copy(),
        "learned_mean": learned_mean.copy(),
    }
    return optimum, controls


def run_cell(*, protocol: Figure5aProtocol, plant: Figure5aStimPlant, frequency: float,
             entropy_weight: float, seed: int, optimizer_config: OptimizerConfig,
             initial_sigma: float, checkpoint_path: Path, dependency_hashes: Mapping[str, str],
             controller_hash: str, clip: float = 0.2, baseline_weight: float = 0.2,
             resume: bool = False, max_candidate_boundaries: int | None = None,
             boundary_callback: Callable[[dict[str, Any]], None] | None = None,
             checkpoint_every_candidates: int = 1,
             fresh_acquisition_required: bool = False,
             source_budget_profile: str = "UNSPECIFIED_DEVELOPMENT") -> dict[str, Any]:
    """Execute or resume a cell; checkpoint atomically after every candidate boundary."""
    if frequency <= 0 or entropy_weight <= 0 or initial_sigma <= 0:
        raise ValueError("frequency, entropy weight, and initial sigma must be positive")
    if checkpoint_every_candidates < 1:
        raise ValueError("checkpoint_every_candidates must be at least one")
    if protocol.circuit_rounds != plant.rounds:
        raise ValueError("protocol and Stim plant circuit rounds differ")
    checkpoint_preexisted = checkpoint_path.exists()
    if checkpoint_preexisted and fresh_acquisition_required:
        raise RuntimeError("fresh source acquisition forbids reuse of a lower-level checkpoint")
    if checkpoint_path.exists():
        if not resume:
            raise RuntimeError(f"checkpoint exists; pass resume=True: {checkpoint_path}")
        state = __import__("json").loads(checkpoint_path.read_text(encoding="utf-8"))
        if state.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise RuntimeError(
                "checkpoint acquisition layout is obsolete; start a fresh cell under the "
                "epoch-aggregated diagnostic-stream contract")
        identity = (state["protocol_hash"], state["plant_hash"], state["frequency"],
                    state["entropy_weight"], state["seed"], state["dependency_hashes"],
                    state.get("controller_hash"), state.get("coordinate_contract"))
        expected = (protocol.protocol_hash, plant.plant_hash, float(frequency), float(entropy_weight),
                    int(seed), dict(dependency_hashes), controller_hash,
                    COORDINATE_CONTRACT)
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
                           controller_hash=controller_hash)
        atomic_json(checkpoint_path, state)
    completed_this_call = 0
    while int(state["epoch"]) < protocol.epochs:
        epoch = int(state["epoch"])
        if state["active_batch"] is None:
            state["active_batch"] = _freeze_batch(policy, protocol.candidates_per_epoch)
            state["policy"] = policy.state_dict(optimizer_state=optimizer.state_dict(), baseline=baseline)
            atomic_json(checkpoint_path, state)
        active = state["active_batch"]
        # Reconstruct epoch-level controls even when resuming exactly after the
        # final candidate boundary, in which case the sampling loop is skipped.
        optimum, controls = source_controls_for_epoch(
            plant, epoch=epoch, frequency=frequency,
            stochastic=np.asarray(active["applied_actions"])[-1],
            learned_mean=np.asarray(active["applied_behavior_mean"]))
        if not all(active["counts"][stream] for stream in DIAGNOSTIC_STREAMS):
            if any(active["counts"][stream] for stream in DIAGNOSTIC_STREAMS):
                raise RuntimeError("partial epoch diagnostic-stream aggregation is invalid")
            diagnostic_qec_cycles = (
                protocol.candidates_per_epoch * protocol.qec_cycles_per_candidate)
            for stream in DIAGNOSTIC_STREAMS:
                observation = plant.sample_detector_observation(
                    controls[stream], epoch=epoch, frequency=frequency,
                    qec_cycles=diagnostic_qec_cycles,
                    seed=plant.stream_seed(
                        seed, f"{stream}:epoch-aggregate", epoch, 0),
                    target_controls=optimum)
                active["counts"][stream].append(observation.raw_total)
            state["active_batch"] = active
            atomic_json(checkpoint_path, state)
        while int(active["next_candidate"]) < protocol.candidates_per_epoch:
            candidate = int(active["next_candidate"])
            optimum, controls = source_controls_for_epoch(
                plant, epoch=epoch, frequency=frequency,
                stochastic=np.asarray(active["applied_actions"][candidate]),
                learned_mean=np.asarray(active["applied_behavior_mean"]))
            observation = plant.sample_detector_observation(
                controls[STOCHASTIC_STREAM], epoch=epoch, frequency=frequency,
                qec_cycles=protocol.qec_cycles_per_candidate,
                seed=plant.stream_seed(seed, STOCHASTIC_STREAM, epoch, candidate),
                target_controls=optimum)
            active["counts"][STOCHASTIC_STREAM].append(observation.raw_total)
            active["stochastic_detector_counts"].append(
                observation.reward_component_counts.tolist())
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
            np.asarray(active["gaussian_actions"]), rewards, plant.mask, policy.mean, policy.sigma,
            baseline, behavior, clip=float(clip), entropy_weight=float(entropy_weight),
            baseline_weight=float(baseline_weight))
        update = optimizer.step(policy.mean, policy.sigma, baseline, loss.grad_mean,
                                loss.grad_sigma, loss.grad_baseline)
        policy.policy_version += 1
        record = {
            "epoch": epoch, "optimum": float(optimum[0]),
            "plant_target_controls": optimum.tolist(),
            "optimal_controls": controls["optimal"].tolist(),
            "controller_mode": "PAPER_DIRECT_SIGMA",
            "parameterization": "direct_sigma",
            "ratio_clipping_mode": loss.diagnostics["ratio_clipping_mode"],
            "baseline_mode": loss.diagnostics["baseline_mode"],
            "coordinate_ratios_clipped_before_sparse_product":
                loss.diagnostics["coordinate_ratios_clipped_before_sparse_product"],
            "component_clip_fraction": loss.diagnostics["component_clip_fraction"],
            "detector_clip_fraction": loss.diagnostics["detector_clip_fraction"],
            "behavior_mean": active["applied_behavior_mean"],
            "gaussian_behavior_mean": active["gaussian_behavior_mean"],
            "behavior_sigma": active["behavior_sigma"],
            "post_update_mean": policy.mean.tolist(),
            "post_update_gaussian_mean": policy.mean.tolist(),
            "post_update_sigma": policy.sigma.tolist(),
            "action_execution": "identity_applied_gaussian",
            "plant_boundary_execution": "none_source_coordinate_identity",
            "likelihood_space": "applied_gaussian",
            "entropy_space": "applied_gaussian",
            "coordinate_contract": COORDINATE_CONTRACT,
            "action_transform_uses_hidden_optimum": False,
            "action_transform_applied": False,
            "maximum_abs_gaussian_applied_delta": float(np.max(np.abs(
                np.asarray(active["gaussian_actions"]) - np.asarray(active["applied_actions"])))),
            "source_optimum_applied_directly": bool(np.array_equal(
                optimum, controls["optimal"])),
            "policy_entropy": entropy(np.asarray(active["behavior_sigma"])),
            "counts": {stream: list(map(int, active["counts"][stream])) for stream in STREAMS},
            "stream_totals": {stream: int(sum(active["counts"][stream])) for stream in STREAMS},
            "stream_acquisition": {
                "mode": DIAGNOSTIC_ACQUISITION_MODE,
                "stochastic_acquisitions": protocol.candidates_per_epoch,
                "diagnostic_acquisitions": len(DIAGNOSTIC_STREAMS),
                "total_circuit_compilations": protocol.candidates_per_epoch + len(DIAGNOSTIC_STREAMS),
                "diagnostic_aggregation_factor": protocol.candidates_per_epoch,
                "qec_cycles_per_stochastic_acquisition": protocol.qec_cycles_per_candidate,
                "qec_cycles_per_diagnostic_acquisition":
                    protocol.candidates_per_epoch * protocol.qec_cycles_per_candidate,
                "all_four_stream_qec_budgets_unchanged": True,
                "stochastic_training_seed_contract_unchanged": True,
                "aggregate_count_distribution":
                    "equal_in_distribution_to_sum_of_candidate_count_batches"},
            "stochastic_detector_counts": active["stochastic_detector_counts"],
            "reward_sigma_gradient_norm": loss.diagnostics["reward_sigma_gradient_norm"],
            "entropy_sigma_gradient_norm": loss.diagnostics["entropy_sigma_gradient_norm"],
            "fraction_at_positivity_guard": update["fraction_at_positivity_guard"],
            "gradient_clipping": {key: update[key] for key in (
                "gradient_clipping_mode", "gradient_clip_threshold",
                "gradient_global_l2_norm_before_clipping",
                "gradient_global_l2_norm_after_clipping", "gradient_global_clip_scale",
                "gradient_component_count", "gradient_clipped_component_count",
                "gradient_clipped_component_fraction",
                "raw_mean_gradient_l2_norm", "raw_sigma_gradient_l2_norm",
                "raw_baseline_gradient_l2_norm", "applied_mean_gradient_l2_norm",
                "applied_sigma_gradient_l2_norm", "applied_baseline_gradient_l2_norm")},
            "candidate_count": protocol.candidates_per_epoch,
            "qec_cycles_per_candidate": protocol.qec_cycles_per_candidate,
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
        "schema_version": ARTIFACT_SCHEMA_VERSION, "complete": True,
        "protocol": state["protocol"], "protocol_hash": state["protocol_hash"],
        "plant_hash": plant.plant_hash, "dependency_hashes": dict(dependency_hashes),
        "controller_hash": controller_hash,
        "implementation_version": FIGURE5A_IMPLEMENTATION_VERSION,
        "control_order_hash": canonical_hash(list(plant.parameter_ids)),
        "frequency": float(frequency), "entropy_weight": float(entropy_weight), "seed": int(seed),
        "parameterization": "DIRECT_SIGMA_SOURCE_EXACT", "control_count": 41,
        "action_execution": "identity_applied_gaussian",
        "plant_boundary_execution": "none_source_coordinate_identity",
        "likelihood_space": "applied_gaussian",
        "entropy_space": "applied_gaussian",
        "coordinate_contract": COORDINATE_CONTRACT,
        "action_transform_applied": False,
        "action_transform_uses_hidden_optimum": False,
        "empirical_relative_normalization_applied": False,
        "mean_bounds_applied": False,
        "detector_count": plant.detector_count, "raw_detector_count": plant.raw_detector_count,
        "reward_representation": "time_translation_equivalence_class_mean_edr",
        "stream_acquisition_contract": {
            "mode": DIAGNOSTIC_ACQUISITION_MODE,
            "stochastic_stream": "one finite-shot acquisition per sampled candidate",
            "diagnostic_streams": list(DIAGNOSTIC_STREAMS),
            "diagnostic_stream_execution":
                "one finite-shot acquisition per epoch at candidates_per_epoch times the per-candidate QEC budget",
            "diagnostic_controls_constant_within_epoch": True,
            "all_four_stream_qec_budgets_unchanged": True,
            "stochastic_training_seed_contract_unchanged": True,
            "exact_DEM_diagnostics_used": False},
        "gradient_clipping_contract": {
            "mode": optimizer_config.gradient_clipping_mode.value,
            "threshold": optimizer_config.gradient_clip_threshold,
            "source_identifiability": "SOURCE_UNSPECIFIED_PREREGISTERED_NUISANCE",
            "applied_before_momentum": True,
            "global_l2_scope": "mean_sigma_and_detector_baseline_joint"},
        "stream_totals": totals,
        "stochastic_ratio": ratios, "learned_mean_ratio": learned_ratios,
        "finite_shot_denominator_nonzero": finite_shot_denominator_nonzero,
        "epoch_records": epoch_records,
        "candidate_qec_cycles": protocol.candidate_qec_cycles,
        "four_stream_qec_cycles": protocol.four_stream_qec_cycles,
        "circuit_compilations": protocol.epochs * (
            protocol.candidates_per_epoch + len(DIAGNOSTIC_STREAMS)),
        "candidate_boundaries_completed": state["candidate_boundaries_completed"],
        "checkpoint_every_candidates": int(checkpoint_every_candidates),
        "no_candidates_dropped": state["candidate_boundaries_completed"] == protocol.epochs * protocol.candidates_per_epoch,
        "epoch_shards": state["epoch_shards"],
        "fresh_acquisition": not checkpoint_preexisted,
        "reused_shard_ids": [],
        "source_budget_profile": str(source_budget_profile),
        "artifact_hash": canonical_hash({key: value for key, value in state.items() if key != "policy"}),
    }
    return artifact


def substitution_identity(counts: Mapping[str, int]) -> dict[str, float]:
    fixed = ratio_from_raw_counts(counts["fixed"], counts["fixed"], counts["optimal"])
    optimal = ratio_from_raw_counts(counts["optimal"], counts["fixed"], counts["optimal"])
    if fixed["source_ratio"] != 0.0 or optimal["source_ratio"] != 1.0:
        raise AssertionError("raw-count substitution identities failed")
    return {"fixed_substitution": fixed["source_ratio"], "optimal_substitution": optimal["source_ratio"]}
