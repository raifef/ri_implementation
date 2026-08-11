"""Independent checkpointed acquisition for the public-analogue scale branch.

The frozen production acquisition module is imported only for its state/batch
serialization helpers.  Its run loop and controller labels are not used.
"""
from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from hdfa_rl_suite.google_pure_source_exact.figure5a.acquisition import (
    STREAMS,
    _behavior,
    _freeze_batch,
    _load_runtime,
    _new_state,
)
from hdfa_rl_suite.google_pure_source_exact.figure5a.contracts import (
    Figure5aProtocol,
    atomic_json as _source_atomic_json,
    canonical_hash,
    ratio_from_raw_counts,
)
from hdfa_rl_suite.google_pure_source_exact.figure5a.plant import Figure5aStimPlant
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.gaussian import (
    DirectSigmaGaussianPolicy,
    entropy,
)
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.losses import (
    total_loss_and_gradients,
)
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.optimizer import (
    DirectSigmaOptimizer,
    OptimizerConfig,
)
from hdfa_rl_suite.google_pure_source_exact.source_normalization import (
    SourceNormalizationBoundary,
    require_v15_boundary_provenance,
)

from .controller import (
    CONTROLLER_MODE,
    PARAMETERIZATION,
    SCALE_OBJECTIVE,
    PublicAnalogueControllerSpec,
)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Retry a transient Windows sharing violation without weakening atomicity."""
    delays = (0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2)
    for attempt, delay in enumerate(delays):
        try:
            _source_atomic_json(path, value)
            return
        except PermissionError:
            if attempt == len(delays) - 1:
                raise
            time.sleep(delay)


def _experimental_identity(spec: PublicAnalogueControllerSpec) -> dict[str, Any]:
    return {
        "controller_mode": CONTROLLER_MODE,
        "parameterization": PARAMETERIZATION,
        "scale_objective": SCALE_OBJECTIVE,
        "source_exact": False,
        "source_scale_hyperparameters_identifiable": False,
        "inherited_entropy_coefficient": spec.inherited_entropy_coefficient,
        "active_dimensions": spec.active_dimensions,
        "effective_entropy_coefficient": spec.effective_entropy_coefficient,
        "frozen_parent_controller_hash": spec.frozen_parent_controller_hash,
    }


def run_experimental_cell(*, protocol: Figure5aProtocol, plant: Figure5aStimPlant,
                          frequency: float, seed: int, optimizer_config: OptimizerConfig,
                          controller: PublicAnalogueControllerSpec, checkpoint_path: Path,
                          dependency_hashes: Mapping[str, str],
                          boundary: SourceNormalizationBoundary,
                          resume: bool = False) -> dict[str, Any]:
    """Run one bounded cell with beta/P entropy and a distinct controller hash."""
    if frequency <= 0 or protocol.candidates_per_epoch < 2:
        raise ValueError("experimental frequency and candidate count must be positive")
    if controller.active_dimensions != len(plant.parameter_ids):
        raise ValueError("controller active dimension does not match the plant")
    identity = _experimental_identity(controller)
    checkpoint_preexisted = checkpoint_path.is_file()
    if checkpoint_preexisted:
        if not resume:
            raise RuntimeError(f"checkpoint exists; pass resume=True: {checkpoint_path}")
        state = __import__("json").loads(checkpoint_path.read_text(encoding="utf-8"))
        expected = {
            "protocol_hash": protocol.protocol_hash,
            "plant_hash": plant.plant_hash,
            "frequency": float(frequency),
            "seed": int(seed),
            "controller_hash": controller.controller_hash,
            "experimental_controller": identity,
            "v15_boundary": boundary.provenance_fields(),
        }
        observed = {key: state.get(key) for key in expected}
        if observed != expected:
            raise RuntimeError(f"experimental checkpoint identity changed: {observed}")
        stored_dependencies = dict(state.get("dependency_hashes", {}))
        current_dependencies = dict(dependency_hashes)
        stable_dependency_keys = set(current_dependencies) - {"experimental_acquisition_code"}
        dependency_mismatch = {
            key: {"stored": stored_dependencies.get(key), "current": current_dependencies.get(key)}
            for key in stable_dependency_keys
            if stored_dependencies.get(key) != current_dependencies.get(key)
        }
        if dependency_mismatch:
            raise RuntimeError(
                f"experimental checkpoint scientific dependency changed: {dependency_mismatch}")
        policy, optimizer, baseline = _load_runtime(state)
    else:
        policy = DirectSigmaGaussianPolicy(
            np.zeros(controller.active_dimensions),
            np.full(controller.active_dimensions, controller.initial_sigma), seed=seed)
        optimizer = DirectSigmaOptimizer(
            controller.active_dimensions, plant.detector_count, optimizer_config)
        baseline = np.zeros(plant.detector_count)
        state = _new_state(
            protocol=protocol, plant=plant, frequency=frequency,
            entropy_weight=controller.effective_entropy_coefficient, seed=seed,
            policy=policy, optimizer=optimizer, baseline=baseline,
            dependency_hashes=dependency_hashes, controller_hash=controller.controller_hash,
            boundary=boundary)
        state["experimental_controller"] = identity
        state["controller_mode"] = CONTROLLER_MODE
        state["parameterization"] = PARAMETERIZATION
        state["scale_objective"] = SCALE_OBJECTIVE
        atomic_json(checkpoint_path, state)

    while int(state["epoch"]) < protocol.epochs:
        epoch = int(state["epoch"])
        if state["active_batch"] is None:
            state["active_batch"] = _freeze_batch(
                policy, plant, boundary, protocol.candidates_per_epoch)
            state["policy"] = policy.state_dict(
                optimizer_state=optimizer.state_dict(), baseline=baseline)
            atomic_json(checkpoint_path, state)
        active = state["active_batch"]
        while int(active["next_candidate"]) < protocol.candidates_per_epoch:
            candidate = int(active["next_candidate"])
            optimum_normalized = plant.optimum(epoch, frequency)
            optimum = boundary.target_to_native(optimum_normalized)
            controls = {
                "fixed": boundary.target_to_native(np.zeros(controller.active_dimensions)),
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
            atomic_json(checkpoint_path, state)

        behavior = _behavior(active)
        stochastic_counts = np.asarray(active["stochastic_detector_counts"], dtype=float)
        rewards = -stochastic_counts / protocol.shots_per_policy
        loss = total_loss_and_gradients(
            np.asarray(active["latent_actions"]), rewards, plant.mask, policy.mean,
            policy.sigma, baseline, behavior, clip=controller.ppo_clip,
            entropy_weight=controller.effective_entropy_coefficient,
            baseline_weight=controller.baseline_loss_weight)
        update = optimizer.step(
            policy.mean, policy.sigma, baseline, loss.grad_mean, loss.grad_sigma,
            loss.grad_baseline, mean_bounds=(-2.0, 2.0))
        policy.policy_version += 1
        record = {
            "epoch": epoch,
            "optimum": float(plant.optimum(epoch, frequency)[0]),
            "controller_mode": CONTROLLER_MODE,
            "controller_hash": controller.controller_hash,
            "frozen_parent_controller_hash": controller.frozen_parent_controller_hash,
            "parameterization": PARAMETERIZATION,
            "scale_objective": SCALE_OBJECTIVE,
            "source_exact": False,
            "source_scale_hyperparameters_identifiable": False,
            "inherited_entropy_coefficient": controller.inherited_entropy_coefficient,
            "effective_entropy_coefficient": controller.effective_entropy_coefficient,
            "entropy_reduction_divisor": controller.active_dimensions,
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
            "post_update_mean": boundary.apply(
                plant.apply_control_transform(policy.mean)).native.tolist(),
            "post_update_normalized_mean": plant.apply_control_transform(policy.mean).tolist(),
            "post_update_latent_mean": policy.mean.tolist(),
            "post_update_sigma": policy.sigma.tolist(),
            "action_execution": "plant_derived_per_coordinate_scaled_tanh",
            "plant_boundary_execution": "v15_source_normalized_to_native_once",
            "likelihood_space": "latent_gaussian",
            "action_transform_uses_hidden_optimum": False,
            "control_limits": plant.control_limits.tolist(),
            "policy_entropy": entropy(np.asarray(active["behavior_sigma"])),
            "counts": {stream: list(map(int, active["counts"][stream])) for stream in STREAMS},
            "stream_totals": {
                stream: int(sum(active["counts"][stream])) for stream in STREAMS},
            "stochastic_detector_counts": active["stochastic_detector_counts"],
            "reward_sigma_gradient_norm": loss.diagnostics["reward_sigma_gradient_norm"],
            "entropy_sigma_gradient_norm": loss.diagnostics["entropy_sigma_gradient_norm"],
            "fraction_at_positivity_guard": update["fraction_at_positivity_guard"],
            "candidate_count": protocol.candidates_per_epoch,
            "qec_cycles_per_candidate": protocol.qec_cycles_per_candidate,
            **boundary.provenance_fields(),
        }
        epoch_directory = checkpoint_path.parent / "checkpoint_epochs"
        epoch_path = epoch_directory / f"epoch-{epoch:04d}.json"
        record_hash = canonical_hash(record)
        atomic_json(epoch_path, {"record_hash": record_hash, "record": record})
        state["epoch_shards"].append({
            "epoch": epoch, "path": str(epoch_path.resolve()), "record_hash": record_hash})
        state["epoch"] = epoch + 1
        state["active_batch"] = None
        state["policy"] = policy.state_dict(
            optimizer_state=optimizer.state_dict(), baseline=baseline)
        atomic_json(checkpoint_path, state)

    records = []
    for shard in state["epoch_shards"]:
        payload = __import__("json").loads(Path(shard["path"]).read_text(encoding="utf-8"))
        if (payload["record_hash"] != shard["record_hash"] or
                canonical_hash(payload["record"]) != shard["record_hash"]):
            raise RuntimeError("experimental epoch shard corruption detected")
        records.append(payload["record"])
    if [row["epoch"] for row in records] != list(range(protocol.epochs)):
        raise RuntimeError("experimental checkpoint has missing or duplicate epoch shards")
    totals = {stream: int(sum(row["stream_totals"][stream] for row in records))
              for stream in STREAMS}
    denominator_nonzero = totals["optimal"] != totals["fixed"]
    ratios = (ratio_from_raw_counts(
        totals["stochastic"], totals["fixed"], totals["optimal"])
              if denominator_nonzero else {"source_ratio": None, "positive_cost_ratio": None})
    learned = (ratio_from_raw_counts(
        totals["learned_mean"], totals["fixed"], totals["optimal"])
               if denominator_nonzero else {"source_ratio": None, "positive_cost_ratio": None})
    artifact = {
        "schema_version": "figure5a-public-analogue-experimental-cell.v1",
        "complete": True,
        "protocol": state["protocol"],
        "protocol_hash": state["protocol_hash"],
        "plant_hash": plant.plant_hash,
        "dependency_hashes": dict(state["dependency_hashes"]),
        "runtime_experimental_acquisition_code_sha256":
            dict(dependency_hashes)["experimental_acquisition_code"],
        "checkpoint_acquisition_code_revision_used": (
            dict(state["dependency_hashes"])["experimental_acquisition_code"] !=
            dict(dependency_hashes)["experimental_acquisition_code"]),
        "controller_hash": controller.controller_hash,
        **identity,
        "frequency": float(frequency),
        "seed": int(seed),
        "control_count": controller.active_dimensions,
        "detector_count": plant.detector_count,
        "stream_totals": totals,
        "stochastic_ratio": ratios,
        "learned_mean_ratio": learned,
        "finite_shot_denominator_nonzero": denominator_nonzero,
        "epoch_records": records,
        "candidate_qec_cycles": protocol.candidate_qec_cycles,
        "four_stream_qec_cycles": protocol.four_stream_qec_cycles,
        "candidate_boundaries_completed": state["candidate_boundaries_completed"],
        "no_candidates_dropped": (
            state["candidate_boundaries_completed"] ==
            protocol.epochs * protocol.candidates_per_epoch),
        "epoch_shards": state["epoch_shards"],
        "fresh_acquisition": not checkpoint_preexisted,
        "source_budget_profile": "V19_SMALL_DYNAMIC_PUBLIC_ANALOGUE",
        **boundary.provenance_fields(),
    }
    require_v15_boundary_provenance(artifact)
    return artifact
