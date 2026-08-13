"""Frozen V18/V19 inputs and exact Figure-5a evaluation helpers."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from hdfa_rl_suite.google_pure_source_exact.figure5a.bounded_action_ablation import Figure5aBoundedActionAblation

from hdfa_rl_suite.google_pure_source_exact.figure5a.contracts import canonical_hash as shard_hash
from hdfa_rl_suite.google_pure_source_exact.figure5a.validation import build_plant
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.gaussian import (
    BehaviorSnapshot,
    component_log_probability,
)
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.losses import (
    total_loss_and_gradients,
)
from hdfa_rl_suite.google_pure_v18.experiments import _boundary, _source_config
from hdfa_rl_suite.google_pure_v19_experimental.dynamic_validation import (
    _controller_spec,
)

from .io import (
    ARTIFACT_ROOT,
    ROOT,
    atomic_json,
    atomic_text,
    canonical_hash,
    file_hash,
    nonfinal,
    read_json,
    relative,
    settings,
)


EXPECTED_V19_CONTROLLER = "14c5243c4053d36b63c8f32537ad0a9309a29d91d78f73ff5401f1e716ae4af5"
EXPECTED_SOURCE_PARENT = "6b24d03aeb0f16ed8c9ed855755ebdb6b5e7cc8a558b4dfd9a646dcd6bfe5aa2"
FORBIDDEN_CAMPAIGNS = (
    "slow", "intermediate", "long-three-frequency", "source-budget", "heldout",
    "reference", "natural-drift", "figure5c", "paired-acceptance",
)


def import_paths() -> dict[str, Path]:
    matched = ROOT / "artifacts/google_pure_v19/experimental_public_analogue_matched"
    return {
        "v18_slow_transfer": ROOT / "artifacts/google_pure_v18/transfer_slow.json",
        "v18_intermediate_transfer": ROOT / "artifacts/google_pure_v18/transfer_intermediate.json",
        "v18_extended_fast_transfer": ROOT /
            "artifacts/google_pure_v18/extended_fast/transfer_fast_extended.json",
        "v19_diagnosis_root_cause": ROOT /
            "artifacts/google_pure_v19/root_cause_classification.json",
        "v19_diagnosis_repair": ROOT / "artifacts/google_pure_v19/minimal_repair.json",
        "v19_diagnosis_validation": ROOT /
            "artifacts/google_pure_v19/postrepair_validation.json",
        "v19_experimental_controller": ROOT /
            "src/hdfa_rl_suite/google_pure_v19_experimental/controller.py",
        "v19_experimental_acquisition": ROOT /
            "src/hdfa_rl_suite/google_pure_v19_experimental/acquisition.py",
        "v19_matched_protocol": ROOT /
            "configs/google_pure_v19/public_analogue_matched_dynamic_validation.json",
        "v19_matched_status": matched / "status.json",
        "v19_matched_slow_transfer": matched / "transfer_slow.json",
        "v19_matched_intermediate_transfer": matched / "transfer_intermediate.json",
        "v19_matched_fast_transfer": matched / "transfer_fast.json",
        "v19_matched_slow_checkpoint": matched / "acquisition/slow/checkpoint.json",
        "v19_matched_intermediate_checkpoint": matched /
            "acquisition/intermediate/checkpoint.json",
        "v19_matched_fast_checkpoint": matched / "acquisition/fast/checkpoint.json",
        "source_style_optimizer_bundle": ROOT /
            "artifacts/google_pure_v16/frozen_source_normalized_optimizer.json",
        "source_style_gaussian": ROOT /
            "src/hdfa_rl_suite/google_pure_source_exact/policy_parameterization/gaussian.py",
        "source_style_losses": ROOT /
            "src/hdfa_rl_suite/google_pure_source_exact/policy_parameterization/losses.py",
        "source_style_optimizer": ROOT /
            "src/hdfa_rl_suite/google_pure_source_exact/policy_parameterization/optimizer.py",
        "normalization_code": ROOT /
            "src/hdfa_rl_suite/google_pure_source_exact/source_normalization.py",
        "normalization_bundle": ROOT /
            "artifacts/google_pure_source_exact/control_normalization/calibration_bundle.json",
        "figure5a_config": ROOT / "configs/google_pure_source_exact/figure5a.json",
        "figure5a_plant": ROOT /
            "src/hdfa_rl_suite/google_pure_source_exact/figure5a/plant.py",
        "figure5a_evaluator": ROOT /
            "src/hdfa_rl_suite/google_pure_source_exact/figure5a/validation.py",
    }


def _observed_imports() -> dict[str, dict[str, str]]:
    paths = import_paths()
    missing = [relative(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing mandatory V20 input: {missing}")
    return {role: {"path": relative(path), "sha256": file_hash(path)}
            for role, path in paths.items()}


def build_import_manifest() -> dict[str, Any]:
    protocol = settings()
    observed = _observed_imports()
    path = ARTIFACT_ROOT / "import_manifest.json"
    if path.is_file():
        previous = read_json(path)
        if previous.get("inputs") != observed:
            changed = {key: {"expected": previous.get("inputs", {}).get(key),
                             "observed": value}
                       for key, value in observed.items()
                       if previous.get("inputs", {}).get(key) != value}
            raise RuntimeError(f"V20 frozen input mismatch: {changed}")
        return previous
    matched = read_json(import_paths()["v19_matched_status"])
    frozen = read_json(import_paths()["source_style_optimizer_bundle"])
    controller = _controller_spec()
    invariants = {
        "frozen_source_style_branch_unchanged": matched.get(
            "frozen_source_branch_unchanged") is True,
        "experimental_parent_unchanged": (
            matched.get("controller_hash") == controller.controller_hash ==
            protocol["frozen_v19_controller_hash"] == EXPECTED_V19_CONTROLLER and
            matched.get("frozen_parent_controller_hash") ==
            controller.frozen_parent_controller_hash == protocol[
                "frozen_source_parent_hash"] == EXPECTED_SOURCE_PARENT),
        "mean_lr_changed": False,
        "entropy_changed": False,
        "sigma_lr_changed": False,
        "normalization_changed": False,
    }
    gates = {
        "v19_execution_complete": matched.get("execution_complete") is True,
        "v19_fast_mean_failed": next(row for row in matched["rows"]
                                     if row["label"] == "fast")[
                                         "stream_decomposition"]["I_mean"] < 0,
        "mean_lr_matches_frozen_bundle": controller.mean_learning_rate ==
            float(frozen["mean_learning_rate"]),
        "sigma_lr_matches_frozen_bundle": controller.sigma_learning_rate ==
            float(frozen["sigma_learning_rate"]),
        "only_v19_entropy_division": math.isclose(
            controller.effective_entropy_coefficient,
            float(frozen["entropy_coefficient"]) / controller.active_dimensions,
            rel_tol=0, abs_tol=1e-15),
        "invariants": all(value is False for key, value in invariants.items()
                          if key.endswith("_changed")) and
            invariants["frozen_source_style_branch_unchanged"] and
            invariants["experimental_parent_unchanged"],
    }
    if not all(gates.values()):
        raise RuntimeError(f"V20 lineage gate failed: {gates}")
    value = nonfinal({
        "pass": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": observed,
        "invariants": invariants,
        "gates": gates,
        "frozen_v19_controller_mode": protocol["frozen_v19_controller_mode"],
        "frozen_v19_controller_hash": EXPECTED_V19_CONTROLLER,
        "frozen_source_parent_hash": EXPECTED_SOURCE_PARENT,
        "only_parent_objective_change": "entropy gradient beta -> beta/41",
        "forbidden_auto_runs": list(FORBIDDEN_CAMPAIGNS),
        "forbidden_auto_runs_launched": [],
    })
    atomic_json(path, value)
    atomic_text(ARTIFACT_ROOT / "import_manifest.md", "\n".join([
        "# V20 frozen lineage manifest", "",
        "V18 transfers, V19 diagnosis and matched acquisitions, the experimental controller, "
        "source-style policy stack, normalization, optimizer, and Figure 5a plant/evaluator are hash-pinned.",
        "",
        "No learning rate, entropy coefficient, scale bound, normalization, or frozen parent was changed.",
    ]))
    return value


def verify_import_manifest() -> dict[str, Any]:
    path = ARTIFACT_ROOT / "import_manifest.json"
    value = read_json(path) if path.is_file() else build_import_manifest()
    if value.get("inputs") != _observed_imports() or value.get("pass") is not True:
        raise RuntimeError("V20 import manifest no longer matches frozen inputs")
    return value


_RUN_CACHE: dict[str, dict[str, Any]] = {}


def load_matched_run(label: str) -> dict[str, Any]:
    if label in _RUN_CACHE:
        return _RUN_CACHE[label]
    if label not in {"slow", "intermediate", "fast"}:
        raise ValueError(f"unknown matched V19 condition: {label}")
    root = ROOT / "artifacts/google_pure_v19/experimental_public_analogue_matched"
    transfer_path = root / f"transfer_{label}.json"
    checkpoint_path = root / "acquisition" / label / "checkpoint.json"
    transfer, checkpoint = read_json(transfer_path), read_json(checkpoint_path)
    if checkpoint.get("controller_hash") != EXPECTED_V19_CONTROLLER:
        raise RuntimeError(f"{label} is not the frozen V19 experimental controller")
    records: list[dict[str, Any]] = []
    for shard in checkpoint["epoch_shards"]:
        payload = read_json(Path(shard["path"]))
        if payload.get("record_hash") != shard.get("record_hash") or \
                shard_hash(payload.get("record")) != shard.get("record_hash"):
            raise RuntimeError(f"corrupt matched V19 shard: {shard['path']}")
        records.append(payload["record"])
    epochs = int(checkpoint["epoch"])
    if len(records) != epochs or [int(row["epoch"]) for row in records] != list(range(epochs)):
        raise RuntimeError(f"{label} V19 checkpoint is incomplete or reordered")
    candidates = int(checkpoint["protocol"]["candidates_per_epoch"])
    controls = len(checkpoint["policy"]["mean"])
    rng = np.random.default_rng(int(checkpoint["seed"]))
    noises = np.asarray([rng.normal(size=(candidates, controls)) for _ in range(epochs)])
    if canonical_hash(rng.bit_generator.state) != canonical_hash(
            checkpoint["policy"]["rng_state"]):
        raise RuntimeError(f"{label} candidate stream cannot be reconstructed")
    result = {
        "label": label,
        "transfer": transfer,
        "checkpoint": checkpoint,
        "records": records,
        "noises": noises,
        "transfer_path": transfer_path,
        "checkpoint_path": checkpoint_path,
    }
    _RUN_CACHE[label] = result
    return result


def selected_fast_epochs(*, count: int | None = None) -> list[int]:
    cfg = settings()
    number = int(count if count is not None else cfg["phase_states"])
    run = load_matched_run("fast")
    period = int(run["transfer"]["period_epochs"])
    stop = int(run["transfer"]["analysis_epoch_window"][1])
    start = stop - period
    epochs = [start + min(period - 1, int(round((index + .5) * period / number - .5)))
              for index in range(number)]
    if len(set(epochs)) != number:
        raise RuntimeError("phase-state selection contains duplicates")
    return epochs


class ExactDetectorEvaluator:
    """Exact Stim detector marginals under the frozen V15 normalization boundary."""

    def __init__(self) -> None:
        self.plant = build_plant(_source_config())
        self.bounded = Figure5aBoundedActionAblation(self.plant)
        self.boundary = _boundary(self.plant)
        self._cache: dict[tuple[float, float, bytes], np.ndarray] = {}

    def normalized_to_latent(self, normalized: np.ndarray) -> np.ndarray:
        return self.bounded.latent_controls_for(np.asarray(normalized, dtype=float))

    def native(self, latent: np.ndarray) -> np.ndarray:
        normalized = self.bounded.apply_control_transform(np.asarray(latent, dtype=float))
        return self.boundary.apply(normalized).native

    def detector_expectations(self, latent: np.ndarray, epoch: float,
                              frequency: float) -> np.ndarray:
        value = np.asarray(latent, dtype="<f8")
        key = (float(epoch), float(frequency), value.tobytes(order="C"))
        cached = self._cache.get(key)
        if cached is not None:
            return cached.copy()
        controls = self.native(value)
        target_normalized = np.full(41, math.sin(2.0 * math.pi * float(frequency) *
                                                 float(epoch)))
        target = self.boundary.target_to_native(target_normalized)
        probabilities = self.plant.probabilities(
            controls, int(math.floor(float(epoch))), frequency, target_controls=target)
        dem = self.plant._circuit_from_probabilities(probabilities).detector_error_model(
            decompose_errors=False, approximate_disjoint_errors=True, flatten_loops=True)
        parity_products = np.ones(self.plant.detector_count, dtype=float)
        for instruction in dem.flattened():
            if instruction.type != "error":
                continue
            probability = float(instruction.args_copy()[0])
            parity: dict[int, int] = defaultdict(int)
            for target_item in instruction.targets_copy():
                if target_item.is_relative_detector_id():
                    parity[int(target_item.val)] ^= 1
            for detector, odd in parity.items():
                if odd:
                    parity_products[detector] *= 1.0 - 2.0 * probability
        result = (1.0 - parity_products) / 2.0
        self._cache[key] = result.copy()
        return result

    def cost(self, latent: np.ndarray, epoch: float, frequency: float) -> float:
        return float(np.sum(self.detector_expectations(latent, epoch, frequency)))

    def normalized_cost(self, normalized: np.ndarray, epoch: float,
                        frequency: float) -> float:
        return self.cost(self.normalized_to_latent(normalized), epoch, frequency)


_EVALUATOR: ExactDetectorEvaluator | None = None


def evaluator() -> ExactDetectorEvaluator:
    global _EVALUATOR
    if _EVALUATOR is None:
        _EVALUATOR = ExactDetectorEvaluator()
    return _EVALUATOR


def replay_gradients(run: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Recreate the exact logged behavior-policy gradients and learned baseline state."""
    controller = _controller_spec()
    plant = evaluator().plant
    checkpoint = run["checkpoint"]
    baseline = np.zeros(plant.detector_count, dtype=float)
    rows: list[dict[str, Any]] = []
    for epoch, record in enumerate(run["records"]):
        mean = np.asarray(record["latent_behavior_mean"], dtype=float)
        sigma = np.asarray(record["behavior_sigma"], dtype=float)
        actions = mean[None, :] + sigma[None, :] * run["noises"][epoch]
        behavior = BehaviorSnapshot(
            mean, sigma, component_log_probability(actions, mean, sigma), epoch)
        shots = int(record["qec_cycles_per_candidate"]) // int(
            checkpoint["protocol"]["circuit_rounds"])
        rewards = -np.asarray(record["stochastic_detector_counts"], dtype=float) / shots
        baseline_before = baseline.copy()
        loss = total_loss_and_gradients(
            actions, rewards, plant.mask, mean, sigma, baseline_before, behavior,
            clip=controller.ppo_clip,
            entropy_weight=controller.effective_entropy_coefficient,
            baseline_weight=controller.baseline_loss_weight)
        advantages = rewards - baseline_before[None, :]
        score_mean = (actions - mean[None, :]) / sigma[None, :]**2
        local_advantage = advantages @ plant.mask.astype(float)
        candidate_update_contributions = local_advantage * score_mean
        update_direction = np.mean(candidate_update_contributions, axis=0)
        if not np.allclose(update_direction, -loss.grad_mean, rtol=1e-10, atol=1e-12):
            raise RuntimeError("candidate mean-gradient replay disagrees with production loss")
        baseline -= controller.baseline_learning_rate * loss.grad_baseline
        rows.append({
            "epoch": epoch,
            "mean": mean,
            "sigma": sigma,
            "actions": actions,
            "rewards": rewards,
            "baseline": baseline_before,
            "loss": loss,
            "candidate_update_contributions": candidate_update_contributions,
            "update_direction": update_direction,
            "delta_mean": np.asarray(record["post_update_latent_mean"], dtype=float) - mean,
            "reward_variance": float(np.mean(np.var(rewards, axis=0, ddof=1))),
            "baseline_error": float(np.linalg.norm(np.mean(rewards, axis=0) - baseline_before)),
        })
    return rows


def exact_update_direction(actions: np.ndarray, mean: np.ndarray, sigma: np.ndarray,
                           baseline: np.ndarray, epoch: float, frequency: float) -> np.ndarray:
    evaluation = evaluator()
    rewards = -np.asarray([
        evaluation.detector_expectations(action, epoch, frequency) for action in actions])
    behavior = BehaviorSnapshot(
        mean, sigma, component_log_probability(actions, mean, sigma), int(epoch))
    controller = _controller_spec()
    loss = total_loss_and_gradients(
        actions, rewards, evaluation.plant.mask, mean, sigma, baseline, behavior,
        clip=controller.ppo_clip,
        entropy_weight=controller.effective_entropy_coefficient,
        baseline_weight=controller.baseline_loss_weight)
    return -loss.grad_mean
