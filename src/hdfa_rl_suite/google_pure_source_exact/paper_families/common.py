"""Shared sparse plant and resumable direct-sigma runtime for paper analogues.

The proprietary simulator is unavailable.  These backends therefore implement the
public controller mathematics and paper geometry without claiming numerical identity
to the unpublished plant.  Simulator targets are evaluation-only and are never passed
to the controller update.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.contracts import (
    JOINT_LEARNED_DETECTOR_BASELINE,
    PositivityGuard,
    SOURCE_ELEMENTWISE_COORDINATE_CLIPPING,
)
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.gaussian import (
    BehaviorSnapshot,
    DirectSigmaGaussianPolicy,
    component_log_probability,
    entropy,
    gaussian_scores,
)
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.optimizer import (
    DirectSigmaOptimizer,
    OptimizerConfig,
)
from hdfa_rl_suite.google_pure_source_exact.source_normalization import (
    SourceNormalizationBoundary,
    require_v15_boundary_provenance,
)


ROOT = Path(__file__).resolve().parents[4]


def canonical_hash(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def controller_config() -> dict[str, Any]:
    value = json.loads((ROOT / "configs/google_pure_source_exact/figure5a.json").read_text(
        encoding="utf-8"))["controller"]
    return dict(value)


def optimizer_config() -> OptimizerConfig:
    value = controller_config()
    return OptimizerConfig(
        float(value["mean_learning_rate"]), float(value["sigma_learning_rate"]),
        float(value["baseline_learning_rate"]),
        minimum_sigma=float(value["minimum_sigma"]),
        maximum_sigma=float(value["maximum_sigma"]),
        positivity_guard=PositivityGuard(value["positivity_guard"]),
    )


_FAMILY_CONTRACTS: dict[str, dict[str, Any]] = {
    "FIGURE5A_REAL_TIME_STEERING": {
        "plant": "DISTANCE3_41_PARAMETER_STIM_QUADRATIC",
        "graph": "STIM_DERIVED_DETECTOR_CONTROL_GRAPH",
    },
    "FIGURE5B_SPARSE_SCALING": {
        "plant": "SURFACE_CODE_SPARSE_QUADRATIC_DIRECT_SIGMA_ANALOGUE",
        "graph": "DISTANCE_LOCAL_SPARSE_CONTROL_GRAPH",
    },
    "FIGURE5C_CONVERGENCE_LAW": {
        "plant": "SURFACE_CODE_SPARSE_QUADRATIC_DIRECT_SIGMA_ANALOGUE",
        "graph": "DISTANCE_LOCAL_SPARSE_CONTROL_GRAPH",
    },
    "NATURAL_DRIFT_SPECTRAL_SUPPRESSION": {
        "plant": "SIX_SOURCE_STRUCTURED_SPARSE_DRIFT_PLANTS",
        "graph": "LOCAL_SPARSE_CONTROL_GRAPH_WITH_DECODED_LER_EVALUATION",
    },
    "RANDOMIZED_RECOVERY_AFTER_SPOIL": {
        "plant": "DISTANCE5_924_COORDINATE_POLICY_SPOIL_ANALOGUE",
        "graph": "DECLARED_DISTANCE5_924_COORDINATE_SPARSE_GRAPH",
    },
    "STEP_RESPONSE_INJECTED_DRIFT": {
        "plant": "DISTANCE5_924_COORDINATE_XY_STEP_ANALOGUE",
        "graph": "DECLARED_DISTANCE5_924_COORDINATE_SPARSE_GRAPH",
    },
}


def amended_family_identities(family: str) -> tuple[str, str]:
    try:
        value = _FAMILY_CONTRACTS[str(family)]
    except KeyError as error:
        raise ValueError(f"no amended family contract for {family}") from error
    return canonical_hash({"schema": "amended-paper-plant.v1", **value}), canonical_hash(
        {"schema": "amended-paper-graph.v1", "graph": value["graph"]})


@dataclass(frozen=True)
class SparseControlPlant:
    """Local quadratic detector model with O(P) storage and explicit physical floor."""

    distance: int
    controls: int
    detectors: int
    seed: int
    irreducible_physical_error: float = 4e-4
    threshold_physical_error: float = 1.79e-3
    curvature: float = 3e-3

    def __post_init__(self) -> None:
        if self.distance < 3 or self.distance % 2 == 0:
            raise ValueError("surface-code distance must be odd and at least three")
        if min(self.controls, self.detectors) <= 0:
            raise ValueError("plant dimensions must be positive")
        if not 0 < self.irreducible_physical_error < self.threshold_physical_error:
            raise ValueError("physical floor must be positive and below threshold")

    @property
    def control_detector(self) -> np.ndarray:
        # A deterministic permutation prevents adjacent serialized controls from all
        # landing in the same detector while retaining one local owner per coordinate.
        step = max(1, self.detectors // 2 + 1)
        while np.gcd(step, self.detectors) != 1:
            step += 1
        return (np.arange(self.controls, dtype=np.int64) * step + self.seed) % self.detectors

    @property
    def plant_hash(self) -> str:
        return canonical_hash({"schema": "sparse-source-plant.v1", "distance": self.distance,
                               "controls": self.controls, "detectors": self.detectors,
                               "seed": self.seed, "floor": self.irreducible_physical_error,
                               "threshold": self.threshold_physical_error,
                               "curvature": self.curvature})

    @property
    def graph_hash(self) -> str:
        return canonical_hash({"schema": "one-owner-local-sparse-graph.v1",
                               "control_detector": self.control_detector.tolist()})

    @property
    def control_ids(self) -> tuple[str, ...]:
        return tuple(f"sparse-d{self.distance}:control:{index}" for index in range(self.controls))

    @property
    def connected_objective_curvature(self) -> np.ndarray:
        """Native coefficient in the connected-detector reward sum per coordinate."""
        degree = np.bincount(self.control_detector, minlength=self.detectors).astype(float)
        return np.full(self.controls, self.curvature) / degree[self.control_detector]

    def expected_detector_rates(self, actions: np.ndarray, target: np.ndarray) -> np.ndarray:
        samples = np.atleast_2d(np.asarray(actions, dtype=float))
        optimum = np.asarray(target, dtype=float)
        if samples.shape[1:] != (self.controls,) or optimum.shape != (self.controls,):
            raise ValueError("action or target shape mismatch")
        squared = np.square(samples - optimum[None, :])
        owners = self.control_detector
        counts = np.bincount(owners, minlength=self.detectors).astype(float)
        rates = np.empty((len(samples), self.detectors), dtype=float)
        for row, values in enumerate(squared):
            local = np.bincount(owners, weights=values, minlength=self.detectors) / counts
            rates[row] = self.irreducible_physical_error + self.curvature * local
        return np.clip(rates, 1e-9, 0.49)

    def sample_counts(self, actions: np.ndarray, target: np.ndarray, *, cycles: int,
                      seed: int) -> np.ndarray:
        if cycles <= 0:
            raise ValueError("cycles must be positive")
        probabilities = self.expected_detector_rates(actions, target)
        return np.random.default_rng(int(seed)).binomial(int(cycles), probabilities)

    def performance(self, action: np.ndarray, target: np.ndarray) -> dict[str, float]:
        physical = float(np.mean(self.expected_detector_rates(action, target)))
        suppression = self.threshold_physical_error / physical
        exponent = (self.distance + 1) / 2
        logical = float(np.clip(0.01 * suppression ** (-exponent), 1e-12, 1.0))
        floor_suppression = self.threshold_physical_error / self.irreducible_physical_error
        logical_floor = float(0.01 * floor_suppression ** (-exponent))
        return {"physical_error": physical, "lambda": suppression,
                "lambda_star": floor_suppression, "logical_error": logical,
                "logical_floor": logical_floor}


def _sparse_source_loss(actions: np.ndarray, rewards: np.ndarray, owners: np.ndarray,
                        mean: np.ndarray, sigma: np.ndarray, baseline: np.ndarray,
                        behavior: BehaviorSnapshot, *, clip: float,
                        entropy_weight: float, baseline_weight: float) -> dict[str, Any]:
    """Source elementwise clipping evaluated without a dense detector-control matrix."""
    current = component_log_probability(actions, mean, sigma)
    coordinate_log_ratio = current - behavior.component_log_probability
    active = (coordinate_log_ratio > np.log1p(-clip)) & (coordinate_log_ratio < np.log1p(clip))
    clipped = np.clip(coordinate_log_ratio, np.log1p(-clip), np.log1p(clip))
    detector_count = len(baseline)
    masked = np.vstack([
        np.bincount(owners, weights=row, minlength=detector_count) for row in clipped
    ])
    if np.any(masked > np.log(np.finfo(float).max)) or np.any(masked < np.log(np.finfo(float).tiny)):
        raise FloatingPointError("masked coordinate-ratio product is outside float64 range")
    local_ratio = np.exp(masked)
    advantages = np.asarray(rewards, dtype=float) - baseline[None, :]
    detector_weight = advantages * local_ratio
    control_weight = detector_weight[:, owners] * active / len(actions)
    score_mean, score_sigma = gaussian_scores(actions, mean, sigma)
    grad_mean = -np.sum(control_weight * score_mean, axis=0)
    reward_grad_sigma = -np.sum(control_weight * score_sigma, axis=0)
    grad_sigma = reward_grad_sigma - float(entropy_weight) / sigma
    grad_baseline = 2.0 * float(baseline_weight) * np.mean(
        baseline[None, :] - rewards, axis=0)
    return {
        "grad_mean": grad_mean, "grad_sigma": grad_sigma,
        "grad_baseline": grad_baseline,
        "diagnostics": {
            "ratio_clipping_mode": SOURCE_ELEMENTWISE_COORDINATE_CLIPPING,
            "baseline_mode": JOINT_LEARNED_DETECTOR_BASELINE,
            "coordinate_ratios_clipped_before_sparse_product": True,
            "component_clip_fraction": float(1.0 - active.mean()),
            "reward_sigma_gradient_norm": float(np.linalg.norm(reward_grad_sigma)),
            "entropy_sigma_gradient_norm": float(np.linalg.norm(float(entropy_weight) / sigma)),
            "policy_entropy": entropy(sigma),
            "dense_parameter_matrix_allocated": False,
        },
    }


def _new_state(*, plant: SparseControlPlant, protocol_hash: str, seed: int,
               initial_mean: np.ndarray, initial_sigma: float,
               boundary: SourceNormalizationBoundary) -> tuple[dict[str, Any],
                                                                        DirectSigmaGaussianPolicy,
                                                                        DirectSigmaOptimizer,
                                                                        np.ndarray]:
    policy = DirectSigmaGaussianPolicy(initial_mean, np.full(plant.controls, initial_sigma), seed=seed)
    optimizer = DirectSigmaOptimizer(plant.controls, plant.detectors, optimizer_config())
    baseline = np.zeros(plant.detectors)
    state = {"schema": "amended-paper-family-checkpoint.v1", "protocol_hash": protocol_hash,
             "plant_hash": plant.plant_hash, "graph_hash": plant.graph_hash, "seed": int(seed),
             "v15_boundary": boundary.provenance_fields(),
             "epoch": 0, "candidate_boundaries": 0, "records": [],
             "policy": policy.state_dict(optimizer_state=optimizer.state_dict(), baseline=baseline)}
    return state, policy, optimizer, baseline


def _load_state(state: Mapping[str, Any]) -> tuple[DirectSigmaGaussianPolicy,
                                                    DirectSigmaOptimizer, np.ndarray]:
    policy = DirectSigmaGaussianPolicy.from_state_dict(dict(state["policy"]))
    optimizer = DirectSigmaOptimizer.from_state_dict(state["policy"]["optimizer_state"])
    baseline = np.asarray(state["policy"]["baseline"], dtype=float)
    return policy, optimizer, baseline


def run_direct_sigma_trace(*, plant: SparseControlPlant, protocol_hash: str, seed: int,
                           epochs: int, candidates: int, cycles_per_candidate: int,
                           entropy_weight: float, checkpoint: Path,
                           target_at_epoch: Callable[[int], np.ndarray],
                           experiment_family: str = "DEVELOPMENT_FIXTURE",
                           initial_mean: np.ndarray | None = None,
                           evaluation: Callable[[int, DirectSigmaGaussianPolicy, np.ndarray],
                                                Mapping[str, Any]] | None = None,
                           max_epochs: int | None = None,
                           fresh_acquisition_required: bool = False,
                           source_budget_profile: str = "UNSPECIFIED_DEVELOPMENT") -> dict[str, Any]:
    """Run or resume a condition with immutable epoch checkpoints.

    Each epoch is one controller candidate batch, the smallest safe restart boundary
    for the high-dimensional scaling runs.  The 924-coordinate step runner retains its
    stricter candidate-boundary checkpoint in its dedicated source-exact module.
    """
    if min(epochs, candidates, cycles_per_candidate) <= 0:
        raise ValueError("acquisition budgets must be positive")
    first_mean = np.zeros(plant.controls) if initial_mean is None else np.asarray(initial_mean, dtype=float)
    config = controller_config()
    boundary = SourceNormalizationBoundary.from_training_objective(
        experiment_family, plant.connected_objective_curvature,
        control_ids=plant.control_ids)
    checkpoint_preexisted = checkpoint.exists()
    if checkpoint_preexisted and fresh_acquisition_required:
        raise RuntimeError("fresh V15 acquisition forbids reuse of a lower-level checkpoint")
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        expected = (protocol_hash, plant.plant_hash, plant.graph_hash, int(seed),
                    boundary.provenance_fields())
        observed = (state["protocol_hash"], state["plant_hash"], state["graph_hash"], state["seed"],
                    state.get("v15_boundary"))
        if observed != expected:
            raise RuntimeError("checkpoint identity changed")
        policy, optimizer, baseline = _load_state(state)
    else:
        state, policy, optimizer, baseline = _new_state(
            plant=plant, protocol_hash=protocol_hash, seed=seed, initial_mean=first_mean,
            initial_sigma=float(config["initial_sigma"]), boundary=boundary)
        atomic_json(checkpoint, state)
    stopping_epoch = epochs if max_epochs is None else min(epochs, int(state["epoch"]) + max_epochs)
    owners = plant.control_detector
    while int(state["epoch"]) < stopping_epoch:
        epoch = int(state["epoch"])
        target_normalized = np.asarray(target_at_epoch(epoch), dtype=float)
        target = boundary.target_to_native(target_normalized)
        batch = policy.sample(candidates)
        applied = boundary.apply(
            batch.actions, control_order_hash=boundary.control_order_hash,
            sensitivity_map_hash=boundary.sensitivity_map_hash)
        detector_counts = plant.sample_counts(
            applied.native, target, cycles=cycles_per_candidate,
            seed=int(canonical_hash([protocol_hash, seed, epoch, "candidate-counts"])[:16], 16))
        rewards = -detector_counts / float(cycles_per_candidate)
        loss = _sparse_source_loss(
            batch.actions, rewards, owners, policy.mean, policy.sigma, baseline,
            batch.behavior, clip=float(config["ppo_clip"]),
            entropy_weight=float(entropy_weight), baseline_weight=float(config["baseline_weight"]))
        before_mean = policy.mean.copy()
        update = optimizer.step(policy.mean, policy.sigma, baseline,
                                loss["grad_mean"], loss["grad_sigma"], loss["grad_baseline"],
                                mean_bounds=(-2.0, 2.0))
        policy.policy_version += 1
        learned_native = boundary.apply(policy.mean).native
        fixed_native = boundary.apply(first_mean).native
        learned = plant.performance(learned_native, target)
        fixed = plant.performance(fixed_native, target)
        oracle = plant.performance(target, target)
        candidate_physical = float(np.mean(detector_counts) / cycles_per_candidate)
        record: dict[str, Any] = {
            "epoch": epoch, "parameterization": "direct_sigma",
            "controller_mode": "PAPER_DIRECT_SIGMA",
            "ratio_clipping_mode": loss["diagnostics"]["ratio_clipping_mode"],
            "baseline_mode": loss["diagnostics"]["baseline_mode"],
            "coordinate_ratios_clipped_before_sparse_product": True,
            "component_clip_fraction": loss["diagnostics"]["component_clip_fraction"],
            "reward_sigma_gradient_norm": loss["diagnostics"]["reward_sigma_gradient_norm"],
            "entropy_sigma_gradient_norm": loss["diagnostics"]["entropy_sigma_gradient_norm"],
            "policy_entropy": loss["diagnostics"]["policy_entropy"],
            "fraction_at_sigma_guard": update["fraction_at_positivity_guard"],
            "mean_motion": float(np.linalg.norm(policy.mean - before_mean)),
            "mean_sigma": float(np.mean(policy.sigma)),
            "candidate_physical_error": candidate_physical,
            "learned": learned, "fixed": fixed, "oracle": oracle,
            **boundary.provenance_fields(),
        }
        if evaluation is not None:
            record["evaluation"] = dict(evaluation(epoch, policy, target_normalized))
        state["records"].append(record)
        state["epoch"] = epoch + 1
        state["candidate_boundaries"] += candidates
        state["policy"] = policy.state_dict(
            optimizer_state=optimizer.state_dict(), baseline=baseline)
        atomic_json(checkpoint, state)
    result = {
        "complete": int(state["epoch"]) == epochs, "epoch": int(state["epoch"]),
        "records": state["records"], "candidate_boundaries": state["candidate_boundaries"],
        "candidate_qec_cycles": int(state["candidate_boundaries"]) * cycles_per_candidate,
        "checkpoint_path": str(checkpoint.resolve()), "plant_hash": plant.plant_hash,
        "graph_hash": plant.graph_hash, "control_count": plant.controls,
        "detector_count": plant.detectors, "controller_observed_target": False,
        "dense_parameter_matrix_allocated": False,
        "fresh_acquisition": not checkpoint_preexisted,
        "reused_shard_ids": [],
        "source_budget_profile": str(source_budget_profile),
        "boundary_trace": boundary.trace(
            np.asarray(state["records"][0].get("boundary_probe", np.eye(1, plant.controls, 0).ravel()),
                       dtype=float)) if state["records"] else None,
        **boundary.provenance_fields(),
    }
    require_v15_boundary_provenance(result)
    return result
