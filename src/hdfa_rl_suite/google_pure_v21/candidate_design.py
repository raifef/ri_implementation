"""Fixed-budget candidate frames and their mathematically matched estimators."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


DESIGN_IDS = ("D0", "D1", "D2", "D3", "D4", "D5")
DESIGN_NAMES = {
    "D0": "IID_GAUSSIAN_K8",
    "D1": "FOUR_ANTITHETIC_GAUSSIAN_PAIRS",
    "D2": "EIGHT_ROTATING_ORTHOGONAL_SPHERE_DIRECTIONS",
    "D3": "FOUR_ROTATING_ORTHOGONAL_ANTITHETIC_SPHERE_PAIRS",
    "D4": "PUBLIC_FACTOR_GRAPH_RANDOM_LOCAL_BLOCKS",
    "D5": "BALANCED_PUBLIC_LOCAL_BLOCK_PAIRS",
}
SOURCE_FIDELITY = {
    "D0": "SOURCE_EXPLICIT",
    "D1": "DIAGNOSTIC_EXTENSION",
    "D2": "DIAGNOSTIC_EXTENSION",
    "D3": "DIAGNOSTIC_EXTENSION",
    "D4": "SOURCE_IMPLIED",
    "D5": "SOURCE_IMPLIED",
}

ORACLE_PROVENANCE_FIELDS = (
    "uses_known_driven_direction", "uses_target_trajectory", "uses_future_phase",
    "uses_population_or_reference_gradient", "uses_hidden_optimum",
    "uses_multi_run_leakage", "uses_posthoc_selected_subspace",
)


def candidate_is_nonoracle(provenance: Mapping[str, Any]) -> bool:
    """Fail closed unless every unavailable-information dependency is absent."""
    return all(provenance.get(field) is False for field in ORACLE_PROVENANCE_FIELDS)


@dataclass(frozen=True)
class CandidateFrame:
    design_id: str
    standardized_directions: np.ndarray
    mean_score_factors: np.ndarray
    sigma_score_factors: np.ndarray | None
    inclusion_probabilities: np.ndarray
    selected_blocks: tuple[int, ...]
    estimator_valid: bool
    sigma_estimator_valid: bool
    physical_covariance_preserved_in_expectation: bool
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        z = np.asarray(self.standardized_directions, dtype=float)
        score = np.asarray(self.mean_score_factors, dtype=float)
        inclusion = np.asarray(self.inclusion_probabilities, dtype=float)
        if z.ndim != 2 or z.shape != score.shape or inclusion.shape != z.shape:
            raise ValueError("candidate frame arrays must have identical K-by-P shape")
        if z.shape[0] != 8 or np.any(inclusion <= 0) or np.any(inclusion > 1):
            raise ValueError("V21 frames require K=8 and valid inclusion probabilities")
        if self.sigma_score_factors is not None and \
                np.asarray(self.sigma_score_factors).shape != z.shape:
            raise ValueError("sigma score factors are misaligned")


def public_factor_graph_blocks(mask: np.ndarray, inventory: Sequence[Any],
                               *, block_count: int = 4) -> tuple[np.ndarray, ...]:
    """Derive deterministic local blocks from public detector topology only."""
    graph = np.asarray(mask, dtype=bool)
    if graph.ndim != 2 or graph.shape[1] != len(inventory) or block_count <= 1:
        raise ValueError("invalid public factor graph")
    keys = []
    for coordinate, item in enumerate(inventory):
        detectors = np.flatnonzero(graph[:, coordinate])
        detector_center = float(np.mean(detectors)) if detectors.size else float("inf")
        qubit_center = float(np.mean(item.qubits))
        keys.append((detector_center, qubit_center, coordinate))
    ordered = np.asarray([row[2] for row in sorted(keys)], dtype=int)
    blocks = tuple(np.asarray(value, dtype=int) for value in np.array_split(ordered, block_count))
    if sorted(np.concatenate(blocks).tolist()) != list(range(graph.shape[1])):
        raise RuntimeError("public blocks do not partition all controls")
    return blocks


def _orthogonal_rows(rng: np.random.Generator, rows: int, dimension: int) -> np.ndarray:
    matrix = rng.normal(size=(dimension, rows))
    q, _ = np.linalg.qr(matrix, mode="reduced")
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=rows)
    return (q * signs[None, :]).T


def generate_frame(design_id: str, *, dimension: int, epoch: int, seed: int,
                   blocks: Sequence[np.ndarray]) -> CandidateFrame:
    """Generate one K=8 frame in standardized policy coordinates."""
    if design_id not in DESIGN_IDS or dimension <= 0 or len(blocks) != 4:
        raise ValueError("unknown design or invalid dimension/block inventory")
    rng = np.random.default_rng(int(seed) + 104729 * int(epoch) + 1009 * DESIGN_IDS.index(design_id))
    inclusion = np.ones((8, dimension), dtype=float)
    selected: tuple[int, ...] = ()
    sigma_score: np.ndarray | None
    if design_id == "D0":
        z = rng.normal(size=(8, dimension))
        mean_score = z.copy()
        sigma_score = z**2 - 1.0
        metadata = {
            "sampling_distribution": "eight independent standard normal vectors",
            "normalization": "Gaussian score z/sigma",
            "bias_properties": "unbiased Monte Carlo estimate of the factorized Gaussian objective",
        }
    elif design_id == "D1":
        base = rng.normal(size=(4, dimension))
        z = np.concatenate([base, -base], axis=0)
        mean_score = z.copy()
        sigma_score = z**2 - 1.0
        metadata = {
            "sampling_distribution": "four iid Gaussian directions and their exact negatives",
            "normalization": "marginal Gaussian score z/sigma; pair dependence retained",
            "bias_properties": "unbiased because every candidate marginal remains N(0,I)",
            "independent_directions": 4,
        }
    elif design_id == "D2":
        q = _orthogonal_rows(rng, 8, dimension)
        z = np.sqrt(dimension) * q
        mean_score = z.copy()
        sigma_score = None
        metadata = {
            "sampling_distribution": "rotating Haar orthogonal rows on radius-sqrt(P) sphere",
            "normalization": "sphere/ellipsoid boundary estimator z/sigma",
            "bias_properties": "unbiased for ellipsoid-ball smoothing, not Gaussian smoothing",
            "independent_directions": 8,
        }
    elif design_id == "D3":
        q = _orthogonal_rows(rng, 4, dimension)
        base = np.sqrt(dimension) * q
        z = np.concatenate([base, -base], axis=0)
        mean_score = z.copy()
        sigma_score = None
        metadata = {
            "sampling_distribution": "four Haar-orthogonal sphere directions and negatives",
            "normalization": "antithetic sphere/ellipsoid estimator z/sigma",
            "bias_properties": "unbiased for ellipsoid-ball smoothing, not Gaussian smoothing",
            "independent_directions": 4,
        }
    else:
        p = 1.0 / len(blocks)
        z = np.zeros((8, dimension), dtype=float)
        inclusion = np.full((8, dimension), p, dtype=float)
        sigma_score = np.zeros_like(z)
        if design_id == "D4":
            chosen = tuple(int(value) for value in rng.integers(0, len(blocks), size=8))
            raw_vectors = [rng.normal(size=len(blocks[block])) for block in chosen]
            metadata = {
                "sampling_distribution": "uniform random public factor-graph block then conditional Gaussian",
                "normalization": "1/p Horvitz-Thompson correction with conditional scale sigma/sqrt(p)",
                "bias_properties": "unbiased for the public block-mixture objective; cross-block coupling is reported",
                "inclusion_probability": p,
            }
        else:
            rotation = int(epoch) % len(blocks)
            order = tuple((rotation + offset) % len(blocks) for offset in range(len(blocks)))
            chosen_list: list[int] = []
            raw_vectors = []
            for block in order:
                raw = rng.normal(size=len(blocks[block]))
                chosen_list.extend([block, block])
                raw_vectors.extend([raw, -raw])
            chosen = tuple(chosen_list)
            metadata = {
                "sampling_distribution": "one antithetic conditional-Gaussian pair per public block",
                "normalization": "balanced 1/p Horvitz-Thompson correction; block order rotates",
                "bias_properties": "unbiased for the balanced public block-mixture objective",
                "inclusion_probability": p,
                "rolling_block_frame_epochs": len(blocks),
            }
        for candidate, (block, raw) in enumerate(zip(chosen, raw_vectors)):
            support = np.asarray(blocks[block], dtype=int)
            z[candidate, support] = np.asarray(raw) / np.sqrt(p)
            sigma_score[candidate, support] = (p * z[candidate, support]**2 - 1.0) / p
        mean_score = z.copy()
        selected = chosen
    return CandidateFrame(
        design_id=design_id,
        standardized_directions=z,
        mean_score_factors=mean_score,
        sigma_score_factors=sigma_score,
        inclusion_probabilities=inclusion,
        selected_blocks=selected,
        estimator_valid=True,
        sigma_estimator_valid=sigma_score is not None,
        physical_covariance_preserved_in_expectation=True,
        metadata={
            **metadata,
            "design_name": DESIGN_NAMES[design_id],
            "source_fidelity": SOURCE_FIDELITY[design_id],
            "coordinate_space": "standardized latent policy coordinates",
            "K": 8,
            "rank": int(np.linalg.matrix_rank(z)),
            "expected_coverage": "rotating full coordinates" if design_id in {"D2", "D3", "D5"}
                else "full in expectation",
            "covariance_preservation": "E[(1/K) sum z_k z_k^T] has unit diagonal",
        },
    )


def estimate_policy_updates(frame: CandidateFrame, rewards: np.ndarray, baseline: np.ndarray,
                            mask: np.ndarray, sigma: np.ndarray) -> dict[str, np.ndarray | None]:
    """Apply the estimator derived for the frame; never substitute the iid score."""
    reward = np.asarray(rewards, dtype=float)
    base = np.asarray(baseline, dtype=float)
    graph = np.asarray(mask, dtype=bool)
    sd = np.asarray(sigma, dtype=float)
    if reward.shape != (8, graph.shape[0]) or base.shape != (graph.shape[0],) or \
            sd.shape != (graph.shape[1],):
        raise ValueError("candidate rewards, baseline, graph, and sigma are not aligned")
    advantages = reward - base[None, :]
    local_advantage = advantages @ graph.astype(float)
    mean_contributions = local_advantage * frame.mean_score_factors / sd[None, :]
    mean_update = np.mean(mean_contributions, axis=0)
    if frame.sigma_score_factors is None:
        sigma_reward_gradient = None
    else:
        sigma_update_contributions = local_advantage * frame.sigma_score_factors / sd[None, :]
        sigma_reward_gradient = -np.mean(sigma_update_contributions, axis=0)
    return {
        "mean_update_direction": mean_update,
        "candidate_mean_update_contributions": mean_contributions,
        "sigma_reward_gradient": sigma_reward_gradient,
        "baseline_gradient": 2.0 * np.mean(base[None, :] - reward, axis=0),
    }
