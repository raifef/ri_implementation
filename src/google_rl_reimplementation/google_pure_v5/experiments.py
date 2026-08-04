"""Shared acquisition runner; disturbance generation remains experiment-specific."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .plant import PureQuadraticPlant
from .reference_agent import PureGoogleReferenceAgent, evidence_from_counts


POLICY_CLASSES = ("fixed_policy", "learned_mean", "stochastic_candidates", "oracle_optimum")


def validate_policy_schema(traces: Mapping[str, Any]) -> None:
    if set(traces) != set(POLICY_CLASSES):
        raise ValueError("fixed, learned-mean, stochastic, and oracle traces must be separate")
    identities = [id(traces[name]) for name in POLICY_CLASSES]
    if len(set(identities)) != len(identities):
        raise ValueError("policy-class arrays may not alias")


def run_matched_trace(
    plant: PureQuadraticPlant,
    optimum_trace: np.ndarray,
    choices: Mapping[str, Any],
    paper: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    optimum_trace = np.asarray(optimum_trace, dtype=float)
    if optimum_trace.ndim != 2 or optimum_trace.shape[1] != plant.spec.control_count:
        raise ValueError("optimum tape shape mismatch")
    rng = np.random.default_rng(seed + 100_000)
    agent = PureGoogleReferenceAgent(
        plant.mask,
        plant.base_optimum,
        plant.native_sensitivity,
        choices,
        seed=seed,
    )
    fixed = plant.base_optimum.copy()
    candidates = int(paper["candidates_per_epoch"])
    cycles = int(paper["effective_cycles_per_candidate"])
    traces = {name: [] for name in POLICY_CLASSES}
    detector = {name: [] for name in POLICY_CLASSES}
    mean_vectors: list[list[float]] = []
    scale_vectors: list[list[float]] = []
    baseline_vectors: list[list[float]] = []
    advantage_means: list[list[float]] = []
    diagnostics: list[dict[str, float]] = []
    for optimum in optimum_trace:
        batch = agent.sample(candidates)
        counts = plant.acquire_counts(
            batch.normalized_actions, optimum, effective_cycles=cycles, rng=rng
        )
        rewards = -counts / float(cycles)
        frozen = agent.baseline.snapshot()
        diagnostic = agent.update(batch, evidence_from_counts(batch, counts, cycles))
        fixed_ler = float(plant.logical_risk(fixed[None, :], optimum)[0])
        mean_ler = float(plant.logical_risk(agent.mean[None, :], optimum)[0])
        stochastic_ler = float(plant.logical_risk(batch.normalized_actions, optimum).mean())
        oracle_ler = float(plant.logical_risk(optimum[None, :], optimum)[0])
        fixed_edr = float(plant.detector_rates(fixed[None, :], optimum).mean())
        mean_edr = float(plant.detector_rates(agent.mean[None, :], optimum).mean())
        stochastic_edr = float(plant.detector_rates(batch.normalized_actions, optimum).mean())
        oracle_edr = float(plant.detector_rates(optimum[None, :], optimum).mean())
        for name, value in zip(POLICY_CLASSES, (fixed_ler, mean_ler, stochastic_ler, oracle_ler)):
            traces[name].append(value)
        for name, value in zip(POLICY_CLASSES, (fixed_edr, mean_edr, stochastic_edr, oracle_edr)):
            detector[name].append(value)
        mean_vectors.append(agent.mean.tolist())
        scale_vectors.append(agent.scale.tolist())
        baseline_vectors.append(frozen.tolist())
        advantage_means.append((rewards - frozen[None, :]).mean(axis=0).tolist())
        diagnostics.append(diagnostic)
    trace_arrays = {name: np.asarray(values, dtype=float) for name, values in traces.items()}
    detector_arrays = {name: np.asarray(values, dtype=float) for name, values in detector.items()}
    validate_policy_schema(trace_arrays)
    validate_policy_schema(detector_arrays)
    return {
        "logical_risk": trace_arrays,
        "detector_rate": detector_arrays,
        "learned_mean_vectors": np.asarray(mean_vectors),
        "policy_scale_vectors": np.asarray(scale_vectors),
        "baseline_vectors": np.asarray(baseline_vectors),
        "advantage_means": np.asarray(advantage_means),
        "diagnostics": diagnostics,
    }


def lag1_autocorrelation(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float).reshape(len(values), -1).mean(axis=1)
    if len(x) < 3 or np.std(x[:-1]) == 0 or np.std(x[1:]) == 0:
        return 0.0
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def percentile_interval(values: list[float] | np.ndarray) -> list[float]:
    x = np.asarray(values, dtype=float)
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]
