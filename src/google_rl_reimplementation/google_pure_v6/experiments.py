"""Matched-tape acquisitions with all policy classes kept physically separate."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .plant import PureQuadraticPlant
from .reference_agent import PureGoogleV6Agent, evidence_from_counts


POLICY_CLASSES = ("fixed_policy", "learned_mean", "stochastic_candidates", "oracle_optimum")


def validate_policy_schema(traces: Mapping[str, Any]) -> None:
    if set(traces) != set(POLICY_CLASSES):
        raise ValueError("fixed, learned-mean, stochastic, and oracle traces must be separate")
    if len({id(traces[key]) for key in POLICY_CLASSES}) != len(POLICY_CLASSES):
        raise ValueError("policy-class traces may not alias")


def run_matched_trace(plant: PureQuadraticPlant, optimum_normalized: np.ndarray, choices: Mapping[str, Any], *,
                      seed: int, candidates: int = 12, cycles: int = 5000,
                      objective_mode: str | None = None) -> dict[str, Any]:
    optimum_normalized = np.asarray(optimum_normalized, dtype=float)
    if optimum_normalized.ndim != 2 or optimum_normalized.shape[1] != plant.spec.control_count:
        raise ValueError("optimum tape shape mismatch")
    rng = np.random.default_rng(seed + 100_000)
    mode = objective_mode or str(choices.get("objective_mode", "source_literal_ppo"))
    agent = PureGoogleV6Agent(plant.mask, plant.spec.base_optimum_normalized,
                             plant.spec.coordinates, choices, seed=seed, objective_mode=mode)
    fixed_native = plant.base_optimum_native.copy()
    traces = {name: [] for name in POLICY_CLASSES}
    detector = {name: [] for name in POLICY_CLASSES}
    mean_vectors, scale_vectors, diagnostics = [], [], []
    for optimum in optimum_normalized:
        optimum_native = plant.spec.coordinates.to_native(optimum)
        batch = agent.sample(candidates)
        counts = plant.acquire_counts(batch.applied_native_actions, optimum_native, cycles=cycles, rng=rng)
        diagnostics.append(agent.update(batch, evidence_from_counts(batch, counts, cycles)))
        mean_native = plant.spec.coordinates.to_native(agent.mean)
        values = (
            plant.logical_risk_native(fixed_native[None, :], optimum_native)[0],
            plant.logical_risk_native(mean_native[None, :], optimum_native)[0],
            plant.logical_risk_native(batch.applied_native_actions, optimum_native).mean(),
            plant.logical_risk_native(optimum_native[None, :], optimum_native)[0],
        )
        edrs = (
            plant.detector_rates_native(fixed_native[None, :], optimum_native).mean(),
            plant.detector_rates_native(mean_native[None, :], optimum_native).mean(),
            plant.detector_rates_native(batch.applied_native_actions, optimum_native).mean(),
            plant.detector_rates_native(optimum_native[None, :], optimum_native).mean(),
        )
        for key, value in zip(POLICY_CLASSES, values):
            traces[key].append(float(value))
        for key, value in zip(POLICY_CLASSES, edrs):
            detector[key].append(float(value))
        mean_vectors.append(agent.mean.copy())
        scale_vectors.append(agent.scale.copy())
    output = {
        "logical_risk": {key: np.asarray(value) for key, value in traces.items()},
        "detector_rate": {key: np.asarray(value) for key, value in detector.items()},
        "learned_mean_vectors": np.asarray(mean_vectors),
        "policy_scale_vectors": np.asarray(scale_vectors),
        "diagnostics": diagnostics,
    }
    validate_policy_schema(output["logical_risk"])
    validate_policy_schema(output["detector_rate"])
    return output


def response_metrics(traces: Mapping[str, np.ndarray], family: str, *, onset: int = 0,
                     minimum_denominator: float = 1e-10) -> dict[str, Any]:
    fixed = np.asarray(traces["fixed_policy"], dtype=float)
    learned = np.asarray(traces["learned_mean"], dtype=float)
    oracle = np.asarray(traces["oracle_optimum"], dtype=float)
    fixed_excess = fixed - oracle
    learned_excess = learned - oracle
    denominator = float(np.sum(np.abs(fixed_excess[onset:])))
    identifiable = denominator > minimum_denominator
    ratio = float(np.sum(np.abs(learned_excess[onset:])) / denominator) if identifiable else None
    result: dict[str, Any] = {
        "family": family,
        "denominator_identifiable": identifiable,
        "fixed_integrated_absolute_excess": denominator,
        "learned_integrated_absolute_excess": float(np.sum(np.abs(learned_excess[onset:]))),
        "integrated_excess_error_ratio_mean_over_fixed": ratio,
    }
    if family == "step":
        peak = float(np.max(np.abs(learned_excess[onset:])))
        threshold = max(0.1 * peak, 1e-12)
        hits = np.flatnonzero(np.abs(learned_excess[onset:]) <= threshold)
        result.update({"peak_excess": peak, "settling_time": int(hits[0]) if len(hits) else None,
                       "steady_state_error": float(np.mean(np.abs(learned_excess[-max(2, len(learned)//8):])))})
    return result


def sine_gain_phase(optimum_scalar: np.ndarray, response_scalar: np.ndarray, frequency_cycles: float) -> dict[str, float]:
    target = np.asarray(optimum_scalar, dtype=float)
    response = np.asarray(response_scalar, dtype=float)
    t = np.arange(len(target), dtype=float)
    basis = np.exp(-2j * np.pi * frequency_cycles * t / len(target))
    target_coefficient = np.sum(target * basis)
    response_coefficient = np.sum(response * basis)
    gain = float(abs(response_coefficient) / max(abs(target_coefficient), 1e-15))
    phase = float(np.angle(response_coefficient / target_coefficient)) if abs(target_coefficient) > 1e-15 else 0.0
    lag_epochs = float(-phase * len(target) / (2.0 * np.pi * frequency_cycles))
    return {"gain": gain, "phase_radians": phase, "phase_lag_epochs": lag_epochs}
