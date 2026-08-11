"""Source-correct production acquisitions with independent evaluation noise."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from hdfa_rl_suite.google_pure_v6.experiments import POLICY_CLASSES, validate_policy_schema
from hdfa_rl_suite.google_pure_v6.plant import PureQuadraticPlant
from hdfa_rl_suite.google_pure_v6.reference_agent import evidence_from_counts

from .controller import build_production_agent, require_resolved_controller


def run_production_trace(plant: PureQuadraticPlant, optimum_normalized: np.ndarray, *, seed: int,
                         candidates: int, cycles: int, logical_evaluation_cycles: int = 20000) -> dict[str, Any]:
    controller = require_resolved_controller()
    tape = np.asarray(optimum_normalized, dtype=float)
    if tape.ndim != 2 or tape.shape[1] != plant.spec.control_count:
        raise ValueError("optimum tape shape mismatch")
    agent = build_production_agent(plant.mask, plant.spec.base_optimum_normalized, plant.spec.coordinates, seed=seed)
    acquisition_rng = np.random.default_rng(seed + 100_000)
    evaluation_rng = np.random.default_rng(seed + 200_000)
    fixed_native = plant.base_optimum_native.copy()
    risk = {name: [] for name in POLICY_CLASSES}
    detector = {name: [] for name in POLICY_CLASSES}
    logical_evaluation = {name: [] for name in POLICY_CLASSES}
    means, scales, diagnostics = [], [], []
    for optimum in tape:
        optimum_native = plant.spec.coordinates.to_native(optimum)
        batch = agent.sample(candidates)
        counts = plant.acquire_counts(batch.applied_native_actions, optimum_native, cycles=cycles, rng=acquisition_rng)
        diagnostics.append(agent.update(batch, evidence_from_counts(batch, counts, cycles)))
        mean_native = plant.spec.coordinates.to_native(agent.mean)
        actions = (fixed_native[None, :], mean_native[None, :], batch.applied_native_actions,
                   optimum_native[None, :])
        for name, action in zip(POLICY_CLASSES, actions):
            logical = plant.logical_risk_native(action, optimum_native)
            value = float(np.mean(logical))
            risk[name].append(value)
            detector[name].append(float(np.mean(plant.detector_rates_native(action, optimum_native))))
            evaluation_probability = float(np.clip(value, 1e-9, 1.0 - 1e-9))
            logical_evaluation[name].append(float(evaluation_rng.binomial(logical_evaluation_cycles,
                                                                           evaluation_probability) /
                                                  logical_evaluation_cycles))
        means.append(agent.mean.copy())
        scales.append(agent.scale.copy())
    output = {
        "logical_risk": {name: np.asarray(values) for name, values in risk.items()},
        "detector_rate": {name: np.asarray(values) for name, values in detector.items()},
        "logical_evaluation": {name: np.asarray(values) for name, values in logical_evaluation.items()},
        "learned_mean_vectors": np.asarray(means),
        "policy_scale_vectors": np.asarray(scales),
        "diagnostics": diagnostics,
        "controller_code_hash": controller["controller_code_hash"],
        "resolved_config_hash": controller["resolved_config_hash"],
        "objective_mode": "source_mapped_v7_production_ppo",
        "baseline_mode": "per_detector_frozen_batch_ema",
        "replay_mode": f"fixed_fifo_{controller['parameters']['replay_capacity_epochs']}_epochs",
        "units_mode": "latent_normalized_likelihood_bounded_native_application",
        "entropy_mode": "once_per_coordinate",
        "accounting": {
            "epochs": len(tape),
            "candidates_per_epoch": candidates,
            "effective_cycles_per_candidate": cycles,
            "candidate_qec_cycles": len(tape) * candidates * cycles,
            "logical_evaluation_cycles": len(tape) * len(POLICY_CLASSES) * logical_evaluation_cycles,
        },
    }
    for key in ("logical_risk", "detector_rate", "logical_evaluation"):
        validate_policy_schema(output[key])
    return output


def trace_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = result["diagnostics"]
    return {
        "mean_gradient_norm": float(np.mean([row["mean_gradient_norm"] for row in diagnostics])),
        "mean_clipping_fraction": float(np.mean([row["clip_fraction"] for row in diagnostics])),
        "mean_policy_scale": float(np.mean(result["policy_scale_vectors"])),
        "scale_floor_hits": int(np.sum(result["policy_scale_vectors"] <= 0.0400000001)),
        "scale_ceiling_hits": int(np.sum(result["policy_scale_vectors"] >= 0.2499999999)),
    }
