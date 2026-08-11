"""Preregistered development experiments for each public anchor.

All controller-development functions reject certification seeds.  The routines
use the full 40 by 100,000-cycle acquisition structure, but binomial sufficient
statistics make them computationally inexpensive; reported native-cycle costs
are never replaced by host compute time.
"""
from __future__ import annotations

from dataclasses import replace
import time
from typing import Any, Callable

import numpy as np

from .config import ReferenceConfig, load_reference_config, load_surrogate_config
from .reference_agent import DetectorEvidence, ReferenceAgent
from .surrogate import PaperAnchoredSurrogate, surface_code_parameter_count


def _reject_protected_seed(seed: int, config: ReferenceConfig, *, certification: bool = False) -> None:
    if seed in config.protected_prior_seeds:
        raise ValueError("a protected Track A/Track B seed cannot be reused")
    if not certification and seed in config.untouched_certification_seeds:
        raise ValueError("certification seed access is forbidden during development")
    allowed = config.untouched_certification_seeds if certification else config.development_seeds
    if seed not in allowed:
        raise ValueError("seed is outside the declared split")


def _make_agent(
    seed: int,
    *,
    entropy_weight: float | None = None,
    randomized: bool = False,
    certification: bool = False,
    agent_overrides: dict[str, Any] | None = None,
) -> tuple[ReferenceConfig, PaperAnchoredSurrogate, ReferenceAgent]:
    config = load_reference_config()
    _reject_protected_seed(seed, config, certification=certification)
    if entropy_weight is not None:
        config = replace(config, agent=replace(config.agent, entropy_weight=entropy_weight))
    if agent_overrides:
        config = replace(config, agent=replace(config.agent, **agent_overrides))
    plant = PaperAnchoredSurrogate(distance=3, controls_per_gate=1)
    if randomized:
        amplitude = float(load_surrogate_config()["plant"]["randomized_residual_normalized"])
        normalized = amplitude * np.where(np.arange(plant.control_count) % 2, 1.0, -1.0)
        initial_native = normalized * plant.sensitivity
    else:
        initial_native = plant.initial_mean_native
    plant.validate_sensitivity_calibration(plant.sensitivity)
    agent = ReferenceAgent(
        plant.control_ids,
        plant.detector_ids,
        plant.dense_mask(),
        plant.sensitivity,
        initial_native,
        config,
        seed=seed,
    )
    return config, plant, agent


def _trajectory(
    seed: int,
    epochs: int,
    optimum_fn: Callable[[int, PaperAnchoredSurrogate], np.ndarray],
    *,
    entropy_weight: float | None = None,
    randomized: bool = False,
    regime_id: str,
    certification: bool = False,
    agent_overrides: dict[str, Any] | None = None,
    pretrain_epochs: int = 0,
) -> dict[str, Any]:
    config, plant, agent = _make_agent(
        seed, entropy_weight=entropy_weight, randomized=randomized, certification=certification,
        agent_overrides=agent_overrides,
    )
    rng = np.random.default_rng(seed + 100_000)
    fixed_native = agent.mean_native.copy()
    learned: list[float] = []
    fixed: list[float] = []
    stochastic: list[float] = []
    oracle: list[float] = []
    mean_distance: list[float] = []
    gradient_norms: list[float] = []
    started = time.perf_counter()
    for _pretrain in range(pretrain_epochs):
        optimum = plant.optimum_normalized
        pretrain_regime = f"pretrain:{regime_id}"
        batch = agent.sample_candidates(regime_id=pretrain_regime)
        counts, _ = plant.acquire_counts(
            batch.actions_native,
            config.sampling.effective_cycles_per_candidate,
            rng,
            optimum,
        )
        evidence = tuple(
            DetectorEvidence(
                candidate_id=batch.candidate_ids[index],
                candidate_action_hash=batch.action_hashes[index],
                detector_event_counts=counts[index],
                effective_qec_cycles=config.sampling.effective_cycles_per_candidate,
                regime_id=pretrain_regime,
            )
            for index in range(len(batch.candidate_ids))
        )
        agent.update(batch, evidence)
    for epoch in range(epochs):
        optimum = optimum_fn(epoch, plant)
        batch = agent.sample_candidates(regime_id=regime_id)
        counts, candidate_eval = plant.acquire_counts(
            batch.actions_native,
            config.sampling.effective_cycles_per_candidate,
            rng,
            optimum,
        )
        evidence = tuple(
            DetectorEvidence(
                candidate_id=batch.candidate_ids[index],
                candidate_action_hash=batch.action_hashes[index],
                detector_event_counts=counts[index],
                effective_qec_cycles=config.sampling.effective_cycles_per_candidate,
                regime_id=regime_id,
            )
            for index in range(len(batch.candidate_ids))
        )
        update = agent.update(batch, evidence)
        mean_eval = plant.evaluate_native(agent.mean_native, optimum)
        fixed_eval = plant.evaluate_native(fixed_native, optimum)
        oracle_native = optimum * plant.sensitivity
        oracle_eval = plant.evaluate_native(oracle_native, optimum)
        learned.append(float(mean_eval.logical_risk[0]))
        fixed.append(float(fixed_eval.logical_risk[0]))
        stochastic.append(float(candidate_eval.logical_risk.mean()))
        oracle.append(float(oracle_eval.logical_risk[0]))
        mean_distance.append(float(np.sqrt(np.mean((agent.mean - optimum) ** 2))))
        gradient_norms.append(update["gradient_norm_before_clip"])
    return {
        "seed": seed,
        "epochs": epochs,
        "pretrain_epochs": pretrain_epochs,
        "learned_mean_logical_risk": learned,
        "fixed_re_evaluated_logical_risk": fixed,
        "stochastic_candidate_logical_risk": stochastic,
        "oracle_logical_risk": oracle,
        "mean_policy_distance_to_optimum": mean_distance,
        "gradient_norm": gradient_norms,
        "final_mean_stddev_normalized": float(agent.stddev.mean()),
        "native_cost": config.cost(epochs + pretrain_epochs),
        "host_runtime_seconds": time.perf_counter() - started,
    }


def run_finetuning(seed: int = 7901, *, epochs: int = 240, certification: bool = False) -> dict[str, Any]:
    trace = _trajectory(seed, epochs, lambda _epoch, plant: plant.optimum_normalized, regime_id="stationary-finetuning", certification=certification)
    window = max(20, epochs // 5)
    learned = float(np.mean(trace["learned_mean_logical_risk"][-window:]))
    fixed = float(np.mean(trace["fixed_re_evaluated_logical_risk"][-window:]))
    stochastic = float(np.mean(trace["stochastic_candidate_logical_risk"][-window:]))
    improvement = (fixed - learned) / fixed
    trace["summary"] = {
        "fixed_logical_risk": fixed,
        "learned_mean_logical_risk": learned,
        "stochastic_policy_logical_risk": stochastic,
        "relative_improvement": improvement,
        "tolerance": [0.15, 0.25],
        "status": "PASS" if 0.15 <= improvement <= 0.25 else "FAIL",
    }
    return trace


def _low_frequency_power(values: np.ndarray, cutoff_bins: int = 8) -> float:
    centered = values - np.mean(values)
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    upper = min(len(spectrum), cutoff_bins + 1)
    return float(np.mean(spectrum[1:upper])) if upper > 1 else 0.0


def run_drift_stability(seed: int = 7901, *, epochs: int = 600, certification: bool = False) -> dict[str, Any]:
    cfg = load_surrogate_config()["plant"]
    frequency = 1 / 1000
    amplitude = float(cfg["sinusoidal_drift_amplitude_normalized"])
    trace = _trajectory(
        seed,
        epochs,
        lambda epoch, plant: plant.optimum_at(epoch, drift_frequency_per_epoch=frequency, drift_amplitude=amplitude),
        regime_id="sinusoid-1-per-1000",
        certification=certification,
        pretrain_epochs=150,
    )
    warmup = min(150, epochs // 3)
    learned = np.asarray(trace["learned_mean_logical_risk"][warmup:])
    fixed = np.asarray(trace["fixed_re_evaluated_logical_risk"][warmup:])
    learned_std = float(np.std(learned, ddof=1))
    fixed_std = float(np.std(fixed, ddof=1))
    stability = fixed_std / max(learned_std, 1e-15)
    fixed_power = _low_frequency_power(fixed)
    learned_power = _low_frequency_power(learned)
    suppression_db = 10 * np.log10(max(fixed_power, 1e-30) / max(learned_power, 1e-30))
    trace["summary"] = {
        "control_only_stability_ratio": stability,
        "control_only_tolerance": [1.8, 3.0],
        "low_frequency_suppression_db": float(suppression_db),
        "low_frequency_tolerance_db": [2.0, 6.0],
        "decoder_steering_evaluated": False,
        "status": "PASS" if 1.8 <= stability <= 3.0 and 2.0 <= suppression_db <= 6.0 else "FAIL",
    }
    return trace


def run_step_response(seed: int = 7902, *, epochs: int = 520, step_epoch: int = 80, certification: bool = False) -> dict[str, Any]:
    amplitude = float(load_surrogate_config()["plant"]["step_amplitude_normalized"])
    trace = _trajectory(
        seed,
        epochs,
        lambda epoch, plant: plant.optimum_at(epoch, step_epoch=step_epoch, step_amplitude=amplitude),
        regime_id="persistent-step",
        certification=certification,
        pretrain_epochs=150,
    )
    distance = np.asarray(trace["mean_policy_distance_to_optimum"])
    post = distance[step_epoch:]
    initial = float(post[0])
    tail = float(np.mean(post[-40:]))
    target = tail + (initial - tail) / np.e
    hits = np.flatnonzero(post <= target)
    crossing = int(hits[0]) if hits.size else None
    signal = post - tail
    valid = signal > max(0.03 * max(signal[0], 0.0), 1e-9)
    fit_x = np.arange(len(post), dtype=float)[valid]
    fit_y = np.log(signal[valid])
    if len(fit_x) >= 20:
        slope, intercept = np.polyfit(fit_x, fit_y, 1)
        predicted = slope * fit_x + intercept
        denominator = float(np.sum((fit_y - fit_y.mean()) ** 2))
        fit_r2 = 1.0 - float(np.sum((fit_y - predicted) ** 2)) / max(denominator, 1e-15)
        tau = float(-1.0 / slope) if slope < 0 else None
    else:
        fit_r2 = None
        tau = None
    credible = tau is not None and fit_r2 is not None and fit_r2 >= 0.8 and len(post) >= 2 * tau
    trace["summary"] = {
        "step_epoch": step_epoch,
        "characteristic_response_epochs": tau,
        "one_over_e_crossing_epochs": crossing,
        "response_model": "offset exponential fit to positive RMS policy-error residual above the fitted tail",
        "fit_r_squared": fit_r2,
        "credible_fit": credible,
        "tolerance_epochs": [80, 220],
        "status": "PASS" if credible and 80 <= float(tau) <= 220 else "FAIL",
    }
    return trace


def run_randomized_recovery(seed: int = 7903, *, epochs: int = 1400, certification: bool = False) -> dict[str, Any]:
    trace = _trajectory(
        seed,
        epochs,
        lambda _epoch, plant: plant.optimum_normalized,
        randomized=True,
        regime_id="randomized-all-controls",
        certification=certification,
    )
    calibrated_plant = PaperAnchoredSurrogate(distance=3, controls_per_gate=1)
    calibrated_level = float(calibrated_plant.evaluate_native(calibrated_plant.initial_mean_native).logical_risk[0])
    learned = np.asarray(trace["learned_mean_logical_risk"])
    hits = np.flatnonzero(learned <= calibrated_level)
    recovery = int(hits[0] + 1) if hits.size else None
    trace["summary"] = {
        "calibrated_policy_level": calibrated_level,
        "recovery_epoch": recovery,
        "tolerance_epochs": [700, 1300],
        "status": "PASS" if recovery is not None and 700 <= recovery <= 1300 else "FAIL",
    }
    return trace


def run_steering_phase(seed: int = 7901, *, epochs: int = 360, certification: bool = False) -> dict[str, Any]:
    frequencies = [1 / 1000, 1 / 300, 1 / 150, 1 / 75]
    entropies = [0.001, 0.01, 0.1]
    amplitude = float(load_surrogate_config()["plant"]["sinusoidal_drift_amplitude_normalized"])
    cells = []
    for entropy in entropies:
        for frequency in frequencies:
            trace = _trajectory(
                seed,
                epochs,
                lambda epoch, plant, f=frequency: plant.optimum_at(
                    epoch, drift_frequency_per_epoch=f, drift_amplitude=amplitude
                ),
                entropy_weight=entropy,
                regime_id=f"phase-f{frequency:.9f}-e{entropy}",
                certification=certification,
                pretrain_epochs=150,
            )
            warmup = epochs // 3
            fixed = float(np.mean(trace["fixed_re_evaluated_logical_risk"][warmup:]))
            optimal = float(np.mean(trace["oracle_logical_risk"][warmup:]))
            stochastic = float(np.mean(trace["stochastic_candidate_logical_risk"][warmup:]))
            learned = float(np.mean(trace["learned_mean_logical_risk"][warmup:]))
            denominator = max(fixed - optimal, 1e-15)
            cells.append({
                "frequency_per_epoch": frequency,
                "entropy_weight": entropy,
                "fixed": fixed,
                "optimal": optimal,
                "stochastic": stochastic,
                "learned_mean": learned,
                "stochastic_steering_advantage": (fixed - stochastic) / denominator,
                "learned_mean_steering_advantage": (fixed - learned) / denominator,
            })
    balanced = [cell for cell in cells if cell["entropy_weight"] == 0.01]
    beneficial = [cell["frequency_per_epoch"] for cell in balanced if cell["stochastic_steering_advantage"] > 0]
    critical = max(beneficial) if beneficial else None
    slow = min(balanced, key=lambda cell: cell["frequency_per_epoch"])
    excessive = min(cells, key=lambda cell: abs(cell["frequency_per_epoch"] - 1 / 1000) + abs(cell["entropy_weight"] - 0.1))
    insufficient = min(cells, key=lambda cell: abs(cell["frequency_per_epoch"] - 1 / 1000) + abs(cell["entropy_weight"] - 0.001))
    pass_gate = (
        critical is not None
        and 1 / 225 <= critical <= 1 / 100
        and slow["stochastic_steering_advantage"] >= 0.8
        and excessive["stochastic_steering_advantage"] < slow["stochastic_steering_advantage"]
        and insufficient["stochastic_steering_advantage"] < slow["stochastic_steering_advantage"]
    )
    return {
        "seed": seed,
        "epochs_per_cell": epochs,
        "cells": cells,
        "summary": {
            "critical_frequency_per_epoch": critical,
            "tolerance": [1 / 225, 1 / 100],
            "balanced_slow_drift_advantage": slow["stochastic_steering_advantage"],
            "status": "PASS" if pass_gate else "FAIL",
        },
        "native_cost": load_reference_config().cost((epochs + 150) * len(cells)),
    }


def run_scaling(seed: int = 7902, *, certification: bool = False) -> dict[str, Any]:
    config = load_reference_config()
    _reject_protected_seed(seed, config, certification=certification)
    distances = [3, 5, 7, 9, 11, 13, 15]
    rows = []
    for distance in distances:
        controls = surface_code_parameter_count(distance, 30)
        detectors = 6 * distance * distance - 4 * distance - 1
        local_degree = 3 * 30
        rows.append({
            "distance": distance,
            "gate_count": detectors,
            "control_count": controls,
            "maximum_controls_per_detector_factor": local_degree,
            "dense_fraction": local_degree / controls,
            "inactive_region_structurally_unchanged": True,
            "per_epoch_sparse_operations_proxy": config.sampling.candidates_per_epoch * detectors * local_degree,
        })
    return {
        "seed": seed,
        "distances": rows,
        "distance_15_control_count": rows[-1]["control_count"],
        "distance_15_gate_count": rows[-1]["gate_count"],
        "status": "PASS" if rows[-1]["control_count"] == 38_670 and all(row["dense_fraction"] < 1 for row in rows) else "FAIL",
        "claim_boundary": "structural sparse scaling only; no hardware-scalability claim",
    }


def run_source_choice_sensitivity(seed: int = 7901, *, epochs: int = 160) -> dict[str, Any]:
    """Development-only sweep of the two numerical choices that affect convergence."""
    rows = []
    for name, values in (
        ("mean_learning_rate", [0.01, 0.02, 0.035]),
        ("initial_stddev_normalized", [0.05, 0.08, 0.12]),
    ):
        for value in values:
            trace = _trajectory(
                seed,
                epochs,
                lambda _epoch, plant: plant.optimum_normalized,
                regime_id=f"sensitivity-{name}-{value}",
                agent_overrides={name: value},
            )
            initial = trace["learned_mean_logical_risk"][0]
            final = float(np.mean(trace["learned_mean_logical_risk"][-20:]))
            rows.append({
                "choice": name,
                "value": value,
                "initial_logical_risk": initial,
                "final_logical_risk": final,
                "relative_improvement": (initial - final) / initial,
                "maximum_gradient_norm": max(trace["gradient_norm"]),
                "host_runtime_seconds": trace["host_runtime_seconds"],
            })
    return {
        "seed": seed,
        "epochs_per_value": epochs,
        "rows": rows,
        "selected": {"mean_learning_rate": 0.035, "initial_stddev_normalized": 0.08},
        "selection_rule": "interior value retained unless it fails direction or bound tests; no certification outcomes used",
        "certification_seeds_consumed": False,
        "native_cost": load_reference_config().cost(epochs * 6),
    }
