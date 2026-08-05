"""Preregistered scale, entropy, adaptation, and held-out dynamic studies."""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from google_rl_reimplementation.google_pure_v6.plant import PureQuadraticPlant, default_spec
from google_rl_reimplementation.google_pure_v6.reference_agent import PureGoogleV6Agent, evidence_from_counts
from google_rl_reimplementation.google_pure_v7.config import canonical_hash, repository_root, sha256_file
from google_rl_reimplementation.google_pure_v7.sine import fit_sine_tracking, wrap_phase

from .common import artifact_root, guard_seed, load_config, protocol_hash, read_json, write_artifact
from .contracts import (
    ControllerConfig,
    TemporalProtocol,
    controller_selection_gates,
    entropy_operationality,
    five_policy_decomposition,
    scale_floor_classification,
    validate_source_choices,
    window_stability,
)


POLICIES = ("fixed", "oracle", "oracle_with_scale", "learned_mean", "sampled_candidates")


def _direction(count: int) -> np.ndarray:
    values = np.linspace(1.0, 0.45, count)
    return values / np.linalg.norm(values)


def _plant_hash(plant: PureQuadraticPlant) -> str:
    return canonical_hash(
        {
            "mask": plant.mask.tolist(),
            "normalized_curvature": plant.spec.normalized_curvature.tolist(),
            "detector_floor": plant.spec.detector_floor.tolist(),
            "native_per_normalized": plant.spec.coordinates.native_per_normalized.tolist(),
        }
    )


def _mean_and_ci(values: Iterable[float]) -> tuple[float, list[float]]:
    array = np.asarray(tuple(values), dtype=float)
    mean = float(np.mean(array))
    if len(array) < 2:
        return mean, [mean, mean]
    half = 1.959963984540054 * float(np.std(array, ddof=1) / np.sqrt(len(array)))
    return mean, [mean - half, mean + half]


def _window_metrics(epoch_costs: Mapping[str, list[float]], start: int, stop: int) -> dict[str, Any]:
    costs = {name: float(np.mean(np.asarray(values, dtype=float)[start:stop])) for name, values in epoch_costs.items()}
    return {"costs": costs, **five_policy_decomposition(costs)}


def run_dynamic_cell(
    config: ControllerConfig,
    *,
    frequency: float,
    phase: float,
    burn_in_periods: int,
    primary_periods: int,
    seed: int,
    candidates: int,
    cycles: int,
    window_tolerance: float = 0.15,
) -> dict[str, Any]:
    """Run one matched cell with a primary and plus/minus-one-period analysis."""
    guard_seed(seed)
    if primary_periods < 2:
        raise ValueError("window sensitivity requires at least two primary periods")
    period = int(round(1.0 / float(frequency)))
    if period <= 0 or not np.isclose(period * frequency, 1.0, rtol=0, atol=1e-10):
        raise ValueError("frequency must have an integer number of epochs per period")
    burn_in_epochs = burn_in_periods * period
    primary_epochs = primary_periods * period
    epochs = burn_in_epochs + (primary_periods + 1) * period
    plant = PureQuadraticPlant(default_spec(6))
    agent = PureGoogleV6Agent(
        plant.mask,
        plant.spec.base_optimum_normalized,
        plant.spec.coordinates,
        config.to_agent_choices(),
        seed=seed,
        objective_mode="source_literal_ppo",
    )
    training_rng = np.random.default_rng(seed + 100_000)
    evaluation_rng = np.random.default_rng(seed + 200_000)
    oracle_scale_rng = np.random.default_rng(seed + 300_000)
    direction = _direction(plant.spec.control_count)
    amplitude = 0.45
    time = np.arange(epochs, dtype=float)
    tape = amplitude * np.sin(2 * np.pi * frequency * time + phase)[:, None] * direction[None, :]
    epoch_costs = {name: [] for name in POLICIES}
    mean_vectors: list[np.ndarray] = []
    scale_vectors: list[np.ndarray] = []
    action_clipping: list[float] = []
    ppo_clipping: list[float] = []
    native_displacement: list[float] = []
    ratio_ranges: list[list[float]] = []
    training_events = 0
    for epoch, optimum in enumerate(tape):
        optimum_native = plant.spec.coordinates.to_native(optimum)
        mean = agent.mean.copy()
        mean_native = plant.spec.coordinates.to_native(mean)
        scale = agent.scale.copy()
        batch = agent.sample(candidates)
        fixed_actions = np.repeat(plant.spec.base_optimum_normalized[None, :], candidates, axis=0)
        oracle_actions = np.repeat(optimum[None, :], candidates, axis=0)
        oracle_with_scale = plant.spec.coordinates.apply_bounds(
            optimum[None, :] + scale[None, :] * oracle_scale_rng.normal(size=(candidates, plant.spec.control_count))
        )
        learned_actions = np.repeat(mean[None, :], candidates, axis=0)
        policy_actions = {
            "fixed": fixed_actions,
            "oracle": oracle_actions,
            "oracle_with_scale": oracle_with_scale,
            "learned_mean": learned_actions,
            "sampled_candidates": batch.applied_normalized_actions,
        }
        for name, normalized_actions in policy_actions.items():
            native_actions = plant.spec.coordinates.to_native(normalized_actions)
            counts = plant.acquire_counts(native_actions, optimum_native, cycles=cycles, rng=evaluation_rng)
            epoch_costs[name].append(float(np.sum(counts) / (candidates * cycles)))
        training_counts = plant.acquire_counts(
            batch.applied_native_actions,
            optimum_native,
            cycles=cycles,
            rng=training_rng,
        )
        training_events += int(np.sum(training_counts))
        latent = np.asarray(batch.latent_normalized_actions)
        applied = np.asarray(batch.applied_normalized_actions)
        action_clipping.append(float(np.mean(latent != applied)))
        native_displacement.append(float(np.sqrt(np.mean((batch.applied_native_actions - mean_native[None, :]) ** 2))))
        diagnostic = agent.update(batch, evidence_from_counts(batch, training_counts, cycles))
        ppo_clipping.append(float(diagnostic["clip_fraction"]))
        ratio_ranges.append([float(diagnostic["ratio_min"]), float(diagnostic["ratio_max"])])
        mean_vectors.append(mean)
        scale_vectors.append(scale)
    primary_start = burn_in_epochs
    primary_stop = primary_start + primary_epochs
    minus_stop = primary_stop - period
    plus_stop = primary_stop + period
    windows = {
        "minus_one_period": _window_metrics(epoch_costs, primary_start, minus_stop),
        "primary": _window_metrics(epoch_costs, primary_start, primary_stop),
        "plus_one_period": _window_metrics(epoch_costs, primary_start, plus_stop),
    }
    stability = window_stability(
        windows["primary"], windows["minus_one_period"], windows["plus_one_period"], tolerance=window_tolerance
    )
    means = np.asarray(mean_vectors)
    scales = np.asarray(scale_vectors)
    projection = means @ direction
    fit = fit_sine_tracking(
        time,
        projection,
        optimum_amplitude=amplitude,
        omega_radians_per_epoch=2 * np.pi * frequency,
        burn_in_epochs=burn_in_epochs,
        minimum_complete_periods=max(3, primary_periods),
        documented_drift_tape=True,
    )
    relative_phase = wrap_phase(float(fit["phase_radians"]) - float(phase))
    phase_lag_epochs = -relative_phase / (2 * np.pi * frequency)
    primary = windows["primary"]
    diff = np.asarray(epoch_costs["fixed"])[primary_start:primary_stop] - np.asarray(epoch_costs["oracle"])[primary_start:primary_stop]
    denominator_se = float(np.std(diff, ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else float("inf")
    scale_change = np.mean(scales, axis=1) / config.initial_scale - 1.0
    adaptation_hits = np.flatnonzero(np.abs(scale_change) >= 0.05)
    expected_impulse = epochs * config.scale_learning_rate * config.entropy_coefficient
    measured_log_change = float(np.log(np.mean(agent.scale) / config.initial_scale))
    floor_fraction = float(np.mean(scales <= config.minimum_scale * (1 + 1e-9)))
    ceiling_fraction = float(np.mean(scales >= config.maximum_scale * (1 - 1e-9)))
    gain_ci = fit["confidence_intervals_95"]["amplitude_gain"]
    return {
        "schema_version": "google-pure-v9-dynamic-cell.v1",
        "controller": config.to_dict(),
        "controller_hash": canonical_hash(config.to_dict()),
        "plant_hash": _plant_hash(plant),
        "graph_hash": canonical_hash(plant.mask.tolist()),
        "protocol_hash": protocol_hash(
            {
                "frequency": frequency,
                "phase": phase,
                "burn_in_periods": burn_in_periods,
                "primary_periods": primary_periods,
                "candidates": candidates,
                "cycles": cycles,
            }
        ),
        "seed": seed,
        "frequency_cycles_per_epoch": float(frequency),
        "period_epochs": period,
        "phase_radians": float(phase),
        "burn_in_epochs": burn_in_epochs,
        "analysis_duration_epochs": primary_epochs,
        "complete_primary_periods": primary_periods,
        "analysis_window": [primary_start, primary_stop],
        "windows": windows,
        "window_sensitivity": stability,
        "five_policy": primary,
        "initial_scale": config.initial_scale,
        "minimum_scale": config.minimum_scale,
        "maximum_scale": config.maximum_scale,
        "final_scale": float(np.mean(agent.scale)),
        "mean_scale": float(np.mean(scales)),
        "scale_trajectory": np.mean(scales, axis=1).tolist(),
        "fraction_at_floor": floor_fraction,
        "fraction_at_ceiling": ceiling_fraction,
        "scale_floor_classification": scale_floor_classification(floor_fraction),
        "scale_adaptation_time_epochs": int(adaptation_hits[0]) if len(adaptation_hits) else None,
        "expected_entropy_only_log_scale_impulse": expected_impulse,
        "measured_log_scale_change": measured_log_change,
        "objective_normalization": "sum detectors then mean candidates",
        "reward_aggregation": "per-detector negative finite-shot event rate",
        "entropy_aggregation": "once per policy coordinate",
        "native_candidate_displacement": float(np.mean(native_displacement)),
        "clipping_fraction": float(np.mean(action_clipping)),
        "ppo_clip_fraction": float(np.mean(ppo_clipping)),
        "importance_ratio_range": [float(np.min(ratio_ranges)), float(np.max(ratio_ranges))],
        "tracking": {
            **fit,
            "relative_phase_radians": relative_phase,
            "phase_lag_epochs_relative_to_optimum": float(phase_lag_epochs),
            "phase_identifiable": bool(fit["fit_r_squared"] >= 0.2 and np.all(np.isfinite(gain_ci))),
        },
        "denominator_standard_error": denominator_se,
        "denominator_resolution_3se": 3 * denominator_se,
        "denominator_statistically_resolved": bool(float(primary["D_fixed"]) > 3 * denominator_se),
        "drift_tape_hash": canonical_hash(tape.tolist()),
        "matched_policy_windows": True,
        "training_qec_cycles": epochs * candidates * cycles,
        "evaluation_qec_cycles_per_policy": epochs * candidates * cycles,
        "training_raw_detector_events": training_events,
        "plant_modified": False,
    }


def corrected_fault_classification() -> dict[str, Any]:
    payload = {
        "schema_version": "google-pure-v9-corrected-v8-classification.v1",
        "exploration": [
            "CURRENT_OPERATIONAL_POLICY_SCALE_MAKES_STEERING_IMPOSSIBLE",
            "MINIMUM_SCALE_FLOOR_EFFECT_NOT_ESTABLISHED",
        ],
        "entropy": ["ENTROPY_IMPLEMENTATION_PASS", "ENTROPY_SWEEP_NOT_OPERATIONAL_UNDER_TESTED_PROTOCOL"],
        "temporal": ["TEMPORAL_IMPLEMENTATION_PASS", "TEMPORAL_EVALUATION_PROTOCOL_FAILURE"],
        "unresolved_causes": [
            "MEAN_TRACKING_BANDWIDTH_FAILURE",
            "SOURCE_UNSPECIFIED_INITIAL_SCALE",
            "SOURCE_UNSPECIFIED_SCALE_LEARNING_RATE",
            "SOURCE_UNSPECIFIED_ENTROPY_NORMALIZATION",
            "SOURCE_UNSPECIFIED_UPDATE_LIFECYCLE",
            "SYNTHETIC_PLANT_NON_COMMENSURABILITY",
        ],
        "remaining_plausible_causes_empty": False,
        "behavior_changed": False,
        "blocking_reasons": [],
    }
    return write_artifact("corrected_v8_fault_classification", payload, "Corrected v8 Fault Classification")


def _base_config(*, initial_scale: float, entropy: float, scale_learning_rate: float) -> ControllerConfig:
    return ControllerConfig(
        initial_scale=initial_scale,
        minimum_scale=0.001,
        maximum_scale=0.25,
        scale_learning_rate=scale_learning_rate,
        entropy_coefficient=entropy,
        mean_learning_rate=0.02,
    )


def _run_protocol_for_stage(
    configs: list[tuple[str, ControllerConfig]],
    *,
    mode: str,
    seed_start: int,
    execute: bool,
) -> list[dict[str, Any]]:
    development = load_config("development.json")
    if mode != "smoke" and not execute:
        raise RuntimeError("development acquisition requires explicit execute=True")
    if mode == "smoke":
        frequencies = (float(development["smoke"]["frequency_cycles_per_epoch"]),)
        phases = (0.0,)
        burn = int(development["smoke"]["burn_in_periods"])
        periods = int(development["smoke"]["primary_periods"])
        candidates = int(development["smoke"]["candidates"])
        cycles = int(development["smoke"]["cycles_per_candidate"])
    else:
        frequencies = (float(development["frequencies_cycles_per_epoch"][-1]),)
        phases = tuple(map(float, development["phases_radians"]))
        burn = int(development["burn_in_periods"])
        periods = int(development["primary_periods"])
        candidates, cycles = 12, 4000
    rows = []
    index = 0
    for config_id, config in configs:
        for frequency in frequencies:
            for phase in phases:
                row = run_dynamic_cell(
                    config,
                    frequency=frequency,
                    phase=phase,
                    burn_in_periods=burn,
                    primary_periods=periods,
                    seed=seed_start + index,
                    candidates=candidates,
                    cycles=cycles,
                    window_tolerance=float(development["window_tolerance"]),
                )
                row["config_id"] = config_id
                row["mode"] = mode
                rows.append(row)
                index += 1
    return rows


def _group(rows: Iterable[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["controller"][key])].append(row)
    output = []
    for value, records in sorted(grouped.items()):
        output.append(
            {
                key: value,
                "initial_scale": float(np.mean([r["initial_scale"] for r in records])),
                "minimum_scale": float(np.mean([r["minimum_scale"] for r in records])),
                "final_scale": float(np.mean([r["final_scale"] for r in records])),
                "mean_scale": float(np.mean([r["mean_scale"] for r in records])),
                "fraction_at_floor": float(np.mean([r["fraction_at_floor"] for r in records])),
                "fraction_at_ceiling": float(np.mean([r["fraction_at_ceiling"] for r in records])),
                "D_fixed": float(np.mean([r["five_policy"]["D_fixed"] for r in records])),
                "D_tracking": float(np.mean([r["five_policy"]["D_tracking"] for r in records])),
                "D_exploration": float(np.mean([r["five_policy"]["D_exploration"] for r in records])),
                "I_mean": float(np.mean([r["five_policy"]["I_mean"] for r in records])),
                "I_candidate": float(np.mean([r["five_policy"]["I_candidate"] for r in records])),
                "I_oracle_with_scale": float(np.mean([r["five_policy"]["I_oracle_with_scale"] for r in records])),
                "native_candidate_displacement": float(np.mean([r["native_candidate_displacement"] for r in records])),
                "clipping_fraction": float(np.mean([r["clipping_fraction"] for r in records])),
                "measured_log_scale_change": float(np.mean([r["measured_log_scale_change"] for r in records])),
                "expected_entropy_only_log_scale_impulse": float(np.mean([r["expected_entropy_only_log_scale_impulse"] for r in records])),
                "tracking_amplitude_gain": float(np.mean([r["tracking"]["amplitude_gain"] for r in records])),
                "phase_lag_epochs": float(np.mean([r["tracking"]["phase_lag_epochs_relative_to_optimum"] for r in records])),
                "window_stable": all(r["window_sensitivity"]["stable"] for r in records),
                "cell_count": len(records),
            }
        )
    return output


def _save_plot(path: Path, x: Iterable[float], y: Iterable[float], *, xlabel: str, ylabel: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    axis.plot(tuple(x), tuple(y), marker="o")
    axis.axhline(0, color="black", linewidth=0.7)
    axis.set(xlabel=xlabel, ylabel=ylabel, title=title)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plan_stage_a(mode: str = "smoke") -> dict[str, Any]:
    development = load_config("development.json")
    grid = tuple(map(float, development["initial_scale_grid"]))
    smoke = development["smoke"]
    plan = {
        "stage": "A_INITIAL_SCALE",
        "mode": mode,
        "runs": len(grid) if mode == "smoke" else len(grid) * len(development["phases_radians"]),
        "initial_scale_grid": list(grid),
        "epochs": (int(smoke["burn_in_periods"]) + int(smoke["primary_periods"]) + 1) * 12 if mode == "smoke" else "frequency-dependent",
        "periods": int(smoke["primary_periods"]) if mode == "smoke" else development["primary_periods"],
        "qec_cycles": "reported exactly by run artifact",
        "seeds": development["development_seeds"],
        "phases": [0.0] if mode == "smoke" else development["phases_radians"],
        "estimated_runtime": "under one minute smoke; explicit development run otherwise",
        "estimated_storage": "under 20 MiB",
        "controller_hash": canonical_hash({"grid": grid, "mean_learning_rate": development["mean_learning_rate"]}),
    }
    plan["protocol_hash"] = protocol_hash(plan)
    return plan


def run_stage_a(*, mode: str = "smoke", execute: bool = False) -> dict[str, Any]:
    development = load_config("development.json")
    validate_source_choices(development["source_classification"])
    configs = [
        (f"initial-{value:g}", _base_config(initial_scale=float(value), entropy=0.0, scale_learning_rate=0.002))
        for value in development["initial_scale_grid"]
    ]
    cells = _run_protocol_for_stage(configs, mode=mode, seed_start=19301, execute=execute)
    rows = _group(cells, "initial_scale")
    plot_root = artifact_root() / "plots"
    _save_plot(plot_root / "i_mean_vs_initial_scale.png", [r["initial_scale"] for r in rows], [r["I_mean"] for r in rows], xlabel="initial scale", ylabel="I_mean", title="Learned-mean improvement")
    _save_plot(plot_root / "i_candidate_vs_initial_scale.png", [r["initial_scale"] for r in rows], [r["I_candidate"] for r in rows], xlabel="initial scale", ylabel="I_candidate", title="Sampled-candidate improvement")
    _save_plot(plot_root / "exploration_damage_vs_sigma_squared.png", [r["initial_scale"] ** 2 for r in rows], [r["D_exploration"] for r in rows], xlabel="initial scale squared", ylabel="exploration damage", title="Exploration scaling")
    payload = {
        "schema_version": "google-pure-v9-stage-a.v1",
        "mode": mode,
        "plan": plan_stage_a(mode),
        "rows": rows,
        "cells": cells,
        "plant_frozen": True,
        "joint_plant_controller_tuning": False,
        "artifact_complete": True,
        "claim_supported": False,
        "blocking_reasons": ["SMOKE_OR_DEVELOPMENT_RESULTS_ARE_NOT_HELD_OUT_EVIDENCE"],
    }
    return write_artifact("stage_a_initial_scale/results", payload, "Stage A Initial-scale Feasibility", markdown_relative="stage_a_initial_scale/report.md")


def plan_stage_b(mode: str = "smoke") -> dict[str, Any]:
    development = load_config("development.json")
    scales = (0.02, 0.06, 0.14)
    entropies = tuple(map(float, development["entropy_grid"]))
    runs = len(scales) * len(entropies) * (1 if mode == "smoke" else len(development["phases_radians"]))
    payload = {
        "stage": "B_ENTROPY_OPERATIONALITY",
        "mode": mode,
        "runs": runs,
        "initial_scales": list(scales),
        "entropy_grid": list(entropies),
        "epochs": "frequency-dependent",
        "periods": development["smoke"]["primary_periods"] if mode == "smoke" else development["primary_periods"],
        "qec_cycles": "reported exactly by run artifact",
        "seeds": development["development_seeds"],
        "phases": [0.0] if mode == "smoke" else development["phases_radians"],
        "estimated_runtime": "under two minutes smoke; explicit development run otherwise",
        "estimated_storage": "under 30 MiB",
        "controller_hash": canonical_hash({"scales": scales, "entropies": entropies}),
    }
    payload["protocol_hash"] = protocol_hash(payload)
    return payload


def run_stage_b(*, mode: str = "smoke", execute: bool = False) -> dict[str, Any]:
    development = load_config("development.json")
    configs = []
    for scale in (0.02, 0.06, 0.14):
        for entropy in map(float, development["entropy_grid"]):
            configs.append((f"scale-{scale:g}-entropy-{entropy:g}", _base_config(initial_scale=scale, entropy=entropy, scale_learning_rate=0.002)))
    cells = _run_protocol_for_stage(configs, mode=mode, seed_start=19401, execute=execute)
    rows = _group(cells, "entropy_coefficient")
    operational = entropy_operationality(rows)
    plot_root = artifact_root() / "plots"
    _save_plot(plot_root / "final_scale_vs_entropy.png", [r["entropy_coefficient"] for r in rows], [r["final_scale"] for r in rows], xlabel="entropy coefficient", ylabel="final scale", title="Entropy and final policy scale")
    _save_plot(plot_root / "candidate_damage_vs_entropy.png", [r["entropy_coefficient"] for r in rows], [r["D_exploration"] for r in rows], xlabel="entropy coefficient", ylabel="exploration damage", title="Entropy and candidate damage")
    if cells:
        _save_plot(plot_root / "scale_trajectory_vs_epoch.png", range(len(cells[0]["scale_trajectory"])), cells[0]["scale_trajectory"], xlabel="epoch", ylabel="mean scale", title="Representative policy-scale trajectory")
    payload = {
        "schema_version": "google-pure-v9-stage-b.v1",
        "mode": mode,
        "plan": plan_stage_b(mode),
        "rows": rows,
        "cells": cells,
        "entropy_implementation_classification": "ENTROPY_IMPLEMENTATION_PASS",
        "operationality": operational,
        "reward_balance_classification": "ENTROPY_REWARD_BALANCE_NOT_SOURCE_IDENTIFIABLE" if not operational["operational"] else "ENTROPY_AXIS_OPERATIONAL",
        "artifact_complete": True,
        "claim_supported": False,
        "blocking_reasons": [] if operational["operational"] else [operational["classification"]],
    }
    return write_artifact("stage_b_entropy/results", payload, "Stage B Entropy Operationality", markdown_relative="stage_b_entropy/report.md")


def plan_stage_c(mode: str = "smoke") -> dict[str, Any]:
    development = load_config("development.json")
    grid = tuple(map(float, development["scale_learning_rate_grid"]))
    payload = {
        "stage": "C_SCALE_ADAPTATION",
        "mode": mode,
        "runs": len(grid) * (1 if mode == "smoke" else len(development["phases_radians"])),
        "scale_learning_rate_grid": list(grid),
        "mean_learning_rate_frozen": development["mean_learning_rate"],
        "epochs": "frequency-dependent",
        "periods": development["smoke"]["primary_periods"] if mode == "smoke" else development["primary_periods"],
        "qec_cycles": "reported exactly by run artifact",
        "seeds": development["development_seeds"],
        "phases": [0.0] if mode == "smoke" else development["phases_radians"],
        "estimated_runtime": "under one minute smoke; explicit development run otherwise",
        "estimated_storage": "under 20 MiB",
        "controller_hash": canonical_hash({"grid": grid, "mean_learning_rate": development["mean_learning_rate"]}),
    }
    payload["protocol_hash"] = protocol_hash(payload)
    return payload


def run_stage_c(*, mode: str = "smoke", execute: bool = False) -> dict[str, Any]:
    development = load_config("development.json")
    configs = [
        (f"scale-lr-{value:g}", _base_config(initial_scale=0.04, entropy=0.01, scale_learning_rate=float(value)))
        for value in development["scale_learning_rate_grid"]
    ]
    cells = _run_protocol_for_stage(configs, mode=mode, seed_start=19601, execute=execute)
    rows = _group(cells, "scale_learning_rate")
    for row in rows:
        row["training_stable"] = bool(
            np.isfinite(row["final_scale"])
            and row["fraction_at_floor"] < 0.5
            and row["fraction_at_ceiling"] < 0.5
            and row["clipping_fraction"] <= 0.01
        )
    payload = {
        "schema_version": "google-pure-v9-stage-c.v1",
        "mode": mode,
        "plan": plan_stage_c(mode),
        "rows": rows,
        "cells": cells,
        "mean_learning_rate_changed_in_primary_sweep": False,
        "artifact_complete": True,
        "claim_supported": False,
        "blocking_reasons": ["HELD_OUT_VALIDATION_REQUIRED"],
    }
    return write_artifact("stage_c_scale_learning_rate/results", payload, "Stage C Scale Learning-rate Study", markdown_relative="stage_c_scale_learning_rate/report.md")


def freeze_held_out_protocol() -> dict[str, Any]:
    config_path = repository_root() / "configs" / "google_pure_v9" / "held_out.json"
    config = read_json(config_path)
    protocol = TemporalProtocol(
        frequencies=tuple(map(float, config["frequencies_cycles_per_epoch"])),
        phases=tuple(map(float, config["phases_radians"])),
        burn_in_periods=int(config["burn_in_periods"]),
        primary_periods=int(config["primary_periods"]),
        extension_periods=int(config["window_extension_periods"]),
        seeds=tuple(map(int, config["held_out_seeds"])),
        mode="validation",
        window_tolerance=float(config["window_tolerance"]),
    )
    payload = {
        "schema_version": "google-pure-v9-held-out-freeze.v1",
        "configuration_path": config_path.relative_to(repository_root()).as_posix(),
        "configuration_sha256": sha256_file(config_path),
        "configuration": config,
        "temporal_contract": protocol.to_dict(),
        "plan": protocol.plan(
            controller_count=len(config["controller_candidates"]),
            candidates=int(config["candidates"]),
            cycles=int(config["cycles_per_candidate"]),
        ),
        "selection_rules_frozen": True,
        "held_out_results_may_not_change_rules": True,
        "certification_seeds_consumed": False,
        "blocking_reasons": [],
    }
    return write_artifact("stage_d_held_out_validation/frozen_protocol", payload, "Frozen Held-out Dynamic Protocol")


def _config_from_mapping(value: Mapping[str, Any]) -> ControllerConfig:
    return ControllerConfig(
        initial_scale=float(value["initial_scale"]),
        minimum_scale=float(value["minimum_scale"]),
        maximum_scale=float(value["maximum_scale"]),
        scale_learning_rate=float(value["scale_learning_rate"]),
        entropy_coefficient=float(value["entropy_coefficient"]),
        mean_learning_rate=float(value["mean_learning_rate"]),
        replay_capacity_epochs=int(value["replay_capacity_epochs"]),
        update_passes=int(value["update_passes"]),
        ppo_clip=float(value["ppo_clip"]),
    )


def _stage_b_operational() -> bool:
    path = artifact_root() / "stage_b_entropy" / "results.json"
    if not path.is_file():
        return False
    return bool(read_json(path).get("operationality", {}).get("operational", False))


def run_held_out_validation(*, mode: str = "smoke", execute: bool = False) -> dict[str, Any]:
    freeze = freeze_held_out_protocol()
    config = load_config("held_out.json")
    if mode != "smoke" and not execute:
        raise RuntimeError("held-out validation requires explicit execute=True")
    if mode == "smoke":
        protocol = TemporalProtocol(
            frequencies=(1 / 12,),
            phases=tuple(map(float, config["phases_radians"])),
            burn_in_periods=1,
            primary_periods=5,
            extension_periods=1,
            seeds=(19701,),
            mode="smoke",
            window_tolerance=float(config["window_tolerance"]),
        )
        candidates, cycles = 6, 1200
    else:
        protocol = TemporalProtocol(
            frequencies=tuple(map(float, config["frequencies_cycles_per_epoch"])),
            phases=tuple(map(float, config["phases_radians"])),
            burn_in_periods=int(config["burn_in_periods"]),
            primary_periods=int(config["primary_periods"]),
            extension_periods=int(config["window_extension_periods"]),
            seeds=tuple(map(int, config["held_out_seeds"])),
            mode=mode,
            window_tolerance=float(config["window_tolerance"]),
        )
        candidates, cycles = int(config["candidates"]), int(config["cycles_per_candidate"])
    cells = []
    seed_counter = 19801
    for candidate in config["controller_candidates"]:
        controller = _config_from_mapping(candidate)
        for frequency in protocol.frequencies:
            for phase in protocol.phases:
                for _seed in protocol.seeds:
                    cell = run_dynamic_cell(
                        controller,
                        frequency=frequency,
                        phase=phase,
                        burn_in_periods=protocol.burn_in_periods,
                        primary_periods=protocol.primary_periods,
                        seed=seed_counter,
                        candidates=candidates,
                        cycles=cycles,
                        window_tolerance=protocol.window_tolerance,
                    )
                    cell["config_id"] = candidate["id"]
                    cell["mode"] = mode
                    cells.append(cell)
                    seed_counter += 1
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        grouped[cell["config_id"]].append(cell)
    summaries = []
    entropy_ok = _stage_b_operational()
    for config_id, records in grouped.items():
        mean_value, mean_ci = _mean_and_ci(r["five_policy"]["I_mean"] for r in records)
        candidate_value, candidate_ci = _mean_and_ci(r["five_policy"]["I_candidate"] for r in records)
        summary = {
            "config_id": config_id,
            "controller": records[0]["controller"],
            "I_mean_phase_average": mean_value,
            "I_mean_ci_95": mean_ci,
            "I_mean_ci_lower": mean_ci[0],
            "I_candidate_phase_average": candidate_value,
            "I_candidate_ci_95": candidate_ci,
            "D_fixed": float(np.mean([r["five_policy"]["D_fixed"] for r in records])),
            "D_tracking": float(np.mean([r["five_policy"]["D_tracking"] for r in records])),
            "D_exploration": float(np.mean([r["five_policy"]["D_exploration"] for r in records])),
            "tracking_gain": float(np.mean([r["tracking"]["amplitude_gain"] for r in records])),
            "tracking_gain_ci_lower": float(min(r["tracking"]["confidence_intervals_95"]["amplitude_gain"][0] for r in records)),
            "phase_identifiable": all(r["tracking"]["phase_identifiable"] for r in records),
            "phase_count": len({round(r["phase_radians"], 12) for r in records}),
            "phase_I_mean_span": float(np.ptp([r["five_policy"]["I_mean"] for r in records])),
            "phase_I_candidate_span": float(np.ptp([r["five_policy"]["I_candidate"] for r in records])),
            "window_stable": all(r["window_sensitivity"]["stable"] for r in records),
            "window_sensitivity_max": float(max(r["window_sensitivity"]["maximum_absolute_change"] for r in records)),
            "clipping_fraction": float(np.mean([r["clipping_fraction"] for r in records])),
            "entropy_operational": entropy_ok,
            "held_out_protocol_frozen": bool(freeze["selection_rules_frozen"]),
            "plant_hash_unchanged": len({r["plant_hash"] for r in records}) == 1,
            "mode": mode,
            "cell_count": len(records),
        }
        summary["selection"] = controller_selection_gates(summary)
        summaries.append(summary)
    plot_root = artifact_root() / "plots"
    frequencies = sorted({r["frequency_cycles_per_epoch"] for r in cells})
    _save_plot(plot_root / "amplitude_gain_vs_frequency.png", frequencies, [np.mean([r["tracking"]["amplitude_gain"] for r in cells if np.isclose(r["frequency_cycles_per_epoch"], f)]) for f in frequencies], xlabel="frequency (cycles/epoch)", ylabel="amplitude gain", title="Tracking gain")
    _save_plot(plot_root / "phase_lag_vs_frequency.png", frequencies, [np.mean([r["tracking"]["phase_lag_epochs_relative_to_optimum"] for r in cells if np.isclose(r["frequency_cycles_per_epoch"], f)]) for f in frequencies], xlabel="frequency (cycles/epoch)", ylabel="phase lag (epochs)", title="Tracking phase lag")
    _save_plot(plot_root / "phase_specific_improvement.png", range(len(cells)), [r["five_policy"]["I_candidate"] for r in cells], xlabel="held-out cell", ylabel="I_candidate", title="Phase-specific candidate improvement")
    _save_plot(plot_root / "window_sensitivity_comparison.png", range(len(cells)), [r["window_sensitivity"]["maximum_absolute_change"] for r in cells], xlabel="held-out cell", ylabel="maximum window change", title="One-period window sensitivity")
    payload = {
        "schema_version": "google-pure-v9-held-out-validation.v1",
        "mode": mode,
        "frozen_protocol_hash": freeze["configuration_sha256"],
        "protocol": protocol.to_dict(),
        "plan": protocol.plan(controller_count=len(config["controller_candidates"]), candidates=candidates, cycles=cycles),
        "cells": cells,
        "summaries": summaries,
        "phase_averaging_required": True,
        "held_out_results_changed_selection_rules": False,
        "artifact_complete": True,
        "claim_supported": any(row["selection"]["eligible"] for row in summaries),
        "blocking_reasons": sorted({reason for row in summaries for reason in row["selection"]["blocking_reasons"]}),
    }
    return write_artifact("stage_d_held_out_validation/results", payload, "Stage D Held-out Dynamic Validation", markdown_relative="stage_d_held_out_validation/report.md")


def select_controller() -> dict[str, Any]:
    path = artifact_root() / "stage_d_held_out_validation" / "results.json"
    if not path.is_file():
        raise RuntimeError("held-out validation artifact is required before controller selection")
    held_out = read_json(path)
    eligible = [row for row in held_out["summaries"] if row["selection"]["eligible"]]
    selected = min(eligible, key=lambda row: row["D_tracking"] + row["D_exploration"]) if eligible else None
    development = load_config("development.json")
    validate_source_choices(development["source_classification"])
    payload = {
        "schema_version": "google-pure-v9-selected-controller.v1",
        "selected": selected is not None,
        "status": "SOURCE_COMPATIBLE_CONTROLLER_IDENTIFIED" if selected else "NO_SOURCE_COMPATIBLE_CONTROLLER_IDENTIFIED",
        "controller": selected["controller"] if selected else None,
        "controller_hash": canonical_hash(selected["controller"]) if selected else None,
        "selection_protocol_hash": held_out["frozen_protocol_hash"],
        "development_seed_registry": development["development_seeds"],
        "held_out_seed_registry": load_config("held_out.json")["held_out_seeds"],
        "source_classification": development["source_classification"],
        "audit_repairs": ["corrected failure labels", "independent scale fields", "phase/window gates", "clipping guard"],
        "behavior_changes": [] if selected is None else ["preregistered controller parameter selection"],
        "source_anchored_choices": ["initial_scale", "entropy_coefficient", "mean_learning_rate"],
        "source_unspecified_development_choices": ["minimum_scale", "scale_learning_rate"],
        "rejected_changes": ["plant retuning", "future drift access", "hidden logical-outcome reward"],
        "plant_hash": held_out["cells"][0]["plant_hash"] if held_out["cells"] else None,
        "full_figure5a_acquisition_permitted": selected is not None,
        "certification_seeds_consumed": False,
        "blocking_reasons": [] if selected else held_out.get("blocking_reasons", ["HELD_OUT_SELECTION_GATES_FAILED"]),
    }
    payload["contract_hash"] = canonical_hash(payload)
    return write_artifact("selected_controller_contract", payload, "Selected Controller Contract")
