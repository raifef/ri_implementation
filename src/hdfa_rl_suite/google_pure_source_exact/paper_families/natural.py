"""Paired decoded-LER natural-drift acquisition using the direct-sigma controller."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from hdfa_rl_suite.google_pure_source_exact.natural_drift_dft.contracts import (
    ALLOWED_STREAM,
    EvaluationTrace,
    SourceDFTConfig,
)
from hdfa_rl_suite.google_pure_source_exact.natural_drift_dft.estimator import (
    preprocess_trace,
    run_spectrum,
)
from hdfa_rl_suite.google_pure_v7.config import repository_root
from hdfa_rl_suite.logical.surface_code import RotatedSurfaceCodeEvaluator, SurfaceCodeMemoryConfig

from .common import SparseControlPlant, canonical_hash, run_direct_sigma_trace


def _natural_target(index: int, horizon: int, controls: int) -> np.ndarray:
    rng = np.random.default_rng(9800 + index)
    t = np.arange(horizon, dtype=float)
    if index == 0:
        scalar = .65 * np.sin(2*np.pi*t/510 + .3) + .35 * np.sin(2*np.pi*t/290 + 1.7)
    elif index == 1:
        scalar = .72 * np.sin(2*np.pi*t/430 + .8) + .28 * np.cos(2*np.pi*t/210 + .2)
    elif index in (2, 4):
        frequencies = rng.uniform(1/650, 1/120, 5)
        phases = rng.uniform(0, 2*np.pi, 5)
        weights = rng.uniform(.4, 1.0, 5)
        scalar = sum(w*np.sin(2*np.pi*f*t+p) for w, f, p in zip(weights, frequencies, phases))
        scalar /= max(float(np.max(np.abs(scalar))), 1e-12)
    else:
        frequencies = np.fft.rfftfreq(horizon)
        spectrum = np.zeros(len(frequencies), dtype=complex)
        active = (frequencies > 0) & (frequencies <= .02)
        spectrum[active] = (rng.normal(size=active.sum()) + 1j*rng.normal(size=active.sum())) / frequencies[active]
        scalar = np.fft.irfft(spectrum, n=horizon)
        scalar /= max(float(np.max(np.abs(scalar))), 1e-12)
    amplitude = (.20, .18, .20, .17, .16, .15)[index]
    stride = (3, 1, 2, 2, 4, 3)[index]
    target = np.zeros((horizon, controls))
    coordinates = np.arange(0, controls, stride)
    signs = np.where(np.arange(len(coordinates)) % 2 == 0, 1.0, -1.0)
    target[:, coordinates] = amplitude * scalar[:, None] * signs[None, :]
    return np.clip(target, -.5, .5)


def _checkpoint(protocol: Mapping[str, Any], condition: Mapping[str, Any]) -> Path:
    identity = canonical_hash({"protocol_hash": protocol["protocol_hash"], "condition": dict(condition)})[:24]
    return repository_root() / "artifacts/google_pure_source_exact/paper_families/checkpoints/natural" / f"{identity}.json"


def acquire_natural_condition(protocol: Mapping[str, Any],
                              condition: Mapping[str, Any]) -> dict[str, Any]:
    config = protocol["config"]
    index, seed = int(condition["plant_index"]), int(condition["seed"])
    horizon = int(config["epochs"])
    controls = int(config["controls"])
    plant = SparseControlPlant(5, controls, 24, seed=9900 + index, curvature=.004)
    tape = _natural_target(index, horizon, controls)
    cadence = int(config["evaluation_cadence_epochs"])
    shots = int(config["logical_evaluation_shots"])
    evaluator = RotatedSurfaceCodeEvaluator(SurfaceCodeMemoryConfig(
        distance=5, rounds=5, shots=shots, basis="z", decoder_noise_probability=.002))

    def evaluate(epoch: int, policy, target: np.ndarray) -> Mapping[str, Any]:
        if epoch % cadence:
            return {"performed": False}
        learned_error = {f"c{i}": float(value) for i, value in enumerate(policy.mean - target)}
        fixed_error = {f"c{i}": float(-value) for i, value in enumerate(target)}
        learned = evaluator.evaluate(learned_error, seed=seed*10_000 + epoch*2 + 1,
                                     policy_hash=f"learned-{policy.policy_version}",
                                     disturbance_state_id=f"natural-{index}-{epoch}")
        fixed = evaluator.evaluate(fixed_error, seed=seed*10_000 + epoch*2 + 2,
                                   policy_hash="fixed-initial",
                                   disturbance_state_id=f"natural-{index}-{epoch}")
        return {"performed": True, "epoch": epoch,
                "learned_mean_ler": (learned.logical_failures + .5) / (shots + 1.0),
                "fixed_initial_ler": (fixed.logical_failures + .5) / (shots + 1.0),
                "learned_raw_failures": learned.logical_failures,
                "fixed_raw_failures": fixed.logical_failures,
                "shots_per_policy": shots, "decoder": learned.decoder,
                "stack_id": learned.stack_id, "stim_version": learned.stim_version,
                "pymatching_version": learned.pymatching_version,
                "finite_shot_positive_correction": "JEFFREYS_HALF_COUNT"}

    result = run_direct_sigma_trace(
        plant=plant, protocol_hash=str(protocol["protocol_hash"]), seed=seed,
        epochs=horizon, candidates=int(config["candidates"]),
        cycles_per_candidate=int(config["cycles_per_candidate"]),
        entropy_weight=float(config["entropy_coefficient"]),
        checkpoint=_checkpoint(protocol, condition), target_at_epoch=lambda epoch: tape[epoch],
        evaluation=evaluate, experiment_family="NATURAL_DRIFT_SPECTRAL_SUPPRESSION",
        fresh_acquisition_required=bool(protocol.get("fresh_acquisition_required", False)),
        source_budget_profile=str(protocol.get("source_budget_profile", protocol["mode"])))
    evaluations = [row["evaluation"] for row in result["records"]
                   if row.get("evaluation", {}).get("performed")]
    trace = EvaluationTrace(
        run_id=f"natural-direct-sigma-{index}-{seed}",
        epochs=tuple(row["epoch"] for row in evaluations),
        learned_mean_ler=tuple(row["learned_mean_ler"] for row in evaluations),
        fixed_initial_ler=tuple(row["fixed_initial_ler"] for row in evaluations),
        stream_kind=ALLOWED_STREAM)
    estimator_config = SourceDFTConfig(
        cadence_epochs=cadence, warmup_epoch=int(config["warmup_epoch"]),
        shared_grid_points=int(config["shared_grid_points"]),
        gaussian_smoothing_sigma_bins=float(config["gaussian_smoothing_sigma_bins"]))
    prepared = preprocess_trace(trace, estimator_config)
    frequency, learned_power = run_spectrum(prepared["learned"], cadence_epochs=cadence)
    _, fixed_power = run_spectrum(prepared["fixed"], cadence_epochs=cadence)
    per_run_filter = 10*np.log10(np.maximum(learned_power, np.finfo(float).tiny) /
                                 np.maximum(fixed_power, np.finfo(float).tiny))
    return {
        "plant_id": f"source-structured-{index}", "plant_index": index, "seed": seed,
        "raw_trace_hash": canonical_hash(tape.tolist()), "plant_instance_hash": result["plant_hash"],
        "graph_instance_hash": result["graph_hash"], "controller_mode": "PAPER_DIRECT_SIGMA",
        "parameterization": "direct_sigma", "ratio_clipping_mode": "SOURCE_ELEMENTWISE_COORDINATE_CLIPPING",
        "baseline_mode": "JOINT_LEARNED_DETECTOR_BASELINE", "candidate_qec_cycles": result["candidate_qec_cycles"],
        "stream_kind": ALLOWED_STREAM, "source_dft_estimator": True,
        "warmup_epoch_excluded": True, "source_epoch_150_normalization": True,
        "spectral_aggregation": "GEOMETRIC_MEAN", "power_db_convention": "10*log10(learned/fixed)",
        "low_frequency_suppression_db_fixed_over_mean": float(-np.median(per_run_filter[:max(1, len(per_run_filter)//4)])),
        "development_classification": "UNDERPOWERED_DEVELOPMENT_VALIDATION",
        "v15_provenance": {key: result[key] for key in (
            "implementation_version", "sensitivity_map_hash", "sensitivity_definition_hash",
            "calibration_bundle_hash", "detector_degree_audit_hash", "boundary_transform_hash",
            "boundary_transform_name", "boundary_apply_count", "control_order_hash",
            "expanded_scale_hash", "fresh_acquisition", "reused_shard_ids", "source_budget_profile")},
        "boundary_trace": result["boundary_trace"],
        "trace": trace.to_dict(), "per_run_spectrum": {"frequency_per_epoch": frequency.tolist(),
            "learned_power": learned_power.tolist(), "fixed_power": fixed_power.tolist(),
            "filter_db": per_run_filter.tolist()},
        "trajectory": {"learned_mean": [row["learned"]["logical_error"] for row in result["records"]],
                       "fixed_policy": [row["fixed"]["logical_error"] for row in result["records"]]},
        "source_structure_match": True, "paper_comparable": False,
        "blocking_reasons": ["the experimental learned-mean/fixed LER traces are not in the public release",
                             "the proprietary natural-drift hardware plant is unavailable"],
    }
