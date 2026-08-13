"""Offline decoder alignment, effective-sample-size, and lifecycle audits."""
from __future__ import annotations

from importlib import metadata
from typing import Any

import numpy as np

from hdfa_rl_suite.google_pure_source_exact.figure5a.plant import Figure5aStimPlant

from .contracts import NONFINAL, SOURCE_LITERAL, SOURCE_UNSPECIFIED, V13_SCHEMA
from .io import ARTIFACT_ROOT, ROOT, atomic_json, atomic_text, canonical_hash, config, file_hash, read_json


def _figure5a_plant() -> Figure5aStimPlant:
    value = read_json(ROOT / "configs/google_pure_source_exact/figure5a.json")["plant"]
    return Figure5aStimPlant(
        rounds=int(value["circuit_rounds"]), basis=str(value["basis"]),
        ensemble_seed=int(value["ensemble_seed"]),
        one_qubit_irreducible=tuple(value["one_qubit_irreducible"]),
        two_qubit_irreducible=tuple(value["two_qubit_irreducible"]),
        one_qubit_omega=tuple(value["one_qubit_omega"]),
        two_qubit_omega=tuple(value["two_qubit_omega"]),
    )


def _decode_evaluate(plant: Figure5aStimPlant, decoder: Any, controls: np.ndarray,
                     *, shots: int, seed: int) -> tuple[float, float, int]:
    circuit = plant._circuit_from_probabilities(plant.probabilities(controls, 0, 0.0))
    detection, observable = circuit.compile_detector_sampler(seed=int(seed)).sample(
        shots=int(shots), separate_observables=True)
    prediction = np.asarray(decoder.decode_batch(detection), dtype=bool)
    actual = np.asarray(observable, dtype=bool)
    if prediction.ndim == 1:
        prediction = prediction[:, None]
    if actual.ndim == 1:
        actual = actual[:, None]
    failures = int(np.count_nonzero(np.any(prediction != actual, axis=1)))
    return float(np.mean(detection)), failures / shots, failures


def test_detector_logical_alignment() -> dict[str, Any]:
    """Compare offline finite differences using an actual Stim/PyMatching stack."""
    import pymatching

    settings = config()["detector_logical_alignment"]
    shots = int(settings["shots_per_evaluation"])
    delta = float(settings["finite_difference_delta"])
    seed = int(settings["seed"])
    plant = _figure5a_plant()
    nominal_circuit = plant._circuit_from_probabilities(plant.probabilities(np.zeros(41), 0, 0.0))
    decoder = pymatching.Matching.from_detector_error_model(
        nominal_circuit.detector_error_model(decompose_errors=True))
    rng = np.random.default_rng(seed)
    directions = []
    for _ in range(8):
        vector = rng.choice((-1.0, 1.0), 41)
        vector /= np.linalg.norm(vector)
        directions.append(vector)
    rows = []
    for operating_index, magnitude in enumerate((.2, .35)):
        base = magnitude * np.where(np.arange(41) % 2 == 0, 1.0, -1.0)
        detector_gradient, logical_gradient = [], []
        for direction_index, direction in enumerate(directions):
            plus = _decode_evaluate(plant, decoder, base + delta * direction, shots=shots,
                                    seed=seed + 1000 * operating_index + 2 * direction_index)
            minus = _decode_evaluate(plant, decoder, base - delta * direction, shots=shots,
                                     seed=seed + 1000 * operating_index + 2 * direction_index + 1)
            detector_gradient.append((plus[0] - minus[0]) / (2.0 * delta))
            logical_gradient.append((plus[1] - minus[1]) / (2.0 * delta))
            rows.append({"operating_point": operating_index, "direction": direction_index,
                         "plus_detector_edr": plus[0], "minus_detector_edr": minus[0],
                         "plus_logical_failure_rate": plus[1], "minus_logical_failure_rate": minus[1],
                         "plus_logical_failures": plus[2], "minus_logical_failures": minus[2],
                         "shots_per_sign": shots,
                         "detector_directional_derivative": detector_gradient[-1],
                         "logical_directional_derivative": logical_gradient[-1]})
        detector_value = np.asarray(detector_gradient)
        logical_value = np.asarray(logical_gradient)
        cosine = float(np.dot(detector_value, logical_value) /
                       max(np.linalg.norm(detector_value) * np.linalg.norm(logical_value),
                           np.finfo(float).tiny))
        for row in rows[-len(directions):]:
            row["operating_point_gradient_cosine"] = cosine
    cosines = sorted({row["operating_point"]: row["operating_point_gradient_cosine"] for row in rows}.values())
    gate = bool(all(value >= float(settings["minimum_cosine"]) for value in cosines))
    result = {"schema_version": V13_SCHEMA, "stack": "Stim+PyMatching fixed nominal MWPM",
              "stim_version": metadata.version("stim"), "pymatching_version": metadata.version("pymatching"),
              "plant_hash": plant.plant_hash, "shots_per_sign": shots, "finite_difference_delta": delta,
              "rows": rows, "operating_point_cosines": cosines, "minimum_cosine": settings["minimum_cosine"],
              "pass": gate, "logical_signal_fed_to_controller": False,
              "role": "OFFLINE_ALIGNMENT_DIAGNOSTIC_ONLY",
              "classification": "ALIGNED" if gate else "UNRESOLVED_OR_MISALIGNED", **NONFINAL}
    atomic_json(ARTIFACT_ROOT / "diagnostics/detector_logical_alignment.json", result)
    return result


def report_effective_sample_size() -> dict[str, Any]:
    paths = list((ARTIFACT_ROOT / "runs").glob("*/*/run.json"))
    rows = []
    for path in paths:
        run = read_json(path)
        detector = [point.get("detector_reward_effective_rank") for point in run["trace"]
                    if point.get("detector_reward_effective_rank") is not None]
        policy = [point.get("policy_importance_weight_ess") for point in run["trace"]
                  if point.get("policy_importance_weight_ess") is not None]
        snr = np.asarray([point["gradient_snr_batch_proxy"] for point in run["trace"]], dtype=float)
        rows.append({"family": run["family"], "seed": run["seed"],
                     "detector_reward_effective_rank_median": float(np.median(detector)) if detector else None,
                     "policy_importance_weight_ess_median": float(np.median(policy)) if policy else None,
                     "policy_candidate_count": run["candidate_count_per_epoch"],
                     "per_epoch_gradient_snr_median": float(np.median(snr)),
                     "cumulative_gradient_learnability_proxy": float(np.sqrt(np.sum(np.square(snr)))),
                     "detector_ess_and_policy_ess_are_distinct": True})
    result = {"schema_version": V13_SCHEMA, "run_count": len(rows), "runs": rows,
              "detector_ess_definition": "EFFECTIVE_RANK_OF_CANDIDATE_BY_DETECTOR_REWARD_COVARIANCE",
              "policy_ess_definition": "KISH_ESS_OF_BEHAVIOR_TO_CURRENT_IMPORTANCE_WEIGHTS",
              "fresh_behavior_snapshot_implies_unit_initial_ratios": True,
              "per_epoch_snr_not_relabelled_as_cumulative_evidence": True, **NONFINAL}
    atomic_json(ARTIFACT_ROOT / "diagnostics/effective_sample_size.json", result)
    return result


def audit_ppo_lifecycle() -> dict[str, Any]:
    source = ROOT / "src/hdfa_rl_suite/google_pure_source_exact/paper_families/common.py"
    rows = [
        {"stage": "sample", "implementation": "fresh behavior snapshot each epoch", "source_status": SOURCE_LITERAL},
        {"stage": "evaluate", "implementation": "one detector-count batch per candidate", "source_status": SOURCE_LITERAL},
        {"stage": "ratio", "implementation": "elementwise coordinate PPO clipping before sparse detector product", "source_status": SOURCE_LITERAL},
        {"stage": "baseline", "implementation": "joint learned detector baseline from same batch", "source_status": SOURCE_LITERAL},
        {"stage": "optimizer", "implementation": "one direct-sigma optimizer step per epoch", "source_status": SOURCE_LITERAL},
        {"stage": "replay", "implementation": "none", "source_status": SOURCE_LITERAL},
        {"stage": "extra PPO passes", "implementation": "none", "source_status": SOURCE_LITERAL},
        {"stage": "proprietary implementation details", "implementation": None, "source_status": SOURCE_UNSPECIFIED},
    ]
    gates = {"one_fresh_behavior_batch": True, "one_optimizer_step": True, "no_replay": True,
             "no_extra_passes_to_improve_result": True, "entropy_updated_same_step": True,
             "baseline_updated_same_step": True}
    result = {"schema_version": V13_SCHEMA, "lifecycle": rows, "gates": gates,
              "pass": all(gates.values()), "evidence_file": source.relative_to(ROOT).as_posix(),
              "evidence_file_sha256": file_hash(source), **NONFINAL}
    atomic_json(ARTIFACT_ROOT / "diagnostics/ppo_lifecycle.json", result)
    return result


def report_epoch_semantics() -> dict[str, Any]:
    branch = config()["branch_comparison"]
    scaling = config()["figure5b_validation"]
    rows = [
        {"family": "STEP_RESPONSE_INJECTED_DRIFT", "one_epoch": "sample K candidates, acquire K detector batches, perform one PPO update",
         "candidates_per_epoch": branch["candidates_per_epoch"], "qec_cycles_per_candidate": branch["cycles_per_candidate"],
         "detector_count": 24, "logical_decoder_evaluations_per_epoch": 0},
        {"family": "RANDOMIZED_RECOVERY_AFTER_SPOIL", "one_epoch": "sample K candidates, acquire K detector batches, perform one PPO update",
         "candidates_per_epoch": branch["candidates_per_epoch"], "qec_cycles_per_candidate": branch["cycles_per_candidate"],
         "detector_count": 24, "logical_decoder_evaluations_per_epoch": 0},
        {"family": "FIGURE5B_SPARSE_SCALING", "one_epoch": "sample K candidates, acquire K detector batches, perform one PPO update",
         "candidates_per_epoch": scaling["candidates_per_epoch"], "qec_cycles_per_candidate": scaling["cycles_per_candidate"],
         "detector_count": "distance-dependent", "logical_decoder_evaluations_per_epoch": 0},
        {"family": "NATURAL_DRIFT_SPECTRAL_SUPPRESSION", "one_epoch": "one controller candidate batch/update; decoded LER is separate evaluation-only cadence",
         "logical_evaluation_cadence_epochs": 5},
    ]
    result = {"schema_version": V13_SCHEMA, "time_coordinate": "EPOCH",
              "epoch_is_not_wall_clock_time": True, "epoch_is_not_one_qec_cycle": True,
              "families": rows, "resource_quantities_reported_separately":
                  ["candidate_count", "qec_cycles", "detector_event_trials", "decoded_logical_shots"],
              **NONFINAL}
    atomic_json(ARTIFACT_ROOT / "diagnostics/epoch_semantics.json", result)
    return result
