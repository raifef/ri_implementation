"""Physical and structural preflight for the source-structured Stim plant."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import canonical_hash, file_sha256
from .plant import Figure5aStimPlant


def build_plant(config: Mapping[str, Any]) -> Figure5aStimPlant:
    value = config["plant"]
    if int(value["distance"]) != 3:
        raise ValueError("Figure 5a paper mode requires distance 3")
    return Figure5aStimPlant(
        rounds=int(value["circuit_rounds"]), basis=str(value["basis"]),
        ensemble_seed=int(value["ensemble_seed"]),
        one_qubit_irreducible=tuple(value["one_qubit_irreducible"]),
        two_qubit_irreducible=tuple(value["two_qubit_irreducible"]),
        one_qubit_omega=tuple(value["one_qubit_omega"]),
        two_qubit_omega=tuple(value["two_qubit_omega"]),
        maximum_probability=float(value["maximum_probability"]),
        action_probability_margin_fraction=float(value["action_probability_margin_fraction"]))


def dependency_hashes(root: Path, config: Mapping[str, Any]) -> dict[str, str]:
    return {name: file_sha256(root / relative) for name, relative in config["dependencies"].items()}


def validate_dependencies(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    normalization_path = root / config["dependencies"]["normalization_bundle"]
    direct_path = root / config["dependencies"]["direct_sigma_status"]
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    direct = json.loads(direct_path.read_text(encoding="utf-8"))
    gates = {
        "empirical_normalization_artifact_complete": bool(normalization["artifact_complete"]),
        "empirical_normalization_math_pass": bool(normalization["mathematical_contract_pass"]),
        "direct_sigma_math_pass": bool(direct["mathematical_contract_pass"]),
        "direct_sigma_structure_match": bool(direct["source_structure_match"]),
        "direct_sigma_parameterization": direct["parameterization"] == "DIRECT_SIGMA_SOURCE_EXACT",
    }
    return {"pass": all(gates.values()), "gates": gates, "hashes": dependency_hashes(root, config),
            "normalization_parameter_count": sum(len(item["gate_ids"]) for item in normalization["control_specs"]),
            "figure5a_coordinate_note": "Figure 5a source p_i coordinates are dimensionless; the prior empirical bundle is imported and hashed but its 36 hardware-control registry is not relabelled as the 41 synthetic gate registry.",
            "coordinate_registry_alignment": "SOURCE_UNSPECIFIED_PREREGISTERED"}


def physical_preflight(root: Path, config: Mapping[str, Any], *, finite_shot_cycles: int = 900_000) -> dict[str, Any]:
    plant = build_plant(config)
    dependencies = validate_dependencies(root, config)
    inventory = plant.inventory
    one_count = sum(item.gate_type == "single_qubit" for item in inventory)
    two_count = sum(item.gate_type == "two_qubit" for item in inventory)
    optimum_zero = plant.optimum(0, 1 / 1000)
    baseline_probability = plant.probabilities(np.zeros(41), 0, 1 / 1000)
    oracle_probability = plant.probabilities(optimum_zero, 0, 1 / 1000)
    coordinate = 0
    omega = inventory[coordinate].omega_sensitivity
    half = np.zeros(41); half[coordinate] = 0.5
    full = np.zeros(41); full[coordinate] = 1.0
    half_excess = plant.probabilities(half, 0, 1 / 1000)[coordinate] - baseline_probability[coordinate]
    full_excess = plant.probabilities(full, 0, 1 / 1000)[coordinate] - baseline_probability[coordinate]
    epoch = 250
    oracle = plant.optimum(epoch, 1 / 1000)
    fixed_counts = plant.sample_detector_counts(np.zeros(41), epoch=epoch, frequency=1 / 1000,
                                                qec_cycles=finite_shot_cycles,
                                                seed=plant.stream_seed(53091, "preflight-fixed", epoch, 0))
    oracle_counts = plant.sample_detector_counts(oracle, epoch=epoch, frequency=1 / 1000,
                                                 qec_cycles=finite_shot_cycles,
                                                 seed=plant.stream_seed(53091, "preflight-oracle", epoch, 0))
    single_gate_seed = plant.stream_seed(53092, "single-gate-crn", 0, 0)
    single_baseline = plant.sample_detector_counts(np.zeros(41), epoch=0, frequency=1 / 1000,
                                                   qec_cycles=finite_shot_cycles,
                                                   seed=single_gate_seed)
    single_full = plant.sample_detector_counts(full, epoch=0, frequency=1 / 1000,
                                               qec_cycles=finite_shot_cycles,
                                               seed=single_gate_seed)
    gates = {
        "exactly_17_one_qubit_controls": one_count == 17,
        "exactly_24_two_qubit_controls": two_count == 24,
        "exactly_41_controls": plant.control_count == 41,
        "zero_miscalibration_at_optimum": np.array_equal(baseline_probability, oracle_probability),
        "quadratic_single_coordinate": np.isclose(half_excess, 0.25 * omega) and np.isclose(full_excess, omega),
        "fixed_optimal_at_epoch_zero": np.array_equal(optimum_zero, np.zeros(41)),
        "shared_phase_and_frequency": np.unique(plant.optimum(137, 1 / 1000)).size == 1,
        "stim_derived_sparse_mask": bool(0 < plant.mask.mean() < 1 and plant.mask.any(axis=0).all()),
        "physical_probabilities": bool(np.all(plant.probabilities(np.full(41, 2.0), 0, 1 / 1000)
                                               < float(config["plant"]["maximum_probability"]))),
        "action_transform_matches_frozen_controller_contract":
            config["controller"].get("action_execution", {}).get("transform") ==
            "plant_derived_per_coordinate_scaled_tanh",
        "action_transform_independent_of_hidden_optimum":
            config["controller"].get("action_execution", {}).get("uses_hidden_optimum") is False,
        "bounded_action_domain_contains_full_optimum_range":
            bool(np.all(plant.control_limits > 1.0)),
        "bounded_action_domain_safe_at_both_drift_extremes": bool(
            np.all(plant.probabilities(plant.apply_control_transform(np.full(41, 1e6)),
                                       750, 1 / 1000) < plant.maximum_probability) and
            np.all(plant.probabilities(plant.apply_control_transform(np.full(41, -1e6)),
                                       250, 1 / 1000) < plant.maximum_probability)),
        "oracle_better_than_fixed_finite_shot": int(oracle_counts.sum()) < int(fixed_counts.sum()),
        "single_gate_measured_curvature_positive": int(single_full.sum()) > int(single_baseline.sum()),
        "dependencies_valid": dependencies["pass"],
    }
    gates = {name: bool(value) for name, value in gates.items()}
    result = {"schema_version": "figure5a-physical-preflight.v1", "pass": all(gates.values()),
              "gates": gates, "plant_hash": plant.plant_hash, "control_count": plant.control_count,
              "detector_count": plant.detector_count, "mask_density": float(plant.mask.mean()),
              "finite_shot_cycles_per_policy": finite_shot_cycles,
              "finite_shot_counts": {"fixed": int(fixed_counts.sum()), "oracle": int(oracle_counts.sum()),
                                     "single_gate_baseline": int(single_baseline.sum()),
                                     "single_gate_miscalibrated": int(single_full.sum())},
              "single_gate_expected_curvature": omega,
              "single_gate_measured_total_event_excess": int(single_full.sum() - single_baseline.sum()),
              "dependencies": dependencies, "inventory_hash": canonical_hash(plant.inventory_rows()),
              "mask_hash": canonical_hash(plant.mask.astype(int).tolist()),
              "action_execution": "plant_derived_per_coordinate_scaled_tanh",
              "control_limits": plant.control_limits.tolist(),
              "blocking_reasons": [name for name, value in gates.items() if not value]}
    return result
