"""Physical and structural preflight for the source-structured Stim plant."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import canonical_hash, file_sha256
from .plant import Figure5aStimPlant
from .normalization import Figure5aEmpiricalBoundary, reward_representation_hash


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
    plant = build_plant(config)
    boundary = Figure5aEmpiricalBoundary.from_artifact(plant, normalization)
    gates = {
        "figure5a_normalization_bound_to_exact_plant": normalization["plant_hash"] == plant.plant_hash,
        "normalization_bound_to_reduced_reward":
            normalization["reward_representation_hash"] == reward_representation_hash(plant),
        "empirical_normalization_math_pass": bool(normalization["mathematical_contract_pass"]),
        "literal_fractional_edr_convention": normalization["edr_unit"] == "fraction"
            and normalization["source_literal_target_edr_increase_fraction"] == 1.0
            and not normalization["percentage_point_conversion_applied"],
        "no_analytic_degree_shortcut": not normalization["analytic_omega_times_degree_shortcut_used"],
        "absolute_scale_nonidentifiability_preserved":
            normalization["absolute_source_scale_identifiable"] is False,
        "relative_curvature_equalization_applied":
            normalization["applied_scale_kind"] == "relative_empirical_curvature_equalization",
        "safe_normalized_action_domain": bool(
            np.all(plant.normalized_control_limits(boundary.native_scale) > 1.0)),
        "direct_sigma_math_pass": bool(direct["mathematical_contract_pass"]),
        "direct_sigma_structure_match": bool(direct["source_structure_match"]),
        "direct_sigma_parameterization": direct["parameterization"] == "DIRECT_SIGMA_SOURCE_EXACT",
    }
    return {"pass": all(gates.values()), "gates": gates, "hashes": dependency_hashes(root, config),
            "normalization_parameter_count": len(normalization["native_scale"]),
            "figure5a_coordinate_note": "Figure 5a uses a plant-bound Figure-S3 sweep over the 41 synthetic gate controls; the unpublished grouping is explicit.",
            "coordinate_registry_alignment": "EXACT_41_CONTROL_PLANT_HASH_BOUND"}


def physical_preflight(root: Path, config: Mapping[str, Any], *, finite_shot_cycles: int = 900_000) -> dict[str, Any]:
    plant = build_plant(config)
    dependencies = validate_dependencies(root, config)
    normalization = json.loads((root / config["dependencies"]["normalization_bundle"]).read_text(
        encoding="utf-8"))
    boundary = Figure5aEmpiricalBoundary.from_artifact(plant, normalization)
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
    fixed_observation = plant.sample_detector_observation(
        np.zeros(41), epoch=epoch, frequency=1 / 1000, qec_cycles=finite_shot_cycles,
        seed=plant.stream_seed(53091, "preflight-fixed", epoch, 0))
    oracle_observation = plant.sample_detector_observation(
        oracle, epoch=epoch, frequency=1 / 1000, qec_cycles=finite_shot_cycles,
        seed=plant.stream_seed(53091, "preflight-oracle", epoch, 0))
    single_gate_seed = plant.stream_seed(53092, "single-gate-crn", 0, 0)
    single_baseline = plant.sample_detector_observation(
        np.zeros(41), epoch=0, frequency=1 / 1000,
        qec_cycles=finite_shot_cycles, seed=single_gate_seed)
    single_full = plant.sample_detector_observation(
        full, epoch=0, frequency=1 / 1000,
        qec_cycles=finite_shot_cycles, seed=single_gate_seed)
    gates = {
        "exactly_17_one_qubit_controls": one_count == 17,
        "exactly_24_two_qubit_controls": two_count == 24,
        "exactly_41_controls": plant.control_count == 41,
        "zero_miscalibration_at_optimum": np.array_equal(baseline_probability, oracle_probability),
        "quadratic_single_coordinate": np.isclose(half_excess, 0.25 * omega) and np.isclose(full_excess, omega),
        "fixed_optimal_at_epoch_zero": np.array_equal(optimum_zero, np.zeros(41)),
        "shared_phase_and_frequency": np.unique(plant.optimum(137, 1 / 1000)).size == 1,
        "stim_derived_sparse_mask": bool(0 < plant.mask.mean() < 1 and plant.mask.any(axis=0).all()),
        "time_translation_reward_reduction": plant.detector_count < plant.raw_detector_count,
        "reduced_reward_partition_is_exact": sum(
            len(group) for group in plant.reward_component_raw_detectors) == plant.raw_detector_count,
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
            np.all(plant.probabilities(boundary.apply(plant.apply_control_transform(
                np.full(41, 1e6), native_scale=boundary.native_scale)).native,
                750, 1 / 1000, target_controls=boundary.target_to_native(
                    plant.optimum(750, 1 / 1000))) < plant.maximum_probability) and
            np.all(plant.probabilities(boundary.apply(plant.apply_control_transform(
                np.full(41, -1e6), native_scale=boundary.native_scale)).native,
                250, 1 / 1000, target_controls=boundary.target_to_native(
                    plant.optimum(250, 1 / 1000))) < plant.maximum_probability)),
        "oracle_better_than_fixed_finite_shot": oracle_observation.raw_total < fixed_observation.raw_total,
        "single_gate_measured_curvature_positive": single_full.raw_total > single_baseline.raw_total,
        "dependencies_valid": dependencies["pass"],
    }
    gates = {name: bool(value) for name, value in gates.items()}
    result = {"schema_version": "figure5a-physical-preflight.v1", "pass": all(gates.values()),
              "gates": gates, "plant_hash": plant.plant_hash, "control_count": plant.control_count,
              "detector_count": plant.detector_count, "raw_detector_count": plant.raw_detector_count,
              "mask_density": float(plant.mask.mean()),
              "finite_shot_cycles_per_policy": finite_shot_cycles,
              "finite_shot_counts": {"fixed": fixed_observation.raw_total,
                                     "oracle": oracle_observation.raw_total,
                                     "single_gate_baseline": single_baseline.raw_total,
                                     "single_gate_miscalibrated": single_full.raw_total},
              "single_gate_expected_curvature": omega,
              "single_gate_measured_total_event_excess":
                  int(single_full.raw_total - single_baseline.raw_total),
              "dependencies": dependencies, "inventory_hash": canonical_hash(plant.inventory_rows()),
              "mask_hash": canonical_hash(plant.mask.astype(int).tolist()),
              "action_execution": "plant_derived_per_coordinate_scaled_tanh",
              "normalized_control_limits":
                  plant.normalized_control_limits(boundary.native_scale).tolist(),
              "native_scale": boundary.native_scale.tolist(),
              "blocking_reasons": [name for name, value in gates.items() if not value]}
    return result
