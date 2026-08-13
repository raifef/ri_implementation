"""Physical and structural preflight for the source-structured Stim plant."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import canonical_hash, file_sha256
from .bounded_action_ablation import Figure5aBoundedActionAblation
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
        two_qubit_omega=tuple(value["two_qubit_omega"]))


def dependency_hashes(root: Path, config: Mapping[str, Any]) -> dict[str, str]:
    return {name: file_sha256(root / relative) for name, relative in config["dependencies"].items()}


def validate_dependencies(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    direct_path = root / config["dependencies"]["direct_sigma_status"]
    direct = json.loads(direct_path.read_text(encoding="utf-8"))
    plant = build_plant(config)
    gates = {
        "direct_sigma_math_pass": bool(direct["mathematical_contract_pass"]),
        "direct_sigma_structure_match": bool(direct["source_structure_match"]),
        "direct_sigma_parameterization": direct["parameterization"] == "DIRECT_SIGMA_SOURCE_EXACT",
        "canonical_policy_coordinate_is_applied_p":
            config["controller"]["action_execution"]["policy_space"] ==
            "applied_dimensionless_gate_controls_p",
        "canonical_action_transform_is_identity":
            config["controller"]["action_execution"]["transform"] == "identity",
        "canonical_empirical_normalization_disabled":
            config["controller"]["action_execution"]["empirical_relative_normalization"] is False,
    }
    ablation_path = root / config["ablations"]["empirical_relative_normalization_bundle"]
    ablation: dict[str, Any] = {"pass": False, "path": str(ablation_path), "blocking_reasons": []}
    try:
        normalization = json.loads(ablation_path.read_text(encoding="utf-8"))
        boundary = Figure5aEmpiricalBoundary.from_artifact(plant, normalization)
        bounded = Figure5aBoundedActionAblation(
            plant, maximum_probability=float(
                config["ablations"]["bounded_action"]["maximum_probability"]),
            action_probability_margin_fraction=float(
                config["ablations"]["bounded_action"]["action_probability_margin_fraction"]))
        ablation_gates = {
            "bound_to_exact_plant": normalization["plant_hash"] == plant.plant_hash,
            "bound_to_reduced_reward":
                normalization["reward_representation_hash"] == reward_representation_hash(plant),
            "empirical_math_pass": bool(normalization["mathematical_contract_pass"]),
            "no_percentage_point_conversion": not normalization["percentage_point_conversion_applied"],
            "no_analytic_degree_shortcut":
                not normalization["analytic_omega_times_degree_shortcut_used"],
            "absolute_scale_nonidentifiability_preserved":
                normalization["absolute_source_scale_identifiable"] is False,
            "noncanonical_status_explicit":
                config["ablations"]["empirical_relative_normalization_status"] ==
                "NONCANONICAL_CONDITIONING_ABLATION",
            "bounded_ablation_domain_safe": bool(
                np.all(bounded.normalized_control_limits(boundary.native_scale) > 1.0)),
        }
        ablation = {"pass": all(ablation_gates.values()), "path": str(ablation_path),
                    "gates": ablation_gates, "calibration_hash": boundary.calibration_hash,
                    "blocking_reasons": [name for name, value in ablation_gates.items() if not value]}
    except (OSError, KeyError, RuntimeError, ValueError) as exc:
        ablation["blocking_reasons"] = [str(exc)]
    return {"pass": all(gates.values()), "gates": gates, "hashes": dependency_hashes(root, config),
            "figure5a_coordinate_note": "Canonical Figure 5a applies Gaussian p directly; empirical relative normalization is a separate noncanonical ablation.",
            "coordinate_registry_alignment": "SOURCE_P_EQUALS_APPLIED_PLANT_P",
            "empirical_relative_normalization_ablation": ablation}


def detector_equivalence_response_audit(
    plant: Figure5aStimPlant, *, seed: int = 53077, random_policies: int = 5,
) -> dict[str, Any]:
    """Verify every declared class has equal exact raw-detector marginals."""
    if random_policies < 3:
        raise ValueError("at least three random policies are required")
    rng = np.random.default_rng(seed)
    policies = rng.normal(0.0, 0.3, size=(random_policies, plant.control_count))
    maximum_spread = 0.0
    failures: list[dict[str, Any]] = []
    for policy_index, policy in enumerate(policies):
        marginals = plant.raw_detector_marginals(
            policy, epoch=0, frequency=1 / 1000,
            target_controls=np.zeros(plant.control_count))
        for component, group in enumerate(plant.reward_component_raw_detectors):
            values = marginals[list(group)]
            spread = float(np.ptp(values))
            maximum_spread = max(maximum_spread, spread)
            if not np.allclose(values, values[0], rtol=1e-12, atol=2e-15):
                failures.append({"policy": policy_index, "component": component,
                                 "raw_detectors": list(group), "spread": spread})
    return {"pass": not failures, "random_policy_count": random_policies,
            "multi_detector_class_count": sum(
                len(group) > 1 for group in plant.reward_component_raw_detectors),
            "maximum_within_class_marginal_spread": maximum_spread,
            "failures": failures}


def physical_preflight(root: Path, config: Mapping[str, Any], *, finite_shot_cycles: int = 900_000) -> dict[str, Any]:
    plant = build_plant(config)
    dependencies = validate_dependencies(root, config)
    equivalence = detector_equivalence_response_audit(plant)
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
        "detector_class_exact_marginal_equivalence": equivalence["pass"],
        "detector_class_reduction_nonvacuous": equivalence["multi_detector_class_count"] > 0,
        "physical_probabilities": bool(np.all(plant.probabilities(np.full(41, 2.0), 0, 1 / 1000)
                                               < plant.probability_ceilings)),
        "canonical_action_is_identity":
            config["controller"].get("action_execution", {}).get("transform") ==
            "identity",
        "canonical_policy_and_plant_coordinates_identical":
            config["controller"].get("action_execution", {}).get("policy_space") ==
            config["controller"].get("action_execution", {}).get("plant_space"),
        "canonical_entropy_is_applied_gaussian_entropy":
            config["controller"].get("action_execution", {}).get("entropy_space") ==
            "applied_gaussian",
        "canonical_mean_is_unbounded":
            config["controller"].get("action_execution", {}).get("mean_bounds") is None,
        "empirical_normalization_excluded_from_canonical_path":
            config["controller"].get("action_execution", {}).get(
                "empirical_relative_normalization") is False,
        "identity_action_independent_of_hidden_optimum":
            config["controller"].get("action_execution", {}).get("uses_hidden_optimum") is False,
        "published_peak_optimum_is_literal_one_vector":
            np.array_equal(oracle, np.ones(plant.control_count)),
        "oracle_better_than_fixed_finite_shot": oracle_observation.raw_total < fixed_observation.raw_total,
        "single_gate_measured_curvature_positive": single_full.raw_total > single_baseline.raw_total,
        "dependencies_valid": dependencies["pass"],
    }
    gates = {name: bool(value) for name, value in gates.items()}
    result = {"schema_version": "figure5a-physical-preflight.v1", "pass": all(gates.values()),
              "gates": gates, "plant_hash": plant.plant_hash, "control_count": plant.control_count,
              "controller_hash": canonical_hash(config["controller"]),
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
              "detector_equivalence_response_audit": equivalence,
              "action_execution": "identity_applied_gaussian",
              "coordinate_contract": "SOURCE_GAUSSIAN_P_EQUALS_APPLIED_PLANT_P_V1",
              "action_physicality_contract":
                  "validate each actually encountered applied action against its Stim channel ceiling",
              "blocking_reasons": [name for name, value in gates.items() if not value]}
    return result
