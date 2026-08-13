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
from .plant_calibration import CALIBRATION_ALGORITHM_VERSION, calibration_input


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
        irreducible_global_scale=float(value.get("irreducible_global_scale", 1.0)),
        one_qubit_omega_global_scale=float(
            value.get("one_qubit_omega_global_scale", 1.0)),
        two_qubit_omega_global_scale=float(
            value.get("two_qubit_omega_global_scale", 1.0)),
        omega_coordinate_scales=tuple(value.get("omega_coordinate_scales", [1.0] * 41)),
        one_qubit_injection_mapping=str(value.get(
            "one_qubit_injection_mapping", "per_qubit_operation_aggregate")))


def dependency_hashes(root: Path, config: Mapping[str, Any]) -> dict[str, str]:
    return {name: file_sha256(root / relative) for name, relative in config["dependencies"].items()}


def validate_dependencies(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    direct_path = root / config["dependencies"]["direct_sigma_status"]
    direct = json.loads(direct_path.read_text(encoding="utf-8"))
    plant = build_plant(config)
    calibration = json.loads(
        (root / config["dependencies"]["plant_calibration"]).read_text(encoding="utf-8"))
    marginal_audit = json.loads(
        (root / config["dependencies"]["marginal_exactness"]).read_text(encoding="utf-8"))
    mapping = str(config["plant"]["one_qubit_injection_mapping"])
    curvature_family = str(config["plant"]["curvature_conditioning_family"])
    calibrated_variant = calibration.get("variants", {}).get(mapping, {})
    calibrated_coordinate_scales = np.asarray(
        calibrated_variant.get("omega_coordinate_scales", []), dtype=float)
    configured_coordinate_scales = np.asarray(
        config["plant"]["omega_coordinate_scales"], dtype=float)
    calibrated_curvatures = np.asarray(
        calibrated_variant.get("coordinate_curvatures", []), dtype=float)
    fixed_to_oracle_gap = float(
        calibrated_variant.get("fixed_peak_edr", np.nan)
        - calibrated_variant.get("oracle_edr", np.nan))
    initial_sigma_gap_fraction = (
        float(np.sum(calibrated_curvatures) * config["controller"]["initial_sigma"]**2 /
              fixed_to_oracle_gap)
        if calibrated_curvatures.shape == (41,) and fixed_to_oracle_gap > 0
        else np.nan)
    gates = {
        "direct_sigma_math_pass": bool(direct["mathematical_contract_pass"]),
        "direct_sigma_structure_match": bool(direct["source_structure_match"]),
        "direct_sigma_parameterization": direct["parameterization"] == "DIRECT_SIGMA_SOURCE_EXACT",
        "marginal_exactness_audit_pass": bool(marginal_audit.get("pass")),
        "marginal_exactness_audit_current": bool(
            marginal_audit.get("schema_version") ==
            "figure5a-marginal-exactness-audit.v1"
            and marginal_audit.get("plant_hash") == plant.plant_hash
            and marginal_audit.get("fault_signature_hash") == plant.fault_signature_hash
            and marginal_audit.get("exact_marginal_evaluator_version") ==
            plant.EXACT_MARGINAL_EVALUATOR_VERSION),
        "canonical_policy_coordinate_is_applied_p":
            config["controller"]["action_execution"]["policy_space"] ==
            "applied_dimensionless_gate_controls_p",
        "canonical_action_transform_is_identity":
            config["controller"]["action_execution"]["transform"] == "identity",
        "canonical_empirical_normalization_disabled":
            config["controller"]["action_execution"]["empirical_relative_normalization"] is False,
        "plant_calibration_math_pass": bool(calibration.get("mathematical_contract_pass")),
        "plant_calibration_algorithm_current":
            calibration.get("calibration_algorithm_version") == CALIBRATION_ALGORITHM_VERSION,
        "plant_calibration_input_current": calibration.get("calibration_input_hash") ==
            canonical_hash(calibration_input(config)),
        "selected_curvature_family_current":
            calibration.get("calibration_input", {}).get(
                "selected_curvature_conditioning_family") == curvature_family
            and calibrated_variant.get("curvature_conditioning_family") == curvature_family,
        "selected_plant_mapping_calibrated": bool(calibrated_variant.get("pass")),
        "selected_calibrated_plant_hash_current":
            calibrated_variant.get("plant_hash") == plant.plant_hash,
        "selected_exact_marginal_evaluator_current":
            calibrated_variant.get("exact_marginal_evaluator_version") ==
            plant.EXACT_MARGINAL_EVALUATOR_VERSION
            and calibrated_variant.get("fault_signature_hash") ==
            plant.fault_signature_hash,
        "selected_irreducible_scale_current": np.isclose(
            calibrated_variant.get("irreducible_global_scale", np.nan),
            config["plant"]["irreducible_global_scale"], rtol=0, atol=1e-12),
        "selected_omega_scales_current": bool(
            np.isclose(calibrated_variant.get("one_qubit_omega_global_scale", np.nan),
                       config["plant"]["one_qubit_omega_global_scale"], rtol=0, atol=1e-12)
            and np.isclose(calibrated_variant.get("two_qubit_omega_global_scale", np.nan),
                           config["plant"]["two_qubit_omega_global_scale"], rtol=0, atol=1e-12)
            and calibrated_coordinate_scales.shape == configured_coordinate_scales.shape
            and np.allclose(calibrated_coordinate_scales, configured_coordinate_scales,
                            rtol=0, atol=1e-12)),
        "selected_initial_sigma_has_physical_exploration_criterion": np.isclose(
            initial_sigma_gap_fraction,
            config["controller"]["initial_exploration"]["selected_gap_fraction"],
            rtol=0.05, atol=0),
        "selected_baseline_effective_rate_current": np.isclose(
            2.0 * config["controller"]["baseline_learning_rate"]
            * config["controller"]["baseline_weight"],
            config["controller"]["baseline_dynamics"][
                "selected_effective_update_rate"], rtol=0, atol=1e-12),
    }
    ablation_path = root / config["ablations"]["empirical_relative_normalization_bundle"]
    ablation: dict[str, Any] = {"pass": False, "path": str(ablation_path), "blocking_reasons": []}
    try:
        normalization = json.loads(ablation_path.read_text(encoding="utf-8"))
        boundary = Figure5aEmpiricalBoundary.from_artifact(plant, normalization)
        maximum_probability = config["ablations"]["bounded_action"]["maximum_probability"]
        bounded = Figure5aBoundedActionAblation(
            plant, maximum_probability=(None if maximum_probability is None
                                        else float(maximum_probability)),
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
    gates = {name: bool(value) for name, value in gates.items()}
    return {"pass": all(gates.values()), "gates": gates, "hashes": dependency_hashes(root, config),
            "figure5a_coordinate_note": "Canonical Figure 5a applies Gaussian p directly; empirical relative normalization is a separate noncanonical ablation.",
            "coordinate_registry_alignment": "SOURCE_P_EQUALS_APPLIED_PLANT_P",
            "initial_sigma_implied_fixed_to_oracle_gap_fraction":
                initial_sigma_gap_fraction,
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
    source_opportunities = (int(config["figure_s8_plant_calibration"][
        "source_qec_cycles_per_epoch"]) * plant.raw_detector_count / plant.rounds)
    s8_oracle_events = plant.expected_global_edr(
        optimum_zero, epoch=0, frequency=1 / 1000,
        target_controls=optimum_zero) * source_opportunities
    s8_peak_target = plant.optimum(250, 1 / 1000)
    s8_fixed_peak_events = plant.expected_global_edr(
        np.zeros(plant.control_count), epoch=250, frequency=1 / 1000,
        target_controls=s8_peak_target) * source_opportunities
    graphical_tolerance = float(config["figure_s8_plant_calibration"][
        "graphical_count_tolerance"])
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
    fixed_exact_marginals = plant.exact_raw_detector_marginals(
        np.zeros(41), epoch=epoch, frequency=1 / 1000,
        target_controls=oracle)
    oracle_exact_marginals = plant.exact_raw_detector_marginals(
        oracle, epoch=epoch, frequency=1 / 1000,
        target_controls=oracle)
    fixed_measured_marginals = fixed_observation.raw_counts / fixed_observation.shots
    oracle_measured_marginals = oracle_observation.raw_counts / oracle_observation.shots
    fixed_standard_error = np.sqrt(
        fixed_exact_marginals * (1.0 - fixed_exact_marginals) /
        fixed_observation.shots)
    oracle_standard_error = np.sqrt(
        oracle_exact_marginals * (1.0 - oracle_exact_marginals) /
        oracle_observation.shots)
    maximum_exact_monte_carlo_z = float(max(
        np.max(np.abs(fixed_measured_marginals - fixed_exact_marginals) /
               np.maximum(fixed_standard_error, np.finfo(float).tiny)),
        np.max(np.abs(oracle_measured_marginals - oracle_exact_marginals) /
               np.maximum(oracle_standard_error, np.finfo(float).tiny))))
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
        "exact_fault_signature_tensor_complete": bool(
            len(plant.channel_occurrences) == len(plant.fault_signatures)
            and all(item.shape[1] == plant.raw_detector_count
                    for item in plant.fault_signatures)),
        "exact_marginals_match_finite_shot_within_six_sigma":
            maximum_exact_monte_carlo_z <= 6.0,
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
        "s8_oracle_graphical_count_anchor": abs(
            s8_oracle_events - float(config["figure_s8_plant_calibration"][
                "oracle_detection_events_target"])) <= graphical_tolerance,
        "s8_fixed_peak_graphical_count_anchor": abs(
            s8_fixed_peak_events - float(config["figure_s8_plant_calibration"][
                "fixed_peak_detection_events_target"])) <= graphical_tolerance,
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
              "exact_marginal_evaluator_version":
                  plant.EXACT_MARGINAL_EVALUATOR_VERSION,
              "fault_signature_hash": plant.fault_signature_hash,
              "physical_channel_occurrence_count": len(plant.channel_occurrences),
              "physical_pauli_branch_signature_count": int(sum(
                  item.shape[0] for item in plant.fault_signatures)),
              "maximum_exact_vs_finite_shot_detector_binomial_z_score":
                  maximum_exact_monte_carlo_z,
              "finite_shot_cycles_per_policy": finite_shot_cycles,
              "finite_shot_counts": {"fixed": fixed_observation.raw_total,
                                     "oracle": oracle_observation.raw_total,
                                     "single_gate_baseline": single_baseline.raw_total,
                                     "single_gate_miscalibrated": single_full.raw_total},
              "single_gate_expected_curvature": omega,
              "figure_s8_graphical_count_calibration": {
                  "axis_scale": "10^6 detection events per training epoch",
                  "detector_opportunities_per_epoch": source_opportunities,
                  "oracle_detection_events_per_epoch": s8_oracle_events,
                  "fixed_peak_detection_events_per_epoch": s8_fixed_peak_events,
                  "graphical_tolerance": graphical_tolerance,
              },
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
