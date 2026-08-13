"""Figure-S8-anchored clean-room nuisance calibration for the Figure 5a plant."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import canonical_hash
from .plant import Figure5aStimPlant

CALIBRATION_ALGORITHM_VERSION = "figure5a-exact-s8-curvature-family-calibration.v2"


def calibration_input(config: Mapping[str, Any]) -> dict[str, Any]:
    plant = config["plant"]
    fit = config["figure_s8_plant_calibration"]
    study = config["plant_nuisance_study"]
    return {
        "distance": int(plant["distance"]),
        "basis": str(plant["basis"]),
        "circuit_rounds": int(plant["circuit_rounds"]),
        "ensemble_seed": int(plant["ensemble_seed"]),
        "one_qubit_irreducible": list(plant["one_qubit_irreducible"]),
        "two_qubit_irreducible": list(plant["two_qubit_irreducible"]),
        "one_qubit_omega": list(plant["one_qubit_omega"]),
        "two_qubit_omega": list(plant["two_qubit_omega"]),
        "source_qec_cycles_per_epoch": int(fit["source_qec_cycles_per_epoch"]),
        "oracle_detection_events_target": float(fit["oracle_detection_events_target"]),
        "fixed_peak_detection_events_target": float(
            fit["fixed_peak_detection_events_target"]),
        "graphical_count_tolerance": float(fit["graphical_count_tolerance"]),
        "group_curvature_probe_sigma": float(fit["group_curvature_probe_sigma"]),
        "maximum_group_curvature_ratio": float(fit["maximum_group_curvature_ratio"]),
        "maximum_target_conditioning_residual_ratio": float(
            fit["maximum_target_conditioning_residual_ratio"]),
        "one_qubit_injection_mappings": list(study["one_qubit_injection_mappings"]),
        "curvature_target_seed": int(study["curvature_target_seed"]),
        "development_ensemble_seeds": [
            int(seed) for seed in study["development_ensemble_seeds"]],
        "curvature_conditioning_families": dict(
            study["curvature_conditioning_families"]),
        "selected_curvature_conditioning_family": str(
            study["selected_curvature_conditioning_family"]),
    }


def _target_curvature_factors(
    values: Mapping[str, Any], family: str,
) -> np.ndarray:
    families = values["curvature_conditioning_families"]
    if family not in families:
        raise ValueError("curvature family is outside the preregistered nuisance study")
    specification = families[family]
    standard_deviation = float(specification["log_standard_deviation"])
    maximum = float(specification["maximum_absolute_log_factor"])
    if standard_deviation < 0 or maximum < 0:
        raise ValueError("curvature-family log scales must be nonnegative")
    rng = np.random.default_rng(int(values["curvature_target_seed"]))
    log_factors = np.clip(
        standard_deviation * rng.normal(size=41), -maximum, maximum)
    # The common geometric scale is absorbed into kappa; only relative frozen
    # randomness belongs to the curvature-family definition.
    log_factors -= float(np.mean(log_factors))
    result = np.exp(log_factors)
    result.setflags(write=False)
    return result


def _plant(
    values: Mapping[str, Any], mapping: str, *, irreducible_scale: float = 1.0,
    one_qubit_omega_scale: float = 1.0, two_qubit_omega_scale: float = 1.0,
    omega_coordinate_scales: np.ndarray | None = None,
) -> Figure5aStimPlant:
    return Figure5aStimPlant(
        rounds=int(values["circuit_rounds"]), basis=str(values["basis"]),
        ensemble_seed=int(values["ensemble_seed"]),
        one_qubit_irreducible=tuple(values["one_qubit_irreducible"]),
        two_qubit_irreducible=tuple(values["two_qubit_irreducible"]),
        one_qubit_omega=tuple(values["one_qubit_omega"]),
        two_qubit_omega=tuple(values["two_qubit_omega"]),
        irreducible_global_scale=float(irreducible_scale),
        one_qubit_omega_global_scale=float(one_qubit_omega_scale),
        two_qubit_omega_global_scale=float(two_qubit_omega_scale),
        omega_coordinate_scales=None if omega_coordinate_scales is None else
            tuple(np.asarray(omega_coordinate_scales, dtype=float)),
        one_qubit_injection_mapping=mapping,
    )


def _edr(plant: Figure5aStimPlant, probabilities: np.ndarray) -> float:
    return float(np.mean(
        plant._exact_raw_detector_marginals_from_probabilities(probabilities)))


def _bisect_monotone(
    function: Any, target: float, lower: float, upper: float, *, iterations: int = 55,
) -> float:
    low_value, high_value = float(function(lower)), float(function(upper))
    if not low_value <= target <= high_value:
        raise RuntimeError(
            f"calibration target {target} is not bracketed by [{low_value}, {high_value}]")
    for _ in range(iterations):
        middle = 0.5 * (lower + upper)
        if float(function(middle)) < target:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


def _coordinate_curvatures(
    plant: Figure5aStimPlant, base: np.ndarray, raw_omega: np.ndarray,
    scales: np.ndarray, *, probe_sigma: float,
) -> np.ndarray:
    probe_variance = float(probe_sigma) ** 2
    base_edr = _edr(plant, base)
    values = []
    for index in range(plant.control_count):
        probabilities = base.copy()
        probabilities[index] += raw_omega[index] * scales[index] * probe_variance
        values.append((_edr(plant, probabilities) - base_edr) / probe_variance)
    result = np.asarray(values)
    if not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise RuntimeError("Figure S8 coordinate curvatures must be positive and finite")
    return result


def _fit_coordinate_scales(
    plant: Figure5aStimPlant, base: np.ndarray, raw_omega: np.ndarray,
    *, target_peak_edr: float, probe_sigma: float,
    target_curvature_factors: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Match frozen relative target curvatures, then fit one global magnitude."""
    target_factors = np.asarray(target_curvature_factors, dtype=float)
    if (target_factors.shape != (plant.control_count,)
            or not np.all(np.isfinite(target_factors))
            or np.any(target_factors <= 0)):
        raise ValueError("target curvature factors must contain 41 positive values")

    def global_scale(relative: np.ndarray) -> float:
        maximum = float(np.min(
            (plant.probability_ceilings - base) / (raw_omega * relative)) * 0.99)
        return _bisect_monotone(
            lambda scale: _edr(plant, base + raw_omega * relative * scale),
            target_peak_edr, 0.0, maximum)

    relative = np.ones(plant.control_count, dtype=float)
    history = []
    for iteration in range(8):
        magnitude = global_scale(relative)
        final = relative * magnitude
        curvatures = _coordinate_curvatures(
            plant, base, raw_omega, final, probe_sigma=probe_sigma)
        normalized = curvatures / target_factors
        ratio = float(np.max(normalized) / np.min(normalized))
        history.append({
            "iteration": iteration,
            "overall_scale": magnitude,
            "target_conditioning_residual_ratio": ratio,
        })
        if ratio <= 1.000001:
            break
        common = float(np.exp(np.mean(np.log(normalized))))
        relative *= common * target_factors / curvatures
        relative /= float(np.exp(np.mean(np.log(relative))))
    else:
        raise RuntimeError("coordinate sensitivity equalization did not converge")
    final = relative * global_scale(relative)
    curvatures = _coordinate_curvatures(
        plant, base, raw_omega, final, probe_sigma=probe_sigma)
    peak = _edr(plant, base + raw_omega * final)
    if abs(peak - target_peak_edr) > 1e-10:
        raise RuntimeError("coordinate sensitivity fit missed the fixed-peak target")
    one_global = float(np.exp(np.mean(np.log(final[:17]))))
    two_global = float(np.exp(np.mean(np.log(final[17:]))))
    coordinate = final / np.r_[np.full(17, one_global), np.full(24, two_global)]
    diagnostics: dict[str, Any] = {
        "fixed_peak_edr": peak,
        "target_curvature_factors": target_factors.tolist(),
        "target_curvature_factor_ratio": float(
            np.max(target_factors) / np.min(target_factors)),
        "coordinate_curvatures": curvatures.tolist(),
        "coordinate_curvature_ratio": float(np.max(curvatures) / np.min(curvatures)),
        "target_conditioning_residual_ratio": float(
            np.max(curvatures / target_factors) /
            np.min(curvatures / target_factors)),
        "coordinate_curvature_coefficient_of_variation": float(
            np.std(curvatures) / np.mean(curvatures)),
        "iterations": history,
        "within_group_coordinate_scale_geometric_means": [
            float(np.exp(np.mean(np.log(coordinate[:17])))),
            float(np.exp(np.mean(np.log(coordinate[17:])))),
        ],
    }
    return final, diagnostics


def calibrate_variant(
    config: Mapping[str, Any], mapping: str, curvature_family: str,
    *, ensemble_seed: int | None = None,
) -> dict[str, Any]:
    values = calibration_input(config)
    if ensemble_seed is not None:
        if int(ensemble_seed) not in values["development_ensemble_seeds"]:
            raise ValueError("ensemble seed is outside the preregistered development family")
        values["ensemble_seed"] = int(ensemble_seed)
    if mapping not in values["one_qubit_injection_mappings"]:
        raise ValueError("mapping is outside the preregistered nuisance study")
    target_curvature_factors = _target_curvature_factors(values, curvature_family)
    plant = _plant(values, mapping)
    raw_irreducible = np.asarray([item.irreducible_error for item in plant.inventory])
    raw_omega = np.asarray([item.omega_sensitivity for item in plant.inventory])
    opportunities = (values["source_qec_cycles_per_epoch"] *
                     plant.raw_detector_count / plant.rounds)
    target_oracle_edr = values["oracle_detection_events_target"] / opportunities
    target_peak_edr = values["fixed_peak_detection_events_target"] / opportunities
    maximum_irreducible_scale = float(np.min(
        plant.probability_ceilings / raw_irreducible) * 0.99)
    irreducible_scale = _bisect_monotone(
        lambda scale: _edr(plant, raw_irreducible * scale), target_oracle_edr,
        0.0, maximum_irreducible_scale)
    base = raw_irreducible * irreducible_scale
    final_scales, omega_fit = _fit_coordinate_scales(
        plant, base, raw_omega, target_peak_edr=target_peak_edr,
        probe_sigma=values["group_curvature_probe_sigma"],
        target_curvature_factors=target_curvature_factors)
    one_scale = float(np.exp(np.mean(np.log(final_scales[:17]))))
    two_scale = float(np.exp(np.mean(np.log(final_scales[17:]))))
    coordinate_scales = final_scales / np.r_[
        np.full(17, one_scale), np.full(24, two_scale)]
    calibrated = _plant(
        values, mapping, irreducible_scale=irreducible_scale,
        one_qubit_omega_scale=one_scale, two_qubit_omega_scale=two_scale,
        omega_coordinate_scales=coordinate_scales)
    calibrated_irreducible = [
        item.irreducible_error for item in calibrated.inventory]
    calibrated_omega = [item.omega_sensitivity for item in calibrated.inventory]
    optimum = np.ones(calibrated.control_count)
    oracle_edr = calibrated.expected_global_edr(
        optimum, epoch=250, frequency=1 / 1000, target_controls=optimum)
    fixed_peak_edr = calibrated.expected_global_edr(
        np.zeros(calibrated.control_count), epoch=250, frequency=1 / 1000,
        target_controls=optimum)
    probe = float(values["group_curvature_probe_sigma"])
    base_controls = np.zeros(calibrated.control_count)
    one_controls = base_controls.copy(); one_controls[:17] = probe
    two_controls = base_controls.copy(); two_controls[17:] = probe
    one_group_curvature = (calibrated.expected_global_edr(
        one_controls, epoch=0, frequency=1 / 1000,
        target_controls=base_controls) - oracle_edr) / probe**2
    two_group_curvature = (calibrated.expected_global_edr(
        two_controls, epoch=0, frequency=1 / 1000,
        target_controls=base_controls) - oracle_edr) / probe**2
    one_curvature = one_group_curvature / 17
    two_curvature = two_group_curvature / 24
    coordinate_curvatures = []
    for index in range(calibrated.control_count):
        plus_controls = base_controls.copy(); plus_controls[index] = probe
        minus_controls = base_controls.copy(); minus_controls[index] = -probe
        plus_edr = calibrated.exact_global_edr(
            plus_controls, epoch=0, frequency=1 / 1000,
            target_controls=base_controls)
        minus_edr = calibrated.exact_global_edr(
            minus_controls, epoch=0, frequency=1 / 1000,
            target_controls=base_controls)
        coordinate_curvatures.append(
            (plus_edr + minus_edr - 2.0 * oracle_edr) / (2.0 * probe**2))
    coordinate_array = np.asarray(coordinate_curvatures)
    conditioning_residual = coordinate_array / target_curvature_factors
    occurrence_counts = np.asarray([
        len(item.circuit_locations) for item in calibrated.inventory], dtype=float)
    mask_degrees = np.sum(calibrated.mask, axis=0).astype(float)
    calibrated_omega_array = np.asarray(calibrated_omega, dtype=float)
    physical_edr_sensitivities = coordinate_array / calibrated_omega_array

    def correlation(left: np.ndarray, right: np.ndarray) -> float | None:
        if np.std(left) == 0 or np.std(right) == 0:
            return None
        return float(np.corrcoef(left, right)[0, 1])

    parameter_diagnostics = [{
        "parameter_id": item.parameter_id,
        "gate_type": item.gate_type,
        "occurrences": int(occurrence_counts[index]),
        "occurrences_per_round": float(occurrence_counts[index] / calibrated.rounds),
        "mask_degree": int(mask_degrees[index]),
        "omega_sensitivity": float(calibrated_omega_array[index]),
        "measured_control_curvature": float(coordinate_array[index]),
        "curvature_divided_by_omega_A": float(physical_edr_sensitivities[index]),
        "target_curvature_factor": float(target_curvature_factors[index]),
    } for index, item in enumerate(calibrated.inventory)]
    oracle_events, fixed_events = oracle_edr * opportunities, fixed_peak_edr * opportunities
    count_tolerance = float(values["graphical_count_tolerance"])
    group_ratio = max(one_curvature, two_curvature) / min(one_curvature, two_curvature)
    gates = {
        "oracle_count_matches_graphical_anchor":
            abs(oracle_events - values["oracle_detection_events_target"]) <= count_tolerance,
        "fixed_peak_count_matches_graphical_anchor":
            abs(fixed_events - values["fixed_peak_detection_events_target"]) <= count_tolerance,
        "one_two_qubit_group_curvatures_conditioned":
            group_ratio <= float(values["maximum_group_curvature_ratio"]),
        "all_coordinate_curvatures_match_frozen_family_target":
            float(np.max(conditioning_residual) / np.min(conditioning_residual)) <=
            float(values["maximum_target_conditioning_residual_ratio"]),
        "all_coordinate_curvatures_positive": bool(np.all(coordinate_array > 0)),
        "alternative_one_qubit_location_count_exact":
            mapping != "one_location_per_cycle" or all(
                len(item.circuit_locations) == calibrated.rounds
                for item in calibrated.inventory[:17]),
    }
    return {
        "mapping": mapping,
        "ensemble_seed": int(values["ensemble_seed"]),
        "curvature_conditioning_family": curvature_family,
        "irreducible_global_scale": irreducible_scale,
        "one_qubit_omega_global_scale": one_scale,
        "two_qubit_omega_global_scale": two_scale,
        "omega_coordinate_scales": coordinate_scales.tolist(),
        "raw_irreducible_draws": raw_irreducible.tolist(),
        "raw_omega_draws": raw_omega.tolist(),
        "calibrated_irreducible": calibrated_irreducible,
        "calibrated_omega": calibrated_omega,
        "raw_random_draws_preserved_before_detector_sensitivity_conditioning": True,
        "coordinate_conditioning_method":
            "iterated_exact_physical_Pauli_channel_local_EDR_curvature_matching_to_frozen_random_family_targets",
        "exact_marginal_evaluator_version": calibrated.EXACT_MARGINAL_EVALUATOR_VERSION,
        "fault_signature_hash": calibrated.fault_signature_hash,
        "detector_opportunities_per_epoch": opportunities,
        "oracle_edr": oracle_edr,
        "fixed_peak_edr": fixed_peak_edr,
        "oracle_detection_events_per_epoch": oracle_events,
        "fixed_peak_detection_events_per_epoch": fixed_events,
        "one_qubit_group_curvature": one_curvature,
        "two_qubit_group_curvature": two_curvature,
        "one_qubit_aggregate_group_curvature": one_group_curvature,
        "two_qubit_aggregate_group_curvature": two_group_curvature,
        "group_curvature_comparison": "aggregate Gaussian group curvature divided by controls in group",
        "group_curvature_ratio": group_ratio,
        "coordinate_curvatures": coordinate_curvatures,
        "target_curvature_factors": target_curvature_factors.tolist(),
        "target_curvature_factor_ratio": float(
            np.max(target_curvature_factors) / np.min(target_curvature_factors)),
        "target_conditioning_residual_ratio": float(
            np.max(conditioning_residual) / np.min(conditioning_residual)),
        "coordinate_curvature_ratio": float(np.max(coordinate_array) / np.min(coordinate_array)),
        "coordinate_curvature_coefficient_of_variation": float(
            np.std(coordinate_array) / np.mean(coordinate_array)),
        "parameter_diagnostics": parameter_diagnostics,
        "mapping_sensitivity_correlations": {
            "A_vs_occurrence_count": correlation(
                physical_edr_sensitivities, occurrence_counts),
            "A_vs_mask_degree": correlation(
                physical_edr_sensitivities, mask_degrees),
            "control_curvature_vs_occurrence_count": correlation(
                coordinate_array, occurrence_counts),
            "control_curvature_vs_mask_degree": correlation(
                coordinate_array, mask_degrees),
        },
        "one_qubit_location_counts": [
            len(item.circuit_locations) for item in calibrated.inventory[:17]],
        "two_qubit_location_counts": [
            len(item.circuit_locations) for item in calibrated.inventory[17:]],
        "detector_count": calibrated.detector_count,
        "raw_detector_count": calibrated.raw_detector_count,
        "mask_density": float(calibrated.mask.mean()),
        "mask_hash": canonical_hash(calibrated.mask.astype(int).tolist()),
        "plant_hash": calibrated.plant_hash,
        "inventory_hash": canonical_hash([asdict(item) for item in calibrated.inventory]),
        "omega_fit_diagnostics": omega_fit,
        "gates": gates,
        "pass": all(gates.values()),
        "blocking_reasons": [name for name, passed in gates.items() if not passed],
    }


def calibrate_all_variants(config: Mapping[str, Any]) -> dict[str, Any]:
    values = calibration_input(config)
    family_variants = {
        family: {
            mapping: calibrate_variant(config, mapping, family)
            for mapping in values["one_qubit_injection_mappings"]
        }
        for family in values["curvature_conditioning_families"]
    }
    selected_family = values["selected_curvature_conditioning_family"]
    variants = family_variants[selected_family]
    additional_development_ensembles = {
        str(seed): {
            family: {
                mapping: calibrate_variant(
                    config, mapping, family, ensemble_seed=int(seed))
                for mapping in values["one_qubit_injection_mappings"]
            }
            for family in values["curvature_conditioning_families"]
        }
        for seed in values["development_ensemble_seeds"]
        if int(seed) != int(values["ensemble_seed"])
    }
    payload = {
        "schema_version": "figure5a-s8-plant-calibration.v2",
        "calibration_algorithm_version": CALIBRATION_ALGORITHM_VERSION,
        "scientific_status":
            "CLEAN_ROOM_NUISANCE_FIT_TO_APPROXIMATE_GRAPHICAL_FIGURE_S8_ANCHORS",
        "figure_s8_axis_scale": "10^6 detection events per training epoch",
        "source_numerical_values_published": False,
        "canonical_deterministic_evaluator":
            "exact_independent_physical_Pauli_channels_with_mutually_exclusive_branches",
        "calibration_input": values,
        "calibration_input_hash": canonical_hash(values),
        "variants": variants,
        "curvature_family_variants": family_variants,
        "additional_development_ensemble_variants":
            additional_development_ensembles,
        "mapping_comparison": {
            "same_reduced_detector_count": len({
                item["detector_count"] for item in variants.values()}) == 1,
            "same_detector_control_mask": len({
                item["mask_hash"] for item in variants.values()}) == 1,
            "one_qubit_location_count_ranges": {
                mapping: [min(item["one_qubit_location_counts"]),
                          max(item["one_qubit_location_counts"])]
                for mapping, item in variants.items()
            },
            "calibrated_omega_ranges": {
                mapping: [min(item["calibrated_omega"]), max(item["calibrated_omega"])]
                for mapping, item in variants.items()
            },
            "both_match_same_S8_anchors_and_conditioning_gate":
                all(item["pass"] for item in variants.values()),
        },
        "curvature_family_comparison": {
            family: {
                "target_curvature_factor_ratio": next(iter(items.values()))[
                    "target_curvature_factor_ratio"],
                "all_mappings_pass": all(item["pass"] for item in items.values()),
            }
            for family, items in family_variants.items()
        },
        "development_plant_family_count": int(
            len(values["development_ensemble_seeds"])
            * len(values["curvature_conditioning_families"])
            * len(values["one_qubit_injection_mappings"])),
        "mathematical_contract_pass": bool(
            all(item["pass"] for items in family_variants.values()
                for item in items.values())
            and all(item["pass"]
                    for ensembles in additional_development_ensembles.values()
                    for items in ensembles.values() for item in items.values())),
        "long_rl_acquisition_used": False,
        "certification_seeds_consumed": False,
    }
    payload["calibration_hash"] = canonical_hash(payload)
    return payload


def plot_mapping_sensitivity_diagnostics(
    calibration: Mapping[str, Any], output_path: Path,
) -> dict[str, Any]:
    """Plot physical EDR sensitivity against occurrence count and mask degree."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = calibration["variants"]
    mappings = tuple(variants)
    figure, axes = plt.subplots(
        len(mappings), 2, figsize=(10.0, 4.0 * len(mappings)), squeeze=False)
    colors = {"single_qubit": "#1769aa", "two_qubit": "#d95f02"}
    for row, mapping in enumerate(mappings):
        diagnostics = variants[mapping]["parameter_diagnostics"]
        correlations = variants[mapping]["mapping_sensitivity_correlations"]
        for gate_type in colors:
            selected = [item for item in diagnostics if item["gate_type"] == gate_type]
            label = "1Q" if gate_type == "single_qubit" else "2Q"
            axes[row, 0].scatter(
                [item["occurrences_per_round"] for item in selected],
                [item["curvature_divided_by_omega_A"] for item in selected],
                color=colors[gate_type], label=label, alpha=0.85)
            axes[row, 1].scatter(
                [item["mask_degree"] for item in selected],
                [item["curvature_divided_by_omega_A"] for item in selected],
                color=colors[gate_type], label=label, alpha=0.85)
        axes[row, 0].set_xlabel("Physical channel occurrences per QEC cycle")
        axes[row, 1].set_xlabel("Reduced detector-control mask degree")
        for column in range(2):
            axes[row, column].set_ylabel(r"Physical EDR sensitivity $A_i=c_i/\Omega_i$")
            axes[row, column].grid(alpha=0.2)
            axes[row, column].legend(frameon=False)
        occurrence_correlation = correlations["A_vs_occurrence_count"]
        occurrence_text = ("undefined" if occurrence_correlation is None
                           else f"{occurrence_correlation:.3f}")
        axes[row, 0].set_title(
            f"{mapping}\nPearson r={occurrence_text}")
        mask_correlation = correlations["A_vs_mask_degree"]
        mask_text = "undefined" if mask_correlation is None else f"{mask_correlation:.3f}"
        axes[row, 1].set_title(f"{mapping}\nPearson r={mask_text}")
    figure.suptitle(
        "Figure 5a reconstructed-plant mapping sensitivity diagnostic", y=0.995)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    svg_path = output_path.with_suffix(".svg")
    figure.savefig(svg_path, bbox_inches="tight")
    plt.close(figure)
    return {
        "schema_version": "figure5a-plant-mapping-sensitivity-plot.v1",
        "scientific_status": "CLEAN_ROOM_MAPPING_DIAGNOSTIC",
        "calibration_hash": calibration["calibration_hash"],
        "png_path": str(output_path.resolve()),
        "svg_path": str(svg_path.resolve()),
        "mappings": list(mappings),
    }
