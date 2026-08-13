"""Audits for the exact physical-Pauli-channel Figure 5a marginal evaluator."""
from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

from .contracts import canonical_hash
from .plant import Figure5aStimPlant
from .validation import build_plant

MARGINAL_EXACTNESS_AUDIT_VERSION = "figure5a-marginal-exactness-audit.v1"


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return None if denominator == 0 else float(np.vdot(left, right) / denominator)


def _reward_jacobian(
    evaluator: Callable[[np.ndarray], np.ndarray], controls: np.ndarray, *, delta: float,
) -> np.ndarray:
    columns = []
    for coordinate in range(len(controls)):
        plus = controls.copy(); plus[coordinate] += delta
        minus = controls.copy(); minus[coordinate] -= delta
        columns.append((evaluator(plus) - evaluator(minus)) / (2.0 * delta))
    return np.stack(columns, axis=1)


def audit_marginal_exactness(
    config: Mapping[str, Any], *, random_policy_count: int = 100,
    gradient_policy_count: int = 3, monte_carlo_policy_count: int = 3,
    monte_carlo_qec_cycles: int = 900_000, seed: int = 53119,
    control_scale: float = 0.25, gradient_delta: float = 1e-5,
) -> dict[str, Any]:
    """Compare exact, approximate-DEM, and actual finite-shot Stim acquisition."""
    if min(random_policy_count, gradient_policy_count, monte_carlo_policy_count) <= 0:
        raise ValueError("marginal exactness audit counts must be positive")
    plant = build_plant(config)
    if monte_carlo_qec_cycles <= 0 or monte_carlo_qec_cycles % plant.rounds:
        raise ValueError("Monte Carlo QEC cycles must divide into whole circuit shots")
    rng = np.random.default_rng(seed)
    target = np.zeros(plant.control_count)
    policies = rng.normal(
        0.0, float(control_scale),
        size=(max(random_policy_count, gradient_policy_count,
                  monte_carlo_policy_count), plant.control_count))

    comparison_rows = []
    for index, controls in enumerate(policies[:random_policy_count]):
        exact = plant.exact_raw_detector_marginals(
            controls, epoch=0, frequency=1 / 1000, target_controls=target)
        approximate = plant.approximate_dem_raw_detector_marginals(
            controls, epoch=0, frequency=1 / 1000, target_controls=target)
        difference = exact - approximate
        comparison_rows.append({
            "policy_index": index,
            "exact_global_edr": float(np.mean(exact)),
            "approximate_dem_global_edr": float(np.mean(approximate)),
            "global_edr_difference": float(np.mean(difference)),
            "maximum_absolute_detector_marginal_difference": float(
                np.max(np.abs(difference))),
            "detector_marginal_l2_difference": float(np.linalg.norm(difference)),
        })

    gradient_rows = []
    for index, controls in enumerate(policies[:gradient_policy_count]):
        exact_jacobian = _reward_jacobian(
            lambda value: plant.exact_reward_rates(
                value, epoch=0, frequency=1 / 1000, target_controls=target),
            controls, delta=gradient_delta)
        approximate_jacobian = _reward_jacobian(
            lambda value: plant.approximate_dem_reward_rates(
                value, epoch=0, frequency=1 / 1000, target_controls=target),
            controls, delta=gradient_delta)
        difference = exact_jacobian - approximate_jacobian
        exact_norm = float(np.linalg.norm(exact_jacobian))
        column_cosines = [
            _cosine(exact_jacobian[:, coordinate],
                    approximate_jacobian[:, coordinate])
            for coordinate in range(plant.control_count)]
        gradient_rows.append({
            "policy_index": index,
            "reward_jacobian_cosine": _cosine(
                exact_jacobian.ravel(), approximate_jacobian.ravel()),
            "reward_jacobian_relative_l2_difference": float(
                np.linalg.norm(difference) /
                max(exact_norm, np.finfo(float).tiny)),
            "maximum_absolute_reward_jacobian_difference": float(
                np.max(np.abs(difference))),
            "minimum_defined_coordinate_cosine": min(
                value for value in column_cosines if value is not None),
        })

    monte_carlo_rows = []
    shots = monte_carlo_qec_cycles // plant.rounds
    for index, controls in enumerate(policies[:monte_carlo_policy_count]):
        exact = plant.exact_raw_detector_marginals(
            controls, epoch=0, frequency=1 / 1000, target_controls=target)
        observation = plant.sample_detector_observation(
            controls, epoch=0, frequency=1 / 1000,
            qec_cycles=monte_carlo_qec_cycles,
            seed=plant.stream_seed(seed, "marginal-exactness", 0, index),
            target_controls=target)
        measured = observation.raw_counts / float(observation.shots)
        standard_error = np.sqrt(exact * (1.0 - exact) / shots)
        z_scores = np.abs(measured - exact) / np.maximum(
            standard_error, np.finfo(float).tiny)
        monte_carlo_rows.append({
            "policy_index": index,
            "shots": shots,
            "exact_global_edr": float(np.mean(exact)),
            "measured_global_edr": float(np.mean(measured)),
            "global_edr_difference": float(np.mean(measured - exact)),
            "maximum_absolute_detector_marginal_difference": float(
                np.max(np.abs(measured - exact))),
            "maximum_detector_binomial_z_score": float(np.max(z_scores)),
        })

    maximum_edr_difference = max(
        abs(row["global_edr_difference"]) for row in comparison_rows)
    maximum_detector_difference = max(
        row["maximum_absolute_detector_marginal_difference"]
        for row in comparison_rows)
    maximum_gradient_difference = max(
        row["reward_jacobian_relative_l2_difference"] for row in gradient_rows)
    maximum_monte_carlo_z = max(
        row["maximum_detector_binomial_z_score"] for row in monte_carlo_rows)
    gates = {
        "fault_signature_tensor_nonempty": bool(
            plant.fault_signatures and len(plant.channel_occurrences)),
        "fault_signature_tensor_matches_raw_detector_count": all(
            item.ndim == 2 and item.shape[1] == plant.raw_detector_count
            for item in plant.fault_signatures),
        "canonical_expected_path_is_exact": bool(np.array_equal(
            plant.expected_reward_rates(
                policies[0], epoch=0, frequency=1 / 1000,
                target_controls=target),
            plant.exact_reward_rates(
                policies[0], epoch=0, frequency=1 / 1000,
                target_controls=target))),
        "exact_results_finite": bool(np.isfinite([
            maximum_edr_difference, maximum_detector_difference,
            maximum_gradient_difference]).all()),
        "monte_carlo_within_six_sigma_per_detector": maximum_monte_carlo_z <= 6.0,
    }
    result = {
        "schema_version": MARGINAL_EXACTNESS_AUDIT_VERSION,
        "scientific_status": "DETERMINISTIC_AND_FINITE_SHOT_PHYSICAL_VALIDATION",
        "plant_hash": plant.plant_hash,
        "fault_signature_hash": plant.fault_signature_hash,
        "exact_marginal_evaluator_version": plant.EXACT_MARGINAL_EVALUATOR_VERSION,
        "physical_channel_occurrence_count": len(plant.channel_occurrences),
        "physical_pauli_branch_signature_count": int(sum(
            item.shape[0] for item in plant.fault_signatures)),
        "random_policy_count": random_policy_count,
        "gradient_policy_count": gradient_policy_count,
        "gradient_delta": gradient_delta,
        "monte_carlo_policy_count": monte_carlo_policy_count,
        "monte_carlo_qec_cycles_per_policy": monte_carlo_qec_cycles,
        "comparison_rows": comparison_rows,
        "gradient_rows": gradient_rows,
        "monte_carlo_rows": monte_carlo_rows,
        "summary": {
            "maximum_absolute_global_edr_exact_minus_approximate":
                maximum_edr_difference,
            "maximum_absolute_detector_marginal_exact_minus_approximate":
                maximum_detector_difference,
            "maximum_reward_jacobian_relative_l2_exact_minus_approximate":
                maximum_gradient_difference,
            "minimum_reward_jacobian_cosine_exact_vs_approximate": min(
                row["reward_jacobian_cosine"] for row in gradient_rows
                if row["reward_jacobian_cosine"] is not None),
            "maximum_monte_carlo_detector_binomial_z_score": maximum_monte_carlo_z,
            "legacy_dem_approximation_material_at_audit_precision": bool(
                maximum_detector_difference > 1e-10 or
                maximum_gradient_difference > 1e-8),
        },
        "gates": gates,
        "pass": all(gates.values()),
        "blocking_reasons": [name for name, passed in gates.items() if not passed],
        "long_controller_acquisition_used": False,
        "certification_seeds_consumed": False,
    }
    result["audit_hash"] = canonical_hash(result)
    return result
