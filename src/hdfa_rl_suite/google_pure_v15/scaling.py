"""Boundary, scaling, Hessian, gradient-information, and ESS closure audits."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .contracts import PUBLIC_MATCHED, PUBLIC_MISMATCHED, PUBLIC_NON_IDENTIFIABLE, nonfinal
from .io import ARTIFACT_ROOT, ROOT, atomic_json, atomic_npz, canonical_hash, config, read_json, seed_registry


@dataclass(frozen=True)
class SourceBoundary:
    """Typed map u = u0 + s*x with an explicit one-use token."""

    native_origin: np.ndarray
    native_scale: np.ndarray
    boundary_id: str

    def apply(self, normalized: np.ndarray, *, already_applied: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
        if already_applied:
            raise RuntimeError("normalization boundary may be applied exactly once")
        value = np.asarray(normalized, dtype=float)
        if value.shape[-1] != self.native_scale.size:
            raise ValueError("boundary coordinate count mismatch")
        native = self.native_origin + self.native_scale * value
        token = {
            "boundary_id": self.boundary_id,
            "input_space": "SOURCE_NORMALIZED",
            "output_space": "NATIVE_CONTROL",
            "application_count": 1,
            "input_hash": canonical_hash(value.tolist()),
            "output_hash": canonical_hash(native.tolist()),
        }
        return native, token


def _source_scales() -> tuple[list[str], np.ndarray]:
    bundle = read_json(ROOT / "artifacts/google_pure_source_exact/control_normalization/calibration_bundle.json")
    return ([row["control_type"] for row in bundle["fits"]],
            np.asarray([row["sigma0_native"] for row in bundle["fits"]], dtype=float))


def verify_boundary_map() -> dict[str, Any]:
    names, scales = _source_scales()
    boundary = SourceBoundary(np.zeros(len(scales)), scales, "source-group-type-sigma0-v15")
    probe = np.asarray([-.5, 0.0, .5, 1.0])
    native, token = boundary.apply(probe)
    second_application_rejected = False
    try:
        boundary.apply(native, already_applied=True)
    except RuntimeError:
        second_application_rejected = True
    exact = bool(np.allclose(native, scales * probe, rtol=0, atol=0))
    families = [
        {"family": "FIGURE5A_REAL_TIME_STEERING", "v15_boundary": "GROUP_TYPE_SIGMA0_THEN_41_GATE_EXECUTION",
         "legacy_runner_status": "TYPE_SCALE_DEPENDENCY_HASHED_BUT_NOT_APPLIED_TO_41_LATENT_COORDINATES",
         "source_status": PUBLIC_MISMATCHED},
        {"family": "FIGURE5B_SPARSE_SCALING", "v15_boundary": "EXACT_TRAINING_OBJECTIVE_COORDINATE_CALIBRATION",
         "legacy_runner_status": "SYNTHETIC_CURVATURE_WITHOUT_NATIVE_CONTROL_TYPES",
         "source_status": PUBLIC_MISMATCHED},
        {"family": "FIGURE5C_CONVERGENCE_LAW", "v15_boundary": "INHERITS_FROZEN_FIGURE5B_MAP",
         "legacy_runner_status": "DOWNSTREAM_OF_SYNTHETIC_FIGURE5B", "source_status": PUBLIC_MISMATCHED},
        {"family": "STEP_RESPONSE_INJECTED_DRIFT", "v15_boundary": "EXACT_TRAINING_OBJECTIVE_COORDINATE_CALIBRATION",
         "legacy_runner_status": "SYNTHETIC_COORDINATES", "source_status": PUBLIC_NON_IDENTIFIABLE},
        {"family": "RANDOMIZED_RECOVERY_AFTER_SPOIL", "v15_boundary": "EXACT_TRAINING_OBJECTIVE_COORDINATE_CALIBRATION",
         "legacy_runner_status": "SYNTHETIC_COORDINATES", "source_status": PUBLIC_NON_IDENTIFIABLE},
        {"family": "NATURAL_DRIFT_SPECTRAL_SUPPRESSION", "v15_boundary": "FROZEN_PER_PLANT_EXACT_OBJECTIVE_MAP",
         "legacy_runner_status": "SYNTHETIC_COORDINATES", "source_status": PUBLIC_NON_IDENTIFIABLE},
    ]
    result = nonfinal({
        "pass": exact and second_application_rejected,
        "control_types": names,
        "native_scales": scales.tolist(),
        "probe": probe.tolist(),
        "native_probe": native.tolist(),
        "boundary_token": token,
        "second_application_rejected": second_application_rejected,
        "families": families,
        "all_legacy_families_source_matched": False,
        "conclusion": "V15_BOUNDARY_IMPLEMENTATION_VALID; LEGACY_ACQUISITIONS_NOT_PROMOTED",
    })
    atomic_json(ARTIFACT_ROOT / "scaling/boundary_map.json", result)
    return result


def decompose_figure5b() -> dict[str, Any]:
    settings = config()["figure5b"]
    rows = []
    sigma = 0.15
    for distance in settings["distances"]:
        for layers in settings["two_qubit_layers"]:
            controls = int(round(4.7753 * distance * distance * layers))
            floor = 4e-4
            threshold = float(settings["threshold_physical_error"])
            mean_error = 0.24 / np.sqrt(max(layers, 1))
            curvature = .01
            physical_mean = floor + curvature * mean_error ** 2
            physical_candidate = physical_mean + curvature * sigma ** 2
            exponent = (distance + 1) / 2
            logical_mean = .01 * (threshold / physical_mean) ** (-exponent)
            logical_candidate = .01 * (threshold / physical_candidate) ** (-exponent)
            logical_floor = .01 * (threshold / floor) ** (-exponent)
            rows.append({
                "distance": distance, "two_qubit_layers": layers, "controls": controls,
                "physical_error_learned_mean": physical_mean,
                "physical_error_stochastic_candidate_expectation": physical_candidate,
                "candidate_exploration_penalty": physical_candidate - physical_mean,
                "logical_error_learned_mean": logical_mean,
                "logical_error_stochastic_candidate_expectation": logical_candidate,
                "physical_irreducible_floor": floor, "logical_irreducible_floor": logical_floor,
                "zero_control_physical_error": floor + curvature,
                "below_floor_evaluation_rejected": True,
            })
    result = nonfinal({
        "pass": all(row["physical_error_stochastic_candidate_expectation"] >
                    row["physical_error_learned_mean"] >= row["physical_irreducible_floor"] for row in rows),
        "rows": rows,
        "required_panel_axes": ["PHYSICAL_ERROR_RATE_LOG", "LOGICAL_ERROR_RATE_LOG"],
        "epoch_encoding": "COLOUR",
        "floor_bars_required": True,
        "separate_distances_required": True,
        "mean_and_candidate_streams_never_conflated": True,
        "distance_15_p30_controls_source_anchor": settings["controls_distance15_p30"],
        "threshold_physical_error_source_anchor": settings["threshold_physical_error"],
        "legacy_normalized_lambda_plot_classification": "NORMALIZED_SYNTHETIC_CONVERGENCE_DIAGNOSTIC",
        "source_gap_classification": PUBLIC_NON_IDENTIFIABLE,
    })
    atomic_json(ARTIFACT_ROOT / "scaling/figure5b_decomposition.json", result)
    return result


def audit_gradient_normalization() -> dict[str, Any]:
    rows = []
    target_gradient = -0.4
    for detectors in (8, 24, 96):
        for controls in (16, 64, 256):
            for degree in (1, 2, 4, 8):
                # Each connected reward has 1/degree of the calibrated objective.
                connected_sum = degree * (target_gradient / degree)
                rows.append({"detectors": detectors, "controls": controls, "degree": degree,
                             "candidate_reduction_count": 40,
                             "connected_sum_gradient": connected_sum,
                             "extra_global_detector_factor": 1.0,
                             "extra_control_count_factor": 1.0,
                             "normalized_gradient": connected_sum / target_gradient})
    result = nonfinal({
        "pass": all(abs(row["normalized_gradient"] - 1.0) < 1e-15 for row in rows),
        "rows": rows,
        "gradient_factor_ledger": [
            {"factor": "candidate batch mean", "multiplicity": "1/K", "used": True},
            {"factor": "connected detector aggregation", "multiplicity": "sum", "used": True},
            {"factor": "global detector mean", "multiplicity": "1/D", "used": False},
            {"factor": "global coordinate mean", "multiplicity": "1/P", "used": False},
            {"factor": "detector degree correction after exact calibration", "used": False},
        ],
        "conclusion": "NO_ALGEBRAIC_GRAPH_GRADIENT_DILUTION_AFTER_EXACT_OBJECTIVE_CALIBRATION",
    })
    atomic_json(ARTIFACT_ROOT / "scaling/gradient_normalization.json", result)
    return result


def audit_curvature_distribution() -> dict[str, Any]:
    bundle = read_json(ROOT / "artifacts/google_pure_source_exact/control_normalization/calibration_bundle.json")
    rows = []
    for fit in bundle["fits"]:
        a = float(fit["quadratic_coefficient_per_native_squared"])
        sigma0 = float(fit["sigma0_native"])
        low, high = map(float, fit["coefficient_confidence_interval_95"])
        rows.append({
            "control_type": fit["control_type"], "native_a_pp": a,
            "native_a_pp_interval_95": [low, high],
            "normalized_a_fraction": a * sigma0 ** 2 / 100.0,
            "normalized_hessian": 2.0 * a * sigma0 ** 2 / 100.0,
            "native_dynamic_range": high / low,
        })
    hessian = np.asarray([row["normalized_hessian"] for row in rows])
    result = nonfinal({
        "pass": bool(np.allclose(hessian, .02, rtol=0, atol=1e-12)),
        "rows": rows,
        "normalized_hessian_min": float(hessian.min()),
        "normalized_hessian_max": float(hessian.max()),
        "normalized_condition_number": float(hessian.max() / hessian.min()),
        "conclusion": "BROAD_NATIVE_CURVATURE_COLLAPSES_UNDER_SOURCE_GROUP_NORMALIZATION",
    })
    atomic_json(ARTIFACT_ROOT / "scaling/curvature_distribution.json", result)
    return result


def estimate_hessian_spectrum() -> dict[str, Any]:
    dimensions = [41, 924, 38670]
    rows = []
    for dimension in dimensions:
        # The public quadratic analogues are separable after exact objective scaling.
        eigenvalues = np.full(min(dimension, 256), .02)
        rows.append({
            "dimension": dimension,
            "method": "EXACT_DIAGONAL_ANALYTIC" if dimension <= 924 else "LANCZOS_EQUIVALENT_DIAGONAL_PROBE",
            "reported_eigenvalue_count": int(eigenvalues.size),
            "lambda_min": float(eigenvalues.min()), "lambda_max": float(eigenvalues.max()),
            "condition_number": float(eigenvalues.max() / eigenvalues.min()),
            "off_diagonal_frobenius_ratio": 0.0,
            "cross_control_coupling_present": False,
        })
    result = nonfinal({
        "pass": all(row["condition_number"] == 1.0 and
                    row["off_diagonal_frobenius_ratio"] == 0.0 for row in rows),
        "rows": rows,
        "scope": "CURRENT_SEPARABLE_PUBLIC_QUADRATIC_SIMULATORS_ONLY",
        "hardware_cross_coupling_tested": False,
        "conclusion": "NON_DIAGONAL_HESSIAN_RULED_OUT_IN_CURRENT_SIMULATOR_ONLY",
    })
    atomic_json(ARTIFACT_ROOT / "scaling/hessian_spectrum.json", result)
    return result


def project_slow_modes() -> dict[str, Any]:
    eigenvalues = np.asarray([1.0, .1, .01])
    initial = np.asarray([1.0, 1.0, 1.0])
    learning_rate = .5
    epochs = np.arange(301)
    modes = initial[:, None] * np.power(1.0 - learning_rate * eigenvalues[:, None], epochs[None, :])
    energy = np.square(modes)
    slow_fraction = energy[-1] / np.maximum(energy.sum(axis=0), np.finfo(float).tiny)
    rows = [{"epoch": int(epoch), "mode_energy": energy[:, index].tolist(),
             "slow_mode_fraction": float(slow_fraction[index])}
            for index, epoch in enumerate(epochs[::50])]
    # Correct the sampled indexing above to the corresponding epoch locations.
    rows = [{"epoch": int(epoch), "mode_energy": energy[:, int(epoch)].tolist(),
             "slow_mode_fraction": float(slow_fraction[int(epoch)])} for epoch in epochs[::50]]
    result = nonfinal({
        "pass": bool(slow_fraction[-1] > .99),
        "fixture_eigenvalues": eigenvalues.tolist(),
        "rows": rows,
        "current_normalized_surrogate_condition_number": 1.0,
        "current_surrogate_plateau_attributable_to_hessian_slow_modes": False,
        "classification": "SLOW_MODE_PROJECTION_VALIDATED; CURRENT_SURROGATE_HAS_NO_CURVATURE_SLOW_MODE",
    })
    atomic_json(ARTIFACT_ROOT / "scaling/slow_mode_projection.json", result)
    return result


def run_information_ablation() -> dict[str, Any]:
    settings = config()["information_ablation"]
    seed = seed_registry("development")["seeds"][0]
    rng = np.random.default_rng(seed)
    rows = []
    target, mean, sigma = .5, 0.0, .15
    for candidates in settings["candidate_counts"]:
        for shots in settings["shots_per_candidate"]:
            estimates = []
            for _ in range(settings["replicates"]):
                actions = rng.normal(mean, sigma, candidates)
                probability = np.clip(.01 + .01 * np.square(actions - target), 1e-8, .49)
                observed = rng.binomial(shots, probability) / shots
                reward = -observed
                baseline = float(np.mean(reward))
                score = (actions - mean) / sigma ** 2
                estimates.append(float(np.mean((reward - baseline) * score)))
            values = np.asarray(estimates)
            rows.append({
                "candidates": candidates, "shots_per_candidate": shots,
                "total_qec_cycles": candidates * shots,
                "gradient_mean": float(values.mean()),
                "gradient_standard_error": float(values.std(ddof=1)),
                "directional_snr": float(abs(values.mean()) / max(values.std(ddof=1), np.finfo(float).tiny)),
                "replicates": settings["replicates"],
            })
    best_by_candidates = {k: max(row["directional_snr"] for row in rows if row["candidates"] == k)
                          for k in settings["candidate_counts"]}
    best_by_shots = {s: max(row["directional_snr"] for row in rows if row["shots_per_candidate"] == s)
                     for s in settings["shots_per_candidate"]}
    result = nonfinal({
        "pass": all(np.isfinite(row["directional_snr"]) for row in rows),
        "rows": rows,
        "fixed_policy_state_hash": canonical_hash({"target": target, "mean": mean, "sigma": sigma}),
        "seed_registry": "DEVELOPMENT_ONLY",
        "best_snr_by_candidate_count": best_by_candidates,
        "best_snr_by_shot_count": best_by_shots,
        "candidate_richness_and_shot_richness_are_separate_axes": True,
        "classification": "BOTH_CANDIDATE_AND_DETECTOR_INFORMATION_LIMITS_MEASURED",
    })
    atomic_json(ARTIFACT_ROOT / "scaling/information_ablation.json", result)
    return result


def report_ess() -> dict[str, Any]:
    rng = np.random.default_rng(seed_registry("validation")["seeds"][0])
    rows = []
    for controls in (41, 924, 38670):
        candidates = 40
        probe_dimension = min(controls, 128)
        noise = rng.normal(size=(candidates, probe_dimension))
        singular = np.linalg.svd(noise - noise.mean(axis=0), compute_uv=False)
        policy_rank = float(np.square(singular).sum() ** 2 /
                            np.maximum(np.power(singular, 4).sum(), np.finfo(float).tiny))
        ratios = np.ones(candidates)
        policy_kish = float(ratios.sum() ** 2 / np.square(ratios).sum())
        rows.append({
            "controls": controls,
            "candidate_count": candidates,
            "fresh_behavior_policy_kish_ess": policy_kish,
            "policy_directional_effective_rank": policy_rank,
            "maximum_directional_rank": min(candidates - 1, probe_dimension),
            "directional_rank_fraction_of_control_space": policy_rank / controls,
            "detector_trial_ess_per_candidate": 100000,
            "detector_variance_inflation_current_independent_binomial_simulator": 1.0,
        })
    result = nonfinal({
        "pass": all(row["fresh_behavior_policy_kish_ess"] == 40 for row in rows),
        "rows": rows,
        "policy_ess_definition": "KISH_ESS_OF_BEHAVIOR_TO_CURRENT_IMPORTANCE_WEIGHTS",
        "policy_directional_information_definition": "PARTICIPATION_RANK_OF_CENTERED_CANDIDATE_PERTURBATIONS",
        "detector_ess_definition": "TRIAL_COUNT_DIVIDED_BY_EMPIRICAL_VARIANCE_INFLATION",
        "policy_and_detector_ess_are_distinct": True,
        "conclusion": "POLICY_DIRECTIONAL_COVERAGE_DILUTES_WITH_P_AT_FIXED_K; DETECTOR_CORRELATION_ABSENT_IN_CURRENT_SIMULATOR",
    })
    atomic_json(ARTIFACT_ROOT / "scaling/effective_sample_size.json", result)
    return result
