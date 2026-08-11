"""Independent validation gates for empirical normalization artifacts."""
from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import (
    PAPER_NORMALIZATION_METHOD,
    SCHEMA_VERSION,
    CalibrationBundle,
    ControlTypeSpec,
    FitRules,
    SensitivityFit,
    SourceIdentifiability,
    SweepResult,
    canonical_hash,
)
from .edr_measurement import DetectorEventEvaluator
from .normalized_coordinates import CoordinateVector, EmpiricalCoordinateSystem
from .quadratic_fit import coefficient_stability


def build_calibration_bundle_from_sweeps(control_specs: Sequence[ControlTypeSpec],
                                         sweeps: Sequence[SweepResult],
                                         fits: Sequence[SensitivityFit],
                                         fit_rules: FitRules, *,
                                         config_hash: str, source_contract_hash: str,
                                         full_scale_completed: bool = False,
                                         quantitative_match: bool = False,
                                         extra_blocking_reasons: Sequence[str] = ()) -> CalibrationBundle:
    specs, sweeps, fits = tuple(control_specs), tuple(sweeps), tuple(fits)
    if not specs or len(specs) != len(sweeps) or len(specs) != len(fits):
        raise ValueError("specs, sweeps, and fits must be non-empty and aligned")
    references = {canonical_hash(asdict(item.reference)) for item in sweeps}
    plants = {item.plant_hash for item in sweeps}
    if len(references) != 1 or len(plants) != 1:
        raise ValueError("all control types must use one frozen reference and plant")
    blockers = list(extra_blocking_reasons)
    if not full_scale_completed:
        blockers.append("full source-scale validation has not been completed")
    if not quantitative_match:
        blockers.append("quantitative equivalence to the published experiment is not established")
    blockers.append("public source leaves sweep budget, fit interval, and synthetic gain distribution unspecified")
    source_map = {
        "equation": SourceIdentifiability.SOURCE_LITERAL.value,
        "simultaneous_control_type_sweep": SourceIdentifiability.SOURCE_LITERAL.value,
        "stim_detector_sampling": SourceIdentifiability.SOURCE_LITERAL.value,
        "sweep_grid": SourceIdentifiability.SOURCE_UNSPECIFIED_PREREGISTERED.value,
        "fit_rules": SourceIdentifiability.SOURCE_UNSPECIFIED_PREREGISTERED.value,
        "synthetic_gain_distribution": SourceIdentifiability.SOURCE_UNSPECIFIED_PREREGISTERED.value,
        "proprietary_controller_code": SourceIdentifiability.NOT_PUBLICLY_IDENTIFIABLE.value,
    }
    return CalibrationBundle(
        schema_version=SCHEMA_VERSION,
        normalization_method=PAPER_NORMALIZATION_METHOD,
        reference=sweeps[0].reference,
        control_specs=specs,
        fits=fits,
        fit_rules=fit_rules,
        config_hash=config_hash,
        plant_hash=next(iter(plants)),
        source_contract_hash=source_contract_hash,
        source_identifiability=source_map,
        artifact_complete=True,
        mathematical_contract_pass=True,
        protocol_contract_pass=True,
        source_structure_match=True,
        quantitative_match=bool(quantitative_match),
        paper_comparable=bool(full_scale_completed and quantitative_match and not blockers),
        blocking_reasons=tuple(dict.fromkeys(blockers)),
    )


def validate_round_trip(coordinates: EmpiricalCoordinateSystem, *, seed: int = 1) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    normalized_values = rng.normal(0.0, 0.7, len(coordinates.parameter_ids))
    normalized = CoordinateVector(normalized_values, coordinates.parameter_ids, "normalized")
    native = coordinates.to_native(normalized)
    recovered = coordinates.to_normalized(native)
    error = float(np.max(np.abs(recovered.values - normalized.values)))
    return {"maximum_absolute_error": error, "passed": bool(error <= 1e-12)}


def validate_coefficient_remeasurement(first: Sequence[SensitivityFit],
                                       second: Sequence[SensitivityFit],
                                       rules: FitRules) -> dict[str, Any]:
    a = {item.control_type: item for item in first}
    b = {item.control_type: item for item in second}
    if set(a) != set(b):
        raise ValueError("remeasurement control types differ")
    rows = {name: coefficient_stability(a[name], b[name], rules) for name in sorted(a)}
    return {"rows": rows, "passed": all(bool(row["passed"]) for row in rows.values())}


def _measurement_standard_error_percentage_points(events: int, opportunities: int) -> float:
    p = (events + 0.5) / (opportunities + 1.0)
    return 100.0 * math.sqrt(p * (1.0 - p) / opportunities)


def validate_normalized_isotropy(evaluator: DetectorEventEvaluator,
                                  bundle: CalibrationBundle, *,
                                  normalized_sigma: float = 1.0,
                                  candidates: int = 24,
                                  shots_per_candidate: int = 1024,
                                  perturbation_seed: int = 73001,
                                  detector_seed: int = 74001) -> dict[str, Any]:
    if normalized_sigma <= 0:
        raise ValueError("normalized validation sigma must be positive")
    coordinates = EmpiricalCoordinateSystem(bundle)
    baseline = evaluator.measure_joint(
        {}, candidates=candidates, shots_per_candidate=shots_per_candidate,
        perturbation_seed=perturbation_seed, detector_seed=detector_seed)
    baseline_pp = baseline.edr_percentage_points
    baseline_se = _measurement_standard_error_percentage_points(
        baseline.detector_events, baseline.detector_opportunities)
    expected = normalized_sigma ** 2
    rows = []
    for index, spec in enumerate(bundle.control_specs):
        sigma_native = bundle.fit_by_type()[spec.control_type].sigma0_native * normalized_sigma
        measurement = evaluator.measure_joint(
            {spec.control_type: sigma_native}, candidates=candidates,
            shots_per_candidate=shots_per_candidate,
            perturbation_seed=perturbation_seed + 1009 * (index + 1),
            detector_seed=detector_seed + 1013 * (index + 1))
        increase = measurement.edr_percentage_points - baseline_pp
        measurement_se = _measurement_standard_error_percentage_points(
            measurement.detector_events, measurement.detector_opportunities)
        combined_se = math.sqrt(baseline_se ** 2 + measurement_se ** 2)
        relative_error = abs(increase - expected) / expected
        effective_tolerance = bundle.fit_rules.isotropy_relative_tolerance + 3.0 * combined_se / expected
        rows.append({
            "control_type": spec.control_type,
            "sigma0_native": bundle.fit_by_type()[spec.control_type].sigma0_native,
            "native_unit": spec.native_unit,
            "measured_edr_increase_percentage_points": increase,
            "expected_edr_increase_percentage_points": expected,
            "combined_standard_error_percentage_points": combined_se,
            "relative_error": relative_error,
            "effective_relative_tolerance": effective_tolerance,
            "passed": bool(increase > 0 and relative_error <= effective_tolerance),
        })
    increases = np.asarray([row["measured_edr_increase_percentage_points"] for row in rows])
    anisotropy = float(np.ptp(increases) / max(abs(float(np.mean(increases))), np.finfo(float).tiny))
    positive_total = float(np.sum(np.maximum(increases, 0.0)))
    maximum_damage_share = float(np.max(np.maximum(increases, 0.0)) / max(
        positive_total, np.finfo(float).tiny))
    dominance_threshold = min(0.75, 2.0 / len(rows))
    passed = all(row["passed"] for row in rows) and anisotropy <= (
        2 * bundle.fit_rules.isotropy_relative_tolerance + 6 * baseline_se / expected)
    return {
        "method": "independent finite-shot unit-variance perturbation in normalized coordinates",
        "baseline_edr_percentage_points": baseline_pp,
        "rows": rows,
        "relative_peak_to_peak_anisotropy": anisotropy,
        "maximum_single_type_damage_share": maximum_damage_share,
        "unit_dominance_threshold": dominance_threshold,
        "unit_dominated_damage": bool(maximum_damage_share > dominance_threshold),
        "passed": bool(passed),
    }


def validate_covariance_damage_prediction(evaluator: DetectorEventEvaluator,
                                          bundle: CalibrationBundle, *,
                                          normalized_sigma: float = 0.5,
                                          candidates: int = 32,
                                          shots_per_candidate: int = 1536,
                                          perturbation_seed: int = 75001,
                                          detector_seed: int = 76001) -> dict[str, Any]:
    coordinates = EmpiricalCoordinateSystem(bundle)
    dimension = len(coordinates.parameter_ids)
    covariance = np.eye(dimension) * normalized_sigma ** 2
    predicted = coordinates.predict_candidate_damage_percentage_points(
        np.zeros(dimension), covariance)
    sigma_by_type = coordinates.type_sigma_native({
        spec.control_type: normalized_sigma for spec in bundle.control_specs})
    baseline = evaluator.measure_joint(
        {}, candidates=candidates, shots_per_candidate=shots_per_candidate,
        perturbation_seed=perturbation_seed, detector_seed=detector_seed)
    direct = evaluator.measure_joint(
        sigma_by_type, candidates=candidates, shots_per_candidate=shots_per_candidate,
        perturbation_seed=perturbation_seed + 1, detector_seed=detector_seed + 1)
    measured = direct.edr_percentage_points - baseline.edr_percentage_points
    combined_se = math.sqrt(
        _measurement_standard_error_percentage_points(
            baseline.detector_events, baseline.detector_opportunities) ** 2
        + _measurement_standard_error_percentage_points(
            direct.detector_events, direct.detector_opportunities) ** 2)
    relative_error = abs(measured - predicted) / max(abs(predicted), np.finfo(float).tiny)
    effective_tolerance = bundle.fit_rules.prediction_relative_tolerance + 3 * combined_se / max(
        abs(predicted), np.finfo(float).tiny)
    return {
        "hessian_convention": "damage = 0.5*Tr(H*Sigma) + 0.5*mu.T*H*mu",
        "normalized_sigma": normalized_sigma,
        "predicted_damage_percentage_points": predicted,
        "direct_monte_carlo_damage_percentage_points": measured,
        "combined_standard_error_percentage_points": combined_se,
        "relative_error": relative_error,
        "effective_relative_tolerance": effective_tolerance,
        "passed": bool(measured > 0 and relative_error <= effective_tolerance),
    }


def validate_context_invalidation(bundle: CalibrationBundle) -> dict[str, bool]:
    circuit_rejected = detector_rejected = False
    try:
        bundle.validate_context(
            circuit_hash="changed-circuit",
            detector_set_hash=bundle.reference.detector_set_hash)
    except RuntimeError:
        circuit_rejected = True
    try:
        bundle.validate_context(
            circuit_hash=bundle.reference.circuit_hash,
            detector_set_hash="changed-detectors")
    except RuntimeError:
        detector_rejected = True
    return {
        "changed_circuit_rejected": circuit_rejected,
        "changed_detector_set_rejected": detector_rejected,
        "passed": circuit_rejected and detector_rejected,
    }


def audit_no_arbitrary_scale(config: Mapping[str, Any]) -> dict[str, Any]:
    forbidden = {"native_per_normalized", "algebraic_scale", "legacy_sensitivity_scale", "old_scale"}
    findings: list[str] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                child = f"{path}.{key}" if path else str(key)
                if str(key) in forbidden:
                    findings.append(child)
                walk(item, child)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")

    walk(config, "")
    return {
        "forbidden_scale_fields": sorted(forbidden),
        "findings": findings,
        "passed": not findings,
    }


def run_independent_validation(evaluator: DetectorEventEvaluator,
                               bundle: CalibrationBundle, *,
                               candidates: int, shots_per_candidate: int,
                               normalized_isotropy_sigma: float,
                               normalized_joint_sigma: float,
                               seed: int) -> dict[str, Any]:
    round_trip = validate_round_trip(EmpiricalCoordinateSystem(bundle), seed=seed)
    context = validate_context_invalidation(bundle)
    isotropy = validate_normalized_isotropy(
        evaluator, bundle, normalized_sigma=normalized_isotropy_sigma,
        candidates=candidates, shots_per_candidate=shots_per_candidate,
        perturbation_seed=seed + 101, detector_seed=seed + 201)
    prediction = validate_covariance_damage_prediction(
        evaluator, bundle, normalized_sigma=normalized_joint_sigma,
        candidates=candidates, shots_per_candidate=shots_per_candidate,
        perturbation_seed=seed + 301, detector_seed=seed + 401)
    gates = {
        "round_trip": bool(round_trip["passed"]),
        "context_invalidation": bool(context["passed"]),
        "normalized_isotropy": bool(isotropy["passed"]),
        "covariance_damage_prediction": bool(prediction["passed"]),
    }
    return {
        "schema_version": "google-pure-source-exact-normalization-validation.v1",
        "round_trip": round_trip,
        "context_invalidation": context,
        "normalized_isotropy": isotropy,
        "covariance_damage_prediction": prediction,
        "gates": gates,
        "passed": all(gates.values()),
    }
