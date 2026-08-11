from __future__ import annotations

from dataclasses import asdict
import math

import numpy as np
import pytest

from hdfa_rl_suite.google_pure_source_exact.control_normalization.contracts import (
    NON_PAPER_NORMALIZATION_ABLATION,
    SCHEMA_VERSION,
    CalibrationBundle,
    ControlTypeSpec,
    FitRules,
    SourceIdentifiability,
    SweepProtocol,
    build_source_contract,
    canonical_hash,
)
from hdfa_rl_suite.google_pure_source_exact.control_normalization.edr_measurement import (
    QuadraticSyntheticEDREvaluator,
    StimSurfaceCodeEDREvaluator,
)
from hdfa_rl_suite.google_pure_source_exact.control_normalization.normalized_coordinates import (
    CoordinateVector,
    EmpiricalCoordinateSystem,
    EmpiricallyNormalizedGaussianPolicy,
    legacy_algebraic_ablation_label,
)
from hdfa_rl_suite.google_pure_source_exact.control_normalization.perturbation_sweeps import (
    merge_sweep_shards,
    run_control_type_sweep,
    shard_sigmas,
)
from hdfa_rl_suite.google_pure_source_exact.control_normalization.quadratic_fit import (
    FitRejected,
    fit_all_sweeps,
    fit_detector_sensitivity,
)
from hdfa_rl_suite.google_pure_source_exact.control_normalization.validation import (
    audit_no_arbitrary_scale,
    build_calibration_bundle_from_sweeps,
    validate_covariance_damage_prediction,
    validate_normalized_isotropy,
)


def make_spec(name: str = "xy", *, unit: str = "native", scale: float = 1.0,
              channel: str = "after_clifford_depolarization") -> ControlTypeSpec:
    sigmas = tuple(scale * value for value in (0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0))
    return ControlTypeSpec(
        control_type=name,
        gate_ids=tuple(f"{name}_gate_{index}" for index in range(6)),
        native_unit=unit,
        reference_value_native=3.0 * scale,
        sweep_sigmas_native=sigmas,
        fit_interval_native=(0.0, 2.0 * scale),
        stim_error_channel=channel,
        synthetic_probability_gain=0.001 / (scale * scale),
        source_identifiability=SourceIdentifiability.SOURCE_UNSPECIFIED_PREREGISTERED,
    )


def protocol(seed: int = 101, *, shots: int = 20000) -> SweepProtocol:
    return SweepProtocol(
        candidates_per_sigma=6,
        shots_per_candidate=shots,
        qec_rounds_per_shot=3,
        perturbation_seed=seed,
        detector_seed=seed + 1,
    )


def relaxed_rules() -> FitRules:
    return FitRules(
        minimum_r_squared=0.75,
        maximum_monotonicity_z=4.0,
        maximum_quartic_z=4.0,
        maximum_reduced_chi_squared=8.0,
        minimum_positive_coefficient_z=2.0,
        stability_relative_tolerance=0.3,
        isotropy_relative_tolerance=0.25,
        prediction_relative_tolerance=0.25,
    )


def calibrate(specs: tuple[ControlTypeSpec, ...], true_sigma0: dict[str, float], *, seed: int = 101,
              shots: int = 20000, rules: FitRules | None = None,
              evaluator: QuadraticSyntheticEDREvaluator | None = None):
    rules = rules or relaxed_rules()
    evaluator = evaluator or QuadraticSyntheticEDREvaluator(specs, true_sigma0)
    sweep_protocol = protocol(seed, shots=shots)
    sweeps = tuple(run_control_type_sweep(evaluator, spec, sweep_protocol) for spec in specs)
    fits = fit_all_sweeps(sweeps, rules)
    source = build_source_contract()
    bundle = build_calibration_bundle_from_sweeps(
        specs, sweeps, fits, rules,
        config_hash=canonical_hash({"test": seed}),
        source_contract_hash=source["source_contract_hash"],
    )
    return evaluator, sweeps, fits, bundle


def test_exact_synthetic_recovers_known_sigma0_and_all_gates_are_perturbed() -> None:
    spec = make_spec()
    _, sweeps, fits, _ = calibrate((spec,), {"xy": 2.0}, shots=50000)
    assert fits[0].sigma0_native == pytest.approx(2.0, rel=0.04)
    assert sweeps[0].simultaneous_gate_ids == spec.gate_ids
    assert fits[0].coefficient_confidence_interval_95[0] > 0
    assert fits[0].shot_budget > 0


def test_finite_shot_sigma0_interval_has_reasonable_coverage() -> None:
    spec = make_spec()
    rules = FitRules(
        minimum_r_squared=0.5,
        maximum_monotonicity_z=100.0,
        maximum_quartic_z=100.0,
        maximum_reduced_chi_squared=100.0,
        minimum_positive_coefficient_z=1.0,
        stability_relative_tolerance=0.5,
        isotropy_relative_tolerance=0.5,
        prediction_relative_tolerance=0.5,
    )
    covered = 0
    accepted = 0
    for seed in range(30):
        evaluator = QuadraticSyntheticEDREvaluator((spec,), {"xy": 2.0})
        sweep = run_control_type_sweep(evaluator, spec, protocol(1000 + seed, shots=1200))
        try:
            fit = fit_detector_sensitivity(sweep, rules)
        except FitRejected:
            continue
        accepted += 1
        low, high = fit.sigma0_confidence_interval_95
        covered += int(low <= 2.0 <= high)
    assert accepted >= 27
    assert covered / accepted >= 0.80


def test_native_unit_reexpression_leaves_normalized_coordinates_invariant() -> None:
    original = make_spec(scale=1.0, unit="GHz")
    reexpressed = make_spec(scale=1000.0, unit="MHz")
    _, _, _, first_bundle = calibrate((original,), {"xy": 2.0}, seed=201)
    _, _, _, second_bundle = calibrate((reexpressed,), {"xy": 2000.0}, seed=201)
    first = EmpiricalCoordinateSystem(first_bundle)
    second = EmpiricalCoordinateSystem(second_bundle)
    first_native = CoordinateVector(
        first.reference_native + 0.4 * first.native_per_normalized,
        first.parameter_ids, "native")
    second_native = CoordinateVector(
        second.reference_native + 0.4 * second.native_per_normalized,
        second.parameter_ids, "native")
    assert np.allclose(first.to_normalized(first_native).values, 0.4, atol=1e-12)
    assert np.allclose(second.to_normalized(second_native).values, 0.4, atol=1e-12)
    assert second_bundle.fits[0].sigma0_native / first_bundle.fits[0].sigma0_native == pytest.approx(1000, rel=0.01)


@pytest.mark.parametrize(
    "evaluator",
    [
        lambda spec: QuadraticSyntheticEDREvaluator(
            (spec,), {"xy": 2.0}, quartic_by_type={"xy": 3.0}),
        lambda spec: QuadraticSyntheticEDREvaluator(
            (spec,), {"xy": 2.0}, linear_variance_by_type={"xy": -0.22}),
    ],
)
def test_nonquadratic_or_nonmonotonic_sweeps_fail_closed(evaluator) -> None:
    spec = make_spec()
    sweep = run_control_type_sweep(evaluator(spec), spec, protocol(301, shots=50000))
    with pytest.raises(FitRejected):
        fit_detector_sensitivity(sweep, relaxed_rules())


def test_calibration_invalidates_on_circuit_or_detector_hash_change() -> None:
    spec = make_spec()
    _, _, _, bundle = calibrate((spec,), {"xy": 2.0}, seed=401)
    bundle.validate_context(
        circuit_hash=bundle.reference.circuit_hash,
        detector_set_hash=bundle.reference.detector_set_hash)
    with pytest.raises(RuntimeError, match="circuit"):
        bundle.validate_context(
            circuit_hash="changed", detector_set_hash=bundle.reference.detector_set_hash)
    with pytest.raises(RuntimeError, match="detector"):
        bundle.validate_context(
            circuit_hash=bundle.reference.circuit_hash, detector_set_hash="changed")


def test_equal_normalized_perturbations_have_equal_detector_damage() -> None:
    a = make_spec("a", scale=1.0, channel="after_clifford_depolarization")
    b = make_spec("b", scale=4.0, channel="before_measure_flip_probability")
    evaluator, _, _, bundle = calibrate((a, b), {"a": 2.0, "b": 8.0}, seed=501)
    result = validate_normalized_isotropy(
        evaluator, bundle, normalized_sigma=1.0,
        candidates=10, shots_per_candidate=30000,
        perturbation_seed=502, detector_seed=503)
    assert result["passed"]
    assert not result["unit_dominated_damage"]
    assert result["maximum_single_type_damage_share"] < result["unit_dominance_threshold"]
    increases = [row["measured_edr_increase_percentage_points"] for row in result["rows"]]
    assert max(increases) - min(increases) < 0.12


def test_hessian_covariance_prediction_matches_direct_monte_carlo() -> None:
    a = make_spec("a")
    b = make_spec("b", scale=2.0, channel="before_measure_flip_probability")
    evaluator, _, _, bundle = calibrate((a, b), {"a": 2.0, "b": 4.0}, seed=601)
    result = validate_covariance_damage_prediction(
        evaluator, bundle, normalized_sigma=0.5,
        candidates=12, shots_per_candidate=30000,
        perturbation_seed=602, detector_seed=603)
    assert result["passed"]
    assert result["predicted_damage_percentage_points"] == pytest.approx(0.5, rel=1e-12)


def test_sensitivity_cannot_be_applied_twice_and_policy_uses_empirical_bundle() -> None:
    spec = make_spec()
    _, _, _, bundle = calibrate((spec,), {"xy": 2.0}, seed=701)
    coordinates = EmpiricalCoordinateSystem(bundle)
    normalized = CoordinateVector(np.zeros(6), coordinates.parameter_ids, "normalized")
    native = coordinates.to_native(normalized)
    with pytest.raises(ValueError, match="twice"):
        coordinates.to_native(native)
    policy = EmpiricallyNormalizedGaussianPolicy(coordinates, seed=702)
    batch = policy.sample(4, np.zeros(6), np.ones(6) * 0.1)
    assert batch.sensitivity_application_count == 1
    assert batch.calibration_bundle_hash == bundle.bundle_hash
    assert np.allclose(
        batch.applied_native,
        coordinates.reference_native[None, :] + batch.latent_normalized * coordinates.native_per_normalized[None, :])


def test_source_exact_config_rejects_old_arbitrary_scale_fields_and_labels_ablation() -> None:
    assert audit_no_arbitrary_scale({"normalization_method": "empirical"})["passed"]
    rejected = audit_no_arbitrary_scale({"native_per_normalized": [1.0]})
    assert not rejected["passed"]
    assert legacy_algebraic_ablation_label() == NON_PAPER_NORMALIZATION_ABLATION


def test_stim_evaluator_counts_real_detector_samples() -> None:
    pytest.importorskip("stim")
    spec = ControlTypeSpec(
        control_type="xy",
        gate_ids=("g0", "g1", "g2", "g3"),
        native_unit="relative_amplitude",
        reference_value_native=1.0,
        sweep_sigmas_native=(0.0, 0.005, 0.01, 0.015),
        fit_interval_native=(0.0, 0.015),
        stim_error_channel="after_clifford_depolarization",
        synthetic_probability_gain=4.0,
        source_identifiability=SourceIdentifiability.SOURCE_UNSPECIFIED_PREREGISTERED,
    )
    evaluator = StimSurfaceCodeEDREvaluator((spec,), distance=3, rounds=3)
    measured = evaluator.measure_joint(
        {"xy": 0.01}, candidates=3, shots_per_candidate=64,
        perturbation_seed=801, detector_seed=802)
    assert measured.detector_opportunities == 3 * 64 * evaluator.detector_count
    assert 0 <= measured.detector_events <= measured.detector_opportunities
    assert measured.qec_cycles == 3 * 64 * 3
    assert evaluator.reference.circuit_hash
    assert evaluator.plant_hash


def test_shard_resume_is_exact_and_duplicate_merge_is_rejected(tmp_path) -> None:
    spec = make_spec()
    evaluator = QuadraticSyntheticEDREvaluator((spec,), {"xy": 2.0})
    sweep_protocol = protocol(901, shots=800)
    shards = []
    for index in range(2):
        checkpoint = tmp_path / f"shard-{index}.json"
        sigmas = shard_sigmas(spec.sweep_sigmas_native, index, 2)
        first = run_control_type_sweep(
            evaluator, spec, sweep_protocol, sigmas_native=sigmas,
            checkpoint_path=checkpoint, shard_index=index, shard_count=2)
        resumed = run_control_type_sweep(
            evaluator, spec, sweep_protocol, sigmas_native=sigmas,
            checkpoint_path=checkpoint, resume=True,
            shard_index=index, shard_count=2)
        assert resumed.to_dict() == first.to_dict()
        assert sum(point.qec_cycles for point in resumed.points) == (
            len(sigmas) * sweep_protocol.qec_cycles_per_sigma)
        shards.append(resumed)
    merged = merge_sweep_shards(shards, spec)
    assert [point.sigma_native for point in merged.points] == list(spec.sweep_sigmas_native)
    with pytest.raises(RuntimeError, match="duplicate shard"):
        merge_sweep_shards([shards[0], shards[0]], spec)
