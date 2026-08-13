from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hdfa_rl_suite.google_pure_source_exact.figure5a.acquisition import (
    _freeze_batch,
    run_cell,
    source_controls_for_epoch,
    substitution_identity,
)
from hdfa_rl_suite.google_pure_source_exact.figure5a.bounded_action_ablation import (
    Figure5aBoundedActionAblation,
)
from hdfa_rl_suite.google_pure_source_exact.figure5a.cli import plan
from hdfa_rl_suite.google_pure_source_exact.figure5a.contracts import (
    AcquisitionMode,
    Figure5aProtocol,
    SOURCE_CANDIDATES_PER_EPOCH,
    SOURCE_CANDIDATE_QEC_CYCLES,
    SOURCE_ENTROPY_ANCHORS,
    ratio_from_raw_counts,
)
from hdfa_rl_suite.google_pure_source_exact.figure5a.entropy_scan import (
    build_conditions,
    classify_anchor_rows,
    reduce_dense_rows,
    scan_contract,
)
from hdfa_rl_suite.google_pure_source_exact.figure5a.gradient_stability import (
    gradient_stability_conditions,
    plan_gradient_stability,
)
from hdfa_rl_suite.google_pure_source_exact.figure5a.plant import Figure5aStimPlant
from hdfa_rl_suite.google_pure_source_exact.figure5a.normalization import (
    Figure5aEmpiricalBoundary,
    reward_representation_hash,
)
from hdfa_rl_suite.google_pure_source_exact.figure5a.round_invariance import plan_round_invariance
from hdfa_rl_suite.google_pure_source_exact.figure5a.validation import (
    build_plant,
    detector_equivalence_response_audit,
)
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.gaussian import DirectSigmaGaussianPolicy
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.losses import total_loss_and_gradients
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.optimizer import (
    DirectSigmaOptimizer,
    OptimizerConfig,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/google_pure_source_exact/figure5a.json"


@pytest.fixture(scope="module")
def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def plant(config) -> Figure5aStimPlant:
    return build_plant(config)


@pytest.fixture(scope="module")
def normalization_artifact(config) -> dict:
    path = ROOT / config["ablations"]["empirical_relative_normalization_bundle"]
    return json.loads(path.read_text(encoding="utf-8"))


def test_exact_source_budget_arithmetic_and_reference_gate() -> None:
    protocol = Figure5aProtocol(AcquisitionMode.REFERENCE, 1000, 50, 36000, 3)
    assert protocol.candidate_qec_cycles == SOURCE_CANDIDATE_QEC_CYCLES
    assert protocol.four_stream_qec_cycles == 7_200_000_000
    assert protocol.shots_per_policy == 12000
    protocol.assert_reference()
    with pytest.raises(ValueError, match="exactly 1000"):
        Figure5aProtocol(AcquisitionMode.REFERENCE, 999, 50, 36000, 3)
    with pytest.raises(ValueError, match="cannot be watermarked"):
        Figure5aProtocol(AcquisitionMode.VALIDATION, 1000, 50, 36000, 3)


def test_raw_count_ratio_and_substitution_identities() -> None:
    ratio = ratio_from_raw_counts(150, 200, 100)
    assert ratio == {"source_ratio": 0.5, "positive_cost_ratio": 0.5}
    assert substitution_identity({"fixed": 200, "optimal": 100}) == {
        "fixed_substitution": 0.0, "optimal_substitution": 1.0}


def test_distance3_inventory_is_exact_and_stim_derived(plant) -> None:
    assert plant.control_count == 41
    assert sum(row.gate_type == "single_qubit" for row in plant.inventory) == 17
    assert sum(row.gate_type == "two_qubit" for row in plant.inventory) == 24
    assert plant.mask.shape == (plant.detector_count, 41)
    assert plant.mask.any(axis=0).all() and plant.mask.any(axis=1).all()
    assert 0 < plant.mask.mean() < 1
    assert all(row.circuit_locations and row.detectors_influenced for row in plant.inventory)


def test_reward_components_are_time_translation_reduced_and_round_invariant(config, plant) -> None:
    assert plant.rounds == 25
    assert plant.raw_detector_count == 200
    assert plant.detector_count == 24
    assert plant.detector_count < plant.raw_detector_count
    assert sorted(raw for group in plant.reward_component_raw_detectors for raw in group) == \
        list(range(plant.raw_detector_count))
    longer = json.loads(json.dumps(config))
    longer["plant"]["circuit_rounds"] = 50
    fifty_round_plant = build_plant(longer)
    assert fifty_round_plant.raw_detector_count == 400
    assert fifty_round_plant.detector_count == plant.detector_count
    np.testing.assert_array_equal(fifty_round_plant.mask, plant.mask)
    np.testing.assert_array_equal(
        fifty_round_plant.mask.sum(axis=0), plant.mask.sum(axis=0))


def test_every_time_equivalence_class_has_equal_exact_marginal_response(plant) -> None:
    result = detector_equivalence_response_audit(plant, seed=8021, random_policies=7)
    assert result["multi_detector_class_count"] == 8
    assert result["pass"], result["failures"]
    assert result["maximum_within_class_marginal_spread"] < 2e-15


def test_empirical_normalization_is_plant_and_reward_bound_without_hidden_point01(
        plant, normalization_artifact) -> None:
    artifact = normalization_artifact
    boundary = Figure5aEmpiricalBoundary.from_artifact(plant, artifact)
    assert artifact["plant_hash"] == plant.plant_hash
    assert artifact["scientific_status"] == "EMPIRICAL_RELATIVE_NORMALIZATION_ABLATION"
    assert not artifact["canonical_figure5a_execution"]
    assert artifact["reward_representation_hash"] == reward_representation_hash(plant)
    assert artifact["edr_unit"] == "fraction"
    assert artifact["source_literal_target_edr_increase_fraction"] == 1.0
    assert not artifact["percentage_point_conversion_applied"]
    assert not artifact["analytic_omega_times_degree_shortcut_used"]
    assert not artifact["absolute_source_scale_identifiable"]
    assert not artifact["source_literal_scale_safe_for_published_unit_amplitude_drift"]
    assert np.exp(np.mean(np.log(boundary.native_scale))) == pytest.approx(1.0)
    conditioned = []
    for group in artifact["control_groups"]:
        index = group["control_indices"][0]
        conditioned.append(group["quadratic_coefficient_per_native_squared"] *
                           boundary.native_scale[index] ** 2)
    assert conditioned[0] == pytest.approx(conditioned[1], rel=1e-12)


def test_shared_optimum_quadratic_error_and_physical_probabilities(plant) -> None:
    assert np.array_equal(plant.optimum(0, 1 / 1000), np.zeros(41))
    assert np.unique(plant.optimum(137, 1 / 1000)).size == 1
    baseline = plant.probabilities(np.zeros(41), 0, 1 / 1000)
    for index, item in enumerate(plant.inventory):
        half = np.zeros(41); half[index] = 0.5
        full = np.zeros(41); full[index] = 1.0
        assert np.isclose(plant.probabilities(half, 0, 1 / 1000)[index] - baseline[index],
                          0.25 * item.omega_sensitivity)
        assert np.isclose(plant.probabilities(full, 0, 1 / 1000)[index] - baseline[index],
                          item.omega_sensitivity)
    assert np.all(plant.probabilities(np.full(41, 2.0), 0, 1 / 1000) <
                  plant.probability_ceilings)


def test_bounded_transform_is_retained_only_as_a_separate_ablation(plant) -> None:
    bounded = Figure5aBoundedActionAblation(plant)
    latent = np.linspace(-2.75, 2.75, 41)
    applied = bounded.apply_control_transform(latent)
    np.testing.assert_allclose(
        bounded.latent_controls_for(applied), latent, rtol=1e-12, atol=1e-12)
    assert np.all(np.abs(applied) < bounded.control_limits)
    assert np.all(bounded.control_limits > 1.0)
    assert not hasattr(plant, "control_limits")
    assert not hasattr(plant, "apply_control_transform")

    positive_extreme = bounded.apply_control_transform(np.full(41, 1e6))
    negative_extreme = bounded.apply_control_transform(np.full(41, -1e6))
    assert np.all(plant.probabilities(positive_extreme, 750, 1 / 1000) <
                  plant.probability_ceilings)
    assert np.all(plant.probabilities(negative_extreme, 250, 1 / 1000) <
                  plant.probability_ceilings)


def test_canonical_plant_constructs_without_a_global_bounded_action_domain(config) -> None:
    high_curvature = json.loads(json.dumps(config))
    high_curvature["plant"]["one_qubit_omega"] = [0.2, 0.3]
    high_curvature["plant"]["two_qubit_omega"] = [0.2, 0.3]
    candidate = build_plant(high_curvature)
    assert candidate.control_count == 41
    with pytest.raises(ValueError, match="bounded-action ablation"):
        Figure5aBoundedActionAblation(candidate)
    with pytest.raises(ValueError, match="left the frozen physical range"):
        candidate.probabilities(np.full(41, 2.0), 0, 1 / 1000)


def test_canonical_acquisition_uses_literal_source_optimum_and_gaussian_actions(plant) -> None:
    policy = DirectSigmaGaussianPolicy(np.linspace(-0.2, 0.2, 41), np.full(41, 0.15), seed=8111)
    batch = _freeze_batch(policy, 6)
    np.testing.assert_array_equal(batch["applied_actions"], batch["gaussian_actions"])
    np.testing.assert_array_equal(batch["applied_behavior_mean"], batch["gaussian_behavior_mean"])
    target, controls = source_controls_for_epoch(
        plant, epoch=250, frequency=1 / 1000,
        stochastic=np.asarray(batch["applied_actions"])[0],
        learned_mean=np.asarray(batch["applied_behavior_mean"]))
    np.testing.assert_array_equal(target, np.ones(41))
    np.testing.assert_array_equal(controls["optimal"], target)
    np.testing.assert_array_equal(controls["stochastic"], np.asarray(batch["gaussian_actions"])[0])


def test_configured_omega_is_curvature_in_the_same_applied_policy_coordinates(plant) -> None:
    rng = np.random.default_rng(8112)
    target = rng.normal(0.0, 0.2, 41)
    baseline = plant.probabilities(target, 0, 1 / 1000, target_controls=target)
    delta = 1e-3
    for index, item in enumerate(plant.inventory):
        displaced = target.copy(); displaced[index] += delta
        measured = (plant.probabilities(
            displaced, 0, 1 / 1000, target_controls=target)[index] - baseline[index]) / delta**2
        assert measured == pytest.approx(item.omega_sensitivity, rel=2e-10, abs=1e-12)


def test_detector_sampling_is_deterministic_and_local(plant) -> None:
    controls = np.zeros(41)
    first = plant.sample_detector_counts(controls, epoch=0, frequency=1 / 1000, qec_cycles=300,
                                         seed=plant.stream_seed(7, "test", 0, 0))
    second = plant.sample_detector_counts(controls, epoch=0, frequency=1 / 1000, qec_cycles=300,
                                          seed=plant.stream_seed(7, "test", 0, 0))
    assert np.array_equal(first, second)
    assert first.shape == (plant.detector_count,)
    observation = plant.sample_detector_observation(
        controls, epoch=0, frequency=1 / 1000, qec_cycles=300,
        seed=plant.stream_seed(7, "test", 0, 0))
    assert observation.raw_counts.shape == (plant.raw_detector_count,)
    assert observation.reward_component_counts.shape == (plant.detector_count,)
    assert observation.raw_total == int(observation.raw_counts.sum())


def test_entropy_is_policy_level_and_not_multiplied_by_detector_degree(plant) -> None:
    mean, sigma = np.zeros(41), np.full(41, 0.2)
    batch = DirectSigmaGaussianPolicy(mean, sigma, seed=8).sample(5)
    rewards = np.zeros((5, plant.detector_count))
    baseline = np.zeros(plant.detector_count)
    sparse = total_loss_and_gradients(batch.actions, rewards, plant.mask, mean, sigma, baseline,
                                      batch.behavior, clip=0.2, policy_weight=0,
                                      baseline_weight=0, entropy_weight=0.1)
    dense = total_loss_and_gradients(batch.actions, rewards, np.ones_like(plant.mask), mean, sigma,
                                     baseline, batch.behavior, clip=0.2, policy_weight=0,
                                     baseline_weight=0, entropy_weight=0.1)
    np.testing.assert_array_equal(sparse.grad_sigma, dense.grad_sigma)
    np.testing.assert_allclose(sparse.grad_sigma, -0.1 / sigma)


def test_candidate_boundary_resume_is_bit_exact_and_drops_nothing(tmp_path, plant) -> None:
    protocol = Figure5aProtocol(AcquisitionMode.SMOKE, 2, 3, 50, plant.rounds)
    optimizer = OptimizerConfig(0.08, 0.02, 0.08, minimum_sigma=0.002, maximum_sigma=0.8)
    common = dict(protocol=protocol, plant=plant, frequency=0.1, entropy_weight=0.01,
                  seed=99, optimizer_config=optimizer, initial_sigma=0.15,
                  dependency_hashes={"test": "frozen"}, controller_hash="test-controller")
    mono = run_cell(**common, checkpoint_path=tmp_path / "mono.json")
    partial = run_cell(**common, checkpoint_path=tmp_path / "resume.json", max_candidate_boundaries=4)
    assert not partial["complete"] and partial["next_candidate"] == 1
    changed = {**common, "controller_hash": "changed-controller"}
    with pytest.raises(RuntimeError, match="checkpoint identity changed"):
        run_cell(**changed, checkpoint_path=tmp_path / "resume.json", resume=True)
    resumed = run_cell(**common, checkpoint_path=tmp_path / "resume.json", resume=True)
    boundary = run_cell(**common, checkpoint_path=tmp_path / "last-boundary.json",
                        max_candidate_boundaries=3)
    assert not boundary["complete"] and boundary["next_candidate"] == 3
    boundary_resumed = run_cell(
        **common, checkpoint_path=tmp_path / "last-boundary.json", resume=True)
    assert resumed["stream_totals"] == mono["stream_totals"]
    assert resumed["epoch_records"] == mono["epoch_records"]
    assert boundary_resumed["epoch_records"] == mono["epoch_records"]
    assert resumed["no_candidates_dropped"]
    assert resumed["candidate_boundaries_completed"] == 6
    assert resumed["schema_version"] == "figure5a-cell.v6"
    assert resumed["raw_detector_count"] > resumed["detector_count"]
    assert resumed["reward_representation"] == "time_translation_equivalence_class_mean_edr"
    assert resumed["gradient_clipping_contract"]["applied_before_momentum"]
    assert resumed["action_execution"] == "identity_applied_gaussian"
    assert resumed["likelihood_space"] == "applied_gaussian"
    assert resumed["entropy_space"] == "applied_gaussian"
    assert not resumed["action_transform_applied"]
    assert not resumed["empirical_relative_normalization_applied"]
    assert not resumed["mean_bounds_applied"]
    assert not resumed["action_transform_uses_hidden_optimum"]
    assert all(len(record["counts"]["stochastic"]) == 3
               for record in resumed["epoch_records"])
    assert all(len(record["counts"][stream]) == 1 for record in resumed["epoch_records"]
               for stream in ("fixed", "optimal", "learned_mean"))
    assert resumed["circuit_compilations"] == 2 * (3 + 3)
    assert resumed["stream_acquisition_contract"]["mode"] == \
        "figure5a-finite-shot-epoch-aggregate.v1"
    assert resumed["stream_acquisition_contract"]["all_four_stream_qec_budgets_unchanged"]
    assert resumed["stream_acquisition_contract"]["stochastic_training_seed_contract_unchanged"]
    assert not resumed["stream_acquisition_contract"]["exact_DEM_diagnostics_used"]
    assert all(record["stream_acquisition"]["total_circuit_compilations"] == 6
               for record in resumed["epoch_records"])
    assert all(record["action_execution"] == "identity_applied_gaussian"
               and record["likelihood_space"] == "applied_gaussian"
               and record["maximum_abs_gaussian_applied_delta"] == 0.0
               and record["source_optimum_applied_directly"]
               for record in resumed["epoch_records"])
    assert all(record["optimum"] == plant.optimum(record["epoch"], 0.1)[0]
               for record in resumed["epoch_records"])
    assert all(record["gradient_clipping"]["gradient_clipping_mode"] == "none"
               for record in resumed["epoch_records"])
    assert all(set(("fraction_at_sigma_min", "fraction_at_sigma_max",
                    "unclipped_sigma_min", "unclipped_sigma_max")) <= set(record)
               for record in resumed["epoch_records"])

    obsolete = json.loads((tmp_path / "mono.json").read_text(encoding="utf-8"))
    obsolete["schema_version"] = "figure5a-cell-checkpoint.v4"
    (tmp_path / "obsolete.json").write_text(json.dumps(obsolete), encoding="utf-8")
    with pytest.raises(RuntimeError, match="acquisition layout is obsolete"):
        run_cell(**common, checkpoint_path=tmp_path / "obsolete.json", resume=True)


def test_epoch_constant_streams_are_aggregated_without_changing_qec_budgets(
        tmp_path, plant, monkeypatch) -> None:
    protocol = Figure5aProtocol(AcquisitionMode.SMOKE, 1, 3, 50, plant.rounds)
    optimizer = OptimizerConfig(0.08, 0.02, 0.08, minimum_sigma=0.002, maximum_sigma=0.8)
    calls: list[int] = []
    original = plant.sample_detector_observation

    def recording_sample(*args, **kwargs):
        calls.append(int(kwargs["qec_cycles"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(plant, "sample_detector_observation", recording_sample)
    result = run_cell(
        protocol=protocol, plant=plant, frequency=0.1, entropy_weight=0.01,
        seed=102, optimizer_config=optimizer, initial_sigma=0.15,
        dependency_hashes={"test": "frozen"}, controller_hash="test-controller",
        checkpoint_path=tmp_path / "aggregated.json")
    assert calls.count(50) == 3
    assert calls.count(150) == 3
    assert len(calls) == 6
    assert sum(calls) == protocol.four_stream_qec_cycles
    assert result["four_stream_qec_cycles"] == 4 * 3 * 50
    assert result["circuit_compilations"] == 6


def test_batched_checkpoint_flush_is_bit_exact(tmp_path, plant) -> None:
    protocol = Figure5aProtocol(AcquisitionMode.SMOKE, 2, 3, 50, plant.rounds)
    optimizer = OptimizerConfig(0.08, 0.02, 0.08, minimum_sigma=0.002, maximum_sigma=0.8)
    common = dict(protocol=protocol, plant=plant, frequency=0.1, entropy_weight=0.01,
                  seed=101, optimizer_config=optimizer, initial_sigma=0.15,
                  dependency_hashes={"test": "frozen"}, controller_hash="test-controller")
    mono = run_cell(**common, checkpoint_path=tmp_path / "mono-batched.json",
                    checkpoint_every_candidates=3)
    partial = run_cell(**common, checkpoint_path=tmp_path / "resume-batched.json",
                       checkpoint_every_candidates=3, max_candidate_boundaries=2)
    assert not partial["complete"] and partial["next_candidate"] == 2
    resumed = run_cell(**common, checkpoint_path=tmp_path / "resume-batched.json", resume=True,
                       checkpoint_every_candidates=3)
    assert resumed["checkpoint_every_candidates"] == 3
    assert resumed["stream_totals"] == mono["stream_totals"]
    assert resumed["epoch_records"] == mono["epoch_records"]


def test_source_entropy_anchors_and_dense_scan_share_frozen_contract(config, plant) -> None:
    conditions = build_conditions(config, mode=AcquisitionMode.SMOKE, scan="anchors")
    assert tuple(sorted({row["entropy_weight"] for row in conditions})) == SOURCE_ENTROPY_ANCHORS
    protocol = Figure5aProtocol(AcquisitionMode.SMOKE, 2, 3, 50, plant.rounds)
    anchors = scan_contract(config, mode=AcquisitionMode.SMOKE, scan="anchors", protocol=protocol,
                            plant_hash=plant.plant_hash, controller_hash="controller")
    dense = scan_contract(config, mode=AcquisitionMode.SMOKE, scan="dense", protocol=protocol,
                          plant_hash=plant.plant_hash, controller_hash="controller")
    assert anchors["plant_hash"] == dense["plant_hash"]
    assert anchors["controller_hash"] == dense["controller_hash"]
    assert anchors["validation_watermark"] and dense["validation_watermark"]


def test_dynamic_validation_spans_a_quarter_drift_cycle_without_using_certification_budget(
        config, plant) -> None:
    profile = config["profiles"][AcquisitionMode.DYNAMIC_VALIDATION.value]
    protocol = Figure5aProtocol(
        AcquisitionMode.DYNAMIC_VALIDATION, profile["epochs"],
        profile["candidates_per_epoch"], profile["qec_cycles_per_candidate"],
        plant.rounds)
    assert protocol.epochs >= 250
    assert protocol.candidates_per_epoch < SOURCE_CANDIDATES_PER_EPOCH
    assert protocol.qec_cycles_per_candidate < 36_000
    assert plant.optimum(protocol.epochs, 1 / 1000)[0] == pytest.approx(1.0)
    conditions = build_conditions(
        config, mode=AcquisitionMode.DYNAMIC_VALIDATION, scan="anchors")
    assert {row["entropy_weight"] for row in conditions} == set(SOURCE_ENTROPY_ANCHORS)
    assert len({row["seed"] for row in conditions}) == 3


def test_anchor_classification_is_quantitative_and_seed_stable() -> None:
    rows = []
    for seed in (1, 2):
        for weight, ent, stochastic, learned in ((0.001, -5.0, 0.1, 0.2),
                                                  (0.01, -3.0, 0.8, 0.85),
                                                  (0.1, -1.0, 0.3, 0.9)):
            rows.append({"seed": seed, "entropy_weight": weight,
                         "epoch_records": [{"policy_entropy": ent,
                                            "fraction_at_sigma_max": 0.0}],
                         "stochastic_ratio": {"source_ratio": stochastic},
                         "learned_mean_ratio": {"source_ratio": learned}})
    result = classify_anchor_rows(rows, {"minimum_entropy_separation": 0.05,
        "minimum_high_exploration_gap": 0.02, "minimum_low_tracking_gap": 0.02,
        "middle_must_maximize_stochastic_ratio": True,
        "minimum_middle_stochastic_ratio": 0.5,
        "minimum_high_learned_mean_ratio": 0.0,
        "maximum_sigma_cap_fraction": 0.01})
    assert result["anchor_classification_pass"] and result["stable_over_seeds"]


def test_anchor_classification_fails_closed_on_zero_denominator() -> None:
    rows = []
    for weight in SOURCE_ENTROPY_ANCHORS:
        rows.append({"seed": 1, "entropy_weight": weight,
                     "epoch_records": [{"policy_entropy": float(weight),
                                        "fraction_at_sigma_max": 0.0}],
                     "stochastic_ratio": {"source_ratio": None},
                     "learned_mean_ratio": {"source_ratio": None}})
    result = classify_anchor_rows(rows, {"minimum_entropy_separation": 0.05,
        "minimum_high_exploration_gap": 0.02, "minimum_low_tracking_gap": 0.02,
        "middle_must_maximize_stochastic_ratio": True,
        "minimum_middle_stochastic_ratio": 0.5,
        "minimum_high_learned_mean_ratio": 0.0,
        "maximum_sigma_cap_fraction": 0.01})
    assert not result["anchor_classification_pass"]
    assert "zero finite-shot" in result["seed_rows"][0]["blocking_reasons"][0]


def test_anchor_absolute_gates_reject_ordered_but_objectively_bad_controller(config) -> None:
    rows = []
    for weight, entropy, stochastic, learned in (
            (0.001, -5.0, -2.0, -1.5),
            (0.01, -3.0, -1.0, -0.8),
            (0.1, -1.0, -1.5, 0.1)):
        rows.append({
            "seed": 1, "entropy_weight": weight,
            "epoch_records": [{"policy_entropy": entropy, "fraction_at_sigma_max": 0.0}],
            "stochastic_ratio": {"source_ratio": stochastic},
            "learned_mean_ratio": {"source_ratio": learned},
        })
    result = classify_anchor_rows(rows, config["anchor"]["classification"])
    assert not result["anchor_classification_pass"]
    assert "middle_stochastic_absolute_advantage" in \
        result["seed_rows"][0]["blocking_reasons"]


def test_anchor_sigma_cap_contact_fails_closed(config) -> None:
    rows = []
    for weight, entropy, stochastic, learned in (
            (0.001, -5.0, 0.1, 0.2),
            (0.01, -3.0, 0.8, 0.85),
            (0.1, -1.0, 0.3, 0.9)):
        rows.append({
            "seed": 1, "entropy_weight": weight,
            "epoch_records": [{"policy_entropy": entropy,
                               "fraction_at_sigma_max": 0.05 if weight == 0.1 else 0.0}],
            "stochastic_ratio": {"source_ratio": stochastic},
            "learned_mean_ratio": {"source_ratio": learned},
        })
    result = classify_anchor_rows(rows, config["anchor"]["classification"])
    assert not result["anchor_classification_pass"]
    assert "sigma_cap_not_dominant" in result["seed_rows"][0]["blocking_reasons"]


def test_dense_reducer_extracts_rmax_surface_and_zero_crossing(config) -> None:
    rows = []
    values = {
        0.001: {0.001: 0.4, 0.01: 0.7},
        0.006: {0.001: 0.1, 0.01: 0.2},
        0.010: {0.001: -0.2, 0.01: -0.1},
    }
    for frequency, entropy_rows in values.items():
        for entropy_weight, stochastic in entropy_rows.items():
            for seed in (1, 2):
                rows.append({
                    "frequency": frequency, "entropy_weight": entropy_weight, "seed": seed,
                    "stochastic_ratio": {"source_ratio": stochastic},
                    "learned_mean_ratio": {"source_ratio": stochastic + 0.1},
                })
    result = reduce_dense_rows(rows, config["dense_scan"]["classification"])
    assert result["dense_acceptance_pass"]
    assert [row["r_max"] for row in result["stochastic_r_max_envelope"]] == [0.7, 0.2, -0.1]
    assert result["crossing_bracket_frequency_per_epoch"] == [0.006, 0.01]
    assert result["estimated_steerability_threshold_frequency_per_epoch"] == \
        pytest.approx(0.008666666666666666)


def test_optimizer_reports_both_sigma_bounds_and_unclipped_extrema() -> None:
    mean = np.zeros(2)
    sigma = np.asarray([0.79, 0.01])
    baseline = np.zeros(1)
    optimizer = DirectSigmaOptimizer(
        2, 1, OptimizerConfig(0.1, 0.1, 0.1, minimum_sigma=0.002, maximum_sigma=0.8))
    update = optimizer.step(
        mean, sigma, baseline, np.zeros(2), np.asarray([-1.0, 1.0]), np.zeros(1))
    assert update["unclipped_sigma_max"] == pytest.approx(0.89)
    assert update["unclipped_sigma_min"] == pytest.approx(-0.09)
    assert update["fraction_at_sigma_max"] == 0.5
    assert update["fraction_at_sigma_min"] == 0.5


def test_reference_planner_reports_exact_cost(config, tmp_path) -> None:
    result = plan(CONFIG_PATH, tmp_path, mode=AcquisitionMode.REFERENCE, scan="anchors")
    assert result["condition_count"] == 9
    assert result["candidate_training_qec_cycles"] == 9 * 1_800_000_000
    assert result["four_stream_total_qec_cycles"] == 9 * 7_200_000_000
    assert result["circuit_compilations"] == 9 * 1000 * 53
    assert result["naive_unaggregated_circuit_compilations"] == 9 * 1000 * 200
    assert result["circuit_compilation_reduction_factor"] == pytest.approx(200 / 53)
    assert result["reference_launch_requires_explicit_allow"]
    assert result["reference_launch_blocked_by_unfrozen_hyperparameters"]


def test_gradient_stability_plan_preserves_50_candidates_and_development_seeds(config) -> None:
    result = plan_gradient_stability(config)
    conditions = gradient_stability_conditions(config)
    assert result["source_candidate_count_preserved"]
    assert {row["qec_cycles_per_candidate"] for row in conditions} == {2000, 10000, 36000}
    assert {row["gradient_clipping_mode"] for row in conditions} == {"per_component", "global_l2"}
    assert not ({row["seed"] for row in conditions} & set(config["seed_registry"]["certification_reserved"]))
    assert len({row["condition_id"] for row in conditions}) == len(conditions)


def test_round_invariance_plan_equalizes_qec_cycles_without_launching(config) -> None:
    result = plan_round_invariance(config)
    assert result["rounds"] == [5, 10, 25, 50]
    assert result["primary_rounds"] == 25
    assert result["finite_qec_cycles"] == 4 * 3 * 50 * 36000
    assert result["long_run_not_launched_by_plan"]
