from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hdfa_rl_suite.google_pure_source_exact.figure5a.acquisition import run_cell, substitution_identity
from hdfa_rl_suite.google_pure_source_exact.figure5a.cli import plan
from hdfa_rl_suite.google_pure_source_exact.figure5a.contracts import (
    AcquisitionMode,
    Figure5aProtocol,
    SOURCE_CANDIDATE_QEC_CYCLES,
    SOURCE_ENTROPY_ANCHORS,
    ratio_from_raw_counts,
)
from hdfa_rl_suite.google_pure_source_exact.figure5a.entropy_scan import (
    build_conditions,
    classify_anchor_rows,
    scan_contract,
)
from hdfa_rl_suite.google_pure_source_exact.figure5a.plant import Figure5aStimPlant
from hdfa_rl_suite.google_pure_source_exact.figure5a.normalization import (
    Figure5aEmpiricalBoundary,
    reward_representation_hash,
)
from hdfa_rl_suite.google_pure_source_exact.figure5a.validation import build_plant
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.gaussian import DirectSigmaGaussianPolicy
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.losses import total_loss_and_gradients
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.optimizer import OptimizerConfig


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
    path = ROOT / config["dependencies"]["normalization_bundle"]
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
    assert plant.raw_detector_count == 24
    assert plant.detector_count == 16
    assert plant.detector_count < plant.raw_detector_count
    assert sorted(raw for group in plant.reward_component_raw_detectors for raw in group) == \
        list(range(plant.raw_detector_count))
    longer = json.loads(json.dumps(config))
    longer["plant"]["circuit_rounds"] = 5
    five_round_plant = build_plant(longer)
    assert five_round_plant.raw_detector_count == 40
    assert five_round_plant.detector_count == plant.detector_count
    np.testing.assert_array_equal(five_round_plant.mask, plant.mask)
    np.testing.assert_array_equal(
        five_round_plant.mask.sum(axis=0), plant.mask.sum(axis=0))


def test_empirical_normalization_is_plant_and_reward_bound_without_hidden_point01(
        plant, normalization_artifact) -> None:
    artifact = normalization_artifact
    boundary = Figure5aEmpiricalBoundary.from_artifact(plant, artifact)
    assert artifact["plant_hash"] == plant.plant_hash
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
    assert np.all(plant.probabilities(np.full(41, 2.0), 0, 1 / 1000) < plant.maximum_probability)


def test_latent_action_transform_is_invertible_hidden_target_free_and_physically_safe(plant) -> None:
    latent = np.linspace(-2.75, 2.75, 41)
    applied = plant.apply_control_transform(latent)
    np.testing.assert_allclose(plant.latent_controls_for(applied), latent, rtol=1e-12, atol=1e-12)
    assert np.all(np.abs(applied) < plant.control_limits)
    assert np.all(plant.control_limits > 1.0)

    positive_extreme = plant.apply_control_transform(np.full(41, 1e6))
    negative_extreme = plant.apply_control_transform(np.full(41, -1e6))
    assert np.all(plant.probabilities(positive_extreme, 750, 1 / 1000) <
                  plant.maximum_probability)
    assert np.all(plant.probabilities(negative_extreme, 250, 1 / 1000) <
                  plant.maximum_probability)


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
    protocol = Figure5aProtocol(AcquisitionMode.SMOKE, 2, 3, 30, 3)
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
    assert resumed["stream_totals"] == mono["stream_totals"]
    assert resumed["epoch_records"] == mono["epoch_records"]
    assert resumed["no_candidates_dropped"]
    assert resumed["candidate_boundaries_completed"] == 6
    assert resumed["schema_version"] == "figure5a-cell.v3"
    assert resumed["raw_detector_count"] > resumed["detector_count"]
    assert resumed["reward_representation"] == "time_translation_equivalence_class_mean_edr"
    assert resumed["action_execution"] == "plant_derived_per_coordinate_scaled_tanh"
    assert resumed["likelihood_space"] == "latent_gaussian"
    assert resumed["action_transform_invertible"]
    assert not resumed["action_transform_uses_hidden_optimum"]
    assert all(len(record["counts"][stream]) == 3 for record in resumed["epoch_records"]
               for stream in ("fixed", "optimal", "stochastic", "learned_mean"))
    assert all(record["action_execution"] == "plant_derived_per_coordinate_scaled_tanh"
               and record["likelihood_space"] == "latent_gaussian"
               for record in resumed["epoch_records"])
    assert all(record["optimum"] == plant.optimum(record["epoch"], 0.1)[0]
               for record in resumed["epoch_records"])


def test_batched_checkpoint_flush_is_bit_exact(tmp_path, plant) -> None:
    protocol = Figure5aProtocol(AcquisitionMode.SMOKE, 2, 3, 30, 3)
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
    protocol = Figure5aProtocol(AcquisitionMode.SMOKE, 2, 3, 30, 3)
    anchors = scan_contract(config, mode=AcquisitionMode.SMOKE, scan="anchors", protocol=protocol,
                            plant_hash=plant.plant_hash, controller_hash="controller")
    dense = scan_contract(config, mode=AcquisitionMode.SMOKE, scan="dense", protocol=protocol,
                          plant_hash=plant.plant_hash, controller_hash="controller")
    assert anchors["plant_hash"] == dense["plant_hash"]
    assert anchors["controller_hash"] == dense["controller_hash"]
    assert anchors["validation_watermark"] and dense["validation_watermark"]


def test_anchor_classification_is_quantitative_and_seed_stable() -> None:
    rows = []
    for seed in (1, 2):
        for weight, ent, stochastic, learned in ((0.001, -5.0, 0.1, 0.2),
                                                  (0.01, -3.0, 0.8, 0.85),
                                                  (0.1, -1.0, 0.3, 0.9)):
            rows.append({"seed": seed, "entropy_weight": weight,
                         "epoch_records": [{"policy_entropy": ent}],
                         "stochastic_ratio": {"source_ratio": stochastic},
                         "learned_mean_ratio": {"source_ratio": learned}})
    result = classify_anchor_rows(rows, {"minimum_entropy_separation": 0.05,
        "minimum_high_exploration_gap": 0.02, "minimum_low_tracking_gap": 0.02,
        "middle_must_maximize_stochastic_ratio": True})
    assert result["anchor_classification_pass"] and result["stable_over_seeds"]


def test_anchor_classification_fails_closed_on_zero_denominator() -> None:
    rows = []
    for weight in SOURCE_ENTROPY_ANCHORS:
        rows.append({"seed": 1, "entropy_weight": weight,
                     "epoch_records": [{"policy_entropy": float(weight)}],
                     "stochastic_ratio": {"source_ratio": None},
                     "learned_mean_ratio": {"source_ratio": None}})
    result = classify_anchor_rows(rows, {"minimum_entropy_separation": 0.05,
        "minimum_high_exploration_gap": 0.02, "minimum_low_tracking_gap": 0.02,
        "middle_must_maximize_stochastic_ratio": True})
    assert not result["anchor_classification_pass"]
    assert "zero finite-shot" in result["seed_rows"][0]["blocking_reasons"][0]


def test_reference_planner_reports_exact_cost(config, tmp_path) -> None:
    result = plan(CONFIG_PATH, tmp_path, mode=AcquisitionMode.REFERENCE, scan="anchors")
    assert result["condition_count"] == 9
    assert result["candidate_training_qec_cycles"] == 9 * 1_800_000_000
    assert result["four_stream_total_qec_cycles"] == 9 * 7_200_000_000
    assert result["reference_launch_requires_explicit_allow"]
