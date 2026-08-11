from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from hdfa_rl_suite.google_pure_v6 import DISCLAIMER, OUTCOME_CLASSES
from hdfa_rl_suite.google_pure_v6.baseline import DetectorBaseline
from hdfa_rl_suite.google_pure_v6.config import CERTIFICATION_SEEDS, config_dir, guard_seed, repository_root
from hdfa_rl_suite.google_pure_v6.experiments import POLICY_CLASSES, run_matched_trace
from hdfa_rl_suite.google_pure_v6.factor_graph import global_importance_ratio, local_importance_ratios
from hdfa_rl_suite.google_pure_v6.metrics import spectral_metrics, stability_metrics
from hdfa_rl_suite.google_pure_v6.plant import PureQuadraticPlant, default_spec, optimum_tape
from hdfa_rl_suite.google_pure_v6.policy import FactorizedGaussianPolicy, component_log_probability, gaussian_scores
from hdfa_rl_suite.google_pure_v6.snapshot import EXPECTED_HEADLINE, current_v5_headline
from hdfa_rl_suite.google_pure_v6.studies import freeze_repaired_drift_protocol, run_certification
from hdfa_rl_suite.google_pure_v6.update import ppo_objective_and_gradient


def test_outcome_hierarchy_is_frozen_and_disclaimer_exact():
    assert OUTCOME_CLASSES == (
        "BENCHMARK_FAILURE", "REPORTING_CONVENTION_FAILURE", "UNIT_OR_NORMALIZATION_FAILURE",
        "EXPLORATION_CALIBRATION_FAILURE", "BANDWIDTH_MISMATCH", "REPLAY_STALENESS",
        "BASELINE_FAILURE", "OBJECTIVE_TRANSCRIPTION_FAILURE", "SYNTHETIC_TASK_NON_COMMENSURABILITY",
        "GENUINE_CONTROLLER_FAILURE", "PURE_GOOGLE_STYLE_SYNTHETIC_REPRODUCTION_CERTIFIED",
        "PARTIAL_PURE_REPRODUCTION",
    )
    assert DISCLAIMER == "This is an open synthetic reproduction of the published Google-style RL algorithm. Google’s proprietary controller code and hardware control dynamics were unavailable."


def test_v5_headline_is_reproduced_exactly_without_importing_v5():
    assert current_v5_headline() == EXPECTED_HEADLINE
    source = Path(__file__).parents[1] / "src" / "hdfa_rl_suite" / "google_pure_v6"
    for path in source.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                text = ast.unparse(node)
                assert "google_pure_v5" not in text
                assert ".stage" not in text


def test_metric_orientations_are_reciprocal_and_sign_reversed():
    stability = stability_metrics(np.asarray([0.0, 2.0, 0.0, 2.0]), np.asarray([0.5, 1.5, 0.5, 1.5]))
    assert stability["stability_suppression_factor_fixed_over_mean"] == pytest.approx(2.0)
    assert stability["stability_residual_ratio_mean_over_fixed"] == pytest.approx(0.5)
    spectral = spectral_metrics(10.0, 1.0)
    assert spectral["low_frequency_suppression_db_fixed_over_mean"] == pytest.approx(10.0)
    assert spectral["low_frequency_residual_db_mean_over_fixed"] == pytest.approx(-10.0)


def test_gaussian_scores_match_finite_difference():
    action = np.asarray([[0.1, -0.3]])
    mean = np.asarray([0.02, -0.1])
    log_scale = np.log(np.asarray([0.25, 0.4]))
    score_mean, score_scale = gaussian_scores(action, mean, log_scale)
    step = 1e-6
    for index in range(2):
        plus, minus = mean.copy(), mean.copy()
        plus[index] += step
        minus[index] -= step
        numeric = (component_log_probability(action, plus, log_scale).sum() - component_log_probability(action, minus, log_scale).sum()) / (2*step)
        assert score_mean[0, index] == pytest.approx(numeric, abs=1e-6)
        plus, minus = log_scale.copy(), log_scale.copy()
        plus[index] += step
        minus[index] -= step
        numeric = (component_log_probability(action, mean, plus).sum() - component_log_probability(action, mean, minus).sum()) / (2*step)
        assert score_scale[0, index] == pytest.approx(numeric, abs=1e-6)


def test_local_ratios_are_current_over_collection_and_not_global():
    actions = np.asarray([[0.2, -0.1, 0.3]])
    old_mean, old_log = np.zeros(3), np.log(np.asarray([0.3, 0.3, 0.3]))
    mean, log_scale = np.asarray([0.05, 0.0, -0.02]), np.log(np.asarray([0.25, 0.35, 0.28]))
    old = component_log_probability(actions, old_mean, old_log)
    mask = np.asarray([[1, 1, 0], [0, 1, 1]], bool)
    local = local_importance_ratios(actions, mean, log_scale, old, mask)
    manual = np.exp((component_log_probability(actions, mean, log_scale) - old) @ mask.T)
    assert np.allclose(local, manual)
    assert not np.allclose(local[:, 0], global_importance_ratio(actions, mean, log_scale, old))


def test_negative_advantage_clipping_truth_table_and_sparse_gradient():
    mask = np.asarray([[1, 0]], bool)
    actions = np.asarray([[0.0, 0.4]])
    current_mean, current_log = np.zeros(2), np.zeros(2)
    # Collection log probability is chosen to make detector-local rho=0.7.
    current = component_log_probability(actions, current_mean, current_log)
    collection = current.copy()
    collection[0, 0] -= np.log(0.7)
    objective, gm, gs, diagnostic = ppo_objective_and_gradient(actions, np.asarray([[-1.0]]), mask,
        current_mean, current_log, collection, clip=0.2, entropy_coefficient=0.0)
    assert objective == pytest.approx(-0.8)
    assert gm[0] == 0.0 and gs[0] == 0.0
    assert gm[1] == 0.0 and gs[1] == 0.0
    assert diagnostic["clip_fraction"] == pytest.approx(1.0)


def test_entropy_gradient_is_once_per_coordinate_not_detector_degree():
    actions = np.asarray([[0.1, -0.1, 0.2]])
    mean, log_scale = np.zeros(3), np.zeros(3)
    old = component_log_probability(actions, mean, log_scale)
    mask = np.asarray([[1, 0, 1], [0, 1, 1], [0, 0, 1]], bool)
    _, gm, gs, diagnostic = ppo_objective_and_gradient(actions, np.zeros((1, 3)), mask, mean, log_scale, old,
        clip=0.2, entropy_coefficient=0.003)
    assert np.allclose(gm, 0.0)
    assert np.allclose(gs, 0.003)
    assert diagnostic["control_detector_degree"] == [1, 1, 3]


def test_baseline_freeze_and_ema():
    baseline = DetectorBaseline(2, coefficient=0.2)
    rewards = np.asarray([[-1.0, -0.4], [-0.6, -0.2]])
    frozen = baseline.snapshot()
    advantages = baseline.advantages(rewards, frozen)
    updated = baseline.update(rewards)
    assert np.array_equal(advantages, rewards)
    assert np.allclose(updated, 0.2 * rewards.mean(axis=0))
    assert np.array_equal(frozen, np.zeros(2))


def test_latent_likelihood_applied_action_and_native_action_are_separate():
    spec = default_spec(3)
    policy = FactorizedGaussianPolicy(np.asarray([0.95, -0.95, 0.0]), spec.coordinates, initial_scale=0.5, seed=12)
    batch = policy.sample(30, policy_version=0, epoch=0, environment_time=0, graph_version="g")
    assert np.any(batch.latent_normalized_actions != batch.applied_normalized_actions)
    expected = component_log_probability(batch.latent_normalized_actions, batch.collection_mean, batch.collection_log_scale)
    assert np.allclose(batch.collection_component_log_probability, expected)
    assert np.allclose(batch.applied_native_actions, spec.coordinates.to_native(batch.applied_normalized_actions))


def test_units_roundtrip_and_native_plant_equivalence():
    spec = default_spec(6)
    plant = PureQuadraticPlant(spec)
    x = np.linspace(-0.8, 0.7, 6)
    u = spec.coordinates.to_native(x)
    assert np.allclose(spec.coordinates.to_normalized(u), x)
    assert np.allclose(plant.detector_rates_native(u[None], plant.base_optimum_native),
                       plant.detector_rates_normalized(x[None], spec.base_optimum_normalized))


def test_repaired_protocol_is_one_sided_and_hash_frozen():
    artifact = freeze_repaired_drift_protocol()
    assert artifact["protocol"]["strobe"]["pattern"] == [0.0, 1.0]
    assert artifact["protocol"]["strobe"]["symmetric_sign_flip_forbidden"] is True
    assert artifact["cross_family_aggregation_forbidden"] is True


def test_four_policy_traces_are_distinct_and_non_aliasing():
    spec = default_spec(4)
    tape = optimum_tape("step", 8, 0.15, controls=4)
    choices = {
        "initial_scale": 0.05, "scale_bounds": [0.025, 0.18], "normalized_bounds": [-1.0, 1.0],
        "mean_learning_rate": 0.02, "scale_learning_rate": 0.001, "baseline_coefficient": 0.08,
        "replay_capacity_epochs": 0, "ppo_clip": 0.2, "entropy_coefficient": 0.0001, "update_passes": 1,
    }
    result = run_matched_trace(PureQuadraticPlant(spec), tape, choices, seed=7, candidates=4, cycles=500)
    assert set(result["logical_risk"]) == set(POLICY_CLASSES)
    assert len({id(result["logical_risk"][key]) for key in POLICY_CLASSES}) == 4


def test_development_cannot_touch_certification_seeds_and_certification_is_blocked():
    with pytest.raises(ValueError):
        guard_seed(CERTIFICATION_SEEDS[0])
    with pytest.raises((RuntimeError, FileNotFoundError)):
        run_certification(seed=CERTIFICATION_SEEDS[0], confirm=True)


def test_all_v6_entry_points_are_registered():
    text = (repository_root() / "pyproject.toml").read_text(encoding="utf-8")
    expected = (
        "snapshot-v5", "migrate-v5-metric-schema", "audit-source-compliance", "validate-gaussian-scores",
        "audit-local-ratios", "audit-ppo-clipping", "audit-entropy-normalization", "audit-objective-aggregation",
        "audit-baseline", "audit-replay", "audit-units", "validate-quadratic-gradients", "audit-candidate-damage",
        "freeze-repaired-drift-protocol", "run-repaired-drift-unchanged", "run-sine-bandwidth",
        "run-natural-drift-retention", "run-exploration-calibration", "run-hyperparameter-study",
        "run-static-validation", "run-scaling-retention", "run-recovery-retention", "run-development-scorecard",
        "freeze-certification", "run-certification",
    )
    for suffix in expected:
        assert f"hdfa-google-v6-{suffix}" in text
