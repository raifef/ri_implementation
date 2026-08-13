from __future__ import annotations

import math

import numpy as np
import pytest

from hdfa_rl_suite.google_pure_v20.core import batch_snr, candidate_snr
from hdfa_rl_suite.google_pure_v21.benchmark import (
    _frame_condition,
    _metric_row,
    normalize_exploration_damage,
    pareto_dominates,
)
from hdfa_rl_suite.google_pure_v21.candidate_design import (
    CandidateFrame,
    SOURCE_FIDELITY,
    candidate_is_nonoracle,
    estimate_policy_updates,
    generate_frame,
)
from hdfa_rl_suite.google_pure_v21.diagnostics import (
    _variance_components,
    projection_components,
)
from hdfa_rl_suite.google_pure_v21.io import settings
from hdfa_rl_suite.google_pure_v21.lineage import FORBIDDEN_CAMPAIGNS, verify_import_manifest
from hdfa_rl_suite.google_pure_v21.online import (
    apply_safe_control_transform,
    normalized_safety_limits,
)
from hdfa_rl_suite.google_pure_source_exact.figure5a.validation import build_plant
from hdfa_rl_suite.google_pure_v19_experimental.dynamic_validation import (
    _boundary,
    _source_config,
)


def _blocks(dimension: int) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(part, dtype=int) for part in np.array_split(
        np.arange(dimension), 4))


def _linear_rewards(actions: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    return np.asarray(actions) * np.asarray(gradient)[None, :]


def test_projection_retention_is_an_exact_orthogonal_decomposition() -> None:
    gradient = np.asarray([1.0, -2.0, 4.0, 3.0])
    basis = np.asarray([1.0, 1.0, 0.0, 0.0])
    retained, discarded = projection_components(gradient, basis)
    np.testing.assert_allclose(retained + discarded, gradient, atol=0, rtol=0)
    assert retained @ discarded == pytest.approx(0.0, abs=1e-15)
    assert np.linalg.norm(retained) / np.linalg.norm(gradient) == pytest.approx(
        math.sqrt(.5) / math.sqrt(30.0))


def test_variance_decomposition_conserves_total_by_construction() -> None:
    rng = np.random.default_rng(4)
    directions = rng.normal(size=(12, 5))
    shots = rng.normal(scale=.2, size=(48, 5))
    totals = np.repeat(directions, 4, axis=0) + shots
    result = _variance_components(directions, shots, totals)
    assert result["conservation_error"] == pytest.approx(0.0, abs=1e-14)
    assert result["V_direction"] >= 0 and result["V_shot"] >= 0


def test_antithetic_estimator_is_exact_for_coordinate_quadratic_with_unit_frame() -> None:
    dimension = 4
    base = np.ones((4, dimension))
    z = np.concatenate([base, -base])
    frame = CandidateFrame(
        "D1", z, z, z**2 - 1.0, np.ones_like(z), (), True, True, True, {})
    sigma = np.full(dimension, .3)
    target = np.asarray([.2, -.4, .7, -.1])
    actions = sigma[None, :] * z
    rewards = -.5 * (actions - target[None, :])**2
    estimate = estimate_policy_updates(
        frame, rewards, np.zeros(dimension), np.eye(dimension, dtype=bool), sigma)
    np.testing.assert_allclose(estimate["mean_update_direction"], target, atol=1e-14)


def test_orthogonal_sphere_frame_has_correct_scaling_and_covariance_in_expectation() -> None:
    dimension = 12
    gradient = np.linspace(.2, 1.3, dimension)
    sigma = np.full(dimension, .25)
    estimates = []
    covariance = np.zeros((dimension, dimension))
    for epoch in range(512):
        frame = generate_frame("D2", dimension=dimension, epoch=epoch, seed=9,
                               blocks=_blocks(dimension))
        z = frame.standardized_directions
        assert np.allclose(z @ z.T, dimension * np.eye(8), atol=1e-10)
        covariance += z.T @ z / 8
        estimates.append(estimate_policy_updates(
            frame, _linear_rewards(sigma[None, :] * z, gradient),
            np.zeros(dimension), np.eye(dimension, dtype=bool), sigma
        )["mean_update_direction"])
    np.testing.assert_allclose(np.mean(estimates, axis=0), gradient, rtol=.08, atol=.03)
    np.testing.assert_allclose(covariance / 512, np.eye(dimension), rtol=.12, atol=.06)


def test_random_block_estimator_applies_inclusion_probability_correction() -> None:
    dimension = 8
    gradient = np.linspace(.4, 1.1, dimension)
    sigma = np.full(dimension, .2)
    estimates = []
    for epoch in range(1024):
        frame = generate_frame("D4", dimension=dimension, epoch=epoch, seed=31,
                               blocks=_blocks(dimension))
        actions = sigma[None, :] * frame.standardized_directions
        estimates.append(estimate_policy_updates(
            frame, _linear_rewards(actions, gradient), np.zeros(dimension),
            np.eye(dimension, dtype=bool), sigma)["mean_update_direction"])
    np.testing.assert_allclose(np.mean(estimates, axis=0), gradient, rtol=.12, atol=.06)


def test_balanced_blocks_cover_every_coordinate_and_rotate_order() -> None:
    blocks = _blocks(12)
    frames = [generate_frame("D5", dimension=12, epoch=epoch, seed=8, blocks=blocks)
              for epoch in range(12)]
    for frame in frames:
        assert sorted(frame.selected_blocks) == [0, 0, 1, 1, 2, 2, 3, 3]
        assert np.all(frame.inclusion_probabilities == .25)
        for candidate, block in enumerate(frame.selected_blocks):
            support = set(np.flatnonzero(frame.standardized_directions[candidate]))
            assert support == set(blocks[block])
    stacked = np.concatenate([frame.standardized_directions for frame in frames])
    condition = _frame_condition(stacked)
    assert condition["rank"] == 12
    assert np.isfinite(condition["condition_number_nonzero_support"])


def test_corrected_snr_uses_standard_deviation_and_batch_sqrt_k() -> None:
    values = np.asarray([1.0, 2.0, 3.0, 4.0])
    expected = abs(values.mean()) / values.std(ddof=1)
    assert candidate_snr(values) == pytest.approx(expected)
    assert batch_snr(values) == pytest.approx(math.sqrt(len(values)) * expected)


def test_gradient_metrics_use_full_reference_not_zero_orthogonal_motion() -> None:
    reference = np.asarray([1.0, 0.0])
    estimate = np.asarray([1.0, .5])
    contributions = np.tile(estimate, (8, 1))
    row = _metric_row(estimate, reference, reference, contributions)
    assert row["squared_error"] == pytest.approx(.25)
    assert row["directional_magnitude_ratio"] == pytest.approx(1.0)
    assert row["reference_gradient_capture"] == pytest.approx(1.0)
    assert row["orthogonal_error_power"] == pytest.approx(.25)


def test_fixed_budget_and_fast_only_protocol_are_exact() -> None:
    protocol = settings()
    budget = protocol["candidate_budget"]
    assert budget == {"K": 8, "M": 12000, "B": 96000}
    assert budget["K"] * budget["M"] == budget["B"]
    assert protocol["automatic_acquisition_frequencies"] == [1 / 150]
    assert protocol["automatic_campaigns_permitted"] == []


def test_damage_normalization_and_pareto_logic_fail_closed() -> None:
    assert normalize_exploration_damage(.2, .5, .1) == pytest.approx(.5)
    with pytest.raises(ValueError):
        normalize_exploration_damage(.2, .1, .1)
    assert pareto_dominates(.8, .4, 1.0, .4)
    assert pareto_dominates(1.0, .3, 1.0, .4)
    assert not pareto_dominates(.8, .5, 1.0, .4)
    assert not pareto_dominates(1.0, .4, 1.0, .4)


def test_invalid_sphere_sigma_estimators_are_blocked_online() -> None:
    for design in ("D2", "D3"):
        frame = generate_frame(design, dimension=12, epoch=0, seed=2,
                               blocks=_blocks(12))
        assert frame.estimator_valid
        assert not frame.sigma_estimator_valid
        assert frame.sigma_score_factors is None


def test_generalization_provenance_rejects_target_informed_frames() -> None:
    clean = {
        "uses_known_driven_direction": False,
        "uses_target_trajectory": False,
        "uses_future_phase": False,
        "uses_population_or_reference_gradient": False,
        "uses_hidden_optimum": False,
        "uses_multi_run_leakage": False,
        "uses_posthoc_selected_subspace": False,
    }
    assert candidate_is_nonoracle(clean)
    for field in clean:
        tainted = {**clean, field: True}
        assert not candidate_is_nonoracle(tainted)
    assert not candidate_is_nonoracle({})


def test_v15_normalized_safety_envelope_is_phase_independent_and_physical() -> None:
    plant = build_plant(_source_config())
    boundary = _boundary(plant)
    limits = normalized_safety_limits(plant, boundary)
    from hdfa_rl_suite.google_pure_source_exact.figure5a.bounded_action_ablation import Figure5aBoundedActionAblation
    assert np.all(limits <= Figure5aBoundedActionAblation(plant).control_limits)
    for target_sign in (-1.0, 1.0):
        target = boundary.target_to_native(np.full(41, target_sign))
        opposite = apply_safe_control_transform(
            np.full(41, -target_sign * 1e6), plant, boundary)
        native = boundary.apply(opposite).native
        probabilities = plant.probabilities(
            native, 0, 1 / 150, target_controls=target)
        assert np.all(probabilities < plant.probability_ceilings)


def test_source_fidelity_labels_and_frozen_lineage_remain_explicit() -> None:
    assert SOURCE_FIDELITY == {
        "D0": "SOURCE_EXPLICIT", "D1": "DIAGNOSTIC_EXTENSION",
        "D2": "DIAGNOSTIC_EXTENSION", "D3": "DIAGNOSTIC_EXTENSION",
        "D4": "SOURCE_IMPLIED", "D5": "SOURCE_IMPLIED",
    }
    manifest = verify_import_manifest()
    assert manifest["invariants"]["source_style_branch_unchanged"]
    assert manifest["invariants"]["v19_parent_unchanged"]
    assert manifest["invariants"]["v20_projection_not_promoted_to_baseline"]


def test_no_forbidden_campaign_can_be_auto_launched() -> None:
    protocol = settings()
    assert set(FORBIDDEN_CAMPAIGNS) <= set(protocol["forbidden_auto_runs"])
    assert protocol["automatic_campaigns_permitted"] == []
