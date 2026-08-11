from __future__ import annotations

import math

import numpy as np

from hdfa_rl_suite.google_pure_v19.core import (
    PUBLIC_ANALOGUE_SCALE_OBJECTIVE,
    aggregate_damage,
    aggregation_scaling_fixture,
    classify_bound_activity,
    coordinate_quadratic_damage,
    frozen_sigma_sweep,
    lambda_squared_fit,
    phase_aligned_distance,
    phase_bin_means,
    public_analogue_entropy_gradient,
    quadratic_damage,
    sigma_equilibrium,
)
from hdfa_rl_suite.google_pure_v19.diagnostics import (
    FORBIDDEN_CAMPAIGNS,
    _stationary_public_fixture,
    verify_import_manifest,
)
from hdfa_rl_suite.google_pure_v19.io import ARTIFACT_ROOT, NONFINAL, ROOT, read_json


def test_quadratic_damage_matches_analytic_diagonal_gaussian_expectation():
    hessian = np.asarray([2.0, 4.0])
    sigma = np.asarray([0.5, 0.25])
    coordinate = coordinate_quadratic_damage(hessian, sigma)
    np.testing.assert_allclose(coordinate, [0.25, 0.125])
    assert quadratic_damage(hessian, sigma) == 0.375
    assert math.isclose(float(np.sum(coordinate)), quadratic_damage(hessian, sigma))


def test_family_and_neighborhood_aggregations_conserve_coordinate_damage():
    damage = np.asarray([1.0, 2.0, 3.0, 4.0])
    family = aggregate_damage(damage, ["one", "one", "two", "two"])
    neighborhood = aggregate_damage(damage, ["left", "middle", "middle", "right"])
    assert family == {"one": 3.0, "two": 7.0}
    assert neighborhood == {"left": 1.0, "middle": 5.0, "right": 4.0}
    assert sum(family.values()) == sum(neighborhood.values()) == float(np.sum(damage))


def test_duplicated_independent_dimensions_preserve_per_coordinate_objective_ratio():
    rows = aggregation_scaling_fixture(
        [1, 2, 8, 41, 82], curvature=0.02, sigma=0.5, entropy_weight=0.01)
    assert {row["per_coordinate_ratio"] for row in rows} == {2.0}
    for row in rows:
        assert row["reward_gradient_l1"] == row["controls"] * 0.01
        assert math.isclose(row["entropy_gradient_l1"], row["controls"] * 0.02)


def test_source_and_public_analogue_equilibria_have_zero_scale_gradient():
    hessian = np.asarray([0.01, 0.04, 0.16])
    beta = 0.01
    source = sigma_equilibrium(hessian, beta)
    public = sigma_equilibrium(hessian, beta, entropy_divisor=len(hessian))
    np.testing.assert_allclose(hessian * source - beta / source, 0.0, atol=1e-15)
    np.testing.assert_allclose(
        hessian * public + public_analogue_entropy_gradient(public, beta, len(hessian)),
        0.0, atol=1e-15)
    fixture = _stationary_public_fixture(hessian, beta, len(hessian))
    assert fixture["maximum_absolute_equilibrium_error"] < 1e-10


def test_bound_activity_classification_distinguishes_objective_truncation():
    assert classify_bound_activity(1.3, 0.8) == "strongly truncating the objective optimum"
    assert classify_bound_activity(0.8, 0.8) == "equilibrium-limiting"
    assert classify_bound_activity(0.75, 0.8) == "occasionally active"
    assert classify_bound_activity(0.2, 0.8) == "inactive"


def test_phase_bins_and_phase_aligned_distance_are_explicit():
    phases = np.asarray([0.1, 1.0, 3.3, 5.9])
    values = np.asarray([[1.0], [3.0], [7.0], [9.0]])
    binned = phase_bin_means(phases, values, bins=4)
    np.testing.assert_array_equal(binned["counts"], [2, 0, 1, 1])
    assert binned["means"][0, 0] == 2.0
    assert binned["means"][2, 0] == 7.0
    assert phase_aligned_distance(np.ones(4), np.full(4, 1.25)) == 0.25


def test_frozen_sigma_sweep_is_read_only_and_exactly_lambda_squared():
    mean = np.asarray([0.3])
    sigma = np.asarray([0.2])
    noises = np.asarray([[-1.0], [1.0]])
    before_mean, before_sigma = mean.copy(), sigma.copy()
    multipliers = [0.0, 0.25, 0.5, 0.75, 1.0]
    result = frozen_sigma_sweep(mean, sigma, noises, multipliers,
                                lambda value: float(np.sum(np.square(value))))
    np.testing.assert_array_equal(mean, before_mean)
    np.testing.assert_array_equal(sigma, before_sigma)
    assert result["policy_state_unchanged"] is True
    assert result["lambda_zero_equals_mean"] is True
    damage = np.asarray([row["damage"] for row in result["rows"]])
    fit = lambda_squared_fit(np.asarray(multipliers), damage)
    assert fit["r_squared"] == 1.0
    assert math.isclose(fit["slope"], sigma[0] ** 2, abs_tol=1e-15)


def test_frozen_v18_inputs_are_hash_pinned_and_reconstructible():
    manifest = verify_import_manifest()
    assert manifest["pass"] is True
    assert len(manifest["inputs"]) >= 20
    assert manifest["v18_evidence_frozen_before_v19"] is True


def test_causal_repair_is_gated_and_does_not_mutate_production_or_mean_policy():
    root_cause = read_json(ARTIFACT_ROOT / "root_cause_classification.json")
    repair = read_json(ARTIFACT_ROOT / "minimal_repair.json")
    validation = read_json(ARTIFACT_ROOT / "postrepair_validation.json")
    sweep = read_json(ARTIFACT_ROOT / "frozen_sigma_counterfactual_sweep.json")
    assert root_cause["classification"] == "SCALE_OBJECTIVE_EQUILIBRIUM_TOO_EXPLORATORY"
    assert repair["causal_parent_classification"] == root_cause["classification"]
    assert repair["repair"] == PUBLIC_ANALOGUE_SCALE_OBJECTIVE
    assert repair["source_exact"] is False
    assert repair["production_source_exact_controller_changed"] is False
    assert repair["mean_controller_hash_before"] == repair["mean_controller_hash_after"]
    assert sweep["checkpoint_hashes_before"] == sweep["checkpoint_hashes_after"]
    assert validation["pass"] is True
    assert all(row["mean_policy_unchanged"] and row["stationary_fixture_converged"]
               for row in validation["frozen_mean_sampled_policy_checks"])


def test_required_v19_artifacts_are_nonfinal_and_no_forbidden_campaign_ran():
    required = {
        "import_manifest", "exploration_damage_quadratic_comparison",
        "exploration_damage_dimension_decomposition", "entropy_reward_aggregation_audit",
        "sigma_equilibrium_derivation", "sigma_equilibrium_comparison",
        "phase_conditioned_sigma_gradients", "phase_aligned_sigma_limit_cycle",
        "frozen_sigma_counterfactual_sweep", "root_cause_classification",
        "minimal_repair", "postrepair_validation", "status",
    }
    assert all((ARTIFACT_ROOT / f"{name}.json").is_file() for name in required)
    assert (ARTIFACT_ROOT / "FINAL_REPORT.md").is_file()
    for name in required:
        value = read_json(ARTIFACT_ROOT / f"{name}.json")
        for flag, expected in NONFINAL.items():
            assert value[flag] == expected
        assert value.get("forbidden_auto_runs_launched", []) == []
    status = read_json(ARTIFACT_ROOT / "status.json")
    assert set(status["forbidden_auto_runs"]) == set(FORBIDDEN_CAMPAIGNS)


def test_v19_commands_are_registered_once_and_cannot_auto_launch_campaigns():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    commands = (
        "hdfa-google-v19-import-manifest",
        "hdfa-google-v19-audit-exploration-damage",
        "hdfa-google-v19-decompose-exploration-damage",
        "hdfa-google-v19-audit-entropy-reward-aggregation",
        "hdfa-google-v19-derive-sigma-equilibrium",
        "hdfa-google-v19-audit-phase-sigma-gradients",
        "hdfa-google-v19-run-frozen-sigma-sweep",
        "hdfa-google-v19-classify-root-cause",
        "hdfa-google-v19-run-minimal-repair-validation",
        "hdfa-google-v19-status",
        "hdfa-google-v19-report",
    )
    assert all(pyproject.count(command) == 1 for command in commands)
    protocol = read_json(ROOT / "configs/google_pure_v19/protocol.json")
    assert protocol["automatic_campaigns_permitted"] == []
    assert set(protocol["forbidden_auto_runs"]) == set(FORBIDDEN_CAMPAIGNS)
