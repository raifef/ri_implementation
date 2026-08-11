"""Permanent V16 optimizer-consistency and physical-matching regression gates."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hdfa_rl_suite.google_pure_v16.contracts import LEGACY_INHERITED, NONFINAL
from hdfa_rl_suite.google_pure_v16.coordinate import (
    audit_coordinate_transform,
    audit_entropy_covariance,
    audit_gradient_covariance,
    audit_ppo_covariance,
    run_covariance_fixture,
)
from hdfa_rl_suite.google_pure_v16.experiments import (
    run_matched_figure5b,
    run_matched_step,
)
from hdfa_rl_suite.google_pure_v16.imports import build_import_manifest, verify_import_manifest
from hdfa_rl_suite.google_pure_v16.io import ROOT
from hdfa_rl_suite.google_pure_v16.optimizer_audits import (
    audit_baseline_reward_scaling,
    audit_direct_sigma,
    audit_local_contraction,
    audit_native_exploration,
    audit_optimizer_sources,
    calibrate_optimizer,
    freeze_optimizer,
    run_source_entropy_anchors,
)


@pytest.fixture(scope="module", autouse=True)
def frozen_imports() -> dict:
    build_import_manifest()
    return verify_import_manifest()


@pytest.fixture(scope="module")
def frozen_optimizer() -> dict:
    return freeze_optimizer()


@pytest.fixture(scope="module")
def matched_step(frozen_optimizer: dict) -> dict:
    return run_matched_step()


@pytest.fixture(scope="module")
def matched_figure5b(frozen_optimizer: dict) -> dict:
    return run_matched_figure5b()


def test_coordinate_transform_contract_is_explicit() -> None:
    result = audit_coordinate_transform()
    assert result["pass"]
    assert result["gradient_rule"] == "grad_x = S^T grad_u"
    assert "A_x = eta_u S^-2" in result["scalar_native_optimizer_rule"]


def test_strict_covariance_fixture_matches_native_candidates_rewards_and_updates() -> None:
    result = run_covariance_fixture()
    assert result["pass"]
    assert all(result["hard_invariants"].values())
    assert max(result["native_reward_max_abs_errors"].values()) == 0.0


def test_mean_and_direct_sigma_finite_difference_covariance() -> None:
    result = audit_gradient_covariance()
    assert result["pass"]
    assert result["finite_difference_mean_max_abs_error"] < 2e-10
    assert result["finite_difference_direct_sigma_max_abs_error"] < 2e-10


def test_ppo_ratio_and_elementwise_clip_are_coordinate_invariant() -> None:
    result = audit_ppo_covariance()
    assert result["pass"]
    assert result["coordinate_ratio_invariant"]
    assert result["clip_epsilon_invariant"]


def test_entropy_jacobian_and_gradient_covariance() -> None:
    result = audit_entropy_covariance()
    assert result["pass"]
    assert np.isclose(result["relative_magnitude_ratio_v15_over_v12"],
                      result["inherited_reward_to_entropy_relative_magnitude"]["V15"] /
                      result["inherited_reward_to_entropy_relative_magnitude"]["V12"])


def test_optimizer_source_audit_has_no_silent_legacy_value() -> None:
    result = audit_optimizer_sources()
    assert result["pass"]
    assert result["legacy_inherited_count"] == 0
    assert all(row["source_class"] != LEGACY_INHERITED for row in result["rows"])


def test_independent_calibration_does_not_read_headline_or_heldout_outputs() -> None:
    result = calibrate_optimizer()
    assert result["pass"]
    assert not result["selection_used_v12_performance"]
    assert not result["selection_used_v15_step_or_figure5b_outcomes"]
    assert not result["selection_used_paper_headline_outputs"]
    assert not result["selection_used_heldout_seeds"]
    assert result["protocol"]["heldout_seeds"] == []
    assert result["protocol"]["development_seeds"]
    assert not set(result["protocol"]["development_seeds"]) & set(result["protocol"]["heldout_seeds"])


def test_frozen_optimizer_is_direct_sigma_and_nonfinal(frozen_optimizer: dict) -> None:
    assert frozen_optimizer["parameterization"] == "DIRECT_SIGMA_SOURCE_EXACT"
    assert frozen_optimizer["v12_curvature_ratio_used_for_selection"] is False
    assert frozen_optimizer["normalization"] == "V15_SOURCE_NORMALIZED"
    assert frozen_optimizer["heldout_seeds"] == []
    for field, value in NONFINAL.items():
        assert frozen_optimizer[field] == value


def test_native_exploration_and_direct_sigma_are_separated(frozen_optimizer: dict) -> None:
    exploration = audit_native_exploration()
    sigma = audit_direct_sigma()
    assert exploration["pass"]
    assert sigma["pass"]
    assert sigma["v15_inherited_entropy_reward_ratio_over_v12"] > 3.8
    assert all(row["reward_and_entropy_sigma_gradients_separate"] for row in sigma["rows"])


def test_source_entropy_anchor_ordering_is_preserved() -> None:
    result = run_source_entropy_anchors()
    assert result["pass"]
    assert [row["source_regime"] for row in result["rows"]] == [
        "TOO_LITTLE", "BALANCED", "TOO_MUCH"]
    assert result["normalization"] == "V15_SOURCE_NORMALIZED"
    assert result["sensitivity_map_hash"]


def test_matched_step_holds_native_physics_and_tapes_fixed(matched_step: dict) -> None:
    assert matched_step["all_native_starts_identical"]
    assert matched_step["all_native_covariances_identical"]
    assert matched_step["all_native_targets_identical"]
    assert matched_step["random_tapes_paired"]
    assert matched_step["optimizer_retuned_after_matched_result"] is False
    assert matched_step["paper_130_epoch_target_used_for_selection"] is False
    contract = matched_step["physical_target_contract"]
    assert contract["native_targets_identical_across_abc"]
    assert np.array_equal(contract["native_target_before_step"],
                          np.zeros(len(contract["native_target_before_step"])))
    assert np.array_equal(np.asarray(contract["native_target_after_step"]) -
                          np.asarray(contract["native_target_before_step"]),
                          contract["native_step_vector"])


def test_matched_figure5b_uses_fractional_residual_reduction(matched_figure5b: dict) -> None:
    assert matched_figure5b["same_native_start_per_seed"]
    assert matched_figure5b["same_native_covariance_per_seed"]
    assert matched_figure5b["same_native_optimum_plant_evaluation_and_tapes"]
    assert matched_figure5b["raw_delta_lambda_is_not_main_metric"]
    for row in matched_figure5b["rows"]:
        assert all("fractional_residual_reduction" in record for record in row["records"])
        assert all("edr" in record and "logical_error" in record for record in row["records"])


def test_local_contraction_matches_declared_model(frozen_optimizer: dict) -> None:
    result = audit_local_contraction()
    assert result["pass"]
    assert all(row["maximum_abs_disagreement"] < 1e-15 for row in result["rows"])
    diagnostic = result["coordinate_covariance_diagnostic"]
    assert diagnostic["classification"] == "COORDINATE_COVARIANCE_DIAGNOSTIC_ONLY"
    assert diagnostic["pass"]
    assert diagnostic["used_for_production_selection"] is False


def test_baseline_reward_scaling_distinguishes_matched_and_unmatched_cases() -> None:
    result = audit_baseline_reward_scaling()
    assert result["pass"]
    assert result["control_coordinate_change_only"]["baseline_loss_coefficient_invariant"]
    assert result["unmatched_inherited_normalized_policy"]["relative_policy_entropy_weight_changes"]
    assert result["all_policy_baseline_entropy_contributions_separately_inspectable"]
    assert all(row["policy_mean_reward_gradient_norm"] > 0 for row in
               result["matched_native_gradient_contributions"])


def test_v16_does_not_modify_or_execute_figure5c() -> None:
    package = ROOT / "src/hdfa_rl_suite/google_pure_v16"
    text = "\n".join(path.read_text(encoding="utf-8") for path in package.glob("*.py"))
    assert "run_figure5c" not in text
    assert not any(path.name.lower().startswith("figure5c") for path in package.rglob("*"))


def test_v12_cannot_become_production_normalization(frozen_optimizer: dict) -> None:
    assert frozen_optimizer["normalization"] == "V15_SOURCE_NORMALIZED"
    assert "V12" not in frozen_optimizer["normalization"]
    assert frozen_optimizer["v12_curvature_ratio_used_for_selection"] is False


def test_no_v16_api_auto_launches_long_or_heldout_work() -> None:
    protocol = json.loads((ROOT / "configs/google_pure_v16/protocol.json").read_text(encoding="utf-8"))
    assert protocol["heldout_seeds"] == []
    assert protocol["matched_step"]["epochs"] < 1000
    assert protocol["matched_figure5b"]["epochs"] < 1000
    package = ROOT / "src/hdfa_rl_suite/google_pure_v16"
    text = "\n".join(path.read_text(encoding="utf-8") for path in package.glob("*.py"))
    assert "source_budget_auto_launched\": False" in text
