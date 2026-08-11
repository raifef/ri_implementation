"""Permanent V15 regression tests for the complete open-issue closure contract."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hdfa_rl_suite.google_pure_v15 import contracts
from hdfa_rl_suite.google_pure_v15.decoder import run_decoder_steering_offline
from hdfa_rl_suite.google_pure_v15.dynamics import (
    audit_mean_scale_conditioning, audit_scale_floor, classify_residual_decay)
from hdfa_rl_suite.google_pure_v15.fidelity import (
    analyse_figure5c, audit_objective_alignment, audit_ppo_lifecycle,
    build_source_gap_register, fit_step_response, model_figure5a_latency,
    plan_natural_drift_power, report_resource_semantics, verify_provenance)
from hdfa_rl_suite.google_pure_v15.gate import build_heldout_freeze, reference_gate_status
from hdfa_rl_suite.google_pure_v15.imports import build_import_manifest, verify_import_manifest
from hdfa_rl_suite.google_pure_v15.io import ARTIFACT_ROOT, ROOT, read_json
from hdfa_rl_suite.google_pure_v15.ledger import build_fault_ledger
from hdfa_rl_suite.google_pure_v15.reporting import build_report, build_status
from hdfa_rl_suite.google_pure_v15.scaling import (
    SourceBoundary, audit_curvature_distribution, audit_gradient_normalization,
    decompose_figure5b, estimate_hessian_spectrum, project_slow_modes, report_ess,
    run_information_ablation, verify_boundary_map)
from hdfa_rl_suite.google_pure_v15.sensitivity import (
    audit_detector_degree_normalization, audit_source_sensitivity_definition,
    calibrate_multi_point_sensitivity, propagate_calibration_uncertainty,
    verify_calibration_firewall)


@pytest.fixture(scope="module")
def v15_artifacts() -> dict:
    functions = [
        build_import_manifest,
        audit_source_sensitivity_definition,
        audit_detector_degree_normalization,
        calibrate_multi_point_sensitivity,
        propagate_calibration_uncertainty,
        verify_calibration_firewall,
        verify_boundary_map,
        decompose_figure5b,
        audit_gradient_normalization,
        audit_curvature_distribution,
        estimate_hessian_spectrum,
        project_slow_modes,
        run_information_ablation,
        report_ess,
        audit_mean_scale_conditioning,
        audit_scale_floor,
        classify_residual_decay,
        audit_objective_alignment,
        analyse_figure5c,
        model_figure5a_latency,
        fit_step_response,
        plan_natural_drift_power,
        audit_ppo_lifecycle,
        verify_provenance,
        report_resource_semantics,
        build_source_gap_register,
        run_decoder_steering_offline,
    ]
    values = {function.__name__: function() for function in functions}
    values["freeze"] = build_heldout_freeze()
    values["gate"] = reference_gate_status()
    values["ledger"] = build_fault_ledger()
    values["status"] = build_status()
    values["report"] = build_report()
    return values


def test_ledger_contains_exactly_a_through_z(v15_artifacts: dict) -> None:
    ledger = v15_artifacts["ledger"]
    assert [row["issue"] for row in ledger["issues"]] == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    assert ledger["issue_count"] == 26
    assert ledger["all_terminal"]
    assert ledger["all_evidence_present"]
    assert set(row["status"] for row in ledger["issues"]) <= contracts.TERMINAL_STATUSES


def test_source_definition_uses_group_variance_width_not_first_derivative(v15_artifacts: dict) -> None:
    value = v15_artifacts["audit_source_sensitivity_definition"]
    assert value["pass"]
    assert value["source_grouping"] == "SIMULTANEOUS_GATE_GROUP_BY_CONTROL_TYPE"
    assert not value["paper_mode_uses_v13_per_coordinate_map"]
    for row in value["calibration_rows"]:
        assert row["first_derivative_at_optimum"] == 0
        assert not row["first_derivative_is_valid_sensitivity"]
        assert row["sigma0_native"] == pytest.approx(1 / np.sqrt(row["a_pp"]), abs=1e-12)
        assert row["one_normalized_variance_edr_fraction"] == .01


def test_detector_degree_is_not_counted_twice(v15_artifacts: dict) -> None:
    value = v15_artifacts["audit_detector_degree_normalization"]
    assert value["pass"]
    assert not value["extra_degree_multiplier_used"]
    assert not value["global_detector_mean_used"]
    assert all(row["normalized_curvature"] == pytest.approx(1) for row in value["rows"])


def test_multipoint_calibration_has_state_cubic_and_hysteresis_audits(v15_artifacts: dict) -> None:
    value = v15_artifacts["calibrate_multi_point_sensitivity"]
    assert value["pass"]
    assert Path(ROOT / value["raw_npz"]).is_file()
    assert all(row["operating_state_hash"] for row in value["rows"])
    assert all(len(row["offsets_in_sigma0"]) == 7 for row in value["rows"])
    assert all(row["cubic_fraction"] <= .15 for row in value["rows"])
    assert not value["hardware_hysteresis_tested"]


def test_uncertainty_and_seed_firewall_are_fail_closed(v15_artifacts: dict) -> None:
    uncertainty = v15_artifacts["propagate_calibration_uncertainty"]
    firewall = v15_artifacts["verify_calibration_firewall"]
    assert uncertainty["pass"] and firewall["pass"]
    assert not uncertainty["downstream_outcomes_used"]
    assert firewall["pairwise_seed_overlaps"] == []
    assert firewall["forbidden_downstream_imports"] == []
    assert not firewall["heldout_seed_access_during_calibration"]


def test_boundary_is_typed_and_single_use(v15_artifacts: dict) -> None:
    assert v15_artifacts["verify_boundary_map"]["pass"]
    boundary = SourceBoundary(np.zeros(2), np.asarray([2.0, 3.0]), "fixture")
    native, token = boundary.apply(np.asarray([.5, -1.0]))
    assert np.allclose(native, [1.0, -3.0])
    assert token["application_count"] == 1
    with pytest.raises(RuntimeError, match="exactly once"):
        boundary.apply(native, already_applied=True)


def test_figure5b_keeps_mean_candidate_physical_logical_and_floors_separate(v15_artifacts: dict) -> None:
    value = v15_artifacts["decompose_figure5b"]
    assert value["pass"]
    assert value["required_panel_axes"] == ["PHYSICAL_ERROR_RATE_LOG", "LOGICAL_ERROR_RATE_LOG"]
    assert value["floor_bars_required"]
    assert all(row["physical_error_stochastic_candidate_expectation"] >
               row["physical_error_learned_mean"] >= row["physical_irreducible_floor"]
               for row in value["rows"])


def test_gradient_curvature_hessian_and_slow_mode_contracts(v15_artifacts: dict) -> None:
    gradient = v15_artifacts["audit_gradient_normalization"]
    curvature = v15_artifacts["audit_curvature_distribution"]
    hessian = v15_artifacts["estimate_hessian_spectrum"]
    slow = v15_artifacts["project_slow_modes"]
    assert gradient["pass"] and curvature["pass"] and hessian["pass"] and slow["pass"]
    assert all(row["normalized_hessian"] == pytest.approx(.02) for row in curvature["rows"])
    assert all(row["off_diagonal_frobenius_ratio"] == 0 for row in hessian["rows"])
    assert not slow["current_surrogate_plateau_attributable_to_hessian_slow_modes"]


def test_information_and_ess_axes_are_distinct(v15_artifacts: dict) -> None:
    information = v15_artifacts["run_information_ablation"]
    ess = v15_artifacts["report_ess"]
    assert information["pass"] and ess["pass"]
    assert information["candidate_richness_and_shot_richness_are_separate_axes"]
    assert ess["policy_and_detector_ess_are_distinct"]
    assert all(row["fresh_behavior_policy_kish_ess"] == 40 for row in ess["rows"])


def test_direct_sigma_conditioning_and_floor_are_explicit(v15_artifacts: dict) -> None:
    conditioning = v15_artifacts["audit_mean_scale_conditioning"]
    floor = v15_artifacts["audit_scale_floor"]
    assert conditioning["pass"] and floor["pass"]
    assert conditioning["mean_and_scale_gradients_reported_separately"]
    assert floor["initial_exploration_penalty_is_not_negligible"]
    assert all(row["mean_physical_edr_floor_fraction"] < 1e-7 for row in floor["rows"])


def test_finite_horizon_is_never_promoted_to_asymptote(v15_artifacts: dict) -> None:
    value = v15_artifacts["classify_residual_decay"]
    assert value["pass"]
    assert value["finite_horizon_never_relabelled_as_asymptote"]
    assert all(row["classification"] in {
        "STILL_DECAYING_AT_HORIZON", "EMPIRICAL_PLATEAU_WITHIN_HORIZON",
        "NO_IDENTIFIABLE_CONVERGENCE"} for row in value["rows"])


def test_objective_alignment_and_figure5c_remain_honest(v15_artifacts: dict) -> None:
    alignment = v15_artifacts["audit_objective_alignment"]
    figure5c = v15_artifacts["analyse_figure5c"]
    assert alignment["surrogate_alignment_established"]
    assert not alignment["circuit_level_alignment_established"]
    assert not alignment["paper_hardware_alignment_established"]
    assert not figure5c["fit_valid"]
    assert figure5c["legacy_stored_zero_values_rejected"]


def test_latency_and_step_metrics_use_source_relevant_definitions(v15_artifacts: dict) -> None:
    latency = v15_artifacts["model_figure5a_latency"]
    step = v15_artifacts["fit_step_response"]
    assert latency["pass"] and not latency["hardware_latency_identified"]
    assert step["target_definition"] == "ABSOLUTE_INJECTED_TARGET_FRACTION_0.9"
    assert step["observed_final_excursion_fraction_never_used_as_threshold"]
    assert {row["selected_model"] for row in step["rows"]} <= {
        "SINGLE_EXPONENTIAL", "DEAD_TIME_EXPONENTIAL", "TWO_EXPONENTIAL"}
    assert step["median_response_time_90_epochs"] != step["paper_anchor_epochs"]


def test_natural_power_plan_uses_complete_pairs_and_resolvable_dft(v15_artifacts: dict) -> None:
    value = v15_artifacts["plan_natural_drift_power"]
    assert value["planned_complete_paired_runs"] == 48
    assert value["frequency_bins_are_not_replicates"]
    assert value["uncertainty_unit"] == "COMPLETE_PAIRED_RUN"
    assert not value["current_trace_resolves_lowest_scan_frequency"]
    assert value["four_period_minimum_epochs"] > value["current_epochs"]
    assert not value["long_run_auto_launched"]


def test_lifecycle_provenance_and_resources_do_not_promote_smoke(v15_artifacts: dict) -> None:
    lifecycle = v15_artifacts["audit_ppo_lifecycle"]
    provenance = v15_artifacts["verify_provenance"]
    resources = v15_artifacts["report_resource_semantics"]
    assert lifecycle["pass"] and provenance["pass"] and resources["pass"]
    assert not lifecycle["replay_used"] and not lifecycle["extra_passes_used"]
    assert provenance["status_inherits_provenance_without_promotion"]
    assert not resources["current_reference_profile_matches_public_source_budget"]


def test_decoder_steering_blocks_without_primary_backend_and_data(v15_artifacts: dict) -> None:
    value = v15_artifacts["run_decoder_steering_offline"]
    assert value["pass"]
    assert value["execution_status"] == "BLOCKED_PREREQUISITES"
    assert not value["proxy_is_paper_equivalent"]
    assert not value["fixture_is_scientific_evidence"]
    assert set(value["required_arms"]) == {
        "fixed_controls_fixed_prior", "learned_controls_fixed_prior",
        "fixed_controls_steered_prior", "learned_controls_steered_prior"}


def test_immutable_imports_freeze_and_reference_gate_have_no_force_path(v15_artifacts: dict) -> None:
    assert verify_import_manifest()["pass"]
    freeze = v15_artifacts["freeze"]
    gate = v15_artifacts["gate"]
    assert freeze["force_override_allowed"] is False
    assert freeze["heldout_seeds_consumed"] is False
    assert gate["status"] == "REFERENCE_GATE_CLOSED"
    assert not gate["pass"]
    assert gate["force_override_allowed"] is False
    assert not gate["promotion_performed"]


def test_source_gap_classes_are_exhaustive(v15_artifacts: dict) -> None:
    register = v15_artifacts["build_source_gap_register"]
    assert register["pass"]
    assert register["all_claims_classified"]
    assert set(row["classification"] for row in register["rows"]) <= contracts.SOURCE_GAP_CLASSES
    assert not register["paper_equivalence_claim_permitted"]


def test_all_requested_cli_names_are_registered() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    requested = [
        "audit-source-sensitivity-definition", "audit-detector-degree-normalization",
        "calibrate-multi-point-sensitivity", "propagate-calibration-uncertainty",
        "verify-calibration-firewall", "verify-boundary-map", "decompose-figure5b",
        "audit-gradient-normalization", "audit-curvature-distribution",
        "estimate-hessian-spectrum", "project-slow-modes", "run-information-ablation",
        "report-ess", "audit-mean-scale-conditioning", "audit-scale-floor",
        "classify-residual-decay", "audit-objective-alignment", "analyse-figure5c",
        "model-figure5a-latency", "fit-step-response", "plan-natural-drift-power",
        "audit-ppo-lifecycle", "verify-state-chain", "verify-candidate-lineage",
        "report-resource-semantics", "run-decoder-steering-offline",
        "build-heldout-freeze", "reference-gate-status", "status", "report",
    ]
    assert all(f"hdfa-google-v15-{name} =" in text for name in requested)


def test_status_and_22_section_report_are_nonfinal(v15_artifacts: dict) -> None:
    status = v15_artifacts["status"]
    report = v15_artifacts["report"]
    assert status["open_issue_closure_complete"]
    assert status["reference_gate_status"] == "REFERENCE_GATE_CLOSED"
    assert not status["final_evidence"]
    assert report["section_count"] == 22
    assert report["issue_count"] == 26
    assert not report["paper_equivalence_claim_permitted"]
    assert Path(report["report"]).is_file()
