from __future__ import annotations

import tomllib

import numpy as np

from hdfa_rl_suite.google_pure_paper_reproduction import panel_b, panel_c
from hdfa_rl_suite.google_pure_v12.contracts import CURRENT_CLASSIFICATIONS, fail_closed_status
from hdfa_rl_suite.google_pure_v12.directional import (
    _cases,
    _rates_for_runtime,
    audit_directional_gradient,
    audit_directional_sensitivity,
    audit_factor_graph_direction,
    audit_units,
    audit_update_efficiency,
    compare_protocols,
    reference_directional_curvature,
)
from hdfa_rl_suite.google_pure_v12.imports import build_import_manifest, validate_import_manifest
from hdfa_rl_suite.google_pure_v12.io import ROOT, read_json
from hdfa_rl_suite.google_pure_v12.lineage import (
    _load_merged,
    audit_figure5b_lineage,
    audit_figure5c_lineage,
    convergence_derivative,
    validate_figure5c_derivative,
)
from hdfa_rl_suite.google_pure_v12.spectral import (
    analyse_natural_drift_uncertainty,
    power_ratio_db,
    validate_natural_drift_sign,
)


def test_frozen_classifications_are_exact_and_nonpromotional():
    assert CURRENT_CLASSIFICATIONS == {
        "FIGURE5A_REAL_TIME_STEERING": "PARTIAL",
        "FIGURE5B_SPARSE_SCALING": "INVALID_DIAGNOSTIC",
        "FIGURE5C_CONVERGENCE_LAW": "INVALID_DIAGNOSTIC",
        "NATURAL_DRIFT_SPECTRAL_SUPPRESSION": "PARTIAL",
        "RANDOMIZED_RECOVERY_AFTER_SPOIL": "FAILED",
        "STEP_RESPONSE_INJECTED_DRIFT": "FAILED",
        "DIRECT_SIGMA_INTEGRATION": "OPERATIONAL",
        "FINAL_GOOGLE_BASELINE": "NOT_READY",
        "HDFA_COMPARISON": "NOT_READY",
    }
    status = fail_closed_status(gates={"all": True})
    assert status["development_gates_passed"]
    assert status["final_evidence"] is False
    assert status["paper_equivalence_claim_permitted"] is False


def test_immutable_import_manifest_hashes_every_source():
    manifest = build_import_manifest()
    assert manifest["import_count"] >= 30
    assert manifest["controller_mode"] == "PAPER_DIRECT_SIGMA"
    assert validate_import_manifest()["pass"]


def test_exactly_three_directional_cases_and_connected_graphs():
    assert tuple(_cases()) == ("BEST_SLOW_FIGURE5A", "FAILED_STEP_RESPONSE", "FAILED_RANDOMIZED_RECOVERY")
    result = audit_factor_graph_direction()
    assert len(result["cases"]) == 3
    assert result["all_directions_connected"]


def test_directional_sensitivity_localizes_unit_scale_attenuation():
    result = audit_directional_sensitivity()
    by_name = {row["case"]: row for row in result["cases"]}
    reference = reference_directional_curvature()
    assert reference > 0
    assert by_name["FAILED_STEP_RESPONSE"]["raw_coordinate_curvature_median"] < reference / 100
    assert by_name["FAILED_RANDOMIZED_RECOVERY"]["raw_coordinate_curvature_median"] < reference / 100
    assert all(len(row["finite_differences"]) >= 3 for row in result["cases"])


def test_directional_score_gradient_and_optimizer_update_are_explicit():
    gradient = audit_directional_gradient()
    assert all(row["score_formula"] == "q=v^T Sigma^-1 (x-mu)" for row in gradient["cases"])
    assert all(len(row["z_advantage_times_q"]) == row["candidate_count"] for row in gradient["cases"])
    update = audit_update_efficiency()
    assert update["optimizer_fault_detected"] is False
    assert all(row["update_efficiency"] in {1.0, None} for row in update["cases"])


def test_step_and_spoil_units_roundtrip_and_apply_sensitivity_once():
    for kind in ("step", "spoil"):
        result = audit_units(kind)
        assert result["roundtrip_pass"]
        assert result["sensitivity_application_count"] == 1
        assert result["target_available_to_controller"] is False


def test_boundary_repair_equalizes_per_coordinate_detector_curvature():
    reference = reference_directional_curvature()
    for name in ("FAILED_STEP_RESPONSE", "FAILED_RANDOMIZED_RECOVERY"):
        case = _cases()[name]
        target = case.target
        base = _rates_for_runtime(case, target[None, :], target, repaired=True)[0]
        coordinate = 0 if name == "FAILED_STEP_RESPONSE" else int(np.flatnonzero(case.mean)[0])
        delta = 1e-3
        plus = target.copy(); plus[coordinate] += delta
        rates = _rates_for_runtime(case, plus[None, :], target, repaired=True)[0]
        owner = int(case.owners[coordinate])
        assert np.isclose((rates[owner] - base[owner]) / delta ** 2, reference, rtol=1e-7)


def test_protocol_diff_finds_no_architecture_or_target_access_change():
    result = compare_protocols()
    assert result["architecture_change"] is False
    assert result["hidden_target_used_by_controller"] is False
    assert result["tuning_performed_before_diagnosis"] is False
    assert len(result["cases"]) == 3


def test_figure5b_lineage_localizes_acquisition_not_merge():
    result = audit_figure5b_lineage()
    assert result["raw_quantities_vary"]
    assert result["lineage_verdict"] == "ACQUISITION_DIRECTIONAL_SCALE_ATTENUATION"
    assert result["classification"] == "INVALID_DIAGNOSTIC"
    assert result["trajectory_table_row_count"] > result["merged_row_count"]
    assert result["original_per_epoch_checkpoint_ids_retained"] is False


def test_figure5c_never_substitutes_zero_for_unidentifiable_fit():
    result = audit_figure5c_lineage()
    assert result["nonzero_derivative_condition_count"] == result["condition_count"]
    assert result["identifiable_fit_condition_count"] == 0
    assert all(row["gamma_times_100"] is None for row in result["conditions"])
    fixture = validate_figure5c_derivative()
    assert fixture["pass"]
    no_window = convergence_derivative(np.linspace(.1, .2, 20), 1.0)
    assert no_window["gamma_times_100"] is None


def test_existing_panel_validators_now_fail_closed_on_noninformative_rows():
    _, figure5b = _load_merged("FIGURE5B_SPARSE_SCALING")
    valid_b, reasons_b, _ = panel_b.validation(figure5b["rows"], "validation")
    assert not valid_b
    assert any("visibility gate" in reason for reason in reasons_b)
    _, figure5c = _load_merged("FIGURE5C_CONVERGENCE_LAW")
    valid_c, reasons_c, _ = panel_c.validation(figure5c["rows"], "validation")
    assert not valid_c
    assert any("R-squared gate" in reason for reason in reasons_c)


def test_natural_drift_power_sign_exact_and_run_level_uncertainty():
    assert float(power_ratio_db(1.0, 1.0)) == 0.0
    assert np.isclose(float(power_ratio_db(.5, 1.0)), -3.010299956639812)
    assert np.isclose(float(power_ratio_db(2.0, 1.0)), 3.010299956639812)
    assert validate_natural_drift_sign()["pass"]
    result = analyse_natural_drift_uncertainty()
    assert result["run_count"] == 6
    assert result["resampling_unit"] == "run_id"
    assert result["frequency_bins_are_replicates"] is False
    assert result["power_db_convention"] == "10*log10(P_learned/P_fixed)"


def test_all_v12_cli_commands_are_registered():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = project["project"]["scripts"]
    required = {
        "hdfa-google-v12-audit-directional-sensitivity", "hdfa-google-v12-audit-factor-graph-direction",
        "hdfa-google-v12-audit-directional-gradient", "hdfa-google-v12-audit-gradient-snr",
        "hdfa-google-v12-audit-update-efficiency", "hdfa-google-v12-audit-step-units",
        "hdfa-google-v12-audit-spoil-units", "hdfa-google-v12-compare-fig5a-step",
        "hdfa-google-v12-audit-figure5b-lineage", "hdfa-google-v12-audit-figure5c-lineage",
        "hdfa-google-v12-validate-figure5c-derivative", "hdfa-google-v12-validate-natural-drift-sign",
        "hdfa-google-v12-analyse-natural-drift-uncertainty", "hdfa-google-v12-run-directional-comparison",
        "hdfa-google-v12-status", "hdfa-google-v12-report",
    }
    assert required <= scripts.keys()
