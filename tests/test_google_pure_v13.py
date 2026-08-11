from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hdfa_rl_suite.google_pure_v13.contracts import NONFINAL, V12_FINDINGS
from hdfa_rl_suite.google_pure_v13.diagnostics import audit_ppo_lifecycle, report_epoch_semantics
from hdfa_rl_suite.google_pure_v13.findings import write_v12_findings_contract
from hdfa_rl_suite.google_pure_v13.imports import build_import_manifest, verify_import_manifest
from hdfa_rl_suite.google_pure_v13.io import ARTIFACT_ROOT, ROOT, canonical_hash, config
from hdfa_rl_suite.google_pure_v13.natural import plan_natural_drift_power, run_natural_drift
from hdfa_rl_suite.google_pure_v13.reporting import build_status
from hdfa_rl_suite.google_pure_v13.runtime import STEP, run_v13_arm
from hdfa_rl_suite.google_pure_v13.scaling import (audit_figure5b_contract,
                                                  validate_figure5c_fit)
from hdfa_rl_suite.google_pure_v13.sensitivity import (
    CoordinateBatch,
    SensitivityBoundary,
    calibrate_edr_sensitivity,
    require_native_boundary,
    validate_sensitivity_map,
)
from hdfa_rl_suite.google_pure_v13.step import _fit_curve


def _calibration() -> dict:
    path = ARTIFACT_ROOT / "sensitivity_calibration/scales.json"
    return calibrate_edr_sensitivity() if not path.is_file() else {
        "validation": validate_sensitivity_map(),
        "scales": json.loads(path.read_text(encoding="utf-8")),
    }


def test_namespace_protocol_and_all_cli_entries_exist():
    assert (ROOT / "src/hdfa_rl_suite/google_pure_v13").is_dir()
    assert (ROOT / "configs/google_pure_v13/protocol.json").is_file()
    names = {
        "calibrate-edr-sensitivity", "validate-sensitivity-map", "compare-normalization-branches",
        "run-step-validation", "fit-step-response", "verify-state-chain", "verify-candidate-lineage",
        "audit-figure5b-contract", "audit-figure5b-convergence", "run-figure5b-validation",
        "analyse-figure5c", "validate-figure5c-fit", "plan-natural-drift-power",
        "run-natural-drift", "analyse-natural-drift", "test-detector-logical-alignment",
        "report-effective-sample-size", "audit-ppo-lifecycle", "report-epoch-semantics",
        "status", "report",
    }
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert all(f"hdfa-google-v13-{name} =" in pyproject for name in names)


def test_immutable_imports_and_v12_findings_are_frozen():
    manifest = build_import_manifest()
    assert manifest["import_count"] >= 20
    assert verify_import_manifest()["pass"]
    assert write_v12_findings_contract()["findings"] == V12_FINDINGS


def test_symmetric_calibration_has_raw_counts_intervals_fits_and_source_literal_kappa():
    result = _calibration()
    assert result["validation"]["pass"]
    scales = result["scales"]
    assert scales["kappa_ref_edr_fraction"] == .01
    assert scales["source_classification"]["kappa_reference"] == "SOURCE_LITERAL"
    assert scales["scale_count"] == 1848
    assert all(abs(row["conditioned_edr_coefficient_fraction_per_normalized_squared"] - .01) < 1e-12
               for row in scales["scales"])
    raw = json.loads((ARTIFACT_ROOT / "sensitivity_calibration/raw.json").read_text(encoding="utf-8"))
    assert raw["row_count"] == 1848 * 9
    assert all(row["detector_event_count"] >= 0 and len(row["edr_interval_95"]) == 2 for row in raw["rows"])


def test_boundary_applies_exactly_once_and_rejects_all_known_mapping_faults():
    _calibration()
    boundary = SensitivityBoundary.from_artifact(STEP)
    normalized = CoordinateBatch(np.zeros(924), boundary.control_order_hash,
                                 sensitivity_map_hash=boundary.sensitivity_map_hash)
    with pytest.raises(RuntimeError):  # missing boundary
        require_native_boundary(normalized, control_order_hash=boundary.control_order_hash,
                                sensitivity_map_hash=boundary.sensitivity_map_hash)
    native = boundary.apply(normalized).native
    require_native_boundary(native, control_order_hash=boundary.control_order_hash,
                            sensitivity_map_hash=boundary.sensitivity_map_hash)
    with pytest.raises(RuntimeError):  # double application / inverse-space input
        boundary.apply(native)
    with pytest.raises(RuntimeError):
        boundary.apply(CoordinateBatch(np.zeros(924), "reordered",
                                       sensitivity_map_hash=boundary.sensitivity_map_hash))
    with pytest.raises(RuntimeError):
        boundary.apply(CoordinateBatch(np.zeros(924), boundary.control_order_hash,
                                       sensitivity_map_hash="stale-map"))
    with pytest.raises(RuntimeError):  # inverse scale pretending to be the same map
        SensitivityBoundary(1.0 / boundary.scales, np.zeros(924),
                            control_order_hash=boundary.control_order_hash,
                            sensitivity_map_hash=boundary.sensitivity_map_hash,
                            expected_scale_hash=canonical_hash(boundary.scales.tolist()))


def test_tiny_direct_sigma_run_has_continuous_state_and_candidate_reward_lineage():
    _calibration()
    result = run_v13_arm(STEP, seed=70001, epochs=4, candidates=4,
                        cycles_per_candidate=1000, entropy_coefficient=.001, persist=False)
    assert result["controller_mode"] == "PAPER_DIRECT_SIGMA"
    assert result["parameterization"] == "direct_sigma"
    assert result["sensitivity_application_count"] == 1
    assert result["state_chain_pass"] and result["candidate_lineage_pass"]
    assert len(result["candidate_lineage"]) == 16
    assert result["qec_cycles"] == 16_000
    assert result["detector_event_trials"] == 16_000 * 24


def test_step_fit_uses_target_relative_exponential_timescale():
    tau = 130.0
    time = np.arange(800)
    values = 1.02 * (1.0 - np.exp(-time / tau))
    fit = _fit_curve(values)
    assert abs(fit["tau_epochs"] - tau) / tau < .01
    assert abs(fit["asymptotic_target_fraction"] - 1.02) < .01
    assert fit["r_squared"] > .999


def test_figure5_contract_and_derivative_fixture_are_fail_closed():
    contract = audit_figure5b_contract()
    assert not contract["normalized_lambda_only_plot_permitted_as_figure5b"]
    assert contract["required_plot_fields"] == ["physical_error", "logical_error", "epoch_colour",
                                                 "irreducible_floor", "distance"]
    assert validate_figure5c_fit()["pass"]


def test_natural_power_uses_complete_runs_and_never_auto_executes():
    plan = plan_natural_drift_power()
    assert plan["uncertainty_unit"] == "COMPLETE_PAIRED_RUN"
    assert not plan["frequency_bins_are_replicates"]
    assert not plan["inferential_smoothing_used"]
    assert plan["planned_complete_runs"] >= 6
    run = run_natural_drift(execute_long=False)
    assert not run["executed"]
    assert run["reason"] == "EXPLICIT_EXECUTE_LONG_REQUIRED"


def test_ppo_lifecycle_epoch_semantics_and_status_never_promote():
    assert audit_ppo_lifecycle()["pass"]
    semantics = report_epoch_semantics()
    assert semantics["epoch_is_not_one_qec_cycle"]
    status = build_status()
    for key, expected in NONFINAL.items():
        assert status[key] is expected
    assert status["classifications"]["FINAL_GOOGLE_BASELINE"] == "NOT_READY"


def test_protocol_keeps_long_runs_explicit_and_source_kappa_frozen():
    value = config()
    assert value["sensitivity_calibration"]["kappa_ref_edr_percentage_points"] == 1.0
    assert not value["long_runs"]["auto_launch"]
    assert value["long_runs"]["explicit_execute_required"]

