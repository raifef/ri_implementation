import json
from pathlib import Path
import numpy as np
import pytest

from hdfa_rl_suite.google_pure_source_exact.identity import PAPER_DIRECT_SIGMA, build_direct_sigma_identity
from hdfa_rl_suite.google_pure_source_exact.integration import run_tiny_integration
from hdfa_rl_suite.google_pure_paper_reproduction.direct_path import protocol_identity_reasons, require_amended_acquisition
from hdfa_rl_suite.google_pure_paper_reproduction.experiment_families import ExperimentFamily
from hdfa_rl_suite.google_pure_paper_reproduction.paper_figures import acquire, build_protocol
from hdfa_rl_suite.google_pure_paper_reproduction.natural_drift import diagnostic_dft, validation as natural_validation
from hdfa_rl_suite.google_pure_paper_reproduction.panel_b import validation as panel_b_validation
from hdfa_rl_suite.google_pure_paper_reproduction.randomized_recovery import validation as recovery_validation
from hdfa_rl_suite.google_pure_v7.response import estimate_step_response

ROOT=Path(__file__).resolve().parents[2]

def test_tiny_integration_proves_amended_execution_path_without_promotion(tmp_path):
    result=run_tiny_integration(tmp_path)
    assert result["pass"] and result["controller_mode"]==PAPER_DIRECT_SIGMA
    assert result["parameterization"]=="direct_sigma" and result["control_count"]==41
    assert all(result["gates"].values())
    assert set(result["policy_decomposition_counts"])=={"fixed","oracle","oracle_with_policy_sigma","learned_mean","sampled_candidates"}
    assert result["training_qec_cycles"]>0 and not result["scientifically_valid"] and not result["final_evidence"]

def test_legacy_controller_protocol_is_rejected_before_paper_acquisition():
    expected=build_direct_sigma_identity(ROOT)
    legacy={"experiment_family":"FIGURE5A_REAL_TIME_STEERING","controller_mode":"source_mapped_v7_production_ppo",
        "controller_hash":"old","controller_code_hash":"old-code","parameterization":"legacy_log_scale",
        "plant_hash":"v6-default-quadratic-6","graph_hash":"local-detector-control-mask-v6"}
    reasons=protocol_identity_reasons(legacy)
    assert any("controller_mode mismatch" in reason for reason in reasons)
    assert any("plant contract mismatch" in reason for reason in reasons)
    assert expected["controller_hash"] not in {legacy["controller_hash"],legacy["controller_code_hash"]}

def test_every_amended_paper_family_passes_identity_preflight_without_acquisition(tmp_path):
    run_tiny_integration(tmp_path)
    # The bridge reads the canonical workspace manifest, which is produced by the
    # repository integration command and must match the current code identity.
    run_tiny_integration(ROOT/"artifacts/google_pure_source_exact/direct_sigma_integration")
    synthetic=[family for family in ExperimentFamily if family.value not in {
        "PUBLIC_ENDPOINT_DATA_REPRODUCTION","PUBLIC_TABLE_REPRODUCTION"}]
    for family in synthetic:
        protocol=build_protocol(family.value,mode="paper-scale")
        assert not protocol_identity_reasons(protocol)
        require_amended_acquisition(protocol)

def test_ninety_percent_response_is_relative_to_injected_target():
    onset=20; trace=np.zeros(240); trace[onset:]=.3*(1-np.exp(-np.arange(220)/40))
    result=estimate_step_response(trace,onset_epoch=onset,target=1.0,sustained_epochs=10)
    assert result["final_response"]==pytest.approx(.3,abs=.01)
    assert result["response_time_90_epochs"] is None
    assert result["response_classification"]=="TARGET_90_NOT_REACHED_WITHIN_HORIZON"

def test_all_censored_recovery_is_negative_evidence():
    rows=[{"not_a_step_response":True,"randomized_fraction":.5,"recovery_epoch":None} for _ in range(9)]
    valid,reasons,metrics=recovery_validation(rows,"paper-scale")
    assert not valid and reasons and metrics["censored_count"]==9
    assert metrics["median_recovery_epoch"] is None and not metrics["median_identifiable"]
    assert metrics["outcome"]=="RECOVERY_NOT_REACHED_WITHIN_HORIZON"

def test_legacy_natural_drift_and_panel_b_cannot_be_promoted():
    base=np.linspace(1,.7,800); learned=np.linspace(1,.6,800)
    natural_rows=[{"plant_id":str(i),"power_db_convention":"10*log10(power ratio)",
        "low_frequency_suppression_db_fixed_over_mean":1.25,"trajectory":{"learned_mean":learned.tolist(),"fixed_policy":base.tolist()},
        "stream_kind":"LOGICAL_RISK_PROXY_DIAGNOSTIC","source_dft_estimator":False,
        "warmup_epoch_excluded":False,"source_epoch_150_normalization":False,"spectral_aggregation":"LEGACY_WELCH_BAND_DIAGNOSTIC"} for i in range(6)]
    valid,reasons,metrics=natural_validation(natural_rows,"paper-scale")
    assert not valid and metrics["median_suppression_db"] is None
    diagnostic=diagnostic_dft(natural_rows); assert diagnostic["spectral_aggregation"]=="GEOMETRIC_MEAN"
    panel_rows=[{"logical_floor":.01,"logical_initial":.1,"dense_parameter_matrix_allocated":False,
        "paper_physical_error_axis_present":False,"paper_logical_error_axis_present":False,"irreducible_floor_bars_present":False}]
    valid,reasons,metrics=panel_b_validation(panel_rows,"paper-scale")
    assert not valid and metrics["panel_label"]=="SOURCE_STRUCTURED_SYNTHETIC_SCALING_ANALOGUE"
