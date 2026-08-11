import json
from pathlib import Path
import numpy as np
import pytest
from hdfa_rl_suite.google_pure_evidence_v8.evidence_contracts import EvidenceGate
from hdfa_rl_suite.google_pure_evidence_v8.experiment_families import ExperimentFamily,forbid_joint_score,require_control_only
from hdfa_rl_suite.google_pure_evidence_v8.step_response import estimate_target_response,run_step_response
from hdfa_rl_suite.google_pure_evidence_v8.recovery import run_recovery
from hdfa_rl_suite.google_pure_evidence_v8.natural_drift import run_natural_drift
from hdfa_rl_suite.google_pure_evidence_v8.figure5b import DISTANCES,run_figure5b
from hdfa_rl_suite.google_pure_evidence_v8.figure5c import run_figure5c
from hdfa_rl_suite.google_pure_evidence_v8.claim_registry import build_claim_registry,validate_scorecard
from hdfa_rl_suite.google_pure_evidence_v8.hdfa_readiness import report_hdfa_readiness
from hdfa_rl_suite.google_pure_evidence_v8.manifest_validation import build_protocol_preflight

def test_artifact_alone_is_not_final_evidence():
 assert not EvidenceGate("x",True,False,False,False,"CLAIM_NOT_SUPPORTED").final_evidence
def test_invalid_diagnostic_can_never_be_final():
 with pytest.raises(ValueError):EvidenceGate("x",True,True,True,True,"INVALID_DIAGNOSTIC")
def test_exact_public_reproduction_requires_every_gate():
 with pytest.raises(ValueError):EvidenceGate("x",True,True,False,False,"PUBLIC_DATA_EXACT_REPRODUCTION")
 assert EvidenceGate("x",True,True,True,True,"PUBLIC_DATA_EXACT_REPRODUCTION").final_evidence
def test_every_cross_family_score_is_rejected_and_decoder_modes_cannot_mix():
 with pytest.raises(RuntimeError):forbid_joint_score([ExperimentFamily.FIGURE5B_SPARSE_SCALING.value,ExperimentFamily.STEP_RESPONSE_INJECTED_DRIFT.value])
 with pytest.raises(RuntimeError):require_control_only([{"decoder_assistance":"CONTROL_ONLY"},{"decoder_assistance":"DECODER_ASSISTED"}])
def test_target_response_does_not_normalize_by_failed_achievement():
 trace=np.r_[np.zeros(20),np.linspace(0,.1,80)];result=estimate_target_response(trace,onset=20,target=1)
 assert result["response_time_90_epochs"] is None and result["final_residual"]>.8
def test_step_bundle_contains_target_candidate_fixed_and_residual():
 r=run_step_response();data=np.load("artifacts/google_pure_evidence_v8/step_response/raw_traces.npz")
 assert r["piecewise_constant_optimum_verified"] and {"candidate_projection","fixed_projection","residual","optimum"}<=set(data.files)
 assert Path("artifacts/google_pure_evidence_v8/step_response/protocol.json").exists() and not r["evidence_gate"]["final_evidence"]
def test_recovery_is_real_policy_spoil_with_fraction_threshold_and_censoring():
 r=run_recovery();data=np.load("artifacts/google_pure_evidence_v8/recovery/raw_traces.npz")
 assert r["spoil_protocol"].startswith("explicit bounded randomized") and any(k.endswith("fractional_recovery") for k in data.files)
 assert all({"spoiled_policy_hash","independent_floor","threshold_90pct","censored","exponential_fit"}<=row.keys() for row in r["rows"])
def test_natural_drift_is_paired_and_smoke_fails_identifiability():
 r=run_natural_drift();data=np.load("artifacts/google_pure_evidence_v8/natural_drift/raw_traces.npz")
 assert any(k.endswith("fixed_policy") for k in data.files) and any(k.endswith("oracle_optimum") for k in data.files)
 assert r["sensitivity_records"] and not r["low_frequency_identifiable"] and r["evidence_gate"]["evidence_status"]=="INVALID_DIAGNOSTIC"
def test_figure5b_uses_exact_distances_and_both_figures():
 r=run_figure5b();assert sorted(set(x["distance"] for x in r["rows"]))==list(DISTANCES) and r["distance_15_control_count"]==38670
 assert Path("artifacts/google_pure_evidence_v8/figure5b/paper_axes_figure.png").exists() and Path("artifacts/google_pure_evidence_v8/figure5b/normalized_diagnostic.png").exists()
def test_figure5c_has_21_independent_phase_and_time_fits():
 r=run_figure5c();assert r["cell_count"]==21 and len(r["distance_independence_tables"])==3 and not r["unfavorable_points_dropped"]
 assert all({"phase_space_constrained","phase_space_free","phase_space_robust","time_domain","negative_derivatives_dropped"}<=row.keys() for row in r["rows"])
 assert Path("artifacts/google_pure_evidence_v8/figure5c/sensitivity.png").exists()
def test_claim_registry_has_required_fields_and_cannot_form_scorecard():
 r=build_claim_registry();required={"paper_quantity","paper_value","paper_uncertainty","comparison_legitimacy","same_run_required","cannot_be_jointly_scored_with","status"};assert all(required<=row.keys() for row in r["rows"])
 with pytest.raises(RuntimeError):validate_scorecard(r["rows"][:2])
def test_readiness_is_derived_and_comparison_stays_blocked():
 r=report_hdfa_readiness();assert not r["definitive_comparison_permitted"] and r["outcome"]=="HDFA_COMPARISON_NOT_CAUSALLY_IDENTIFIABLE" and not all(r["matching_checks"].values())
def test_protocol_preflight_validates_files_and_gates():
 r=build_protocol_preflight();assert r["protocol_gate_pass"] and not r["reference_acquisition_permitted"]
def test_all_evidence_cli_entries_registered():
 text=Path("pyproject.toml").read_text(encoding="utf-8")
 for suffix in ("build-contracts","validate-manifests","run-natural-drift","run-step-response","run-recovery","run-figure5b","run-figure5c","build-claim-registry","build-paper-comparison","report-hdfa-readiness","status"):assert f"hdfa-google-evidence-v8-{suffix} =" in text
