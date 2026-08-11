from __future__ import annotations

from pathlib import Path

import numpy as np

from hdfa_rl_suite.google_pure_paper_reproduction.experiment_families import ExperimentFamily
from hdfa_rl_suite.google_pure_paper_reproduction.paper_figures import build_protocol
from hdfa_rl_suite.google_pure_paper_reproduction.randomized_recovery import validation as recovery_validation
from hdfa_rl_suite.google_pure_paper_reproduction.step_response import validation as step_validation
from hdfa_rl_suite.google_pure_source_exact.identity import build_direct_sigma_identity
from hdfa_rl_suite.google_pure_source_exact.paper_families.common import SparseControlPlant, run_direct_sigma_trace
from hdfa_rl_suite.google_pure_source_exact.step_response_130.estimator import estimate_response


ROOT=Path(__file__).resolve().parents[2]


def test_all_synthetic_family_protocols_freeze_direct_sigma_and_amended_identities():
    expected=build_direct_sigma_identity(ROOT)
    families=[family for family in ExperimentFamily if family.value not in {
        "PUBLIC_ENDPOINT_DATA_REPRODUCTION","PUBLIC_TABLE_REPRODUCTION"}]
    for family in families:
        protocol=build_protocol(family.value,mode="reference")
        assert protocol["controller_mode"]=="PAPER_DIRECT_SIGMA"
        assert protocol["parameterization"]=="direct_sigma"
        assert protocol["controller_hash"]==expected["controller_hash"]
        assert protocol["controller_code_hash"]==expected["controller_code_hash"]
        assert protocol["execution_path"]=="AMENDED_DIRECT_SIGMA_SOURCE_STRUCTURED_ANALOGUE"
        assert protocol["plant_hash"] and protocol["graph_hash"]


def test_sparse_runtime_executes_direct_sigma_without_dense_mask_or_target_access(tmp_path):
    plant=SparseControlPlant(3,41,8,seed=17)
    target=np.linspace(-.1,.1,41)
    result=run_direct_sigma_trace(plant=plant,protocol_hash="tiny",seed=19,epochs=3,
        candidates=4,cycles_per_candidate=200,entropy_weight=.001,
        checkpoint=tmp_path/"trace.json",target_at_epoch=lambda _:target,
        initial_mean=np.full(41,.3))
    assert result["complete"] and result["candidate_qec_cycles"]==2400
    assert not result["dense_parameter_matrix_allocated"] and not result["controller_observed_target"]
    assert all(row["controller_mode"]=="PAPER_DIRECT_SIGMA" for row in result["records"])
    assert all(row["coordinate_ratios_clipped_before_sparse_product"] for row in result["records"])
    assert all(row["baseline_mode"]=="JOINT_LEARNED_DETECTOR_BASELINE" for row in result["records"])


def test_step_estimator_never_uses_observed_final_excursion_as_target():
    onset=20;trace=np.zeros(240);trace[onset:]=.3*(1-np.exp(-np.arange(220)/35))
    result=estimate_response(trace,onset_epoch=onset,bootstrap_samples=20)
    assert result["response_fraction_of_injected_target"]<.35
    assert result["response_time_90_epochs"] is None
    assert result["response_classification"]=="TARGET_90_NOT_REACHED_WITHIN_HORIZON"


def test_censoring_and_non_public_analogue_never_promote():
    recovery_rows=[{"not_a_step_response":True,"randomized_fraction":.5,"recovery_epoch":None,
        "controller_mode":"PAPER_DIRECT_SIGMA","parameterization":"direct_sigma",
        "source_structure_match":True} for _ in range(3)]
    valid,_,metrics=recovery_validation(recovery_rows,"reference")
    assert not valid and metrics["censored_count"]==3 and not metrics["paper_comparable"]
    response={"response_time_90_epochs":None,"response_fraction_of_injected_target":.3,"fit_valid":False}
    step_rows=[{"policy_spoil_applied":False,"controller_mode":"PAPER_DIRECT_SIGMA",
        "parameterization":"direct_sigma","source_structure_match":True,"response":response}]
    valid,_,metrics=step_validation(step_rows,"reference")
    assert not valid and metrics["outcome"]=="STEP_TARGET_NOT_REACHED_WITHIN_HORIZON"
    assert not metrics["paper_comparable"]
