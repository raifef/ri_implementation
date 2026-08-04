from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from google_rl_reimplementation.google_pure_v7 import ACTIVE_CERTIFICATION_SEEDS, RETIRED_SEEDS
from google_rl_reimplementation.google_pure_v7.config import repository_root
from google_rl_reimplementation.google_pure_v7.controller import (CONTROLLER_MODE, RESOLVED_PARAMETERS,
    require_resolved_controller, resolve_production_controller)
from google_rl_reimplementation.google_pure_v7.gates import gate_from_result
from google_rl_reimplementation.google_pure_v7.hyperparameters import select_passing_configuration
from google_rl_reimplementation.google_pure_v7.natural import natural_ensemble_comparable
from google_rl_reimplementation.google_pure_v7.scorecard import PRIMARY_ARTIFACTS, run_development_scorecard
from google_rl_reimplementation.google_pure_v7.sine import (InvalidSineDiagnostic, classify_bandwidth_cutoff,
    fit_sine_tracking, wrap_phase)
from google_rl_reimplementation.google_pure_v7.snapshot import EXPECTED_V6_HEADLINE, current_v6_headline, snapshot_v6
from google_rl_reimplementation.google_pure_v7.timescale_studies import run_long_step, run_production_repaired_drift


def _known_sine(*, periods=5, period=40, gain=.7, phase=-.3, amplitude=.2):
    time=np.arange(periods*period,dtype=float); omega=2*np.pi/period
    values=.02+amplitude*gain*np.sin(omega*time+phase)
    return time,values,omega


def test_v6_snapshot_reproduces_authoritative_state_and_preserves_seeds():
    assert current_v6_headline()==EXPECTED_V6_HEADLINE
    result=snapshot_v6()
    assert result["status"]=="PASS"
    assert result["active_certification_seeds_unused"] is True
    assert result["retired_seed_state_preserved"] is True


def test_sine_estimator_recovers_known_gain_phase_and_uses_optimum_denominator():
    time,values,omega=_known_sine()
    result=fit_sine_tracking(time,values,optimum_amplitude=.2,omega_radians_per_epoch=omega,burn_in_epochs=0)
    assert result["amplitude_gain"]==pytest.approx(.7,abs=1e-10)
    assert wrap_phase(result["phase_radians"]+.3)==pytest.approx(0,abs=1e-10)
    assert result["denominator_source"]=="moving_optimum_amplitude_not_fixed_policy"


def test_zero_optimum_sine_amplitude_is_rejected():
    time,values,omega=_known_sine()
    with pytest.raises(InvalidSineDiagnostic):
        fit_sine_tracking(time,values,optimum_amplitude=0,omega_radians_per_epoch=omega,burn_in_epochs=0)


def test_nonfinite_gain_input_is_rejected():
    time,values,omega=_known_sine(); values[3]=np.nan
    with pytest.raises(InvalidSineDiagnostic):
        fit_sine_tracking(time,values,optimum_amplitude=.2,omega_radians_per_epoch=omega,burn_in_epochs=0)


def test_fewer_than_three_complete_periods_is_rejected():
    time,values,omega=_known_sine(periods=2)
    with pytest.raises(InvalidSineDiagnostic):
        fit_sine_tracking(time,values,optimum_amplitude=.2,omega_radians_per_epoch=omega,burn_in_epochs=0)


def test_rank_deficient_phase_fit_is_rejected():
    time=np.arange(20,dtype=float); values=np.ones(20)
    with pytest.raises(InvalidSineDiagnostic):
        fit_sine_tracking(time,values,optimum_amplitude=.2,omega_radians_per_epoch=2*np.pi,burn_in_epochs=0)


def test_null_or_nonfinite_cutoff_is_rejected_and_out_of_sweep_is_bounded():
    with pytest.raises(InvalidSineDiagnostic): classify_bandwidth_cutoff([.2,.5],[1.0,None])
    with pytest.raises(InvalidSineDiagnostic): classify_bandwidth_cutoff([.2,.5],[1.0,np.inf])
    result=classify_bandwidth_cutoff([.2,.5],[.6,.4])
    assert result["classification"]=="BELOW_SWEEP" and result["value"] is not None


def test_artifact_status_pass_cannot_override_quantitative_failure():
    artifact={"status":"PASS","suppression":.9}
    gate=gate_from_result(artifact,required_fields=("status","suppression"),mechanism_checks=(True,),
                          performance_checks=(artifact["suppression"]>1,),performance_reasons=("suppression failed",))
    assert gate.artifact_complete and gate.mechanism_valid
    assert not gate.performance_pass and not gate.passes


def test_no_passing_hyperparameter_set_returns_no_configuration():
    contract={"hard_filters":{"no_drift_stationarity_max_damage":.004,"slow_drift_suppression_min_exclusive":1.,
        "integrated_excess_ratio_max_exclusive":1.,"natural_drift_suppression_min_exclusive":0.,
        "candidate_damage_max":.004,"clipping_fraction_max":.5,"scaling_deterioration_max":.15}}
    failed={"candidate_id":"least-bad","no_drift_mean_damage":0.,"objective_and_units_valid":True,
        "amplitude_gain":.1,"phase_radians":0.,"slow_suppression_factor":.99,"slow_suppression_ci_95":[.8,1.2],
        "integrated_excess_ratio":1.01,"natural_suppression_db":.1,"candidate_damage":1e-4,
        "clipping_fraction":.1,"scaling_deterioration":.01,"recovery_pass":True}
    result=select_passing_configuration([failed],contract)
    assert result["status"]=="NO_PASSING_CONFIGURATION" and result["selected"] is None


def test_resolved_controller_is_explicit_source_correct_and_unique():
    result=resolve_production_controller()
    assert result["controller_mode"]==CONTROLLER_MODE
    assert result["parameters"]==RESOLVED_PARAMETERS
    assert result["legacy_v5_defaults_used"] is False and result["all_parameters_explicit"] is True
    assert require_resolved_controller()["resolved_config_hash"]==result["resolved_config_hash"]


def test_unresolved_controller_blocks_final_tests(monkeypatch):
    import google_rl_reimplementation.google_pure_v7.controller as module
    monkeypatch.setattr(module,"read_artifact",lambda name:{"controller_mode":CONTROLLER_MODE})
    with pytest.raises(RuntimeError): require_resolved_controller()


def test_legacy_objective_blocks_final_benchmark(monkeypatch):
    controller=resolve_production_controller(); template={"resolved_config_hash":controller["resolved_config_hash"],
        "mechanism_valid":True,"performance_pass":True,"objective_mode":CONTROLLER_MODE}
    import google_rl_reimplementation.google_pure_v7.timescale_studies as module
    def fake(name):
        value=dict(template)
        if name=="timescale_matched_sine": value["objective_mode"]="legacy_v5_component_clipping_diagnostic_only"
        return value
    monkeypatch.setattr(module,"read_artifact",fake)
    with pytest.raises(RuntimeError): run_production_repaired_drift()


def test_incomplete_natural_ensemble_blocks_direct_comparison():
    valid,reason=natural_ensemble_comparable([{"raw_trace_hash":str(i)} for i in range(3)])
    assert not valid and "incomplete" in reason


def _scorecard_fixture(controller_hash:str, *, natural_pass=True, different_hash=False, legacy=False):
    artifacts={}
    for name in PRIMARY_ARTIFACTS:
        artifacts[name]={"artifact_complete":True,"mechanism_valid":True,"performance_pass":True,
                         "blocking_reasons":[],"resolved_config_hash":controller_hash,
                         "objective_mode":CONTROLLER_MODE,"status":"PASS"}
    if not natural_pass:
        artifacts["natural_drift_full_ensemble"].update(performance_pass=False,blocking_reasons=["material regression"])
    if different_hash: artifacts["scaling_final_controller"]["resolved_config_hash"]="different"
    if legacy: artifacts["timescale_matched_sine"]["objective_mode"]="legacy_v5_component_clipping_diagnostic_only"
    return artifacts


def test_natural_material_regression_blocks_certification(monkeypatch):
    controller=resolve_production_controller(); artifacts=_scorecard_fixture(controller["resolved_config_hash"],natural_pass=False)
    import google_rl_reimplementation.google_pure_v7.scorecard as module
    monkeypatch.setattr(module,"read_artifact",lambda name:artifacts[name])
    result=run_development_scorecard()
    assert result["certification_ready"] is False
    assert result["outcome_class"]=="NATURAL_DRIFT_RETENTION_FAILURE"


def test_different_controller_hashes_block_certification(monkeypatch):
    controller=resolve_production_controller(); artifacts=_scorecard_fixture(controller["resolved_config_hash"],different_hash=True)
    import google_rl_reimplementation.google_pure_v7.scorecard as module
    monkeypatch.setattr(module,"read_artifact",lambda name:artifacts[name])
    result=run_development_scorecard()
    assert not result["certification_ready"] and not result["one_resolved_controller_hash"]


def test_legacy_objective_blocks_certification(monkeypatch):
    controller=resolve_production_controller(); artifacts=_scorecard_fixture(controller["resolved_config_hash"],legacy=True)
    import google_rl_reimplementation.google_pure_v7.scorecard as module
    monkeypatch.setattr(module,"read_artifact",lambda name:artifacts[name])
    result=run_development_scorecard()
    assert not result["certification_ready"] and not result["legacy_objective_absent"]


def test_smoke_run_is_explicitly_excluded_from_performance_gate():
    resolve_production_controller()
    result=run_long_step(smoke=True,epochs=48)
    assert result["run_class"]=="SMOKE_TEST_ONLY"
    assert result["performance_pass"] is False


def test_long_commands_require_explicit_execute():
    resolve_production_controller()
    with pytest.raises(RuntimeError): run_long_step(smoke=False,execute=False,epochs=64)


def test_v7_imports_no_v5_runtime_and_registers_all_commands():
    source=repository_root()/"src/google_rl_reimplementation/google_pure_v7"
    for path in source.glob("*.py"):
        tree=ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                assert "google_pure_v5" not in ast.unparse(node)
    text=(repository_root()/"pyproject.toml").read_text(encoding="utf-8")
    commands=("snapshot-v6","supersede-certification","validate-scientific-gates","resolve-production-controller",
        "validate-sine-estimator","run-long-step-smoke","run-long-step","freeze-timescale-sine",
        "run-timescale-sine-smoke","run-timescale-sine","run-timescale-strobe","run-production-repaired-drift",
        "run-replay-age-audit","run-natural-ablation","run-full-natural-ensemble","run-hyperparameter-study",
        "run-exploration-study","run-final-recovery","run-final-scaling","run-development-scorecard",
        "freeze-certification","run-certification")
    assert all(f"google-rl-v7-{name}" in text for name in commands)


def test_seed_cohort_is_retained_and_retired_seed_stays_retired():
    assert ACTIVE_CERTIFICATION_SEEDS==tuple(range(12101,12113))
    assert RETIRED_SEEDS==(10101,)
