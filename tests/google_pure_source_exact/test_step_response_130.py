import numpy as np
import pytest
from hdfa_rl_suite.google_pure_source_exact.step_response_130 import StepProtocol, SourceStepPlant, build_control_inventory, build_run_plan, estimate_response
from hdfa_rl_suite.google_pure_source_exact.step_response_130.contracts import controller_observation
from hdfa_rl_suite.google_pure_source_exact.step_response_130.acquisition import run_step_analogue
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.optimizer import OptimizerConfig

def test_inventory_and_reference_budget_are_exact():
    inventory = build_control_inventory()
    assert inventory["control_count"] == 924
    assert sum(row["is_injected_direction"] for row in inventory["coordinates"]) == 1
    plan = build_run_plan(StepProtocol("reference", certification=True))
    assert plan["source_budget"]["total_training_effective_cycles"] == 2_880_000_000
    assert "figure5a_metric" in plan["explicit_exclusions"]
    with pytest.raises(ValueError): build_run_plan(StepProtocol("bad", candidates_per_epoch=39, certification=True))

def test_controller_has_no_oracle_fields():
    assert set(controller_observation([0.0]*924, [1.0])) == {"policy_mean", "detector_rewards"}

def test_hidden_step_is_piecewise_constant_and_physical_anchors_hold():
    plant=SourceStepPlant(onset_epoch=5)
    assert np.all(plant.hidden_target(4)==0)
    assert plant.hidden_target(5)[0]==pytest.approx(.5)
    fixed=np.zeros(924); oracle=plant.hidden_target(5)
    assert plant.expected_edr(fixed,5).sum()>plant.expected_edr(oracle,5).sum()
    assert plant.expected_edr(fixed,5,drift_enabled=False).sum()==pytest.approx(plant.expected_edr(oracle,5).sum())

def test_acquisition_checkpoints_at_candidate_boundary(tmp_path):
    protocol=StepProtocol("smoke",candidates_per_epoch=3,cycles_per_candidate=200,epochs=30,onset_epoch=5)
    result=run_step_analogue(protocol,tmp_path/"checkpoint.json",OptimizerConfig(.001,.0001,.01),max_candidate_boundaries=2)
    assert not result["complete"] and result["next_candidate"]==2
    resumed=run_step_analogue(protocol,tmp_path/"checkpoint.json",OptimizerConfig(.001,.0001,.01),resume=True,max_candidate_boundaries=1)
    assert not resumed["complete"] and resumed["next_candidate"]==3

def test_compact_epoch_checkpointing_preserves_directional_result(tmp_path):
    protocol=StepProtocol("smoke",candidates_per_epoch=1,cycles_per_candidate=20,epochs=23,onset_epoch=3)
    optimizer=OptimizerConfig(.001,.0001,.01)
    full=run_step_analogue(protocol,tmp_path/"full.json",optimizer)
    compact=run_step_analogue(protocol,tmp_path/"compact.json",optimizer,
                              checkpoint_every_candidates=1,compact_records=True)
    assert compact["record_storage"]=="compact_directional"
    assert compact["checkpoint_every_candidates"]==1
    assert "mean_after" not in compact["records"][0]
    assert "mean_after_direction" in compact["records"][0]
    assert compact["response"]==full["response"]
    assert [row["learned_mean_edr"] for row in compact["records"]] == [
        row["learned_mean_edr"] for row in full["records"]]

def test_fit_recovers_tau_and_rejects_unsettled_trace():
    rng = np.random.default_rng(4); onset, tau = 30, 55.; t = np.arange(420)
    y = .1 + .7*(1-np.exp(-np.maximum(t-onset, 0)/tau)) + rng.normal(0, .006, len(t))
    result = estimate_response(y, onset_epoch=onset, bootstrap_samples=50)
    assert result["fit_valid"] and result["tau_ci_95"][0] < tau < result["tau_ci_95"][1]
    assert all(not row["censored"] for row in result["crossings"])
    short = 1-np.exp(-np.maximum(np.arange(70)-10, 0)/500)
    rejected = estimate_response(short, onset_epoch=10, bootstrap_samples=20)
    assert not rejected["fit_valid"] and rejected["requires_extension"]
