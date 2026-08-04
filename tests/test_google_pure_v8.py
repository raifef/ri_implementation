from pathlib import Path
import numpy as np
import pytest
from google_rl_reimplementation.google_pure_v6.plant import default_spec
from google_rl_reimplementation.google_pure_v6.policy import FactorizedGaussianPolicy
from google_rl_reimplementation.google_pure_v6.update import ppo_objective_and_gradient
from google_rl_reimplementation.google_pure_v8.contracts import normalized_edr_improvement,cost_decomposition,local_ratio,frequency_contract
from google_rl_reimplementation.google_pure_v8.diagnostics import _freeze_batch,run_cell
from google_rl_reimplementation.google_pure_v8.audits import audit_baselines,audit_native_units

def test_edr_endpoints_and_affine_invariance():
 assert normalized_edr_improvement(4,4,2)==0 and normalized_edr_improvement(4,2,2)==1
 base=normalized_edr_improvement(4,3,2)
 for a,b in ((2,7),(.25,-3),(10,0)): assert normalized_edr_improvement(a*4+b,a*3+b,a*2+b)==pytest.approx(base)
def test_edr_denominator_fails_closed():
 for values in ((1,0,1),(1,0,2),(np.nan,0,1)):
  with pytest.raises(ValueError): normalized_edr_improvement(*values)
def test_cost_decomposition_identity():
 x=cost_decomposition(5,3,2.5,1);assert x["direct_improvement"]==pytest.approx(x["decomposition_improvement"])
def test_local_ratio_direction_and_exact_mask():
 actions=np.array([[0.,9.]]);mask=np.array([[True,False]]);z=np.zeros(2);ls=np.zeros(2)
 assert local_ratio(actions,z,ls,z,ls,mask)[0,0]==pytest.approx(1)
 assert local_ratio(actions,z,ls,np.array([1.,0.]),ls,mask)[0,0]>1
 assert local_ratio(actions,np.array([1.,0.]),ls,z,ls,mask)[0,0]<1
 assert local_ratio(actions,np.array([0.,20.]),ls,z,ls,mask)[0,0]==pytest.approx(1)
def test_collection_snapshot_is_actually_read_only():
 spec=default_spec(3);policy=FactorizedGaussianPolicy(np.zeros(3),spec.coordinates,initial_scale=.1,seed=1);batch=policy.sample(4,policy_version=0,epoch=0,environment_time=0,graph_version="x");_freeze_batch(batch)
 with pytest.raises(ValueError):batch.collection_mean[0]=3
 with pytest.raises(ValueError):batch.collection_component_log_probability[0,0]=3
def test_global_entropy_exactly_once_not_detector_degree():
 spec=default_spec(6);policy=FactorizedGaussianPolicy(np.zeros(6),spec.coordinates,initial_scale=.1,seed=2);batch=policy.sample(8,policy_version=0,epoch=0,environment_time=0,graph_version="x");adv=np.zeros((8,spec.mask.shape[0]));_,_,gs,diag=ppo_objective_and_gradient(batch.latent_normalized_actions,adv,spec.mask,np.zeros(6),np.log(np.full(6,.1)),batch.collection_component_log_probability,clip=.2,entropy_coefficient=.03)
 assert np.allclose(gs,.03) and diag["effective_entropy_gradient"]==[.03]*6
def test_matched_finite_shot_policy_accounting():
 cell=run_cell(frequency=1/12,epochs=24,candidates=6,cycles=3000,seed=14901);rows=cell["policy_accounting"];cycles={r["effective_qec_cycles"] for r in rows.values()}
 assert len(cycles)==1 and cell["evaluation_protocol"].startswith("matched independent finite-shot")
 assert all(r["edr_per_cycle"]==pytest.approx(r["raw_detector_event_count"]/r["effective_qec_cycles"]) for r in rows.values())
 assert cell["behaviour_snapshots_immutable"] and len(cell["candidate_native_variance"])==6
def test_frequency_units_and_native_contracts():
 x=frequency_contract(.25);assert x["period_epochs"]==4 and x["angular_frequency_radians_per_epoch"]==pytest.approx(np.pi/2)
 assert audit_native_units()["classification"]=="PASS" and audit_baselines()["classification"]=="PASS"
def test_pure_v8_has_no_outside_workflow_controller_imports():
 text="\n".join(p.read_text(encoding="utf-8") for p in Path("src/google_rl_reimplementation/google_pure_v8").glob("*.py"));assert "google_rl_reimplementation.stage" not in text and "google_rl_reimplementation.product" not in text
def test_all_v8_cli_entries_registered():
 text=Path("pyproject.toml").read_text(encoding="utf-8")
 for name in ("snapshot","audit-mathematical-contracts","audit-figure5a-edr","run-figure5a-feasibility","audit-exploration-floor","audit-entropy-scale","audit-native-units","audit-clipping-likelihood","audit-ppo-lifecycle","audit-baselines","audit-temporal-protocol","run-compact-fault-matrix","report-root-cause","status"):
  assert f"google-rl-v8-{name} =" in text
