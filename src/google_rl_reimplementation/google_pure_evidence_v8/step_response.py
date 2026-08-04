"""Target-relative injected-step response with an explicit persistent optimum."""
from __future__ import annotations
import numpy as np
from google_rl_reimplementation.google_pure_v6.plant import PureQuadraticPlant,default_spec
from google_rl_reimplementation.google_pure_v6.reference_agent import evidence_from_counts
from google_rl_reimplementation.google_pure_v7.config import canonical_hash
from google_rl_reimplementation.google_pure_v7.controller import build_production_agent,require_resolved_controller
from .common import bundle_complete,prompt1_report,require_reference_authorization,root,write
from .evidence_contracts import EvidenceGate
from .experiment_families import ExperimentFamily,family_metadata

def _crossing(values:np.ndarray,threshold:float)->int|None:
    hit=np.flatnonzero(values>=threshold);return int(hit[0]) if len(hit) else None

def estimate_target_response(response:np.ndarray,*,onset:int,target:float=1.0,sustained:int=12)->dict:
    values=np.asarray(response,float);pre=float(np.mean(values[max(0,onset-10):onset]));post=values[onset:];time=np.arange(len(post),dtype=float)
    crossings={f"response_time_{label}_epochs":_crossing(post,pre+fraction*(target-pre)) for label,fraction in (("50",.5),("63_2",.632),("90",.9))}
    tolerance=.05*abs(target-pre);settled=None
    for i in range(max(0,len(post)-sustained+1)):
        if np.all(np.abs(post[i:i+sustained]-target)<=tolerance):settled=i;break
    taus=np.geomspace(1,max(10,2*len(post)),500);fits=[]
    for tau in taus:
        basis=np.column_stack((np.ones(len(time)),np.exp(-time/tau)));beta,_,rank,_=np.linalg.lstsq(basis,post,rcond=None)
        if rank==2:
            residual=post-basis@beta;fits.append((float(residual@residual),float(tau),beta))
    sse,tau,beta=min(fits,key=lambda x:x[0]);total=float(np.sum((post-np.mean(post))**2));r2=1-sse/total if total>0 else 1.;variance=sse/max(1,len(post)-3);accepted=[x[1] for x in fits if x[0]<=sse+3.841458820694124*variance]
    fit={"valid":bool(r2>=.8 and tau>taus[0]*1.001 and tau<taus[-1]/1.001),"tau_epochs":tau,"tau_profile_confidence_interval_95_epochs":[float(min(accepted)),float(max(accepted))],
      "r_infinity":float(beta[0]),"r_zero_minus_r_infinity":float(beta[1]),"fit_r_squared":float(r2),"fit_sse":sse,"credibility_gate":"R2>=0.8 and tau interior to preregistered grid"}
    return {"pre_step_response":pre,"target_response":target,"final_response":float(np.mean(post[-max(10,len(post)//10):])),"final_residual":float(target-np.mean(post[-max(10,len(post)//10):])),
      **crossings,"settling_time_95_epochs":settled,"settling_tolerance_absolute":tolerance,"overshoot":float(max(0,np.max(post)-target)),
      "integrated_absolute_tracking_error":float(np.sum(np.abs(target-post))),"exponential_fit":fit,"response_classification":"SETTLED" if settled is not None else "NO_SETTLING_WITHIN_HORIZON"}

def run_step_response(*,mode:str="smoke",execute:bool=False)->dict:
    require_reference_authorization(mode,execute);epochs,candidates,cycles=(180,12,3000) if mode=="smoke" else (720,40,100000);onset=epochs//4
    spec=default_spec(6);plant=PureQuadraticPlant(spec);direction=np.linspace(1,.45,6);direction/=np.linalg.norm(direction);delta=.35*direction
    tape=np.zeros((epochs,6));tape[onset:]=delta
    if not (np.allclose(tape[:onset],0) and np.allclose(tape[onset:],delta)):raise RuntimeError("step optimum is not piecewise constant")
    agent=build_production_agent(plant.mask,spec.base_optimum_normalized,spec.coordinates,seed=15221);rng=np.random.default_rng(115221)
    means=[];candidate_means=[];candidate_risk=[];scale=[]
    for optimum in tape:
        mean=agent.mean.copy();batch=agent.sample(candidates);counts=plant.acquire_counts(batch.applied_native_actions,spec.coordinates.to_native(optimum),cycles=cycles,rng=rng)
        means.append(mean);candidate_means.append(np.mean(batch.applied_normalized_actions,axis=0));candidate_risk.append(float(np.mean(plant.logical_risk_native(batch.applied_native_actions,spec.coordinates.to_native(optimum)))));scale.append(agent.scale.copy())
        agent.update(batch,evidence_from_counts(batch,counts,cycles))
    means=np.asarray(means);candidate_means=np.asarray(candidate_means);pre=means[:onset].mean(axis=0);den=float(delta@delta)
    projection=(means-pre)@delta/den;candidate_projection=(candidate_means-pre)@delta/den;fixed=np.zeros(epochs);optimum_projection=(tape@delta)/den;residual=1-projection
    response=estimate_target_response(projection,onset=onset,target=1.0,sustained=max(8,epochs//20));target=root()/"step_response";target.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target/"raw_traces.npz",optimum=tape,learned_mean=means,projection=projection,residual=residual,candidate_projection=candidate_projection,fixed_projection=fixed,optimum_projection=optimum_projection,candidate_risk=candidate_risk,policy_scale=scale)
    protocol={"schema_version":"google-pure-evidence-v8-step-protocol.v2","mode":mode,"onset_epoch":onset,"epochs":epochs,"pre_optimum":np.zeros(6).tolist(),"post_optimum":delta.tolist(),"weighting_matrix":"identity","coordinates":"normalized","piecewise_constant_verified":True,"target_response":1.0}
    protocol=write("protocol",protocol,"Injected-step Protocol",directory=target)
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    fig,(ax0,ax1)=plt.subplots(2,1,figsize=(9,7),sharex=True,constrained_layout=True);ax0.plot(optimum_projection,"k--",label="injected optimum");ax0.plot(projection,label="learned mean");ax0.plot(candidate_projection,alpha=.7,label="candidate stream");ax0.plot(fixed,label="fixed");ax0.legend();ax0.set_ylabel("normalized response")
    ax1.plot(residual,label="residual 1-R");ax1.axhline(.1,color="tab:red",ls="--",label="90% threshold");ax1.axhline(0,color="k",lw=.6);ax1.set(xlabel="epoch",ylabel="target residual");ax1.legend();fig.savefig(target/"figure.png",dpi=180);plt.close(fig)
    result={"schema_version":"google-pure-evidence-v8-step-results.v2",**family_metadata(ExperimentFamily.STEP_RESPONSE_INJECTED_DRIFT),"mode":mode,"controller_hash":require_resolved_controller()["resolved_config_hash"],
      "protocol_hash":protocol["artifact_hash"],"plant_hash":canonical_hash({"mask":spec.mask.tolist(),"curvature":spec.normalized_curvature.tolist()}),"graph_hash":canonical_hash(spec.mask.tolist()),"seed_registry_hash":canonical_hash([15221]),
      "observable_definition":"target-relative W-weighted projection delta^T W(mu-mu_pre)/(delta^T W delta), W=I","evaluation_budget":{"epochs":epochs,"candidates":candidates,"cycles_per_candidate":cycles},
      "optimum_trajectory_stored":True,"piecewise_constant_optimum_verified":True,"projection_definition":"projection = delta^T W(mu-mu_pre)/(delta^T W delta), W=I","response":response,"final_vector_residual":float(np.linalg.norm(means[-1]-delta)),"candidate_stream_response_stored":True,"fixed_baseline_stored":True,
      "prompt1_hash":prompt1_report()["artifact_hash"]}
    complete,missing=bundle_complete(target,["raw_traces.npz","protocol.json","protocol.md","figure.png"],result,["response","projection_definition","prompt1_hash"])
    blockers=["PROMPT1_GATE_NOT_PASSED"] if not prompt1_report()["prompt1_gate_pass"] else []
    if mode=="smoke":blockers.append("SMOKE_NOT_REFERENCE_EVIDENCE")
    gate=EvidenceGate("step.injected_persistent_optimum",complete,True,False,False,"SYNTHETIC_MECHANISM_EVIDENCE",tuple(blockers+missing));result["evidence_gate"]=gate.to_dict();result["blocking_reasons"]=list(gate.blocking_reasons)
    return write("results",result,"Injected-drift Step-response Results",directory=target,json_name="results.json",md_name="report.md")
