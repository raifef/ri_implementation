"""Recovery from an explicitly spoiled, bounded production-policy mean."""
from __future__ import annotations
import numpy as np
from hdfa_rl_suite.google_pure_v6.plant import PureQuadraticPlant,default_spec
from hdfa_rl_suite.google_pure_v6.reference_agent import PureGoogleV6Agent,evidence_from_counts
from hdfa_rl_suite.google_pure_v7.config import canonical_hash
from hdfa_rl_suite.google_pure_v7.controller import agent_choices,require_resolved_controller
from .common import bundle_complete,prompt1_report,require_reference_authorization,root,write
from .evidence_contracts import EvidenceGate
from .experiment_families import ExperimentFamily,family_metadata
from .uncertainty import bootstrap_interval,wilson_interval

def _cross(values:np.ndarray,threshold:float)->int|None:
    hit=np.flatnonzero(np.asarray(values)<=threshold);return int(hit[0]) if len(hit) else None

def _fit_excess(excess:np.ndarray)->dict:
    values=np.asarray(excess,float);idx=np.flatnonzero(values>max(values[0]*.01,1e-12))
    if len(idx)<8:return {"valid":False,"reason":"fewer than eight positive points"}
    x=idx.astype(float);y=np.log(values[idx]);slope,intercept=np.polyfit(x,y,1);pred=intercept+slope*x;sse=float(np.sum((y-pred)**2));total=float(np.sum((y-np.mean(y))**2));r2=1-sse/total if total else 1.;se=float(np.sqrt(sse/max(1,len(x)-2)/max(np.sum((x-np.mean(x))**2),1e-30)))
    valid=bool(slope<0 and r2>=.8);rate=-float(slope)
    return {"valid":valid,"rate_per_epoch":rate,"rate_ci_95":[max(0,float(-slope-1.96*se)),max(0,float(-slope+1.96*se))],"tau_epochs":1/rate if rate>0 else None,"fit_r_squared":float(r2),"point_count":len(x),"credibility_gate":"negative slope, R2>=0.8, at least 8 positive points"}

def run_recovery(*,mode:str="smoke",execute:bool=False)->dict:
    require_reference_authorization(mode,execute);epochs,candidates,cycles,reps=(180,12,3000,3) if mode=="smoke" else (1000,40,100000,12)
    spec=default_spec(6);plant=PureQuadraticPlant(spec);controller=require_resolved_controller();rows=[];curves={};fractions={};mean_vectors={}
    for severity in (.15,.30,.50):
      for rep in range(reps):
        seed=15300+rep+int(severity*100);direction=np.random.default_rng(seed+7000).normal(size=6);direction/=np.linalg.norm(direction);spoiled=severity*direction
        agent=PureGoogleV6Agent(plant.mask,spoiled,spec.coordinates,agent_choices(controller),seed=seed,objective_mode="source_literal_ppo");rng=np.random.default_rng(seed+100000);excess=[];candidate_excess=[];means=[]
        optimum_native=plant.base_optimum_native;floor=float(plant.logical_risk_native(optimum_native[None,:],optimum_native)[0])
        for _ in range(epochs):
          mean=agent.mean.copy();batch=agent.sample(candidates);counts=plant.acquire_counts(batch.applied_native_actions,optimum_native,cycles=cycles,rng=rng)
          mean_risk=float(plant.logical_risk_native(spec.coordinates.to_native(mean)[None,:],optimum_native)[0]);candidate_risk=float(np.mean(plant.logical_risk_native(batch.applied_native_actions,optimum_native)))
          excess.append(max(0,mean_risk-floor));candidate_excess.append(max(0,candidate_risk-floor));means.append(mean);agent.update(batch,evidence_from_counts(batch,counts,cycles))
        excess=np.asarray(excess);fraction=1-excess/max(excess[0],1e-30);threshold=float(.1*excess[0]);crossing=_cross(excess,threshold);fit=_fit_excess(excess);key=f"s{severity:.2f}_r{rep}"
        curves[key]=excess;fractions[key]=fraction;mean_vectors[key]=np.asarray(means)
        rows.append({"seed":seed,"replicate":rep,"nominal_severity":severity,"spoiled_policy_hash":canonical_hash(spoiled.tolist()),"spoiled_policy_vector":spoiled.tolist(),"policy_within_bounds":bool(np.all(np.abs(spoiled)<=1)),
          "measured_initial_degradation":float(excess[0]),"independent_floor":floor,"observation_horizon":epochs,"threshold_90pct":threshold,"crossing_epoch":crossing,"crossing_occurred":crossing is not None,"censored":crossing is None,
          "exponential_fit":fit,"final_residual":float(excess[-1]),"integrated_excess":float(np.sum(excess)),"fractional_recovery_final":float(fraction[-1])})
    summaries=[]
    for severity in (.15,.30,.50):
      keys=[k for k in curves if k.startswith(f"s{severity:.2f}")];stack=np.stack([curves[k] for k in keys]);crossings=[r["crossing_epoch"] for r in rows if r["nominal_severity"]==severity and r["crossing_epoch"] is not None];reached=len(crossings)
      summaries.append({"severity":severity,"replicates":len(keys),"median_excess":np.median(stack,axis=0).tolist(),"q25_excess":np.quantile(stack,.25,axis=0).tolist(),"q75_excess":np.quantile(stack,.75,axis=0).tolist(),
        "median_observed_crossing":float(np.median(crossings)) if crossings else None,"crossing_ci_95":bootstrap_interval(crossings) if len(crossings)>1 else None,"reached_fraction":reached/len(keys),"reached_fraction_wilson_95":wilson_interval(reached,len(keys))})
    target=root()/"recovery";target.mkdir(parents=True,exist_ok=True);np.savez_compressed(target/"raw_traces.npz",**{f"{k}__excess":v for k,v in curves.items()},**{f"{k}__fractional_recovery":v for k,v in fractions.items()},**{f"{k}__mean":v for k,v in mean_vectors.items()})
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5));colors={.15:"tab:blue",.3:"tab:orange",.5:"tab:green"}
    for row in rows:
      key=f"s{row['nominal_severity']:.2f}_r{row['replicate']}";ax.plot(curves[key],color=colors[row["nominal_severity"]],alpha=.2)
      if row["crossing_epoch"] is not None:ax.plot(row["crossing_epoch"],row["threshold_90pct"],"o",color=colors[row["nominal_severity"]],ms=3)
    for s in summaries:
      med=np.asarray(s["median_excess"]);q1=np.asarray(s["q25_excess"]);q3=np.asarray(s["q75_excess"]);t=np.arange(epochs);ax.plot(t,med,color=colors[s["severity"]],label=f"severity {s['severity']}");ax.fill_between(t,q1,q3,color=colors[s["severity"]],alpha=.15);ax.axhline(.1*med[0],color=colors[s["severity"]],ls="--",lw=.7)
    positive_floor=.5*min(float(np.min(v[v>0])) for v in curves.values() if np.any(v>0));ax.axhline(positive_floor,color="k",lw=.7,label="oracle floor (0; displayed at plotting epsilon)");ax.set(xlabel="epoch",ylabel="excess logical risk above oracle floor");ax.set_yscale("log");ax.legend();fig.tight_layout();fig.savefig(target/"figure.png",dpi=180);plt.close(fig)
    result={"schema_version":"google-pure-evidence-v8-recovery-results.v2",**family_metadata(ExperimentFamily.RANDOMIZED_RECOVERY_AFTER_SPOIL),"mode":mode,"controller_hash":controller["resolved_config_hash"],
      "protocol_hash":canonical_hash({"epochs":epochs,"severities":[.15,.3,.5],"reps":reps,"spoil":"bounded randomized initial policy mean"}),"plant_hash":canonical_hash({"mask":spec.mask.tolist(),"curvature":spec.normalized_curvature.tolist()}),"graph_hash":canonical_hash(spec.mask.tolist()),
      "seed_registry_hash":canonical_hash([r["seed"] for r in rows]),"observable_definition":"E(t)=L_mean(t)-L_oracle_floor; F(t)=1-E(t)/E(0); observed 90% crossing E<=0.1E0","evaluation_budget":{"epochs":epochs,"candidates":candidates,"cycles_per_candidate":cycles,"replicates":reps},
      "spoil_protocol":"explicit bounded randomized production-policy mean at fixed calibrated optimum","rows":rows,"severity_summaries":summaries,"threshold_and_censoring_explicit":True,"fit_never_substitutes_for_observed_crossing":True,"hardware_claim":False,"prompt1_hash":prompt1_report()["artifact_hash"]}
    complete,missing=bundle_complete(target,["raw_traces.npz","figure.png"],result,["rows","severity_summaries","prompt1_hash"]);blockers=["PROMPT1_GATE_NOT_PASSED"] if not prompt1_report()["prompt1_gate_pass"] else []
    if mode=="smoke":blockers.append("SMOKE_NOT_REFERENCE_EVIDENCE")
    gate=EvidenceGate("recovery.spoiled_policy_90pct",complete,True,False,False,"SYNTHETIC_MECHANISM_EVIDENCE",tuple(blockers+missing));result["evidence_gate"]=gate.to_dict();result["blocking_reasons"]=list(gate.blocking_reasons)
    return write("results",result,"Randomized-policy Recovery Results",directory=target,json_name="results.json",md_name="report.md")
