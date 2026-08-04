"""Quantitative Figure 5c phase-space and independent time-domain fits."""
from __future__ import annotations
import numpy as np
from google_rl_reimplementation.google_pure_v7.config import canonical_hash
from google_rl_reimplementation.google_pure_v7.figure5.panel_b import scaling_trace
from google_rl_reimplementation.google_pure_v7.figure5.accounting import total_controls
from .common import bundle_complete,prompt1_report,require_reference_authorization,root,write
from .evidence_contracts import EvidenceGate
from .experiment_families import ExperimentFamily,family_metadata

DISTANCES=(3,5,7,9,11,13,15);PARAMETERS=(1,10,30)

def _ols(x:np.ndarray,y:np.ndarray,*,origin:bool=False)->dict:
    x=np.asarray(x,float);y=np.asarray(y,float);n=len(x)
    if origin:
      slope=float(x@y/max(x@x,1e-30));intercept=0.;pred=slope*x;dof=max(1,n-1);se=float(np.sqrt(np.sum((y-pred)**2)/dof/max(x@x,1e-30)));total=float(y@y)
    else:
      design=np.column_stack((np.ones(n),x));beta,_,_,_=np.linalg.lstsq(design,y,rcond=None);intercept,slope=map(float,beta);pred=design@beta;dof=max(1,n-2);variance=float(np.sum((y-pred)**2)/dof);cov=variance*np.linalg.pinv(design.T@design);se=float(np.sqrt(max(cov[1,1],0)));total=float(np.sum((y-np.mean(y))**2))
    sse=float(np.sum((y-pred)**2));return {"slope":slope,"intercept":intercept,"slope_ci_95":[slope-1.96*se,slope+1.96*se],"r_squared":1-sse/total if total>0 else 1.,"point_count":n,"sse":sse}

def _huber(x:np.ndarray,y:np.ndarray)->dict:
    x=np.asarray(x,float);y=np.asarray(y,float);design=np.column_stack((np.ones(len(x)),x));beta=np.linalg.lstsq(design,y,rcond=None)[0]
    for _ in range(30):
      residual=y-design@beta;scale=1.4826*np.median(np.abs(residual-np.median(residual)))+1e-12;z=np.abs(residual)/(1.345*scale);w=np.ones_like(z);mask=z>1;w[mask]=1/z[mask];new=np.linalg.lstsq(design*w[:,None]**.5,y*w**.5,rcond=None)[0]
      if np.linalg.norm(new-beta)<1e-12:break
      beta=new
    pred=design@beta;sse=float(np.sum((y-pred)**2));total=float(np.sum((y-np.mean(y))**2));variance=sse/max(1,len(x)-2);cov=variance*np.linalg.pinv(design.T@(w[:,None]*design));se=float(np.sqrt(max(cov[1,1],0)))
    return {"slope":float(beta[1]),"intercept":float(beta[0]),"slope_ci_95":[float(beta[1]-1.96*se),float(beta[1]+1.96*se)],"r_squared":1-sse/total if total>0 else 1.,"point_count":len(x),"sse":sse,"method":"Huber IRLS, tuning 1.345"}

def _cell_fit(ratio:np.ndarray,*,start_fraction:float=.1,derivative_method:str="central")->dict:
    ratio=np.asarray(ratio,float);epochs=len(ratio);start=max(2,int(start_fraction*epochs));stop=epochs-2;t=np.arange(epochs,dtype=float)
    if derivative_method=="central":derivative=np.gradient(ratio)
    elif derivative_method=="forward":derivative=np.r_[np.diff(ratio),np.nan]
    else:raise ValueError("unknown derivative method")
    x=1-ratio;indices=np.arange(start,stop);px=x[indices];py=100*derivative[indices]
    constrained=_ols(px,py,origin=True);free=_ols(px,py);robust=_huber(px,py)
    positive=indices[x[indices]>0];time_fit=_ols(t[positive],np.log(x[positive]));gamma=-time_fit["slope"];time_ci=[-time_fit["slope_ci_95"][1],-time_fit["slope_ci_95"][0]];phase_ci=[constrained["slope_ci_95"][0]/100,constrained["slope_ci_95"][1]/100]
    agree=max(phase_ci[0],time_ci[0])<=min(phase_ci[1],time_ci[1])
    return {"fit_interval":[start,stop],"derivative_method":derivative_method,"derivative_window":"three-point central" if derivative_method=="central" else "one-step forward",
      "boundary_handling":"exclude first/last two epochs","negative_derivative_count":int(np.sum(py<0)),"negative_derivatives_dropped":False,
      "constrained_through_origin":constrained,"free_intercept":free,"robust":robust,"phase_space_gamma_per_epoch":constrained["slope"]/100,"phase_space_gamma_ci_95":[v/100 for v in constrained["slope_ci_95"]],
      "time_domain":{"gamma_per_epoch":float(gamma),"gamma_ci_95":[float(v) for v in time_ci],"amplitude":float(np.exp(time_fit["intercept"])),"fit_r_squared":time_fit["r_squared"],"point_count":time_fit["point_count"]},
      "phase_time_agree_within_uncertainty":bool(agree),"x":px,"y":py}

def run_figure5c(*,mode:str="smoke",execute:bool=False)->dict:
    require_reference_authorization(mode,execute);epochs=128 if mode=="smoke" else 1000;rows=[];traces={};sensitivity=[]
    for p in PARAMETERS:
      for d in DISTANCES:
        seed=15500+p+d;a=scaling_trace(d,p,seed,epochs);fit=_cell_fit(a["lambda_ratio"]);traces[(p,d)]=a
        rows.append({"distance":d,"parameters_per_gate":p,"seed":seed,"total_controls":total_controls(d,p),"phase_space_constrained":fit["constrained_through_origin"],"phase_space_free":fit["free_intercept"],"phase_space_robust":fit["robust"],
          "phase_space_gamma_per_epoch":fit["phase_space_gamma_per_epoch"],"phase_space_gamma_ci_95":fit["phase_space_gamma_ci_95"],"time_domain":fit["time_domain"],"phase_time_agree_within_uncertainty":fit["phase_time_agree_within_uncertainty"],
          "derivative_method":fit["derivative_method"],"derivative_window":fit["derivative_window"],"fit_interval":fit["fit_interval"],"boundary_handling":fit["boundary_handling"],"negative_derivative_count":fit["negative_derivative_count"],"negative_derivatives_dropped":False,"point_count":fit["constrained_through_origin"]["point_count"]})
        for method in ("central","forward"):
          for fraction in (.05,.1,.2):
            sf=_cell_fit(a["lambda_ratio"],start_fraction=fraction,derivative_method=method);sensitivity.append({"distance":d,"parameters_per_gate":p,"method":method,"start_fraction":fraction,"phase_gamma":sf["phase_space_gamma_per_epoch"],"time_gamma":sf["time_domain"]["gamma_per_epoch"]})
    distance_tables=[]
    for p in PARAMETERS:
      subset=[r for r in rows if r["parameters_per_gate"]==p];d=np.asarray([r["distance"] for r in subset],float);n=np.log([r["total_controls"] for r in subset]);gamma=np.asarray([r["time_domain"]["gamma_per_epoch"] for r in subset]);by_d=_ols(d,gamma);by_n=_ols(n,gamma);deterioration=float((gamma[-1]-gamma[0])/max(abs(gamma[0]),1e-30));tol=.2
      distance_tables.append({"parameters_per_gate":p,"gamma_by_distance":gamma.tolist(),"slope_versus_distance":by_d,"slope_versus_log_controls":by_n,"d3_to_d15_relative_change":deterioration,"practical_relative_tolerance":tol,"within_practical_tolerance":abs(deterioration)<=tol})
    target=root()/"figure5c";target.mkdir(parents=True,exist_ok=True)
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    fig,(ax0,ax1)=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for d in DISTANCES:
      fit=_cell_fit(traces[(30,d)]["lambda_ratio"]);ax0.plot(fit["x"],fit["y"],".",ms=2,label=f"d={d}")
    ax0.set(xlabel=r"$1-\Lambda/\Lambda^\star$",ylabel=r"$10^2\,\partial_t\Lambda/\Lambda^\star$",title="Phase-space fits, P=30");ax0.legend(ncol=2,fontsize=7)
    for p in PARAMETERS:
      subset=[r for r in rows if r["parameters_per_gate"]==p];g=[r["time_domain"]["gamma_per_epoch"] for r in subset];lo=[r["time_domain"]["gamma_ci_95"][0] for r in subset];hi=[r["time_domain"]["gamma_ci_95"][1] for r in subset];ax1.errorbar(DISTANCES,g,yerr=[np.asarray(g)-lo,np.asarray(hi)-g],marker="o",label=f"P={p}")
    ax1.set(xlabel="distance",ylabel=r"time-domain $\gamma$ per epoch",title="Distance-independence table");ax1.legend();fig.savefig(target/"figure.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5));
    for method in ("central","forward"):
      for fraction in (.05,.1,.2):
        values=[r["phase_gamma"] for r in sensitivity if r["method"]==method and r["start_fraction"]==fraction and r["parameters_per_gate"]==30];ax.plot(DISTANCES,values,marker="o",label=f"{method}, start={fraction}")
    ax.set(xlabel="distance",ylabel="phase-space gamma per epoch",title="Derivative/window sensitivity, P=30");ax.legend(fontsize=8);fig.tight_layout();fig.savefig(target/"sensitivity.png",dpi=180);plt.close(fig)
    valid=all(np.isfinite(r["phase_space_gamma_per_epoch"]) and np.isfinite(r["time_domain"]["gamma_per_epoch"]) for r in rows);agreement=sum(r["phase_time_agree_within_uncertainty"] for r in rows)
    result={"schema_version":"google-pure-evidence-v8-5c.v2",**family_metadata(ExperimentFamily.FIGURE5C_CONVERGENCE_LAW),"mode":mode,"controller_hash":"analytic-scaling-model-v7",
      "protocol_hash":canonical_hash({"P":PARAMETERS,"d":DISTANCES,"epochs":epochs,"fit_start_fraction":.1,"derivative":"central"}),"plant_hash":"analytic-sparse-paper-anchored-surrogate","graph_hash":"surface-code-local-count-graph-v1","seed_registry_hash":canonical_hash([r["seed"] for r in rows]),
      "observable_definition":"x=1-Lambda/Lambda*, y=100*d_t Lambda/Lambda*; phase slope m=100 gamma plus independent log-x time fit","evaluation_budget":{"cells":len(rows),"epochs_per_cell":epochs},"cell_count":len(rows),"rows":rows,
      "distance_independence_tables":distance_tables,"sensitivity_records":sensitivity,"phase_time_agreement_count":agreement,"phase_time_cell_count":len(rows),"unfavorable_points_dropped":False,"classification":"CONSTRUCTED_ANALYTIC_CONVERGENCE","prompt1_hash":prompt1_report()["artifact_hash"]}
    write("report",{"schema_version":"google-pure-evidence-v8-5c-report.v2","classification":result["classification"],"distance_independence_tables":distance_tables,"phase_time_agreement_count":agreement,"phase_time_cell_count":len(rows),"prompt1_hash":result["prompt1_hash"]},"Figure 5c Scientific Report",directory=target,json_name="report.json",md_name="report.md")
    complete,missing=bundle_complete(target,["figure.png","sensitivity.png","report.md"],result,["rows","distance_independence_tables","prompt1_hash"]);blockers=["CONSTRUCTED_ANALYTIC_TRAJECTORIES_NOT_EMPIRICAL_PPO"]
    if not prompt1_report()["prompt1_gate_pass"]:blockers.append("PROMPT1_GATE_NOT_PASSED")
    status="PAPER_ANCHORED_SYNTHETIC_EVIDENCE" if valid else "INVALID_DIAGNOSTIC";gate=EvidenceGate("figure5c.analytic_convergence_law",complete,valid,False,False,status,tuple(blockers+missing));result["evidence_gate"]=gate.to_dict();result["blocking_reasons"]=list(gate.blocking_reasons)
    return write("slopes",result,"Figure 5c Convergence Slopes",directory=target,json_name="slopes.json",md_name="slopes.md")
