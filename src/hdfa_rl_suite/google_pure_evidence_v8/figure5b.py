"""Paper-axis Figure 5b evidence with explicit analytic-model provenance."""
from __future__ import annotations
import numpy as np
from hdfa_rl_suite.google_pure_v7.config import canonical_hash
from hdfa_rl_suite.google_pure_v7.figure5.panel_b import scaling_trace
from hdfa_rl_suite.google_pure_v7.figure5.accounting import total_controls
from .common import bundle_complete,prompt1_report,require_reference_authorization,root,write
from .evidence_contracts import EvidenceGate
from .experiment_families import ExperimentFamily,family_metadata

DISTANCES=(3,5,7,9,11,13,15)

def run_figure5b(*,mode:str="smoke",execute:bool=False)->dict:
    require_reference_authorization(mode,execute);epochs,reps=(128,3) if mode=="smoke" else (1000,12);arrays={};rows=[]
    for d in DISTANCES:
      for rep in range(reps):
        seed=15400+100*d+rep;a=scaling_trace(d,30,seed,epochs);arrays.update({f"d{d}_r{rep}_{k}":v for k,v in a.items()})
        rows.append({"distance":d,"replicate":rep,"seed":seed,"parameters_per_gate":30,"total_controls":total_controls(d,30),"epochs":epochs,
          "initial_physical_error":float(a["physical_error"][0]),"final_physical_error":float(a["physical_error"][-1]),"initial_ler":float(a["logical_learned"][0]),"final_ler":float(a["logical_learned"][-1]),
          "independent_irreducible_floor":float(a["logical_floor"][-1]),"final_lambda_ratio":float(a["lambda_ratio"][-1]),"finite_shot_detector_sampling":False,"analytic_recurrence":True})
    target=root()/"figure5b";target.mkdir(parents=True,exist_ok=True);np.savez_compressed(target/"raw_trajectories.npz",**arrays)
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(15,8),constrained_layout=True);mappable=None
    for ax,d in zip(axes.flat,DISTANCES):
      for rep in range(reps):
        physical=arrays[f"d{d}_r{rep}_physical_error"];ler=arrays[f"d{d}_r{rep}_logical_learned"];mappable=ax.scatter(physical,ler,c=np.arange(epochs),cmap="viridis",s=7,alpha=.55)
      ax.axhline(arrays[f"d{d}_r0_logical_floor"][0],color="k",ls="--",lw=.7,label="floor");ax.set(xlabel="physical error rate",ylabel="LER",title=f"d={d}");ax.set_yscale("log")
    axes.flat[-1].axis("off");fig.colorbar(mappable,ax=axes.ravel().tolist(),label="epoch");fig.savefig(target/"paper_axes_figure.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    for d in DISTANCES:
      stack=np.stack([arrays[f"d{d}_r{r}_lambda_ratio"] for r in range(reps)]);mean=stack.mean(axis=0);lo=np.quantile(stack,.025,axis=0);hi=np.quantile(stack,.975,axis=0);ax.plot(mean,label=f"d={d}");ax.fill_between(np.arange(epochs),lo,hi,alpha=.08)
    ax.set(xlabel="epoch",ylabel=r"$\Lambda/\Lambda^\star$",title="Normalized sparse convergence diagnostic (not Figure 5b)");ax.legend(ncol=2);fig.tight_layout();fig.savefig(target/"normalized_diagnostic.png",dpi=180);plt.close(fig)
    result={"schema_version":"google-pure-evidence-v8-5b.v2",**family_metadata(ExperimentFamily.FIGURE5B_SPARSE_SCALING),"mode":mode,"controller_hash":"analytic-scaling-model-v7",
      "protocol_hash":canonical_hash({"distances":DISTANCES,"P":30,"epochs":epochs,"replicates":reps}),"plant_hash":"analytic-sparse-paper-anchored-surrogate","graph_hash":"surface-code-local-count-graph-v1","seed_registry_hash":canonical_hash([r["seed"] for r in rows]),
      "observable_definition":"paper-axis physical error rate versus LER, epoch colour, explicit independent floor; normalized Lambda diagnostic separate","evaluation_budget":{"epochs_per_distance":epochs,"distances":len(DISTANCES),"replicates":reps},
      "classification":"ANALYTIC_SCALING_MODEL","normalized_old_plot_role":"NORMALIZED_SPARSE_CONVERGENCE_DIAGNOSTIC","distances":list(DISTANCES),"distance_15_control_count":total_controls(15,30),"paper_axis_transforms":{"x":"physical error rate","y":"LER logarithmic","colour":"epoch"},
      "uncertainty":"2.5-97.5 percent replicate envelope","trajectories_share_analytic_recurrence":True,"distance_normalization_constructed":True,"rows":rows,"prompt1_hash":prompt1_report()["artifact_hash"]}
    complete,missing=bundle_complete(target,["raw_trajectories.npz","paper_axes_figure.png","normalized_diagnostic.png"],result,["rows","paper_axis_transforms","prompt1_hash"]);blockers=["ANALYTIC_RECURRENCE_NOT_EMPIRICAL_PPO_SCALING","PROMPT1_GATE_NOT_PASSED"] if not prompt1_report()["prompt1_gate_pass"] else ["ANALYTIC_RECURRENCE_NOT_EMPIRICAL_PPO_SCALING"]
    gate=EvidenceGate("figure5b.analytic_sparse_scaling",complete,True,False,False,"PAPER_ANCHORED_SYNTHETIC_EVIDENCE",tuple(blockers+missing));result["evidence_gate"]=gate.to_dict();result["blocking_reasons"]=list(gate.blocking_reasons)
    return write("summary",result,"Figure 5b Sparse Scaling Evidence",directory=target,json_name="summary.json",md_name="report.md")
