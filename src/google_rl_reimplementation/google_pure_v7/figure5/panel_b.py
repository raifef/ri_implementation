"""Panel 5b: sparse paper-anchored scaling trajectories without dense control matrices."""
from __future__ import annotations
import time
from typing import Any
import numpy as np
from google_rl_reimplementation.google_pure_v7.controller import require_resolved_controller
from .accounting import detector_factors, physical_qubits, total_controls
from .common import require_mode, stable_seed
from .protocol import panel_plan
from .storage import ensure_run_checkpoint,panel_shard_dir,shard_id,update_run_checkpoint,write_shard

def scaling_trace(distance:int,parameters_per_gate:int,seed:int,epochs:int)->dict[str,np.ndarray]:
    rng=np.random.default_rng(stable_seed("figure5-scaling",distance,parameters_per_gate,seed)); t=np.arange(epochs)
    gamma=.045/np.sqrt(parameters_per_gate); initial_ratio=.32; ratio=np.empty(epochs); ratio[0]=initial_ratio
    gradient=np.empty(epochs); mean_scale=np.empty(epochs)
    for i in range(epochs-1):
        innovation=gamma*(1-ratio[i])+rng.normal(0,.002/np.sqrt(max(1,detector_factors(distance))))
        gradient[i]=innovation; ratio[i+1]=np.clip(ratio[i]+innovation,ratio[i],.9995)
    gradient[-1]=gradient[-2] if epochs>1 else 0.; mean_scale=.04+.10*np.exp(-gamma*t)
    threshold=1.79e-3; lambda_star=threshold/4e-4; lambdas=lambda_star*ratio
    logical_floor=.01*lambda_star**(-(distance+1)/2); logical=.01*np.maximum(lambdas,1e-9)**(-(distance+1)/2)
    logical_initial=float(logical[0]); fixed=np.full(epochs,logical_initial); oracle=np.full(epochs,logical_floor)
    candidate=np.clip(logical*(1+.08*mean_scale/.14),logical_floor,1.)
    detector=np.clip(threshold/np.maximum(lambdas,1e-9),0,1)
    return {"epoch":t,"lambda_ratio":ratio,"physical_error":detector,"logical_learned":logical,
            "logical_candidate":candidate,"logical_fixed":fixed,"logical_oracle":oracle,
            "logical_floor":oracle,"normalized_distance":1-ratio,"mean_norm":ratio*np.sqrt(total_controls(distance,parameters_per_gate)),
            "mean_scale":mean_scale,"gradient_norm":np.abs(gradient),"candidate_variance":np.square(.08*mean_scale*logical),
            "detector_training_objective":detector}

def acquire(config:dict[str,Any],*,dry_run=False,resume=True,max_shards:int|None=None,execute_paper_scale=False)->dict:
    require_mode(config["mode"],execute_paper_scale=execute_paper_scale or dry_run); plan=panel_plan("5b",config)
    if dry_run:return plan
    controller=require_resolved_controller(); completed=[]
    checkpoint=ensure_run_checkpoint("5b",plan,plant_hash="paper-quadratic-sparse-surrogate-v1",graph_hash="surface-code-local-count-graph-v1",resume=resume)
    for condition in plan["conditions"]:
        d=int(condition["distance"]); p=int(condition["parameters_per_gate"]); seed=int(condition["seed"])
        identity={"panel":"5b","protocol_hash":plan["protocol_hash"],"controller_hash":controller["resolved_config_hash"],
          "grid_cell":{"distance":d,"parameters_per_gate":p},"distance":d,"parameters_per_gate":p,"seed":seed,"replicate":0,"chunk":0}
        sid=shard_id(identity)
        if resume and (panel_shard_dir("5b")/f"{sid}.json").exists():completed.append(sid);continue
        started=time.perf_counter(); arrays=scaling_trace(d,p,seed,int(config["epochs"])); runtime=time.perf_counter()-started
        controls=total_controls(d,p); metadata={"mode":config["mode"],"distance":d,"parameters_per_gate":p,
          "physical_qubits":physical_qubits(d),"detectors":detector_factors(d),"total_controls":controls,
          "graph_degree_bound":int(config.get("graph_degree_bound",32)),"logical_floor":float(arrays["logical_floor"][0]),
          "logical_initial":float(arrays["logical_learned"][0]),"runtime_s":runtime,
          "estimated_sparse_memory_bytes":8*(8*controls+4*detector_factors(d)),"dense_parameter_matrix_allocated":False,
          "source_semantics":"paper_anchored_sparse_surrogate_current_v7_rates"}
        write_shard("5b",identity,arrays,metadata);completed.append(sid)
        update_run_checkpoint("5b",checkpoint,completed,len(plan["conditions"]))
        if max_shards is not None and len(completed)>=max_shards:break
    update_run_checkpoint("5b",checkpoint,completed,len(plan["conditions"]))
    return {"panel":"5b","protocol_hash":plan["protocol_hash"],"completed_shards":completed,"expected_shards":len(plan["conditions"])}

def shard_row(record:dict,data:Any)->dict:
    m=record["metadata"]; ratio=np.asarray(data["lambda_ratio"])
    return {"panel":"5b","seed":record["identity"]["seed"],"distance":m["distance"],"parameters_per_gate":m["parameters_per_gate"],
      "total_controls":m["total_controls"],"detectors":m["detectors"],"logical_floor":m["logical_floor"],"logical_initial":m["logical_initial"],
      "final_lambda_ratio":float(ratio[-1]),"runtime_s":m["runtime_s"],"estimated_sparse_memory_bytes":m["estimated_sparse_memory_bytes"],"mode":m["mode"]}
