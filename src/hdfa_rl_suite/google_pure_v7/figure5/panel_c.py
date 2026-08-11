"""Panel 5c: source-axis local convergence derivatives and fits."""
from __future__ import annotations
from typing import Any
import numpy as np
from hdfa_rl_suite.google_pure_v7.controller import require_resolved_controller
from .accounting import total_controls
from .common import require_mode
from .panel_b import scaling_trace
from .protocol import panel_plan
from .storage import ensure_run_checkpoint,panel_shard_dir,shard_id,update_run_checkpoint,write_shard

def acquire(config:dict[str,Any],*,dry_run=False,resume=True,max_shards:int|None=None,execute_paper_scale=False)->dict:
    require_mode(config["mode"],execute_paper_scale=execute_paper_scale or dry_run);plan=panel_plan("5c",config)
    if dry_run:return plan
    controller=require_resolved_controller();completed=[]
    checkpoint=ensure_run_checkpoint("5c",plan,plant_hash="paper-quadratic-sparse-surrogate-v1",graph_hash="surface-code-local-count-graph-v1",resume=resume)
    for condition in plan["conditions"]:
        d=int(condition["distance"]);p=int(condition["parameters_per_gate"]);seed=int(condition["seed"])
        identity={"panel":"5c","protocol_hash":plan["protocol_hash"],"controller_hash":controller["resolved_config_hash"],
          "grid_cell":{"distance":d,"parameters_per_gate":p},"distance":d,"parameters_per_gate":p,"seed":seed,"replicate":0,"chunk":0}
        sid=shard_id(identity)
        if resume and (panel_shard_dir("5c")/f"{sid}.json").exists():completed.append(sid);continue
        base=scaling_trace(d,p,seed,int(config["epochs"]));ratio=np.asarray(base["lambda_ratio"])
        speed=np.diff(ratio);x=1-ratio[:-1];y=100*speed
        keep=(x>float(config.get("local_fit_min_distance",1e-4)))&(x<float(config.get("local_fit_max_distance",.7)))
        slope=float(np.dot(x[keep],y[keep])/np.dot(x[keep],x[keep])) if np.any(keep) else float("nan")
        arrays={**base,"x_distance":x,"normalized_speed":y,"fit_mask":keep.astype(np.uint8)}
        meta={"mode":config["mode"],"distance":d,"parameters_per_gate":p,"total_controls":total_controls(d,p),
          "gamma_times_100":slope,"sensitivity_method":"centered/local-forward finite difference; origin-constrained fit",
          "source_x_axis":"1-Lambda/Lambda*","source_y_axis":"1e2 d_t Lambda/Lambda*","raw_trajectory_preserved":True}
        write_shard("5c",identity,arrays,meta);completed.append(sid)
        update_run_checkpoint("5c",checkpoint,completed,len(plan["conditions"]))
        if max_shards is not None and len(completed)>=max_shards:break
    update_run_checkpoint("5c",checkpoint,completed,len(plan["conditions"]))
    return {"panel":"5c","protocol_hash":plan["protocol_hash"],"completed_shards":completed,"expected_shards":len(plan["conditions"])}

def shard_row(record:dict,data:Any)->dict:
    m=record["metadata"];x=np.asarray(data["x_distance"]);y=np.asarray(data["normalized_speed"])
    return {"panel":"5c","seed":record["identity"]["seed"],"distance":m["distance"],"parameters_per_gate":m["parameters_per_gate"],
      "total_controls":m["total_controls"],"gamma_times_100":m["gamma_times_100"],"x_distance":float(np.mean(x)),
      "normalized_speed":float(np.mean(y)),"source_x_axis":m["source_x_axis"],"source_y_axis":m["source_y_axis"],"mode":m["mode"]}
