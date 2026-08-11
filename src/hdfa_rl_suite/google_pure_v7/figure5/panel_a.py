"""Panel 5a: finite-shot candidate-level real-time steerability acquisition."""
from __future__ import annotations
import math
from typing import Any
import numpy as np
from hdfa_rl_suite.google_pure_v6.plant import PureQuadraticPlant, default_spec
from hdfa_rl_suite.google_pure_v6.reference_agent import evidence_from_counts
from hdfa_rl_suite.google_pure_v7.controller import build_production_agent, require_resolved_controller
from .accounting import acquisition_accounting
from .common import require_mode
from .protocol import panel_plan
from .storage import ensure_run_checkpoint, shard_id, update_run_checkpoint, write_shard

def _condition(plan: dict, condition: dict) -> tuple[dict,dict]:
    config=plan["config"]; epochs=int(config["epochs"]); candidates=int(config["candidates"]); cycles=int(config["cycles_per_candidate"])
    frequency=float(condition["frequency"]); entropy=float(condition["entropy_coefficient"]); seed=int(condition["seed"])
    plant=PureQuadraticPlant(default_spec(int(config.get("controls",6))))
    agent=build_production_agent(plant.mask,plant.spec.base_optimum_normalized,plant.spec.coordinates,seed=seed)
    # This is the Figure-5 scan coordinate. All other resolved controller fields remain frozen.
    agent.choices["entropy_coefficient"]=entropy
    rng=np.random.default_rng(seed+100_000); direction=np.linspace(1.0,.45,plant.spec.control_count); direction/=np.linalg.norm(direction)
    candidate_cost=[]; candidate_actions=[]; fixed=[]; learned=[]; oracle=[]; scales=[]; gradients=[]; variances=[]
    fixed_native=plant.base_optimum_native.copy()
    for epoch in range(epochs):
        optimum=np.sin(2*np.pi*frequency*epoch)*float(config.get("drift_amplitude",.45))*direction
        optimum_native=plant.spec.coordinates.to_native(optimum); batch=agent.sample(candidates)
        counts=plant.acquire_counts(batch.applied_native_actions,optimum_native,cycles=cycles,rng=rng)
        candidate_cost.append(np.mean(counts/cycles,axis=1)); candidate_actions.append(batch.applied_normalized_actions)
        diag=agent.update(batch,evidence_from_counts(batch,counts,cycles)); gradients.append(diag["mean_gradient_norm"])
        mean_native=plant.spec.coordinates.to_native(agent.mean)
        fixed.append(float(np.mean(plant.detector_rates_native(fixed_native[None,:],optimum_native))))
        learned.append(float(np.mean(plant.detector_rates_native(mean_native[None,:],optimum_native))))
        oracle.append(float(np.mean(plant.detector_rates_native(optimum_native[None,:],optimum_native))))
        scales.append(float(np.mean(agent.scale))); variances.append(float(np.var(candidate_cost[-1],ddof=1)) if candidates>1 else 0.)
    candidate_cost=np.asarray(candidate_cost); fixed=np.asarray(fixed); learned=np.asarray(learned); oracle=np.asarray(oracle)
    c_candidate=float(candidate_cost.mean()); c_fixed=float(fixed.mean()); c_mean=float(learned.mean()); c_oracle=float(oracle.mean())
    denominator=c_fixed-c_oracle
    improvement_candidate=(c_fixed-c_candidate)/denominator; improvement_mean=(c_fixed-c_mean)/denominator
    arrays={"epoch":np.arange(epochs),"candidate_cost":candidate_cost,"candidate_actions":np.asarray(candidate_actions),
            "fixed_cost":fixed,"learned_mean_cost":learned,"oracle_cost":oracle,"mean_scale":scales,
            "gradient_norm":gradients,"candidate_variance":variances}
    accounting=acquisition_accounting(epochs=epochs,candidates=candidates,cycles_per_candidate=cycles)
    metadata={"frequency":frequency,"nominal_entropy_coefficient":entropy,
              "effective_policy_entropy":float(np.mean(np.log(np.asarray(scales)*math.sqrt(2*math.pi*math.e)))),
              "improvement_candidate":improvement_candidate,"improvement_mean":improvement_mean,
              "candidate_cost":c_candidate,"fixed_cost":c_fixed,"mean_cost":c_mean,"oracle_cost":c_oracle,
              **accounting,"source_semantics":"paper_anchored_synthetic_current_v7_controller"}
    return arrays,metadata

def acquire(config: dict[str,Any], *, dry_run=False, resume=True, max_shards:int|None=None, execute_paper_scale=False) -> dict:
    require_mode(config["mode"],execute_paper_scale=execute_paper_scale or dry_run); plan=panel_plan("5a",config)
    if dry_run:return plan
    if int(config["epochs"])*int(config["candidates"])*int(config["cycles_per_candidate"])>=1_800_000_000 and not execute_paper_scale:
        raise RuntimeError("the 1.8-billion-cycle Figure 5a contract requires --execute-paper-scale")
    controller=require_resolved_controller(); completed=[]
    from .storage import panel_shard_dir
    checkpoint=ensure_run_checkpoint("5a",plan,plant_hash=f"v6-default-quadratic-{config.get('controls',6)}",graph_hash="graph-v6.1",resume=resume)
    for condition in plan["conditions"]:
        identity={"panel":"5a","protocol_hash":plan["protocol_hash"],"controller_hash":controller["resolved_config_hash"],
                  "grid_cell":{"frequency":condition["frequency"],"entropy":condition["entropy_coefficient"]},
                  "distance":3,"parameters_per_gate":1,"seed":condition["seed"],"replicate":0,"chunk":0}
        sid=shard_id(identity)
        if resume and (panel_shard_dir("5a")/f"{sid}.json").exists(): completed.append(sid); continue
        arrays,meta=_condition(plan,condition); write_shard("5a",identity,arrays,{**meta,"mode":config["mode"]}); completed.append(sid)
        update_run_checkpoint("5a",checkpoint,completed,len(plan["conditions"]))
        if max_shards is not None and len(completed)>=max_shards: break
    update_run_checkpoint("5a",checkpoint,completed,len(plan["conditions"]))
    return {"panel":"5a","protocol_hash":plan["protocol_hash"],"completed_shards":completed,"expected_shards":len(plan["conditions"])}

def shard_row(record:dict,data:Any)->dict:
    m=record["metadata"]; return {"panel":"5a","seed":record["identity"]["seed"],"frequency":m["frequency"],
      "entropy_coefficient":m["nominal_entropy_coefficient"],"effective_policy_entropy":m["effective_policy_entropy"],
      "improvement_candidate":m["improvement_candidate"],"improvement_mean":m["improvement_mean"],
      "candidate_cost":m["candidate_cost"],"fixed_cost":m["fixed_cost"],"mean_cost":m["mean_cost"],
      "oracle_cost":m["oracle_cost"],"candidate_cycles":m["candidate_qec_cycles"],"mode":m["mode"]}
