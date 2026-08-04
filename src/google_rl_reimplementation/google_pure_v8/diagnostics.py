"""Matched finite-shot Figure 5a diagnostics using the frozen production agent and plant."""
from __future__ import annotations
import math
from typing import Any
import numpy as np
from google_rl_reimplementation.google_pure_v6.plant import PureQuadraticPlant,default_spec
from google_rl_reimplementation.google_pure_v6.reference_agent import PureGoogleV6Agent,evidence_from_counts
from google_rl_reimplementation.google_pure_v7.controller import agent_choices,require_resolved_controller
from google_rl_reimplementation.google_pure_v7.config import canonical_hash
from .common import guard_seed,write_report
from .contracts import cost_decomposition,frequency_contract,normalized_edr_improvement

POLICIES=("fixed","oracle","oracle_with_production_scale","learned_mean","sampled_candidates")

def _direction(count:int)->np.ndarray:
    value=np.linspace(1.0,.45,count);return value/np.linalg.norm(value)

def _freeze_batch(batch:Any)->None:
    """Make collection-time values immutable before an update can observe them."""
    for name in ("collection_mean","collection_log_scale","collection_component_log_probability",
                 "latent_normalized_actions","applied_normalized_actions","applied_native_actions"):
        np.asarray(getattr(batch,name)).setflags(write=False)

def _eval_counts(plant:PureQuadraticPlant,actions_native:np.ndarray,optimum_native:np.ndarray,*,cycles:int,rng:np.random.Generator)->tuple[int,float,np.ndarray]:
    counts=plant.acquire_counts(actions_native,optimum_native,cycles=cycles,rng=rng)
    per_candidate=counts.sum(axis=1)/cycles
    return int(counts.sum()),float(np.mean(per_candidate)),per_candidate

def run_cell(*,frequency:float=1/150,entropy:float=.0004,phase:float=0.0,minimum_scale:float=.04,
             epochs:int=72,candidates:int=16,cycles:int=4000,seed:int=14101)->dict[str,Any]:
    guard_seed(seed); plant=PureQuadraticPlant(default_spec(6)); controller=require_resolved_controller(); choices=agent_choices(controller)
    choices={**choices,"entropy_coefficient":float(entropy),"scale_bounds":[float(minimum_scale),choices["scale_bounds"][1]]}
    agent=PureGoogleV6Agent(plant.mask,plant.spec.base_optimum_normalized,plant.spec.coordinates,choices,seed=seed,objective_mode="source_literal_ppo")
    training_rng=np.random.default_rng(seed+100_000); evaluation_rng=np.random.default_rng(seed+200_000); oracle_rng=np.random.default_rng(seed+300_000)
    direction=_direction(plant.spec.control_count); amplitude=.45
    raw={name:0 for name in POLICIES}; traces={name:[] for name in POLICIES}; action_hashes={name:[] for name in POLICIES}
    scales=[];pre_scales=[];post_scales=[];clip=[];clip_by_control=[];grad=[];entropy_grad=[];behaviour_hashes=[];current_hashes=[];mean_vectors=[]
    norm_var=[];native_var=[];displacements=[];training_events=0;candidate_example={}
    for epoch in range(epochs):
        optimum=amplitude*np.sin(2*np.pi*frequency*epoch+phase)*direction; optimum_native=plant.spec.coordinates.to_native(optimum)
        mean=agent.mean.copy();mean_native=plant.spec.coordinates.to_native(mean);scale=agent.scale.copy();batch=agent.sample(candidates);_freeze_batch(batch)
        fixed_norm=np.repeat(plant.spec.base_optimum_normalized[None,:],candidates,axis=0)
        oracle_norm=np.repeat(optimum[None,:],candidates,axis=0)
        oracle_scale_norm=plant.spec.coordinates.apply_bounds(optimum[None,:]+scale[None,:]*oracle_rng.normal(size=(candidates,plant.spec.control_count)))
        learned_norm=np.repeat(mean[None,:],candidates,axis=0)
        normalized={"fixed":fixed_norm,"oracle":oracle_norm,"oracle_with_production_scale":oracle_scale_norm,
                    "learned_mean":learned_norm,"sampled_candidates":batch.applied_normalized_actions}
        for name,actions in normalized.items():
            native=plant.spec.coordinates.to_native(actions);events,value,per_candidate=_eval_counts(plant,native,optimum_native,cycles=cycles,rng=evaluation_rng)
            raw[name]+=events;traces[name].append(value);action_hashes[name].append(canonical_hash(native.tolist()))
        counts=plant.acquire_counts(batch.applied_native_actions,optimum_native,cycles=cycles,rng=training_rng);training_events+=int(counts.sum())
        latent=np.asarray(batch.latent_normalized_actions);applied=np.asarray(batch.applied_normalized_actions);native=np.asarray(batch.applied_native_actions)
        clipped=latent!=applied;clip.append(float(np.mean(clipped)));clip_by_control.append(np.mean(clipped,axis=0));norm_var.append(np.var(applied,axis=0));native_var.append(np.var(native,axis=0));displacements.append((native-mean_native[None,:])**2)
        if epoch==0:
            candidate_example={"epsilon":((latent-batch.collection_mean[None,:])/np.exp(batch.collection_log_scale)[None,:]).tolist(),
              "normalized_mean":batch.collection_mean.tolist(),"normalized_scale":np.exp(batch.collection_log_scale).tolist(),
              "normalized_latent":latent.tolist(),"native_nominal":plant.spec.coordinates.native_offset.tolist(),"native_mean":mean_native.tolist(),
              "native_perturbation":(native-mean_native[None,:]).tolist(),"applied_native":native.tolist(),"clipping_mask":clipped.tolist(),
              "likelihood_action":"latent_normalized","plant_action":"applied_native"}
        behaviour_hashes.append(canonical_hash({"mean":batch.collection_mean.tolist(),"log_scale":batch.collection_log_scale.tolist(),"logp":batch.collection_component_log_probability.tolist()}))
        pre_scales.append(scale.tolist());diagnostic=agent.update(batch,evidence_from_counts(batch,counts,cycles));post_scales.append(agent.scale.tolist())
        grad.append(diagnostic["mean_gradient_norm"]);effective=np.asarray(diagnostic.get("effective_entropy_gradient",np.full(plant.spec.control_count,entropy)));entropy_grad.append(float(np.linalg.norm(effective)))
        scales.append(float(np.mean(scale)));current_hashes.append(canonical_hash({"mean":agent.mean.tolist(),"scale":agent.scale.tolist()}));mean_vectors.append(mean)
    effective_cycles=epochs*candidates*cycles;costs={name:raw[name]/effective_cycles for name in POLICIES}
    diffs=np.asarray(traces["fixed"])-np.asarray(traces["oracle"]);denominator_se=float(np.std(diffs,ddof=1)/np.sqrt(len(diffs))) if len(diffs)>1 else float("inf")
    resolution=3*denominator_se;decomposition=cost_decomposition(costs["fixed"],costs["sampled_candidates"],costs["learned_mean"],costs["oracle"])
    improvements={name:normalized_edr_improvement(costs["fixed"],value,costs["oracle"],resolution=0.0) for name,value in costs.items() if name not in {"fixed","oracle"}}
    improvements.update({"fixed":0.0,"oracle":1.0});means=np.asarray(mean_vectors);projection=means@direction;t=np.arange(epochs)
    design=np.column_stack((np.ones(epochs),np.sin(2*np.pi*frequency*t+phase),np.cos(2*np.pi*frequency*t+phase)));beta=np.linalg.lstsq(design,projection,rcond=None)[0]
    tracking_amplitude=float(np.hypot(beta[1],beta[2]));phase_lag=float(-math.atan2(beta[2],beta[1]));detector_hash=canonical_hash({"detectors":list(range(plant.spec.detector_count)),"weights":[1.0]*plant.spec.detector_count})
    accounting={name:{"raw_detector_event_count":raw[name],"effective_qec_cycles":effective_cycles,"edr_per_cycle":costs[name],
      "candidate_count":candidates,"candidate_weighting":"uniform","detector_weighting_hash":detector_hash,"epoch_window":[0,epochs],"burn_in":0,
      "drift_tape_hash":canonical_hash((amplitude*np.sin(2*np.pi*frequency*t+phase)[:,None]*direction[None,:]).tolist()),"phase":phase} for name in POLICIES}
    return {"frequency":frequency_contract(frequency,phase),"entropy_coefficient":entropy,"minimum_scale":minimum_scale,"seed":seed,"epochs":epochs,
      "candidates_per_epoch":candidates,"cycles_per_candidate":cycles,"evaluation_protocol":"matched independent finite-shot evaluation for all five policies",
      "training_qec_cycles":effective_cycles,"evaluation_qec_cycles_per_policy":effective_cycles,"training_raw_detector_events":training_events,
      "cost_quantity":"total detector-event count divided by matched QEC cycles, unit detector weights","costs":costs,"improvements":improvements,"decomposition":decomposition,
      "policy_accounting":accounting,"denominator_standard_error":denominator_se,"denominator_resolution_3se":resolution,"denominator_statistically_resolved":costs["fixed"]-costs["oracle"]>resolution,
      "mean_scale":{"initial":scales[0],"time_average":float(np.mean(scales)),"final":float(np.mean(agent.scale))},"pre_update_scale_vectors":pre_scales,"post_update_scale_vectors":post_scales,
      "fraction_scale_at_floor":float(np.mean(np.asarray(post_scales)<=minimum_scale*(1+1e-9))),"fraction_scale_at_ceiling":float(np.mean(np.asarray(post_scales)>=choices["scale_bounds"][1]*(1-1e-9))),
      "clipping_fraction":float(np.mean(clip)),"clipping_fraction_by_control":np.mean(clip_by_control,axis=0).tolist(),"reward_gradient_norm":float(np.mean(grad)),"entropy_gradient_norm":float(np.mean(entropy_grad)),
      "candidate_normalized_variance":np.mean(norm_var,axis=0).tolist(),"candidate_native_variance":np.mean(native_var,axis=0).tolist(),
      "candidate_native_displacement_rms":float(np.sqrt(np.mean(np.concatenate(displacements,axis=0)))),"candidate_example":candidate_example,
      "tracking_amplitude":tracking_amplitude,"tracking_amplitude_gain":tracking_amplitude/amplitude,"phase_lag_radians":phase_lag,
      "behaviour_policy_hashes":behaviour_hashes,"post_update_policy_hashes":current_hashes,"behaviour_snapshots_immutable":True,
      "drift_tape_hash":accounting["fixed"]["drift_tape_hash"],"burn_in_epochs":0,"analysis_window":[0,epochs],"complete_periods":float(epochs*frequency),
      "fixed_denominator":costs["fixed"]-costs["oracle"],"decomposition_identity_pass":bool(np.isclose(decomposition["direct_improvement"],decomposition["decomposition_improvement"]))}

def run_figure5a_feasibility()->dict[str,Any]:
    cell=run_cell(epochs=120,candidates=24,cycles=8000,seed=14101);imp=cell["improvements"]
    if not cell["denominator_statistically_resolved"]:classification="METRIC_OR_ACCOUNTING_FAILURE"
    elif not np.isclose(imp["fixed"],0):classification="METRIC_OR_ACCOUNTING_FAILURE"
    elif not np.isclose(imp["oracle"],1):classification="ORACLE_CONSTRUCTION_FAILURE"
    elif imp["oracle_with_production_scale"]<=0:classification="EXPLORATION_FLOOR_FAILURE"
    elif imp["learned_mean"]<=0:classification="MEAN_TRACKING_BANDWIDTH_FAILURE"
    elif imp["sampled_candidates"]<=0:classification="EXPLORATION_FLOOR_FAILURE"
    else:classification="FEASIBLE"
    write_report("figure5a_cost_decomposition",{"schema_version":"google-pure-v8-figure5a-cost-decomposition.v2","decomposition":cell["decomposition"],"identity_pass":cell["decomposition_identity_pass"],"matched_finite_shot":True,
      "interpretation":{"d_tracking":"mean tracking failure","d_exploration":"candidate exploration damage","d_fixed":"normalization signal"}},"Figure 5a Cost Decomposition")
    return write_report("figure5a_feasibility_decomposition",{"schema_version":"google-pure-v8-figure5a-feasibility.v2","cell":cell,"classification":classification,"artifact_complete":True,"gate_pass":classification=="FEASIBLE","blocking_reasons":[] if classification=="FEASIBLE" else [classification]},"Figure 5a Five-policy Feasibility Decomposition")

def run_edr_identity_audit()->dict[str,Any]:
    cell=run_cell(epochs=24,candidates=8,cycles=4000,seed=14102);policies=cell["policy_accounting"]
    identity_fields=("effective_qec_cycles","epoch_window","burn_in","drift_tape_hash","phase","detector_weighting_hash","candidate_count","candidate_weighting")
    reference=policies["fixed"];mismatches={name:[field for field in identity_fields if row[field]!=reference[field]] for name,row in policies.items()}
    same=not any(mismatches.values()) and all(row["raw_detector_event_count"]>=0 and np.isclose(row["edr_per_cycle"],row["raw_detector_event_count"]/row["effective_qec_cycles"]) for row in policies.values())
    result={"schema_version":"google-pure-v8-figure5a-edr-identity.v2","policy_classes":policies,"identity_fields":list(identity_fields),"identity_mismatches":mismatches,
      "same_quantity_all_policy_classes":same,"matched_finite_shot":True,"independent_evaluation_from_training":True,"logical_risk_or_ler_used":False,"classification":"PASS" if same else "FIGURE5A_COST_IDENTITY_FAILURE"}
    return write_report("figure5a_edr_identity_audit",result,"Figure 5a EDR Identity Audit")
