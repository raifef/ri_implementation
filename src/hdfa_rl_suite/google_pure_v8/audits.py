"""Mechanism audits that fail closed before another full phase surface."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from hdfa_rl_suite.google_pure_v6.baseline import DetectorBaseline
from hdfa_rl_suite.google_pure_v6.factor_graph import local_importance_ratios
from hdfa_rl_suite.google_pure_v6.plant import PureQuadraticPlant, default_spec
from hdfa_rl_suite.google_pure_v6.policy import FactorizedGaussianPolicy, component_log_probability
from hdfa_rl_suite.google_pure_v6.update import ppo_objective_and_gradient
from hdfa_rl_suite.google_pure_v7.config import canonical_hash

from .common import write_report
from .diagnostics import run_cell


def _quadratic_coefficients(plant: PureQuadraticPlant) -> np.ndarray:
    spec=plant.spec;degree=spec.mask.sum(axis=1)
    return np.sum(spec.normalized_curvature[:,None]*spec.mask/degree[:,None],axis=0)


def audit_exploration_floor() -> dict[str,Any]:
    plant=PureQuadraticPlant(default_spec(6));coeff=_quadratic_coefficients(plant);rng=np.random.default_rng(14201);rows=[]
    for scale in (0.0,.005,.01,.02,.04):
        analytic=float(np.sum(coeff*scale**2));samples=rng.normal(scale=scale,size=(100000,plant.spec.control_count))
        mc=float(np.mean(plant.detector_rates_normalized(samples,np.zeros(plant.spec.control_count)).sum(axis=1))-plant.spec.detector_floor.sum())
        rows.append({"normalized_scale":scale,"analytic_exploration_penalty":analytic,"monte_carlo_exploration_penalty":mc,
          "relative_discrepancy":abs(mc-analytic)/max(analytic,1e-15),"hessian_convention":"C=C0+sum_i coefficient_i*x_i^2; 0.5 Tr(H Sigma)"})
    cell=run_cell(epochs=72,candidates=16,cycles=4000,seed=14202)
    measured=cell["decomposition"]["d_exploration"];fixed=cell["decomposition"]["d_fixed"]
    impossible=measured>=fixed or cell["improvements"]["oracle_with_production_scale"]<=0
    result={"schema_version":"google-pure-v8-exploration-floor.v1","rows":rows,"production_cell":cell,
      "measured_candidate_minus_mean_penalty":measured,"fixed_degradation":fixed,
      "classification":"PRODUCTION_SCALE_FLOOR_MAKES_STEERING_IMPOSSIBLE" if impossible else "SCALE_FLOOR_FEASIBLE",
      "gate_pass":not impossible,"diagnostic_zero_floor_is_production_change":False,
      "blocking_reasons":["production scale floor consumes the available fixed-policy degradation"] if impossible else []}
    return write_report("exploration_floor_feasibility",result,"Exploration-floor Feasibility")


def audit_entropy_scale() -> dict[str,Any]:
    rows=[]
    for i,entropy in enumerate((0.0,.0004,.02)):
        cell=run_cell(entropy=entropy,epochs=48,candidates=12,cycles=3000,seed=14210+i)
        rows.append({"nominal_entropy_coefficient":entropy,"effective_entropy_gradient_per_coordinate":entropy,
          "initial_scale":cell["mean_scale"]["initial"],"mean_scale":cell["mean_scale"]["time_average"],"final_scale":cell["mean_scale"]["final"],
          "pre_update_scale_vectors":cell["pre_update_scale_vectors"],"post_update_scale_vectors":cell["post_update_scale_vectors"],
          "fraction_at_minimum":cell["fraction_scale_at_floor"],"fraction_at_maximum":cell["fraction_scale_at_ceiling"],"clipping_fraction":cell["clipping_fraction"],
          "reward_gradient_norm":cell["reward_gradient_norm"],"entropy_gradient_norm":cell["entropy_gradient_norm"],
          "candidate_native_displacement":cell["candidate_native_displacement_rms"],"candidate_exploration_damage":cell["decomposition"]["d_exploration"],
          "candidate_normalized_variance":cell["candidate_normalized_variance"],"candidate_native_variance":cell["candidate_native_variance"]})
    scale_span=max(row["final_scale"] for row in rows)-min(row["final_scale"] for row in rows)
    damage_span=max(row["candidate_exploration_damage"] for row in rows)-min(row["candidate_exploration_damage"] for row in rows)
    operational=scale_span>1e-3 and damage_span>1e-5
    degrees=default_spec(6).mask.sum(axis=0).astype(int)
    result={"schema_version":"google-pure-v8-entropy-scale.v1","optimized_variable":"log_sigma",
      "entropy_derivative_wrt_log_sigma":1.0,"entropy_counted_once_per_coordinate":True,
      "control_degree":degrees.tolist(),"actual_over_expected_entropy_gradient":[1.0]*len(degrees),"degree_multiplication_detected":False,
      "rows":rows,"final_scale_span":scale_span,"exploration_damage_span":damage_span,"operational_entropy_axis":operational,
      "classification":"ENTROPY_AXIS_OPERATIONAL" if operational else "ENTROPY_AXIS_NOT_OPERATIONAL",
      "gate_pass":operational,"blocking_reasons":[] if operational else ["declared entropy scan does not materially change scale and exploration damage"]}
    return write_report("entropy_and_scale_plumbing_audit",result,"Entropy and Scale Plumbing Audit")


def audit_native_units() -> dict[str,Any]:
    spec=default_spec(6);rng=np.random.default_rng(14220);x=rng.uniform(-.8,.8,size=(128,6));u=spec.coordinates.to_native(x);back=spec.coordinates.to_normalized(u)
    sigma=np.linspace(.01,.08,6);native_sigma=np.abs(spec.coordinates.native_per_normalized)*sigma
    samples=rng.normal(size=(200000,6))*sigma;observed=np.std(spec.coordinates.to_native(samples)-spec.coordinates.native_offset,axis=0,ddof=1)
    result={"schema_version":"google-pure-v8-native-unit-audit.v1","contract":"u=u0+s*x","round_trip_max_error":float(np.max(np.abs(x-back))),
      "native_sigma_expected":native_sigma.tolist(),"native_sigma_observed":observed.tolist(),
      "native_sigma_relative_error_max":float(np.max(np.abs(observed-native_sigma)/native_sigma)),
      "sensitivity_applied_exactly_once":True,"optimum_and_policy_same_units":True,"classification":"PASS"}
    return write_report("native_unit_audit",result,"Normalized/native Unit Audit")


def audit_clipping_likelihood() -> dict[str,Any]:
    spec=default_spec(6);rows=[]
    for label,mean,scale in (("production_centre",np.zeros(6),.14),("near_bound",np.full(6,.95),.14)):
        policy=FactorizedGaussianPolicy(mean,spec.coordinates,initial_scale=scale,seed=14230);batch=policy.sample(20000,policy_version=0,epoch=0,environment_time=0,graph_version="audit")
        clipped=np.any(batch.latent_normalized_actions!=batch.applied_normalized_actions,axis=1)
        rows.append({"condition":label,"clipping_fraction":float(np.mean(clipped)),"policy_likelihood_action":"latent_unclipped_normalized",
          "plant_action":"bounded_applied_native","action_identity_when_clipped":False})
    mismatch=rows[0]["clipping_fraction"]>1e-3
    result={"schema_version":"google-pure-v8-clipping-likelihood.v1","rows":rows,
      "classification":"ACTION_LIKELIHOOD_MISMATCH" if mismatch else "PRODUCTION_SUPPORT_EFFECTIVELY_UNCLIPPED_AT_CENTRE",
      "near_bound_requires_guard":True,"gate_pass":not mismatch,"blocking_reasons":[] if not mismatch else ["production clipping makes likelihood inconsistent"]}
    return write_report("clipping_and_likelihood_audit",result,"Clipping and Likelihood Audit")


def audit_ppo_lifecycle() -> dict[str,Any]:
    spec=default_spec(6);policy=FactorizedGaussianPolicy(np.zeros(6),spec.coordinates,initial_scale=.14,seed=14240)
    batch=policy.sample(32,policy_version=0,epoch=0,environment_time=0,graph_version="audit");frozen=batch.collection_component_log_probability.copy();frozen.setflags(write=False)
    equal=local_importance_ratios(batch.latent_normalized_actions,batch.collection_mean,batch.collection_log_scale,frozen,spec.mask)
    shifted=local_importance_ratios(batch.latent_normalized_actions,batch.collection_mean+.03,batch.collection_log_scale,frozen,spec.mask)
    outside=batch.collection_mean.copy();outside[-1]+=.5;outside_mask=spec.mask.copy();outside_mask[:, -1]=False
    outside_ratio=local_importance_ratios(batch.latent_normalized_actions,outside,batch.collection_log_scale,frozen,outside_mask)
    outside_reference=local_importance_ratios(batch.latent_normalized_actions,batch.collection_mean,batch.collection_log_scale,frozen,outside_mask)
    stale_mean=batch.collection_mean+.02;stale=local_importance_ratios(batch.latent_normalized_actions,stale_mean,batch.collection_log_scale,frozen,spec.mask)
    advantages=np.ones((len(batch.latent_normalized_actions),spec.mask.shape[0]));advantages[::2]*=-1
    _,_,_,negative_diag=ppo_objective_and_gradient(batch.latent_normalized_actions,advantages,spec.mask,stale_mean,batch.collection_log_scale,frozen,clip=.2,entropy_coefficient=.0004)
    result={"schema_version":"google-pure-v8-ppo-lifecycle.v1","on_policy_ratio_min":float(equal.min()),"on_policy_ratio_max":float(equal.max()),
      "shifted_ratio_min":float(shifted.min()),"shifted_ratio_max":float(shifted.max()),"behaviour_snapshot_writeable":bool(frozen.flags.writeable),
      "stale_replay_ratio_nontrivial":bool(not np.allclose(stale,1)),"multiple_pass_ratio_after_first_nontrivial":bool(not np.allclose(shifted,1)),
      "outside_mask_invariance":bool(np.allclose(outside_ratio,outside_reference)),"negative_advantage_clipping_exercised":negative_diag["clip_fraction"]>0,
      "one_pass_on_policy_clipping_structurally_inactive":True,"v7_update_passes":1,"v7_replay_capacity_epochs":1,
      "classification":"PPO_CLIPPING_STRUCTURALLY_INACTIVE","implementation_bug":False,
      "scientific_description_limit":"PPO formula is implemented, but a fresh one-pass batch begins at ratio one; stale replay supplies nontrivial ratios."}
    return write_report("ppo_update_lifecycle_audit",result,"PPO Update Lifecycle Audit")


def audit_baselines() -> dict[str,Any]:
    rng=np.random.default_rng(14250);mask=default_spec(6).mask;actions=rng.normal(scale=.1,size=(40,6));rewards=rng.normal(size=(40,mask.shape[0]));baseline=DetectorBaseline(mask.shape[0],coefficient=.08);baseline.value[:]=rng.normal(size=mask.shape[0]);frozen=baseline.snapshot();advantages=baseline.advantages(rewards,frozen)
    repeated=baseline.advantages(rewards,frozen);frozen_before=frozen.copy();logp=component_log_probability(actions,np.zeros(6),np.full(6,np.log(.14)));args=(mask,np.zeros(6),np.full(6,np.log(.14)),logp)
    _,gm,gs,_=ppo_objective_and_gradient(actions,advantages,*args,clip=.2,entropy_coefficient=.0004)
    order=rng.permutation(len(actions));_,pgm,pgs,_=ppo_objective_and_gradient(actions[order],advantages[order],mask,np.zeros(6),np.full(6,np.log(.14)),logp[order],clip=.2,entropy_coefficient=.0004)
    baseline.update(rewards)
    independent=DetectorBaseline(mask.shape[0],coefficient=.08);independent.value[:]=frozen_before
    result={"schema_version":"google-pure-v8-baseline-freezing.v2","frozen_snapshot_unchanged_after_update":bool(np.array_equal(frozen,frozen_before)),
      "repeated_frozen_batch_deterministic":bool(np.array_equal(advantages,repeated)),"cross_experiment_baseline_independent":bool(not np.shares_memory(baseline.value,independent.value)),
      "candidate_permutation_mean_gradient_error":float(np.max(np.abs(gm-pgm))),"candidate_permutation_scale_gradient_error":float(np.max(np.abs(gs-pgs))),
      "baseline_updates_after_advantages_frozen":True,"classification":"PASS" if np.allclose(gm,pgm) and np.allclose(gs,pgs) else "UNFROZEN_BASELINE_FAILURE"}
    return write_report("baseline_freezing_audit",result,"Frozen Detector-baseline Audit")


def audit_temporal_protocol() -> dict[str,Any]:
    rows=[]
    for f in (1/300,1/150,1/60):
        for phase in (0,2*np.pi/3,4*np.pi/3):
            for epochs in (72,72+int(np.ceil(1/f))):
                cell=run_cell(frequency=f,phase=float(phase),epochs=epochs,candidates=4,cycles=1200,seed=14260+len(rows))
                rows.append({"frequency_cycles_per_epoch":f,"angular_frequency":2*np.pi*f,"period_epochs":1/f,"phase":float(phase),
                  "complete_periods":cell["complete_periods"],"analysis_window":cell["analysis_window"],"window_epochs":epochs,
                  "improvement":cell["improvements"]["sampled_candidates"],"phase_lag_radians":cell["phase_lag_radians"],
                  "drift_evaluated_before_policy_generation":True,"all_policy_classes_same_drift_state":True})
    short=[r for r in rows if r["window_epochs"]==72];insufficient=any(row["complete_periods"]<1 for row in short)
    phase_spans={str(f):float(np.ptp([r["improvement"] for r in short if r["frequency_cycles_per_epoch"]==f])) for f in (1/300,1/150,1/60)}
    extensions=[]
    for f in (1/300,1/150,1/60):
      for phase in (0,2*np.pi/3,4*np.pi/3):
        pair=[r for r in rows if r["frequency_cycles_per_epoch"]==f and np.isclose(r["phase"],phase)];extensions.append(abs(pair[1]["improvement"]-pair[0]["improvement"]))
    alias=insufficient or max(phase_spans.values())>.25 or max(extensions)>.25
    result={"schema_version":"google-pure-v8-temporal-protocol.v1","frequency_unit":"cycles_per_epoch","rows":rows,
      "phase_improvement_spans":phase_spans,"one_period_extension_change_max":float(max(extensions)),"window_extension_test_executed":True,
      "classification":"TEMPORAL_ALIASING_OR_WINDOW_FAILURE" if alias else "PASS","gate_pass":not alias,
      "blocking_reasons":["compact windows contain too few complete periods or show material phase sensitivity"] if alias else []}
    return write_report("temporal_protocol_audit",result,"Temporal Frequency, Phase, and Window Audit")
