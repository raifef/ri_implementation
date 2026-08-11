"""Candidate-boundary resumable public step-response acquisition."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.gaussian import BehaviorSnapshot, DirectSigmaGaussianPolicy
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.losses import total_loss_and_gradients
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.optimizer import DirectSigmaOptimizer, OptimizerConfig
from .contracts import StepProtocol
from .estimator import estimate_response
from .plant import SourceStepPlant
from hdfa_rl_suite.google_pure_source_exact.source_normalization import (
    SourceNormalizationBoundary,
    require_v15_boundary_provenance,
)

def _write(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True); temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8"); temporary.replace(path)

def run_step_analogue(protocol: StepProtocol, checkpoint: Path, optimizer_config: OptimizerConfig,
                      *, initial_sigma: float=.15, entropy_weight: float=.001, baseline_weight: float=.2,
                      clip: float=.2, resume: bool=False, max_candidate_boundaries: int|None=None,
                      checkpoint_every_candidates: int=1, compact_records: bool=False,
                      experiment_family: str="STEP_RESPONSE_INJECTED_DRIFT",
                      fresh_acquisition_required: bool=False,
                      source_budget_profile: str="UNSPECIFIED_DEVELOPMENT") -> dict:
    if checkpoint_every_candidates < 1:
        raise ValueError("checkpoint_every_candidates must be at least one")
    protocol.validate(); plant=SourceStepPlant(protocol.controls, direction_coordinate=protocol.direction_coordinate,
        target_delta=protocol.target_delta_normalized, onset_epoch=protocol.onset_epoch)
    boundary=SourceNormalizationBoundary.from_training_objective(
        experiment_family, plant.sensitivity,
        control_ids=tuple(f"{experiment_family}:control:{index}" for index in range(protocol.controls)))
    checkpoint_preexisted=checkpoint.exists()
    if checkpoint_preexisted and fresh_acquisition_required:
        raise RuntimeError("fresh V15 acquisition forbids reuse of a lower-level checkpoint")
    if checkpoint.exists():
        if not resume: raise RuntimeError("checkpoint exists; pass resume=True")
        state=json.loads(checkpoint.read_text(encoding="utf-8"))
        if (state["plant_hash"]!=plant.plant_hash or state["protocol"]!=protocol.__dict__ or
                state.get("v15_boundary") != boundary.provenance_fields()):
            raise RuntimeError("checkpoint identity changed")
        if int(state.get("checkpoint_every_candidates", 1)) != checkpoint_every_candidates:
            raise RuntimeError("checkpoint cadence changed")
        expected_storage = "compact_directional" if compact_records else "full_policy"
        if state.get("record_storage", "full_policy") != expected_storage:
            raise RuntimeError("checkpoint record storage changed")
        policy=DirectSigmaGaussianPolicy.from_state_dict(state["policy"])
        optimizer=DirectSigmaOptimizer.from_state_dict(state["policy"]["optimizer_state"])
        baseline=np.asarray(state["policy"]["baseline"],dtype=float)
    else:
        policy=DirectSigmaGaussianPolicy(np.zeros(protocol.controls),np.full(protocol.controls,initial_sigma),seed=protocol.seed)
        optimizer=DirectSigmaOptimizer(protocol.controls,plant.detectors,optimizer_config); baseline=np.zeros(plant.detectors)
        state={"schema":"step-response-checkpoint.v1","protocol":protocol.__dict__,"plant_hash":plant.plant_hash,
               "v15_boundary":boundary.provenance_fields(),
               "epoch":0,"active":None,"records":[],"candidate_boundaries":0,
               "checkpoint_every_candidates":int(checkpoint_every_candidates),
               "record_storage":"compact_directional" if compact_records else "full_policy",
               "policy":policy.state_dict(optimizer_state=optimizer.state_dict(),baseline=baseline)}; _write(checkpoint,state)
    boundaries=0
    while state["epoch"]<protocol.epochs:
        epoch=int(state["epoch"])
        if state["active"] is None:
            batch=policy.sample(protocol.candidates_per_epoch)
            state["active"]={"actions":batch.actions.tolist(),"noise":batch.standardized_noise.tolist(),
                "mean":batch.behavior.mean.tolist(),"sigma":batch.behavior.sigma.tolist(),
                "logp":batch.behavior.component_log_probability.tolist(),"version":batch.behavior.policy_version,
                "next":0,"rewards":[],"candidate_edr":[],"fixed_edr":[],"mean_edr":[],"oracle_edr":[]}; _write(checkpoint,state)
        active=state["active"]
        while active["next"]<protocol.candidates_per_epoch:
            candidate=int(active["next"])
            target_normalized=plant.hidden_target(epoch)
            target_native=boundary.target_to_native(target_normalized)
            candidate_native=boundary.apply(np.asarray(active["actions"][candidate])).native
            fixed_native=boundary.apply(np.zeros(protocol.controls)).native
            mean_native=boundary.apply(np.asarray(active["mean"])).native
            controls={"candidate":candidate_native,"fixed":fixed_native,
                "learned_mean":mean_native,"oracle":target_native}
            counts=plant.common_random_counts(controls,epoch,protocol.cycles_per_candidate,
                seed=protocol.seed+1_000_003*epoch+candidate,
                target_controls=target_native)
            active["rewards"].append((-counts["candidate"]/protocol.cycles_per_candidate).tolist())
            for key,stream in (("candidate_edr","candidate"),("fixed_edr","fixed"),("mean_edr","learned_mean"),("oracle_edr","oracle")):
                active[key].append(float(np.sum(counts[stream])/(protocol.cycles_per_candidate*plant.detectors)))
            active["next"]+=1; state["candidate_boundaries"]+=1; state["active"]=active
            flush_boundary = (active["next"] % checkpoint_every_candidates == 0 or
                              active["next"] == protocol.candidates_per_epoch)
            if flush_boundary: _write(checkpoint,state)
            boundaries+=1
            if max_candidate_boundaries is not None and boundaries>=max_candidate_boundaries:
                if not flush_boundary: _write(checkpoint,state)
                return {"complete":False,"epoch":epoch,"next_candidate":active["next"],"checkpoint":str(checkpoint.resolve())}
        behavior=BehaviorSnapshot(np.asarray(active["mean"]),np.asarray(active["sigma"]),np.asarray(active["logp"]),active["version"])
        loss=total_loss_and_gradients(np.asarray(active["actions"]),np.asarray(active["rewards"]),plant.mask,
            policy.mean,policy.sigma,baseline,behavior,clip=clip,entropy_weight=entropy_weight,baseline_weight=baseline_weight)
        update=optimizer.step(policy.mean,policy.sigma,baseline,loss.grad_mean,loss.grad_sigma,loss.grad_baseline,mean_bounds=(-2.,2.)); policy.policy_version+=1
        record={"epoch":epoch,
            "candidate_edr":active["candidate_edr"],"learned_mean_edr":float(np.mean(active["mean_edr"])),
            "fixed_edr":float(np.mean(active["fixed_edr"])),"oracle_edr":float(np.mean(active["oracle_edr"])),
            "direction_snr":float(abs(policy.mean[protocol.direction_coordinate]) / max(policy.sigma[protocol.direction_coordinate],1e-12)),
            "fraction_at_sigma_guard":update["fraction_at_positivity_guard"],"reward_sigma_gradient_norm":loss.diagnostics["reward_sigma_gradient_norm"],
            **boundary.provenance_fields()}
        if compact_records:
            keep=np.ones(protocol.controls,dtype=bool); keep[protocol.direction_coordinate]=False
            record.update({
                "mean_before_direction":float(active["mean"][protocol.direction_coordinate]),
                "mean_after_direction":float(policy.mean[protocol.direction_coordinate]),
                "sigma_direction":float(policy.sigma[protocol.direction_coordinate]),
                "off_direction_mean_l2":float(np.linalg.norm(policy.mean[keep])),
                "off_direction_sigma_mean":float(np.mean(policy.sigma[keep])),
            })
        else:
            record.update({"mean_before":active["mean"],"mean_after":policy.mean.tolist(),
                           "sigma":policy.sigma.tolist()})
        state["records"].append(record)
        state["epoch"]+=1; state["active"]=None; state["policy"]=policy.state_dict(optimizer_state=optimizer.state_dict(),baseline=baseline); _write(checkpoint,state)
    response=np.asarray([(
        row["mean_after_direction"] if "mean_after_direction" in row else
        row["mean_after"][protocol.direction_coordinate]) / protocol.target_delta_normalized
        for row in state["records"]])
    result={"complete":True,"response":estimate_response(response,onset_epoch=protocol.onset_epoch,bootstrap_seed=protocol.seed),
            "records":state["records"],"candidate_boundaries":state["candidate_boundaries"],
            "plant_hash":plant.plant_hash,"paper_comparable":False,
            "checkpoint_every_candidates":int(checkpoint_every_candidates),
            "record_storage":state["record_storage"],
            "fresh_acquisition":not checkpoint_preexisted,"reused_shard_ids":[],
            "source_budget_profile":str(source_budget_profile),
            "boundary_trace":boundary.trace(np.eye(1, protocol.controls, protocol.direction_coordinate).ravel()),
            **boundary.provenance_fields()}
    require_v15_boundary_provenance(result)
    return result
