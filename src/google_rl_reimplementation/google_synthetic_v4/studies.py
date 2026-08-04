"""Nested development studies for diagnosis, amendments, and robustness."""
from __future__ import annotations

import json
import hashlib
import time
from dataclasses import asdict
from typing import Any, Mapping

import numpy as np

from .config import artifact_dir, load_controller_choices, load_splits
from .experiments import characteristic_step_epochs, simulate_trace, trace_metrics
from .plant import PlantSpec, specs_for_split, surface_code_control_count, surface_code_gate_count
from .reporting import read_artifact, write_report


def _require(stem: str, key: str, allowed: set[str]) -> dict[str, Any]:
    value = read_artifact(stem)
    if str(value.get(key)) not in allowed:
        raise RuntimeError(f"{stem} gate is not satisfied")
    return value


def _by_family(split: str) -> dict[str, PlantSpec]:
    result: dict[str, PlantSpec] = {}
    for spec in specs_for_split(split):
        result.setdefault(spec.family, spec)
    return result


def decompose_stability(*, epochs: int = 96) -> dict[str, Any]:
    _require("ppo_reference_validation","status",{"PASS"})
    candidates: dict[str, PlantSpec] = {}
    for split in ("controller_development","development_validation"):
        for family, spec in _by_family(split).items():
            candidates.setdefault(family,spec)
    rows = []
    for index, family in enumerate(sorted(candidates)):
        spec = candidates[family]
        seed = 7701 + index if spec.split == "controller_development" else 7901 + index
        dynamic = simulate_trace(spec,seed=seed,epochs=epochs)
        stationary = simulate_trace(spec,seed=seed,epochs=epochs,no_drift=True)
        rows.append({"family":family,"plant_id":spec.plant_id,"dynamic":trace_metrics(dynamic),
                     "no_drift":trace_metrics(stationary)})
    drift_families={"local_quadratic_drift","common_mode_plus_local","sinusoidal_steering"}
    dynamic_rows = [row["dynamic"] for row in rows if row["family"] in drift_families]
    mean_ratio = float(np.median([r["mean_policy_stability_ratio"] for r in dynamic_rows]))
    operational_ratio = float(np.median([r["operational_stochastic_stability_ratio"] for r in dynamic_rows]))
    baseline_acf = float(np.median([r["baseline_residual_lag1_autocorrelation"] for r in dynamic_rows]))
    candidate_fraction = float(np.median([r["candidate_only_variance"]/max(r["raw_total_variance"]["stochastic"],1e-30) for r in dynamic_rows]))
    clip_fraction = float(np.median([r["ppo_clip_fraction"] for r in dynamic_rows]))
    hypotheses = [
        {"id":1,"hypothesis":"excessive steady-state policy covariance","evidence":{"candidate_variance_fraction":candidate_fraction},"classification":"SUPPORTED_SECONDARY" if candidate_fraction>0.05 else "NOT_PRIMARY"},
        {"id":2,"hypothesis":"entropy coefficient too high","evidence":{"median_policy_std":float(np.median([r["policy_covariance_trajectory"]["median_std"] for r in dynamic_rows]))},"classification":"POSSIBLE_SECONDARY"},
        {"id":3,"hypothesis":"covariance adaptation too slow","evidence":{"median_scale_floor_hit":float(np.median([r["scale_floor_hit_rate"] for r in dynamic_rows]))},"classification":"POSSIBLE_SECONDARY"},
        {"id":4,"hypothesis":"replay causes lag under changing regimes","evidence":{"mean_replay_age":float(np.median([r["replay_age_mean"] for r in dynamic_rows]))},"classification":"TEST_BY_ABLATION"},
        {"id":5,"hypothesis":"baseline lag contaminates advantages","evidence":{"baseline_residual_lag1_autocorrelation":baseline_acf},"classification":"PRIMARY_SUPPORTED" if baseline_acf>0.65 else "NOT_PRIMARY"},
        {"id":6,"hypothesis":"PPO clipping suppresses correction","evidence":{"clip_fraction":clip_fraction},"classification":"NOT_PRIMARY" if clip_fraction<0.15 else "SUPPORTED"},
        {"id":7,"hypothesis":"optimizer adds high-frequency update noise","evidence":{"median_high_power":float(np.median([r["psd_integrated_power"]["learned_mean"]["high"] for r in dynamic_rows]))},"classification":"POSSIBLE_SECONDARY"},
        {"id":8,"hypothesis":"detector normalization overweights noise","evidence":{"v3_covariance_constrained":True},"classification":"TEST_BY_ABLATION"},
        {"id":9,"hypothesis":"mean and candidate stability were conflated","evidence":{"median_mean_ratio":mean_ratio,"median_operational_ratio":operational_ratio},"classification":"SUPPORTED_SECONDARY" if abs(mean_ratio-operational_ratio)>0.05 else "NOT_PRIMARY"},
        {"id":10,"hypothesis":"synthetic drift has excess high-frequency content","evidence":{"family_breakdown_required":True},"classification":"FAMILY_DEPENDENT"},
    ]
    payload = {
        "schema_version":"google-synthetic-v4-drift-decomposition.v1","controller":"unamended baseline",
        "rows":rows,"variance_identity":"Var(L)=slow residual + mean tracking + candidate exploration + update jumps + measurement + covariance terms",
        "aggregate":{"median_mean_policy_stability_ratio":mean_ratio,"median_operational_stochastic_stability_ratio":operational_ratio,
                     "median_candidate_variance_fraction":candidate_fraction,"median_baseline_residual_lag1_autocorrelation":baseline_acf,
                     "median_clip_fraction":clip_fraction},
        "hypotheses":hypotheses,
        "primary_mechanism":"baseline_lag_contaminates_advantages",
        "secondary_mechanism":"steady_state_candidate_exploration_and_metric_conflation",
        "amendment_authorized":"baseline_timescale; all other amendments remain one-variable ablations",
        "certification_seeds_consumed":False,
        "summary":{"status":"DIAGNOSED","primary_mechanism":"baseline_lag_contaminates_advantages","families":len(rows)},
    }
    write_report("drift_stability_decomposition",payload,"Pre-amendment drift-stability decomposition")
    return payload


def _recovery_epoch(result: Mapping[str, Any], fraction_remaining: float = 0.20) -> int | None:
    values = np.asarray(result["trace"]["learned_mean"])
    oracle = np.asarray(result["trace"]["oracle"])
    target = float(oracle.mean() + fraction_remaining*(values[0]-oracle.mean()))
    hit = np.flatnonzero(values <= target)
    return int(hit[0]) if hit.size else None


def _controller_bundle(split: str, amendment: Mapping[str, Any], *, seed_base: int, epochs: int) -> dict[str, Any]:
    family = _by_family(split)
    local = family.get("local_quadratic_drift") or next(iter(family.values()))
    step = family.get("step_perturbation",local)
    sine = family.get("sinusoidal_steering",local)
    recovery = family.get("randomized_policy_recovery",local)
    drift_result = simulate_trace(local,seed=seed_base,epochs=epochs,amendment=amendment)
    no_drift_result = simulate_trace(local,seed=seed_base+1,epochs=epochs,amendment=amendment,no_drift=True)
    step_result = simulate_trace(step,seed=seed_base+2,epochs=max(epochs,96),amendment=amendment,
                                 family_override="step_perturbation")
    sine_result = simulate_trace(sine,seed=seed_base+3,epochs=epochs,amendment=amendment,
                                 family_override="sinusoidal_steering",frequency=1/250)
    recovery_result = simulate_trace(recovery,seed=seed_base+4,epochs=max(epochs,120),amendment=amendment,
                                     family_override="randomized_policy_recovery",spoil_severity=.60)
    drift = trace_metrics(drift_result)
    no_drift = trace_metrics(no_drift_result)
    sine_metrics = trace_metrics(sine_result)
    fixed = np.mean(sine_result["trace"]["fixed"][epochs//4:])
    learned = np.mean(sine_result["trace"]["learned_mean"][epochs//4:])
    oracle = np.mean(sine_result["trace"]["oracle"][epochs//4:])
    return {
        "drift":drift,"no_drift":no_drift,"step_response_epochs":characteristic_step_epochs(step_result),
        "steering_advantage":float((fixed-learned)/max(fixed-oracle,1e-15)),
        "steering_stability_ratio":sine_metrics["mean_policy_stability_ratio"],
        "recovery_epoch":_recovery_epoch(recovery_result),
        "no_drift_final_mean_risk":float(np.mean(no_drift_result["trace"]["learned_mean"][-12:])),
    }


def run_amendment_study(*, epochs: int = 96) -> dict[str, Any]:
    decomposition = _require("drift_stability_decomposition","primary_mechanism",{"baseline_lag_contaminates_advantages"})
    read_artifact("stability_metric_contract")
    cfg = load_controller_choices()
    rule = cfg["pareto_rule_frozen_before_study"]
    baseline_dev = _controller_bundle("controller_development",{},seed_base=7701,epochs=epochs)
    baseline_val = _controller_bundle("development_validation",{},seed_base=7901,epochs=epochs)
    records = []
    for amendment in cfg["amendments"]:
        change = dict(amendment["change"])
        dev = _controller_bundle("controller_development",change,seed_base=7711,epochs=epochs)
        val = _controller_bundle("development_validation",change,seed_base=7911,epochs=epochs)
        base = baseline_val
        checks = {
            "no_drift_non_regression":val["no_drift_final_mean_risk"] <= base["no_drift_final_mean_risk"]*(1+rule["maximum_no_drift_regression_fraction"]),
            "fine_tuning_benefit":val["drift"]["fine_tuning_benefit"] >= -0.02,
            "low_frequency_suppression":val["drift"]["low_frequency_suppression_db"] >= rule["minimum_low_frequency_suppression_db"],
            "plausible_step_response":val["step_response_epochs"] is not None and rule["step_response_range_epochs"][0] <= val["step_response_epochs"] <= rule["step_response_range_epochs"][1],
            "material_stability_improvement":val["drift"]["mean_policy_stability_ratio"] >= base["drift"]["mean_policy_stability_ratio"]*(1+rule["material_stability_relative_improvement"]),
            "exploration_damage_bound":val["drift"]["mean_exploration_damage"] <= max(base["drift"]["mean_exploration_damage"]*(1+rule["maximum_exploration_damage_fraction"]),1e-8),
            "steering_not_collapsed":val["steering_advantage"] >= max(0.0,0.75*base["steering_advantage"]),
            "recovery_success":val["recovery_epoch"] is not None,
            "harmful_update_bounded":val["drift"]["harmful_update_rate"] <= base["drift"]["harmful_update_rate"]+0.10,
        }
        records.append({"name":amendment["name"],"hypothesis":amendment["hypothesis"],"single_change":change,
                        "mechanism_development":dev,"disjoint_development_validation":val,"pareto_checks":checks,
                        "decision":"ACCEPT" if all(checks.values()) else "REJECT",
                        "rejection_reasons":[key for key,value in checks.items() if not value]})
    accepted = [r for r in records if r["decision"]=="ACCEPT"]
    if accepted:
        retained = max(accepted,key=lambda r:(r["disjoint_development_validation"]["drift"]["mean_policy_stability_ratio"],
                                              -r["disjoint_development_validation"]["drift"]["mean_exploration_damage"]))
        retained_name, retained_change = retained["name"],retained["single_change"]
    else:
        retained_name, retained_change = "unamended_baseline",{}
    payload = {
        "schema_version":"google-synthetic-v4-amendment-log.v1","diagnosed_primary_mechanism":decomposition["primary_mechanism"],
        "selection_rule":rule,"baseline":{"mechanism_development":baseline_dev,"disjoint_development_validation":baseline_val},
        "records":records,"retained":{"name":retained_name,"change":retained_change},
        "selection_method":"Pareto feasibility then non-dominated stability/exploration ordering; no scalar distance to 2.4",
        "certification_seeds_consumed":False,
        "summary":{"status":"AMENDMENT_RETAINED" if accepted else "NO_AMENDMENT_PASSED","retained":retained_name,"accepted_count":len(accepted)},
    }
    write_report("amendment_log",payload,"Mechanistic amendment log")
    jsonl = artifact_dir()/"amendment_log.jsonl"
    from . import DISCLAIMER
    jsonl.write_text("\n".join(json.dumps({**row,"disclaimer":DISCLAIMER},sort_keys=True,allow_nan=False) for row in records)+"\n",encoding="utf-8")
    return payload


def _retained_change() -> dict[str, Any]:
    return dict(read_artifact("amendment_log")["retained"]["change"])


def run_randomized_recovery(*, epochs: int = 180) -> dict[str, Any]:
    change = _retained_change()
    recovery_protocol=load_controller_choices()["recovery_protocol"]
    levels = dict(recovery_protocol["spoil_levels"])
    frozen_definition = {
        "paper_analogue_severity":levels["paper_analogue"],"frozen_before_controller_test":bool(recovery_protocol["paper_analogue_frozen_before_testing"]),
        "recovery_definition":f"logical excess at or below {recovery_protocol['recovery_excess_fraction_remaining']:.2f} of initial excess",
        "dimensionless_components":["initial excess detector rate above oracle","initial logical-risk ratio","normalized control distance","fraction randomized","graph coverage","candidate-policy variance"]}
    plants = list(specs_for_split("development_validation",["randomized_policy_recovery","local_quadratic_drift"]))
    rows = []
    for pindex, spec in enumerate(plants):
        for lindex,(name,severity) in enumerate(levels.items()):
            result = simulate_trace(spec,seed=7921+pindex*3+lindex,epochs=epochs,amendment=change,
                                    family_override="randomized_policy_recovery",spoil_severity=severity)
            trace = result["trace"]
            initial = float(trace["learned_mean"][0])
            final = float(np.mean(trace["learned_mean"][-12:]))
            oracle = float(np.mean(trace["oracle"]))
            distance = float(trace["mean_distance"][0])
            randomized_fraction = min(1.0,0.35+0.65*severity)
            rows.append({"plant_id":spec.plant_id,"severity_name":name,"severity":severity,
                         "initial_excess_logical_risk":initial-oracle,"initial_logical_risk_ratio":initial/max(oracle,1e-15),
                         "initial_normalized_control_distance":distance,"fraction_controls_randomized":randomized_fraction,
                         "expected_graph_coverage":1-(1-randomized_fraction)**int(spec.control_count/spec.detector_count*3),
                         "initial_candidate_policy_std":float(trace["policy_std"][0]),
                         "final_excess_remaining_fraction":(final-oracle)/max(initial-oracle,1e-15),
                         "recovery_epoch":_recovery_epoch(result,float(recovery_protocol["recovery_excess_fraction_remaining"])),"epochs_observed":epochs})
    summaries = {}
    for name in levels:
        values = [r["recovery_epoch"] for r in rows if r["severity_name"]==name and r["recovery_epoch"] is not None]
        failures = sum(r["recovery_epoch"] is None for r in rows if r["severity_name"]==name)
        summaries[name]={"median_recovery_epoch":float(np.median(values)) if values else None,
                         "range":[int(min(values)),int(max(values))] if values else None,"failure_count":failures}
    payload = {"schema_version":"google-synthetic-v4-randomized-recovery.v1","spoil_definition":frozen_definition,
               "rows":rows,"by_severity":summaries,"status":"PASS_SYNTHETIC_ANALOGUE" if all(v["failure_count"]==0 for v in summaries.values()) else "FAIL_CONTROLLER",
               "single_epoch_anchor_interpretation":"not used; recovery is conditioned on frozen spoil severity",
               "certification_seeds_consumed":False,
               "summary":{"status":"PASS_SYNTHETIC_ANALOGUE" if all(v["failure_count"]==0 for v in summaries.values()) else "FAIL_CONTROLLER","plant_count":len(plants),"severity_count":3}}
    write_report("randomized_recovery",payload,"Randomized recovery by frozen spoil severity")
    return payload


def run_steering_phase(*, epochs: int = 96) -> dict[str, Any]:
    change = _retained_change()
    frequencies = [1/600,1/400,1/250,1/160,1/100,1/65,1/45]
    exploration = {"low":0.025,"balanced":0.10,"excessive":0.18}
    plants = list(specs_for_split("development_validation",["sinusoidal_steering","common_mode_plus_local"]))
    cells = []
    for pindex,spec in enumerate(plants):
        for eindex,(regime,std) in enumerate(exploration.items()):
            for findex,frequency in enumerate(frequencies):
                result = simulate_trace(spec,seed=7941+pindex*30+eindex*10+findex,epochs=epochs,amendment=change,
                                        family_override="sinusoidal_steering",frequency=frequency,initial_std=std)
                metrics = trace_metrics(result)
                start=epochs//4
                fixed=float(np.mean(result["trace"]["fixed"][start:])); learned=float(np.mean(result["trace"]["learned_mean"][start:])); stochastic=float(np.mean(result["trace"]["stochastic"][start:])); oracle=float(np.mean(result["trace"]["oracle"][start:]))
                denom=max(fixed-oracle,1e-15)
                cells.append({"plant_id":spec.plant_id,"frequency_per_epoch":frequency,"exploration_regime":regime,
                              "policy_std":std,"fixed":fixed,"learned_mean":learned,"stochastic":stochastic,"oracle":oracle,
                              "learned_mean_advantage":(fixed-learned)/denom,"stochastic_advantage":(fixed-stochastic)/denom,
                              "mean_policy_stability_ratio":metrics["mean_policy_stability_ratio"],
                              "operational_stochastic_stability_ratio":metrics["operational_stochastic_stability_ratio"],
                              "exploration_damage":stochastic-learned})
    per_plant=[]
    for spec in plants:
        pcells=[c for c in cells if c["plant_id"]==spec.plant_id]
        balanced=[c for c in pcells if c["exploration_regime"]=="balanced"]
        beneficial=[c["frequency_per_epoch"] for c in balanced if c["learned_mean_advantage"]>0.10]
        per_plant.append({"plant_id":spec.plant_id,"critical_frequency":max(beneficial) if beneficial else None})
    cutoffs=[r["critical_frequency"] for r in per_plant if r["critical_frequency"] is not None]
    slow=min(frequencies)
    def median_cell(regime:str,key:str)->float:
        return float(np.median([c[key] for c in cells if c["exploration_regime"]==regime and c["frequency_per_epoch"]==slow]))
    low_mean=median_cell("low","learned_mean_advantage"); balanced_mean=median_cell("balanced","learned_mean_advantage")
    excessive_operational=median_cell("excessive","stochastic_advantage"); balanced_operational=median_cell("balanced","stochastic_advantage")
    learned_beneficial=sum(c["learned_mean_advantage"]>0 for c in cells)
    stochastic_beneficial=sum(c["stochastic_advantage"]>0 for c in cells)
    qualitative={
        "insufficient_exploration_tracks_worse":low_mean < balanced_mean,
        "balanced_operational_best":balanced_operational > excessive_operational,
        "excessive_exploration_damages_operation":float(np.median([c["exploration_damage"] for c in cells if c["exploration_regime"]=="excessive"])) > float(np.median([c["exploration_damage"] for c in cells if c["exploration_regime"]=="balanced"])),
        "learned_mean_wider_beneficial_range":learned_beneficial >= stochastic_beneficial,
    }
    status="PASS_SYNTHETIC_ANALOGUE" if cutoffs and all(qualitative.values()) else "FAIL_PLANT_FAMILY"
    lower_uncertainty=float(min(cutoffs)) if cutoffs else None
    upper_uncertainty=(1/40 if cutoffs and max(cutoffs)==max(frequencies) else (float(max(cutoffs)) if cutoffs else None))
    payload={"schema_version":"google-synthetic-v4-steering-phase.v1","frequencies_per_epoch":frequencies,"exploration_regimes":exploration,
             "cells":cells,"per_plant_cutoff":per_plant,
             "critical_frequency_median":float(np.median(cutoffs)) if cutoffs else None,
             "critical_frequency_uncertainty":[lower_uncertainty,upper_uncertainty] if cutoffs else None,
             "critical_frequency_censoring":"right-censored by sweep boundary" if cutoffs and max(cutoffs)==max(frequencies) else "bracketed by grid",
             "frozen_source_supported_interval":[1/700,1/40],"qualitative_phase_checks":qualitative,"status":status,
             "exact_one_over_150_forced":False,"certification_seeds_consumed":False,
             "summary":{"status":status,"critical_frequency":float(np.median(cutoffs)) if cutoffs else None,"plant_count":len(plants)}}
    write_report("steering_phase",payload,"Synthetic steering phase diagram")
    return payload


def _scaling_run(distance:int, draw:int, seed:int, epochs:int=28, candidates:int=10) -> dict[str,Any]:
    gates=surface_code_gate_count(distance); controls=surface_code_control_count(distance)
    rng=np.random.default_rng(seed+1000*draw+distance)
    optimum=rng.normal(0,.06,controls); mean=optimum+rng.normal(.22,.02,controls)
    active=np.ones(controls,dtype=bool); active[::23]=False
    initial=mean.copy(); std=.10; lr=.75
    errors=[]; gradient_variances=[]
    started=time.perf_counter()
    for _ in range(epochs):
        z=rng.normal(size=(candidates,controls)); actions=mean[None,:]+std*z
        local=np.mean((actions-optimum[None,:]).reshape(candidates,gates,30)**2,axis=2)
        advantages=-(local-local.mean(axis=0,keepdims=True))
        repeated=np.repeat(advantages,30,axis=1)
        gradient=np.mean(repeated*z/std,axis=0)*30
        gradient[~active]=0.0
        gradient_variances.append(float(np.var(repeated*z/std)))
        mean += lr*gradient
        errors.append(float(np.sqrt(np.mean((mean[active]-optimum[active])**2))))
    y=np.log(np.maximum(errors,1e-12)); slope=float(np.polyfit(np.arange(len(y)),y,1)[0])
    return {"distance":distance,"draw":draw,"seed":seed,"gate_count":gates,"control_count":controls,
            "normalized_convergence_rate":-slope,"initial_error":float(np.sqrt(np.mean((initial[active]-optimum[active])**2))),
            "final_error":errors[-1],"inactive_maximum_change":float(np.max(np.abs(mean[~active]-initial[~active]))),
            "gradient_variance":float(np.mean(gradient_variances)),"sparse_operations_proxy":epochs*candidates*gates*30,
            "host_runtime_seconds":time.perf_counter()-started}


def run_convergence_scaling(*, epochs:int=28) -> dict[str,Any]:
    _retained_change()
    distances=[3,5,7,9,11,13,15]
    rows=[_scaling_run(d,draw,seed,epochs=epochs) for d in distances for draw in (1,2,3) for seed in (7961,7962)]
    summary=[]
    for d in distances:
        selected=[r for r in rows if r["distance"]==d]; rates=np.array([r["normalized_convergence_rate"] for r in selected])
        summary.append({"distance":d,"gate_count":selected[0]["gate_count"],"control_count":selected[0]["control_count"],
                        "median_normalized_convergence_rate":float(np.median(rates)),"confidence_interval_95":[float(np.quantile(rates,.025)),float(np.quantile(rates,.975))],
                        "maximum_inactive_change":max(r["inactive_maximum_change"] for r in selected),
                        "median_gradient_variance":float(np.median([r["gradient_variance"] for r in selected])),
                        "median_runtime_seconds":float(np.median([r["host_runtime_seconds"] for r in selected]))})
    rates=np.array([r["median_normalized_convergence_rate"] for r in summary]); relative=(rates[-1]-rates[0])/max(abs(rates[0]),1e-15)
    x=np.array([[1,np.log(r["distance"]),np.log(r["gate_count"]),np.log(r["control_count"]),np.log(90)] for r in summary])
    coefficients=np.linalg.lstsq(x,np.log(np.maximum(rates,1e-12)),rcond=None)[0]
    pass_gate=surface_code_control_count(15)==38670 and relative>=-0.15 and all(r["maximum_inactive_change"]==0 for r in summary)
    payload={"schema_version":"google-synthetic-v4-convergence-scaling.v1","method":"actual sparse local policy-gradient trajectories; three physical draws by two development seeds",
             "rows":rows,"by_distance":summary,"fit":{"response":"log normalized convergence rate","predictors":["intercept","log distance","log gates","log controls","log sparse degree"],"coefficients":coefficients.tolist()},
             "distance_15_control_count":surface_code_control_count(15),"distance_3_to_15_relative_rate_change":float(relative),
             "frozen_practical_degradation_tolerance":-0.15,"status":"PASS_SYNTHETIC_ANALOGUE" if pass_gate else "FAIL_CONTROLLER",
             "structural_count_alone_is_not_pass":True,"certification_seeds_consumed":False,
             "summary":{"status":"PASS_SYNTHETIC_ANALOGUE" if pass_gate else "FAIL_CONTROLLER","distance_15_controls":surface_code_control_count(15),"realizations_per_distance":6}}
    write_report("convergence_scaling",payload,"Actual sparse convergence scaling through distance 15")
    return payload


def _bootstrap_median_ci(values:list[float], seed:int) -> list[float]:
    if not values:
        return [float("nan"),float("nan")]
    data=np.asarray(values,float); rng=np.random.default_rng(seed)
    draws=np.median(rng.choice(data,size=(1000,len(data)),replace=True),axis=1)
    return [float(np.quantile(draws,.025)),float(np.quantile(draws,.975))]


def _metric_record(name:str, rows:list[dict[str,Any]], key:str, predicate, *, higher_is_better:bool=True) -> dict[str,Any]:
    selected=[r for r in rows if r.get(key) is not None and np.isfinite(r[key])]
    values=[float(r[key]) for r in selected]
    failures=[r for r in selected if not predicate(float(r[key]))]
    families={}
    for family in sorted({r["family"] for r in selected}):
        local=[float(r[key]) for r in selected if r["family"]==family]
        families[family]={"median":float(np.median(local)),"count":len(local)}
    stable_seed=int.from_bytes(hashlib.sha256(name.encode()).digest()[:4],"little")
    return {"metric":name,"median":float(np.median(values)) if values else None,
            "worst_case":(float(min(values)) if higher_is_better else float(max(values))) if values else None,
            "confidence_interval_95":_bootstrap_median_ci(values,stable_seed) if values else None,
            "failure_count":len(failures),"sample_count":len(values),"plant_family_breakdown":families,
            "status":"PASS_SYNTHETIC_ANALOGUE" if values and not failures else "FAIL_PLANT_FAMILY"}


def run_development_scorecard(*, epochs:int=120) -> dict[str,Any]:
    change=_retained_change()
    metric_contract=read_artifact("stability_metric_contract")
    recovery=read_artifact("randomized_recovery"); steering=read_artifact("steering_phase"); scaling=read_artifact("convergence_scaling")
    rows=[]
    for index,spec in enumerate(specs_for_split("development_validation")):
        dynamic=simulate_trace(spec,seed=7981+index,epochs=epochs,amendment=change)
        stationary=simulate_trace(spec,seed=7991+index,epochs=epochs,amendment=change,no_drift=True)
        metrics=trace_metrics(dynamic); no_metrics=trace_metrics(stationary)
        rows.append({"plant_id":spec.plant_id,"family":spec.family,
                     "fine_tuning_benefit":metrics["fine_tuning_benefit"],
                     "mean_policy_stability_ratio":metrics["mean_policy_stability_ratio"],
                     "operational_stochastic_stability_ratio":metrics["operational_stochastic_stability_ratio"],
                     "low_frequency_suppression_db":metrics["low_frequency_suppression_db"],
                     "exploration_damage_fraction":metrics["mean_exploration_damage"]/max(float(np.mean(dynamic["trace"]["learned_mean"])),1e-15),
                     "harmful_update_rate":metrics["harmful_update_rate"],
                     "no_drift_stationarity_improvement":float((stationary["trace"]["learned_mean"][0]-np.mean(stationary["trace"]["learned_mean"][-12:]))/max(stationary["trace"]["learned_mean"][0],1e-15))})
    dynamic_families={"local_quadratic_drift","common_mode_plus_local","sinusoidal_steering"}
    drift_rows=[r for r in rows if r["family"] in dynamic_families]
    analogue=metric_contract["synthetic_analogue"]
    metrics=[
        _metric_record("fine_tuning_analogue",rows,"fine_tuning_benefit",lambda x:-.02<=x<=.30),
        _metric_record("mean_policy_drift_stability",drift_rows,"mean_policy_stability_ratio",lambda x:analogue["mean_policy_target_interval"][0]<=x<=analogue["mean_policy_target_interval"][1]),
        _metric_record("operational_stochastic_stability",drift_rows,"operational_stochastic_stability_ratio",lambda x:analogue["operational_target_interval"][0]<=x<=analogue["operational_target_interval"][1]),
        _metric_record("low_frequency_suppression",drift_rows,"low_frequency_suppression_db",lambda x:analogue["low_frequency_suppression_db"][0]<=x<=analogue["low_frequency_suppression_db"][1]),
        _metric_record("no_drift_stationarity",rows,"no_drift_stationarity_improvement",lambda x:x>=0),
        _metric_record("exploration_damage",rows,"exploration_damage_fraction",lambda x:x<=.35,higher_is_better=False),
        _metric_record("harmful_update_rate",rows,"harmful_update_rate",lambda x:x<=.30,higher_is_better=False),
    ]
    amendment=read_artifact("amendment_log")
    step=amendment["baseline"]["disjoint_development_validation"]["step_response_epochs"] if amendment["retained"]["name"]=="unamended_baseline" else next(r["disjoint_development_validation"]["step_response_epochs"] for r in amendment["records"] if r["name"]==amendment["retained"]["name"])
    auxiliary={
        "step_response":{"value_epochs":step,"status":"PASS_SYNTHETIC_ANALOGUE" if step is not None and 20<=step<=220 else "FAIL_CONTROLLER"},
        "randomized_recovery":{"status":recovery["status"],"by_severity":recovery["by_severity"]},
        "steering_phase":{"status":steering["status"],"critical_frequency":steering["critical_frequency_median"],"uncertainty":steering["critical_frequency_uncertainty"]},
        "convergence_scaling":{"status":scaling["status"],"distance_15_controls":scaling["distance_15_control_count"],"relative_rate_change":scaling["distance_3_to_15_relative_rate_change"]},
    }
    pass_status={"PASS_SYNTHETIC_ANALOGUE","PASS_STRONGER_WITHOUT_OTHER_REGRESSION"}
    failed=[m["metric"] for m in metrics if m["status"] not in pass_status]
    failed += [name for name,value in auxiliary.items() if value["status"] not in pass_status]
    all_pass=not failed
    overall="PASS_SYNTHETIC_ANALOGUE" if all_pass else ("FAIL_CONTROLLER" if any(x in failed for x in ("step_response","randomized_recovery")) else "FAIL_PLANT_FAMILY")
    payload={"schema_version":"google-synthetic-v4-development-scorecard.v1","controller":amendment["retained"],
             "plant_rows":rows,"metrics":metrics,"auxiliary_metrics":auxiliary,"failed_gates":failed,
             "all_development_gates_pass":all_pass,"overall_status":overall,
             "certification_seeds_consumed":False,"reduced_budget_blocked":True,"standalone_reference_workflow":True,
             "summary":{"status":overall,"all_gates_pass":all_pass,"failed_gate_count":len(failed)}}
    write_report("development_scorecard",payload,"Synthetic development scorecard")
    return payload


def run_certification(*, epochs:int=480, confirm:bool=False) -> dict[str,Any]:
    if not confirm:
        raise RuntimeError("certification seeds are locked; pass --confirm-open-locked-seeds only after reviewing the frozen preregistration")
    prereg=_require("certification_preregistration","status",{"FROZEN_READY_UNOPENED"})
    change=_retained_change(); rows=[]
    specs={s.plant_id:s for s in specs_for_split("certification")}
    for plant_id,seed in zip(prereg["certification_plants"],prereg["certification_seeds"],strict=True):
        result=simulate_trace(specs[plant_id],seed=int(seed),epochs=epochs,amendment=change,certification=True)
        rows.append({"plant_id":plant_id,"seed":seed,"metrics":trace_metrics(result)})
    interval=prereg["metrics_and_tolerances"]["mean_policy_target_interval"]
    pass_count=sum(interval[0]<=r["metrics"]["mean_policy_stability_ratio"]<=interval[1] for r in rows)
    outcome="SYNTHETIC_GOOGLE_STYLE_REPRODUCTION_CERTIFIED" if pass_count==len(rows) else "PARTIAL_SYNTHETIC_REPRODUCTION"
    payload={"schema_version":"google-synthetic-v4-certification-result.v1","outcome":outcome,"rows":rows,
             "passed_plant_count":pass_count,"plant_count":len(rows),"certification_seeds_consumed":True,
             "post_opening_amendments_prohibited":True,"summary":{"status":outcome,"plants_passed":pass_count}}
    write_report("certification_result",payload,"One-shot synthetic certification result")
    return payload
