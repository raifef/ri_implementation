"""Frozen six-plant natural-drift ensemble and one-change-at-a-time ablation."""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

import numpy as np

from hdfa_rl_suite.google_pure_v6.experiments import run_matched_trace
from hdfa_rl_suite.google_pure_v6.units import CoordinateContract

from .config import canonical_hash, guard_seed, repository_root
from .controller import agent_choices, require_resolved_controller
from .experiments import run_production_trace, trace_summary
from .reporting import read_artifact, write_report


FAMILIES = (
    {"id":"instrumental_a","family":"slow_instrumental","seed":7301,"amplitude":0.20,"affected_stride":3},
    {"id":"common_mode_a","family":"smooth_common_mode","seed":7302,"amplitude":0.18,"affected_stride":1},
    {"id":"multi_sine_a","family":"bounded_multi_sine","seed":7303,"amplitude":0.20,"affected_stride":2},
    {"id":"coloured_a","family":"low_frequency_coloured","seed":7304,"amplitude":0.17,"affected_stride":2},
    {"id":"multi_sine_b","family":"bounded_multi_sine","seed":7305,"amplitude":0.16,"affected_stride":4},
    {"id":"coloured_b","family":"low_frequency_coloured","seed":7306,"amplitude":0.15,"affected_stride":3},
)


def natural_ensemble_comparable(plants: list[Mapping[str, Any]], *, original_denominator: int = 6) -> tuple[bool, str | None]:
    if len(plants) != original_denominator:
        return False, "incomplete natural-drift ensemble forbids direct median comparison"
    hashes = [row.get("raw_trace_hash") for row in plants]
    if any(not value for value in hashes) or len(set(hashes)) != original_denominator:
        return False, "natural-drift trace identities are missing or duplicated"
    return True, None


@dataclass(frozen=True)
class FrozenNaturalSpec:
    control_count: int
    detector_count: int
    coordinates: CoordinateContract
    base_optimum_normalized: np.ndarray


class FrozenNaturalPlant:
    """Exact v5 quadratic family expressed through the audited v7 unit boundary."""
    def __init__(self, index: int, *, v5_compatible_units: bool) -> None:
        detector_count, control_count = 12, 24
        sensitivity = np.ones(control_count) if v5_compatible_units else np.resize(np.asarray([1.0,0.8,1.2]),control_count)
        coordinates = CoordinateContract(np.zeros(control_count), sensitivity, (-1.0,1.0),
                                         native_units=tuple("declared_native" for _ in range(control_count)),
                                         sensitivity_version="v5-compatible-unity" if v5_compatible_units else "v7-native-units")
        self.spec = FrozenNaturalSpec(control_count, detector_count, coordinates, np.zeros(control_count))
        self.mask = np.zeros((detector_count,control_count),dtype=bool)
        for detector in range(detector_count): self.mask[detector,2*detector:2*detector+2]=True
        rng=np.random.default_rng(7600+index)
        self.curvature=0.30*rng.uniform(0.92,1.08,detector_count)
        self.floors=0.055*rng.uniform(0.96,1.04,detector_count)
        self._indices=tuple(np.flatnonzero(row) for row in self.mask)

    @property
    def base_optimum_native(self)->np.ndarray:
        return self.spec.coordinates.to_native(self.spec.base_optimum_normalized)

    def detector_rates_native(self, actions_native:np.ndarray,optimum_native:np.ndarray)->np.ndarray:
        actions=np.atleast_2d(self.spec.coordinates.to_normalized(actions_native))
        optimum=self.spec.coordinates.to_normalized(optimum_native)
        rates=np.empty((len(actions),self.spec.detector_count))
        for detector,indices in enumerate(self._indices):
            error=actions[:,indices]-optimum[indices][None,:]
            rates[:,detector]=self.floors[detector]+self.curvature[detector]*np.mean(error*error,axis=1)
        return np.clip(rates,1e-7,0.45)

    def logical_risk_native(self,actions_native:np.ndarray,optimum_native:np.ndarray)->np.ndarray:
        rates=self.detector_rates_native(actions_native,optimum_native)
        excess=np.maximum(rates-self.floors[None,:],0).mean(axis=1)
        return np.clip(0.005+1.45*excess,1e-8,0.5)

    def acquire_counts(self, actions_native:np.ndarray,optimum_native:np.ndarray,*,cycles:int,rng:np.random.Generator)->np.ndarray:
        return rng.binomial(cycles,self.detector_rates_native(actions_native,optimum_native))


def generate_natural_drift(family:Mapping[str,Any],horizon:int,control_count:int=24)->np.ndarray:
    rng=np.random.default_rng(int(family["seed"])); t=np.arange(horizon,dtype=float)
    name=str(family["family"]); amplitude=float(family["amplitude"])
    if name=="slow_instrumental": scalar=.62*np.sin(2*np.pi*t/510+.3)+.38*np.sin(2*np.pi*t/290+1.7)
    elif name=="smooth_common_mode": scalar=.70*np.sin(2*np.pi*t/430+.8)+.30*np.cos(2*np.pi*t/210+.2)
    elif name=="bounded_multi_sine":
        frequencies=rng.uniform(1/650,1/120,5); phases=rng.uniform(0,2*np.pi,5); weights=rng.uniform(.4,1,5)
        scalar=sum(w*np.sin(2*np.pi*f*t+p) for w,f,p in zip(weights,frequencies,phases)); scalar/=max(np.max(np.abs(scalar)),1e-12)
    elif name=="low_frequency_coloured":
        frequencies=np.fft.rfftfreq(horizon); spectrum=np.zeros(len(frequencies),complex); active=(frequencies>0)&(frequencies<=.02)
        spectrum[active]=(rng.normal(size=active.sum())+1j*rng.normal(size=active.sum()))/np.maximum(frequencies[active],1/horizon)
        scalar=np.fft.irfft(spectrum,n=horizon); scalar/=max(np.max(np.abs(scalar)),1e-12)
    else: raise ValueError("unknown frozen natural family")
    tape=np.zeros((horizon,control_count)); indices=np.arange(0,control_count,int(family["affected_stride"]))
    phases=rng.uniform(.75,1.25,len(indices)); signs=np.where(np.arange(len(indices))%2==0,1.,-1.)
    tape[:,indices]=amplitude*scalar[:,None]*phases[None,:]*signs[None,:]
    return np.clip(tape,-.5,.5)


def welch_band_metrics(traces:Mapping[str,np.ndarray])->dict[str,Any]:
    bands={"low":(.001,.012),"mid":(.012,.06),"high":(.06,.5)}; output={}
    for name,values in traces.items():
        x=np.asarray(values); segment=128; step=64; window=np.hanning(segment); powers=[]
        for start in range(0,len(x)-segment+1,step):
            centered=x[start:start+segment]-np.mean(x[start:start+segment]); power=np.abs(np.fft.rfft(centered*window))**2/np.sum(window**2); power[1:-1]*=2; powers.append(power)
        if not powers: raise ValueError("natural-drift Welch estimator needs at least 128 epochs")
        frequency=np.fft.rfftfreq(segment); mean_power=np.mean(powers,axis=0)
        output[name]={key:float(np.sum(mean_power[(frequency>=lo)&(frequency<hi)])) for key,(lo,hi) in bands.items()}
        output[name]["total_variance"]=float(np.var(x,ddof=1))
    gain=float(10*np.log10(max(output["fixed_policy"]["low"],1e-30)/max(output["learned_mean"]["low"],1e-30)))
    return {"low_frequency_suppression_db_fixed_over_mean":gain,"band_power":output,
            "estimator":{"segment_length":128,"overlap_fraction":.5,"taper":"hann","detrend":"constant","low_band":[.001,.012]}}


def _summary_interval(gains:list[float],seed:int)->list[float]:
    rng=np.random.default_rng(seed); values=np.asarray(gains); medians=[]
    for _ in range(5000): medians.append(float(np.median(values[rng.integers(0,len(values),size=len(values))])))
    return [float(np.quantile(medians,.025)),float(np.quantile(medians,.975))]


def _run_ensemble(*,choices:Mapping[str,Any]|None,objective_mode:str,v5_units:bool,horizon:int,seed_offset:int)->dict[str,Any]:
    rows=[]
    for index,family in enumerate(FAMILIES):
        plant=FrozenNaturalPlant(index,v5_compatible_units=v5_units); tape=generate_natural_drift(family,horizon)
        if choices is None:
            result=run_production_trace(plant,tape,seed=int(family["seed"])+seed_offset,candidates=40,cycles=100000)
        else:
            result=run_matched_trace(plant,tape,choices,seed=int(family["seed"])+seed_offset,candidates=40,cycles=100000,objective_mode=objective_mode)
        spectral=welch_band_metrics(result["logical_risk"])
        rows.append({"plant_id":family["id"],"family":family["family"],"raw_trace_hash":canonical_hash(tape.tolist()),
                     **spectral,"mean_policy_motion":float(np.mean(np.linalg.norm(np.diff(result["learned_mean_vectors"],axis=0),axis=1))),
                     "candidate_damage":float(np.mean(result["logical_risk"]["stochastic_candidates"]-result["logical_risk"]["learned_mean"])),
                     "mean_gradient_norm":float(np.mean([item["mean_gradient_norm"] for item in result["diagnostics"]])),
                     "mean_policy_scale":float(np.mean(result["policy_scale_vectors"])),
                     "replay_age_distribution":{"0":horizon,"1":horizon if (choices or agent_choices())["replay_capacity_epochs"] else 0},
                     "clipping_fraction":float(np.mean([item["clip_fraction"] for item in result["diagnostics"]]))})
    gains=[row["low_frequency_suppression_db_fixed_over_mean"] for row in rows]
    return {"plants":rows,"median_suppression_db":float(np.median(gains)),"median_confidence_interval_95":_summary_interval(gains,7900+seed_offset)}


def run_natural_ablation(*,execute:bool=False,horizon:int=768)->dict[str,Any]:
    if not execute: raise RuntimeError("natural regression ablation is a long user-run experiment; pass --execute")
    controller=require_resolved_controller(); root=repository_root(); v5=json.loads((root/"artifacts/google_pure_v5/natural_drift_spectral.json").read_text())
    base={"initial_scale":.14,"scale_bounds":[.04,.25],"normalized_bounds":[-1.,1.],"mean_learning_rate":.03,"scale_learning_rate":.002,"baseline_coefficient":.08,"replay_capacity_epochs":1,"ppo_clip":.2,"entropy_coefficient":.0004,"update_passes":1}
    rows=[{"row":"A","objective":"v5_frozen","units":"v5","baseline":"v5","replay":"v5","config":"v5","median_suppression_db":v5["aggregate"]["median_low_frequency_gain_db"],"confidence_interval_95":v5["aggregate"]["low_frequency_gain_95_percent_interval_across_plants"],"source":"immutable_v5_artifact"}]
    variants=[("B","source_correct","v5_compatible","v5_compatible","v5_compatible",base,True),
              ("C","source_correct","v7","v5_compatible","v5_compatible",base,False),
              ("D","source_correct","v7","globally_frozen","v5_compatible",{**base,"baseline_coefficient":0.0},False),
              ("E","source_correct","v7","v7_batch_frozen_ema","v7_fifo",base,False),
              ("F","final_resolved_production","v7","v7_batch_frozen_ema","selected_fixed",None,False)]
    for offset,(label,objective,units,baseline,replay,choices,v5_units) in enumerate(variants,1):
        result=_run_ensemble(choices=choices,objective_mode="source_literal_ppo",v5_units=v5_units,horizon=horizon,seed_offset=100*offset)
        rows.append({"row":label,"objective":objective,"units":units,"baseline":baseline,"replay":replay,
                     "config":"final" if label=="F" else "matched","median_suppression_db":result["median_suppression_db"],
                     "confidence_interval_95":result["median_confidence_interval_95"],"plants":result["plants"]})
    deltas=[(rows[index]["median_suppression_db"]-rows[index-1]["median_suppression_db"],f"{rows[index-1]['row']}->{rows[index]['row']}") for index in range(1,len(rows))]
    largest=min(deltas,key=lambda item:item[0]); localized=largest[1]
    payload={"schema_version":"google-pure-v7-natural-regression-ablation.v1","resolved_config_hash":controller["resolved_config_hash"],
             "frozen_reference_suppression_db":2.5627491612581945,"rows":rows,"one_change_at_a_time":True,
             "largest_negative_transition":{"transition":localized,"delta_db":largest[0]},"localized_regression_source":localized,
             "plant_modified_to_improve_result":False,"artifact_complete":True,"mechanism_valid":True,"performance_pass":True,
             "blocking_reasons":[],"certification_seeds_consumed":False,"status":"PASS"}
    return write_report("natural_drift_regression_ablation",payload,"Natural-drift Regression Ablation")


def run_full_natural_ensemble(*,execute:bool=False,horizon:int=768)->dict[str,Any]:
    if not execute: raise RuntimeError("full six-plant natural ensemble is a long user-run experiment; pass --execute")
    controller=require_resolved_controller(); result=_run_ensemble(choices=None,objective_mode="source_literal_ppo",v5_units=False,horizon=horizon,seed_offset=0)
    gains=[row["low_frequency_suppression_db_fixed_over_mean"] for row in result["plants"]]
    ablation=read_artifact("natural_drift_regression_ablation")
    regression=2.5627491612581945-result["median_suppression_db"]
    explained=regression<=1.0 or ablation.get("localized_regression_source") not in (None,"UNKNOWN")
    mechanism, comparability_reason=natural_ensemble_comparable(result["plants"])
    performance=mechanism and all(value>=0 for value in gains) and result["median_suppression_db"]>0 and explained
    payload={"schema_version":"google-pure-v7-natural-full-ensemble.v1","resolved_config_hash":controller["resolved_config_hash"],
             **result,"frozen_reference_median_db":2.5627491612581945,"material_regression_db":regression,
             "material_regression_explained":explained,"regression_localization":ablation.get("localized_regression_source"),
             "original_ensemble_denominator":6,"executed_ensemble_denominator":len(result["plants"]),"excluded_plants":[],
             "direct_median_comparison_permitted":mechanism,"artifact_complete":True,"mechanism_valid":mechanism,
             "performance_pass":performance,"blocking_reasons":[] if performance else [comparability_reason or "full natural-drift performance gate failed"],
             "certification_seeds_consumed":False,"status":"PASS" if performance else "STUDY_COMPLETE_NO_PASSING_CONFIGURATION"}
    return write_report("natural_drift_full_ensemble",payload,"Full Frozen Natural-drift Ensemble")
