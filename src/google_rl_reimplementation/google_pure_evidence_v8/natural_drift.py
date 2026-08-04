"""Paired natural-drift low-frequency suppression with identifiability diagnostics."""
from __future__ import annotations
import numpy as np
from google_rl_reimplementation.google_pure_v7.config import canonical_hash
from google_rl_reimplementation.google_pure_v7.controller import require_resolved_controller
from google_rl_reimplementation.google_pure_v7.experiments import run_production_trace
from google_rl_reimplementation.google_pure_v7.natural import FAMILIES,FrozenNaturalPlant,generate_natural_drift
from .common import bundle_complete,prompt1_report,require_reference_authorization,root,write
from .evidence_contracts import EvidenceGate
from .experiment_families import ExperimentFamily,family_metadata
from .uncertainty import bootstrap_interval

def _psd(x:np.ndarray,*,segment:int,detrend:str="constant",window_name:str="hann")->tuple[np.ndarray,np.ndarray,int]:
    x=np.asarray(x,float);step=segment//2;window=np.hanning(segment) if window_name=="hann" else np.ones(segment);rows=[];t=np.arange(segment)
    for start in range(0,len(x)-segment+1,step):
      y=x[start:start+segment].copy()
      if detrend=="constant":y-=np.mean(y)
      elif detrend=="linear":y-=np.polyval(np.polyfit(t,y,1),t)
      else:raise ValueError("unknown detrending method")
      p=np.abs(np.fft.rfft(y*window))**2/np.sum(window**2);p[1:-1]*=2;rows.append(p)
    if not rows:raise ValueError("duration shorter than one PSD segment")
    return np.fft.rfftfreq(segment),np.mean(rows,axis=0),len(rows)

def _band_power(freq:np.ndarray,power:np.ndarray,band:tuple[float,float])->float:
    selected=(freq>=band[0])&(freq<band[1]);df=float(freq[1]-freq[0]);return float(np.sum(power[selected])*df)

def run_natural_drift(*,mode:str="smoke",execute:bool=False)->dict:
    require_reference_authorization(mode,execute);horizon,indices,candidates,cycles,segment=(256,range(2),12,3000,128) if mode=="smoke" else (2048,range(6),40,100000,512)
    traces={};psds={};rows=[];low=(.001,.012);sensitivity=[]
    for index in indices:
      family=FAMILIES[index];plant=FrozenNaturalPlant(index,v5_compatible_units=False);tape=generate_natural_drift(family,horizon);result=run_production_trace(plant,tape,seed=15110+index,candidates=candidates,cycles=cycles);key=family["id"]
      traces[key]={name:np.asarray(value) for name,value in result["logical_risk"].items()};powers={};segments=0
      for name,values in traces[key].items():freq,power,segments=_psd(values,segment=segment);powers[name]=power
      fixed=_band_power(freq,powers["fixed_policy"],low);mean=_band_power(freq,powers["learned_mean"],low);candidate=_band_power(freq,powers["stochastic_candidates"],low);band_modes=int(np.sum((freq>=low[0])&(freq<low[1])))
      row={"plant_id":key,"seed":15110+index,"drift_tape_hash":canonical_hash(tape.tolist()),"fixed_low_power":fixed,"mean_low_power":mean,"candidate_low_power":candidate,
        "mean_suppression_db":float(10*np.log10(max(fixed,1e-30)/max(mean,1e-30))),"candidate_suppression_db":float(10*np.log10(max(fixed,1e-30)/max(candidate,1e-30))),
        "independent_low_frequency_modes":band_modes,"welch_segments":segments,"frequency_resolution":float(freq[1]-freq[0]),"low_frequency_band":list(low),"duration":horizon,"burn_in":0,"detrending":"segment constant","window":"Hann","PSD_estimator":f"Welch {segment}, 50% overlap","sampling_cadence":"one sample per epoch"};rows.append(row);psds[key]=(freq,powers)
      for detrend in ("constant","linear"):
        for window in ("hann","boxcar"):
          f,p,_=_psd(traces[key]["learned_mean"],segment=segment,detrend=detrend,window_name=window)
          for band in (low,(.002,.01),(.001,.02)):
            sensitivity.append({"plant_id":key,"detrending":detrend,"window":window,"band":list(band),"mean_low_power":_band_power(f,p,band)})
    target=root()/"natural_drift";target.mkdir(parents=True,exist_ok=True);np.savez_compressed(target/"raw_traces.npz",**{f"{k}__{n}":v for k,d in traces.items() for n,v in d.items()});np.savez_compressed(target/"psd_results.npz",**{f"{k}__frequency":v[0] for k,v in psds.items()},**{f"{k}__{n}":p for k,v in psds.items() for n,p in v[1].items()})
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True);first=next(iter(traces));axes[0,0].plot(traces[first]["fixed_policy"],label="fixed");axes[0,0].plot(traces[first]["learned_mean"],label="mean");axes[0,0].plot(traces[first]["stochastic_candidates"],alpha=.7,label="candidate");axes[0,0].plot(traces[first]["oracle_optimum"],label="oracle");axes[0,0].legend();axes[0,0].set_title("paired traces")
    f,p=psds[first];axes[0,1].loglog(f[1:],p["fixed_policy"][1:],label="fixed");axes[0,1].loglog(f[1:],p["learned_mean"][1:],label="mean");axes[0,1].axvspan(*low,alpha=.2);axes[0,1].legend();axes[0,1].set_title("paired PSD and LF band")
    mean_db=[r["mean_suppression_db"] for r in rows];candidate_db=[r["candidate_suppression_db"] for r in rows];axes[1,0].bar(np.arange(len(rows))-.18,mean_db,.36,label="mean");axes[1,0].bar(np.arange(len(rows))+.18,candidate_db,.36,label="candidate");axes[1,0].axhline(0,color="k",lw=.6);axes[1,0].legend();axes[1,0].set_title("per-run positive suppression (dB)")
    axes[1,1].boxplot([mean_db,candidate_db],tick_labels=["mean","candidate"]);axes[1,1].axhline(0,color="k",lw=.6);axes[1,1].set_title("complete-run uncertainty units");fig.savefig(target/"figure.png",dpi=180);plt.close(fig)
    identifiable=mode=="reference" and min(r["independent_low_frequency_modes"] for r in rows)>=4 and len(rows)>=4;mechanism=identifiable
    result={"schema_version":"google-pure-evidence-v8-natural.v2",**family_metadata(ExperimentFamily.NATURAL_DRIFT_SPECTRAL_SUPPRESSION),"mode":mode,"controller_hash":require_resolved_controller()["resolved_config_hash"],
      "protocol_hash":canonical_hash({"horizon":horizon,"indices":list(indices),"candidates":candidates,"cycles":cycles,"segment":segment,"band":low}),"plant_hash":"frozen-natural-v7","graph_hash":"paired-local-v7","seed_registry_hash":canonical_hash([15110+i for i in indices]),
      "observable_definition":"10log10(integrated fixed LF PSD / integrated policy LF PSD), positive is suppression","evaluation_budget":{"epochs":horizon,"candidates":candidates,"cycles_per_candidate":cycles},"rows":rows,"sensitivity_records":sensitivity,
      "median_mean_suppression_db":float(np.median(mean_db)),"median_candidate_suppression_db":float(np.median(candidate_db)),"mean_suppression_ci_95":bootstrap_interval(mean_db),"candidate_suppression_ci_95":bootstrap_interval(candidate_db),
      "complete_plant_seed_runs_are_resampling_units":True,"low_frequency_identifiable":identifiable,"old_plot_reclassification":"DESCRIPTIVE_LEARNED_MEAN_TRACES_ONLY","prompt1_hash":prompt1_report()["artifact_hash"]}
    complete,missing=bundle_complete(target,["raw_traces.npz","psd_results.npz","figure.png"],result,["rows","sensitivity_records","prompt1_hash"]);blockers=[]
    if not identifiable:blockers.append("INSUFFICIENT_DURATION_FOR_LOW_FREQUENCY_CLAIM")
    if not prompt1_report()["prompt1_gate_pass"]:blockers.append("PROMPT1_GATE_NOT_PASSED")
    status="PAPER_ANCHORED_SYNTHETIC_EVIDENCE" if identifiable else "INVALID_DIAGNOSTIC";gate=EvidenceGate("natural.low_frequency_4db",complete,mechanism,False,False,status,tuple(blockers+missing));result["evidence_gate"]=gate.to_dict();result["blocking_reasons"]=list(gate.blocking_reasons)
    return write("summary",result,"Natural-drift Spectral Suppression",directory=target,json_name="summary.json",md_name="report.md")
