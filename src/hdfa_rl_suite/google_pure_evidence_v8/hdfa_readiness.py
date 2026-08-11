from __future__ import annotations
import json
from .common import prompt1_report,root,write

def _artifact(path:str)->dict:
    p=root()/path;return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

def _mechanism_status(value:dict,*,analytic:bool=False)->str:
    gate=value.get("evidence_gate",{})
    if not value:return "INVALID_DIAGNOSTIC"
    if not gate.get("mechanism_valid",False):return "INVALID_DIAGNOSTIC"
    if analytic:return "NOT_PUBLICLY_IDENTIFIABLE"
    if gate.get("paper_comparable",False) and gate.get("claim_supported",False):return "READY"
    return "PARTIAL"

def report_hdfa_readiness()->dict:
    p1=prompt1_report();natural=_artifact("natural_drift/summary.json");step=_artifact("step_response/results.json");recovery=_artifact("recovery/results.json");b=_artifact("figure5b/summary.json");c=_artifact("figure5c/slopes.json")
    rows=[{"category":"STATIC_SPARSE_OPTIMIZATION","status":_mechanism_status(b,analytic=True),"basis":"Figure 5b artifact"},
      {"category":"LOCAL_CONVERGENCE_LAW","status":_mechanism_status(c,analytic=True),"basis":"Figure 5c artifact"},
      {"category":"RANDOMIZED_RECOVERY","status":_mechanism_status(recovery),"basis":"spoiled-policy recovery gate"},
      {"category":"REAL_TIME_STEERING","status":"FAILED" if not p1["prompt1_gate_pass"] else "PARTIAL","basis":"Prompt 1 root-cause gate"},
      {"category":"NATURAL_DRIFT_SUPPRESSION","status":_mechanism_status(natural),"basis":"paired PSD gate"},
      {"category":"STEP_RESPONSE","status":_mechanism_status(step),"basis":"target-relative response gate"},
      {"category":"FIGURE5_SCALING","status":"NOT_PUBLICLY_IDENTIFIABLE" if b.get("classification")=="ANALYTIC_SCALING_MODEL" or c.get("classification")=="CONSTRUCTED_ANALYTIC_CONVERGENCE" else "PARTIAL","basis":"scaling provenance"}]
    matching={"identical_plants":False,"identical_seeds":False,"identical_qec_cycle_budgets":False,"identical_fixed_baselines":False,"identical_evaluation_windows":False,"identical_observables":False,"hyperparameters_frozen_before_holdout":False,"no_unresolved_pure_rl_fault":bool(p1["prompt1_gate_pass"])}
    permitted=all(v for v in matching.values()) and all(r["status"]=="READY" for r in rows)
    result={"schema_version":"google-pure-evidence-v8-hdfa-baseline-readiness.v2","rows":rows,"prompt1_hash":p1["artifact_hash"],"prompt1_gate_pass":p1["prompt1_gate_pass"],"matching_checks":matching,"definitive_comparison_permitted":permitted,"outcome":"HDFA_COMPARISON_CAUSALLY_IDENTIFIABLE" if permitted else "HDFA_COMPARISON_NOT_CAUSALLY_IDENTIFIABLE"}
    return write("hdfa_baseline_readiness",result,"HDFA Baseline Readiness",json_name="hdfa_baseline_readiness.json",md_name="hdfa_baseline_readiness.md")
