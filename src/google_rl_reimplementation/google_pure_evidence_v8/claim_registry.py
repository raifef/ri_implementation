from __future__ import annotations
import json
from typing import Any
from google_rl_reimplementation.google_pure_v7.config import repository_root
from .common import prompt1_report,root,write
from .experiment_families import ExperimentFamily,RUN_FAMILIES,forbid_joint_score,require_control_only

CLAIM_STATUSES=("PUBLIC_DATA_EXACT_REPRODUCTION","PAPER_ANCHORED_SYNTHETIC_MATCH","QUALITATIVE_MATCH_ONLY","MISMATCH","INVALID_DIAGNOSTIC","NOT_PUBLICLY_IDENTIFIABLE","NOT_YET_RUN")

def _load(relative:str)->dict[str,Any]|None:
    path=root()/relative;return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

def validate_scorecard(rows:list[dict[str,Any]],*,paper_explicitly_simultaneous:bool=False)->None:
    forbid_joint_score([r["experiment_family"] for r in rows],paper_explicitly_simultaneous=paper_explicitly_simultaneous);require_control_only(rows)

def build_claim_registry()->dict:
    specs=[
      ("figure5a.real_time_sampled_candidate_steering",ExperimentFamily.FIGURE5A_REAL_TIME_STEERING,"../google_pure_v8/root_cause_report.json","normalized EDR improvement surface","source panel values","source uncertainty","MISMATCH"),
      ("natural.low_frequency_4db",ExperimentFamily.NATURAL_DRIFT_SPECTRAL_SUPPRESSION,"natural_drift/summary.json","low-frequency suppression in dB","not locally identifiable","not locally identifiable",None),
      ("step.injected_persistent_optimum",ExperimentFamily.STEP_RESPONSE_INJECTED_DRIFT,"step_response/results.json","target-relative step response","approximately 130 epochs","source uncertainty unavailable",None),
      ("recovery.spoiled_policy_90pct",ExperimentFamily.RANDOMIZED_RECOVERY_AFTER_SPOIL,"recovery/results.json","90% recovery from spoiled policy","approximately 1000 epochs","source uncertainty unavailable",None),
      ("figure5b.sparse_scaling",ExperimentFamily.FIGURE5B_SPARSE_SCALING,"figure5b/summary.json","physical error and LER scaling panel","paper panel supplied locally","paper graphical uncertainty",None),
      ("figure5c.convergence_law",ExperimentFamily.FIGURE5C_CONVERGENCE_LAW,"figure5c/slopes.json","distance-independent phase/time convergence rate","paper panel supplied locally","paper graphical uncertainty",None)]
    rows=[]
    for claim,family,path,paper_quantity,paper_value,paper_uncertainty,forced in specs:
      value=_load(path);gate=value.get("evidence_gate",{}) if value else {}
      if forced:status=forced
      elif value is None:status="NOT_YET_RUN"
      elif gate.get("evidence_status")=="INVALID_DIAGNOSTIC":status="INVALID_DIAGNOSTIC"
      elif value.get("classification") in {"ANALYTIC_SCALING_MODEL","CONSTRUCTED_ANALYTIC_CONVERGENCE"}:status="QUALITATIVE_MATCH_ONLY"
      elif value.get("mode")=="smoke":status="QUALITATIVE_MATCH_ONLY"
      elif not gate.get("paper_comparable",False):status="NOT_PUBLICLY_IDENTIFIABLE"
      else:status="PAPER_ANCHORED_SYNTHETIC_MATCH"
      rows.append({"claim_id":claim,"experiment_family":family.value,"run_family":RUN_FAMILIES[family.value],"paper_quantity":paper_quantity,"paper_value":paper_value,"paper_uncertainty":paper_uncertainty,
        "reproduction_quantity":value.get("observable_definition") if value else None,"reproduction_value":value.get("median_mean_suppression_db",value.get("response",value.get("classification"))) if value else None,
        "comparison_legitimacy":bool(gate.get("paper_comparable",False)),"same_run_required":True,"same_run_not_required":False,"cannot_be_jointly_scored_with":[x.value for x in ExperimentFamily if x is not family],
        "decoder_assistance":"CONTROL_ONLY","status":status,"final_evidence":bool(gate.get("final_evidence",False)),"artifact":(root()/path).resolve().relative_to(repository_root()).as_posix()})
    for row in rows:validate_scorecard([row])
    result={"schema_version":"google-pure-evidence-v8-paper-claim-registry.v2","allowed_statuses":list(CLAIM_STATUSES),"rows":rows,"joint_cross_family_score":False,"joint_scorecard_validation":"rejects every multi-family scorecard unless paper_explicitly_simultaneous=True","decoder_and_control_claims_separate":True,"prompt1_hash":prompt1_report()["artifact_hash"]}
    return write("paper_claim_registry",result,"Paper Claim Registry",json_name="paper_claim_registry.json",md_name="paper_claim_registry.md")
