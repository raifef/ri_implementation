from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from hdfa_rl_suite.google_pure_v7.config import repository_root
from .common import prompt1_report,root,write
from .evidence_contracts import EvidenceGate

EXPECTED=("evidence_gate_contract.json","experiment_family_contract.json","natural_drift/raw_traces.npz","natural_drift/psd_results.npz","natural_drift/summary.json","natural_drift/report.md","natural_drift/figure.png",
 "step_response/raw_traces.npz","step_response/protocol.json","step_response/results.json","step_response/report.md","step_response/figure.png","recovery/raw_traces.npz","recovery/results.json","recovery/report.md","recovery/figure.png",
 "figure5b/raw_trajectories.npz","figure5b/summary.json","figure5b/report.md","figure5b/paper_axes_figure.png","figure5b/normalized_diagnostic.png","figure5c/slopes.json","figure5c/slopes.md","figure5c/report.md","figure5c/figure.png","figure5c/sensitivity.png",
 "paper_claim_registry.json","paper_claim_registry.md","paper_comparison/comparison.json","paper_comparison/report.md","hdfa_baseline_readiness.json","hdfa_baseline_readiness.md","supersession_manifest.json","supersession_manifest.md")
GATED=("natural_drift/summary.json","step_response/results.json","recovery/results.json","figure5b/summary.json","figure5c/slopes.json")

def build_protocol_preflight()->dict[str,Any]:
    base=root();missing=[name for name in EXPECTED if not (base/name).is_file()];invalid=[]
    for name in GATED:
      path=base/name
      if not path.exists():continue
      try:
        gate=json.loads(path.read_text(encoding="utf-8"))["evidence_gate"];EvidenceGate(gate["exact_claim_id"],gate["artifact_complete"],gate["mechanism_valid"],gate["claim_supported"],gate["paper_comparable"],gate["evidence_status"],tuple(gate["blocking_reasons"]))
      except Exception as exc:invalid.append({"artifact":name,"error":str(exc)})
    config_missing=[name for name in ("configs/google_pure_evidence_v8/smoke.json","configs/google_pure_evidence_v8/reference.json") if not (repository_root()/name).is_file()]
    result={"schema_version":"google-pure-evidence-v8-protocol-preflight.v1","expected_files":list(EXPECTED),"missing_files":missing,"invalid_evidence_gates":invalid,"missing_configs":config_missing,
      "protocol_gate_pass":not missing and not invalid and not config_missing,"prompt1_hash":prompt1_report()["artifact_hash"],"prompt1_gate_pass":prompt1_report()["prompt1_gate_pass"],"reference_acquisition_permitted":not missing and not invalid and not config_missing and prompt1_report()["prompt1_gate_pass"]}
    return write("evidence_protocol_preflight",result,"Evidence Protocol Preflight")

def validate_manifests()->dict[str,Any]:
    base=repository_root()/"artifacts/google_pure_paper_reproduction";prompt1=prompt1_report();rows=[]
    for path in sorted((base/"validation").glob("*.json")):
      value=json.loads(path.read_text(encoding="utf-8"));generic=value.get("final_evidence") is True;mode=value.get("mode")
      rows.append({"path":str(path),"old_status":value.get("status"),"old_final_evidence":generic,"v8_status":"INVALID_DIAGNOSTIC" if mode in {"smoke","validation"} else ("CLAIM_NOT_SUPPORTED" if not value.get("valid") else "PAPER_ANCHORED_SYNTHETIC_EVIDENCE"),"prompt1_hash_bound":prompt1["artifact_hash"]})
    superseded=[{"legacy_artifact":"claim_registry.json","superseded_by":"paper_claim_registry.json","reason":"old registry lacked required claim statuses and enforced anti-conflation"},
      {"legacy_artifact":"hdfa_readiness.json","superseded_by":"hdfa_baseline_readiness.json","reason":"old readiness rows were declarative rather than derived"},
      {"legacy_artifact":"step_response/summary.json","superseded_by":"step_response/results.json","reason":"old response normalized crossings by achieved motion"},
      {"legacy_artifact":"recovery/summary.json","superseded_by":"recovery/results.json","reason":"old run did not explicitly spoil a policy"}]
    write("supersession_manifest",{"schema_version":"google-pure-evidence-v8-supersession.v1","rows":superseded,"legacy_artifacts_preserved":True,"legacy_artifacts_valid_for_new_claims":False,"prompt1_hash":prompt1["artifact_hash"]},"Superseded v8 Evidence Artifacts")
    preflight=build_protocol_preflight();result={"schema_version":"google-pure-evidence-v8-manifest-validation.v2","rows":rows,"old_manifests_modified":False,"generic_final_evidence_accepted":False,"prompt1_gate_pass":prompt1["prompt1_gate_pass"],"protocol_preflight_hash":preflight["artifact_hash"],"status":"PASS" if preflight["protocol_gate_pass"] else "FAIL_CLOSED"}
    return write("manifest_validation",result,"Legacy Manifest Evidence Reclassification")
