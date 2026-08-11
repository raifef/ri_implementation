from __future__ import annotations
from dataclasses import asdict,dataclass
from typing import Any
from .common import prompt1_report,write

STATUSES=("SYNTHETIC_MECHANISM_EVIDENCE","PAPER_ANCHORED_SYNTHETIC_EVIDENCE","PUBLIC_DATA_EXACT_REPRODUCTION","INVALID_DIAGNOSTIC","CLAIM_NOT_SUPPORTED","NOT_PUBLICLY_IDENTIFIABLE")
FINAL_CAPABLE_STATUSES=("PAPER_ANCHORED_SYNTHETIC_EVIDENCE","PUBLIC_DATA_EXACT_REPRODUCTION")

@dataclass(frozen=True)
class EvidenceGate:
    exact_claim_id:str
    artifact_complete:bool
    mechanism_valid:bool
    claim_supported:bool
    paper_comparable:bool
    evidence_status:str
    blocking_reasons:tuple[str,...]=()
    def __post_init__(self):
        if not self.exact_claim_id.strip():raise ValueError("an exact claim ID is required")
        if self.evidence_status not in STATUSES:raise ValueError("unknown evidence status")
        if self.claim_supported and (not self.artifact_complete or not self.mechanism_valid):raise ValueError("claim support requires complete, valid mechanism evidence")
        if self.paper_comparable and not self.claim_supported:raise ValueError("paper comparability requires claim support")
        if self.evidence_status=="INVALID_DIAGNOSTIC" and (self.mechanism_valid or self.claim_supported or self.paper_comparable):raise ValueError("invalid diagnostics cannot be mechanism-valid or claim-bearing")
        if self.evidence_status in {"CLAIM_NOT_SUPPORTED","NOT_PUBLICLY_IDENTIFIABLE"} and (self.claim_supported or self.paper_comparable):raise ValueError("negative evidence statuses cannot support a claim")
        if self.evidence_status=="PUBLIC_DATA_EXACT_REPRODUCTION" and not (self.artifact_complete and self.mechanism_valid and self.claim_supported and self.paper_comparable):raise ValueError("exact public-data reproduction requires all four gates")
        if self.evidence_status=="SYNTHETIC_MECHANISM_EVIDENCE" and self.paper_comparable:raise ValueError("mechanism-only synthetic evidence cannot be paper comparable")
        if self.blocking_reasons and self.paper_comparable:raise ValueError("paper-comparable evidence cannot retain blockers")
    @property
    def final_evidence(self)->bool:
        return (self.evidence_status in FINAL_CAPABLE_STATUSES and self.artifact_complete and self.mechanism_valid
                and self.claim_supported and self.paper_comparable and not self.blocking_reasons)
    def to_dict(self)->dict[str,Any]:return {**asdict(self),"final_evidence":self.final_evidence}

def build_gate_contract()->dict[str,Any]:
    prompt1=prompt1_report()
    result={"schema_version":"google-pure-evidence-v8-gate-contract.v1","independent_statuses":["artifact_complete","mechanism_valid","claim_supported","paper_comparable"],
      "final_evidence_formula":"all four statuses AND exact claim ID","allowed_evidence_statuses":list(STATUSES),
      "final_capable_evidence_statuses":list(FINAL_CAPABLE_STATUSES),
      "prompt1_root_cause_hash":prompt1["artifact_hash"],"prompt1_gate_pass":prompt1["prompt1_gate_pass"],
      "artifact_existence_never_implies_claim_support":True}
    return write("evidence_gate_contract",result,"Evidence Gate Contract")
