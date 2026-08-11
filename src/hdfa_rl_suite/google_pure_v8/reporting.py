from __future__ import annotations

import json
from typing import Any

from hdfa_rl_suite.google_pure_v7.config import canonical_hash
from hdfa_rl_suite.google_pure_v7.controller import require_resolved_controller

from .common import root, write_report


REQUIRED=("pre_repair_snapshot","mathematical_contracts","figure5a_edr_identity_audit","figure5a_feasibility_decomposition",
 "exploration_floor_feasibility","entropy_and_scale_plumbing_audit","native_unit_audit","clipping_and_likelihood_audit",
 "ppo_update_lifecycle_audit","baseline_freezing_audit","temporal_protocol_audit","compact_fault_isolation_matrix")


def _read(name: str) -> dict[str,Any]:
    path=root()/f"{name}.json"
    if not path.exists(): raise RuntimeError(f"missing v8 implementation audit: {name}")
    return json.loads(path.read_text(encoding="utf-8"))


def root_cause_report() -> dict[str,Any]:
    items={name:_read(name) for name in REQUIRED};matrix=items["compact_fault_isolation_matrix"]
    dominant=[x for x in matrix["dominant_classifications"] if x!="NO_IMPLEMENTATION_FAULT_DETECTED"]
    floor=items["exploration_floor_feasibility"];entropy=items["entropy_and_scale_plumbing_audit"]
    if not entropy["operational_entropy_axis"] and "ENTROPY_AXIS_NOT_OPERATIONAL" not in dominant:dominant.append("ENTROPY_AXIS_NOT_OPERATIONAL")
    confirmed=[x for x in dominant if x in {"ENTROPY_AXIS_NOT_OPERATIONAL","NATIVE_UNIT_AMPLIFICATION","ACTION_LIKELIHOOD_MISMATCH","TEMPORAL_ALIASING","MEAN_TRACKING_BANDWIDTH_FAILURE"}]
    rejected=[]
    if items["figure5a_edr_identity_audit"]["classification"]=="PASS": rejected.append("METRIC_OR_ACCOUNTING_FAILURE")
    if items["native_unit_audit"]["classification"]=="PASS": rejected.append("NATIVE_UNIT_AMPLIFICATION")
    if items["baseline_freezing_audit"]["classification"]=="PASS": rejected.append("UNFROZEN_BASELINE_FAILURE")
    controller=require_resolved_controller();all_blockers=list(dict.fromkeys(dominant+([] if floor.get("gate_pass") else ["EXPLORATION_FLOOR_FAILURE"])));gate=not all_blockers
    frozen={"schema_version":"google-pure-v8-repaired-controller-contract.v1","base_controller_hash":controller["resolved_config_hash"],
      "base_controller_code_hash":controller["controller_code_hash"],"accepted_behavior_changes":[],
      "accepted_repairs":["machine-readable EDR normalization and denominator gate","matched finite-shot five-policy feasibility evaluator","recursive immutable audit snapshots","read-only behaviour snapshots in v8 diagnostics","claim-safe failure classification"],
      "source_compatible_controller_change_identified":False,"reason":"public information does not uniquely identify a replacement scale floor or entropy sweep normalization",
      "full_reference_acquisition_permitted":gate,"audit_hashes":{name:value["artifact_hash"] for name,value in items.items()}}
    frozen["contract_hash"]=canonical_hash(frozen);write_report("repaired_controller_contract",frozen,"Frozen v8 Controller/Audit Contract")
    result={"schema_version":"google-pure-v8-root-cause-report.v1","confirmed_implementation_faults":confirmed,
      "confirmed_metric_or_accounting_faults":[] if "METRIC_OR_ACCOUNTING_FAILURE" in rejected else ["UNRESOLVED"],
      "benchmark_impossibility_findings":[floor["classification"]] if not floor.get("gate_pass") else [],
      "remaining_plausible_causes":["SYNTHETIC_PLANT_NON_COMMENSURABILITY"] if "MEAN_TRACKING_BANDWIDTH_FAILURE" in confirmed else [],
      "minimal_accepted_repairs":frozen["accepted_repairs"],"rejected_hypotheses":rejected,
      "full_scale_acquisition_blocked":not gate,"prompt1_gate_pass":gate,"repaired_contract_hash":frozen["contract_hash"],
      "exact_next_commands":["hdfa-google-v8-run-compact-fault-matrix","hdfa-google-v8-report-root-cause"] if not gate else ["hdfa-google-paper-fig5a-acquire --mode reference"],
      "blocking_reasons":all_blockers if not gate else []}
    return write_report("root_cause_report",result,"Pure Google-style RL v8 Root Cause")


def status() -> dict[str,Any]:
    files={name:(root()/f"{name}.json").exists() for name in (*REQUIRED,"root_cause_report","repaired_controller_contract")}
    result={"schema_version":"google-pure-v8-status.v1","artifacts":files,"complete":all(files.values()),"certification_seeds_consumed":False}
    return write_report("status",result,"Pure v8 Status")
