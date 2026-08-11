from __future__ import annotations

import json
from typing import Any, Callable

from hdfa_rl_suite.google_pure_v7.controller import require_resolved_controller

from .audits import (audit_baselines,audit_clipping_likelihood,audit_entropy_scale,audit_exploration_floor,
                     audit_native_units,audit_ppo_lifecycle,audit_temporal_protocol)
from .contracts import build_mathematical_contracts
from .diagnostics import run_edr_identity_audit,run_figure5a_feasibility
from .matrix import run_compact_fault_matrix
from .reporting import root_cause_report,status
from .snapshot import snapshot


def _compact(value: Any) -> Any:
    if isinstance(value,dict): return {k:(len(v) if k in {"rows","source_hashes","config_hashes","test_hashes"} else _compact(v)) for k,v in value.items()}
    if isinstance(value,list) and len(value)>20:return {"item_count":len(value),"preview":value[:3]}
    if isinstance(value,list):return [_compact(v) for v in value]
    return value


def _run(fn: Callable[[],dict[str,Any]], *, estimate: str="under 10 seconds", cycles: int=0) -> None:
    controller=require_resolved_controller();print(json.dumps({"estimated_runtime":estimate,"estimated_qec_cycles":cycles,
      "controller_hash":controller["resolved_config_hash"],"certification_seed_involved":False},indent=2),flush=True)
    print(json.dumps(_compact(fn()),indent=2,sort_keys=True,default=str))


def snapshot_main():_run(snapshot)
def contracts_main():_run(build_mathematical_contracts)
def edr_main():_run(run_edr_identity_audit,cycles=24*8*2000)
def feasibility_main():_run(run_figure5a_feasibility,cycles=120*24*8000)
def floor_main():_run(audit_exploration_floor,cycles=72*16*4000)
def entropy_main():_run(audit_entropy_scale,cycles=3*48*12*3000)
def units_main():_run(audit_native_units)
def clipping_main():_run(audit_clipping_likelihood)
def ppo_main():_run(audit_ppo_lifecycle)
def baselines_main():_run(audit_baselines)
def temporal_main():_run(audit_temporal_protocol,cycles=9*72*8*2000)
def matrix_main():_run(run_compact_fault_matrix,estimate="approximately 10-30 seconds",cycles=27*72*12*3000)
def report_main():_run(root_cause_report)
def status_main():_run(status)

