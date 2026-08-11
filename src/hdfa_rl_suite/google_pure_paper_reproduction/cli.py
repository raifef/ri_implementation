"""Console entry points for the full-paper pure baseline workflow."""
from __future__ import annotations

import argparse
import json
from typing import Any

from .claim_registry import build_claim_registry
from .experiment_families import ExperimentFamily
from .paper_figures import acquire, build_protocol, merge_protocol, plot_protocol
from .paper_tables import build_values_table
from .public_data import reproduce_public_data
from .reporting import audit_all, baseline_readiness, next_user_commands, reproduction_overview, status
from .side_by_side import compare_all, compare_panel
from .source_registry import build_source_contract
from .validation import validate_protocol


def _console_view(value: Any, *, key: str | None = None) -> Any:
    """Remove bulk trajectories/conditions while retaining an auditable CLI summary."""
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for child_key, child_value in value.items():
            if child_key in {"rows", "conditions"} and isinstance(child_value, list):
                output[f"{child_key[:-1]}_count"] = len(child_value)
                continue
            if child_key == "trajectory" and isinstance(child_value, dict):
                output["trajectory_fields"] = sorted(child_value)
                continue
            output[child_key] = _console_view(child_value, key=child_key)
        return output
    if isinstance(value, list):
        if len(value) > 24:
            return {"item_count": len(value), "preview": [_console_view(item) for item in value[:3]], "truncated_for_console": True}
        return [_console_view(item) for item in value]
    return value


def _print(value: Any) -> None:
    print(json.dumps(_console_view(value), indent=2, sort_keys=True, default=str))


def _family_main(family: str, action: str, panel: str | None = None) -> None:
    parser=argparse.ArgumentParser(prog=f"hdfa-google-paper-{(panel or family.lower()).replace('_','-')}-{action}")
    parser.add_argument("--mode",choices=("smoke","validation","reference","paper-scale"),default="smoke")
    parser.add_argument("--config")
    if action=="acquire":
        parser.add_argument("--max-shards",type=int);parser.add_argument("--execute-paper-scale",action="store_true")
        parser.add_argument("--worker-count",type=int,default=1,
                            help="number of deterministic condition partitions")
        parser.add_argument("--worker-index",type=int,default=0,
                            help="zero-based partition executed by this process")
    if action=="merge": parser.add_argument("--allow-partial",action="store_true")
    if action=="compare": parser.add_argument("--paper-image")
    args=parser.parse_args();protocol=build_protocol(family,mode=args.mode,config_path=args.config)
    if action=="plan": result=protocol
    elif action=="acquire": result=acquire(protocol,max_shards=args.max_shards,
        execute_paper_scale=args.execute_paper_scale,
        worker_index=args.worker_index,worker_count=args.worker_count)
    elif action=="merge": result=merge_protocol(protocol,allow_partial=args.allow_partial)
    elif action=="validate": result=validate_protocol(protocol)
    elif action=="plot": result=plot_protocol(protocol)
    elif action=="compare":
        if panel is None: raise RuntimeError("compare is only valid for paper figure panels")
        result=compare_panel(panel,protocol,paper_image=args.paper_image)
    else: raise ValueError(action)
    _print(result)


def _make(family: str, action: str, panel: str | None = None):
    return lambda: _family_main(family,action,panel)


fig5a_plan_main=_make(ExperimentFamily.FIGURE5A_REAL_TIME_STEERING.value,"plan","figure5a")
fig5a_acquire_main=_make(ExperimentFamily.FIGURE5A_REAL_TIME_STEERING.value,"acquire","figure5a")
fig5a_merge_main=_make(ExperimentFamily.FIGURE5A_REAL_TIME_STEERING.value,"merge","figure5a")
fig5a_validate_main=_make(ExperimentFamily.FIGURE5A_REAL_TIME_STEERING.value,"validate","figure5a")
fig5a_plot_main=_make(ExperimentFamily.FIGURE5A_REAL_TIME_STEERING.value,"plot","figure5a")
fig5a_compare_main=_make(ExperimentFamily.FIGURE5A_REAL_TIME_STEERING.value,"compare","figure5a")
fig5b_plan_main=_make(ExperimentFamily.FIGURE5B_SPARSE_SCALING.value,"plan","figure5b")
fig5b_acquire_main=_make(ExperimentFamily.FIGURE5B_SPARSE_SCALING.value,"acquire","figure5b")
fig5b_merge_main=_make(ExperimentFamily.FIGURE5B_SPARSE_SCALING.value,"merge","figure5b")
fig5b_validate_main=_make(ExperimentFamily.FIGURE5B_SPARSE_SCALING.value,"validate","figure5b")
fig5b_plot_main=_make(ExperimentFamily.FIGURE5B_SPARSE_SCALING.value,"plot","figure5b")
fig5b_compare_main=_make(ExperimentFamily.FIGURE5B_SPARSE_SCALING.value,"compare","figure5b")
fig5c_plan_main=_make(ExperimentFamily.FIGURE5C_CONVERGENCE_LAW.value,"plan","figure5c")
fig5c_acquire_main=_make(ExperimentFamily.FIGURE5C_CONVERGENCE_LAW.value,"acquire","figure5c")
fig5c_merge_main=_make(ExperimentFamily.FIGURE5C_CONVERGENCE_LAW.value,"merge","figure5c")
fig5c_validate_main=_make(ExperimentFamily.FIGURE5C_CONVERGENCE_LAW.value,"validate","figure5c")
fig5c_plot_main=_make(ExperimentFamily.FIGURE5C_CONVERGENCE_LAW.value,"plot","figure5c")
fig5c_compare_main=_make(ExperimentFamily.FIGURE5C_CONVERGENCE_LAW.value,"compare","figure5c")

natural_plan_main=_make(ExperimentFamily.NATURAL_DRIFT_SPECTRAL_SUPPRESSION.value,"plan")
natural_acquire_main=_make(ExperimentFamily.NATURAL_DRIFT_SPECTRAL_SUPPRESSION.value,"acquire")
natural_merge_main=_make(ExperimentFamily.NATURAL_DRIFT_SPECTRAL_SUPPRESSION.value,"merge")
natural_validate_main=_make(ExperimentFamily.NATURAL_DRIFT_SPECTRAL_SUPPRESSION.value,"validate")
natural_plot_main=_make(ExperimentFamily.NATURAL_DRIFT_SPECTRAL_SUPPRESSION.value,"plot")

recovery_plan_main=_make(ExperimentFamily.RANDOMIZED_RECOVERY_AFTER_SPOIL.value,"plan")
recovery_acquire_main=_make(ExperimentFamily.RANDOMIZED_RECOVERY_AFTER_SPOIL.value,"acquire")
recovery_merge_main=_make(ExperimentFamily.RANDOMIZED_RECOVERY_AFTER_SPOIL.value,"merge")
recovery_validate_main=_make(ExperimentFamily.RANDOMIZED_RECOVERY_AFTER_SPOIL.value,"validate")
recovery_plot_main=_make(ExperimentFamily.RANDOMIZED_RECOVERY_AFTER_SPOIL.value,"plot")

step_plan_main=_make(ExperimentFamily.STEP_RESPONSE_INJECTED_DRIFT.value,"plan")
step_acquire_main=_make(ExperimentFamily.STEP_RESPONSE_INJECTED_DRIFT.value,"acquire")
step_merge_main=_make(ExperimentFamily.STEP_RESPONSE_INJECTED_DRIFT.value,"merge")
step_validate_main=_make(ExperimentFamily.STEP_RESPONSE_INJECTED_DRIFT.value,"validate")
step_plot_main=_make(ExperimentFamily.STEP_RESPONSE_INJECTED_DRIFT.value,"plot")


def _nonfigure_compare(family: str) -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--mode",choices=("smoke","validation","reference","paper-scale"),default="smoke");parser.add_argument("--config");args=parser.parse_args()
    protocol=build_protocol(family,mode=args.mode,config_path=args.config);validation=validate_protocol(protocol)
    _print({"experiment_family":family,"mode":args.mode,"numeric_checks":__import__("hdfa_rl_suite.google_pure_paper_reproduction.comparison_metrics",fromlist=["family_checks"]).family_checks(family,validation),"verdict":"SMOKE_RENDER_ONLY" if args.mode=="smoke" else "QUALITATIVE_MATCH_ONLY"})


natural_compare_main=lambda:_nonfigure_compare(ExperimentFamily.NATURAL_DRIFT_SPECTRAL_SUPPRESSION.value)
recovery_compare_main=lambda:_nonfigure_compare(ExperimentFamily.RANDOMIZED_RECOVERY_AFTER_SPOIL.value)
step_compare_main=lambda:_nonfigure_compare(ExperimentFamily.STEP_RESPONSE_INJECTED_DRIFT.value)


def claim_registry_main() -> None: _print(build_claim_registry())
def source_contract_main() -> None: _print(build_source_contract())
def status_main() -> None: _print(status())
def audit_all_main() -> None: _print(audit_all())


def public_data_main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--recompute",action="store_true");args=parser.parse_args();_print(reproduce_public_data(recompute=args.recompute))


def side_by_side_all_main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--mode",choices=("smoke","validation","reference","paper-scale"),default="smoke");args=parser.parse_args()
    protocols={"figure5a":build_protocol(ExperimentFamily.FIGURE5A_REAL_TIME_STEERING.value,mode=args.mode),"figure5b":build_protocol(ExperimentFamily.FIGURE5B_SPARSE_SCALING.value,mode=args.mode),"figure5c":build_protocol(ExperimentFamily.FIGURE5C_CONVERGENCE_LAW.value,mode=args.mode)};_print(compare_all(protocols))


def values_table_main() -> None: _print(build_values_table())
def readiness_main() -> None: reproduction_overview();next_user_commands();_print(baseline_readiness())
