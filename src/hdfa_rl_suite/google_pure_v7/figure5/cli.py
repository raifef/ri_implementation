"""Console entry points for the isolated Figure 5 pipelines."""
from __future__ import annotations
import argparse,csv,json
from typing import Any
from .common import read_config
from .merge import merge_all,merge_panel
from .plotting import plot_all,plot_panel
from .protocol import freeze_protocols,panel_plan,plan_all
from .reporting import report_all,report_panel,status
from .seed_registry import write_registry
from .source_contract import build_source_contract
from .validation import validate_rows,write_logical_contract

def _emit(value:Any)->None: print(json.dumps(value,indent=2,sort_keys=True,default=str));return None
def _parser(panel:str|None=None)->argparse.ArgumentParser:
    p=argparse.ArgumentParser()
    if panel:p.add_argument("--config",default=f"panel_{panel[-1]}_smoke.yaml")
    p.add_argument("--mode",choices=("smoke","validation","reference","paper-scale"))
    p.add_argument("--dry-run",action="store_true");p.add_argument("--resume",action=argparse.BooleanOptionalAction,default=True)
    p.add_argument("--max-shards",type=int);p.add_argument("--execute-paper-scale",action="store_true")
    p.add_argument("--allow-partial",action="store_true");p.add_argument("--formats",default="png,svg,pdf")
    if panel=="5a":p.add_argument("--frequencies");p.add_argument("--entropy-coefficients")
    if panel in ("5b","5c"):p.add_argument("--distances");p.add_argument("--parameters-per-gate")
    return p
def _config(panel:str,args:argparse.Namespace)->dict:
    cfg=read_config(args.config)
    if args.mode:cfg["mode"]=args.mode
    for name,cast in (("frequencies",float),("entropy_coefficients",float),("distances",int),("parameters_per_gate",int)):
        value=getattr(args,name,None)
        if value:cfg[name]=[cast(x) for x in value.split(",")]
    return cfg
def _rows(panel:str)->tuple[list[dict],str]:
    from .common import figure5_root
    path=figure5_root()/"merged"/panel/"summary.csv";manifest=json.loads((path.parent/"merge_manifest.json").read_text(encoding="utf-8"))
    with path.open(encoding="utf-8",newline="") as stream:raw=list(csv.DictReader(stream))
    rows=[]
    for row in raw:
        converted={}
        for key,value in row.items():
            try:converted[key]=float(value) if any(ch in value.lower() for ch in ".e") else int(value)
            except (ValueError,AttributeError):converted[key]=value
        rows.append(converted)
    return rows,manifest["mode"]
def _panel(panel:str,action:str)->Any:
    args=_parser(panel).parse_args();cfg=_config(panel,args)
    if action=="plan":result=panel_plan(panel,cfg)
    elif action=="acquire":
        module=__import__(f"hdfa_rl_suite.google_pure_v7.figure5.panel_{panel[-1]}",fromlist=["acquire"])
        result=module.acquire(cfg,dry_run=args.dry_run,resume=args.resume,max_shards=args.max_shards,execute_paper_scale=args.execute_paper_scale)
    elif action=="merge":result=merge_panel(panel,cfg,allow_partial=args.allow_partial)
    elif action=="validate":rows,mode=_rows(panel);result=validate_rows(panel,rows,mode=mode)
    elif action=="plot":result=plot_panel(panel,formats=tuple(args.formats.split(",")),partial=args.allow_partial)
    elif action=="report":result=report_panel(panel)
    else:raise ValueError(action)
    if isinstance(result,dict) and "rows" in result:result={k:v for k,v in result.items() if k!="rows"}
    return _emit(result)

def source_contract_main():return _emit({"source":build_source_contract(),"logical":write_logical_contract()})
def freeze_protocols_main():return _emit(freeze_protocols())
def seed_registry_main():return _emit(write_registry())
def plan_all_main():return _emit(plan_all())
def fig5a_plan_main():return _panel("5a","plan")
def fig5a_acquire_main():return _panel("5a","acquire")
def fig5a_merge_main():return _panel("5a","merge")
def fig5a_validate_main():return _panel("5a","validate")
def fig5a_plot_main():return _panel("5a","plot")
def fig5a_report_main():return _panel("5a","report")
def fig5b_plan_main():return _panel("5b","plan")
def fig5b_acquire_main():return _panel("5b","acquire")
def fig5b_merge_main():return _panel("5b","merge")
def fig5b_validate_main():return _panel("5b","validate")
def fig5b_plot_main():return _panel("5b","plot")
def fig5b_report_main():return _panel("5b","report")
def fig5c_plan_main():return _panel("5c","plan")
def fig5c_acquire_main():return _panel("5c","acquire")
def fig5c_merge_main():return _panel("5c","merge")
def fig5c_validate_main():return _panel("5c","validate")
def fig5c_plot_main():return _panel("5c","plot")
def fig5c_report_main():return _panel("5c","report")
def _all_configs(mode:str="smoke")->dict[str,dict]:
    suffix="smoke" if mode=="smoke" else "reference";result={p:read_config(f"panel_{p[-1]}_{suffix}.yaml") for p in ("5a","5b","5c")}
    if mode not in ("smoke","reference"):
        for config in result.values():config["mode"]=mode
    return result
def merge_all_main():args=_parser().parse_args();return _emit(merge_all(_all_configs(args.mode or "smoke"),allow_partial=args.allow_partial))
def validate_all_main():return _emit({p:validate_rows(p,*_rows(p)[0:1],mode=_rows(p)[1]) for p in ("5a","5b","5c")})
def plot_all_main():args=_parser().parse_args();return _emit(plot_all(formats=tuple(args.formats.split(",")),partial=args.allow_partial))
def report_all_main():return _emit(report_all())
def status_main():return _emit(status())
