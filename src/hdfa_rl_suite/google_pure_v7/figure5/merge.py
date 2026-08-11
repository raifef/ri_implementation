"""Deterministic shard merge with explicit completeness and tidy exports."""
from __future__ import annotations
import csv, json
from pathlib import Path
from typing import Any
import numpy as np
from hdfa_rl_suite.google_pure_v7.config import canonical_hash, sha256_file
from .common import atomic_json, atomic_text, figure5_root
from .protocol import panel_plan
from .storage import discover_shards

def _module(panel:str):
    if panel=="5a": from . import panel_a as module
    elif panel=="5b": from . import panel_b as module
    elif panel=="5c": from . import panel_c as module
    else: raise ValueError(panel)
    return module

def merge_panel(panel:str,config:dict[str,Any],*,allow_partial=False)->dict:
    plan=panel_plan(panel,config); discovered=discover_shards(panel); selected=[]
    for record,data in discovered:
        if record["identity"]["protocol_hash"]==plan["protocol_hash"]:selected.append((record,data))
    rows=[_module(panel).shard_row(r,d) for r,d in selected]
    expected=plan["condition_count"]; complete=len(rows)==expected
    if not complete and not allow_partial: raise RuntimeError(f"panel {panel} incomplete: {len(rows)}/{expected} shards")
    root=figure5_root()/"merged"/panel; root.mkdir(parents=True,exist_ok=True)
    fields=sorted({key for row in rows for key in row})
    csv_path=root/"summary.csv"
    with csv_path.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    atomic_text(root/"summary.jsonl","".join(json.dumps(row,sort_keys=True,allow_nan=False)+"\n" for row in rows))
    numeric={field:np.asarray([row[field] for row in rows]) for field in fields if rows and all(isinstance(row.get(field),(int,float)) for row in rows)}
    np.savez_compressed(root/"summary.npz",**numeric)
    trajectory=[]
    for record,data in selected:
        ident=record["identity"]; meta=record["metadata"]
        if panel=="5b":
            for i,epoch in enumerate(data["epoch"]): trajectory.append({"seed":ident["seed"],"distance":ident["distance"],"parameters_per_gate":ident["parameters_per_gate"],"epoch":int(epoch),"lambda_ratio":float(data["lambda_ratio"][i]),"logical_learned":float(data["logical_learned"][i]),"logical_candidate":float(data["logical_candidate"][i]),"logical_fixed":float(data["logical_fixed"][i]),"logical_oracle":float(data["logical_oracle"][i])})
        if panel=="5c":
            for i,x in enumerate(data["x_distance"]): trajectory.append({"seed":ident["seed"],"distance":ident["distance"],"parameters_per_gate":ident["parameters_per_gate"],"x_distance":float(x),"normalized_speed":float(data["normalized_speed"][i]),"fit":bool(data["fit_mask"][i])})
    atomic_text(root/"trajectories.jsonl","".join(json.dumps(row,sort_keys=True,allow_nan=False)+"\n" for row in trajectory))
    if panel=="5c":
        fit_fields=("seed","distance","parameters_per_gate","total_controls","gamma_times_100","mode")
        with (root/"fits.csv").open("w",encoding="utf-8",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=fit_fields);writer.writeheader();writer.writerows({k:r[k] for k in fit_fields} for r in rows)
    result={"schema_version":"google-pure-v7-figure5-merge.v1","panel":panel,"mode":config["mode"],"protocol_hash":plan["protocol_hash"],
      "expected_shards":expected,"merged_shards":len(rows),"complete":complete,"partial":not complete,"row_count":len(rows),
      "trajectory_rows":len(trajectory),"summary_sha256":sha256_file(csv_path),"status":"DATA_COMPLETE" if complete else "ACQUISITION_PARTIAL"}
    atomic_json(root/"merge_manifest.json",result);return {**result,"rows":rows}

def merge_all(configs:dict[str,dict],*,allow_partial=False)->dict:
    result={p:merge_panel(p,configs[p],allow_partial=allow_partial) for p in ("5a","5b","5c")}
    manifest={"schema_version":"google-pure-v7-figure5-merge-all.v1","panels":{p:{k:v for k,v in r.items() if k!="rows"} for p,r in result.items()},"complete":all(r["complete"] for r in result.values())}
    atomic_json(figure5_root()/"merged"/"merge_all_manifest.json",manifest);return manifest
