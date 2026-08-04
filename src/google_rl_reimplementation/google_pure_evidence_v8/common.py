from __future__ import annotations
import json
from pathlib import Path
from typing import Any,Iterable
from google_rl_reimplementation.google_pure_v7.config import canonical_hash,repository_root
from google_rl_reimplementation.google_pure_v7.figure5.common import atomic_json,atomic_text

def root()->Path:
    path=repository_root()/"artifacts/google_pure_evidence_v8";path.mkdir(parents=True,exist_ok=True);return path

def write(name:str,payload:dict[str,Any],title:str,*,directory:Path|None=None,
          json_name:str|None=None,md_name:str|None=None)->dict[str,Any]:
    target=directory or root();target.mkdir(parents=True,exist_ok=True);payload=dict(payload);payload.pop("artifact_hash",None);payload["artifact_hash"]=canonical_hash(payload)
    atomic_json(target/(json_name or f"{name}.json"),payload);lines=[f"# {title}",""]+[f"- **{k}**: `{json.dumps(v,default=str)}`" for k,v in payload.items() if k not in {"rows","traces","records"}]
    rows=payload.get("rows")
    if isinstance(rows,list) and rows:
        columns=list(dict.fromkeys(k for row in rows if isinstance(row,dict) for k in row))[:10]
        lines += ["","## Records","","| "+" | ".join(columns)+" |","| "+" | ".join("---" for _ in columns)+" |"]
        for row in rows:
            lines.append("| "+" | ".join(str(row.get(k,"" )).replace("|","\\|").replace("\n"," ")[:160] for k in columns)+" |")
    if payload.get("blocking_reasons"):lines+=["","## Blocking reasons",""]+[f"- {x}" for x in payload["blocking_reasons"]]
    atomic_text(target/(md_name or f"{name}.md"),"\n".join(lines)+"\n");return payload

def bundle_complete(directory:Path,required_files:Iterable[str],payload:dict[str,Any],required_fields:Iterable[str])->tuple[bool,list[str]]:
    missing=[str(name) for name in required_files if not (directory/str(name)).is_file()]
    missing += [f"field:{name}" for name in required_fields if name not in payload]
    return not missing,missing

def prompt1_report()->dict[str,Any]:
    path=repository_root()/"artifacts/google_pure_v8/root_cause_report.json"
    if not path.exists():raise RuntimeError("Prompt 1 root-cause report is required")
    return json.loads(path.read_text(encoding="utf-8"))

def require_reference_authorization(mode:str,execute:bool)->None:
    if mode not in {"smoke","reference"}: raise ValueError("mode must be smoke or reference")
    if mode=="reference" and not execute: raise RuntimeError("reference evidence requires --execute")
    if mode=="reference" and not prompt1_report()["prompt1_gate_pass"]:
        raise RuntimeError("reference evidence is blocked until the Prompt 1 scientific gate passes")
    if mode=="reference":
        preflight=root()/"evidence_protocol_preflight.json"
        if not preflight.exists() or not json.loads(preflight.read_text(encoding="utf-8")).get("protocol_gate_pass",False):
            raise RuntimeError("reference evidence is blocked until the Prompt 2 protocol preflight passes")
