"""Atomic NPZ shards with content hashes and strict duplicate/corruption checks."""
from __future__ import annotations
import os, tempfile
from pathlib import Path
from typing import Any, Mapping
import numpy as np
from hdfa_rl_suite.google_pure_v7.config import canonical_hash, sha256_file
from .common import atomic_json, figure5_root

IDENTITY_FIELDS=("panel","protocol_hash","controller_hash","grid_cell","distance","parameters_per_gate","seed","replicate","chunk")

def shard_id(identity: Mapping[str, Any]) -> str:
    material={key:identity.get(key) for key in IDENTITY_FIELDS}
    return canonical_hash(material)

def panel_shard_dir(panel: str) -> Path:
    return figure5_root()/"shards"/panel

def ensure_run_checkpoint(panel:str,plan:Mapping[str,Any],*,plant_hash:str,graph_hash:str,resume:bool)->dict:
    """Freeze all resume-critical identities before the first atomic condition shard."""
    path=figure5_root()/"manifests"/f"{panel}_{plan['protocol_hash'][:12]}_acquisition_checkpoint.json"
    expected={"schema_version":"google-pure-v7-figure5-checkpoint.v1","panel":panel,
      "protocol_hash":plan["protocol_hash"],"config_hash":plan["config_hash"],
      "controller_code_hash":plan["controller_code_hash"],"resolved_controller_hash":plan["resolved_controller_hash"],
      "graph_hash":graph_hash,"plant_hash":plant_hash,"rng_contract":"condition-hash/counter-derived",
      "checkpoint_boundary":"between atomic condition shards","completed_shards":[]}
    if path.exists():
        import json
        old=json.loads(path.read_text(encoding="utf-8"))
        keys=("panel","protocol_hash","config_hash","controller_code_hash","resolved_controller_hash","graph_hash","plant_hash")
        changed=[key for key in keys if old.get(key)!=expected[key]]
        if resume and changed:raise RuntimeError(f"resume rejected: changed checkpoint identities {changed}")
        if resume:return old
    atomic_json(path,expected);return expected

def update_run_checkpoint(panel:str,checkpoint:Mapping[str,Any],completed:list[str],expected:int)->dict:
    result={**dict(checkpoint),"completed_shards":list(completed),"expected_shards":expected,
      "complete":len(set(completed))==expected}
    atomic_json(figure5_root()/"manifests"/f"{panel}_{result['protocol_hash'][:12]}_acquisition_checkpoint.json",result);return result

def write_shard(panel: str, identity: Mapping[str, Any], arrays: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict:
    sid=shard_id(identity); root=panel_shard_dir(panel); root.mkdir(parents=True,exist_ok=True)
    npz=root/f"{sid}.npz"; meta=root/f"{sid}.json"
    fd,tmp=tempfile.mkstemp(prefix=f".{sid}.",suffix=".npz",dir=root); os.close(fd)
    try:
        np.savez_compressed(tmp,**{key:np.asarray(value) for key,value in arrays.items()}); os.replace(tmp,npz)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    record={"schema_version":"google-pure-v7-figure5-shard.v1","shard_id":sid,"identity":dict(identity),
            "npz":npz.name,"npz_sha256":sha256_file(npz),"arrays":{k:list(np.asarray(v).shape) for k,v in arrays.items()},
            "metadata":dict(metadata),"finalized":True}
    atomic_json(meta,record); return record

def discover_shards(panel: str) -> list[tuple[dict,np.lib.npyio.NpzFile]]:
    root=panel_shard_dir(panel); output=[]; seen=set()
    if not root.exists(): return output
    import json
    for path in sorted(root.glob("*.json")):
        record=json.loads(path.read_text(encoding="utf-8")); sid=record.get("shard_id")
        if sid in seen: raise RuntimeError(f"duplicate shard id: {sid}")
        seen.add(sid); npz=root/record["npz"]
        if not record.get("finalized") or not npz.exists() or sha256_file(npz)!=record["npz_sha256"]:
            raise RuntimeError(f"corrupt or incomplete shard: {sid}")
        output.append((record,np.load(npz,allow_pickle=False)))
    return output
