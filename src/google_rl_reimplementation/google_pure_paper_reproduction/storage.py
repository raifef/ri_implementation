"""Atomic, resumable storage with strict shard compatibility checks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from google_rl_reimplementation.google_pure_v7.config import canonical_hash, repository_root
from google_rl_reimplementation.google_pure_v7.figure5.common import atomic_json, atomic_text

from .experiment_families import assert_merge_compatible, require_family
from .provenance import make_provenance


REQUIRED_DIRS = (
    "source_contract", "claim_registry", "experiment_protocols", "paper_assets",
    "public_data_reproduction", "synthetic_reproduction", "side_by_side", "tables",
    "reports", "manifests", "validation", "figures",
)


def artifact_root() -> Path:
    return repository_root() / "artifacts" / "google_pure_paper_reproduction"


def initialise_layout() -> Path:
    root = artifact_root()
    for name in REQUIRED_DIRS:
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def protocol_path(family: str, mode: str) -> Path:
    return initialise_layout() / "experiment_protocols" / f"{require_family(family).lower()}_{mode}.json"


def save_protocol(protocol: Mapping[str, Any]) -> Path:
    return atomic_json(protocol_path(protocol["experiment_family"], protocol["mode"]), dict(protocol))


def load_protocol(family: str, mode: str) -> dict[str, Any]:
    path = protocol_path(family, mode)
    if not path.exists():
        raise RuntimeError(f"missing plan: run the {require_family(family)} plan command for mode {mode}")
    return json.loads(path.read_text(encoding="utf-8"))


def _run_dir(protocol: Mapping[str, Any]) -> Path:
    slugs = {
        "FIGURE5A_REAL_TIME_STEERING": "fig5a", "FIGURE5B_SPARSE_SCALING": "fig5b",
        "FIGURE5C_CONVERGENCE_LAW": "fig5c", "NATURAL_DRIFT_SPECTRAL_SUPPRESSION": "natural",
        "RANDOMIZED_RECOVERY_AFTER_SPOIL": "recovery", "STEP_RESPONSE_INJECTED_DRIFT": "step",
    }
    return initialise_layout() / "synthetic_reproduction" / slugs[protocol["experiment_family"]] / protocol["protocol_hash"][:16]


def write_shard(protocol: Mapping[str, Any], condition: Mapping[str, Any], data: Mapping[str, Any]) -> dict[str, Any]:
    identity = {"family": protocol["experiment_family"], "protocol_hash": protocol["protocol_hash"], "condition": dict(condition)}
    sid = canonical_hash(identity)
    path = _run_dir(protocol) / "shards" / f"{sid}.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("identity") != identity or canonical_hash(old.get("data")) != canonical_hash(dict(data)):
            raise RuntimeError(f"duplicate shard identity with changed content: {sid}")
        return old
    provenance = make_provenance(
        protocol["experiment_family"], protocol_hash=protocol["protocol_hash"], mode=protocol["mode"],
        plant_hash=protocol["plant_hash"], graph_hash=protocol["graph_hash"], shard_ids=[sid],
    )
    record = {
        "schema_version": "google-paper-reproduction-shard.v1", "shard_id": sid,
        "identity": identity, "provenance": provenance, "data": dict(data), "finalized": True,
    }
    atomic_json(path, record)
    return record


def discover(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    root = _run_dir(protocol) / "shards"
    if not root.exists():
        return []
    output, seen = [], set()
    for path in sorted(root.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if not row.get("finalized") or row.get("shard_id") in seen:
            raise RuntimeError(f"stale, incomplete, or duplicate shard {path.name}")
        seen.add(row["shard_id"])
        output.append(row)
    return output


def merge(protocol: Mapping[str, Any], *, allow_partial: bool = False) -> dict[str, Any]:
    # Stream shards so a reference Figure 5a merge never retains all candidate
    # action tensors in memory.  Raw trajectories remain immutable in shards;
    # the merged phase-grid artifact needs only condition-level scalars.
    shard_root = _run_dir(protocol) / "shards"
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(shard_root.glob("*.json")) if shard_root.exists() else []:
        record = json.loads(path.read_text(encoding="utf-8"))
        sid = record.get("shard_id")
        if not record.get("finalized") or not sid or sid in seen:
            raise RuntimeError(f"stale, incomplete, or duplicate shard {path.name}")
        seen.add(sid)
        data = record["data"]
        if protocol["experiment_family"] == "FIGURE5A_REAL_TIME_STEERING" and "trajectory" in data:
            trajectory = data["trajectory"]
            data = {key: value for key, value in data.items() if key != "trajectory"}
            data["raw_trajectory_shard_id"] = sid
            data["raw_trajectory_fields"] = sorted(trajectory)
        records.append({"shard_id": sid, "provenance": record["provenance"], "data": data})
    assert_merge_compatible(records)
    expected = int(protocol["condition_count"])
    complete = len(records) == expected
    if len(records) > expected:
        raise RuntimeError(f"unexpected extra shards: {len(records)}/{expected}")
    if not complete and not allow_partial:
        raise RuntimeError(f"incomplete acquisition: {len(records)}/{expected}")
    ids = [row["shard_id"] for row in records]
    provenance = make_provenance(
        protocol["experiment_family"], protocol_hash=protocol["protocol_hash"], mode=protocol["mode"],
        plant_hash=protocol["plant_hash"], graph_hash=protocol["graph_hash"], complete=complete, shard_ids=ids,
    )
    result = {
        "schema_version": "google-paper-reproduction-merge.v1", "experiment_family": protocol["experiment_family"],
        "mode": protocol["mode"], "protocol_hash": protocol["protocol_hash"], "expected_shards": expected,
        "merged_shards": len(records), "complete": complete, "partial": not complete,
        "shard_ids": ids, "rows": [row["data"] for row in records], "provenance": provenance,
        "raw_trajectories_retained_in_shards": True,
    }
    target = _run_dir(protocol) / "merged.json"
    atomic_json(target, result)
    atomic_json(initialise_layout() / "manifests" / f"{protocol['experiment_family'].lower()}_{protocol['mode']}_merge.json", {k: v for k, v in result.items() if k != "rows"})
    return result


def load_merged(protocol: Mapping[str, Any]) -> dict[str, Any]:
    path = _run_dir(protocol) / "merged.json"
    if not path.exists():
        raise RuntimeError("missing merged data")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["protocol_hash"] != protocol["protocol_hash"]:
        raise RuntimeError("stale merged artifact has the wrong protocol hash")
    return value


__all__ = ["artifact_root", "atomic_json", "atomic_text", "discover", "initialise_layout", "load_merged", "load_protocol", "merge", "save_protocol", "write_shard"]
