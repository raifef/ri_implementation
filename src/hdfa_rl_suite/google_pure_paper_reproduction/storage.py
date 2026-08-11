"""Atomic, resumable storage with strict shard compatibility checks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from hdfa_rl_suite.google_pure_v7.config import canonical_hash, repository_root
from hdfa_rl_suite.google_pure_v7.figure5.common import atomic_json, atomic_text

from .experiment_families import assert_merge_compatible, require_family
from .provenance import make_provenance
from hdfa_rl_suite.google_pure_source_exact.source_normalization import (
    canonical_hash as v15_canonical_hash,
    require_v15_boundary_provenance,
)

FIGURE5A_FAMILY = "FIGURE5A_REAL_TIME_STEERING"
FIGURE5A_PROTOCOL_KEYS = (
    "implementation_version", "coordinate_contract", "action_execution",
    "plant_boundary_execution", "likelihood_space", "entropy_space",
    "empirical_relative_normalization_applied", "mean_bounds_applied",
)


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


def _validated_execution(protocol: Mapping[str, Any], data: Mapping[str, Any]) -> dict[str, Any]:
    if protocol["experiment_family"] == FIGURE5A_FAMILY:
        execution = dict(data.get("source_coordinate_provenance") or {})
        for key in FIGURE5A_PROTOCOL_KEYS:
            if execution.get(key) != protocol.get(key):
                raise RuntimeError(f"Figure 5a source-coordinate driver/protocol mismatch for {key}")
        if not execution.get("control_order_hash"):
            raise RuntimeError("Figure 5a source-coordinate provenance lacks a control-order hash")
        if bool(protocol.get("fresh_acquisition_required")) and not execution.get("fresh_acquisition"):
            raise RuntimeError("fresh acquisition protocol received reused lower-level state")
        if execution.get("reused_shard_ids"):
            raise RuntimeError("a newly written Figure 5a shard cannot inherit reused shard ids")
        return execution
    boundary = dict(data.get("v15_provenance") or {})
    require_v15_boundary_provenance(boundary)
    for key in (
        "implementation_version", "sensitivity_map_hash", "sensitivity_definition_hash",
        "calibration_bundle_hash", "detector_degree_audit_hash", "boundary_transform_hash",
        "boundary_transform_name", "source_budget_profile",
    ):
        if boundary.get(key) != protocol.get(key):
            raise RuntimeError(f"V15 driver/protocol mismatch for {key}")
    if bool(protocol.get("fresh_acquisition_required")) and not boundary.get("fresh_acquisition"):
        raise RuntimeError("fresh acquisition protocol received reused lower-level state")
    if boundary.get("reused_shard_ids"):
        raise RuntimeError("a newly written V15 shard cannot inherit reused shard ids")
    return boundary


def _validate_shard(protocol: Mapping[str, Any], row: Mapping[str, Any]) -> None:
    if row.get("identity", {}).get("protocol_hash") != protocol["protocol_hash"]:
        raise RuntimeError("stale shard has the wrong protocol hash")
    provenance = row.get("provenance", {})
    if protocol["experiment_family"] == FIGURE5A_FAMILY:
        for key in (*FIGURE5A_PROTOCOL_KEYS, "experiment_driver_hash", "source_budget_profile"):
            if provenance.get(key) != protocol.get(key):
                raise RuntimeError(f"stale or incompatible Figure 5a shard: {key}")
        if not provenance.get("control_order_hash"):
            raise RuntimeError("stale Figure 5a shard lacks a control-order hash")
        return
    require_v15_boundary_provenance(provenance)
    for key in (
        "implementation_version", "sensitivity_map_hash", "sensitivity_definition_hash",
        "calibration_bundle_hash", "detector_degree_audit_hash", "boundary_transform_hash",
        "boundary_transform_name", "experiment_driver_hash", "source_budget_profile",
    ):
        if provenance.get(key) != protocol.get(key):
            raise RuntimeError(f"stale or incompatible V15 shard: {key}")


def write_shard(protocol: Mapping[str, Any], condition: Mapping[str, Any], data: Mapping[str, Any]) -> dict[str, Any]:
    identity = {"family": protocol["experiment_family"], "protocol_hash": protocol["protocol_hash"], "condition": dict(condition)}
    sid = canonical_hash(identity)
    path = _run_dir(protocol) / "shards" / f"{sid}.json"
    if path.exists():
        if protocol.get("fresh_acquisition_required"):
            raise RuntimeError(f"fresh acquisition refuses pre-existing shard: {sid}")
        old = json.loads(path.read_text(encoding="utf-8"))
        _validate_shard(protocol, old)
        if old.get("identity") != identity or canonical_hash(old.get("data")) != canonical_hash(dict(data)):
            raise RuntimeError(f"duplicate shard identity with changed content: {sid}")
        return old
    execution = _validated_execution(protocol, data)
    source_coordinate = execution if protocol["experiment_family"] == FIGURE5A_FAMILY else None
    boundary = execution if source_coordinate is None else None
    provenance = make_provenance(
        protocol["experiment_family"], protocol_hash=protocol["protocol_hash"], mode=protocol["mode"],
        plant_hash=protocol["plant_hash"], graph_hash=protocol["graph_hash"], shard_ids=[sid],
        v15_provenance=boundary, source_coordinate_provenance=source_coordinate,
        experiment_driver_hash=protocol["experiment_driver_hash"],
        fresh_acquisition=bool(execution["fresh_acquisition"]),
        reused_shard_ids=list(execution["reused_shard_ids"]),
        source_budget_profile=str(execution["source_budget_profile"]),
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
        _validate_shard(protocol, row)
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
        _validate_shard(protocol, record)
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
    first = records[0]["provenance"]
    order_hashes = sorted({row["provenance"]["control_order_hash"] for row in records})
    figure5a = protocol["experiment_family"] == FIGURE5A_FAMILY
    scale_hashes = ([] if figure5a else
                    sorted({row["provenance"]["expanded_scale_hash"] for row in records}))
    execution = dict(first)
    if figure5a:
        if len(order_hashes) != 1:
            raise RuntimeError("Figure 5a shards disagree on the exact 41-control ordering")
        execution["control_order_hash"] = order_hashes[0]
    else:
        execution["control_order_hash"] = v15_canonical_hash(order_hashes)
        execution["expanded_scale_hash"] = v15_canonical_hash(scale_hashes)
    fresh = all(bool(row["provenance"].get("fresh_acquisition")) for row in records)
    reused = sorted({reused_id for row in records
                     for reused_id in row["provenance"].get("reused_shard_ids", [])})
    if protocol.get("fresh_acquisition_required") and (not fresh or reused):
        raise RuntimeError("fresh merge contains reused evidence")
    provenance = make_provenance(
        protocol["experiment_family"], protocol_hash=protocol["protocol_hash"], mode=protocol["mode"],
        plant_hash=protocol["plant_hash"], graph_hash=protocol["graph_hash"], complete=complete, shard_ids=ids,
        v15_provenance=None if figure5a else execution,
        source_coordinate_provenance=execution if figure5a else None,
        experiment_driver_hash=protocol["experiment_driver_hash"],
        fresh_acquisition=fresh, reused_shard_ids=reused,
        source_budget_profile=str(protocol["source_budget_profile"]),
    )
    result = {
        "schema_version": "google-paper-reproduction-merge.v1", "experiment_family": protocol["experiment_family"],
        "mode": protocol["mode"], "protocol_hash": protocol["protocol_hash"], "expected_shards": expected,
        "merged_shards": len(records), "complete": complete, "partial": not complete,
        "shard_ids": ids, "rows": [row["data"] for row in records], "provenance": provenance,
        "control_order_hashes": order_hashes, "expanded_scale_hashes": scale_hashes,
        "fresh_acquisition": fresh, "reused_shard_ids": reused,
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
