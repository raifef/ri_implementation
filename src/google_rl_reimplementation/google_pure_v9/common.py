"""Paths, provenance guards, immutable imports, and artifact writing for v9."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from google_rl_reimplementation.google_pure_v7.config import (
    canonical_hash,
    repository_root,
    sha256_file,
)
from google_rl_reimplementation.google_pure_v7.figure5.common import atomic_json, atomic_text

from . import CERTIFICATION_SEEDS, RETIRED_SEEDS


V8_IMPORTS = (
    "mathematical_contracts",
    "figure5a_edr_identity_audit",
    "figure5a_cost_decomposition",
    "figure5a_feasibility_decomposition",
    "native_unit_audit",
    "ppo_update_lifecycle_audit",
    "baseline_freezing_audit",
    "clipping_and_likelihood_audit",
    "entropy_and_scale_plumbing_audit",
    "temporal_protocol_audit",
    "exploration_floor_feasibility",
    "compact_fault_isolation_matrix",
    "root_cause_report",
    "repaired_controller_contract",
)


def artifact_root() -> Path:
    path = repository_root() / "artifacts" / "google_pure_v9"
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_root() -> Path:
    return repository_root() / "configs" / "google_pure_v9"


def guard_seed(seed: int) -> None:
    value = int(seed)
    if value in CERTIFICATION_SEEDS:
        raise RuntimeError(f"certification seed {value} cannot be consumed by v9 development")
    if value in RETIRED_SEEDS:
        raise RuntimeError(f"retired seed {value} cannot be consumed")


def guard_seed_registry(seeds: Iterable[int]) -> tuple[int, ...]:
    values = tuple(map(int, seeds))
    if len(values) != len(set(values)):
        raise ValueError("seed registry contains duplicates")
    for value in values:
        guard_seed(value)
    return values


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact root is not an object: {path}")
    return value


def load_config(name: str) -> dict[str, Any]:
    return read_json(config_root() / name)


def _markdown(title: str, payload: dict[str, Any]) -> str:
    lines = [f"# {title}", ""]
    for key, value in payload.items():
        if key in {"rows", "cells", "scale_trajectories", "phase_rows"}:
            count = len(value) if isinstance(value, (list, dict)) else None
            lines.append(f"- **{key}**: `{count} records`")
        else:
            lines.append(f"- **{key}**: `{json.dumps(value, sort_keys=True, default=str)}`")
    blockers = payload.get("blocking_reasons", [])
    if blockers:
        lines.extend(["", "## Blocking reasons", ""])
        lines.extend(f"- {item}" for item in blockers)
    return "\n".join(lines) + "\n"


def write_artifact(
    relative: str | Path,
    payload: dict[str, Any],
    title: str,
    *,
    markdown_relative: str | Path | None = None,
) -> dict[str, Any]:
    target = artifact_root() / Path(relative)
    if target.suffix:
        target = target.with_suffix("")
    target.parent.mkdir(parents=True, exist_ok=True)
    value = dict(payload)
    value.setdefault("certification_seeds_consumed", False)
    value.setdefault("seed_10101_consumed", False)
    value["artifact_hash"] = canonical_hash(value)
    json_target = target.with_suffix(".json")
    markdown_target = artifact_root() / Path(markdown_relative) if markdown_relative else target.with_suffix(".md")
    markdown_target.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(json_target, value)
    atomic_text(markdown_target, _markdown(title, value))
    return value


def verify_embedded_hash(payload: dict[str, Any]) -> bool:
    declared = payload.get("artifact_hash")
    unhashed = {key: value for key, value in payload.items() if key != "artifact_hash"}
    return isinstance(declared, str) and declared == canonical_hash(unhashed)


def import_v8_contracts() -> dict[str, Any]:
    source = repository_root() / "artifacts" / "google_pure_v8"
    records: dict[str, Any] = {}
    for name in V8_IMPORTS:
        path = source / f"{name}.json"
        if not path.is_file():
            raise RuntimeError(f"missing immutable v8 input: {name}")
        payload = read_json(path)
        if not verify_embedded_hash(payload):
            raise RuntimeError(f"embedded v8 artifact hash mismatch: {name}")
        records[name] = {
            "path": path.relative_to(repository_root()).as_posix(),
            "file_sha256": sha256_file(path),
            "declared_artifact_hash": payload["artifact_hash"],
            "verified": True,
        }
    manifest = {
        "schema_version": "google-pure-v9-v8-import-manifest.v1",
        "immutable_inputs": records,
        "all_hashes_verified": all(row["verified"] for row in records.values()),
        "prior_artifacts_modified": False,
        "blocking_reasons": [],
    }
    return write_artifact("v8_import_manifest", manifest, "Immutable v8 Import Manifest")


def protocol_hash(payload: dict[str, Any]) -> str:
    return canonical_hash(payload)
