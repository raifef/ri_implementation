"""Shared provenance, evidence, paths, and immutable-import helpers for v10."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from google_rl_reimplementation.google_pure_v7.config import canonical_hash, repository_root, sha256_file
from google_rl_reimplementation.google_pure_v7.figure5.common import atomic_json, atomic_text
from google_rl_reimplementation.google_pure_v9.common import V8_IMPORTS, verify_embedded_hash


def artifact_root() -> Path:
    path = repository_root() / "artifacts" / "google_pure_v10"
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_root() -> Path:
    return repository_root() / "configs" / "google_pure_v10"


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def load_config(name: str) -> dict[str, Any]:
    return read_json(config_root() / name)


def _markdown(title: str, payload: dict[str, Any]) -> str:
    lines = [f"# {title}", ""]
    for key, value in payload.items():
        if key in {"rows", "cells", "sensitivity_records", "raw_records"}:
            lines.append(f"- **{key}**: `{len(value)} records`")
        else:
            lines.append(f"- **{key}**: `{json.dumps(value, sort_keys=True, default=str)}`")
    blockers = payload.get("blocking_reasons", [])
    if blockers:
        lines.extend(["", "## Blocking reasons", ""])
        lines.extend(f"- {reason}" for reason in blockers)
    return "\n".join(lines) + "\n"


def write_artifact(
    relative: str | Path,
    payload: dict[str, Any],
    title: str,
    *,
    markdown_relative: str | Path | None = None,
) -> dict[str, Any]:
    base = artifact_root() / Path(relative)
    if base.suffix:
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    value = dict(payload)
    value.setdefault("artifact_complete", False)
    value.setdefault("mechanism_valid", False)
    value.setdefault("claim_supported", False)
    value.setdefault("paper_comparable", False)
    value.setdefault("blocking_reasons", [])
    value.setdefault("certification_seeds_consumed", False)
    value["artifact_hash"] = canonical_hash(value)
    markdown = artifact_root() / Path(markdown_relative) if markdown_relative else base.with_suffix(".md")
    markdown.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(base.with_suffix(".json"), value)
    atomic_text(markdown, _markdown(title, value))
    return value


def import_audits() -> dict[str, Any]:
    root = repository_root()
    records: dict[str, Any] = {}
    for name in V8_IMPORTS:
        path = root / "artifacts" / "google_pure_v8" / f"{name}.json"
        if not path.is_file():
            raise RuntimeError(f"missing immutable v8 audit: {name}")
        payload = read_json(path)
        if not verify_embedded_hash(payload):
            raise RuntimeError(f"immutable v8 hash mismatch: {name}")
        records[f"v8/{name}"] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "artifact_hash": payload["artifact_hash"],
        }
    v9_root = root / "artifacts" / "google_pure_v9"
    if v9_root.is_dir():
        for path in sorted(v9_root.rglob("*.json")):
            payload = read_json(path)
            if not verify_embedded_hash(payload):
                raise RuntimeError(f"completed v9 artifact hash mismatch: {path.relative_to(root)}")
            records[f"v9/{path.relative_to(v9_root).as_posix()}"] = {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "artifact_hash": payload["artifact_hash"],
            }
    payload = {
        "schema_version": "google-pure-v10-import-manifest.v1",
        "immutable_inputs": records,
        "all_hashes_verified": True,
        "prior_artifacts_modified": False,
        "artifact_complete": True,
        "mechanism_valid": True,
        "claim_supported": True,
        "paper_comparable": False,
        "blocking_reasons": [],
    }
    return write_artifact("import_manifest", payload, "v10 Immutable Import Manifest")
