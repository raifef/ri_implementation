"""Deterministic JSON/Markdown report output."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from . import DISCLAIMER
from .config import artifact_dir


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def write_report(stem: str, payload: dict[str, Any], title: str) -> tuple[Path, Path]:
    root = artifact_dir()
    root.mkdir(parents=True, exist_ok=True)
    complete = _plain({**payload, "disclaimer": DISCLAIMER})
    json_path = root / f"{stem}.json"
    md_path = root / f"{stem}.md"
    json_path.write_text(json.dumps(complete, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    body = [f"# {title}", "", DISCLAIMER, ""]
    summary = complete.get("summary")
    if isinstance(summary, dict):
        body.extend(["## Summary", ""])
        body.extend(f"- **{key.replace('_', ' ')}:** `{value}`" for key, value in summary.items())
        body.append("")
    body.extend(["## Machine-readable record", "", "```json", json.dumps(complete, indent=2, sort_keys=True, allow_nan=False), "```", ""])
    md_path.write_text("\n".join(body), encoding="utf-8")
    return json_path, md_path


def read_artifact(stem: str) -> dict[str, Any]:
    return json.loads((artifact_dir() / f"{stem}.json").read_text(encoding="utf-8"))
