"""Deterministic JSON and Markdown evidence reports."""
from __future__ import annotations

import json
from typing import Any

from . import DISCLAIMER
from .config import artifact_dir


def _markdown(value: Any, depth: int = 0) -> list[str]:
    pad = "  " * depth
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}- **{key}**")
                lines.extend(_markdown(item, depth + 1))
            else:
                lines.append(f"{pad}- **{key}:** `{item}`")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}-")
                lines.extend(_markdown(item, depth + 1))
            else:
                lines.append(f"{pad}- `{item}`")
        return lines
    return [f"{pad}{value}"]


def write_report(name: str, payload: dict[str, Any], title: str) -> dict[str, Any]:
    target = artifact_dir()
    target.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    data.setdefault("disclaimer", DISCLAIMER)
    (target / f"{name}.json").write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (target / f"{name}.md").write_text(
        "\n".join([f"# {title}", "", DISCLAIMER, "", *_markdown(data), ""]), encoding="utf-8"
    )
    return data


def read_artifact(name: str) -> dict[str, Any]:
    path = artifact_dir() / f"{name}.json"
    if not path.exists():
        raise RuntimeError(f"required v7 artifact is missing: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))
