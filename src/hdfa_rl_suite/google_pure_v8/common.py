from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hdfa_rl_suite.google_pure_v7.config import canonical_hash, repository_root
from hdfa_rl_suite.google_pure_v7.figure5.common import atomic_json, atomic_text

CERTIFICATION_SEEDS = set(range(12101, 12113))
RETIRED_SEEDS = {10101}


def root() -> Path:
    path = repository_root() / "artifacts" / "google_pure_v8"
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_root() -> Path:
    return repository_root() / "configs" / "google_pure_v8"


def guard_seed(seed: int) -> None:
    if int(seed) in CERTIFICATION_SEEDS | RETIRED_SEEDS:
        raise RuntimeError(f"protected seed {seed} cannot be consumed by v8 development")


def write_report(name: str, payload: dict[str, Any], title: str) -> dict[str, Any]:
    payload = dict(payload); payload.setdefault("certification_seeds_consumed", False)
    payload["artifact_hash"] = canonical_hash(payload)
    atomic_json(root()/f"{name}.json", payload)
    lines=[f"# {title}", ""]
    for key,value in payload.items():
        if key not in {"rows","conditions","records"}: lines.append(f"- **{key}**: `{json.dumps(value, default=str)}`")
    if payload.get("blocking_reasons"):
        lines += ["", "## Blocking reasons", ""] + [f"- {reason}" for reason in payload["blocking_reasons"]]
    atomic_text(root()/f"{name}.md", "\n".join(lines)+"\n")
    return payload

