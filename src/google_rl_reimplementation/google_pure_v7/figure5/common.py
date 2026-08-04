"""Shared deterministic contracts for all three Figure 5 panels."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any, Mapping

from google_rl_reimplementation.google_pure_v7.config import canonical_hash, repository_root

SOURCE_STATUSES = (
    "EXPLICITLY_SPECIFIED", "DERIVED_FROM_EXPLICIT_SOURCE", "IMPLIED_BY_SOURCE",
    "NOT_PUBLICLY_SPECIFIED", "SYNTHETIC_REPRODUCTION_CHOICE",
)
PIPELINE_STATUSES = (
    "PIPELINE_NOT_BUILT", "PIPELINE_BUILT_UNVALIDATED", "SMOKE_VALIDATED",
    "ACQUISITION_PARTIAL", "DATA_COMPLETE", "SCIENTIFIC_VALIDATION_FAILED",
    "READY_TO_PLOT", "PLOT_COMPLETE",
)
MODES = ("smoke", "validation", "reference", "paper-scale")
SCHEMA = "google-pure-v7-figure5.v1"


def figure5_root() -> Path:
    return repository_root() / "artifacts" / "google_pure_v7" / "figure5"


def config_root() -> Path:
    return repository_root() / "configs" / "google_pure_v7" / "figure5"


def read_config(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_absolute():
        target = config_root() / target
    value = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Figure 5 configuration must be a mapping")
    return value


def atomic_json(path: str | Path, value: Any) -> Path:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
    return target


def atomic_text(path: str | Path, text: str) -> Path:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
    return target


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def environment_manifest() -> dict[str, str]:
    return {"python": platform.python_version(), "platform": platform.platform(), "numpy": __import__("numpy").__version__}


def require_mode(mode: str, *, execute_paper_scale: bool = False) -> None:
    if mode not in MODES: raise ValueError(f"unknown run mode: {mode}")
    if mode == "paper-scale" and not execute_paper_scale:
        raise RuntimeError("paper-scale acquisition requires --execute-paper-scale")


def stable_seed(*parts: Any) -> int:
    return int(canonical_hash(parts)[:16], 16) % (2**63 - 1)


def config_hash(config: Mapping[str, Any]) -> str:
    return canonical_hash(dict(config))

