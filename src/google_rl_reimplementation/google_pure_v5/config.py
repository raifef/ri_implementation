"""Configuration, hashing, and split guards for the pure v5 reproduction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


CERTIFICATION_SEEDS = tuple(range(9101, 9113))


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def config_dir() -> Path:
    return repository_root() / "configs" / "google_pure_v5"


def artifact_dir() -> Path:
    return repository_root() / "artifacts" / "google_pure_v5"


def read_mapping(path: str | Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration root is not a mapping: {path}")
    return value


def load_config(name: str) -> Mapping[str, Any]:
    return read_mapping(config_dir() / name)


def source_choices() -> Mapping[str, Any]:
    return load_config("source_unspecified_choices.yaml")


def paper_scale() -> Mapping[str, Any]:
    return load_config("paper_scale_reference.yaml")


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def guard_seed(seed: int, *, certification: bool = False) -> None:
    if seed in CERTIFICATION_SEEDS and not certification:
        raise ValueError("certification seed access is forbidden during development")
    if certification and seed not in CERTIFICATION_SEEDS:
        raise ValueError("certification requires a preregistered v5 seed")
