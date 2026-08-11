"""Configuration, hashing, and certification-seed guards for pure v6."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


CERTIFICATION_SEEDS = tuple(range(12101, 12113))
RETIRED_DEVELOPMENT_EXPOSED_SEEDS = (10101,)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def config_dir() -> Path:
    return repository_root() / "configs" / "google_pure_v6"


def artifact_dir() -> Path:
    return repository_root() / "artifacts" / "google_pure_v6"


def read_mapping(path: str | Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration root is not a mapping: {path}")
    return value


def load_config(name: str) -> Mapping[str, Any]:
    return read_mapping(config_dir() / name)


def controller_choices(profile: str = "unchanged_v5_equivalent") -> Mapping[str, Any]:
    values = load_config("source_unspecified_choices.yaml")
    if profile not in values["profiles"]:
        raise ValueError(f"unknown v6 controller profile: {profile}")
    return values["profiles"][profile]


def paper_scale() -> Mapping[str, Any]:
    return load_config("paper_scale_reference.yaml")


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def guard_seed(seed: int, *, certification: bool = False) -> None:
    if seed in CERTIFICATION_SEEDS and not certification:
        raise ValueError("v6 certification seed access is forbidden during development")
    if certification and seed not in CERTIFICATION_SEEDS:
        raise ValueError("v6 certification requires a preregistered seed")
