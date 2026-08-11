"""Configuration, hashing, and fail-closed split helpers for v4."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


CERTIFICATION_SEEDS = tuple(range(8101, 8113))


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def config_dir() -> Path:
    return repository_root() / "configs" / "google_synthetic_v4"


def artifact_dir() -> Path:
    return repository_root() / "artifacts" / "google_synthetic_v4"


def read_mapping(path: str | Path) -> Mapping[str, Any]:
    """Read JSON-compatible YAML without adding a YAML dependency."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration root is not a mapping: {path}")
    return value


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reject_certification_seed(seed: int, *, certification: bool = False) -> None:
    if seed in CERTIFICATION_SEEDS and not certification:
        raise ValueError("certification seed access is forbidden during development")
    if certification and seed not in CERTIFICATION_SEEDS:
        raise ValueError("certification requires one of the preregistered seeds 8101-8112")


def load_priors() -> Mapping[str, Any]:
    return read_mapping(config_dir() / "plant_priors.yaml")


def load_ensemble() -> Mapping[str, Any]:
    return read_mapping(config_dir() / "plant_ensemble.yaml")


def load_splits() -> Mapping[str, Any]:
    return read_mapping(config_dir() / "synthetic_splits.yaml")


def load_controller_choices() -> Mapping[str, Any]:
    return read_mapping(config_dir() / "controller_choices.yaml")
