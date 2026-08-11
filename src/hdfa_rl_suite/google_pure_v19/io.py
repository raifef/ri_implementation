"""Deterministic paths and non-final serialization for V19."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_ROOT = ROOT / "artifacts/google_pure_v19"
CONFIG_PATH = ROOT / "configs/google_pure_v19/protocol.json"
SCHEMA_VERSION = "google-pure-v19-exploration-scale-diagnosis.v1"
NONFINAL = {
    "final_evidence": False,
    "scientifically_valid": False,
    "paper_comparable": False,
    "paper_equivalence_claim_permitted": False,
    "reference_evidence_complete": False,
    "heldout_seeds_consumed": False,
    "long_run_auto_launched": False,
    "source_budget_auto_launched": False,
    "reference_campaign_auto_launched": False,
}


def canonical_hash(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def config() -> dict[str, Any]:
    value = read_json(CONFIG_PATH)
    required_lambdas = [0.0, 0.25, 0.5, 0.75, 1.0]
    checks = {
        "schema": value.get("schema_version") == "google-pure-v19-protocol.v1",
        "lambdas": value.get("frozen_sigma_multipliers") == required_lambdas,
        "no_mean_retuning": value.get("mean_controller_retuning_permitted") is False,
        "no_production_mutation": value.get("production_figure5a_changes_permitted") is False,
        "no_automatic_campaigns": value.get("automatic_campaigns_permitted") == [],
        "phase_bins": int(value.get("phase_bins", 0)) >= 4,
        "positive_delta": float(value.get("finite_difference_delta", 0)) > 0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"invalid V19 protocol: {checks}")
    return value


def nonfinal(value: Mapping[str, Any]) -> dict[str, Any]:
    return {**dict(value), "schema_version": SCHEMA_VERSION, **NONFINAL}


def atomic_json(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def atomic_text(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)
    return path
