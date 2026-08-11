"""Paths, hashing, and fail-closed non-final serialization for the V19 experiment."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_ROOT = ROOT / "artifacts/google_pure_v19/experimental_public_analogue"
CONFIG_PATH = ROOT / "configs/google_pure_v19/public_analogue_dynamic_validation.json"
SCHEMA_VERSION = "google-pure-v19-public-analogue-dynamic-validation.v1"
NONFINAL = {
    "final_evidence": False,
    "scientifically_valid": False,
    "paper_comparable": False,
    "paper_equivalence_claim_permitted": False,
    "source_exact": False,
    "source_scale_hyperparameters_identifiable": False,
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
    frequencies = value.get("frequencies", {})
    checks = {
        "schema": value.get("schema_version") ==
                  "google-pure-v19-public-analogue-dynamic-validation.v1",
        "frequency_labels": list(frequencies) == ["slow", "intermediate", "fast"],
        "strict_frequency_order": (
            float(frequencies["slow"]["frequency_per_epoch"]) <
            float(frequencies["intermediate"]["frequency_per_epoch"]) <
            float(frequencies["fast"]["frequency_per_epoch"])),
        "two_analysis_periods": all(int(row["analysis_periods"]) == 2
                                    for row in frequencies.values()),
        "one_transient_period": all(int(row["transient_periods"]) == 1
                                    for row in frequencies.values()),
        "bounded_candidates": all(2 <= int(row["candidates_per_epoch"]) <= 16
                                  for row in frequencies.values()),
        "bounded_cycles": all(0 < int(row["qec_cycles_per_candidate"]) <= 12000
                              for row in frequencies.values()),
        "no_heldout": value.get("heldout_seeds") == [],
        "no_source_budget": value.get("automatic_campaigns_permitted") == [],
        "mean_hyperparameters_frozen": value.get("mean_hyperparameters_changed") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"invalid V19 public-analogue protocol: {checks}")
    for label, row in frequencies.items():
        period = round(1.0 / float(row["frequency_per_epoch"]))
        expected = period * (int(row["transient_periods"]) + int(row["analysis_periods"]))
        if int(row["epochs"]) != expected:
            raise RuntimeError(f"{label} does not contain exact preregistered complete periods")
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
