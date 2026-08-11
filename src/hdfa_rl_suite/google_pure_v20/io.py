"""Paths, provenance, and fail-closed serialization for V20."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_ROOT = ROOT / "artifacts/google_pure_v20"
CONFIG_PATH = ROOT / "configs/google_pure_v20/protocol.json"
SCHEMA_VERSION = "google-pure-v20-fast-mean-diagnosis.v1"
NONFINAL = {
    "final_evidence": False,
    "scientifically_valid": False,
    "paper_comparable": False,
    "paper_equivalence_claim_permitted": False,
    "source_exact": False,
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


def settings() -> dict[str, Any]:
    value = read_json(CONFIG_PATH)
    checks = {
        "schema": value.get("schema_version") == SCHEMA_VERSION,
        "fast_only": value.get("automatic_acquisition_frequencies") == [1 / 150],
        "no_forbidden": value.get("automatic_campaigns_permitted") == [],
        "matched_budget": int(value.get("postrepair", {}).get(
            "candidates_per_epoch", 0)) == 8 and int(value.get("postrepair", {}).get(
                "qec_cycles_per_candidate", 0)) == 12000,
        "one_repair": value.get("maximum_causal_repairs") == 1,
    }
    if not all(checks.values()):
        raise RuntimeError(f"invalid V20 protocol: {checks}")
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


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def write_artifact(name: str, value: Mapping[str, Any], *, title: str,
                   notes: list[str] | None = None) -> dict[str, Any]:
    result = nonfinal(value)
    atomic_json(ARTIFACT_ROOT / f"{name}.json", result)
    if notes is not None:
        atomic_text(ARTIFACT_ROOT / f"{name}.md", "\n".join([
            f"# {title}", "", *notes, "",
            "Bounded development evidence only; no paper-equivalence claim is permitted.",
        ]))
    return result
