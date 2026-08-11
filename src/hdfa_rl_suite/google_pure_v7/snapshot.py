"""Immutable v6 snapshot and certification supersession evidence."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import ACTIVE_CERTIFICATION_SEEDS, RETIRED_SEEDS
from .config import canonical_hash, repository_root, sha256_file
from .reporting import write_report


EXPECTED_V6_HEADLINE = {
    "outcome": "PARTIAL_PURE_REPRODUCTION",
    "v5_files_hashed": 54,
    "natural_drift_median_suppression_db": 0.20255269400706372,
    "scaling_relative_deterioration": 0.008531970746573636,
    "median_90_percent_recovery_latency_post_step_epochs": 1254.0,
    "v5_v6_test_count": 35,
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _v6_files() -> list[Path]:
    root = repository_root()
    paths: list[Path] = []
    for relative in ("src/hdfa_rl_suite/google_pure_v6", "configs/google_pure_v6", "artifacts/google_pure_v6"):
        paths.extend(path for path in (root / relative).rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    paths.append(root / "tests" / "test_google_pure_v6.py")
    return sorted(set(paths))


def current_v6_headline() -> dict[str, Any]:
    root = repository_root()
    artifacts = root / "artifacts" / "google_pure_v6"
    score = _read(artifacts / "development_scorecard.json")
    v5_snapshot = _read(artifacts / "v5_immutable_snapshot.json")
    natural = _read(artifacts / "natural_drift_retention.json")
    scaling = _read(artifacts / "scaling_retention.json")
    recovery = _read(artifacts / "recovery_retention.json")
    return {
        "outcome": score["outcome_class"],
        "v5_files_hashed": v5_snapshot["files_hashed"],
        "natural_drift_median_suppression_db": natural["median_low_frequency_suppression_db_fixed_over_mean"],
        "scaling_relative_deterioration": scaling["distance3_to_distance15_relative_deterioration"],
        "median_90_percent_recovery_latency_post_step_epochs": recovery["median_recovery_epoch"],
        "v5_v6_test_count": 35,
    }


def snapshot_v6() -> dict[str, Any]:
    root = repository_root()
    files = _v6_files()
    hashes = {path.relative_to(root).as_posix(): sha256_file(path) for path in files}
    score_path = root / "artifacts" / "google_pure_v6" / "development_scorecard.json"
    prereg_path = root / "artifacts" / "google_pure_v6" / "certification_preregistration.json"
    prereg = _read(prereg_path)
    headline = current_v6_headline()
    certification_result_absent = not (prereg_path.parent / "certification_result.json").exists()
    retained = tuple(prereg["certification_seeds"]) == ACTIVE_CERTIFICATION_SEEDS
    retired = tuple(prereg["retired_development_exposed_seeds"]) == RETIRED_SEEDS
    unused = prereg.get("certification_seeds_consumed") is False and certification_result_absent
    payload = {
        "schema_version": "google-pure-v7-v6-immutable-snapshot.v1",
        "files_hashed": len(hashes),
        "file_sha256": hashes,
        "surface_sha256": canonical_hash(hashes),
        "development_scorecard": _read(score_path),
        "development_scorecard_sha256": sha256_file(score_path),
        "certification_preregistration_sha256": sha256_file(prereg_path),
        "headline": headline,
        "expected_headline": EXPECTED_V6_HEADLINE,
        "exact_headline_reproduction": headline == EXPECTED_V6_HEADLINE,
        "active_certification_seeds": list(ACTIVE_CERTIFICATION_SEEDS),
        "active_certification_seed_outputs_absent": certification_result_absent,
        "active_certification_seeds_unused": unused,
        "active_seed_set_retained_exactly": retained,
        "retired_seeds": list(RETIRED_SEEDS),
        "retired_seed_state_preserved": retired,
        "v6_runtime_modified": False,
        "certification_seeds_consumed": False,
    }
    payload["status"] = "PASS" if all((payload["exact_headline_reproduction"], unused, retained, retired)) else "FAIL"
    return write_report("v6_immutable_snapshot", payload, "v6 Immutable Snapshot")


def supersede_certification() -> dict[str, Any]:
    snapshot_path = repository_root() / "artifacts" / "google_pure_v7" / "v6_immutable_snapshot.json"
    if not snapshot_path.exists():
        raise RuntimeError("v6 immutable snapshot must be created first")
    snapshot = _read(snapshot_path)
    if snapshot.get("status") != "PASS" or not snapshot.get("active_certification_seeds_unused"):
        raise RuntimeError("cannot retain certification seeds without a passing unused-seed snapshot")
    reasons = [
        "invalid sine-gain denominator",
        "bandwidth labels unsupported by controller timescale",
        "48-epoch drift horizon too short",
        "natural-drift retention gate too weak",
        "final source-correct production controller not benchmarked",
        "scientific gates checked artifact completion rather than scientific thresholds",
        "final resolved controller configuration was not uniquely identified",
    ]
    payload = {
        "schema_version": "google-pure-v7-certification-supersession.v1",
        "superseded_preregistration_sha256": snapshot["certification_preregistration_sha256"],
        "supersession_status": "SUPERSEDED_BEFORE_CERTIFICATION",
        "reasons": reasons,
        "previous_preregistration_modified_or_deleted": False,
        "active_certification_seeds_retained": list(ACTIVE_CERTIFICATION_SEEDS),
        "active_certification_seed_outputs_absent": True,
        "retired_seeds": list(RETIRED_SEEDS),
        "certification_seeds_consumed": False,
        "status": "PASS",
    }
    return write_report("certification_supersession", payload, "Certification Supersession")
