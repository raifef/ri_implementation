"""Read-only v5 evidence snapshot and v5-to-v6 metric-schema migration."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import repository_root, sha256_file
from .reporting import write_report


EXPECTED_HEADLINE = {
    "injected_stability_ratio_legacy_mean_over_fixed": 0.973744458448387,
    "mean_ler_improvement": 0.020195305904267866,
    "step_response_epochs": 81.67554845362952,
    "natural_lf_gain_db": 2.5627491612581945,
    "natural_lf_gain_95_percent_interval_db": [1.2789177213420009, 4.817076702975785],
    "randomized_recovery_epoch": 140.0,
    "scaling_relative_deterioration": 0.00925300490240965,
    "distance_15_control_count": 38670,
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def current_v5_headline() -> dict[str, Any]:
    base = repository_root() / "artifacts" / "google_pure_v5"
    injected = _read(base / "injected_drift_stability.json")
    natural = _read(base / "natural_drift_spectral.json")
    recovery = _read(base / "randomized_recovery.json")
    scaling = _read(base / "convergence_scaling.json")
    d15 = [row for row in scaling["summaries"] if row["distance"] == 15]
    if len(d15) != 1:
        raise RuntimeError("v5 distance-15 scaling row is missing or ambiguous")
    return {
        "injected_stability_ratio_legacy_mean_over_fixed": injected["aggregate"]["median_control_only_stability_ratio"],
        "mean_ler_improvement": injected["aggregate"]["median_relative_mean_ler_improvement"],
        "step_response_epochs": injected["aggregate"]["median_step_response_epochs"],
        "natural_lf_gain_db": natural["aggregate"]["median_low_frequency_gain_db"],
        "natural_lf_gain_95_percent_interval_db": natural["aggregate"]["low_frequency_gain_95_percent_interval_across_plants"],
        "randomized_recovery_epoch": recovery["median_recovery_epoch"],
        "scaling_relative_deterioration": scaling["fits"]["distance3_to_distance15_relative_deterioration"],
        "distance_15_control_count": d15[0]["control_count"],
    }


def snapshot_v5() -> dict[str, Any]:
    root = repository_root()
    paths: list[Path] = []
    for relative in ("src/google_rl_reimplementation/google_pure_v5", "configs/google_pure_v5", "artifacts/google_pure_v5"):
        folder = root / relative
        paths.extend(path for path in folder.rglob("*") if path.is_file() and path.suffix in {".py", ".yaml", ".json", ".md", ".jsonl"})
    hashes = {path.relative_to(root).as_posix(): sha256_file(path) for path in sorted(paths)}
    headline = current_v5_headline()
    exact = headline == EXPECTED_HEADLINE
    payload = {
        "schema_version": "google-pure-v6-v5-immutable-snapshot.v1",
        "v5_runtime_imported_by_v6": False,
        "files_hashed": len(hashes),
        "file_sha256": hashes,
        "headline": headline,
        "expected_headline": EXPECTED_HEADLINE,
        "exact_headline_reproduction": exact,
        "v5_certification_blocked": True,
        "v5_certification_seeds_consumed": False,
        "v6_certification_seeds_consumed": False,
        "status": "PASS" if exact else "FAIL",
    }
    return write_report("v5_immutable_snapshot", payload, "v5 Immutable Snapshot")


def migrate_v5_metric_schema() -> dict[str, Any]:
    headline = current_v5_headline()
    legacy_residual = float(headline["injected_stability_ratio_legacy_mean_over_fixed"])
    suppression = 1.0 / legacy_residual
    payload = {
        "schema_version": "google-pure-v6-metric-contract.v1",
        "migration_only_no_v5_mutation": True,
        "stability": {
            "legacy_v5_field": "median_control_only_stability_ratio",
            "legacy_value_mean_std_over_fixed_std": legacy_residual,
            "stability_suppression_factor_fixed_over_mean": suppression,
            "stability_residual_ratio_mean_over_fixed": legacy_residual,
            "stability_factor_orientation": "fixed_std_over_learned_mean_std",
        },
        "spectral": {
            "low_frequency_suppression_db_fixed_over_mean": headline["natural_lf_gain_db"],
            "low_frequency_residual_db_mean_over_fixed": -float(headline["natural_lf_gain_db"]),
            "spectral_gain_orientation": "10log10(fixed_power/learned_mean_power)",
        },
        "command_and_artifact_schema_migrated": True,
        "status": "PASS",
        "certification_seeds_consumed": False,
    }
    return write_report("metric_contract", payload, "Canonical Metric Contract and v5 Migration")
