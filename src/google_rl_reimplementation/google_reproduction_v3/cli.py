"""Command-line entry points for the v3 Zenodo workflow."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

from .analysis import (
    fit_empirical_statistics,
    fit_surrogate,
    reproduce_public_analysis,
    validate_estimators,
    validate_surrogate,
)
from .dataset_manifest import build_inventory, estimate_inventory_cost, load_json_yaml
from .reporting import (
    build_source_to_dataset_map,
    freeze_data_splits,
    reclassify_v2,
    snapshot_v2,
)


def _workspace() -> Path:
    current = Path.cwd().resolve()
    if (current / "pyproject.toml").exists():
        return current
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("run a v3 command inside the google_rl_reimplementation workspace")


def _paths() -> dict[str, Path]:
    workspace = _workspace()
    return {
        "workspace": workspace,
        "artifact": workspace / "artifacts/google_reproduction_v3",
        "dataset_config": workspace / "configs/google_rl_v3/zenodo_local_dataset.yaml",
        "split_config": workspace / "configs/google_rl_v3/data_splits.yaml",
        "surrogate_config": workspace / "configs/google_rl_v3/empirical_surrogate.yaml",
    }


def _archive(paths: dict[str, Path]) -> Path:
    return Path(load_json_yaml(paths["dataset_config"])["selected_archive"])


def _announce(command: str, runtime: str, read: str, storage: str) -> None:
    print(json.dumps({"command": command, "estimated_runtime": runtime, "estimated_data_read": read, "estimated_storage": storage}, indent=2))
    sys.stdout.flush()


def inventory_main() -> None:
    paths = _paths()
    cost = estimate_inventory_cost(paths["dataset_config"])
    _announce("inventory-zenodo", cost["estimated_runtime"], cost["estimated_read"], cost["estimated_storage"])
    paths["artifact"].mkdir(parents=True, exist_ok=True)
    snapshot_v2(paths["workspace"], paths["artifact"])
    result = build_inventory(paths["dataset_config"], paths["artifact"])
    print(result["identity_status"])


def map_main() -> None:
    paths = _paths()
    _announce("map-paper-to-data", "under 2 seconds", "no bulk members", "under 100 KB")
    result = build_source_to_dataset_map(paths["artifact"])
    print(f"mapped {len(result['entries'])} paper anchors")


def reproduce_main() -> None:
    paths = _paths()
    _announce("reproduce-public-analysis", "20-60 seconds", "approximately 150-300 MB of logical-result members", "under 2 MB")
    result = reproduce_public_analysis(_archive(paths), paths["artifact"])
    print(json.dumps(result["headline_results"], indent=2))


def validate_estimators_main() -> None:
    paths = _paths()
    _announce("validate-estimators", "under 2 seconds", "public reproduction artifact only", "under 100 KB")
    result = validate_estimators(paths["artifact"])
    print(result["status"])


def splits_main() -> None:
    paths = _paths()
    _announce("freeze-data-splits", "under 10 seconds", "ZIP directory and 496 metadata files", "under 1 MB")
    result = freeze_data_splits(_archive(paths), paths["split_config"], paths["artifact"])
    print(result["manifest_sha256"])


def statistics_main() -> None:
    paths = _paths()
    _announce("fit-empirical-statistics", "30-120 seconds", "12 contiguous 2048-shot detector blocks plus logical outcomes", "under 3 MB")
    result = fit_empirical_statistics(_archive(paths), paths["artifact"])
    print(json.dumps(result["aggregate"], indent=2))


def fit_surrogate_main() -> None:
    paths = _paths()
    _announce("fit-surrogate", "under 5 seconds", "fit statistics artifact only", "under 200 KB")
    result = fit_surrogate(paths["artifact"], paths["surrogate_config"])
    print(result["status"])


def validate_surrogate_main() -> None:
    paths = _paths()
    _announce("validate-surrogate", "30-120 seconds", "12 untouched 2048-shot detector blocks", "under 2 MB")
    result = validate_surrogate(_archive(paths), paths["artifact"], paths["surrogate_config"])
    print(result["outcome"])


def reclassify_main() -> None:
    paths = _paths()
    _announce("reclassify-v2", "under 2 seconds", "immutable v2 JSON artifacts only", "under 200 KB")
    result = reclassify_v2(paths["workspace"], paths["artifact"])
    print(result["revised_overall_classification"])


def all_main() -> None:
    commands: list[Callable[[], None]] = [
        inventory_main,
        map_main,
        reproduce_main,
        validate_estimators_main,
        splits_main,
        statistics_main,
        fit_surrogate_main,
        validate_surrogate_main,
        reclassify_main,
    ]
    for command in commands:
        command()


if __name__ == "__main__":
    all_main()
