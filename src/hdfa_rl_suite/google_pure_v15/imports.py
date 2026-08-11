"""Immutable imports for the V15 audit; all available lineage is hash checked."""
from __future__ import annotations

from typing import Any

from .contracts import nonfinal
from .io import ARTIFACT_ROOT, ROOT, atomic_json, canonical_hash, file_hash, read_json


BASE_IMPORTS = [
    ("controller_identity", "artifacts/google_pure_source_exact/direct_sigma_integration/controller_identity.json"),
    ("source_sensitivity_definition", "artifacts/google_pure_source_exact/control_normalization/source_contract.json"),
    ("source_sensitivity_bundle", "artifacts/google_pure_source_exact/control_normalization/calibration_bundle.json"),
    ("source_figure5a_config", "configs/google_pure_source_exact/figure5a.json"),
    ("source_loss", "src/hdfa_rl_suite/google_pure_source_exact/paper_families/common.py"),
    ("v12_imports", "artifacts/google_pure_v12/immutable_import_manifest.json"),
    ("v12_status", "artifacts/google_pure_v12/status.json"),
    ("v12_directional_comparison", "artifacts/google_pure_v12/directional_comparison/comparison.json"),
    ("v12_figure5b_lineage", "artifacts/google_pure_v12/lineage/figure5b_lineage.json"),
    ("v12_figure5c_lineage", "artifacts/google_pure_v12/lineage/figure5c_lineage.json"),
    ("v13_imports", "artifacts/google_pure_v13/import_manifest.json"),
    ("v13_status", "artifacts/google_pure_v13/status.json"),
]


def _paths() -> list[tuple[str, str]]:
    identity = read_json(ROOT / BASE_IMPORTS[0][1])
    rows = list(BASE_IMPORTS)
    rows.extend((f"frozen_controller_code:{path}", path)
                for path in identity["controller_code_files"])
    return rows


def build_import_manifest() -> dict[str, Any]:
    missing = [path for _, path in _paths() if not (ROOT / path).is_file()]
    if missing:
        raise RuntimeError("V15 immutable import failed closed; missing: " + ", ".join(missing))
    identity = read_json(ROOT / BASE_IMPORTS[0][1])
    changed = [path for path, expected in identity["controller_code_files"].items()
               if file_hash(ROOT / path) != expected]
    if changed:
        raise RuntimeError("V15 immutable import failed closed; controller changed: " + ", ".join(changed))
    imports = [{"role": role, "path": path, "sha256": file_hash(ROOT / path),
                "bytes": (ROOT / path).stat().st_size} for role, path in _paths()]
    result = nonfinal({
        "controller_mode": identity["controller_mode"],
        "controller_hash": identity["controller_hash"],
        "controller_code_hash": identity["controller_code_hash"],
        "parameterization": identity["parameterization"],
        "imports": imports,
        "import_count": len(imports),
        "lineage_sequence": ["V12", "V13", "V15"],
        "v14_expected": False,
    })
    payload = {key: value for key, value in result.items() if key != "manifest_hash"}
    result["manifest_hash"] = canonical_hash(payload)
    atomic_json(ARTIFACT_ROOT / "immutable_import_manifest.json", result)
    return result


def verify_import_manifest() -> dict[str, Any]:
    path = ARTIFACT_ROOT / "immutable_import_manifest.json"
    manifest = read_json(path) if path.is_file() else build_import_manifest()
    failures = []
    for row in manifest["imports"]:
        source = ROOT / row["path"]
        if not source.is_file():
            failures.append(f"missing:{row['path']}")
        elif file_hash(source) != row["sha256"]:
            failures.append(f"changed:{row['path']}")
    payload = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if canonical_hash(payload) != manifest["manifest_hash"]:
        failures.append("manifest_self_hash")
    if failures:
        raise RuntimeError("V15 immutable import failed closed: " + ", ".join(failures))
    return nonfinal({"pass": True, "checked_files": len(manifest["imports"]),
                     "manifest_hash": manifest["manifest_hash"],
                     "lineage_sequence": manifest["lineage_sequence"],
                     "v14_expected": False})
