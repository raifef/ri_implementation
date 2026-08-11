"""Immutable import registry for the V12 audit namespace."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import CURRENT_CLASSIFICATIONS, NONFINAL_FIELDS, V12_SCHEMA
from .io import ARTIFACT_ROOT, ROOT, atomic_json, atomic_text, canonical_hash, file_hash, read_json

FAMILIES = {
    "FIGURE5A_REAL_TIME_STEERING": "fig5a",
    "FIGURE5B_SPARSE_SCALING": "fig5b",
    "FIGURE5C_CONVERGENCE_LAW": "fig5c",
    "NATURAL_DRIFT_SPECTRAL_SUPPRESSION": "natural",
    "RANDOMIZED_RECOVERY_AFTER_SPOIL": "recovery",
    "STEP_RESPONSE_INJECTED_DRIFT": "step",
}


def _required_paths() -> list[tuple[str, Path]]:
    identity_path = ROOT / "artifacts/google_pure_source_exact/direct_sigma_integration/controller_identity.json"
    identity = read_json(identity_path)
    paths: list[tuple[str, Path]] = [("controller_identity", identity_path)]
    paths.extend(("controller_code", ROOT / relative) for relative in identity["controller_code_files"])
    paths.extend([
        ("controller_config", ROOT / "configs/google_pure_source_exact/figure5a.json"),
        ("controller_checkpoint", ROOT / "artifacts/google_pure_source_exact/direct_sigma_integration/checkpoint-de2f9061be224d74.json"),
        ("integration_manifest", ROOT / "artifacts/google_pure_source_exact/direct_sigma_integration/manifest.json"),
        ("normalization_bundle", ROOT / "artifacts/google_pure_source_exact/control_normalization/calibration_bundle.json"),
        ("one_hour_manifest", ROOT / "artifacts/google_pure_paper_reproduction/reports/one_hour_validation_manifest.json"),
    ])
    for family, slug in FAMILIES.items():
        stem = family.lower()
        protocol = ROOT / f"artifacts/google_pure_paper_reproduction/experiment_protocols/{stem}_validation.json"
        if protocol.exists():
            protocol_hash = read_json(protocol)["protocol_hash"]
            paths.append(("protocol", protocol))
            paths.append(("raw_merged_data", ROOT / f"artifacts/google_pure_paper_reproduction/synthetic_reproduction/{slug}/{protocol_hash[:16]}/merged.json"))
        paths.extend([
            ("merge_manifest", ROOT / f"artifacts/google_pure_paper_reproduction/manifests/{stem}_validation_merge.json"),
            ("validation", ROOT / f"artifacts/google_pure_paper_reproduction/validation/{stem}_validation.json"),
        ])
    return paths


def build_import_manifest() -> dict[str, Any]:
    missing = [str(path) for _, path in _required_paths() if not path.is_file()]
    if missing:
        raise RuntimeError("V12 immutable import failed closed; missing files: " + ", ".join(missing))
    declared = read_json(ROOT / "artifacts/google_pure_source_exact/direct_sigma_integration/controller_identity.json")
    code_mismatches = [relative for relative, expected in declared["controller_code_files"].items()
                       if file_hash(ROOT / relative) != expected]
    if code_mismatches:
        raise RuntimeError("V12 immutable import failed closed; controller code changed: " +
                           ", ".join(code_mismatches))
    imports = []
    for role, path in _required_paths():
        imports.append({
            "role": role,
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": file_hash(path),
            "bytes": path.stat().st_size,
        })
    controller = read_json(ROOT / "artifacts/google_pure_source_exact/direct_sigma_integration/controller_identity.json")
    integration = read_json(ROOT / "artifacts/google_pure_source_exact/direct_sigma_integration/manifest.json")
    one_hour = read_json(ROOT / "artifacts/google_pure_paper_reproduction/reports/one_hour_validation_manifest.json")
    value = {
        "schema_version": V12_SCHEMA,
        "namespace": "google_pure_v12",
        "controller_mode": controller["controller_mode"],
        "controller_hash": controller["controller_hash"],
        "controller_code_hash": controller["controller_code_hash"],
        "parameterization": controller["parameterization"],
        "integration_manifest_hash": integration["manifest_hash"],
        "one_hour_plan_hash": one_hour["plan_hash"],
        "imports": imports,
        "import_count": len(imports),
        "classifications_at_import": dict(CURRENT_CLASSIFICATIONS),
        **NONFINAL_FIELDS,
    }
    value["manifest_hash"] = canonical_hash(value)
    atomic_json(ARTIFACT_ROOT / "immutable_import_manifest.json", value)
    lines = [
        "# V12 immutable import manifest", "",
        f"Manifest hash: `{value['manifest_hash']}`", "",
        "Every imported file is frozen by SHA-256. Validation fails before analysis if a file is missing or changed.", "",
        "| Role | Path | SHA-256 |", "|---|---|---|",
    ]
    lines.extend(f"| {row['role']} | `{row['path']}` | `{row['sha256']}` |" for row in imports)
    atomic_text(ARTIFACT_ROOT / "immutable_import_manifest.md", "\n".join(lines))
    atomic_json(ARTIFACT_ROOT / "classification_manifest.json", {
        "schema_version": V12_SCHEMA,
        "classifications": CURRENT_CLASSIFICATIONS,
        "basis": "imported one-hour validation evidence before V12 remediation",
        **NONFINAL_FIELDS,
    })
    return value


def validate_import_manifest() -> dict[str, Any]:
    path = ARTIFACT_ROOT / "immutable_import_manifest.json"
    if not path.exists():
        raise RuntimeError("missing V12 immutable import manifest")
    manifest = read_json(path)
    failures = []
    for row in manifest.get("imports", []):
        imported = ROOT / row["path"]
        if not imported.is_file():
            failures.append(f"missing:{row['path']}")
        elif file_hash(imported) != row["sha256"]:
            failures.append(f"hash_mismatch:{row['path']}")
    payload = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if canonical_hash(payload) != manifest.get("manifest_hash"):
        failures.append("manifest_self_hash_mismatch")
    result = {"pass": not failures, "failures": failures, "checked_files": len(manifest.get("imports", [])),
              "manifest_hash": manifest.get("manifest_hash"), **NONFINAL_FIELDS}
    if failures:
        raise RuntimeError("V12 immutable import failed closed: " + ", ".join(failures))
    return result
