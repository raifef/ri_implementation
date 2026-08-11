"""Immutable V13 import manifest over controller, V12, and source contracts."""
from __future__ import annotations

from typing import Any

from .contracts import NONFINAL, V13_SCHEMA
from .io import ARTIFACT_ROOT, ROOT, atomic_json, atomic_text, canonical_hash, file_hash, read_json


def required_paths() -> list[tuple[str, str]]:
    controller = read_json(ROOT / "artifacts/google_pure_source_exact/direct_sigma_integration/controller_identity.json")
    rows = [("controller_code", path) for path in controller["controller_code_files"]]
    rows.extend([
        ("controller_identity", "artifacts/google_pure_source_exact/direct_sigma_integration/controller_identity.json"),
        ("controller_checkpoint", "artifacts/google_pure_source_exact/direct_sigma_integration/checkpoint-de2f9061be224d74.json"),
        ("figure5a_source_config", "configs/google_pure_source_exact/figure5a.json"),
        ("normalization_source_contract", "artifacts/google_pure_source_exact/control_normalization/source_contract.json"),
        ("normalization_bundle", "artifacts/google_pure_source_exact/control_normalization/calibration_bundle.json"),
        ("ppo_contract", "src/hdfa_rl_suite/google_pure_source_exact/policy_parameterization/contracts.py"),
        ("ppo_loss", "src/hdfa_rl_suite/google_pure_source_exact/policy_parameterization/losses.py"),
        ("ppo_optimizer", "src/hdfa_rl_suite/google_pure_source_exact/policy_parameterization/optimizer.py"),
        ("v12_import_manifest", "artifacts/google_pure_v12/immutable_import_manifest.json"),
        ("v12_sensitivity_repair", "src/hdfa_rl_suite/google_pure_v12/directional.py"),
        ("v12_step_spoil_comparison", "artifacts/google_pure_v12/directional_comparison/comparison.json"),
        ("v12_gradient_snr", "artifacts/google_pure_v12/audits/gradient_snr.json"),
        ("v12_update_efficiency", "artifacts/google_pure_v12/audits/update_efficiency.json"),
        ("v12_figure5b_trajectories", "artifacts/google_pure_v12/lineage/figure5b_lineage.json"),
        ("v12_figure5c_finite_differences", "artifacts/google_pure_v12/lineage/figure5c_lineage.json"),
        ("v12_natural_spectra", "artifacts/google_pure_v12/spectral/natural_drift_uncertainty.json"),
        ("v12_status", "artifacts/google_pure_v12/status.json"),
    ])
    return rows


def build_import_manifest() -> dict[str, Any]:
    missing = [path for _, path in required_paths() if not (ROOT / path).is_file()]
    if missing:
        raise RuntimeError("V13 import failed closed; missing: " + ", ".join(missing))
    controller = read_json(ROOT / "artifacts/google_pure_source_exact/direct_sigma_integration/controller_identity.json")
    changed = [path for path, expected in controller["controller_code_files"].items()
               if file_hash(ROOT / path) != expected]
    if changed:
        raise RuntimeError("V13 import failed closed; frozen controller changed: " + ", ".join(changed))
    imports = [{"role": role, "path": path, "sha256": file_hash(ROOT / path),
                "bytes": (ROOT / path).stat().st_size} for role, path in required_paths()]
    result = {"schema_version": V13_SCHEMA, "controller_mode": controller["controller_mode"],
              "controller_hash": controller["controller_hash"],
              "controller_code_hash": controller["controller_code_hash"],
              "parameterization": controller["parameterization"], "imports": imports,
              "import_count": len(imports), **NONFINAL}
    result["manifest_hash"] = canonical_hash(result)
    atomic_json(ARTIFACT_ROOT / "import_manifest.json", result)
    lines = ["# V13 immutable import manifest", "", f"Manifest: `{result['manifest_hash']}`", "",
             "| Role | Path | SHA-256 |", "|---|---|---|"]
    lines.extend(f"| {row['role']} | `{row['path']}` | `{row['sha256']}` |" for row in imports)
    atomic_text(ARTIFACT_ROOT / "import_manifest.md", "\n".join(lines))
    return result


def verify_import_manifest() -> dict[str, Any]:
    manifest = read_json(ARTIFACT_ROOT / "import_manifest.json")
    failures = []
    for row in manifest["imports"]:
        path = ROOT / row["path"]
        if not path.is_file(): failures.append(f"missing:{row['path']}")
        elif file_hash(path) != row["sha256"]: failures.append(f"changed:{row['path']}")
    payload = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if canonical_hash(payload) != manifest["manifest_hash"]: failures.append("manifest_self_hash")
    if failures: raise RuntimeError("V13 import failed closed: " + ", ".join(failures))
    return {"pass": True, "checked_files": len(manifest["imports"]),
            "manifest_hash": manifest["manifest_hash"], **NONFINAL}
