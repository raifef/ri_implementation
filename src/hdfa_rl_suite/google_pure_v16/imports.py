"""Fail-closed import of the frozen V15/V12/source evidence used by V16."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import NONFINAL
from .io import ARTIFACT_ROOT, ROOT, atomic_json, atomic_text, canonical_hash, file_hash, read_json


REQUIRED_IMPORTS: tuple[tuple[str, str], ...] = (
    ("v15_frozen_execution_contract",
     "artifacts/google_pure_v15/immediate_execution_audit/frozen_execution_contract.json"),
    ("v15_calibration_bundle",
     "artifacts/google_pure_source_exact/control_normalization/calibration_bundle.json"),
    ("v15_sensitivity_definition_audit",
     "artifacts/google_pure_v15/sensitivity/source_definition_audit.json"),
    ("v15_detector_degree_audit",
     "artifacts/google_pure_v15/sensitivity/detector_degree_audit.json"),
    ("v15_source_normalized_boundary",
     "src/hdfa_rl_suite/google_pure_source_exact/source_normalization.py"),
    ("v15_direct_sigma_controller",
     "src/hdfa_rl_suite/google_pure_source_exact/policy_parameterization/optimizer.py"),
    ("v15_step_abc",
     "artifacts/google_pure_v15/immediate_execution_audit/abc_step/comparison.json"),
    ("v15_figure5b_abc",
     "artifacts/google_pure_v15/immediate_execution_audit/abc_figure5b/comparison.json"),
    ("v12_outcome_derived_development_map",
     "artifacts/google_pure_v12/directional_comparison/comparison.json"),
    ("source_optimizer_evidence",
     "artifacts/google_pure_source_exact/policy_parameterization/source_contract.json"),
    ("source_optimizer_unspecified_choices",
     "configs/google_rl/source_unspecified_choices.yaml"),
    ("source_entropy_regime_evidence",
     "artifacts/google_pure_source_exact/figure5a/source_contract.json"),
)


def _semantic_checks(role: str, path: Path) -> dict[str, bool]:
    if path.suffix not in {".json"}:
        return {"exists": True, "nonempty": path.stat().st_size > 0}
    value = read_json(path)
    checks: dict[str, bool] = {"exists": True, "nonempty": bool(value)}
    if role == "v15_frozen_execution_contract":
        checks.update({
            "v15_implementation": value.get("implementation_version") == "google_pure_v15",
            "figure5c_frozen": value.get("figure5c_fit_frozen") is True,
            "nonfinal": value.get("final_evidence") is False,
        })
    elif role == "v15_calibration_bundle":
        checks.update({
            "artifact_complete": value.get("artifact_complete") is True,
            "mathematical_contract_pass": value.get("mathematical_contract_pass") is True,
        })
    elif role == "v15_sensitivity_definition_audit":
        checks["audit_pass"] = value.get("pass") is True
    elif role == "v15_detector_degree_audit":
        checks["audit_pass"] = value.get("pass") is True
    elif role in {"v15_step_abc", "v15_figure5b_abc"}:
        checks.update({"nonfinal": value.get("final_evidence") is False,
                       "has_rows": bool(value.get("rows"))})
    elif role == "source_optimizer_evidence":
        checks["direct_sigma_source_contract"] = (
            value.get("paper_parameterization") == "DIRECT_SIGMA_SOURCE_EXACT")
    elif role == "source_entropy_regime_evidence":
        fields = {item.get("field"): item for item in value.get("fields", [])}
        checks["entropy_anchors_present"] = fields.get("entropy_anchors", {}).get("value") == [0.001, 0.01, 0.1]
    return checks


def build_import_manifest() -> dict[str, Any]:
    rows = []
    for role, relative in REQUIRED_IMPORTS:
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"missing mandatory V16 import: {relative}")
        checks = _semantic_checks(role, path)
        if not all(checks.values()):
            raise RuntimeError(f"mandatory V16 import failed semantic checks: {role}: {checks}")
        rows.append({"role": role, "path": relative, "sha256": file_hash(path),
                     "semantic_checks": checks})
    payload = {
        "schema_version": "google-pure-v16-import-manifest.v1",
        "imports": rows,
        "import_count": len(rows),
        "all_imports_valid": True,
        "lineage": ["V12", "V13", "V15", "V16"],
        "v14_exists": False,
        **NONFINAL,
    }
    payload["import_manifest_hash"] = canonical_hash(
        {key: value for key, value in payload.items() if key != "import_manifest_hash"})
    atomic_json(ARTIFACT_ROOT / "import_manifest.json", payload)
    lines = ["# V16 immutable import manifest", "",
             "All required V15, V12-development, and public-source inputs passed fail-closed checks.", "",
             "| Role | Path | SHA-256 |", "|---|---|---|"]
    lines.extend(f"| {row['role']} | `{row['path']}` | `{row['sha256']}` |" for row in rows)
    atomic_text(ARTIFACT_ROOT / "import_manifest.md", "\n".join(lines))
    return payload


def verify_import_manifest() -> dict[str, Any]:
    path = ARTIFACT_ROOT / "import_manifest.json"
    if not path.is_file():
        raise RuntimeError("V16 import manifest is absent; build it before running diagnostics")
    value = read_json(path)
    failures = []
    for row in value.get("imports", []):
        target = ROOT / row["path"]
        if not target.is_file() or file_hash(target) != row["sha256"]:
            failures.append(row["path"])
        elif not all(_semantic_checks(row["role"], target).values()):
            failures.append(row["path"] + ":semantic")
    expected = canonical_hash({key: item for key, item in value.items()
                               if key != "import_manifest_hash"})
    if expected != value.get("import_manifest_hash"):
        failures.append("manifest:self-hash")
    if failures:
        raise RuntimeError("V16 frozen imports changed or failed closed: " + ", ".join(failures))
    return value
