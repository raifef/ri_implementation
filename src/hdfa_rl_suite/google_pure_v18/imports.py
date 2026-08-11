"""Fail-closed imports for the V18 staged identification campaign."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import NONFINAL
from .io import ARTIFACT_ROOT, ROOT, atomic_json, atomic_text, canonical_hash, file_hash, read_json


REQUIRED_IMPORTS: tuple[tuple[str, str], ...] = (
    ("v16_frozen_optimizer", "artifacts/google_pure_v16/frozen_source_normalized_optimizer.json"),
    ("v17_import_manifest", "artifacts/google_pure_v17/import_manifest.json"),
    ("v17_sensitivity_semantics", "artifacts/google_pure_v17/sensitivity_semantics_audit.json"),
    ("v17_step_transfer", "artifacts/google_pure_v17/step_transfer_identification.json"),
    ("v17_deterministic_fixture", "artifacts/google_pure_v17/figure5a_deterministic_fixture.json"),
    ("v17_delta_min_config", "configs/google_pure_v17/protocol.json"),
    ("v17_window_audit", "artifacts/google_pure_v17/figure5a_window_aliasing.json"),
    ("v17_reduced_acceptance", "artifacts/google_pure_v17/reduced_acceptance_v2.json"),
    ("v17_estimators", "src/hdfa_rl_suite/google_pure_v17/estimators.py"),
    ("figure5a_source_config", "configs/google_pure_source_exact/figure5a.json"),
    ("figure5a_target_plant", "src/hdfa_rl_suite/google_pure_source_exact/figure5a/plant.py"),
    ("figure5a_production_acquisition", "src/hdfa_rl_suite/google_pure_source_exact/figure5a/acquisition.py"),
    ("figure5a_metric_contract", "src/hdfa_rl_suite/google_pure_source_exact/figure5a/contracts.py"),
    ("source_normalized_boundary", "src/hdfa_rl_suite/google_pure_source_exact/source_normalization.py"),
    ("direct_sigma_optimizer", "src/hdfa_rl_suite/google_pure_source_exact/policy_parameterization/optimizer.py"),
    ("elementwise_ppo_loss", "src/hdfa_rl_suite/google_pure_source_exact/policy_parameterization/losses.py"),
    ("figure5b_result", "artifacts/google_pure_v16/matched_figure5b/comparison.json"),
    ("figure5b_driver", "src/hdfa_rl_suite/google_pure_v16/experiments.py"),
)


def _semantic_checks(role: str, path: Path) -> dict[str, bool]:
    if path.suffix != ".json":
        text = path.read_text(encoding="utf-8")
        checks = {"exists": True, "nonempty": bool(text)}
        if role == "figure5a_target_plant":
            checks["source_frequency_formula"] = "2.0 * math.pi" in text
        elif role == "figure5a_production_acquisition":
            checks.update({"direct_sigma": '"controller_mode": "PAPER_DIRECT_SIGMA"' in text,
                           "four_streams": 'STREAMS = ("fixed", "optimal", "stochastic", "learned_mean")' in text})
        elif role == "figure5a_metric_contract":
            checks["production_ratio"] = "def ratio_from_raw_counts" in text
        elif role == "source_normalized_boundary":
            checks["variance_damage_target"] = "KAPPA_EDR_FRACTION = 0.01" in text
        elif role == "direct_sigma_optimizer":
            checks["optimizer"] = "class DirectSigmaOptimizer" in text
        elif role == "elementwise_ppo_loss":
            checks["coordinate_clipping"] = "coordinate" in text.lower() and "clip" in text.lower()
        elif role == "figure5b_driver":
            checks["per_epoch_definition"] = '"fractional_residual_reduction"' in text and "lambda_next" in text
        return checks
    value = read_json(path)
    checks: dict[str, bool] = {"exists": True, "nonempty": bool(value)}
    if role == "v16_frozen_optimizer":
        checks.update({"frozen": value.get("frozen_for_matched_causal_validation") is True,
                       "hash": bool(value.get("optimizer_bundle_hash")),
                       "mean_lr": value.get("mean_learning_rate") == .32,
                       "sigma_lr": value.get("sigma_learning_rate") == .08,
                       "initial_sigma": value.get("initial_sigma") == .15,
                       "entropy": value.get("entropy_coefficient") == .01})
    elif role == "v17_import_manifest":
        checks["valid"] = value.get("all_imports_valid") is True
    elif role == "v17_sensitivity_semantics":
        checks.update({"pass": value.get("pass") is True,
                       "variance_semantics": value.get("classification") == "SOURCE_0P01_IS_VARIANCE_DAMAGE"})
    elif role == "v17_step_transfer":
        checks.update({"pass": value.get("pass") is True,
                       "has_fit": bool(value.get("free_gain_delay_tau"))})
    elif role == "v17_deterministic_fixture":
        checks["qualitative_pass"] = value.get("pass") is True
    elif role == "v17_window_audit":
        checks["old_gate_invalid"] = value.get("original_reduced_classification") == "UNDERPOWERED_REDUCED_GATE"
    elif role == "v17_reduced_acceptance":
        checks.update({"blocked": value.get("pass") is False,
                       "zero_complete_pairs": value.get("paired_complete_unit_count") == 0})
    elif role == "figure5a_source_config":
        checks.update({"distance3": value.get("plant", {}).get("distance") == 3,
                       "source_budget": value.get("profiles", {}).get("reference", {}).get("epochs") == 1000})
    elif role == "figure5b_result":
        checks["has_v16_summary"] = "D_V16_FROZEN_OPTIMIZER" in value.get("summaries", {})
    return checks


def build_import_manifest() -> dict[str, Any]:
    rows = []
    for role, relative in REQUIRED_IMPORTS:
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"missing mandatory V18 import: {relative}")
        checks = _semantic_checks(role, path)
        if not all(checks.values()):
            raise RuntimeError(f"mandatory V18 import failed semantic checks: {role}: {checks}")
        rows.append({"role": role, "path": relative, "sha256": file_hash(path),
                     "semantic_checks": checks})
    payload = {"schema_version": "google-pure-v18-import-manifest.v1", "imports": rows,
               "import_count": len(rows), "all_imports_valid": True,
               "v16_optimizer_immutable": True, "lineage": ["V13", "V15", "V16", "V17", "V18"],
               "v14_exists": False, **NONFINAL}
    payload["import_manifest_hash"] = canonical_hash(
        {key: value for key, value in payload.items() if key != "import_manifest_hash"})
    atomic_json(ARTIFACT_ROOT / "import_manifest.json", payload)
    lines = ["# V18 fail-closed import manifest", "", "| Role | Path | SHA-256 |", "|---|---|---|"]
    lines.extend(f"| {row['role']} | `{row['path']}` | `{row['sha256']}` |" for row in rows)
    atomic_text(ARTIFACT_ROOT / "import_manifest.md", "\n".join(lines))
    return payload


def verify_import_manifest() -> dict[str, Any]:
    path = ARTIFACT_ROOT / "import_manifest.json"
    if not path.is_file():
        return build_import_manifest()
    value = read_json(path)
    failures = []
    observed = [(row.get("role"), row.get("path")) for row in value.get("imports", [])]
    if observed != list(REQUIRED_IMPORTS):
        failures.append("manifest:required-import-set")
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
        raise RuntimeError("V18 frozen imports changed or failed closed: " + ", ".join(failures))
    return value
