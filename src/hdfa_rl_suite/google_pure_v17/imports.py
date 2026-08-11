"""Fail-closed V17 imports of V16, V15, and production Figure 5a lineage."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import NONFINAL
from .io import ARTIFACT_ROOT, ROOT, atomic_json, atomic_text, canonical_hash, file_hash, read_json


REQUIRED_IMPORTS: tuple[tuple[str, str], ...] = (
    ("v16_frozen_optimizer", "artifacts/google_pure_v16/frozen_source_normalized_optimizer.json"),
    ("v16_source_normalized_boundary", "src/hdfa_rl_suite/google_pure_source_exact/source_normalization.py"),
    ("v16_sensitivity_definition", "artifacts/google_pure_v15/sensitivity/source_definition_audit.json"),
    ("v16_matched_step", "artifacts/google_pure_v16/matched_step/comparison.json"),
    ("v16_matched_step_driver", "src/hdfa_rl_suite/google_pure_v16/experiments.py"),
    ("v16_reduced_figure5a", "artifacts/google_pure_v16/reduced_acceptance/result.json"),
    ("v16_entropy_audit", "artifacts/google_pure_v16/source_entropy_anchors.json"),
    ("v16_direct_sigma_audit", "artifacts/google_pure_v16/direct_sigma_dynamics.json"),
    ("v16_local_contraction", "artifacts/google_pure_v16/local_contraction_audit.json"),
    ("v16_matched_figure5b", "artifacts/google_pure_v16/matched_figure5b/comparison.json"),
    ("figure5a_source_contract", "artifacts/google_pure_source_exact/figure5a/source_contract.json"),
    ("figure5a_source_config", "configs/google_pure_source_exact/figure5a.json"),
    ("figure5a_target_plant", "src/hdfa_rl_suite/google_pure_source_exact/figure5a/plant.py"),
    ("figure5a_production_evaluator", "src/hdfa_rl_suite/google_pure_source_exact/figure5a/acquisition.py"),
    ("figure5a_metric_contract", "src/hdfa_rl_suite/google_pure_source_exact/figure5a/contracts.py"),
    ("direct_sigma_controller", "src/hdfa_rl_suite/google_pure_source_exact/policy_parameterization/optimizer.py"),
    ("elementwise_ppo_loss", "src/hdfa_rl_suite/google_pure_source_exact/policy_parameterization/losses.py"),
)


def _semantic_checks(role: str, path: Path) -> dict[str, bool]:
    if path.suffix != ".json":
        text = path.read_text(encoding="utf-8")
        checks = {"exists": True, "nonempty": bool(text)}
        if role == "figure5a_target_plant":
            checks.update({"cycles_formula": "2.0 * math.pi" in text,
                           "forty_one_coordinates": "np.full(41" in text})
        elif role == "figure5a_production_evaluator":
            checks.update({"four_streams": 'STREAMS = ("fixed", "optimal", "stochastic", "learned_mean")' in text,
                           "production_ratio": "ratio_from_raw_counts" in text})
        elif role == "figure5a_metric_contract":
            checks["source_ratio_formula"] = "(int(stochastic) - int(fixed)) / denominator" in text
        elif role == "v16_source_normalized_boundary":
            checks["variance_damage_target"] = "KAPPA_EDR_FRACTION = 0.01" in text
        elif role == "v16_matched_step_driver":
            checks.update({"step_seed_runner": "def _run_step_seed" in text,
                           "matched_step_runner": "def run_matched_step" in text})
        elif role == "direct_sigma_controller":
            checks["direct_sigma"] = "DirectSigmaOptimizer" in text
        elif role == "elementwise_ppo_loss":
            checks["coordinate_clipping"] = "coordinate" in text.lower() and "clip" in text.lower()
        return checks
    value = read_json(path)
    checks: dict[str, bool] = {"exists": True, "nonempty": bool(value)}
    if role == "v16_frozen_optimizer":
        checks.update({
            "frozen": value.get("frozen_for_matched_causal_validation") is True,
            "direct_sigma": value.get("parameterization") == "DIRECT_SIGMA_SOURCE_EXACT",
            "mean_lr": value.get("mean_learning_rate") == 0.32,
            "sigma_lr": value.get("sigma_learning_rate") == 0.08,
            "initial_sigma": value.get("initial_sigma") == 0.15,
            "entropy": value.get("entropy_coefficient") == 0.01,
            "nonfinal": value.get("final_evidence") is False,
        })
    elif role == "v16_sensitivity_definition":
        checks.update({"pass": value.get("pass") is True,
                       "variance_semantics": all(row.get("curvature_hessian") == 2 * row.get("a_pp")
                                                 for row in value.get("calibration_rows", []))})
    elif role == "v16_matched_step":
        checks["v16_branch"] = "D_V16_FROZEN_OPTIMIZER" in value.get("summaries", {})
    elif role == "v16_reduced_figure5a":
        checks.update({"has_figure5a": bool(value.get("figure5a", {}).get("rows")),
                       "nonfinal": value.get("final_evidence") is False})
    elif role in {"v16_entropy_audit", "v16_direct_sigma_audit", "v16_local_contraction",
                  "v16_matched_figure5b"}:
        checks.update({"pass": value.get("pass") is True,
                       "nonfinal": value.get("final_evidence") is False})
    elif role == "figure5a_source_contract":
        fields = {row.get("field"): row.get("value") for row in value.get("fields", [])}
        checks.update({"target": fields.get("shared_optimum") == "sin(2*pi*f*t) in every coordinate",
                       "metric": fields.get("performance_ratio") == "(N_stochastic-N_fixed)/(N_optimal-N_fixed)"})
    elif role == "figure5a_source_config":
        checks.update({"controls": value.get("plant", {}).get("distance") == 3,
                       "source_epochs": value.get("profiles", {}).get("reference", {}).get("epochs") == 1000})
    return checks


def build_import_manifest() -> dict[str, Any]:
    rows = []
    for role, relative in REQUIRED_IMPORTS:
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"missing mandatory V17 import: {relative}")
        checks = _semantic_checks(role, path)
        if not all(checks.values()):
            raise RuntimeError(f"mandatory V17 import failed semantic checks: {role}: {checks}")
        rows.append({"role": role, "path": relative, "sha256": file_hash(path),
                     "semantic_checks": checks})
    payload = {
        "schema_version": "google-pure-v17-import-manifest.v1",
        "imports": rows,
        "import_count": len(rows),
        "all_imports_valid": True,
        "lineage": ["V13", "V15", "V16", "V17"],
        "v14_exists": False,
        "v16_optimizer_immutable": True,
        **NONFINAL,
    }
    payload["import_manifest_hash"] = canonical_hash(
        {key: value for key, value in payload.items() if key != "import_manifest_hash"})
    atomic_json(ARTIFACT_ROOT / "import_manifest.json", payload)
    lines = ["# V17 immutable import manifest", "",
             "All frozen V16 and production Figure 5a inputs passed fail-closed checks.", "",
             "| Role | Path | SHA-256 |", "|---|---|---|"]
    lines.extend(f"| {row['role']} | `{row['path']}` | `{row['sha256']}` |" for row in rows)
    atomic_text(ARTIFACT_ROOT / "import_manifest.md", "\n".join(lines))
    return payload


def verify_import_manifest() -> dict[str, Any]:
    path = ARTIFACT_ROOT / "import_manifest.json"
    if not path.is_file():
        return build_import_manifest()
    value = read_json(path)
    failures = []
    observed_set = [(row.get("role"), row.get("path")) for row in value.get("imports", [])]
    if observed_set != list(REQUIRED_IMPORTS):
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
        raise RuntimeError("V17 frozen imports changed or failed closed: " + ", ".join(failures))
    return value
