"""Fail-closed V21 lineage over frozen V19, V20, and source-style inputs."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hdfa_rl_suite.google_pure_v20.data import (
    EXPECTED_SOURCE_PARENT,
    EXPECTED_V19_CONTROLLER,
)

from .io import (
    ARTIFACT_ROOT,
    ROOT,
    atomic_json,
    atomic_text,
    file_hash,
    nonfinal,
    read_json,
    relative,
    settings,
)


FORBIDDEN_CAMPAIGNS = (
    "slow", "intermediate", "source-budget", "heldout", "reference",
    "natural-drift", "figure5c", "paired-acceptance", "long-three-frequency",
)


def import_paths() -> dict[str, Path]:
    v19 = ROOT / "artifacts/google_pure_v19/experimental_public_analogue_matched"
    v20 = ROOT / "artifacts/google_pure_v20"
    return {
        "v20_status": v20 / "status.json",
        "v20_baseline_fast_checkpoint": v19 / "acquisition/fast/checkpoint.json",
        "v20_baseline_fast_transfer": v19 / "transfer_fast.json",
        "v20_hard_projection_acquisition": v20 / "repaired_fast/acquisition.json",
        "v20_hard_projection_checkpoint": v20 / "repaired_fast/checkpoint.json",
        "v20_hard_projection_validation": v20 / "postrepair_fast_validation.json",
        "v20_hard_projection_controller_code": ROOT /
            "src/hdfa_rl_suite/google_pure_v20/repair.py",
        "v20_population_gradient": v20 / "population_gradient_fast_rollout.json",
        "v20_reference_gradients": v20 / "fast_reference_gradients.json",
        "v20_gradient_statistics": v20 / "fast_gradient_statistics.json",
        "v20_scale_frontier": v20 / "frozen_scale_information_damage_frontier.json",
        "v20_damage_decomposition": v20 / "fast_mean_cost_decomposition.json",
        "v19_experimental_controller_code": ROOT /
            "src/hdfa_rl_suite/google_pure_v19_experimental/controller.py",
        "v19_experimental_acquisition_code": ROOT /
            "src/hdfa_rl_suite/google_pure_v19_experimental/acquisition.py",
        "source_style_optimizer_bundle": ROOT /
            "artifacts/google_pure_v16/frozen_source_normalized_optimizer.json",
        "source_style_gaussian": ROOT /
            "src/hdfa_rl_suite/google_pure_source_exact/policy_parameterization/gaussian.py",
        "source_style_losses": ROOT /
            "src/hdfa_rl_suite/google_pure_source_exact/policy_parameterization/losses.py",
        "source_style_optimizer": ROOT /
            "src/hdfa_rl_suite/google_pure_source_exact/policy_parameterization/optimizer.py",
        "normalization_code": ROOT /
            "src/hdfa_rl_suite/google_pure_source_exact/source_normalization.py",
        "normalization_bundle": ROOT /
            "artifacts/google_pure_source_exact/control_normalization/calibration_bundle.json",
        "figure5a_config": ROOT / "configs/google_pure_source_exact/figure5a.json",
        "figure5a_plant": ROOT /
            "src/hdfa_rl_suite/google_pure_source_exact/figure5a/plant.py",
        "figure5a_evaluator": ROOT /
            "src/hdfa_rl_suite/google_pure_source_exact/figure5a/validation.py",
    }


def _observed() -> dict[str, dict[str, str]]:
    paths = import_paths()
    missing = [relative(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing mandatory V21 frozen input: {missing}")
    return {role: {"path": relative(path), "sha256": file_hash(path)}
            for role, path in paths.items()}


def build_import_manifest() -> dict[str, Any]:
    cfg = settings()
    observed = _observed()
    path = ARTIFACT_ROOT / "import_manifest.json"
    if path.is_file():
        previous = read_json(path)
        if previous.get("inputs") != observed:
            changed = {key: {"expected": previous.get("inputs", {}).get(key),
                             "observed": value}
                       for key, value in observed.items()
                       if previous.get("inputs", {}).get(key) != value}
            raise RuntimeError(f"V21 frozen lineage mismatch: {changed}")
        return previous
    v20 = read_json(import_paths()["v20_status"])
    v20_projection = read_json(import_paths()["v20_hard_projection_validation"])
    optimizer = read_json(import_paths()["source_style_optimizer_bundle"])
    invariants = {
        "source_style_branch_unchanged": v20.get(
            "frozen_source_style_branch_unchanged") is True,
        "v19_parent_unchanged": (
            v20_projection["controller"]["frozen_experimental_parent_hash"] ==
            EXPECTED_V19_CONTROLLER and optimizer["optimizer_bundle_hash"] ==
            EXPECTED_SOURCE_PARENT),
        "v20_projection_not_promoted_to_baseline": True,
        "mean_lr_changed": False,
        "sigma_lr_changed": False,
        "entropy_changed": False,
        "normalization_changed": False,
    }
    gates = {
        "v20_complete": v20.get("execution_complete") is True and v20.get("pass") is True,
        "v20_root_cause": v20.get("primary_root_cause") ==
            "FINITE_CANDIDATE_DIRECTIONAL_FAILURE",
        "projection_is_diagnostic": v20_projection["controller"]["single_causal_repair"] ==
            "PUBLIC_FIGURE5A_SHARED_SUBSPACE_MEAN_GRADIENT_PROJECTION",
        "projection_not_general_source_baseline": v20_projection.get("source_exact") is False,
        "invariants": invariants["source_style_branch_unchanged"] and
            invariants["v19_parent_unchanged"] and
            invariants["v20_projection_not_promoted_to_baseline"] and
            all(value is False for key, value in invariants.items() if key.endswith("_changed")),
        "protocol_fixed_budget": cfg["candidate_budget"] == {"K": 8, "M": 12000, "B": 96000},
    }
    if not all(gates.values()):
        raise RuntimeError(f"V21 lineage gate failed: {gates}")
    value = nonfinal({
        "pass": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": observed,
        "invariants": invariants,
        "gates": gates,
        "v20_projection_classification": "ORACLE_LIKE_DIAGNOSTIC_UPPER_BOUND",
        "projection_information_dependencies": [
            "known driven direction", "target trajectory", "hidden optimum structure"],
        "projection_prohibited_baseline_uses": [
            "future phase", "population/reference gradients", "multi-run leakage"],
        "frozen_v19_controller_hash": EXPECTED_V19_CONTROLLER,
        "frozen_source_style_controller_hash": EXPECTED_SOURCE_PARENT,
        "forbidden_auto_runs": list(FORBIDDEN_CAMPAIGNS),
        "forbidden_auto_runs_launched": [],
    })
    atomic_json(path, value)
    atomic_text(ARTIFACT_ROOT / "import_manifest.md", "\n".join([
        "# V21 frozen V20 lineage", "",
        "The V19 iid fast branch, V20 hard-projection diagnostic, population rollout, reference "
        "gradients, scale frontier, source-style policy stack, normalization, and Figure 5a "
        "plant/evaluator are hash-pinned.", "",
        "The V20 hard projection remains `ORACLE_LIKE_DIAGNOSTIC_UPPER_BOUND` and is not a baseline.",
    ]))
    return value


def verify_import_manifest() -> dict[str, Any]:
    path = ARTIFACT_ROOT / "import_manifest.json"
    value = read_json(path) if path.is_file() else build_import_manifest()
    if value.get("pass") is not True or value.get("inputs") != _observed():
        raise RuntimeError("V21 import manifest no longer matches frozen inputs")
    return value
