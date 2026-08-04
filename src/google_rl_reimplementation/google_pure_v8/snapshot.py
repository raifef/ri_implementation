"""Immutable reconstruction of the v7 state before v8 diagnostics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from google_rl_reimplementation.google_pure_v7.config import repository_root, sha256_file
from google_rl_reimplementation.google_pure_v7.controller import require_resolved_controller
from google_rl_reimplementation.google_pure_v7.figure5.seed_registry import validate_registry

from .common import root, write_report


def _hash_files(paths: list[Path]) -> dict[str,str]:
    base=repository_root();return {path.relative_to(base).as_posix():sha256_file(path) for path in paths if path.is_file()}


def snapshot() -> dict[str,Any]:
    base=repository_root(); controller=require_resolved_controller()
    source=list((base/"src/google_rl_reimplementation/google_pure_v6").rglob("*.py"))+list((base/"src/google_rl_reimplementation/google_pure_v7").rglob("*.py"))
    configs=list((base/"configs/google_pure_v6").rglob("*"))+list((base/"configs/google_pure_v7").rglob("*"))
    tests=list((base/"tests").glob("test_google_pure_v6*.py"))+list((base/"tests").glob("test_google_pure_v7*.py"))
    reference=base/"artifacts/google_pure_paper_reproduction/validation/figure5a_real_time_steering_reference.json"
    prior=json.loads(reference.read_text(encoding="utf-8")) if reference.exists() else None
    artifact_candidates=[base/"artifacts/google_pure_v7/resolved_production_controller.json",reference]
    artifact_candidates+=list((base/"artifacts/google_pure_v7/figure5/protocol_freezes").glob("*.json"))
    artifact_candidates+=list((base/"artifacts/google_pure_paper_reproduction/validation").glob("*.json"))
    artifact_hashes=_hash_files(artifact_candidates)
    registry=validate_registry(); protected_ok=not registry["forbidden_used"] and not registry["overlaps"]
    family_status={}
    for path in sorted((base/"artifacts/google_pure_paper_reproduction/validation").glob("*.json")):
        value=json.loads(path.read_text(encoding="utf-8"));family_status[path.stem]={"status":value.get("status"),"valid":value.get("valid"),"mode":value.get("mode"),"artifact_hash":sha256_file(path)}
    missing=[str(path.relative_to(base)) for path in [base/"artifacts/google_pure_v7/resolved_production_controller.json",reference] if not path.is_file()]
    result={"schema_version":"google-pure-v8-pre-repair-snapshot.v1","source_hashes":_hash_files(source),
      "config_hashes":_hash_files(configs),"test_hashes":_hash_files(tests),"resolved_controller":controller,
      "artifact_hashes":artifact_hashes,"prior_experiment_family_status":family_status,
      "controller_hash":controller["resolved_config_hash"],"controller_code_hash":controller["controller_code_hash"],
      "prior_figure5a_reference":prior,"prior_figure5a_reconstructed":prior is not None,
      "snapshot_scope":"recursive v6/v7 source, configs and tests plus controller/protocol/validation artifacts",
      "existing_artifacts_modified":False,"active_certification_seeds_unused":protected_ok,"retired_seed_10101_preserved":10101 in registry["blacklist"],
      "missing_required_inputs":missing,"status":"PASS" if not missing and protected_ok else "FAIL_CLOSED"}
    written=write_report("pre_repair_snapshot",result,"Pre-repair Immutable Snapshot")
    if written["status"]!="PASS":raise RuntimeError(f"pre-repair state cannot be reconstructed: {missing}")
    return written
