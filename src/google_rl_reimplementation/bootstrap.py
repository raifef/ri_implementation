"""Materialize compact provenance fixtures and run fresh smoke prerequisites."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .google_pure_v7.config import repository_root
from .google_pure_v7.controller import resolve_production_controller


HISTORICAL_FILES = (
    "google_pure_v5/injected_drift_stability.json",
    "google_pure_v5/natural_drift_spectral.json",
    "google_pure_v5/randomized_recovery.json",
    "google_pure_v5/convergence_scaling.json",
    "google_pure_v6/development_scorecard.json",
    "google_pure_v6/v5_immutable_snapshot.json",
    "google_pure_v6/natural_drift_retention.json",
    "google_pure_v6/scaling_retention.json",
    "google_pure_v6/recovery_retention.json",
    "google_pure_v6/certification_preregistration.json",
    "google_pure_paper_reproduction/public_data_reproduction/public_data_reproduction.json",
    "google_pure_paper_reproduction/validation/figure5a_real_time_steering_reference.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def materialize_historical_inputs() -> dict[str, Any]:
    """Copy small retained inputs without overwriting user-generated results."""
    root = repository_root()
    fixture_root = root / "fixtures" / "historical_artifacts"
    rows = []
    for relative in HISTORICAL_FILES:
        source = fixture_root / relative
        target = root / "artifacts" / relative
        if not source.is_file():
            raise RuntimeError(f"missing bundled historical input: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        copied = not target.exists()
        if copied:
            shutil.copyfile(source, target)
        rows.append({
            "relative_path": relative,
            "fixture_sha256": _sha256(source),
            "active_sha256": _sha256(target),
            "copied": copied,
            "retained_historical_input": True,
        })
    manifest = {
        "schema_version": "google-rl-standalone-bootstrap.v1",
        "historical_inputs": rows,
        "historical_inputs_are_new_executions": False,
        "historical_inputs_are_final_claim_evidence": False,
    }
    destination = root / "artifacts" / "standalone_bootstrap_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def run_compact_bootstrap() -> dict[str, Any]:
    """Build every compact prerequisite without launching reference-scale work."""
    history = materialize_historical_inputs()
    controller = resolve_production_controller()

    from .google_pure_v8.audits import (
        audit_baselines,
        audit_clipping_likelihood,
        audit_entropy_scale,
        audit_exploration_floor,
        audit_native_units,
        audit_ppo_lifecycle,
        audit_temporal_protocol,
    )
    from .google_pure_v8.contracts import build_mathematical_contracts
    from .google_pure_v8.diagnostics import run_edr_identity_audit, run_figure5a_feasibility
    from .google_pure_v8.matrix import run_compact_fault_matrix
    from .google_pure_v8.reporting import root_cause_report, status as root_status
    from .google_pure_v8.snapshot import snapshot

    root_steps = (
        snapshot,
        build_mathematical_contracts,
        run_edr_identity_audit,
        run_figure5a_feasibility,
        audit_exploration_floor,
        audit_entropy_scale,
        audit_native_units,
        audit_clipping_likelihood,
        audit_ppo_lifecycle,
        audit_baselines,
        audit_temporal_protocol,
        run_compact_fault_matrix,
        root_cause_report,
        root_status,
    )
    root_results = {step.__name__: step() for step in root_steps}

    from .google_pure_evidence_v8.claim_registry import build_claim_registry
    from .google_pure_evidence_v8.evidence_contracts import build_gate_contract
    from .google_pure_evidence_v8.experiment_families import build_contract
    from .google_pure_evidence_v8.figure5b import run_figure5b
    from .google_pure_evidence_v8.figure5c import run_figure5c
    from .google_pure_evidence_v8.manifest_validation import validate_manifests
    from .google_pure_evidence_v8.natural_drift import run_natural_drift
    from .google_pure_evidence_v8.paper_comparison import build_paper_comparison
    from .google_pure_evidence_v8.recovery import run_recovery
    from .google_pure_evidence_v8.step_response import run_step_response

    evidence_steps = (
        build_contract,
        build_gate_contract,
        run_natural_drift,
        run_step_response,
        run_recovery,
        run_figure5b,
        run_figure5c,
        build_claim_registry,
        build_paper_comparison,
        validate_manifests,
    )
    evidence_results = {step.__name__: step() for step in evidence_steps}
    return {
        "schema_version": "google-rl-compact-bootstrap-result.v1",
        "historical_input_count": len(history["historical_inputs"]),
        "controller_hash": controller["resolved_config_hash"],
        "root_cause_gate_pass": root_results["root_cause_report"]["prompt1_gate_pass"],
        "protocol_gate_pass": evidence_results["validate_manifests"]["status"] == "PASS",
        "reference_scale_executed": False,
        "certification_seeds_consumed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="google-rl-bootstrap")
    parser.add_argument("--materialize-only", action="store_true")
    args = parser.parse_args()
    result = materialize_historical_inputs() if args.materialize_only else run_compact_bootstrap()
    print(json.dumps(result, indent=2, sort_keys=True))
