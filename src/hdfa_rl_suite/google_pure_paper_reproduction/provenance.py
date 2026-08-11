"""Immutable provenance records for acquisitions, merges, and plots."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from hdfa_rl_suite.google_pure_v7.config import canonical_hash

from .experiment_families import evidence_class_for, final_evidence_allowed, require_family
from .direct_path import expected_identity
from hdfa_rl_suite.google_pure_source_exact.source_normalization import (
    IMPLEMENTATION_VERSION,
    require_v15_boundary_provenance,
)


def controller_identity() -> dict[str, str]:
    item = expected_identity()
    return {
        "controller_hash": item["controller_hash"],
        "controller_code_hash": item["controller_code_hash"],
        "controller_mode": item["controller_mode"],
        "parameterization": item["parameterization"],
    }


def make_provenance(
    family: str,
    *,
    protocol_hash: str,
    mode: str,
    plant_hash: str,
    graph_hash: str,
    complete: bool = False,
    scientifically_valid: bool = False,
    shard_ids: list[str] | None = None,
    v15_provenance: Mapping[str, Any] | None = None,
    experiment_driver_hash: str = "",
    fresh_acquisition: bool = False,
    reused_shard_ids: list[str] | None = None,
    source_budget_profile: str = "UNSPECIFIED_DEVELOPMENT",
) -> dict[str, Any]:
    identity = controller_identity()
    expected = expected_identity()
    if v15_provenance is None:
        raise RuntimeError("paper acquisition provenance requires the executed V15 boundary record")
    boundary = dict(v15_provenance)
    require_v15_boundary_provenance(boundary)
    payload: dict[str, Any] = {
        "schema_version": "google-paper-reproduction-provenance.v2",
        "experiment_family": require_family(family),
        "evidence_class": evidence_class_for(family),
        "protocol_hash": protocol_hash,
        **identity,
        "expected_controller_mode": expected["controller_mode"],
        "expected_controller_hash": expected["controller_hash"],
        "expected_controller_code_hash": expected["controller_code_hash"],
        "expected_parameterization": expected["parameterization"],
        "direct_sigma_identity_match": identity["controller_mode"] == expected["controller_mode"] and
            identity["controller_hash"] == expected["controller_hash"] and
            identity["controller_code_hash"] == expected["controller_code_hash"] and
            identity["parameterization"] == expected["parameterization"],
        "plant_hash": plant_hash,
        "graph_hash": graph_hash,
        "implementation_version": IMPLEMENTATION_VERSION,
        "experiment_driver_hash": str(experiment_driver_hash),
        "source_budget_profile": str(source_budget_profile),
        "fresh_acquisition": bool(fresh_acquisition),
        "reused_shard_ids": list(reused_shard_ids or []),
        **{key: boundary[key] for key in (
            "sensitivity_map_hash", "sensitivity_definition_hash",
            "calibration_bundle_hash", "detector_degree_audit_hash",
            "boundary_transform_hash", "boundary_transform_name",
            "boundary_apply_count", "control_order_hash", "expanded_scale_hash")},
        "mode": mode,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": bool(complete),
        "scientifically_valid": bool(scientifically_valid),
        "shard_ids": list(shard_ids or []),
        "shard_count": len(shard_ids or []),
        "watermark_required": mode in {"smoke", "validation"},
        "final_evidence": final_evidence_allowed(mode=mode, complete=complete, scientifically_valid=scientifically_valid),
        "pure_google_style_rl_only": True,
        "staged_controller_run": False,
        "certification_seeds_consumed": False,
    }
    payload["provenance_hash"] = canonical_hash(payload)
    return payload
