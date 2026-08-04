"""Immutable provenance records for acquisitions, merges, and plots."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from google_rl_reimplementation.google_pure_v7.config import canonical_hash
from google_rl_reimplementation.google_pure_v7.controller import require_resolved_controller

from .experiment_families import evidence_class_for, final_evidence_allowed, require_family


def controller_identity() -> dict[str, str]:
    item = require_resolved_controller()
    return {
        "controller_hash": item["resolved_config_hash"],
        "controller_code_hash": item["controller_code_hash"],
        "controller_mode": item["controller_mode"],
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
) -> dict[str, Any]:
    identity = controller_identity()
    payload: dict[str, Any] = {
        "schema_version": "google-paper-reproduction-provenance.v1",
        "experiment_family": require_family(family),
        "evidence_class": evidence_class_for(family),
        "protocol_hash": protocol_hash,
        **identity,
        "plant_hash": plant_hash,
        "graph_hash": graph_hash,
        "mode": mode,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": bool(complete),
        "scientifically_valid": bool(scientifically_valid),
        "shard_ids": list(shard_ids or []),
        "shard_count": len(shard_ids or []),
        "watermark_required": mode in {"smoke", "validation"},
        "final_evidence": final_evidence_allowed(mode=mode, complete=complete, scientifically_valid=scientifically_valid),
        "pure_google_style_rl_only": True,
        "standalone_reference_workflow": True,
        "certification_seeds_consumed": False,
    }
    payload["provenance_hash"] = canonical_hash(payload)
    return payload

