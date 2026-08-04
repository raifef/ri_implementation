"""Family-specific scientific gates and smoke/final evidence separation."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .experiment_families import ExperimentFamily, final_evidence_allowed, require_family
from .storage import atomic_json, atomic_text, initialise_layout, load_merged


def _module(family: str):
    value = require_family(family)
    if value == ExperimentFamily.FIGURE5A_REAL_TIME_STEERING.value: from . import panel_a as module
    elif value == ExperimentFamily.FIGURE5B_SPARSE_SCALING.value: from . import panel_b as module
    elif value == ExperimentFamily.FIGURE5C_CONVERGENCE_LAW.value: from . import panel_c as module
    elif value == ExperimentFamily.NATURAL_DRIFT_SPECTRAL_SUPPRESSION.value: from . import natural_drift as module
    elif value == ExperimentFamily.RANDOMIZED_RECOVERY_AFTER_SPOIL.value: from . import randomized_recovery as module
    elif value == ExperimentFamily.STEP_RESPONSE_INJECTED_DRIFT.value: from . import step_response as module
    else: raise ValueError(f"no synthetic validator for {value}")
    return module


def validate_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    merged = load_merged(protocol); rows = merged["rows"]; reasons: list[str] = []
    if merged["experiment_family"] != protocol["experiment_family"]: reasons.append("wrong experiment family")
    if merged["mode"] != protocol["mode"]: reasons.append("mode mismatch")
    if not merged["complete"]: reasons.append("incomplete shard set")
    numeric = [value for row in rows for value in row.values() if isinstance(value, (int, float))]
    if numeric and not np.all(np.isfinite(numeric)): reasons.append("non-finite scalar")
    family_valid, family_reasons, metrics = _module(protocol["experiment_family"]).validation(rows, protocol["mode"])
    reasons.extend(family_reasons); valid = not reasons and family_valid
    final = final_evidence_allowed(mode=protocol["mode"], complete=merged["complete"], scientifically_valid=valid)
    if protocol["mode"] == "smoke": status = "SMOKE_RENDER_ONLY"
    elif protocol["mode"] == "validation": status = "VALIDATION_ONLY" if valid else "SCIENTIFIC_VALIDATION_FAILED"
    else: status = "REFERENCE_EVIDENCE" if final else "SCIENTIFIC_VALIDATION_FAILED"
    result = {"schema_version": "google-paper-validation.v1", "experiment_family": protocol["experiment_family"],
              "mode": protocol["mode"], "protocol_hash": protocol["protocol_hash"], "valid": valid,
              "complete": merged["complete"], "final_evidence": final, "status": status,
              "blocking_reasons": reasons, "metrics": metrics, "smoke_cannot_certify": True}
    root = initialise_layout() / "validation"; stem = f"{protocol['experiment_family'].lower()}_{protocol['mode']}"
    atomic_json(root / f"{stem}.json", result)
    atomic_text(root / f"{stem}.md", f"# {protocol['experiment_family']} validation\n\nStatus: **{status}**\n\n" + ("\n".join(f"- {reason}" for reason in reasons) or "No family-specific blocking issue.\n"))
    return result


def load_validation(protocol: Mapping[str, Any]) -> dict[str, Any]:
    path = initialise_layout() / "validation" / f"{protocol['experiment_family'].lower()}_{protocol['mode']}.json"
    if not path.exists(): raise RuntimeError("missing scientific validation")
    value = __import__("json").loads(path.read_text(encoding="utf-8"))
    if value["protocol_hash"] != protocol["protocol_hash"]: raise RuntimeError("stale validation artifact")
    return value

