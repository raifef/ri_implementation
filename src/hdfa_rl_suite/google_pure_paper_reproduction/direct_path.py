"""Fail-closed bridge from the legacy paper workflow to the amended controller path."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Mapping
from hdfa_rl_suite.google_pure_source_exact.identity import build_direct_sigma_identity, require_direct_sigma_identity
from hdfa_rl_suite.google_pure_source_exact.paper_families.common import amended_family_identities
from hdfa_rl_suite.google_pure_source_exact.source_normalization import (
    IMPLEMENTATION_VERSION, boundary_transform_hash, sensitivity_map_hash_for_family,
)
from hdfa_rl_suite.google_pure_v7.config import repository_root

def expected_identity() -> dict[str, Any]:
    return build_direct_sigma_identity(repository_root())

def integration_manifest() -> dict[str, Any]:
    path=repository_root()/"artifacts/google_pure_source_exact/direct_sigma_integration/manifest.json"
    if not path.exists(): raise RuntimeError("missing tiny direct-sigma integration manifest")
    value=json.loads(path.read_text(encoding="utf-8")); expected=expected_identity()
    require_direct_sigma_identity(value,expected)
    if not value.get("pass") or value.get("final_evidence") or value.get("scientifically_valid"):
        raise RuntimeError("tiny direct-sigma integration did not pass without evidence promotion")
    return value

def protocol_identity_reasons(protocol: Mapping[str, Any]) -> list[str]:
    expected=expected_identity(); reasons=[]
    comparisons={
        "controller_mode":expected["controller_mode"],
        "controller_hash":expected["controller_hash"],
        "controller_code_hash":expected["controller_code_hash"],
        "parameterization":expected["parameterization"],
    }
    for field,wanted in comparisons.items():
        if protocol.get(field)!=wanted: reasons.append(f"{field} mismatch: expected {wanted}, observed {protocol.get(field)}")
    if protocol.get("implementation_version") != IMPLEMENTATION_VERSION:
        reasons.append("V15 implementation version missing from protocol")
    family = str(protocol.get("experiment_family"))
    if protocol.get("sensitivity_map_hash") != sensitivity_map_hash_for_family(family):
        reasons.append("V15 sensitivity map hash mismatch")
    if protocol.get("boundary_transform_hash") != boundary_transform_hash():
        reasons.append("V15 boundary transform hash mismatch")
    try: manifest=integration_manifest()
    except RuntimeError as error: reasons.append(str(error)); manifest=None
    try: expected_plant, expected_graph=amended_family_identities(str(protocol.get("experiment_family")))
    except ValueError as error: reasons.append(str(error))
    else:
        if protocol.get("plant_hash")!=expected_plant: reasons.append("amended family plant contract mismatch")
        if protocol.get("graph_hash")!=expected_graph: reasons.append("amended family graph contract mismatch")
    if manifest is not None and not manifest.get("gates",{}).get("five_policy_decomposition_retained"):
        reasons.append("tiny integration did not retain the five-policy decomposition")
    if manifest is not None and not manifest.get("gates",{}).get("v15_source_boundary_executed"):
        reasons.append("tiny integration did not execute the V15 source boundary")
    return reasons

def require_amended_acquisition(protocol: Mapping[str, Any]) -> None:
    reasons=protocol_identity_reasons(protocol)
    if reasons:
        raise RuntimeError("paper-scale acquisition blocked before execution: "+"; ".join(reasons))
