"""Fail-closed bridge from the legacy paper workflow to the amended controller path."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Mapping
from hdfa_rl_suite.google_pure_source_exact.identity import build_direct_sigma_identity, require_direct_sigma_identity
from hdfa_rl_suite.google_pure_source_exact.figure5a.acquisition import (
    COORDINATE_CONTRACT, FIGURE5A_IMPLEMENTATION_VERSION,
)
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
    family = str(protocol.get("experiment_family"))
    if family == "FIGURE5A_REAL_TIME_STEERING":
        source_fields = {
            "implementation_version": FIGURE5A_IMPLEMENTATION_VERSION,
            "coordinate_contract": COORDINATE_CONTRACT,
            "action_execution": "identity_applied_gaussian",
            "plant_boundary_execution": "none_source_coordinate_identity",
            "likelihood_space": "applied_gaussian",
            "entropy_space": "applied_gaussian",
            "empirical_relative_normalization_applied": False,
            "mean_bounds_applied": False,
        }
        for field, wanted in source_fields.items():
            if protocol.get(field) != wanted:
                reasons.append(f"Figure 5a source-coordinate {field} mismatch")
    else:
        if protocol.get("implementation_version") != IMPLEMENTATION_VERSION:
            reasons.append("V15 implementation version missing from protocol")
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
    if manifest is not None and not any(manifest.get("gates",{}).get(name) for name in (
            "source_gaussian_coordinate_identity_executed",
            "figure5a_empirical_boundary_executed", "v15_source_boundary_executed")):
        reasons.append("tiny integration did not execute the validated source-coordinate policy path")
    return reasons

def require_amended_acquisition(protocol: Mapping[str, Any]) -> None:
    reasons=protocol_identity_reasons(protocol)
    if reasons:
        raise RuntimeError("paper-scale acquisition blocked before execution: "+"; ".join(reasons))
