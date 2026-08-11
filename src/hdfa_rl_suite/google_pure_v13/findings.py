"""Freeze the accepted V12 findings without promoting them."""
from __future__ import annotations

from .contracts import NONFINAL, V12_FINDINGS, V13_SCHEMA
from .io import ARTIFACT_ROOT, atomic_json, atomic_text


def write_v12_findings_contract() -> dict:
    result = {"schema_version": V13_SCHEMA, "findings": dict(V12_FINDINGS),
              "purpose": "frozen input conclusions before V13 calibration", **NONFINAL}
    atomic_json(ARTIFACT_ROOT / "v12_findings_contract.json", result)
    lines = ["# Frozen V12 findings", "", "| Finding | Classification |", "|---|---|"]
    lines.extend(f"| {key} | {value} |" for key, value in V12_FINDINGS.items())
    lines.extend(["", "These are development conclusions and cannot authorize final evidence."])
    atomic_text(ARTIFACT_ROOT / "v12_findings_contract.md", "\n".join(lines))
    return result
