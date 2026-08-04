"""Paper-versus-reproduction values table with claim-specific verdicts."""
from __future__ import annotations

import json
from typing import Any

from .claim_registry import claims
from .storage import artifact_root, atomic_json, atomic_text, initialise_layout

VERDICTS = {"EXACT_PUBLIC_REPRODUCTION", "WITHIN_TOLERANCE_SYNTHETIC_MATCH", "QUALITATIVE_MATCH_ONLY", "MISMATCH", "NOT_PUBLICLY_IDENTIFIABLE", "NOT_YET_RUN"}


def build_values_table() -> dict[str, Any]:
    public_path = artifact_root() / "public_data_reproduction/public_data_reproduction.json"
    public_rows = {}
    if public_path.exists():
        public_rows = {row["claim_id"]: row for row in json.loads(public_path.read_text(encoding="utf-8"))["rows"]}
    output = []
    aliases = {"headline.surface_d7_alphaqubit2": "surface_code_distance_7_alphaqubit2", "headline.color_d5_tesseract": "color_code_distance_5_tesseract_frequency_prior",
               "table.surface_d7_sparse_blossom": "surface_code_distance_7_sparse_blossom_si1000", "table.color_d5_tesseract_si1000": "color_code_distance_5_tesseract_si1000"}
    for claim in claims():
        row = public_rows.get(aliases.get(claim["claim_id"], claim["claim_id"]))
        if row:
            verdict, reproduced = row["verdict"], row["reproduced"]
        elif claim["status"] == "NOT_IDENTIFIABLE_FROM_PUBLIC_INFORMATION":
            verdict, reproduced = "NOT_PUBLICLY_IDENTIFIABLE", None
        else:
            verdict, reproduced = "NOT_YET_RUN", None
        output.append({"claim_id": claim["claim_id"], "paper_value": claim["public_value"], "reproduction_value": reproduced,
                       "unit": claim["unit"], "experiment_families": claim["experiment_families"], "verdict": verdict,
                       "source_location": claim["source_location"]})
    result = {"schema_version": "google-paper-values-table.v1", "rows": output, "allowed_verdicts": sorted(VERDICTS), "master_scalar": None}
    root = initialise_layout() / "tables"; atomic_json(root / "paper_vs_reproduction_values.json", result)
    lines = ["# Paper versus reproduction values", "", "| Claim | Paper | Reproduction | Verdict |", "|---|---:|---:|---|"]
    for row in output:
        lines.append(f"| `{row['claim_id']}` | {row['paper_value'] if row['paper_value'] is not None else '—'} | {row['reproduction_value'] if row['reproduction_value'] is not None else '—'} | `{row['verdict']}` |")
    atomic_text(root / "paper_vs_reproduction_values.md", "\n".join(lines) + "\n")
    return result

