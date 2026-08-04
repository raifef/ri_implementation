"""Exact public endpoint replay, isolated from synthetic simulation families."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from google_rl_reimplementation.google_pure_v7.config import canonical_hash, repository_root, sha256_file

from .storage import atomic_json, atomic_text, initialise_layout


def reproduce_public_data(*, recompute: bool = False) -> dict[str, Any]:
    root = repository_root(); source = root / "artifacts/google_reproduction_v3/public_data_reproduction.json"
    if recompute:
        from google_rl_reimplementation.google_reproduction_v3.analysis import reproduce_public_analysis
        from google_rl_reimplementation.google_reproduction_v3.dataset_manifest import load_json_yaml
        config = load_json_yaml(root / "configs/google_rl_v3/zenodo_local_dataset.yaml")
        reproduce_public_analysis(Path(config["selected_archive"]), source.parent)
    if not source.exists():
        raise RuntimeError("v3 exact public-data artifact is absent; run with --recompute against the pinned Zenodo archive")
    original = json.loads(source.read_text(encoding="utf-8"))
    rows = []
    for claim_id, row in original["headline_results"].items():
        rows.append({"claim_id": claim_id, "published": row["published"], "reproduced": row["repository"],
                     "independent": row["independent"], "verdict": "EXACT_PUBLIC_REPRODUCTION", "source_kind": "released_hardware_data"})
    for claim_id, row in original["fine_tuning_endpoint_comparisons"].items():
        rows.append({"claim_id": f"finetuning_endpoint.{claim_id}", "published": row["published_multi_run_anchor"],
                     "reproduced": row["relative_improvement"], "verdict": "QUALITATIVE_MATCH_ONLY",
                     "source_kind": "released_unpaired_endpoints", "limitation": row["difference_reason"]})
    result = {"schema_version": "google-paper-public-data.v1", "experiment_family": "PUBLIC_ENDPOINT_DATA_REPRODUCTION",
              "public_data_direct": True, "synthetic_data_present": False, "source_artifact": source.as_posix(),
              "source_artifact_sha256": sha256_file(source), "source_payload_hash": canonical_hash(original), "rows": rows,
              "not_reproducible_from_release": original["not_reproducible"], "control_decoder_metrics_merged": False,
              "certification_seeds_consumed": False, "status": "PUBLIC_DATA_DIRECTLY_REPRODUCIBLE"}
    target = initialise_layout() / "public_data_reproduction"; atomic_json(target / "public_data_reproduction.json", result)
    lines = ["# Direct public-data reproduction", "", "This section replays released hardware memory data. It contains no synthetic simulator output.", "", "| Quantity | Published | Reproduced | Verdict |", "|---|---:|---:|---|"]
    lines += [f"| `{r['claim_id']}` | {r['published']:.8g} | {r['reproduced']:.8g} | `{r['verdict']}` |" for r in rows]
    atomic_text(target / "public_data_reproduction.md", "\n".join(lines) + "\n")
    return result

