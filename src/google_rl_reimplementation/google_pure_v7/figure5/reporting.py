"""Status and concise panel/all-pipeline reports."""
from __future__ import annotations
import json
from .common import PIPELINE_STATUSES, atomic_json, atomic_text, figure5_root

def status()->dict:
    root=figure5_root();panels={}
    for panel in ("5a","5b","5c"):
        merge=root/"merged"/panel/"merge_manifest.json";validation=root/"reports"/f"panel_{panel[-1]}_validation.json";plot=root/"figures"/f"panel_{panel[-1]}_plot_manifest.json"
        if plot.exists():state="PLOT_COMPLETE"
        elif validation.exists():
            v=json.loads(validation.read_text(encoding="utf-8"));state="READY_TO_PLOT" if v["valid"] else "SCIENTIFIC_VALIDATION_FAILED"
        elif merge.exists():state=json.loads(merge.read_text(encoding="utf-8"))["status"]
        elif (root/"shards"/panel).exists():state="ACQUISITION_PARTIAL"
        elif (root/"protocol_freezes").exists():state="PIPELINE_BUILT_UNVALIDATED"
        else:state="PIPELINE_NOT_BUILT"
        if state not in PIPELINE_STATUSES:raise RuntimeError(state)
        panels[panel]=state
    result={"schema_version":"google-pure-v7-figure5-status.v1","panels":panels,"certification_seeds_consumed":False}
    atomic_json(root/"manifests"/"status.json",result);return result

def report_panel(panel:str)->dict:
    current=status();result={"panel":panel,"status":current["panels"][panel],"claim":"paper-anchored synthetic reproduction; no hardware or proprietary-simulator equivalence claim"}
    atomic_json(figure5_root()/"reports"/f"panel_{panel[-1]}_report.json",result)
    atomic_text(figure5_root()/"reports"/f"panel_{panel[-1]}_report.md",f"# Panel {panel} Report\n\nStatus: **{result['status']}**\n\n{result['claim']}.\n")
    return result

def report_all()->dict:
    panels={p:report_panel(p) for p in ("5a","5b","5c")}
    source_path=figure5_root()/"source_contract"/"figure5_source_contract.json"
    source=json.loads(source_path.read_text(encoding="utf-8")) if source_path.exists() else {"fields":[]}
    unresolved=[row["field"] for row in source["fields"] if row["status"] in ("NOT_PUBLICLY_SPECIFIED","SYNTHETIC_REPRODUCTION_CHOICE")]
    result={"schema_version":"google-pure-v7-figure5-report.v1","panels":panels,"status":status(),
      "paper_scale_panel_a_cost":{"effective_qec_cycles":1_800_000_000,"epochs":1000,"candidates_per_epoch":50,"cycles_per_candidate":36000},
      "unresolved_or_synthetic_source_fields":unresolved,"verification_scope":"smoke acquisition only; reference and paper-scale not executed",
      "next_commands":["google-rl-v7-fig5a-acquire --config panel_a_reference.yaml --mode paper-scale --execute-paper-scale",
        "google-rl-v7-fig5b-acquire --config panel_b_reference.yaml --mode reference",
        "google-rl-v7-fig5c-acquire --config panel_c_reference.yaml --mode reference"]}
    atomic_json(figure5_root()/"reports"/"figure5_report.json",result)
    lines=["# Figure 5 Pipeline Report","","All results retain source-status and synthetic-reproduction labels.","",
      "## Verification scope","","Only the bounded smoke acquisitions were executed automatically; reference and paper-scale runs were not.","",
      "## Paper-scale panel-a cost","","1,000 epochs × 50 candidates × 36,000 cycles = **1.8 billion effective QEC cycles per condition**.","",
      "## Unresolved or synthetic source fields",""]+[f"- {field}" for field in unresolved]+["","## Next commands",""]+[f"- `{command}`" for command in result["next_commands"]]
    atomic_text(figure5_root()/"reports"/"figure5_report.md","\n".join(lines)+"\n")
    from google_rl_reimplementation.google_pure_v7.config import sha256_file
    manifest_path=figure5_root()/"manifests"/"figure5_manifest.json"
    artifacts=[]
    for path in sorted(figure5_root().rglob("*")):
        if path.is_file() and path!=manifest_path:
            artifacts.append({"path":path.relative_to(figure5_root()).as_posix(),"bytes":path.stat().st_size,"sha256":sha256_file(path)})
    atomic_json(manifest_path,{"schema_version":"google-pure-v7-figure5-manifest.v1","artifacts":artifacts,"artifact_count":len(artifacts),"certification_seeds_consumed":False})
    return result
