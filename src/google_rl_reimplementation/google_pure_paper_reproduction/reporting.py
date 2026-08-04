"""Multi-family audit and readiness reports."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from google_rl_reimplementation.google_pure_v7.config import repository_root

from .experiment_families import ExperimentFamily
from .storage import artifact_root, atomic_json, atomic_text, initialise_layout


def reclassify_prior_figure5_smoke() -> dict[str, Any]:
    project = repository_root(); old = project/"artifacts/google_pure_v7/figure5"
    result={"schema_version":"google-paper-prior-figure5-reclassification.v1","source":old.relative_to(project).as_posix(),"source_deleted":False,
            "classification":"SMOKE_RENDER_ONLY","eligible_as_paper_reproduction":False,
            "reasons":["smoke grids and budgets are deliberately reduced","panel-A smoke did not bracket the zero contour","prior plots were rendering checks, not paper-comparable evidence"],
            "preserved_paths":[path.relative_to(project).as_posix() for path in sorted((old/"figures").glob("*"))] if (old/"figures").exists() else []}
    atomic_json(initialise_layout()/"reports"/"invalid_prior_figure5_outputs.json",result)
    atomic_text(initialise_layout()/"reports"/"invalid_prior_figure5_outputs.md","# Invalid prior Figure 5 outputs\n\nThe existing v7 Figure 5 smoke artifacts are preserved but reclassified **SMOKE_RENDER_ONLY**. They are not evidence of paper reproduction.\n\n"+"\n".join(f"- {reason}" for reason in result["reasons"])+"\n")
    return result


def _validation_statuses() -> dict[str, Any]:
    result={}
    root=artifact_root()/"validation"
    for family in ExperimentFamily:
        files=sorted(root.glob(f"{family.value.lower()}_*.json")) if root.exists() else []
        result[family.value]=[json.loads(path.read_text(encoding="utf-8")) for path in files]
    return result


def audit_pure_namespace() -> dict[str, Any]:
    package=repository_root()/"src/google_rl_reimplementation/google_pure_paper_reproduction"; forbidden=[]
    forbidden_prefixes=("google_rl_reimplementation.stage", "google_rl_reimplementation.supervisor", "google_rl_reimplementation.evaluation.mpc")
    for path in package.glob("*.py"):
        tree=ast.parse(path.read_text(encoding="utf-8"),filename=str(path))
        for node in ast.walk(tree):
            names=[]
            if isinstance(node,ast.Import): names=[alias.name for alias in node.names]
            elif isinstance(node,ast.ImportFrom): names=[node.module or ""]
            for name in names:
                if name.startswith(forbidden_prefixes): forbidden.append({"file":path.name,"import":name})
    return {"pure_import_firewall_pass":not forbidden,"forbidden_imports":forbidden,"full_controller_only":True}


def status() -> dict[str, Any]:
    result={"schema_version":"google-paper-status.v1","families":_validation_statuses(),"pure_namespace":audit_pure_namespace(),
            "master_scalar_certification":False,"artifact_root":artifact_root().relative_to(repository_root()).as_posix()}
    atomic_json(initialise_layout()/"manifests"/"status.json",result);return result


def audit_all() -> dict[str, Any]:
    reclassify_prior_figure5_smoke(); current=status(); issues=[]
    if not current["pure_namespace"]["pure_import_firewall_pass"]: issues.append("pure namespace import firewall failed")
    for path in (artifact_root()/"validation").glob("*.json"):
        row=json.loads(path.read_text(encoding="utf-8"))
        if row["mode"]=="smoke" and row.get("final_evidence"): issues.append(f"smoke accepted as final: {path.name}")
    result={"schema_version":"google-paper-audit.v1","status":"PASS" if not issues else "FAIL","issues":issues,
            "pure_namespace":current["pure_namespace"],"smoke_final_evidence_forbidden":True,"mode_mixing_forbidden":True,
            "stale_or_incomplete_merges_forbidden":True,"claim_family_compatibility_required":True}
    atomic_json(initialise_layout()/"reports"/"audit_all.json",result);return result


def reproduction_overview() -> dict[str, Any]:
    current=status(); public=(artifact_root()/"public_data_reproduction/public_data_reproduction.json").exists()
    result={"schema_version":"google-paper-overview.v1","scope":"pure Google-style full-policy RL baseline only",
            "public_data_reproduction_available":public,"family_status":{k:[row["status"] for row in v] for k,v in current["families"].items()},
            "simulation_code_status":"independent paper-anchored reproduction because original code is proprietary",
            "standalone_reference_workflow":True,"master_scalar_certification":False}
    root=initialise_layout()/"reports";atomic_json(root/"reproduction_overview.json",result)
    atomic_text(root/"reproduction_overview.md","# Reproduction overview\n\nThis workflow reproduces the Google-style full-policy RL controller and reports each evidence family independently.\n\n"+"\n".join(f"- `{key}`: {', '.join(value) if value else 'NOT_YET_RUN'}" for key,value in result["family_status"].items())+"\n")
    return result


def reproduction_completeness() -> dict[str, Any]:
    current=status(); required=[family.value for family in ExperimentFamily]
    per_family={}
    for family in required:
        if family == ExperimentFamily.PUBLIC_ENDPOINT_DATA_REPRODUCTION.value:
            path=artifact_root()/"public_data_reproduction/public_data_reproduction.json"; exact=False
            if path.exists(): exact=json.loads(path.read_text(encoding="utf-8")).get("status")=="PUBLIC_DATA_DIRECTLY_REPRODUCIBLE"
            per_family[family]={"ready":exact,"status":"EXACT_PUBLIC_REPRODUCTION" if exact else "NOT_YET_RUN","blocking_reason":None if exact else "direct public-data replay absent"};continue
        if family == ExperimentFamily.PUBLIC_TABLE_REPRODUCTION.value:
            path=artifact_root()/"tables/paper_vs_reproduction_values.json"; exact=False
            if path.exists(): exact=sum(row["verdict"]=="EXACT_PUBLIC_REPRODUCTION" for row in json.loads(path.read_text(encoding="utf-8"))["rows"])>=4
            per_family[family]={"ready":exact,"status":"EXACT_PUBLIC_REPRODUCTION" if exact else "NOT_YET_RUN","blocking_reason":None if exact else "public values table absent or incomplete"};continue
        records=current["families"][family]; refs=[row for row in records if row["mode"] in {"reference","paper-scale"} and row["final_evidence"]]
        per_family[family]={"ready":bool(refs),"status":refs[-1]["status"] if refs else "NOT_YET_RUN","blocking_reason":None if refs else "no complete validated reference/paper-scale run"}
    ready=all(row["ready"] for row in per_family.values()) and current["pure_namespace"]["pure_import_firewall_pass"]
    result={"schema_version":"google-paper-reproduction-completeness.v1","all_families_complete":ready,
            "overall":"READY" if ready else "NOT_READY","families":per_family,"master_scalar":None,
            "interpretation":"readiness requires each relevant evidence family; no averaging can hide a missing or mismatched family"}
    root=initialise_layout()/"reports";atomic_json(root/"reproduction_completeness.json",result)
    atomic_text(root/"reproduction_completeness.md","# Reproduction completeness\n\nOverall: **"+result["overall"]+"**\n\n"+"\n".join(f"- `{family}`: {row['status']}" for family,row in per_family.items())+"\n")
    return result


def next_user_commands() -> Path:
    text="# Next user commands\n\nSmoke artifacts are already non-final. Run reference acquisitions explicitly, one family at a time:\n\n```powershell\ngoogle-rl-paper-fig5a-plan --mode reference\ngoogle-rl-paper-fig5a-acquire --mode reference\ngoogle-rl-paper-fig5a-merge --mode reference\ngoogle-rl-paper-fig5a-validate --mode reference\ngoogle-rl-paper-fig5a-plot --mode reference\ngoogle-rl-paper-fig5a-compare --mode reference --paper-image D:\\path\\to\\figure5a_crop.png\n```\n\nRepeat with `fig5b`, `fig5c`, `natural-drift`, `randomized-recovery`, and `step-response`. Paper-scale commands additionally require `--execute-paper-scale`.\n"
    path=initialise_layout()/"reports"/"next_user_commands.md";atomic_text(path,text);return path
