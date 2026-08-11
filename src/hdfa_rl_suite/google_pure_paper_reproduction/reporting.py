"""Multi-family audit and readiness reports."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any
import numpy as np

from hdfa_rl_suite.google_pure_v7.config import repository_root

from .experiment_families import ExperimentFamily
from .storage import artifact_root, atomic_json, atomic_text, initialise_layout

def reclassify_legacy_final_artifacts() -> dict[str, Any]:
    """Recompute endpoint gates and permanently demote old-controller final-mode artifacts."""
    from hdfa_rl_suite.google_pure_source_exact.identity import build_direct_sigma_identity
    from hdfa_rl_suite.google_pure_v7.response import estimate_step_response
    from . import natural_drift, panel_a, panel_b, panel_c, randomized_recovery, step_response
    root=artifact_root(); expected=build_direct_sigma_identity(repository_root())
    modules={
        "FIGURE5A_REAL_TIME_STEERING":("fig5a",panel_a),"FIGURE5B_SPARSE_SCALING":("fig5b",panel_b),
        "FIGURE5C_CONVERGENCE_LAW":("fig5c",panel_c),"NATURAL_DRIFT_SPECTRAL_SUPPRESSION":("natural",natural_drift),
        "RANDOMIZED_RECOVERY_AFTER_SPOIL":("recovery",randomized_recovery),"STEP_RESPONSE_INJECTED_DRIFT":("step",step_response)}
    changed=[]
    for path in sorted((root/"validation").glob("*.json")):
        value=json.loads(path.read_text(encoding="utf-8"))
        if value.get("mode") not in {"reference","paper-scale"} or value.get("experiment_family") not in modules: continue
        slug,module=modules[value["experiment_family"]]; merged_path=root/"synthetic_reproduction"/slug/value["protocol_hash"][:16]/"merged.json"
        if not merged_path.exists(): continue
        merged=json.loads(merged_path.read_text(encoding="utf-8")); rows=merged["rows"]
        provenance=merged.get("provenance",{})
        if provenance.get("controller_mode")==expected["controller_mode"] and \
                provenance.get("controller_hash")==expected["controller_hash"] and \
                provenance.get("controller_code_hash")==expected["controller_code_hash"] and \
                provenance.get("parameterization")=="direct_sigma":
            continue
        if value["experiment_family"]=="STEP_RESPONSE_INJECTED_DRIFT":
            repaired=[]
            for row in rows:
                copy=dict(row); trace=np.asarray(row["trajectory"]["normalized_projected_policy_response"],dtype=float)
                copy["response"]=estimate_step_response(trace,onset_epoch=int(row["onset_epoch"]),target=1.0,
                    sustained_epochs=min(25,max(3,len(trace)//10))); repaired.append(copy)
            _,family_reasons,metrics=module.validation(repaired,value["mode"])
        else:
            _,family_reasons,metrics=module.validation(rows,value["mode"])
        actual_mode=provenance.get("controller_mode", "UNKNOWN_MISSING_PROVENANCE")
        actual_hash=provenance.get("controller_hash"); actual_code_hash=provenance.get("controller_code_hash")
        reasons=list(dict.fromkeys([*family_reasons,
            f"controller_mode mismatch: expected {expected['controller_mode']}, observed {actual_mode}",
            f"controller_hash mismatch: expected {expected['controller_hash']}, observed {actual_hash}",
            f"controller_code_hash mismatch: expected {expected['controller_code_hash']}, observed {actual_code_hash}",
            f"parameterization mismatch: expected direct_sigma, observed {provenance.get('parameterization', 'UNKNOWN_MISSING_PROVENANCE')}",
            "legacy synthetic acquisition cannot inherit amended-path validity"] ))
        status="RECOVERY_NOT_REACHED_WITHIN_HORIZON" if metrics.get("outcome")=="RECOVERY_NOT_REACHED_WITHIN_HORIZON" else (
            "STEP_TARGET_NOT_REACHED_WITHIN_HORIZON" if metrics.get("outcome")=="STEP_TARGET_NOT_REACHED_WITHIN_HORIZON" else "LEGACY_DIAGNOSTIC_ONLY")
        repaired={**value,"valid":False,"scientifically_valid":False,"final_evidence":False,"status":status,
            "blocking_reasons":reasons,"metrics":metrics,"controller_mode":actual_mode,"controller_hash":actual_hash,
            "controller_code_hash":actual_code_hash,"parameterization":provenance.get("parameterization", "UNKNOWN_MISSING_PROVENANCE"),
            "expected_controller_mode":expected["controller_mode"],"expected_controller_hash":expected["controller_hash"],
            "expected_controller_code_hash":expected["controller_code_hash"],"expected_parameterization":"direct_sigma",
            "plant_hash":provenance.get("plant_hash"),"graph_hash":provenance.get("graph_hash"),
            "provenance_inherited_without_promotion":True,"supersedes_invalid_promotion":True}
        atomic_json(path,repaired); atomic_text(path.with_suffix(".md"),f"# {value['experiment_family']} validation\n\nStatus: **{status}**\n\n"+"\n".join(f"- {reason}" for reason in reasons)+"\n")
        changed.append(str(path))
    result={"schema_version":"legacy-final-artifact-reclassification.v1","changed":changed,"count":len(changed),
        "final_evidence_for_legacy_controller":False,"expected_controller_hash":expected["controller_hash"]}
    atomic_json(root/"reports"/"legacy_final_artifact_reclassification.json",result); return result


def reclassify_prior_figure5_smoke() -> dict[str, Any]:
    old = repository_root()/"artifacts/google_pure_v7/figure5"
    result={"schema_version":"google-paper-prior-figure5-reclassification.v1","source":str(old),"source_deleted":False,
            "classification":"SMOKE_RENDER_ONLY","eligible_as_paper_reproduction":False,
            "reasons":["smoke grids and budgets are deliberately reduced","panel-A smoke did not bracket the zero contour","prior plots were rendering checks, not paper-comparable evidence"],
            "preserved_paths":[str(path) for path in sorted((old/"figures").glob("*"))] if (old/"figures").exists() else []}
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
    package=repository_root()/"src/hdfa_rl_suite/google_pure_paper_reproduction"; forbidden=[]
    forbidden_prefixes=("hdfa_rl_suite.stage", "hdfa_rl_suite.supervisor", "hdfa_rl_suite.evaluation.mpc")
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
            "master_scalar_certification":False,"artifact_root":str(artifact_root())}
    atomic_json(initialise_layout()/"manifests"/"status.json",result);return result


def audit_all() -> dict[str, Any]:
    reclassify_prior_figure5_smoke(); reclassify_legacy_final_artifacts(); current=status(); issues=[]
    if not current["pure_namespace"]["pure_import_firewall_pass"]: issues.append("pure namespace import firewall failed")
    for path in (artifact_root()/"validation").glob("*.json"):
        row=json.loads(path.read_text(encoding="utf-8"))
        if row["mode"]=="smoke" and row.get("final_evidence"): issues.append(f"smoke accepted as final: {path.name}")
        if row.get("final_evidence") and row.get("controller_mode") != "PAPER_DIRECT_SIGMA":
            issues.append(f"legacy controller accepted as final: {path.name}")
        if row.get("final_evidence") and row.get("metrics",{}).get("censored_count",0):
            issues.append(f"censored endpoint accepted as final: {path.name}")
    result={"schema_version":"google-paper-audit.v1","status":"PASS" if not issues else "FAIL","issues":issues,
            "pure_namespace":current["pure_namespace"],"smoke_final_evidence_forbidden":True,"mode_mixing_forbidden":True,
            "stale_or_incomplete_merges_forbidden":True,"claim_family_compatibility_required":True}
    atomic_json(initialise_layout()/"reports"/"audit_all.json",result);return result


def reproduction_overview() -> dict[str, Any]:
    current=status(); public=(artifact_root()/"public_data_reproduction/public_data_reproduction.json").exists()
    result={"schema_version":"google-paper-overview.v1","scope":"pure Google-style full-policy RL baseline only",
            "public_data_reproduction_available":public,"family_status":{k:[row["status"] for row in v] for k,v in current["families"].items()},
            "simulation_code_status":"independent paper-anchored reproduction because original code is proprietary",
            "no_hdfa_or_staged_controller_run":True,"master_scalar_certification":False}
    root=initialise_layout()/"reports";atomic_json(root/"reproduction_overview.json",result)
    atomic_text(root/"reproduction_overview.md","# Reproduction overview\n\nThis workflow freezes a pure Google-style full-policy RL comparator. It does not run a staged, predictive, residual, or HDFA controller.\n\n"+"\n".join(f"- `{key}`: {', '.join(value) if value else 'NOT_YET_RUN'}" for key,value in result["family_status"].items())+"\n")
    return result


def baseline_readiness() -> dict[str, Any]:
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
    result={"schema_version":"google-paper-baseline-readiness.v1","ready_for_later_hdfa_comparison":ready,
            "overall":"READY" if ready else "NOT_READY","families":per_family,"master_scalar":None,
            "interpretation":"readiness requires each relevant evidence family; no averaging can hide a missing or mismatched family"}
    root=initialise_layout()/"reports";atomic_json(root/"baseline_readiness_for_hdfa.json",result)
    atomic_text(root/"baseline_readiness_for_hdfa.md","# Baseline readiness for later comparison\n\nOverall: **"+result["overall"]+"**\n\n"+"\n".join(f"- `{family}`: {row['status']}" for family,row in per_family.items())+"\n")
    return result


def next_user_commands() -> Path:
    text="# Next user commands\n\nSmoke artifacts are already non-final. Run reference acquisitions explicitly, one family at a time:\n\n```powershell\nhdfa-google-paper-fig5a-plan --mode reference\nhdfa-google-paper-fig5a-acquire --mode reference\nhdfa-google-paper-fig5a-merge --mode reference\nhdfa-google-paper-fig5a-validate --mode reference\nhdfa-google-paper-fig5a-plot --mode reference\nhdfa-google-paper-fig5a-compare --mode reference --paper-image D:\\path\\to\\figure5a_crop.png\n```\n\nRepeat with `fig5b`, `fig5c`, `natural-drift`, `randomized-recovery`, and `step-response`. Paper-scale commands additionally require `--execute-paper-scale`.\n"
    path=initialise_layout()/"reports"/"next_user_commands.md";atomic_text(path,text);return path
