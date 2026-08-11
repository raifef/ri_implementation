from __future__ import annotations
import json
from hdfa_rl_suite.google_pure_v7.config import repository_root
from .common import prompt1_report,root,write

def build_paper_comparison()->dict:
    assets=repository_root()/"artifacts/google_pure_paper_reproduction/paper_assets";local=[p for p in assets.rglob("*") if p.is_file()] if assets.exists() else []
    registry_path=root()/"paper_claim_registry.json";registry=json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {"rows":[]};target=root()/"paper_comparison";target.mkdir(parents=True,exist_ok=True);rows=[]
    for claim in registry["rows"]:
      family=claim["experiment_family"].lower();directory=target/family;directory.mkdir(parents=True,exist_ok=True);matching=[p for p in local if family.split("_")[0] in p.stem.lower()]
      checklist={"quantity_definition":bool(claim.get("reproduction_quantity")),"axes":"recorded in reproduction artifact","units":"recorded in reproduction artifact","normalization":"recorded in reproduction artifact","run_family":claim["run_family"],"controller_mode":"hash required","evaluation_budget":"required","uncertainty":"required","visual_grammar":"secondary","scientific_conclusion":claim["status"]}
      verdict="NOT_PUBLICLY_IDENTIFIABLE" if not matching else claim["status"]
      item={"schema_version":"google-pure-evidence-v8-family-comparison.v2","claim_id":claim["claim_id"],"experiment_family":claim["experiment_family"],"paper_reference_assets":[str(p) for p in matching],"reproduction_artifact":claim["artifact"],"numerical_comparison":{"paper_value":claim["paper_value"],"paper_uncertainty":claim["paper_uncertainty"],"reproduction_value":claim["reproduction_value"]},"scientific_checklist":checklist,"pixel_similarity_metric_used":False,"side_by_side_composite_created":False,"composite_blocker":"LOCAL_PAPER_PANEL_NOT_SUPPLIED" if not matching else "COMPOSITE_REQUIRES_EXPLICIT_PANEL_MAPPING","verdict":verdict,"prompt1_hash":prompt1_report()["artifact_hash"]}
      write("comparison",item,f"{claim['experiment_family']} Comparison",directory=directory,json_name="comparison.json",md_name="report.md");rows.append({"claim_id":claim["claim_id"],"experiment_family":claim["experiment_family"],"verdict":verdict,"paper_asset_count":len(matching),"directory":str(directory)})
    return write("comparison",{"schema_version":"google-pure-evidence-v8-paper-comparison.v2","local_assets_only":True,"local_asset_count":len(local),"pixel_similarity_metric_used":False,"comparison_is_evidence_layer":True,"rows":rows,"supported_composite_count":sum(r["paper_asset_count"]>0 for r in rows),"overall":"NOT_PUBLICLY_IDENTIFIABLE" if not local else "CLAIM_SPECIFIC_VERDICTS","prompt1_hash":prompt1_report()["artifact_hash"]},"Paper Comparison Evidence Layer",directory=target,json_name="comparison.json",md_name="report.md")
