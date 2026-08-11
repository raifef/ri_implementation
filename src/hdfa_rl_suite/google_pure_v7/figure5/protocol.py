"""Immutable protocol plans and their hashes."""
from __future__ import annotations
from itertools import product
from typing import Any
from hdfa_rl_suite.google_pure_v7.controller import require_resolved_controller
from hdfa_rl_suite.google_pure_v7.config import canonical_hash
from .common import atomic_json, atomic_text, config_hash, figure5_root, read_config
from .seed_registry import SEEDS, validate_registry

def panel_plan(panel: str, config: dict[str, Any]) -> dict[str, Any]:
    mode = str(config["mode"]); seeds = tuple(config.get("seeds", SEEDS[panel][mode]))
    if panel == "5a":
        grid = [{"frequency": f, "entropy_coefficient": e} for f, e in product(config["frequencies"], config["entropy_coefficients"])]
    else:
        grid = [{"distance": d, "parameters_per_gate": p} for d, p in product(config["distances"], config["parameters_per_gate"])]
    conditions = [{**cell, "seed": seed} for cell in grid for seed in seeds]
    controller = require_resolved_controller()
    payload = {"schema_version": "google-pure-v7-figure5-plan.v1", "panel": panel, "mode": mode,
               "config": config, "config_hash": config_hash(config), "controller_code_hash": controller["controller_code_hash"],
               "resolved_controller_hash": controller["resolved_config_hash"], "grid": grid, "seeds": list(seeds),
               "conditions": conditions, "condition_count": len(conditions), "certification_seeds_consumed": False}
    payload["protocol_hash"] = canonical_hash(payload)
    return payload

def freeze_protocols() -> dict[str, Any]:
    validate_registry(); output = {}
    for panel in ("5a", "5b", "5c"):
        for suffix in ("smoke", "reference"):
            plan = panel_plan(panel, read_config(f"panel_{panel[-1]}_{suffix}.yaml")); name = f"panel_{panel[-1]}_{suffix}"
            atomic_json(figure5_root()/"protocol_freezes"/f"{name}.json", plan); output[name] = plan["protocol_hash"]
    summary = {"schema_version":"google-pure-v7-figure5-protocol-freeze.v1","protocols":output,"status":"PIPELINE_BUILT_UNVALIDATED"}
    atomic_json(figure5_root()/"protocol_freezes"/"protocol_index.json", summary)
    atomic_text(figure5_root()/"protocol_freezes"/"protocol_index.md", "# Figure 5 Protocol Freezes\n\n"+"\n".join(f"- {k}: `{v}`" for k,v in output.items())+"\n")
    return summary

def plan_all() -> dict[str, Any]:
    plans = {}
    for panel in ("5a", "5b", "5c"):
        for suffix in ("smoke", "reference"):
            name=f"panel_{panel[-1]}_{suffix}"; plan=panel_plan(panel, read_config(name+".yaml"))
            atomic_json(figure5_root()/"run_plans"/(name+".json"), plan); plans[name]=plan
    result={"schema_version":"google-pure-v7-figure5-all-plans.v1","plans":{k:v["protocol_hash"] for k,v in plans.items()}}
    atomic_json(figure5_root()/"run_plans"/"all.json",result); return result
