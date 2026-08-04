"""Protocol planning, pure-controller acquisition, and paper-geometry rendering."""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from google_rl_reimplementation.google_pure_v7.config import canonical_hash
from google_rl_reimplementation.google_pure_v7.controller import require_resolved_controller
from google_rl_reimplementation.google_pure_v7.natural import FAMILIES

from .experiment_families import ExperimentFamily, RunMode, guard_seed, require_family
from .storage import atomic_json, initialise_layout, load_merged, merge, save_protocol, write_shard
from .validation import load_validation

F = ExperimentFamily


def default_config(family: str, mode: str) -> dict[str, Any]:
    family, mode = require_family(family), RunMode(mode).value
    smoke, validation = mode == "smoke", mode == "validation"
    if family == F.FIGURE5A_REAL_TIME_STEERING.value:
        return {"mode": mode, "epochs": 8 if smoke else (120 if validation else 1000), "candidates": 8 if smoke else (20 if validation else 50),
                "cycles_per_candidate": 1000 if smoke else (10000 if validation else 36000), "controls": 6, "drift_amplitude": .45,
                "frequencies": [1/300, 1/150, 1/75] if smoke else np.geomspace(1/600, 1/40, 13 if not validation else 7).tolist(),
                "entropy_coefficients": [1e-4, 4e-4, 1e-3] if smoke else np.geomspace(3e-5, 3e-3, 11 if not validation else 7).tolist(),
                "seeds": [13101, 13102] if smoke else ([13111, 13112] if validation else [13121, 13122, 13123])}
    if family in {F.FIGURE5B_SPARSE_SCALING.value, F.FIGURE5C_CONVERGENCE_LAW.value}:
        return {"mode": mode, "epochs": 24 if smoke else (128 if validation else 1000),
                "distances": [3, 15] if smoke else [3,5,7,9,11,13,15], "parameters_per_gate": [1,10,30],
                "seeds": [13201] if smoke else ([13211,13212] if validation else [13221,13222,13223]),
                "local_fit_min_distance": 1e-4, "local_fit_max_distance": .7}
    if family == F.NATURAL_DRIFT_SPECTRAL_SUPPRESSION.value:
        return {"mode": mode, "epochs": 128 if smoke else 768, "candidates": 8 if smoke else 40,
                "cycles_per_candidate": 2000 if smoke else 100000, "plant_indices": [0] if smoke else list(range(6)),
                "seeds": [13301] if smoke else [13311+i for i in range(6)]}
    if family == F.RANDOMIZED_RECOVERY_AFTER_SPOIL.value:
        return {"mode": mode, "epochs": 48 if smoke else (300 if validation else 1000), "candidates": 8 if smoke else 40,
                "cycles_per_candidate": 2000 if smoke else 100000, "controls": 6,
                "severities": [.45] if smoke else [.25,.45,.65], "seeds": [13401] if smoke else ([13411,13412] if validation else [13421,13422,13423])}
    if family == F.STEP_RESPONSE_INJECTED_DRIFT.value:
        return {"mode": mode, "epochs": 80 if smoke else (360 if validation else 720), "candidates": 8 if smoke else 40,
                "cycles_per_candidate": 2000 if smoke else 100000, "controls": 6,
                "severities": [.35] if smoke else [.25,.45,.65], "seeds": [13501] if smoke else ([13511,13512] if validation else [13521,13522,13523])}
    raise ValueError(f"no synthetic config for {family}")


def _conditions(family: str, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    seeds = list(map(int, config["seeds"])); [guard_seed(seed) for seed in seeds]
    if family == F.FIGURE5A_REAL_TIME_STEERING.value:
        return [{"frequency": float(f), "entropy_coefficient": float(e), "seed": seed} for f,e,seed in product(config["frequencies"], config["entropy_coefficients"], seeds)]
    if family in {F.FIGURE5B_SPARSE_SCALING.value, F.FIGURE5C_CONVERGENCE_LAW.value}:
        return [{"distance": int(d), "parameters_per_gate": int(p), "seed": seed} for d,p,seed in product(config["distances"], config["parameters_per_gate"], seeds)]
    if family == F.NATURAL_DRIFT_SPECTRAL_SUPPRESSION.value:
        if len(seeds) not in {1, len(config["plant_indices"])}: raise ValueError("natural-drift seeds must be singleton or one per plant")
        return [{"plant_index": int(index), "seed": seeds[0] if len(seeds)==1 else seeds[i]} for i,index in enumerate(config["plant_indices"])]
    return [{"severity": float(s), "seed": seed} for s,seed in product(config["severities"], seeds)]


def _identities(family: str) -> tuple[str, str]:
    if family == F.FIGURE5A_REAL_TIME_STEERING.value: return "v6-default-quadratic-6", "local-detector-control-mask-v6"
    if family in {F.FIGURE5B_SPARSE_SCALING.value, F.FIGURE5C_CONVERGENCE_LAW.value}: return "paper-quadratic-sparse-surrogate-v1", "surface-code-local-count-graph-v1"
    if family == F.NATURAL_DRIFT_SPECTRAL_SUPPRESSION.value: return "frozen-six-plant-natural-v7", "paired-local-detector-control-graph-v7"
    return "v6-default-quadratic-6", "local-detector-control-mask-v6"


def build_protocol(family: str, *, mode: str = "smoke", config_path: str | Path | None = None) -> dict[str, Any]:
    family = require_family(family); config = default_config(family, mode)
    if config_path:
        override = json.loads(Path(config_path).read_text(encoding="utf-8")); config.update(override); config["mode"] = RunMode(mode).value
    conditions = _conditions(family, config); controller = require_resolved_controller(); plant_hash, graph_hash = _identities(family)
    payload = {"schema_version": "google-paper-protocol.v1", "experiment_family": family, "mode": RunMode(mode).value,
               "config": config, "conditions": conditions, "condition_count": len(conditions), "plant_hash": plant_hash, "graph_hash": graph_hash,
               "controller_hash": controller["resolved_config_hash"], "controller_code_hash": controller["controller_code_hash"],
               "source_contract_version": "google-paper-source-contract.v1", "pure_google_style_rl_only": True,
               "certification_seeds_consumed": False}
    payload["protocol_hash"] = canonical_hash(payload); save_protocol(payload); return payload


def _module(family: str):
    if family == F.FIGURE5A_REAL_TIME_STEERING.value: from . import panel_a as module
    elif family == F.FIGURE5B_SPARSE_SCALING.value: from . import panel_b as module
    elif family == F.FIGURE5C_CONVERGENCE_LAW.value: from . import panel_c as module
    elif family == F.NATURAL_DRIFT_SPECTRAL_SUPPRESSION.value: from . import natural_drift as module
    elif family == F.RANDOMIZED_RECOVERY_AFTER_SPOIL.value: from . import randomized_recovery as module
    elif family == F.STEP_RESPONSE_INJECTED_DRIFT.value: from . import step_response as module
    else: raise ValueError(family)
    return module


def acquire(protocol: Mapping[str, Any], *, max_shards: int | None = None, execute_paper_scale: bool = False) -> dict[str, Any]:
    if protocol["mode"] == "paper-scale" and not execute_paper_scale: raise RuntimeError("paper-scale acquisition requires --execute-paper-scale")
    module = _module(protocol["experiment_family"]); completed = []
    for condition in protocol["conditions"]:
        data = module.acquire_condition(protocol, condition); record = write_shard(protocol, condition, data); completed.append(record["shard_id"])
        if max_shards is not None and len(completed) >= max_shards: break
    result = {"experiment_family": protocol["experiment_family"], "mode": protocol["mode"], "protocol_hash": protocol["protocol_hash"],
              "completed_this_call": len(completed), "expected_shards": protocol["condition_count"], "shard_ids": completed}
    atomic_json(initialise_layout()/"manifests"/f"{protocol['experiment_family'].lower()}_{protocol['mode']}_acquisition.json", result)
    return result


def merge_protocol(protocol: Mapping[str, Any], *, allow_partial: bool = False) -> dict[str, Any]:
    return merge(protocol, allow_partial=allow_partial)


def _watermark(fig: Any, mode: str, *, scientifically_valid: bool = True) -> None:
    label = None
    if not scientifically_valid: label = "SCIENTIFIC MISMATCH"
    elif mode in {"smoke", "validation"}: label = "SMOKE RENDER ONLY" if mode == "smoke" else "VALIDATION ONLY"
    if label:
        fig.text(.5, .5, label, ha="center", va="center", fontsize=28, color="crimson", alpha=.18, rotation=24, weight="bold")


def plot_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    validation = load_validation(protocol); merged = load_merged(protocol)
    if not validation["complete"]: raise RuntimeError("cannot plot incomplete merged data")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    family, rows = protocol["experiment_family"], merged["rows"]
    if family == F.FIGURE5A_REAL_TIME_STEERING.value:
        fig = plt.figure(figsize=(11,4.4), constrained_layout=True); ax=fig.add_subplot(121); surface=fig.add_subplot(122, projection="3d")
        fs=sorted({r["frequency"] for r in rows}); es=sorted({r["entropy_coefficient"] for r in rows}); z=np.full((len(es),len(fs)),np.nan)
        for yi,e in enumerate(es):
            for xi,f in enumerate(fs): z[yi,xi]=np.mean([r["improvement_candidate"] for r in rows if r["frequency"]==f and r["entropy_coefficient"]==e])
        x,y=np.meshgrid(fs,es); image=ax.pcolormesh(x,y,z,shading="auto",cmap="coolwarm"); ax.set(xscale="log",yscale="log",xlabel="drift frequency (epoch$^{-1}$)",ylabel="entropy coefficient",title="sampled-candidate aggregate")
        if np.nanmin(z)<=0<=np.nanmax(z): ax.contour(x,y,z,levels=[0],colors="black")
        ax.axvline(1/150,color="black",ls="--",lw=.8);fig.colorbar(image,ax=ax,label="normalized improvement")
        surface.plot_surface(np.log10(x),np.log10(y),z,cmap="coolwarm",alpha=.9);surface.set(xlabel="log10 frequency",ylabel="log10 entropy",zlabel="improvement",title="phase surface")
        stem="figure5a_reproduction"
    elif family == F.FIGURE5B_SPARSE_SCALING.value:
        fig,ax=plt.subplots(figsize=(7.2,4.8),constrained_layout=True); cmap=plt.get_cmap("viridis")
        for row in rows:
            t=np.asarray(row["trajectory"]["epoch"]); lam=np.asarray(row["trajectory"]["lambda_ratio"]); ax.scatter(t,lam,c=t,cmap=cmap,s=7,alpha=.45)
            ax.hlines(1.0,t[0],t[-1],colors="grey",lw=.35)
        ax.set(xlabel="epoch (colour progresses with epoch)",ylabel=r"$\Lambda/\Lambda^*$",title="Figure 5b sparse scaling: d=3…15; P=1,10,30");ax.set_ylim(0,1.05); stem="figure5b_reproduction"
    elif family == F.FIGURE5C_CONVERGENCE_LAW.value:
        fig,ax=plt.subplots(figsize=(7.2,4.8),constrained_layout=True); distances=sorted({r["distance"] for r in rows}); colors={d:plt.get_cmap("viridis")(i/max(1,len(distances)-1)) for i,d in enumerate(distances)}
        for row in rows:
            tr=row["trajectory"]; keep=np.asarray(tr["fit_mask"],bool);x=np.asarray(tr["x_distance"])[keep];y=np.asarray(tr["normalized_speed"])[keep];ax.scatter(x,y,s=6,alpha=.2,color=colors[row["distance"]])
        for p,alpha in zip((1,10,30),(.45,.7,1.0)):
            subset=[r for r in rows if r["parameters_per_gate"]==p]; slope=np.mean([r["gamma_times_100"] for r in subset]);line=np.linspace(0,.7,100);ax.plot(line,slope*line,color="red",alpha=alpha,label=f"P={p}")
        ax.set(xlabel=r"$1-\Lambda/\Lambda^*$",ylabel=r"$10^2\,\partial_t\Lambda/\Lambda^*$",title="Figure 5c local convergence law");ax.legend();stem="figure5c_reproduction"
    else:
        fig,ax=plt.subplots(figsize=(7.2,4.8),constrained_layout=True)
        if family == F.NATURAL_DRIFT_SPECTRAL_SUPPRESSION.value:
            for row in rows: ax.plot(row["trajectory"]["learned_mean"],alpha=.75,label=row["plant_id"])
            ax.set(xlabel="epoch",ylabel="logical-risk proxy",title="Natural-drift learned-mean traces");ax.legend(fontsize=7);stem="natural_drift_reproduction"
        elif family == F.RANDOMIZED_RECOVERY_AFTER_SPOIL.value:
            for row in rows: ax.plot(row["trajectory"]["learned_mean_excess_logical_risk"],label=f"severity={row['severity']}")
            ax.set(xlabel="epoch after spoil",ylabel="excess logical-risk proxy",yscale="log",title="Randomized-policy recovery");ax.legend();stem="randomized_recovery_reproduction"
        else:
            for row in rows: ax.plot(row["trajectory"]["normalized_projected_policy_response"],label=f"severity={row['severity']}");ax.axvline(row["onset_epoch"],color="black",lw=.5)
            ax.set(xlabel="epoch",ylabel="normalized projected response",title="Injected-drift step response");ax.legend();stem="step_response_reproduction"
    _watermark(fig, protocol["mode"], scientifically_valid=validation["valid"]); root=initialise_layout()/"figures"; png=root/f"{stem}.png";svg=root/f"{stem}.svg";fig.savefig(png,dpi=180);fig.savefig(svg);plt.close(fig)
    result={"experiment_family":family,"mode":protocol["mode"],"protocol_hash":protocol["protocol_hash"],"files":[str(png),str(svg)],
            "watermarked":protocol["mode"] in {"smoke","validation"} or not validation["valid"],"final_evidence":validation["final_evidence"],"status":validation["status"]}
    atomic_json(root/f"{stem}_manifest.json",result);return result
