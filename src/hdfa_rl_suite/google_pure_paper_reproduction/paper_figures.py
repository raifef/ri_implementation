"""Protocol planning, pure-controller acquisition, and paper-geometry rendering."""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from hdfa_rl_suite.google_pure_v7.config import canonical_hash
from hdfa_rl_suite.google_pure_source_exact.paper_families.common import amended_family_identities
from hdfa_rl_suite.google_pure_source_exact.figure5a.acquisition import (
    COORDINATE_CONTRACT, FIGURE5A_IMPLEMENTATION_VERSION,
)
from hdfa_rl_suite.google_pure_source_exact.source_normalization import (
    BOUNDARY_TRANSFORM_NAME,
    IMPLEMENTATION_VERSION,
    boundary_transform_hash,
    file_hash,
    sensitivity_map_hash_for_family,
    source_normalization_inputs,
)

from .experiment_families import ExperimentFamily, RunMode, guard_seed, require_family
from .storage import atomic_json, discover, initialise_layout, load_merged, merge, save_protocol, write_shard
from .validation import load_validation
from .direct_path import expected_identity, require_amended_acquisition

F = ExperimentFamily


def default_config(family: str, mode: str) -> dict[str, Any]:
    family, mode = require_family(family), RunMode(mode).value
    smoke, validation = mode == "smoke", mode == "validation"
    if family == F.FIGURE5A_REAL_TIME_STEERING.value:
        return {"mode":mode,"epochs":2 if smoke else (20 if validation else 1000),
                "candidates":3 if smoke else (10 if validation else 50),
                "cycles_per_candidate":30 if smoke else (300 if validation else 36000),"controls":41,
                "frequencies":[.001] if smoke else ([.001,1/150] if validation else [.0005,.001,.002,1/300,.005,1/150,.01,1/75,.02]),
                "entropy_coefficients":[.01] if smoke else ([.001,.01,.1] if validation else [.001,.0017782794100389228,.0031622776601683794,.005623413251903491,.01,.01778279410038923,.03162277660168379,.05623413251903491,.1]),
                "seeds":[53101] if smoke else ([53201] if validation else [930001,930002,930003])}
    if family in {F.FIGURE5B_SPARSE_SCALING.value, F.FIGURE5C_CONVERGENCE_LAW.value}:
        return {"mode": mode, "epochs": 4 if smoke else (64 if validation else 1000),
                "candidates":4 if smoke else (10 if validation else 50),
                "cycles_per_candidate":300 if smoke else (10000 if validation else 36000),
                "entropy_coefficient":.001,
                "distances": [3, 15] if smoke else [3,5,7,9,11,13,15], "parameters_per_gate": [1,10,30],
                "seeds": [54101] if smoke else ([54201,54202] if validation else [940001,940002,940003]),
                "local_fit_min_distance": 1e-4, "local_fit_max_distance": .7}
    if family == F.NATURAL_DRIFT_SPECTRAL_SUPPRESSION.value:
        return {"mode":mode,"epochs":180 if smoke else (300 if validation else 825),
                "candidates":3 if smoke else (10 if validation else 40),
                "cycles_per_candidate":300 if smoke else (10000 if validation else 100000),
                "controls":41,"plant_indices":[0,1] if smoke else list(range(6)),
                "seeds":[55101,55102] if smoke else ([55201+i for i in range(6)] if validation else [950001+i for i in range(6)]),
                "evaluation_cadence_epochs":5,"warmup_epoch":150,"logical_evaluation_shots":64 if smoke else (256 if validation else 4096),
                "shared_grid_points":256,"gaussian_smoothing_sigma_bins":5.0,"entropy_coefficient":.001}
    if family == F.RANDOMIZED_RECOVERY_AFTER_SPOIL.value:
        return {"mode":mode,"epochs":20 if smoke else (200 if validation else 1000),
                "candidates":3 if smoke else (10 if validation else 40),
                "cycles_per_candidate":300 if smoke else (10000 if validation else 100000),"controls":924,
                "entropy_coefficient":.001,"sustained_epochs":3 if smoke else 25,
                "severities":[.45],"seeds":[56101] if smoke else ([56201,56202] if validation else [960001,960002,960003])}
    if family == F.STEP_RESPONSE_INJECTED_DRIFT.value:
        return {"mode":mode,"epochs":30 if smoke else (240 if validation else 720),
                "onset_epoch":5 if smoke else (20 if validation else 60),"direction_coordinate":0,
                "candidates":3 if smoke else (10 if validation else 40),
                "cycles_per_candidate":300 if smoke else (10000 if validation else 100000),"controls":924,
                "entropy_coefficient":.001,"severities":[.5],
                "seeds":[57101] if smoke else ([57201,57202] if validation else [91301,91302,91303])}
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
    return amended_family_identities(family)


def _experiment_driver_hash(family: str) -> str:
    module = _module(family)
    backend = Path(__file__).resolve().parents[1] / "google_pure_source_exact" / "paper_families"
    family_backend = {
        F.FIGURE5A_REAL_TIME_STEERING.value: backend.parent / "figure5a" / "acquisition.py",
        F.FIGURE5B_SPARSE_SCALING.value: backend / "scaling.py",
        F.FIGURE5C_CONVERGENCE_LAW.value: backend / "scaling.py",
        F.NATURAL_DRIFT_SPECTRAL_SUPPRESSION.value: backend / "natural.py",
        F.RANDOMIZED_RECOVERY_AFTER_SPOIL.value: backend / "recovery.py",
        F.STEP_RESPONSE_INJECTED_DRIFT.value: backend.parent / "step_response_130" / "acquisition.py",
    }[family]
    return canonical_hash({
        "paper_driver": file_hash(Path(module.__file__).resolve()),
        "source_backend": file_hash(family_backend),
    })


def build_protocol(family: str, *, mode: str = "smoke", config_path: str | Path | None = None,
                   workflow_mode: str = "SMOKE_ACQUISITION",
                   acquisition_run_id: str = "REUSABLE_ACQUISITION",
                   fresh_acquisition_required: bool = False) -> dict[str, Any]:
    family = require_family(family); config = default_config(family, mode)
    if config_path:
        override = json.loads(Path(config_path).read_text(encoding="utf-8")); config.update(override); config["mode"] = RunMode(mode).value
    conditions = _conditions(family, config); plant_hash, graph_hash = _identities(family)
    expected = expected_identity()
    per_condition_cycles = int(config.get("epochs",0))*int(config.get("candidates",0))*int(config.get("cycles_per_candidate",0))
    if family in {F.FIGURE5B_SPARSE_SCALING.value,F.FIGURE5C_CONVERGENCE_LAW.value}:
        from hdfa_rl_suite.google_pure_v7.figure5.accounting import total_controls
        maximum_controls=max(total_controls(int(row["distance"]),int(row["parameters_per_gate"])) for row in conditions)
    else: maximum_controls=int(config.get("controls",41))
    payload = {"schema_version": "google-paper-protocol.v4", "experiment_family": family, "mode": RunMode(mode).value,
               "config": config, "conditions": conditions, "condition_count": len(conditions), "plant_hash": plant_hash, "graph_hash": graph_hash,
               "controller_hash": expected["controller_hash"], "controller_code_hash": expected["controller_code_hash"],
               "controller_mode": expected["controller_mode"], "parameterization": expected["parameterization"],
               "expected_controller_mode": expected["controller_mode"],
               "expected_controller_hash": expected["controller_hash"],
               "expected_controller_code_hash": expected["controller_code_hash"],
               "expected_parameterization": expected["parameterization"],
               "execution_path": "AMENDED_DIRECT_SIGMA_SOURCE_STRUCTURED_ANALOGUE",
               "experiment_driver_hash": _experiment_driver_hash(family),
               "source_budget_profile": str(config.get("profile_name", mode)),
               "workflow_mode": str(workflow_mode),
               "acquisition_run_id": str(acquisition_run_id),
               "fresh_acquisition_required": bool(fresh_acquisition_required),
               "source_contract_version": "google-paper-source-contract.v1", "pure_google_style_rl_only": True,
               "certification_seeds_consumed": False,
               "acquisition_plan":{"candidate_qec_cycles_per_condition":per_condition_cycles,
                   "total_candidate_qec_cycles":per_condition_cycles*len(conditions),
                   "checkpoint_boundary":"candidate" if family==F.STEP_RESPONSE_INJECTED_DRIFT.value else "epoch_candidate_batch",
                   "estimated_action_values_per_epoch":int(config.get("candidates",0))*maximum_controls,
                   "long_run_not_launched_by_plan":True}}
    if family == F.FIGURE5A_REAL_TIME_STEERING.value:
        payload.update({
            "implementation_version": FIGURE5A_IMPLEMENTATION_VERSION,
            "coordinate_contract": COORDINATE_CONTRACT,
            "action_execution": "identity_applied_gaussian",
            "plant_boundary_execution": "none_source_coordinate_identity",
            "likelihood_space": "applied_gaussian",
            "entropy_space": "applied_gaussian",
            "empirical_relative_normalization_applied": False,
            "mean_bounds_applied": False,
        })
    else:
        inputs = source_normalization_inputs()
        payload.update({
            "implementation_version": IMPLEMENTATION_VERSION,
            "sensitivity_map_hash": sensitivity_map_hash_for_family(family),
            "sensitivity_definition_hash": inputs["sensitivity_definition_hash"],
            "calibration_bundle_hash": inputs["calibration_bundle_hash"],
            "detector_degree_audit_hash": inputs["detector_degree_audit_hash"],
            "boundary_transform_name": BOUNDARY_TRANSFORM_NAME,
            "boundary_transform_hash": boundary_transform_hash(),
        })
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


def assigned_conditions(protocol: Mapping[str, Any], *, worker_index: int = 0,
                        worker_count: int = 1) -> list[tuple[int, Mapping[str, Any]]]:
    """Return one deterministic, disjoint partition of the frozen condition list."""
    if worker_count < 1:
        raise ValueError("worker_count must be at least one")
    if not 0 <= worker_index < worker_count:
        raise ValueError("worker_index must lie in [0, worker_count)")
    return [(index, condition) for index, condition in enumerate(protocol["conditions"])
            if index % worker_count == worker_index]


def acquire(protocol: Mapping[str, Any], *, max_shards: int | None = None,
            execute_paper_scale: bool = False, worker_index: int = 0,
            worker_count: int = 1) -> dict[str, Any]:
    if protocol["mode"] == "paper-scale" and not execute_paper_scale: raise RuntimeError("paper-scale acquisition requires --execute-paper-scale")
    if protocol["mode"] in {"reference", "paper-scale"}:
        require_amended_acquisition(protocol)
    assignment = assigned_conditions(protocol, worker_index=worker_index, worker_count=worker_count)
    existing = {record["shard_id"] for record in discover(protocol)}
    if existing and protocol.get("fresh_acquisition_required"):
        raise RuntimeError("fresh V15 acquisition forbids reuse of existing paper shards")
    module = _module(protocol["experiment_family"]); completed = []; skipped = []
    for _, condition in assignment:
        identity = {"family": protocol["experiment_family"], "protocol_hash": protocol["protocol_hash"],
                    "condition": dict(condition)}
        shard_id = canonical_hash(identity)
        if shard_id in existing:
            skipped.append(shard_id)
            continue
        data = module.acquire_condition(protocol, condition); record = write_shard(protocol, condition, data); completed.append(record["shard_id"])
        if max_shards is not None and len(completed) >= max_shards: break
    result = {"experiment_family": protocol["experiment_family"], "mode": protocol["mode"], "protocol_hash": protocol["protocol_hash"],
              "workflow_mode": protocol.get("workflow_mode"),
              "fresh_acquisition_required": bool(protocol.get("fresh_acquisition_required", False)),
              "fresh_acquisition": len(skipped) == 0,
              "reused_shard_ids": skipped,
              "worker_index": worker_index, "worker_count": worker_count,
              "assigned_shards": len(assignment), "preexisting_assigned_shards": len(skipped),
              "completed_this_call": len(completed), "expected_shards": protocol["condition_count"],
              "shard_ids": completed, "skipped_shard_ids": skipped}
    stem = f"{protocol['experiment_family'].lower()}_{protocol['mode']}_acquisition"
    if worker_count > 1:
        stem += f"_worker_{worker_index:03d}_of_{worker_count:03d}"
    atomic_json(initialise_layout()/"manifests"/f"{stem}.json", result)
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
        fig,ax=plt.subplots(figsize=(7.2,5.2),constrained_layout=True); cmap=plt.get_cmap("viridis")
        for row in [item for item in rows if item["parameters_per_gate"]==30]:
            t=np.asarray(row["trajectory"]["epoch"]);physical=np.asarray(row["trajectory"]["physical_error"]);logical=np.asarray(row["trajectory"]["logical_learned"])
            ax.plot(physical,logical,color="0.75",lw=.55,zorder=1);ax.scatter(physical,logical,c=t,cmap=cmap,s=9,alpha=.72,label=f"d={row['distance']}")
            width=max(float(physical[-1])*.08,1e-7);ax.hlines(row["logical_floor"],physical[-1]-width,physical[-1]+width,colors="crimson",lw=2.2)
        ax.set(xscale="log",yscale="log",xlabel="physical error rate",ylabel="logical error rate",title="Figure 5b source-structured sparse scaling analogue")
        plotted=[value for row in rows if row["parameters_per_gate"]==30 for value in row["trajectory"]["logical_learned"]]
        floors=[row["logical_floor"] for row in rows if row["parameters_per_gate"]==30]
        if plotted and floors: ax.set_ylim(max(min(floors)*.6,1e-12),max(plotted)*1.6)
        handles,labels=ax.get_legend_handles_labels();dedup=dict(zip(labels,handles));ax.legend(dedup.values(),dedup.keys(),fontsize=7,ncol=2);stem="figure5b_reproduction"
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
            plt.close(fig); fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
            for row in rows:
                trace=row["trace"];axes[0].plot(trace["epochs"],trace["learned_mean_ler"],alpha=.75,label=f"{row['plant_id']} learned")
                axes[0].plot(trace["epochs"],trace["fixed_initial_ler"],alpha=.55,ls="--",label=f"{row['plant_id']} fixed")
            diagnostic=validation["metrics"]["source_dft_analysis"]; frequency=np.asarray(diagnostic["frequency_per_epoch"])
            axes[1].loglog(frequency,diagnostic["learned_geometric_psd"],label="learned mean")
            axes[1].loglog(frequency,diagnostic["fixed_geometric_psd"],label="fixed",ls="--")
            raw=np.asarray(diagnostic["raw_filter_db"])
            axes[2].semilogx(frequency,raw,alpha=.45,label="raw ratio"); axes[2].semilogx(frequency,diagnostic["smoothed_guide_to_eye_db"],label="Gaussian guide")
            axes[2].axhline(0,color="black",lw=.6)
            axes[0].axvline(150,color="black",ls=":",lw=.7);axes[0].set(xlabel="epoch",ylabel="decoded logical error rate",title="Paired learned-mean/fixed evaluations")
            axes[1].set(xlabel="frequency (epoch$^{-1}$)",ylabel="DFT power",title="Geometric-average spectra")
            axes[2].set(xlabel="frequency (epoch$^{-1}$)",ylabel="10 log10(learned/fixed), dB",title="Filter ratio with uncertainty")
            for item in axes: item.legend(fontsize=6)
            fig.suptitle("UNDERPOWERED_DEVELOPMENT_VALIDATION — SOURCE SECTION-III ESTIMATOR ON LOCAL TRACES",color="crimson");stem="natural_drift_reproduction"
        elif family == F.RANDOMIZED_RECOVERY_AFTER_SPOIL.value:
            for row in rows: ax.plot(row["trajectory"]["learned_mean_excess_logical_risk"],label=f"severity={row['severity']}")
            censored=validation["metrics"].get("censored_count",0); total=validation["metrics"].get("run_count",len(rows))
            ax.set(xlabel="epoch after spoil",ylabel="excess logical-risk proxy",yscale="log",title="Randomized-policy recovery")
            ax.text(.5,.93,f"{censored} / {total} RUNS CENSORED",transform=ax.transAxes,ha="center",color="crimson",weight="bold")
            ax.legend();stem="randomized_recovery_reproduction"
        else:
            for row in rows: ax.plot(row["trajectory"]["normalized_projected_policy_response"],label=f"severity={row['severity']}");ax.axvline(row["onset_epoch"],color="black",lw=.5)
            ax.axhline(1.0,color="black",ls="--",label="injected target");ax.axhline(.9,color="grey",ls=":",label="90% target")
            ax.set(xlabel="epoch",ylabel="normalized projected response",title="Injected-drift step response (target-relative crossings)")
            ax.text(.5,.04,f"censored: {validation['metrics'].get('censored_count',0)} / {len(rows)}",transform=ax.transAxes,ha="center",color="crimson")
            ax.legend();stem="step_response_reproduction"
    _watermark(fig, protocol["mode"], scientifically_valid=validation["valid"]); root=initialise_layout()/"figures"; png=root/f"{stem}.png";svg=root/f"{stem}.svg";fig.savefig(png,dpi=180);fig.savefig(svg);plt.close(fig)
    result={"experiment_family":family,"mode":protocol["mode"],"protocol_hash":protocol["protocol_hash"],"files":[str(png),str(svg)],
            "development_classification": ("UNDERPOWERED_DEVELOPMENT_VALIDATION"
                if family == F.NATURAL_DRIFT_SPECTRAL_SUPPRESSION.value else None),
            "watermarked":protocol["mode"] in {"smoke","validation"} or not validation["valid"],"final_evidence":validation["final_evidence"],"status":validation["status"]}
    atomic_json(root/f"{stem}_manifest.json",result);return result
