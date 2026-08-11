"""Auditable paper-panel/reproduction comparison canvases."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from hdfa_rl_suite.google_pure_v7.config import sha256_file

from .comparison_metrics import family_checks
from .experiment_families import ExperimentFamily
from .storage import atomic_json, atomic_text, initialise_layout
from .validation import load_validation

PANEL_FAMILY = {
    "figure5a": ExperimentFamily.FIGURE5A_REAL_TIME_STEERING.value,
    "figure5b": ExperimentFamily.FIGURE5B_SPARSE_SCALING.value,
    "figure5c": ExperimentFamily.FIGURE5C_CONVERGENCE_LAW.value,
}


def register_paper_asset(panel: str, path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.exists() or source.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        raise ValueError("paper asset must be a local raster panel crop")
    result = {"panel": panel, "path": str(source), "sha256": sha256_file(source), "user_supplied": True,
              "copyright_republication_claim": False, "purpose": "local visual comparison"}
    atomic_json(initialise_layout()/"paper_assets"/f"{panel}.json", result); return result


def _asset(panel: str) -> Path | None:
    manifest = initialise_layout()/"paper_assets"/f"{panel}.json"
    if not manifest.exists(): return None
    value = json.loads(manifest.read_text(encoding="utf-8")); path = Path(value["path"])
    if not path.exists() or sha256_file(path) != value["sha256"]: raise RuntimeError("paper asset is stale or changed")
    return path


def compare_panel(panel: str, protocol: Mapping[str, Any], *, paper_image: str | Path | None = None) -> dict[str, Any]:
    if panel not in PANEL_FAMILY or PANEL_FAMILY[panel] != protocol["experiment_family"]: raise RuntimeError("wrong experiment family for panel comparison")
    if paper_image: register_paper_asset(panel, paper_image)
    validation = load_validation(protocol); reproduction = initialise_layout()/"figures"/f"{panel}_reproduction.png"
    if not reproduction.exists(): raise RuntimeError("missing reproduction plot")
    source = _asset(panel)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    if source:
        axes[0].imshow(plt.imread(source)); axes[0].set_title("Published paper panel (local user asset)")
    else:
        axes[0].text(.5,.5,"PAPER PANEL IMAGE\nNOT LOCALLY SUPPLIED",ha="center",va="center",fontsize=16,color="dimgray");axes[0].set_facecolor("#f2f2f2")
    axes[0].axis("off"); axes[1].imshow(plt.imread(reproduction)); axes[1].set_title("Independent reproduction");axes[1].axis("off")
    if protocol["mode"] in {"smoke","validation"}: fig.text(.5,.5,"SMOKE RENDER ONLY" if protocol["mode"]=="smoke" else "VALIDATION ONLY",ha="center",va="center",fontsize=30,color="crimson",alpha=.18,rotation=20,weight="bold")
    target=initialise_layout()/"side_by_side"/f"{panel}_side_by_side.png";fig.savefig(target,dpi=180);plt.close(fig)
    checks=family_checks(protocol["experiment_family"],validation)
    verdict = "SMOKE_RENDER_ONLY" if protocol["mode"]=="smoke" else ("MISMATCH" if not validation["valid"] else ("NOT_YET_RUN" if source is None else ("QUALITATIVE_MATCH_ONLY" if not validation["final_evidence"] else "WITHIN_TOLERANCE_SYNTHETIC_MATCH")))
    result={"schema_version":"google-paper-side-by-side.v1","panel":panel,"experiment_family":protocol["experiment_family"],
            "mode":protocol["mode"],"protocol_hash":protocol["protocol_hash"],"paper_asset_present":source is not None,
            "reproduction":str(reproduction),"output":str(target),"numeric_checks":checks,"verdict":verdict,
            "plot_similarity_alone_is_reproduction":False,"final_evidence":validation["final_evidence"] and source is not None}
    atomic_json(target.with_suffix(".json"),result)
    atomic_text(target.with_suffix(".md"),f"# {panel} side-by-side audit\n\nVerdict: **{verdict}**\n\n- Paper asset present: {source is not None}\n- Numeric checks: {len(checks)}\n- Plot similarity alone is never accepted as reproduction.\n")
    return result


def compare_all(protocols: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    panels={panel:compare_panel(panel,protocols[panel]) for panel in PANEL_FAMILY}
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(13,15),constrained_layout=True)
    for ax,panel in zip(axes,PANEL_FAMILY): ax.imshow(plt.imread(panels[panel]["output"]));ax.axis("off")
    target=initialise_layout()/"side_by_side"/"figure5_full_side_by_side.png";fig.savefig(target,dpi=160);plt.close(fig)
    result={"panels":panels,"output":str(target),"all_final":all(row["final_evidence"] for row in panels.values())}
    atomic_json(target.with_suffix(".json"),result);return result
