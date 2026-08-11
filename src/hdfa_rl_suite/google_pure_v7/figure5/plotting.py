"""Matplotlib-only rendering from merged, validated data; never runs simulations."""
from __future__ import annotations
import csv, json
from pathlib import Path
from typing import Any
import numpy as np
from .common import atomic_json, atomic_text, figure5_root

def _read_rows(panel:str)->list[dict]:
    path=figure5_root()/"merged"/panel/"summary.csv"
    if not path.exists():raise RuntimeError(f"missing merged panel {panel}")
    with path.open(encoding="utf-8",newline="") as stream:return list(csv.DictReader(stream))

def _require_validation(panel:str,partial:bool)->None:
    path=figure5_root()/"reports"/f"panel_{panel[-1]}_validation.json"
    if not path.exists() and not partial:raise RuntimeError(f"panel {panel} is not validated")
    if path.exists() and not json.loads(path.read_text(encoding="utf-8"))["valid"]:raise RuntimeError(f"panel {panel} validation failed")

def plot_panel(panel:str,*,formats=("png","svg","pdf"),partial=False)->dict:
    _require_validation(panel,partial);rows=_read_rows(panel)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6.4,4.4),constrained_layout=True)
    if panel=="5a":
        frequencies=sorted({float(r["frequency"]) for r in rows});entropies=sorted({float(r["entropy_coefficient"]) for r in rows})
        z=np.full((len(entropies),len(frequencies)),np.nan);se=np.full_like(z,np.nan)
        for yi,e in enumerate(entropies):
            for xi,f in enumerate(frequencies):
                vals=[float(r["improvement_candidate"]) for r in rows if float(r["frequency"])==f and float(r["entropy_coefficient"])==e]
                if vals:
                    z[yi,xi]=np.mean(vals);se[yi,xi]=np.std(vals,ddof=1)/np.sqrt(len(vals)) if len(vals)>1 else 0.
        image=ax.imshow(z,origin="lower",aspect="auto",extent=[min(frequencies),max(frequencies),min(entropies),max(entropies)],cmap="coolwarm")
        if len(frequencies)>1 and len(entropies)>1 and np.nanmin(z)<=0<=np.nanmax(z):
            ax.contour(frequencies,entropies,z,levels=[0],colors="black",linewidths=1)
            for boundary in (z-se,z+se):
                if np.nanmin(boundary)<=0<=np.nanmax(boundary):ax.contour(frequencies,entropies,boundary,levels=[0],colors="black",linewidths=.6,linestyles="dashed")
        fig.colorbar(image,ax=ax,label="sampled-candidate improvement");ax.set(xlabel="drift frequency (epochs$^{-1}$)",ylabel="entropy coefficient",title="Figure 5a synthetic steerability")
    elif panel=="5b":
        path=figure5_root()/"merged"/panel/"trajectories.jsonl"; trace=[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
        for d in sorted({r["distance"] for r in trace}):
            subset=[r for r in trace if r["distance"]==d and r["parameters_per_gate"]==max(x["parameters_per_gate"] for x in trace)]
            epochs=sorted({r["epoch"] for r in subset});groups=[[r["logical_learned"] for r in subset if r["epoch"]==e] for e in epochs];means=np.asarray([np.mean(v) for v in groups]);std=np.asarray([np.std(v,ddof=1) if len(v)>1 else 0 for v in groups])
            ax.plot(epochs,means,label=f"d={d}");ax.fill_between(epochs,np.maximum(means-std,1e-12),means+std,alpha=.15)
        ax.set_yscale("log");ax.set(xlabel="epoch",ylabel="logical metric (lower is better)",title="Figure 5b sparse scaling trajectories");ax.legend(ncol=2,fontsize=8)
    else:
        path=figure5_root()/"merged"/panel/"trajectories.jsonl";trace=[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x and json.loads(x)["fit"]]
        for p in sorted({r["parameters_per_gate"] for r in trace}):
            subset=[r for r in trace if r["parameters_per_gate"]==p];xs=np.asarray([r["x_distance"] for r in subset]);ys=np.asarray([r["normalized_speed"] for r in subset]);ax.scatter(xs,ys,s=7,alpha=.25,label=f"P={p}")
            slope=np.dot(xs,ys)/np.dot(xs,xs);line=np.linspace(0,float(xs.max()),100);ax.plot(line,slope*line,linewidth=1)
        ax.set(xlabel=r"$1-\Lambda/\Lambda^*$",ylabel=r"$10^2\,\partial_t\Lambda/\Lambda^*$",title="Figure 5c convergence law");ax.legend()
    root=figure5_root()/"figures";root.mkdir(parents=True,exist_ok=True);files=[]
    for extension in formats:
        target=root/f"panel_{panel[-1]}.{extension}";fig.savefig(target,dpi=180 if extension=="png" else None);files.append(str(target))
    plt.close(fig);result={"panel":panel,"files":files,"partial":partial,"status":"PLOT_COMPLETE"}
    atomic_json(root/f"panel_{panel[-1]}_plot_manifest.json",result);return result

def plot_all(*,formats=("png","svg","pdf"),partial=False)->dict:
    panels={p:plot_panel(p,formats=formats,partial=partial) for p in ("5a","5b","5c")}
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    root=figure5_root()/"figures";fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    for axis,panel,label in zip(axes,("a","b","c"),("a","b","c")):
        axis.imshow(plt.imread(root/f"panel_{panel}.png"));axis.axis("off");axis.set_title(f"({label})",loc="left",fontweight="bold")
    combined=[]
    for extension in formats:
        target=root/f"figure5_combined.{extension}";fig.savefig(target,dpi=180 if extension=="png" else None);combined.append(str(target))
    plt.close(fig)
    caption=("Figure 5 synthetic reproduction. (a) Sampled-candidate performance uses the public lower-is-better normalization; "
      "the zero contour marks parity with a fixed policy. (b) Sparse analytic trajectories preserve the public distance/control-count mapping "
      "but are not Google simulator output. (c) Source-axis local derivatives test the origin-constrained convergence law. Bands/points aggregate independent seeds.")
    atomic_text(root/"figure5_caption.md",caption+"\n");result={"panels":panels,"combined":combined,"caption":caption}
    atomic_json(root/"figure5_combined_plot_manifest.json",result);return result
