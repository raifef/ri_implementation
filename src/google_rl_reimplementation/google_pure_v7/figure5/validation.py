"""Logical metric contract plus panel scientific validation."""
from __future__ import annotations
from typing import Any, Mapping
import numpy as np
from .accounting import total_controls
from .common import atomic_json, atomic_text, figure5_root

def logical_metric(raw_counts: Any, metadata: Mapping[str, Any]) -> float:
    failures=float(np.asarray(raw_counts).sum()); trials=int(metadata["trials"])
    if trials<=0 or failures<0 or failures>trials: raise ValueError("invalid logical counts")
    return failures/trials

def logical_floor(plant: Any, protocol: Mapping[str, Any]) -> float:
    value=float(protocol["irreducible_logical_floor"])
    if not 0<=value<1: raise ValueError("invalid independent logical floor")
    return value

def normalized_progress(value: float, floor: float, initial: float) -> float:
    if initial<=floor: raise ValueError("initial metric must exceed floor")
    return (initial-value)/(initial-floor)

def validate_rows(panel: str, rows: list[dict], *, mode: str) -> dict:
    reasons=[];metrics={}
    if not rows: reasons.append("no merged rows")
    numeric=[v for row in rows for v in row.values() if isinstance(v,(int,float))]
    if numeric and not np.all(np.isfinite(numeric)): reasons.append("non-finite numeric value")
    if panel=="5a" and rows:
        if not all("improvement_candidate" in r and "improvement_mean" in r for r in rows): reasons.append("missing dual normalization")
        if any(r["candidate_cycles"]<=0 for r in rows): reasons.append("invalid cycle accounting")
        values=[r["improvement_candidate"] for r in rows];metrics["candidate_improvement_range"]=[float(min(values)),float(max(values))]
        if mode!="smoke" and not (min(values)<=0<=max(values)):reasons.append("zero steerability contour is not bracketed")
    if panel=="5b" and rows:
        if total_controls(15,30)!=38670: reasons.append("d15 P30 control mapping failed")
        if any(r["logical_floor"]>=r["logical_initial"] for r in rows): reasons.append("floor not independent below initial")
    if panel=="5c" and rows:
        if any(r["x_distance"]<0 for r in rows): reasons.append("negative distance from optimum")
        if not all("normalized_speed" in r for r in rows): reasons.append("missing source-axis derivative")
        cvs=[]
        for p in sorted({r["parameters_per_gate"] for r in rows}):
            by_distance={d:np.mean([r["gamma_times_100"] for r in rows if r["parameters_per_gate"]==p and r["distance"]==d]) for d in {r["distance"] for r in rows if r["parameters_per_gate"]==p}}
            values=np.asarray(list(by_distance.values()),dtype=float)
            if len(values)>1 and np.mean(values):cvs.append(float(np.std(values,ddof=1)/abs(np.mean(values))))
        metrics["gamma_distance_cv_by_parameters_per_gate"]=cvs
        if mode!="smoke" and any(value>.15 for value in cvs):reasons.append("convergence-rate distance-independence tolerance exceeded")
    status="SMOKE_VALIDATED" if not reasons and mode=="smoke" else ("READY_TO_PLOT" if not reasons else "SCIENTIFIC_VALIDATION_FAILED")
    result={"schema_version":"google-pure-v7-figure5-validation.v1","panel":panel,"mode":mode,
            "row_count":len(rows),"valid":not reasons,"blocking_reasons":reasons,"metrics":metrics,"status":status}
    root=figure5_root()/"reports"; atomic_json(root/f"panel_{panel[-1]}_validation.json",result)
    atomic_text(root/f"panel_{panel[-1]}_validation.md",f"# Panel {panel} Validation\n\nStatus: **{status}**\n\n"+"\n".join(f"- {r}" for r in reasons)+"\n")
    return result

def write_logical_contract() -> dict:
    result={"schema_version":"google-pure-v7-figure5-logical-metric.v1","orientation":"lower_is_better",
            "logical_metric":"failures / trials","logical_floor":"independent protocol value, never learned trace minimum",
            "normalized_progress":"(initial-value)/(initial-floor)","initial":0.0,"optimum":1.0}
    root=figure5_root()/"source_contract"; atomic_json(root/"logical_metric_contract.json",result)
    atomic_text(root/"logical_metric_contract.md","# Logical Metric Contract\n\nLower is better; normalized progress is 0 initially and 1 at the independently specified floor.\n")
    return result
