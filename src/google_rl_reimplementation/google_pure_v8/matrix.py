from __future__ import annotations

from typing import Any
import numpy as np

from google_rl_reimplementation.google_pure_v7.config import canonical_hash

from .common import write_report
from .diagnostics import run_cell


def run_compact_fault_matrix() -> dict[str,Any]:
    rows=[];frequencies=(1/300,1/150,1/60);entropies=(0.0,.0004,.02);phases=(0.0,2*np.pi/3,4*np.pi/3)
    for f in frequencies:
        for entropy in entropies:
            for phase in phases:
                cell=run_cell(frequency=f,entropy=entropy,phase=float(phase),epochs=72,candidates=12,cycles=3000,seed=14301+len(rows))
                rows.append({"frequency":f,"entropy":entropy,"phase":float(phase),"policy_evaluations":cell["costs"],
                  "normalized_improvements":cell["improvements"],**cell["decomposition"],"initial_sigma":cell["mean_scale"]["initial"],
                  "mean_sigma":cell["mean_scale"]["time_average"],"final_sigma":cell["mean_scale"]["final"],
                  "fraction_at_floor":cell["fraction_scale_at_floor"],"entropy_gradient_norm":cell["entropy_gradient_norm"],
                  "reward_gradient_norm":cell["reward_gradient_norm"],"native_displacement":cell["candidate_native_displacement_rms"],
                  "normalized_variance":cell["candidate_normalized_variance"],"native_variance":cell["candidate_native_variance"],
                  "raw_policy_accounting":cell["policy_accounting"],"clipping_fraction":cell["clipping_fraction"],"tracking_amplitude":cell["tracking_amplitude"],"phase_lag_radians":cell["phase_lag_radians"],
                  "period_count":cell["complete_periods"],"burn_in":cell["burn_in_epochs"],"analysis_window":cell["analysis_window"],
                  "behaviour_policy_hash":cell["behaviour_policy_hashes"][0],"current_policy_hash":cell["post_update_policy_hashes"][-1]})
    classifications=[]
    if any(r["normalized_improvements"]["oracle_with_production_scale"]<=0 for r in rows): classifications.append("EXPLORATION_FLOOR_FAILURE")
    entropy_scale={e:np.mean([r["final_sigma"] for r in rows if r["entropy"]==e]) for e in entropies}
    if np.ptp(list(entropy_scale.values()))<1e-3: classifications.append("ENTROPY_AXIS_NOT_OPERATIONAL")
    if any(r["period_count"]<1 for r in rows): classifications.append("TEMPORAL_ALIASING")
    if np.median([r["normalized_improvements"]["learned_mean"] for r in rows])<=0: classifications.append("MEAN_TRACKING_BANDWIDTH_FAILURE")
    if not classifications: classifications=["NO_IMPLEMENTATION_FAULT_DETECTED"]
    result={"schema_version":"google-pure-v8-compact-fault-matrix.v1","protocol":{"frequencies":list(frequencies),"entropies":list(entropies),
      "phases":list(map(float,phases)),"policy_evaluations":["fixed deterministic","oracle deterministic","oracle with production scale","learned mean deterministic","full sampled candidates"],
      "epochs":72,"candidates":12,"cycles_per_candidate":3000},"protocol_hash":canonical_hash({"f":frequencies,"e":entropies,"p":list(map(float,phases)),"epochs":72,"candidates":12,"cycles":3000}),
      "rows":rows,"entropy_final_scale_means":entropy_scale,"dominant_classifications":classifications,
      "full_surface_gate_pass":classifications==["NO_IMPLEMENTATION_FAULT_DETECTED"],
      "blocking_reasons":classifications if classifications!=["NO_IMPLEMENTATION_FAULT_DETECTED"] else []}
    return write_report("compact_fault_isolation_matrix",result,"Compact Figure 5a Fault-isolation Matrix")
