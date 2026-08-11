"""Executable metric, PPO, coordinate, and temporal contracts."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from hdfa_rl_suite.google_pure_v6.factor_graph import local_importance_ratios
from hdfa_rl_suite.google_pure_v6.policy import component_log_probability

from .common import write_report


def normalized_edr_improvement(fixed: float, candidate: float, oracle: float, *, resolution: float = 0.0) -> float:
    values=np.asarray([fixed,candidate,oracle],dtype=float)
    denominator=float(fixed-oracle)
    if not np.all(np.isfinite(values)) or denominator <= max(0.0,float(resolution)):
        raise ValueError("Figure 5a normalization denominator is non-positive, non-finite, or unresolved")
    return float((fixed-candidate)/denominator)


def cost_decomposition(fixed: float, candidate: float, mean: float, oracle: float) -> dict[str,float]:
    d_fixed=float(fixed-oracle);d_tracking=float(mean-oracle);d_exploration=float(candidate-mean)
    direct=normalized_edr_improvement(fixed,candidate,oracle)
    decomposed=float(1-(d_tracking+d_exploration)/d_fixed)
    if not np.isclose(direct,decomposed,rtol=1e-12,atol=1e-12): raise RuntimeError("EDR decomposition identity failed")
    return {"d_fixed":d_fixed,"d_tracking":d_tracking,"d_exploration":d_exploration,
            "direct_improvement":direct,"decomposition_improvement":decomposed}


def local_ratio(actions: np.ndarray, current_mean: np.ndarray, current_log_scale: np.ndarray,
                behaviour_mean: np.ndarray, behaviour_log_scale: np.ndarray, mask: np.ndarray) -> np.ndarray:
    frozen=component_log_probability(np.asarray(actions),np.asarray(behaviour_mean).copy(),np.asarray(behaviour_log_scale).copy())
    frozen.setflags(write=False)
    return local_importance_ratios(actions,current_mean,current_log_scale,frozen,mask)


def frequency_contract(frequency_cycles_per_epoch: float, phase: float = 0.0) -> dict[str,float]:
    f=float(frequency_cycles_per_epoch)
    if not np.isfinite(f) or f <= 0: raise ValueError("frequency must be positive cycles per epoch")
    return {"frequency_cycles_per_epoch":f,"angular_frequency_radians_per_epoch":2*math.pi*f,
            "period_epochs":1/f,"initial_phase_radians":float(phase)}


def build_mathematical_contracts() -> dict[str,Any]:
    result={"schema_version":"google-pure-v8-mathematical-contracts.v1",
      "figure5a":{"metric":"normalized_candidate_edr_improvement","numerator":"fixed_edr - candidate_edr",
        "denominator":"fixed_edr - oracle_edr","fixed_substitution_expected":0.0,"oracle_substitution_expected":1.0,
        "higher_is_better":True,"denominator_gate":"positive, finite, statistically resolved"},
      "ppo":{"ratio":"product over i in S_j of pi_current_i / pi_behaviour_i","behaviour_snapshot":"immutable value copy",
        "local_mask_only":True,"denominator_detached":True},
      "coordinates":{"native":"u0 + s*x","normalized":"(u-u0)/s","native_sigma":"abs(s)*normalized_sigma"},
      "temporal":{"optimum":"u0 + A*sin(2*pi*f*t + phase)","frequency_unit":"cycles_per_epoch"},
      "entropy":{"optimized_variable":"log_sigma","gradient_per_coordinate":1.0,"counted":"exactly once globally"}}
    return write_report("mathematical_contracts",result,"Machine-readable Mathematical Contracts")

