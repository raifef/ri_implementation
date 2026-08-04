"""Level-6 sinusoidal steering-frequency sweep."""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from .config import GoogleRLConfig
from .drift_tracking import _run_agent, one_control_landscape


DEFAULT_PERIODS = (600, 400, 300, 225, 200, 175, 150, 125, 100, 75)


def run_steering_frequency_sweep(
        config: GoogleRLConfig, *, seed: int = 5501,
        periods_epochs: Sequence[int] = DEFAULT_PERIODS,
        minimum_improvement: float = 0.0) -> dict[str, Any]:
    rows = []
    for index, period in enumerate(periods_epochs):
        epochs = max(600, 2*int(period))
        landscape = one_control_landscape(
            lambda epoch, period=period: .80*math.sin(2*math.pi*epoch/period),
            curvature=.015)
        run = _run_agent(config, landscape, epochs, seed=seed+101*index)
        trajectory = run["trajectory"]
        warm = min(int(period)//2, epochs//4)
        fixed = float(np.mean([item["fixed_edr"] for item in trajectory[warm:]]))
        oracle = float(np.mean([item["oracle_edr"] for item in trajectory[warm:]]))
        stochastic = float(np.mean([
            item["aggregate_exploration_edr"] for item in trajectory[warm:]]))
        learned = float(np.mean([item["mean_policy_edr"] for item in trajectory[warm:]]))
        denominator = oracle-fixed
        stochastic_improvement = ((stochastic-fixed)/denominator
                                  if abs(denominator) > 1e-15 else 0.)
        learned_improvement = ((learned-fixed)/denominator
                               if abs(denominator) > 1e-15 else 0.)
        rows.append({
            "period_epochs": int(period),
            "frequency_per_epoch": 1/float(period),
            "epochs": epochs,
            "fixed_edr": fixed,
            "oracle_edr": oracle,
            "aggregate_exploration_edr": stochastic,
            "mean_policy_edr": learned,
            "normalized_stochastic_improvement": stochastic_improvement,
            "normalized_mean_policy_improvement": learned_improvement,
            "beats_fixed": stochastic_improvement > minimum_improvement,
        })
    passing = [row for row in rows if row["beats_fixed"]]
    critical = min((row["period_epochs"] for row in passing), default=None)
    slower = [row for row in rows if row["period_epochs"] >= 300]
    faster = [row for row in rows if row["period_epochs"] <= 125]
    gates = {
        "slow_drift_is_trackable": bool(slower and max(
            row["normalized_stochastic_improvement"] for row in slower) > .15),
        "benefit_declines_with_frequency": rows[0]["normalized_stochastic_improvement"]
        > rows[-1]["normalized_stochastic_improvement"],
        "fast_drift_not_sufficiently_trackable": bool(faster and any(
            not row["beats_fixed"] for row in faster)),
        "anchored_transition_near_public_period": critical is not None and 100 <= critical <= 225,
        "mean_and_exploration_reported_separately": all(
            row["mean_policy_edr"] <= row["aggregate_exploration_edr"]+1e-12
            for row in rows),
    }
    return {
        "schema_version": "google-rl-steering-frequency.v1",
        "evidence_layer": "public-anchor-aligned analytical detector surrogate",
        "config_name": config.name,
        "public_critical_period_anchor_epochs": 150,
        "preregistered_acceptable_period_range": [100, 225],
        "minimum_normalized_improvement": minimum_improvement,
        "critical_period_epochs": critical,
        "critical_frequency_per_epoch": 1/critical if critical else None,
        "gates": gates,
        "passed": all(gates.values()),
        "frequency_rows": rows,
    }
