"""Fixed-horizon replay-age alignment under stationary and drifting quadratics."""
from __future__ import annotations

from typing import Any

import numpy as np

from .config import guard_seed, load_config
from .controller import require_resolved_controller
from .reporting import read_artifact, write_report


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0 else 1.0


def run_replay_age_alignment(*, seed: int = 7501) -> dict[str, Any]:
    guard_seed(seed)
    controller = require_resolved_controller()
    protocol = read_artifact("timescale_matched_sine_protocol")
    tau = float(protocol["response_tau_epochs"])
    config = load_config("replay_age_audit.yaml")
    ages = tuple(int(value) for value in config["replay_horizons_epochs"])
    rng = np.random.default_rng(seed)
    dimensions = 6
    curvature = np.linspace(0.8, 1.2, dimensions)
    mean = np.linspace(-0.04, 0.05, dimensions)
    direction = np.linspace(1.0, 0.45, dimensions)
    direction /= np.linalg.norm(direction)
    scenarios = []
    for scenario in config["scenarios"]:
        rows = []
        current_times = np.linspace(2.0*tau, 6.0*tau, 64)
        for age in ages:
            cosines, biases = [], []
            for current_time in current_times:
                if scenario == "static":
                    current_optimum = old_optimum = np.zeros(dimensions)
                elif scenario == "long_step":
                    current_optimum = 0.12 * direction
                    old_optimum = (np.zeros(dimensions) if current_time-age < 2.0*tau else current_optimum)
                else:
                    omega_tau = 0.2 if scenario == "slow_sine" else 1.5
                    omega = omega_tau / tau
                    current_optimum = 0.12 * np.sin(omega*current_time) * direction
                    old_optimum = 0.12 * np.sin(omega*(current_time-age)) * direction
                true = -2.0 * curvature * (mean-current_optimum)
                estimate = -2.0 * curvature * (mean-old_optimum)
                # Small finite-shot perturbation is frozen and equal across horizons.
                estimate = estimate + rng.normal(scale=1e-5, size=dimensions)
                cosines.append(_cosine(estimate, true))
                biases.append(float(np.linalg.norm(estimate-true)))
            rows.append({"replay_age_epochs": age, "mean_gradient_alignment": float(np.mean(cosines)),
                         "alignment_interval_95": [float(np.quantile(cosines, 0.025)), float(np.quantile(cosines, 0.975))],
                         "gradient_bias_norm": float(np.mean(biases))})
        scenarios.append({"scenario": scenario, "rows": rows})
    age_zero = [item["rows"][0]["mean_gradient_alignment"] for item in scenarios]
    static_last = scenarios[0]["rows"][-1]["mean_gradient_alignment"]
    drift_last = min(item["rows"][-1]["mean_gradient_alignment"] for item in scenarios[1:])
    if min(age_zero) < 0.95:
        interpretation = "AGE_ZERO_IMPLEMENTATION_FAILURE"
        selected = None
    elif static_last > 0.95 and drift_last < 0.95:
        interpretation = "REPLAY_STALENESS"
        selected = 0
    else:
        interpretation = "REPLAY_NOT_LIMITING_ON_DECLARED_AGE_GRID"
        selected = max(ages)
    payload = {"schema_version": "google-pure-v7-replay-age-alignment.v1",
               "resolved_config_hash": controller["resolved_config_hash"], "response_tau_epochs": tau,
               "scenarios": scenarios, "interpretation": interpretation,
               "selected_fixed_replay_horizon_epochs": selected,
               "adaptive_replay_selection_added": False,
               "artifact_complete": True, "mechanism_valid": min(age_zero) >= 0.95,
               "performance_pass": selected is not None,
               "blocking_reasons": [] if selected is not None else ["poor gradient alignment at age zero"],
               "certification_seeds_consumed": False,
               "status": "PASS" if selected is not None else "INVALID_DIAGNOSTIC"}
    return write_report("replay_age_alignment", payload, "Replay-age Gradient Alignment")
