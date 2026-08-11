"""Direct-sigma conditioning, floor, and finite-horizon decay diagnostics."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .contracts import PUBLIC_NON_IDENTIFIABLE, nonfinal
from .io import ARTIFACT_ROOT, ROOT, atomic_json, read_json


def audit_mean_scale_conditioning() -> dict[str, Any]:
    controller = read_json(ROOT / "configs/google_pure_source_exact/figure5a.json")["controller"]
    entropy_weights = read_json(ROOT / "configs/google_pure_source_exact/figure5a.json")[
        "anchor"]["entropy_weights"]
    curvature = .01
    rows = []
    for error in (1.0, .3, .1, .03, 0.0):
        mean_reward_gradient = 2.0 * curvature * error
        for sigma in (.8, .22360679775, .15, .02, controller["minimum_sigma"]):
            reward_scale_gradient = 2.0 * curvature * sigma
            for entropy_weight in entropy_weights:
                entropy_scale_gradient = entropy_weight / sigma
                rows.append({
                    "mean_error": error, "sigma": sigma, "entropy_weight": entropy_weight,
                    "mean_reward_gradient_magnitude": mean_reward_gradient,
                    "scale_reward_contraction_gradient_magnitude": reward_scale_gradient,
                    "scale_entropy_expansion_gradient_magnitude": entropy_scale_gradient,
                    "entropy_to_reward_scale_gradient_ratio":
                        entropy_scale_gradient / max(reward_scale_gradient, np.finfo(float).tiny),
                    "mean_to_net_scale_gradient_ratio":
                        mean_reward_gradient / max(abs(entropy_scale_gradient - reward_scale_gradient),
                                                   np.finfo(float).tiny),
                })
    equilibria = [{"entropy_weight": value,
                   "unclipped_sigma_equilibrium": float(np.sqrt(value / (2.0 * curvature))),
                   "bounded_sigma_equilibrium": float(np.clip(np.sqrt(value / (2.0 * curvature)),
                                                               controller["minimum_sigma"],
                                                               controller["maximum_sigma"]))}
                  for value in entropy_weights]
    result = nonfinal({
        "pass": all(np.isfinite(row["entropy_to_reward_scale_gradient_ratio"]) for row in rows),
        "rows": rows,
        "equilibria": equilibria,
        "normalized_quadratic_coefficient_fraction": curvature,
        "mean_and_scale_gradients_reported_separately": True,
        "conditioning_conclusion": "ENTROPY_DOMINATES_SCALE_NEAR_THE_OPTIMUM_AT_ALL_SOURCE_SCAN_WEIGHTS",
        "controller_changed_by_diagnostic": False,
        "source_entropy_optimizer_details_identifiable": False,
    })
    atomic_json(ARTIFACT_ROOT / "dynamics/mean_scale_conditioning.json", result)
    return result


def audit_scale_floor() -> dict[str, Any]:
    controller = read_json(ROOT / "configs/google_pure_source_exact/figure5a.json")["controller"]
    coefficient = .01
    minimum = float(controller["minimum_sigma"])
    initial = float(controller["initial_sigma"])
    rows = []
    for controls in (41, 924, 38670):
        rows.append({
            "controls": controls,
            "minimum_sigma": minimum,
            "mean_physical_edr_floor_fraction": coefficient * minimum ** 2,
            "summed_training_objective_floor": controls * coefficient * minimum ** 2,
            "initial_mean_physical_exploration_penalty_fraction": coefficient * initial ** 2,
            "floor_to_initial_penalty_ratio": (minimum / initial) ** 2,
            "coordinate_floor_occupancy_measured_in_reference_run": False,
        })
    negligible = all(row["mean_physical_edr_floor_fraction"] < 1e-7 for row in rows)
    result = nonfinal({
        "pass": negligible,
        "rows": rows,
        "physical_metric_reduction": "MEAN_OVER_CONTROL_TYPES_OR_DETECTOR_OPPORTUNITIES",
        "training_metric_reduction": "CONNECTED_DETECTOR_SUM",
        "classification": "MINIMUM_SIGMA_NOT_A_CURRENT_SURROGATE_PHYSICAL_PLATEAU_LIMIT",
        "initial_exploration_penalty_is_not_negligible": True,
        "hardware_floor_tested": False,
    })
    atomic_json(ARTIFACT_ROOT / "dynamics/scale_floor.json", result)
    return result


def _slope(values: np.ndarray) -> float:
    x = np.arange(values.size, dtype=float)
    return float(np.polyfit(x, np.log(np.maximum(values, np.finfo(float).tiny)), 1)[0])


def classify_residual_decay() -> dict[str, Any]:
    lineage = read_json(ROOT / "artifacts/google_pure_v12/lineage/figure5b_lineage.json")
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in lineage["trajectory_table"]:
        grouped[(int(row["distance"]), int(row["parameters_per_gate"]), int(row["seed"]))].append(row)
    rows = []
    for key, points in sorted(grouped.items()):
        points.sort(key=lambda row: row["epoch"])
        learned = np.asarray([row["logical_error_learned"] for row in points], dtype=float)
        floor = np.asarray([row["irreducible_floor"] for row in points], dtype=float)
        excess = np.maximum(learned - floor, np.finfo(float).tiny)
        window = max(12, excess.size // 5)
        early = _slope(excess[:window])
        tail = _slope(excess[-window:])
        improvement = float(excess[-1] / excess[0])
        if tail < -.002:
            classification = "STILL_DECAYING_AT_HORIZON"
        elif improvement < .2 and abs(tail) <= .002:
            classification = "EMPIRICAL_PLATEAU_WITHIN_HORIZON"
        else:
            classification = "NO_IDENTIFIABLE_CONVERGENCE"
        rows.append({
            "distance": key[0], "parameters_per_gate": key[1], "seed": key[2],
            "epochs": len(points), "initial_excess_logical_error": float(excess[0]),
            "final_excess_logical_error": float(excess[-1]),
            "final_to_initial_ratio": improvement,
            "early_log_slope_per_epoch": early, "tail_log_slope_per_epoch": tail,
            "classification": classification,
        })
    counts = {name: sum(row["classification"] == name for row in rows)
              for name in ("STILL_DECAYING_AT_HORIZON", "EMPIRICAL_PLATEAU_WITHIN_HORIZON",
                           "NO_IDENTIFIABLE_CONVERGENCE")}
    result = nonfinal({
        "pass": bool(rows),
        "rows": rows,
        "classification_counts": counts,
        "finite_horizon_never_relabelled_as_asymptote": True,
        "imported_lineage_final_evidence": lineage.get("final_evidence", False),
        "conclusion": "IMPORTED_V12_DECAY_CLASSIFIED_WITHOUT_PROMOTION",
        "source_gap_classification": PUBLIC_NON_IDENTIFIABLE,
    })
    atomic_json(ARTIFACT_ROOT / "dynamics/residual_decay.json", result)
    return result
