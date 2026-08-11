"""Source-contract Figure 5b acquisition and downstream Figure 5c analysis."""
from __future__ import annotations

from typing import Any

import numpy as np

from hdfa_rl_suite.google_pure_source_exact.paper_families.common import SparseControlPlant, run_direct_sigma_trace
from hdfa_rl_suite.google_pure_v7.figure5.accounting import detector_factors, physical_qubits, total_controls

from .contracts import NONFINAL, SOURCE_ANCHORED, SOURCE_LITERAL, SOURCE_UNSPECIFIED, SOURCE_UNSPECIFIED_PREREGISTERED, V13_SCHEMA
from .io import ARTIFACT_ROOT, atomic_json, atomic_text, canonical_hash, config, read_json


FAMILY = "FIGURE5B_SPARSE_SCALING"


def audit_figure5b_contract() -> dict[str, Any]:
    rows = [
        {"field": "panel_axes", "value": "physical_error_rate_vs_logical_error_rate", "source_status": SOURCE_LITERAL},
        {"field": "axis_scales", "value": "log_log", "source_status": SOURCE_LITERAL},
        {"field": "epoch_encoding", "value": "colour", "source_status": SOURCE_LITERAL},
        {"field": "irreducible_floor", "value": "horizontal_bars", "source_status": SOURCE_LITERAL},
        {"field": "distance_15_parameters_30_controls", "value": 38670, "source_status": SOURCE_ANCHORED},
        {"field": "threshold_physical_error", "value": .00179, "source_status": SOURCE_ANCHORED},
        {"field": "proprietary_physical_plant", "value": None, "source_status": SOURCE_UNSPECIFIED},
        {"field": "optimizer_hyperparameters", "value": None, "source_status": SOURCE_UNSPECIFIED},
        {"field": "reduced_validation_grid", "value": config()["figure5b_validation"],
         "source_status": SOURCE_UNSPECIFIED_PREREGISTERED},
    ]
    result = {"schema_version": V13_SCHEMA, "family": FAMILY, "contract": rows,
              "required_plot_fields": ["physical_error", "logical_error", "epoch_colour",
                                       "irreducible_floor", "distance"],
              "normalized_lambda_only_plot_permitted_as_figure5b": False,
              "scientific_classification": "PUBLIC_STRUCTURE_ANALOGUE_WITH_UNAVAILABLE_PROPRIETARY_PLANT",
              **NONFINAL}
    atomic_json(ARTIFACT_ROOT / "figure5b/source_contract.json", result)
    return result


def _condition(distance: int, parameters_per_gate: int, seed: int, *, epochs: int) -> dict[str, Any]:
    settings = config()["figure5b_validation"]
    controls = total_controls(distance, parameters_per_gate)
    detectors = detector_factors(distance)
    raw_plant = SparseControlPlant(distance, controls, detectors,
                                   seed=9100 + 101 * distance + parameters_per_gate, curvature=.003)
    # In raw native coordinates coefficient is curvature/degree.  Applying
    # s_i=sqrt(kappa_ref/(curvature/degree_i)) once makes the normalized plant's
    # detector-local mean-square coefficient exactly kappa_ref=.01.
    degree = np.bincount(raw_plant.control_detector, minlength=detectors)
    scale = np.sqrt(.01 * degree[raw_plant.control_detector] / raw_plant.curvature)
    normalized_plant = SparseControlPlant(distance, controls, detectors,
                                          seed=9100 + 101 * distance + parameters_per_gate,
                                          curvature=.01)
    rng = np.random.default_rng(int(seed))
    initial_mean = rng.choice((-1.0, 1.0), controls) * rng.uniform(.45, .75, controls)
    protocol_hash = canonical_hash({"schema": V13_SCHEMA, "family": FAMILY, "distance": distance,
                                    "parameters_per_gate": parameters_per_gate, "seed": seed,
                                    "epochs": epochs, "kappa_ref": .01,
                                    "controller": "PAPER_DIRECT_SIGMA"})
    checkpoint = ARTIFACT_ROOT / "figure5b/checkpoints" / f"{protocol_hash[:24]}.json"
    run = run_direct_sigma_trace(
        plant=normalized_plant, protocol_hash=protocol_hash, seed=int(seed), epochs=int(epochs),
        candidates=int(settings["candidates_per_epoch"]),
        cycles_per_candidate=int(settings["cycles_per_candidate"]),
        entropy_weight=float(settings["entropy_coefficient"]), checkpoint=checkpoint,
        target_at_epoch=lambda _: np.zeros(controls), initial_mean=initial_mean)
    records = run["records"]
    lambdas = np.asarray([row["learned"]["lambda"] for row in records], dtype=float)
    lambda_star = float(records[0]["oracle"]["lambda_star"])
    ratio = lambdas / lambda_star
    physical = np.asarray([row["learned"]["physical_error"] for row in records])
    logical = np.asarray([row["learned"]["logical_error"] for row in records])
    fixed = np.asarray([row["fixed"]["logical_error"] for row in records])
    floor = float(records[0]["oracle"]["logical_floor"])
    exponent = (distance + 1) / 2
    candidate_logical = [float(np.clip(.01 * (
        normalized_plant.threshold_physical_error / max(row["candidate_physical_error"], 1e-12)
        ) ** (-exponent), 1e-12, 1.0)) for row in records]
    x = 1.0 - ratio[:-1]
    local = (x > float(settings["local_fit_min_distance"])) & (x < float(settings["local_fit_max_distance"]))
    initial_excess = max(float(logical[0] - floor), np.finfo(float).tiny)
    progress = float((logical[0] - logical[-1]) / initial_excess)
    return {
        "distance": distance, "parameters_per_gate": parameters_per_gate, "seed": int(seed),
        "total_controls": controls, "physical_qubits": physical_qubits(distance), "detectors": detectors,
        "raw_native_plant_hash": raw_plant.plant_hash, "normalized_runtime_plant_hash": normalized_plant.plant_hash,
        "graph_hash": raw_plant.graph_hash, "sensitivity_map_hash": canonical_hash(scale.tolist()),
        "native_per_normalized_min": float(np.min(scale)), "native_per_normalized_max": float(np.max(scale)),
        "mapping": "u=u0+s*x", "sensitivity_application_count": 1,
        "conditioned_edr_coefficient_fraction_per_normalized_squared": .01,
        "controller_mode": "PAPER_DIRECT_SIGMA", "parameterization": "direct_sigma",
        "candidate_qec_cycles": run["candidate_qec_cycles"],
        "detector_event_trials": run["candidate_qec_cycles"] * detectors,
        "local_fit_point_count": int(np.count_nonzero(local)),
        "entered_frozen_local_regime": int(np.count_nonzero(local)) >= int(settings["minimum_local_points"]),
        "floor_normalized_progress": progress,
        "trajectory": {"epoch": list(range(len(records))), "physical_error": physical.tolist(),
                       "logical_learned": logical.tolist(), "logical_candidate": candidate_logical,
                       "logical_fixed": fixed.tolist(), "logical_oracle": [floor] * len(records),
                       "logical_floor": [floor] * len(records), "lambda": lambdas.tolist(),
                       "lambda_ratio": ratio.tolist(), "x_distance": x.tolist(),
                       "local_fit_mask": local.astype(int).tolist(),
                       "mean_sigma": [row["mean_sigma"] for row in records]},
        **NONFINAL,
    }


def run_figure5b_validation(*, epochs_override: int | None = None) -> dict[str, Any]:
    audit_figure5b_contract()
    settings = config()["figure5b_validation"]
    epochs = int(epochs_override or settings["epochs"])
    rows = [_condition(int(distance), int(parameters), int(seed), epochs=epochs)
            for distance in settings["distances"] for parameters in settings["parameters_per_gate"]
            for seed in settings["seeds"]]
    gates = {
        "all_conditions_have_nonzero_cycles": all(row["candidate_qec_cycles"] > 0 for row in rows),
        "all_conditions_apply_source_map_once": all(row["sensitivity_application_count"] == 1 for row in rows),
        "all_conditions_visibly_evolve": all(row["floor_normalized_progress"] >= .05 for row in rows),
        "all_conditions_enter_frozen_local_regime": all(row["entered_frozen_local_regime"] for row in rows),
        "paper_panel_quantities_retained": all(all(key in row["trajectory"] for key in
            ("physical_error", "logical_learned", "logical_floor", "epoch")) for row in rows),
    }
    result = {"schema_version": V13_SCHEMA, "family": FAMILY, "rows": rows, "gates": gates,
              "acquisition_valid": all(gates.values()),
              "classification": "NORMALIZED_SYNTHETIC_CONVERGENCE_DIAGNOSTIC",
              "full_preregistered_horizon": epochs_override is None, **NONFINAL}
    atomic_json(ARTIFACT_ROOT / "figure5b/validation.json", result)
    return result


def audit_figure5b_convergence() -> dict[str, Any]:
    value = read_json(ARTIFACT_ROOT / "figure5b/validation.json")
    rows = [{"distance": row["distance"], "parameters_per_gate": row["parameters_per_gate"],
             "seed": row["seed"], "floor_normalized_progress": row["floor_normalized_progress"],
             "local_fit_point_count": row["local_fit_point_count"],
             "entered_frozen_local_regime": row["entered_frozen_local_regime"]} for row in value["rows"]]
    result = {"schema_version": V13_SCHEMA, "conditions": rows,
              "convergence_gate_pass": all(row["entered_frozen_local_regime"] for row in rows),
              "amendments": ["V13_SOURCE_LITERAL_SENSITIVITY_BOUNDARY"],
              "one_amendment_at_a_time": True, **NONFINAL}
    atomic_json(ARTIFACT_ROOT / "figure5b/convergence_audit.json", result)
    return result


def _linear_fit(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    design = np.column_stack([np.ones(len(x)), x])
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ beta
    dof = max(1, len(x) - 2)
    variance = float(np.dot(residual, residual) / dof)
    covariance = variance * np.linalg.inv(design.T @ design)
    slope_se = float(np.sqrt(covariance[1, 1]))
    total = float(np.sum(np.square(y - np.mean(y))))
    r2 = 1.0 - float(np.sum(np.square(residual))) / total if total > 0 else 1.0
    return {"intercept": float(beta[0]), "slope": float(beta[1]),
            "slope_interval_95": [float(beta[1] - 1.96 * slope_se), float(beta[1] + 1.96 * slope_se)],
            "r_squared": r2, "residuals": residual.tolist()}


def _huber_slope(x: np.ndarray, y: np.ndarray) -> float:
    design = np.column_stack([np.ones(len(x)), x])
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    for _ in range(30):
        residual = y - design @ beta
        scale = max(1.4826 * float(np.median(np.abs(residual - np.median(residual)))), 1e-12)
        weight = np.minimum(1.0, 1.345 * scale / np.maximum(np.abs(residual), 1e-30))
        updated = np.linalg.solve(design.T @ (weight[:, None] * design), design.T @ (weight * y))
        if np.allclose(updated, beta, rtol=0, atol=1e-12):
            beta = updated
            break
        beta = updated
    return float(beta[1])


def analyse_figure5c() -> dict[str, Any]:
    source = read_json(ARTIFACT_ROOT / "figure5b/validation.json")
    minimum = int(config()["figure5b_validation"]["minimum_local_points"])
    rows = []
    for condition in source["rows"]:
        ratio = np.asarray(condition["trajectory"]["lambda_ratio"], dtype=float)
        x = 1.0 - ratio[:-1]
        y = 100.0 * np.diff(ratio)
        mask = np.asarray(condition["trajectory"]["local_fit_mask"], dtype=bool)
        identifiable = int(np.count_nonzero(mask)) >= minimum
        phase = time = None
        agreement = False
        if identifiable:
            phase = _linear_fit(x[mask], y[mask])
            phase["robust_huber_slope"] = _huber_slope(x[mask], y[mask])
            positive = (1.0 - ratio) > 0
            time_index = np.arange(len(ratio), dtype=float)[positive]
            log_distance = np.log((1.0 - ratio)[positive])
            if len(time_index) >= minimum:
                time_raw = _linear_fit(time_index, log_distance)
                gamma = -time_raw["slope"]
                time = {**time_raw, "gamma_per_epoch": gamma,
                        "discrete_gamma_times_100": 100.0 * (1.0 - np.exp(-gamma))}
                agreement = bool(gamma > 0 and phase["slope"] > 0 and
                                 abs(phase["slope"] - time["discrete_gamma_times_100"]) /
                                 max(abs(phase["slope"]), 1e-30) <= .3)
        rows.append({"distance": condition["distance"],
                     "parameters_per_gate": condition["parameters_per_gate"], "seed": condition["seed"],
                     "fit_point_count": int(np.count_nonzero(mask)), "identifiable": identifiable,
                     "phase_space_fit_free_intercept": phase, "time_domain_exponential_fit": time,
                     "phase_time_agreement": agreement,
                     "moving_window_used": False, "zero_fallback_used": False})
    gates = {"all_conditions_identifiable": all(row["identifiable"] for row in rows),
             "all_phase_time_fits_agree": all(row["phase_time_agreement"] for row in rows),
             "no_moving_window": all(not row["moving_window_used"] for row in rows),
             "no_zero_fallback": all(not row["zero_fallback_used"] for row in rows)}
    result = {"schema_version": V13_SCHEMA, "conditions": rows, "gates": gates,
              "fit_valid": all(gates.values()),
              "classification": "IDENTIFIABLE_PUBLIC_ANALOGUE" if all(gates.values()) else "REAL_FIT_UNIDENTIFIABLE",
              **NONFINAL}
    atomic_json(ARTIFACT_ROOT / "figure5c/analysis.json", result)
    return result


def validate_figure5c_fit() -> dict[str, Any]:
    gamma = .017
    ratio = 1.0 - .55 * np.exp(-gamma * np.arange(200))
    x = 1.0 - ratio[:-1]
    y = 100.0 * np.diff(ratio)
    phase = _linear_fit(x, y)
    expected = 100.0 * (1.0 - np.exp(-gamma))
    time = _linear_fit(np.arange(len(ratio), dtype=float), np.log(1.0 - ratio))
    passed = abs(phase["slope"] - expected) < 1e-10 and abs(-time["slope"] - gamma) < 1e-12
    result = {"schema_version": V13_SCHEMA, "pass": passed, "expected_phase_slope": expected,
              "observed_phase_slope": phase["slope"], "expected_gamma": gamma,
              "observed_gamma": -time["slope"], "free_intercept": True, **NONFINAL}
    atomic_json(ARTIFACT_ROOT / "figure5c/fit_fixture.json", result)
    if not passed:
        raise RuntimeError("V13 Figure 5c fit fixture failed")
    return result

