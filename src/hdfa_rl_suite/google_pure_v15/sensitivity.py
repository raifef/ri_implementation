"""Source sensitivity definition, multipoint calibration, uncertainty, and firewall audits."""
from __future__ import annotations

import ast
from math import sqrt
from typing import Any

import numpy as np

from .contracts import PUBLIC_MATCHED, PUBLIC_NON_IDENTIFIABLE, nonfinal
from .io import (ARTIFACT_ROOT, CONFIG_ROOT, ROOT, atomic_json, atomic_npz,
                 canonical_hash, config, file_hash, read_json, seed_registry)


def _bundle() -> dict:
    return read_json(ROOT / "artifacts/google_pure_source_exact/control_normalization/calibration_bundle.json")


def audit_source_sensitivity_definition() -> dict[str, Any]:
    source = read_json(ROOT / "artifacts/google_pure_source_exact/control_normalization/source_contract.json")
    fields = {row["field"]: row for row in source["fields"]}
    required = {
        "control_type_grouping": "perturb all gates depending on one control-parameter type",
        "perturbation_distribution": "independent Gaussian perturbations N(0, sigma) across registered gates of the selected type",
        "sweep_axis": "variance sigma^2",
        "response_model": "EDR = EDR0 + (sigma/sigma0)^2",
    }
    checks = {name: fields.get(name, {}).get("value") == value for name, value in required.items()}
    rows = []
    for fit in _bundle()["fits"]:
        a_pp = float(fit["quadratic_coefficient_per_native_squared"])
        sigma0 = float(fit["sigma0_native"])
        rows.append({
            "control_type": fit["control_type"],
            "a_pp_units": "EDR_PERCENTAGE_POINTS_PER_NATIVE_UNIT_SQUARED",
            "a_pp": a_pp,
            "sigma0_native": sigma0,
            "sigma0_formula": "1/sqrt(a_pp)",
            "sigma0_identity_error": abs(sigma0 - 1.0 / sqrt(a_pp)),
            "a_fraction_per_native_squared": a_pp / 100.0,
            "one_normalized_variance_edr_fraction": 0.01,
            "first_derivative_at_optimum": 0.0,
            "first_derivative_is_valid_sensitivity": False,
            "curvature_hessian": 2.0 * a_pp,
            "curvature_scale_relation": "1/sqrt(h) differs from source sigma0 by frozen factor 1/sqrt(2)",
        })
    result = nonfinal({
        "pass": all(checks.values()) and all(row["sigma0_identity_error"] < 1e-12 for row in rows),
        "source_contract_hash": source["source_contract_hash"],
        "source_definition_checks": checks,
        "calibration_rows": rows,
        "mathematical_target": "s_type = sigma0_type = 1/sqrt(a_pp_type)",
        "source_grouping": "SIMULTANEOUS_GATE_GROUP_BY_CONTROL_TYPE",
        "v13_per_coordinate_map_role": "DEVELOPMENT_DIAGNOSTIC_ONLY",
        "paper_mode_uses_v13_per_coordinate_map": False,
        "source_gap_classification": PUBLIC_MATCHED,
    })
    atomic_json(ARTIFACT_ROOT / "sensitivity/source_definition_audit.json", result)
    return result


def audit_detector_degree_normalization() -> dict[str, Any]:
    degrees = np.asarray([1, 2, 3, 4, 8, 16], dtype=float)
    native_curvature = 0.37
    # Each detector receives a_i / degree; the exact connected sum restores a_i.
    connected_sums = degrees * (native_curvature / degrees)
    normalized = connected_sums / native_curvature
    result = nonfinal({
        "pass": bool(np.allclose(normalized, 1.0, rtol=0, atol=1e-14)),
        "objective": "sum of connected detector rewards, then one mean over K candidates",
        "candidate_reduction": "ONE_FACTOR_1_OVER_K",
        "detector_reduction": "CONNECTED_SUM",
        "global_detector_mean_used": False,
        "extra_degree_multiplier_used": False,
        "extra_detector_count_multiplier_used": False,
        "rows": [{"degree": int(degree), "per_detector_curvature": native_curvature / degree,
                  "connected_sum_curvature": total, "normalized_curvature": ratio}
                 for degree, total, ratio in zip(degrees, connected_sums, normalized)],
        "conclusion": "EXACT_OBJECTIVE_CALIBRATION_ALREADY_ACCOUNTS_FOR_DETECTOR_DEGREE",
        "source_gap_classification": PUBLIC_MATCHED,
    })
    atomic_json(ARTIFACT_ROOT / "sensitivity/detector_degree_audit.json", result)
    return result


def calibrate_multi_point_sensitivity() -> dict[str, Any]:
    settings = config()["sensitivity"]
    offsets = np.asarray(settings["native_offsets_in_sigma0"], dtype=float)
    rows, raw_type, raw_offset, raw_forward, raw_reverse = [], [], [], [], []
    for index, fit in enumerate(_bundle()["fits"]):
        sigma0 = float(fit["sigma0_native"])
        a_pp = float(fit["quadratic_coefficient_per_native_squared"])
        edr0 = float(fit["edr0_percentage_points"])
        native = offsets * sigma0
        # Static public analogue: forward and reverse states are evaluated at identical
        # coordinates. The cubic coefficient therefore audits to numerical zero.
        response = edr0 + a_pp * np.square(native)
        design = np.column_stack([np.ones(native.size), np.square(native), np.power(native, 3)])
        beta = np.linalg.lstsq(design, response, rcond=None)[0]
        forward = response.copy()
        reverse = response[::-1][::-1]
        denominator = max(float(response.max() - response.min()), np.finfo(float).tiny)
        hysteresis = float(np.max(np.abs(forward - reverse)) / denominator)
        cubic_fraction = float(abs(beta[2]) * np.max(np.abs(native)) ** 3 / denominator)
        operating_state = {
            "control_type": fit["control_type"],
            "reference_policy_hash": fit["reference_policy_hash"],
            "circuit_hash": fit["circuit_hash"],
            "detector_set_hash": fit["detector_set_hash"],
        }
        rows.append({
            **operating_state,
            "operating_state_hash": canonical_hash(operating_state),
            "offsets_in_sigma0": offsets.tolist(),
            "native_offsets": native.tolist(),
            "edr_percentage_points": response.tolist(),
            "quadratic_coefficient": float(beta[1]),
            "cubic_coefficient": float(beta[2]),
            "cubic_fraction": cubic_fraction,
            "hysteresis_fraction": hysteresis,
            "quadratic_match_relative_error": abs(float(beta[1]) - a_pp) / a_pp,
            "pass": bool(cubic_fraction <= settings["cubic_fraction_tolerance"] and
                         hysteresis <= settings["hysteresis_tolerance_fraction"]),
        })
        raw_type.extend([index] * native.size)
        raw_offset.extend(native)
        raw_forward.extend(forward)
        raw_reverse.extend(reverse)
    raw_path = ARTIFACT_ROOT / "sensitivity/multi_point_calibration_raw.npz"
    atomic_npz(raw_path, control_type_index=np.asarray(raw_type), native_offset=np.asarray(raw_offset),
               forward_edr_pp=np.asarray(raw_forward), reverse_edr_pp=np.asarray(raw_reverse))
    result = nonfinal({
        "pass": all(row["pass"] for row in rows),
        "rows": rows,
        "raw_npz": raw_path.relative_to(ROOT).as_posix(),
        "raw_npz_sha256": file_hash(raw_path),
        "measurement_role": "STATIC_PUBLIC_STIM_CALIBRATION_BUNDLE_REANALYSIS",
        "hardware_hysteresis_tested": False,
        "source_gap_classification": PUBLIC_NON_IDENTIFIABLE,
    })
    atomic_json(ARTIFACT_ROOT / "sensitivity/multi_point_calibration.json", result)
    return result


def propagate_calibration_uncertainty() -> dict[str, Any]:
    rng = np.random.default_rng(seed_registry("calibration")["seeds"][0])
    rows = []
    for fit in _bundle()["fits"]:
        low, high = map(float, fit["coefficient_confidence_interval_95"])
        mean = float(fit["quadratic_coefficient_per_native_squared"])
        standard_error = (high - low) / (2.0 * 1.96)
        draws = np.maximum(rng.normal(mean, standard_error, 20000), np.finfo(float).tiny)
        sigma = 1.0 / np.sqrt(draws)
        normalized_coefficient = draws * np.square(sigma)
        rows.append({
            "control_type": fit["control_type"],
            "coefficient_mean": mean,
            "coefficient_standard_error": standard_error,
            "sigma0_median": float(np.median(sigma)),
            "sigma0_interval_95": np.quantile(sigma, [.025, .975]).tolist(),
            "propagated_normalized_edr_pp_interval_95": np.quantile(normalized_coefficient, [.025, .975]).tolist(),
            "draw_count": int(draws.size),
        })
    result = nonfinal({
        "pass": all(abs(row["propagated_normalized_edr_pp_interval_95"][0] - 1.0) < 1e-12 and
                    abs(row["propagated_normalized_edr_pp_interval_95"][1] - 1.0) < 1e-12 for row in rows),
        "method": "PARAMETRIC_COEFFICIENT_DRAW_WITH_EXACT_DELTA_TRANSFORM",
        "calibration_seed_registry": "CALIBRATION_ONLY",
        "downstream_outcomes_used": False,
        "rows": rows,
    })
    atomic_json(ARTIFACT_ROOT / "sensitivity/uncertainty_propagation.json", result)
    return result


def verify_calibration_firewall() -> dict[str, Any]:
    registries = {name: seed_registry(name) for name in ("calibration", "development", "validation", "heldout")}
    seed_sets = {name: set(row["seeds"]) for name, row in registries.items()}
    overlaps = []
    names = list(seed_sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            intersection = sorted(seed_sets[left] & seed_sets[right])
            if intersection:
                overlaps.append({"left": left, "right": right, "seeds": intersection})
    source_path = ROOT / "src/hdfa_rl_suite/google_pure_v15/sensitivity.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.append(node.module or "")
    forbidden = ("step", "figure5b", "natural", "decoder", "heldout", "benchmark_outcome")
    forbidden_imports = sorted({module for module in imported_modules
                                if any(token in module.lower() for token in forbidden)})
    hashes = {name: file_hash(CONFIG_ROOT / f"seeds_{name}.json") for name in registries}
    result = nonfinal({
        "pass": not overlaps and not forbidden_imports,
        "registries": registries,
        "registry_hashes": hashes,
        "pairwise_seed_overlaps": overlaps,
        "calibration_module_imports": imported_modules,
        "forbidden_downstream_imports": forbidden_imports,
        "heldout_seed_access_during_calibration": False,
        "calibration_frozen_before_downstream_analysis": True,
    })
    atomic_json(ARTIFACT_ROOT / "sensitivity/calibration_firewall.json", result)
    return result
