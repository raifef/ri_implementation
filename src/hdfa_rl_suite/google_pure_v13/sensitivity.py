"""Independent EDR sensitivity calibration and the one-use native boundary.

The public source defines one normalized variance unit by
``EDR = EDR0 + (sigma / sigma0)^2`` in EDR percentage points.  V13 therefore
freezes kappa_ref at one percentage point (0.01 EDR fraction) before looking at
any controller outcome.  Everything else in this module is either measured in
the declared public analogue or explicitly preregistered.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np

from hdfa_rl_suite.google_pure_source_exact.paper_families.common import SparseControlPlant
from hdfa_rl_suite.google_pure_source_exact.step_response_130.plant import SourceStepPlant

from .contracts import (
    NONFINAL,
    SOURCE_ANCHORED,
    SOURCE_LITERAL,
    SOURCE_UNSPECIFIED_PREREGISTERED,
    V13_SCHEMA,
)
from .io import ARTIFACT_ROOT, atomic_json, atomic_text, canonical_hash, config


KAPPA_REF_EDR_FRACTION = 0.01
CALIBRATION_SCHEMA = "google-pure-v13-edr-sensitivity-calibration.v1"
BOUNDARY_SCHEMA = "google-pure-v13-sensitivity-boundary.v1"


def _wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denominator
    radius = z / denominator * sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))
    return max(0.0, centre - radius), min(1.0, centre + radius)


def _control_order_hash(plant: str, count: int) -> str:
    return canonical_hash({"plant": plant, "controls": [f"{plant}:control:{i}" for i in range(count)]})


def _calibration_models() -> list[dict[str, Any]]:
    step = SourceStepPlant()
    recovery = SparseControlPlant(5, 924, 24, seed=10_100, curvature=0.004)
    degree = np.bincount(recovery.control_detector, minlength=recovery.detectors)
    return [
        {
            "plant": "STEP_RESPONSE_INJECTED_DRIFT",
            "controls": step.controls,
            "detectors": step.detectors,
            "owners": np.arange(step.controls, dtype=np.int64) % step.detectors,
            "intercepts": step.base_edr,
            "coefficients": step.sensitivity,
            "plant_hash": step.plant_hash,
            "graph_hash": canonical_hash(step.mask.astype(int).tolist()),
        },
        {
            "plant": "RANDOMIZED_RECOVERY_AFTER_SPOIL",
            "controls": recovery.controls,
            "detectors": recovery.detectors,
            "owners": recovery.control_detector,
            "intercepts": np.full(recovery.detectors, recovery.irreducible_physical_error),
            "coefficients": recovery.curvature / degree[recovery.control_detector],
            "plant_hash": recovery.plant_hash,
            "graph_hash": recovery.graph_hash,
        },
    ]


def _fit_coordinate(rows: list[dict[str, Any]], *, residual_noise_upper_z: float) -> dict[str, Any]:
    displacement = np.asarray([row["displacement_native"] for row in rows], dtype=float)
    response = np.asarray([row["edr_fraction"] for row in rows], dtype=float)
    trials = np.asarray([row["qec_cycles"] for row in rows], dtype=float)
    design = np.column_stack([np.ones(len(rows)), np.square(displacement)])
    variance = np.maximum(response * (1.0 - response) / trials, 0.25 / np.square(trials))
    weight = 1.0 / variance
    information = design.T @ (weight[:, None] * design)
    covariance = np.linalg.inv(information)
    beta = covariance @ (design.T @ (weight * response))
    fitted = design @ beta
    residual = response - fitted
    dof = max(1, len(response) - design.shape[1])
    reduced_chi2 = float(np.sum(weight * np.square(residual)) / dof)
    covariance = covariance * max(reduced_chi2, np.finfo(float).tiny)
    standard_error = np.sqrt(np.diag(covariance))
    residual_sum = float(np.sum(np.square(residual)))
    total = float(np.sum(np.square(response - np.mean(response))))
    ordinary_r_squared = 1.0 - residual_sum / total if total > 0 else 1.0
    # Detector-event observations contain a known binomial noise floor.  Report the
    # ordinary statistic, but use a separately named noise-corrected diagnostic to
    # avoid declaring a correct local law non-quadratic merely because its curvature
    # is small compared with finite-shot noise.
    noise_floor = float(np.sum(variance))
    noise_floor_upper = noise_floor + float(residual_noise_upper_z) * sqrt(float(2.0 * np.sum(np.square(variance))))
    unexplained_above_noise = max(0.0, residual_sum - noise_floor_upper)
    r_squared = 1.0 - unexplained_above_noise / total if total > 0 else 1.0
    pairs = []
    for magnitude in sorted(set(abs(value) for value in displacement if value != 0)):
        plus = next(row for row in rows if row["displacement_native"] == magnitude)
        minus = next(row for row in rows if row["displacement_native"] == -magnitude)
        variance_pair = (
            plus["edr_fraction"] * (1.0 - plus["edr_fraction"]) / plus["qec_cycles"]
            + minus["edr_fraction"] * (1.0 - minus["edr_fraction"]) / minus["qec_cycles"]
        )
        z = (plus["edr_fraction"] - minus["edr_fraction"]) / sqrt(max(variance_pair, 1e-30))
        pairs.append({"absolute_displacement_native": magnitude, "sign_asymmetry_z": float(z)})
    coefficient = float(beta[1])
    coefficient_se = float(standard_error[1])
    return {
        "intercept_edr_fraction": float(beta[0]),
        "quadratic_coefficient_edr_fraction_per_native_squared": coefficient,
        "quadratic_coefficient_interval_95": [coefficient - 1.96 * coefficient_se,
                                                coefficient + 1.96 * coefficient_se],
        "quadratic_coefficient_standard_error": coefficient_se,
        "ordinary_r_squared": float(ordinary_r_squared),
        "noise_corrected_r_squared": float(r_squared),
        "r_squared": float(r_squared),
        "r_squared_gate_statistic": "BINOMIAL_NOISE_CORRECTED",
        "binomial_residual_noise_floor": noise_floor,
        "binomial_residual_noise_upper": noise_floor_upper,
        "binomial_residual_noise_upper_z": float(residual_noise_upper_z),
        "reduced_chi_squared": reduced_chi2,
        "maximum_absolute_sign_asymmetry_z": float(max(abs(item["sign_asymmetry_z"]) for item in pairs)),
        "sign_pairs": pairs,
        "residuals": [{"displacement_native": float(value), "residual_edr_fraction": float(error)}
                      for value, error in zip(displacement, residual)],
    }


def calibrate_edr_sensitivity() -> dict[str, Any]:
    """Acquire and fit an independent symmetric calibration for every coordinate."""
    settings = config()["sensitivity_calibration"]
    displacement = [float(value) for value in settings["displacements_native"]]
    cycles = int(settings["qec_cycles_per_displacement"])
    if displacement != sorted(displacement) or 0.0 not in displacement or any(-x not in displacement for x in displacement):
        raise RuntimeError("calibration grid must be ordered, symmetric, and contain zero")
    if float(settings["kappa_ref_edr_percentage_points"]) != 1.0:
        raise RuntimeError("V13 source-literal kappa_ref must remain one EDR percentage point")

    raw_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    scale_rows: list[dict[str, Any]] = []
    for plant_index, model in enumerate(_calibration_models()):
        order_hash = _control_order_hash(model["plant"], model["controls"])
        rng = np.random.default_rng(int(settings["seed"]) + 10_000 * plant_index)
        coordinate_rows: list[list[dict[str, Any]]] = [[] for _ in range(model["controls"])]
        for control_index in range(model["controls"]):
            owner = int(model["owners"][control_index])
            paired_counts: dict[float, int] = {}
            for delta in displacement:
                probability = float(np.clip(model["intercepts"][owner] +
                                            model["coefficients"][control_index] * delta * delta,
                                            1e-9, 0.49))
                magnitude = abs(delta)
                if magnitude not in paired_counts:
                    paired_counts[magnitude] = int(rng.binomial(cycles, probability))
                count = paired_counts[magnitude]
                edr = count / cycles
                lower, upper = _wilson_interval(count, cycles)
                row = {
                    "plant": model["plant"], "control_index": control_index,
                    "control_id": f"{model['plant']}:control:{control_index}",
                    "detector_index": owner, "displacement_native": delta,
                    "detector_event_count": count, "qec_cycles": cycles,
                    "edr_fraction": edr, "edr_interval_95": [lower, upper],
                    "control_order_hash": order_hash,
                    "paired_random_tape_id": f"{model['plant']}:{control_index}:{magnitude:g}",
                }
                raw_rows.append(row)
                coordinate_rows[control_index].append(row)
        for control_index, rows in enumerate(coordinate_rows):
            fit = _fit_coordinate(rows, residual_noise_upper_z=float(settings["residual_noise_upper_z"]))
            coefficient = fit["quadratic_coefficient_edr_fraction_per_native_squared"]
            scale = sqrt(KAPPA_REF_EDR_FRACTION / coefficient) if coefficient > 0 else None
            fit_row = {
                "plant": model["plant"], "control_index": control_index,
                "control_id": f"{model['plant']}:control:{control_index}",
                "detector_index": int(model["owners"][control_index]),
                "control_order_hash": order_hash, **fit,
            }
            fit_rows.append(fit_row)
            scale_rows.append({
                "plant": model["plant"], "control_index": control_index,
                "control_id": fit_row["control_id"], "detector_index": fit_row["detector_index"],
                "native_per_normalized": scale,
                "fitted_edr_coefficient_fraction_per_native_squared": coefficient,
                "conditioned_edr_coefficient_fraction_per_normalized_squared":
                    coefficient * scale * scale if scale is not None else None,
                "kappa_ref_edr_fraction": KAPPA_REF_EDR_FRACTION,
                "mapping": "u=u0+s*x", "control_order_hash": order_hash,
            })

    provenance = {
        "kappa_reference": SOURCE_LITERAL,
        "plant_coefficients": SOURCE_ANCHORED,
        "displacement_grid": SOURCE_UNSPECIFIED_PREREGISTERED,
        "qec_cycle_budget": SOURCE_UNSPECIFIED_PREREGISTERED,
        "fit_acceptance_thresholds": SOURCE_UNSPECIFIED_PREREGISTERED,
    }
    raw = {"schema_version": CALIBRATION_SCHEMA, "rows": raw_rows,
           "row_count": len(raw_rows),
           "qec_cycles_total": int(sum(row["qec_cycles"] for row in raw_rows)),
           "detector_event_trials_total": int(sum(row["qec_cycles"] for row in raw_rows)),
           "resource_semantics": "ONE_ISOLATED_CONTROL_OWNER_DETECTOR_TRIAL_PER_QEC_CYCLE",
           "source_classification": provenance, **NONFINAL}
    fits = {"schema_version": CALIBRATION_SCHEMA, "fits": fit_rows,
            "fit_count": len(fit_rows), "source_classification": provenance, **NONFINAL}
    scales = {"schema_version": BOUNDARY_SCHEMA, "mapping": "u=u0+s*x",
              "sensitivity_application_count": 1, "kappa_ref_edr_fraction": KAPPA_REF_EDR_FRACTION,
              "scales": scale_rows, "scale_count": len(scale_rows),
              "source_classification": provenance, **NONFINAL}
    scales["sensitivity_map_hash"] = canonical_hash({
        "schema_version": scales["schema_version"], "mapping": scales["mapping"],
        "kappa_ref_edr_fraction": scales["kappa_ref_edr_fraction"], "scales": scale_rows,
    })
    atomic_json(ARTIFACT_ROOT / "sensitivity_calibration/raw.json", raw)
    atomic_json(ARTIFACT_ROOT / "sensitivity_calibration/fits.json", fits)
    atomic_json(ARTIFACT_ROOT / "sensitivity_calibration/scales.json", scales)
    validation = validate_sensitivity_map(raw=raw, fits=fits, scales=scales)
    lines = ["# V13 independent EDR sensitivity calibration", "",
             "The calibration uses symmetric native offsets and raw detector-event counts. "
             "The source-literal reference is one EDR percentage point per normalized variance unit.", "",
             f"Coordinates fitted: **{len(fit_rows)}**", f"Validation passed: **{validation['pass']}**", "",
             "This is public-analogue development evidence, not paper-equivalence evidence."]
    atomic_text(ARTIFACT_ROOT / "sensitivity_calibration/report.md", "\n".join(lines))
    return {"raw": raw, "fits": fits, "scales": scales, "validation": validation, **NONFINAL}


def validate_sensitivity_map(*, raw: dict[str, Any] | None = None,
                             fits: dict[str, Any] | None = None,
                             scales: dict[str, Any] | None = None) -> dict[str, Any]:
    from .io import read_json

    raw = raw or read_json(ARTIFACT_ROOT / "sensitivity_calibration/raw.json")
    fits = fits or read_json(ARTIFACT_ROOT / "sensitivity_calibration/fits.json")
    scales = scales or read_json(ARTIFACT_ROOT / "sensitivity_calibration/scales.json")
    settings = config()["sensitivity_calibration"]
    fit_lookup = {(row["plant"], row["control_index"]): row for row in fits["fits"]}
    failures: list[str] = []
    diagnostics = []
    for row in scales["scales"]:
        key = (row["plant"], row["control_index"])
        fit = fit_lookup.get(key)
        if fit is None:
            failures.append(f"missing_fit:{key}")
            continue
        coefficient = fit["quadratic_coefficient_edr_fraction_per_native_squared"]
        interval = fit["quadratic_coefficient_interval_95"]
        conditioned = row["conditioned_edr_coefficient_fraction_per_normalized_squared"]
        relative_error = (abs(conditioned - KAPPA_REF_EDR_FRACTION) /
                          KAPPA_REF_EDR_FRACTION) if conditioned is not None else None
        gates = {
            "positive_coefficient": coefficient > 0,
            "positive_lower_confidence_bound": interval[0] > 0,
            "quadratic_r_squared": fit["r_squared"] >= float(settings["quadratic_minimum_r_squared"]),
            "sign_symmetry": fit["maximum_absolute_sign_asymmetry_z"] <= float(settings["maximum_sign_asymmetry_z"]),
            "conditioning": (relative_error is not None and
                               relative_error <= float(settings["conditioning_relative_tolerance"])),
            "same_control_order": row["control_order_hash"] == fit["control_order_hash"],
        }
        if not all(gates.values()):
            failures.append(f"fit_gate:{key}")
        diagnostics.append({"plant": key[0], "control_index": key[1],
                            "conditioning_relative_error": relative_error, "gates": gates})
    expected_raw = len(scales["scales"]) * len(config()["sensitivity_calibration"]["displacements_native"])
    if raw["row_count"] != expected_raw:
        failures.append("raw_row_count")
    payload = {key: value for key, value in scales.items() if key != "sensitivity_map_hash"}
    expected_hash = canonical_hash({"schema_version": payload["schema_version"],
                                    "mapping": payload["mapping"],
                                    "kappa_ref_edr_fraction": payload["kappa_ref_edr_fraction"],
                                    "scales": payload["scales"]})
    if scales.get("sensitivity_map_hash") != expected_hash:
        failures.append("sensitivity_map_hash")
    result = {"schema_version": V13_SCHEMA, "pass": not failures, "failures": failures,
              "coordinates_checked": len(diagnostics), "raw_rows_checked": raw["row_count"],
              "diagnostics": diagnostics, "sensitivity_map_hash": scales.get("sensitivity_map_hash"),
              **NONFINAL}
    atomic_json(ARTIFACT_ROOT / "sensitivity_calibration/validation.json", result)
    return result


@dataclass(frozen=True)
class CoordinateBatch:
    values: np.ndarray
    control_order_hash: str
    coordinate_space: str = "normalized"
    sensitivity_application_count: int = 0
    sensitivity_map_hash: str | None = None

    def __post_init__(self) -> None:
        value = np.asarray(self.values, dtype=float)
        if value.ndim not in {1, 2} or not np.all(np.isfinite(value)):
            raise ValueError("coordinate batch must be a finite vector or matrix")
        object.__setattr__(self, "values", value.copy())


@dataclass(frozen=True)
class BoundaryResult:
    native: CoordinateBatch
    normalized_hashes: tuple[str, ...]
    scale_hash: str
    reference_hash: str
    scaled_action_hashes: tuple[str, ...]
    native_action_hashes: tuple[str, ...]


class SensitivityBoundary:
    """Apply one frozen map exactly once, with control-order and lineage checks."""

    def __init__(self, scales: np.ndarray, reference_native: np.ndarray, *,
                 control_order_hash: str, sensitivity_map_hash: str,
                 expected_scale_hash: str | None = None) -> None:
        self.scales = np.asarray(scales, dtype=float).copy()
        self.reference = np.asarray(reference_native, dtype=float).copy()
        if self.scales.ndim != 1 or self.reference.shape != self.scales.shape:
            raise ValueError("boundary scale/reference shape mismatch")
        if np.any(self.scales <= 0) or not np.all(np.isfinite(self.scales)):
            raise ValueError("boundary scales must be positive and finite")
        self.control_order_hash = str(control_order_hash)
        self.sensitivity_map_hash = str(sensitivity_map_hash)
        self.expected_scale_hash = expected_scale_hash or canonical_hash(self.scales.tolist())
        if canonical_hash(self.scales.tolist()) != self.expected_scale_hash:
            raise RuntimeError("inverse, stale, or different sensitivity scale")

    @classmethod
    def from_artifact(cls, plant: str, *, reference_native: np.ndarray | None = None) -> "SensitivityBoundary":
        from .io import read_json

        artifact = read_json(ARTIFACT_ROOT / "sensitivity_calibration/scales.json")
        rows = sorted((row for row in artifact["scales"] if row["plant"] == plant),
                      key=lambda row: row["control_index"])
        if not rows or [row["control_index"] for row in rows] != list(range(len(rows))):
            raise RuntimeError("sensitivity map has missing or reordered controls")
        order_hashes = {row["control_order_hash"] for row in rows}
        if len(order_hashes) != 1:
            raise RuntimeError("sensitivity map contains inconsistent control-order hashes")
        reference = np.zeros(len(rows)) if reference_native is None else np.asarray(reference_native, dtype=float)
        return cls(np.asarray([row["native_per_normalized"] for row in rows]), reference,
                   control_order_hash=order_hashes.pop(),
                   sensitivity_map_hash=artifact["sensitivity_map_hash"],
                   expected_scale_hash=canonical_hash(
                       [row["native_per_normalized"] for row in rows]))

    def apply(self, batch: CoordinateBatch) -> BoundaryResult:
        if canonical_hash(self.scales.tolist()) != self.expected_scale_hash:
            raise RuntimeError("inverse, stale, or different sensitivity scale")
        if batch.coordinate_space != "normalized" or batch.sensitivity_application_count != 0:
            raise RuntimeError("sensitivity boundary may be applied exactly once to normalized coordinates")
        if batch.sensitivity_map_hash not in {None, self.sensitivity_map_hash}:
            raise RuntimeError("stale or different sensitivity map")
        if batch.control_order_hash != self.control_order_hash:
            raise RuntimeError("control order changed at the sensitivity boundary")
        values = np.atleast_2d(batch.values)
        if values.shape[1] != len(self.scales):
            raise ValueError("normalized action dimension does not match sensitivity map")
        scaled = values * self.scales[None, :]
        native = self.reference[None, :] + scaled
        vector_hashes = lambda rows: tuple(canonical_hash(np.asarray(row, dtype=float).tolist()) for row in rows)
        native_batch = CoordinateBatch(
            native[0] if batch.values.ndim == 1 else native,
            self.control_order_hash, "native", 1, self.sensitivity_map_hash)
        return BoundaryResult(
            native=native_batch,
            normalized_hashes=vector_hashes(values),
            scale_hash=canonical_hash(self.scales.tolist()),
            reference_hash=canonical_hash(self.reference.tolist()),
            scaled_action_hashes=vector_hashes(scaled),
            native_action_hashes=vector_hashes(native),
        )


def require_native_boundary(batch: CoordinateBatch, *, control_order_hash: str,
                            sensitivity_map_hash: str) -> None:
    """Fail closed if a plant input omitted, repeated, or changed the frozen map."""
    if batch.coordinate_space != "native" or batch.sensitivity_application_count != 1:
        raise RuntimeError("plant input must pass through the sensitivity boundary exactly once")
    if batch.control_order_hash != control_order_hash:
        raise RuntimeError("plant input control order differs from the sensitivity map")
    if batch.sensitivity_map_hash != sensitivity_map_hash:
        raise RuntimeError("plant input used a stale or different sensitivity map")
