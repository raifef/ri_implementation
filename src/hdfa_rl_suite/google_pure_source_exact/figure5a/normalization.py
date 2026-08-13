"""Noncanonical Figure-S3-style conditioning ablation for the Figure 5a plant.

The paper defines EDR as a probability and writes

    EDR = EDR0 + (sigma / sigma0)**2.

Taken literally, ``sigma0`` corresponds to a unit increase in fractional
EDR, not to one percentage point.  On the public Figure-5a model that scale
is outside the physically valid domain: it cannot coexist with the published
unit-amplitude sinusoidal optimum.  We therefore retain the literal sigma0 as
a measured diagnostic but apply only its *relative* anisotropy correction,
fixing the otherwise unidentified common gauge by preserving the weighted
geometric-mean native scale.  This is a clean-room convention, not a claim
about Google's unpublished absolute coordinate scale.  This module is never
loaded by the canonical Figure 5a acquisition; it exists only to quantify the
effect of an empirical relative-conditioning ablation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from hdfa_rl_suite.google_pure_source_exact.source_normalization import (
    BOUNDARY_TRANSFORM_NAME,
    BoundaryApplication,
    boundary_transform_hash,
    canonical_hash,
)

from .bounded_action_ablation import Figure5aBoundedActionAblation
from .plant import Figure5aStimPlant


SCHEMA_VERSION = "figure5a-empirical-normalization.v1"
IMPLEMENTATION_VERSION = "google_pure_source_exact_figure5a_v2"
NORMALIZATION_METHOD = "FIG_S3_STIM_EDR_RELATIVE_EQUALIZATION"
SOURCE_LITERAL_TARGET_EDR_INCREASE_FRACTION = 1.0
APPLIED_COMMON_SCALE_GAUGE = "WEIGHTED_GEOMETRIC_MEAN_NATIVE_SCALE_EQUALS_ONE"
CONTROL_GROUPING_IDENTIFIABILITY = "SOURCE_UNSPECIFIED_PREREGISTERED"
SCIENTIFIC_STATUS = "EMPIRICAL_RELATIVE_NORMALIZATION_ABLATION"


def reward_representation_hash(plant: Figure5aStimPlant) -> str:
    return canonical_hash({
        "representation": "time_translation_equivalence_class_mean_edr",
        "raw_detector_count": plant.raw_detector_count,
        "groups": [list(group) for group in plant.reward_component_raw_detectors],
        "mask": plant.mask.astype(int).tolist(),
    })


def _control_groups(plant: Figure5aStimPlant) -> tuple[tuple[str, np.ndarray], ...]:
    single = np.asarray([
        index for index, item in enumerate(plant.inventory) if item.gate_type == "single_qubit"
    ], dtype=int)
    two = np.asarray([
        index for index, item in enumerate(plant.inventory) if item.gate_type == "two_qubit"
    ], dtype=int)
    if len(single) != 17 or len(two) != 24:
        raise RuntimeError("Figure 5a empirical control groups no longer cover 17 + 24 controls")
    return (("single_qubit_gate_miscalibration", single),
            ("two_qubit_gate_miscalibration", two))


def _quadratic_fit(sigmas: np.ndarray, edr: np.ndarray) -> dict[str, float]:
    x = np.square(np.asarray(sigmas, dtype=float))
    y = np.asarray(edr, dtype=float)
    design = np.column_stack([np.ones_like(x), x])
    intercept, coefficient = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = intercept + coefficient * x
    residual = float(np.sum(np.square(y - fitted)))
    total = float(np.sum(np.square(y - np.mean(y))))
    r_squared = 1.0 if total == 0.0 and residual == 0.0 else 1.0 - residual / total
    if coefficient <= 0 or not np.isfinite(coefficient):
        raise RuntimeError("Figure 5a empirical detector-sensitivity curvature is not positive")
    return {
        "edr0_fraction": float(intercept),
        "quadratic_coefficient_per_native_squared": float(coefficient),
        "source_literal_sigma0_native": float(np.sqrt(
            SOURCE_LITERAL_TARGET_EDR_INCREASE_FRACTION / coefficient)),
        "r_squared": float(r_squared),
        "maximum_absolute_fit_residual_fraction": float(np.max(np.abs(y - fitted))),
    }


def build_empirical_normalization(
    plant: Figure5aStimPlant,
    *,
    sweep_sigmas_native: Sequence[float] = (0.0, 0.1, 0.2, 0.3),
    candidates_per_sigma: int = 24,
    seed: int = 53401,
    minimum_r_squared: float = 0.98,
) -> dict[str, Any]:
    """Measure Gaussian perturbation sweeps through the actual Stim DEM.

    Detector marginals are evaluated exactly from the Stim detector error
    model.  The Gaussian expectation is still Monte Carlo over a frozen set of
    candidate perturbations, matching the source perturbation distribution
    without adding finite-shot fit noise.  Independent finite-shot checks live
    in the physical preflight.
    """
    sigmas = np.asarray(tuple(float(value) for value in sweep_sigmas_native), dtype=float)
    if (len(sigmas) < 4 or sigmas[0] != 0.0 or np.any(np.diff(sigmas) <= 0)
            or candidates_per_sigma < 8):
        raise ValueError("normalization sweep requires zero plus three sigmas and at least eight candidates")
    rows: list[dict[str, Any]] = []
    fits_by_group: list[tuple[np.ndarray, dict[str, float]]] = []
    for group_index, (name, indices) in enumerate(_control_groups(plant)):
        rng = np.random.default_rng(int(seed) + 1009 * group_index)
        standardized = rng.normal(size=(int(candidates_per_sigma), len(indices)))
        # Antithetic completion makes the finite candidate expectation exactly
        # centred and improves the quadratic fit without changing N(0, sigma).
        standardized = np.concatenate([standardized, -standardized], axis=0)
        edr_values = []
        for sigma in sigmas:
            candidate_edr = []
            for noise in standardized:
                controls = np.zeros(plant.control_count, dtype=float)
                controls[indices] = sigma * noise
                candidate_edr.append(plant.expected_global_edr(
                    controls, epoch=0, frequency=1 / 1000,
                    target_controls=np.zeros(plant.control_count)))
            edr_values.append(float(np.mean(candidate_edr)))
        fit = _quadratic_fit(sigmas, np.asarray(edr_values))
        if fit["r_squared"] < float(minimum_r_squared):
            raise RuntimeError(f"Figure 5a empirical normalization fit failed for {name}")
        fits_by_group.append((indices, fit))
        rows.append({
            "control_type": name,
            "control_indices": indices.astype(int).tolist(),
            "control_ids": [plant.parameter_ids[index] for index in indices],
            "sweep_sigmas_native": sigmas.tolist(),
            "measured_global_edr_fraction": edr_values,
            "gaussian_candidates_per_sigma": int(2 * candidates_per_sigma),
            "gaussian_candidate_seed": int(seed) + 1009 * group_index,
            **fit,
        })
    # The paper does not identify whether the EDR in its fit is a fraction or
    # a percentage-valued observable for the purpose of the coefficient, and
    # the literal fractional scale is incompatible with the published
    # unit-amplitude Figure-5a drift on this Stim plant.  A common multiplier
    # is therefore not identifiable.  Relative equalization is still
    # identifiable from the measured curvatures: choose a coordinate-volume
    # preserving gauge (weighted geometric mean scale = 1).
    log_reference_curvature = sum(
        len(indices) * np.log(fit["quadratic_coefficient_per_native_squared"])
        for indices, fit in fits_by_group
    ) / plant.control_count
    reference_curvature = float(np.exp(log_reference_curvature))
    scales = np.zeros(plant.control_count, dtype=float)
    for indices, fit in fits_by_group:
        scales[indices] = np.sqrt(
            reference_curvature / fit["quadratic_coefficient_per_native_squared"])
    literal_scales = np.zeros(plant.control_count, dtype=float)
    for indices, fit in fits_by_group:
        literal_scales[indices] = fit["source_literal_sigma0_native"]
    literal_scale_safe = True
    literal_scale_failure = None
    try:
        Figure5aBoundedActionAblation(plant).normalized_control_limits(literal_scales)
    except ValueError as exc:
        literal_scale_safe = False
        literal_scale_failure = str(exc)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "normalization_method": NORMALIZATION_METHOD,
        "scientific_status": SCIENTIFIC_STATUS,
        "canonical_figure5a_execution": False,
        "plant_hash": plant.plant_hash,
        "reward_representation_hash": reward_representation_hash(plant),
        "raw_detector_count": plant.raw_detector_count,
        "reward_component_count": plant.detector_count,
        "edr_unit": "fraction",
        "fit_equation": "EDR_fraction = EDR0_fraction + (sigma_native/sigma0_native)^2",
        "source_literal_target_edr_increase_fraction":
            SOURCE_LITERAL_TARGET_EDR_INCREASE_FRACTION,
        "percentage_point_conversion_applied": False,
        "analytic_omega_times_degree_shortcut_used": False,
        "absolute_source_scale_identifiable": False,
        "applied_scale_kind": "relative_empirical_curvature_equalization",
        "applied_common_scale_gauge": APPLIED_COMMON_SCALE_GAUGE,
        "applied_reference_curvature_fraction_per_native_squared": reference_curvature,
        "applied_weighted_geometric_mean_native_scale": float(np.exp(np.mean(np.log(scales)))),
        "source_literal_native_scale": literal_scales.tolist(),
        "source_literal_scale_safe_for_published_unit_amplitude_drift": literal_scale_safe,
        "source_literal_scale_incompatibility": literal_scale_failure,
        "measurement": "exact Stim DEM detector marginals averaged over frozen Gaussian candidates",
        "control_grouping_source_identifiability": CONTROL_GROUPING_IDENTIFIABILITY,
        "control_groups": rows,
        "native_scale": scales.tolist(),
        "minimum_r_squared": float(minimum_r_squared),
        "artifact_complete": True,
        "mathematical_contract_pass": True,
        "source_structure_match": True,
        "paper_comparable": False,
        "blocking_reasons": [
            "the literal fractional-EDR sigma0 is physically incompatible with the published unit-amplitude drift on this reconstructed plant",
            "the common absolute normalization gauge is not publicly identifiable",
            "Figure 5a synthetic control-type grouping is not published",
            "normalization sweep grid and candidate count are preregistered clean-room choices",
            "proprietary optimizer hyperparameters remain unavailable",
        ],
    }
    payload["calibration_hash"] = canonical_hash(payload)
    return payload


@dataclass(frozen=True)
class Figure5aEmpiricalBoundary:
    control_ids: tuple[str, ...]
    native_scale: np.ndarray
    plant_hash: str
    calibration_hash: str
    reward_representation_hash: str

    def __post_init__(self) -> None:
        scale = np.asarray(self.native_scale, dtype=float)
        if (scale.shape != (len(self.control_ids),) or np.any(scale <= 0)
                or not np.all(np.isfinite(scale))):
            raise ValueError("invalid Figure 5a empirical sensitivity scale")
        scale = scale.copy(); scale.setflags(write=False)
        object.__setattr__(self, "native_scale", scale)

    @classmethod
    def from_artifact(cls, plant: Figure5aStimPlant,
                      artifact: Mapping[str, Any]) -> "Figure5aEmpiricalBoundary":
        value = dict(artifact)
        failures = []
        if value.get("schema_version") != SCHEMA_VERSION:
            failures.append("schema changed")
        if value.get("normalization_method") != NORMALIZATION_METHOD:
            failures.append("normalization method changed")
        if (value.get("scientific_status") != SCIENTIFIC_STATUS or
                value.get("canonical_figure5a_execution") is not False):
            failures.append("normalization ablation was promoted to canonical execution")
        if value.get("plant_hash") != plant.plant_hash:
            failures.append("plant hash changed")
        if value.get("reward_representation_hash") != reward_representation_hash(plant):
            failures.append("reward representation changed")
        if value.get("edr_unit") != "fraction" or value.get("percentage_point_conversion_applied"):
            failures.append("EDR unit is not the literal fractional source convention")
        if (float(value.get("source_literal_target_edr_increase_fraction", -1)) !=
                SOURCE_LITERAL_TARGET_EDR_INCREASE_FRACTION):
            failures.append("source-literal Figure-S3 coefficient changed")
        if value.get("applied_common_scale_gauge") != APPLIED_COMMON_SCALE_GAUGE:
            failures.append("unregistered absolute normalization gauge")
        if value.get("absolute_source_scale_identifiable") is not False:
            failures.append("absolute source scale was promoted without evidence")
        claimed_hash = value.get("calibration_hash")
        unhashed = {key: item for key, item in value.items() if key != "calibration_hash"}
        if claimed_hash != canonical_hash(unhashed):
            failures.append("calibration hash mismatch")
        if failures:
            raise RuntimeError("stale Figure 5a empirical normalization: " + "; ".join(failures))
        return cls(tuple(plant.parameter_ids), np.asarray(value["native_scale"], dtype=float),
                   plant.plant_hash, str(claimed_hash), str(value["reward_representation_hash"]))

    @property
    def control_order_hash(self) -> str:
        return canonical_hash(list(self.control_ids))

    @property
    def expanded_scale_hash(self) -> str:
        return canonical_hash(self.native_scale.tolist())

    @property
    def sensitivity_map_hash(self) -> str:
        return self.calibration_hash

    @property
    def boundary_transform_hash(self) -> str:
        return boundary_transform_hash()

    def apply(self, normalized: np.ndarray, *, application_count: int = 0,
              control_order_hash: str | None = None,
              sensitivity_map_hash: str | None = None) -> BoundaryApplication:
        if application_count != 0:
            raise RuntimeError("normalization boundary may be applied exactly once")
        if control_order_hash not in {None, self.control_order_hash}:
            raise RuntimeError("control order changed at the Figure 5a plant boundary")
        if sensitivity_map_hash not in {None, self.sensitivity_map_hash}:
            raise RuntimeError("stale Figure 5a empirical sensitivity map")
        value = np.asarray(normalized, dtype=float)
        if value.ndim not in {1, 2} or value.shape[-1] != len(self.control_ids):
            raise ValueError("normalized controls do not match the Figure 5a registry")
        native = self.native_scale * value
        return BoundaryApplication(native, {
            **self.provenance_fields(),
            "input_space": "SOURCE_NORMALIZED",
            "output_space": "NATIVE_CONTROL",
            "normalized_action_hash": canonical_hash(value.tolist()),
            "native_action_hash": canonical_hash(native.tolist()),
        })

    def target_to_native(self, normalized_target: np.ndarray) -> np.ndarray:
        value = np.asarray(normalized_target, dtype=float)
        if value.shape != (len(self.control_ids),) or not np.all(np.isfinite(value)):
            raise ValueError("target does not match the Figure 5a registry")
        return self.native_scale * value

    def trace(self, normalized: np.ndarray, *, indices: Sequence[int] | None = None) -> dict[str, Any]:
        value = np.asarray(normalized, dtype=float)
        selected = tuple(indices or np.flatnonzero(np.abs(value) > 0)[:16].tolist() or (0,))
        rows = [{
            "control_index": int(index),
            "control_id": self.control_ids[index],
            "x_i": float(value[index]),
            "s_i": float(self.native_scale[index]),
            "u_i": float(self.native_scale[index] * value[index]),
        } for index in selected]
        return {"schema_version": "figure5a-boundary-trace.v2", "rows": rows,
                "same_scale_for_mean_candidates_and_evaluation": True,
                "second_scaling_inside_plant": False, **self.provenance_fields()}

    def provenance_fields(self) -> dict[str, Any]:
        return {
            "implementation_version": IMPLEMENTATION_VERSION,
            "normalization_method": NORMALIZATION_METHOD,
            "scientific_status": SCIENTIFIC_STATUS,
            "canonical_figure5a_execution": False,
            "source_literal_normalization_target_edr_fraction":
                SOURCE_LITERAL_TARGET_EDR_INCREASE_FRACTION,
            "normalization_edr_unit": "fraction",
            "percentage_point_conversion_applied": False,
            "analytic_omega_times_degree_shortcut_used": False,
            "absolute_source_scale_identifiable": False,
            "applied_common_scale_gauge": APPLIED_COMMON_SCALE_GAUGE,
            "calibration_hash": self.calibration_hash,
            "sensitivity_map_hash": self.sensitivity_map_hash,
            "reward_representation_hash": self.reward_representation_hash,
            "boundary_transform_hash": self.boundary_transform_hash,
            "boundary_transform_name": BOUNDARY_TRANSFORM_NAME,
            "boundary_apply_count": 1,
            "control_order_hash": self.control_order_hash,
            "expanded_scale_hash": self.expanded_scale_hash,
        }


def empirical_boundary_for_plant(
    plant: Figure5aStimPlant,
    artifact: Mapping[str, Any] | None = None,
) -> Figure5aEmpiricalBoundary:
    """Return a validated boundary, caching only a deterministic local build."""
    if artifact is not None:
        return Figure5aEmpiricalBoundary.from_artifact(plant, artifact)
    cached = getattr(plant, "_empirical_normalization_artifact", None)
    if cached is None:
        cached = build_empirical_normalization(plant)
        setattr(plant, "_empirical_normalization_artifact", cached)
    return Figure5aEmpiricalBoundary.from_artifact(plant, cached)


def require_figure5a_boundary_provenance(value: Mapping[str, Any]) -> None:
    if value.get("implementation_version") != IMPLEMENTATION_VERSION:
        raise RuntimeError("Figure 5a result did not use the empirical source boundary")
    if value.get("normalization_method") != NORMALIZATION_METHOD:
        raise RuntimeError("Figure 5a normalization method changed")
    if (value.get("scientific_status") != SCIENTIFIC_STATUS or
            value.get("canonical_figure5a_execution") is not False):
        raise RuntimeError("Figure 5a normalization is an ablation, not canonical execution")
    if value.get("normalization_edr_unit") != "fraction":
        raise RuntimeError("Figure 5a normalization silently changed EDR units")
    if value.get("percentage_point_conversion_applied") is not False:
        raise RuntimeError("unsupported one-percentage-point normalization returned")
    if value.get("analytic_omega_times_degree_shortcut_used") is not False:
        raise RuntimeError("analytic Omega-times-degree normalization returned")
    if (float(value.get("source_literal_normalization_target_edr_fraction", -1)) !=
            SOURCE_LITERAL_TARGET_EDR_INCREASE_FRACTION):
        raise RuntimeError("Figure-S3 coefficient convention changed")
    if value.get("absolute_source_scale_identifiable") is not False:
        raise RuntimeError("unidentified Figure 5a absolute scale was promoted")
    if value.get("applied_common_scale_gauge") != APPLIED_COMMON_SCALE_GAUGE:
        raise RuntimeError("Figure 5a normalization gauge changed")
    for name in ("calibration_hash", "reward_representation_hash", "sensitivity_map_hash",
                 "boundary_transform_hash", "control_order_hash", "expanded_scale_hash"):
        if not value.get(name):
            raise RuntimeError(f"missing Figure 5a normalization provenance: {name}")
