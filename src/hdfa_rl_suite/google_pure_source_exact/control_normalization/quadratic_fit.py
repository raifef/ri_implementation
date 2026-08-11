"""Frozen weighted quadratic fit for ``EDR0 + (sigma/sigma0)^2``."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np

from .contracts import FitRules, SensitivityFit, SweepResult


@dataclass(frozen=True)
class FitDiagnostics:
    reasons: tuple[str, ...]
    values: dict[str, Any]


class FitRejected(RuntimeError):
    def __init__(self, diagnostics: FitDiagnostics):
        self.diagnostics = diagnostics
        super().__init__("; ".join(diagnostics.reasons))


def _point_arrays(sweep: SweepResult) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    lower, upper = sweep.fit_interval_native
    points = [point for point in sweep.points if lower <= point.sigma_native <= upper]
    points.sort(key=lambda point: point.sigma_native)
    if len(points) < 4:
        raise FitRejected(FitDiagnostics(("fewer than four points in frozen fit interval",), {}))
    x = np.asarray([point.sigma_native ** 2 for point in points], dtype=float)
    y = np.asarray([point.edr_percentage_points for point in points], dtype=float)
    # Jeffreys smoothing prevents zero estimated variance at finite shot count.
    probability = np.asarray([
        (point.detector_events + 0.5) / (point.detector_opportunities + 1.0)
        for point in points
    ])
    binomial_variance = 10000.0 * probability * (1.0 - probability) / np.asarray(
        [point.detector_opportunities for point in points], dtype=float)
    cluster_variance = []
    for point, fallback in zip(points, binomial_variance):
        if len(point.candidate_detector_events) >= 2:
            candidate_rates = 100.0 * np.asarray(
                point.candidate_detector_events, dtype=float) / point.candidate_detector_opportunities
            cluster_variance.append(float(np.var(candidate_rates, ddof=1) / len(candidate_rates)))
        else:
            cluster_variance.append(float(fallback))
    # The larger variance protects both finite detector sampling and the
    # candidate-level Gaussian perturbation mixture.
    variance = np.maximum(binomial_variance, np.asarray(cluster_variance))
    variance = np.maximum(variance, np.finfo(float).tiny)
    return x, y, variance, points


def _weighted_fit(design: np.ndarray, y: np.ndarray,
                  variance: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    weights = 1.0 / variance
    information = design.T @ (weights[:, None] * design)
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError as error:
        raise FitRejected(FitDiagnostics(("singular weighted fit",), {})) from error
    beta = covariance @ (design.T @ (weights * y))
    residual = y - design @ beta
    chi_squared = float(np.sum(np.square(residual) / variance))
    return beta, covariance, residual, chi_squared


def fit_detector_sensitivity(sweep: SweepResult, rules: FitRules) -> SensitivityFit:
    x, y, variance, points = _point_arrays(sweep)
    design = np.column_stack((np.ones_like(x), x))
    beta, covariance, residual, chi_squared = _weighted_fit(design, y, variance)
    intercept, coefficient = map(float, beta)
    coefficient_se = math.sqrt(max(0.0, float(covariance[1, 1])))
    coefficient_z = coefficient / max(coefficient_se, np.finfo(float).tiny)
    degrees_of_freedom = len(x) - 2
    reduced_chi_squared = chi_squared / degrees_of_freedom
    total = float(np.sum(np.square(y - np.mean(y))))
    r_squared = 1.0 - float(np.sum(np.square(residual))) / max(total, np.finfo(float).tiny)

    monotonicity = []
    for left, right, left_var, right_var in zip(y[:-1], y[1:], variance[:-1], variance[1:]):
        monotonicity.append(float((left - right) / math.sqrt(left_var + right_var)))
    monotonicity_max_z = max([0.0, *monotonicity])

    scaled_x = x / max(float(np.max(x)), np.finfo(float).tiny)
    quartic_design = np.column_stack((np.ones_like(x), scaled_x, np.square(scaled_x)))
    quartic_beta, quartic_covariance, _, _ = _weighted_fit(quartic_design, y, variance)
    quartic_se = math.sqrt(max(0.0, float(quartic_covariance[2, 2])))
    quartic_z = float(quartic_beta[2] / max(quartic_se, np.finfo(float).tiny))

    reasons: list[str] = []
    if not math.isfinite(coefficient) or coefficient <= 0:
        reasons.append("quadratic coefficient is not positive")
    if coefficient_z < rules.minimum_positive_coefficient_z:
        reasons.append("quadratic coefficient is not resolved from zero")
    if r_squared < rules.minimum_r_squared:
        reasons.append("quadratic fit has insufficient R-squared")
    if reduced_chi_squared > rules.maximum_reduced_chi_squared:
        reasons.append("quadratic fit has excessive reduced chi-squared")
    if monotonicity_max_z > rules.maximum_monotonicity_z:
        reasons.append("sweep is significantly non-monotonic")
    if abs(quartic_z) > rules.maximum_quartic_z:
        reasons.append("a quartic-in-sigma contribution is statistically resolved")
    coefficient_low = coefficient - rules.confidence_z * coefficient_se
    coefficient_high = coefficient + rules.confidence_z * coefficient_se
    if coefficient_low <= 0:
        reasons.append("coefficient confidence interval includes zero")
    diagnostics = {
        "edr0_percentage_points": intercept,
        "coefficient": coefficient,
        "coefficient_standard_error": coefficient_se,
        "coefficient_z": coefficient_z,
        "coefficient_ci": [coefficient_low, coefficient_high],
        "r_squared": r_squared,
        "reduced_chi_squared": reduced_chi_squared,
        "quartic_z_score": quartic_z,
        "monotonicity_max_z": monotonicity_max_z,
        "fit_sigmas_native": [point.sigma_native for point in points],
        "fit_edr_percentage_points": y.tolist(),
    }
    if reasons:
        raise FitRejected(FitDiagnostics(tuple(reasons), diagnostics))

    sigma0 = 1.0 / math.sqrt(coefficient)
    sigma0_ci = (1.0 / math.sqrt(coefficient_high), 1.0 / math.sqrt(coefficient_low))
    return SensitivityFit(
        control_type=sweep.control_type,
        native_unit=sweep.native_unit,
        edr0_percentage_points=intercept,
        quadratic_coefficient_per_native_squared=coefficient,
        sigma0_native=sigma0,
        sigma0_confidence_interval_95=sigma0_ci,
        coefficient_confidence_interval_95=(coefficient_low, coefficient_high),
        parameter_covariance=(
            (float(covariance[0, 0]), float(covariance[0, 1])),
            (float(covariance[1, 0]), float(covariance[1, 1])),
        ),
        fit_interval_native=sweep.fit_interval_native,
        fit_point_count=len(points),
        r_squared=r_squared,
        reduced_chi_squared=reduced_chi_squared,
        quartic_z_score=quartic_z,
        monotonicity_max_z=monotonicity_max_z,
        fit_rules_hash=rules.rules_hash,
        detector_set_hash=sweep.reference.detector_set_hash,
        circuit_hash=sweep.reference.circuit_hash,
        reference_policy_hash=sweep.reference.reference_policy_hash,
        shot_budget=sum(point.candidates * point.shots_per_candidate for point in points),
        qec_cycle_budget=sum(point.qec_cycles for point in points),
        uncertainty_method=(
            "candidate-cluster-aware WLS covariance (bounded below by binomial variance) "
            "with delta-transformed 95% interval"),
        passed=True,
        blocking_reasons=(),
    )


def coefficient_stability(first: SensitivityFit, second: SensitivityFit,
                          rules: FitRules) -> dict[str, float | bool]:
    if first.control_type != second.control_type:
        raise ValueError("stability comparison requires the same control type")
    a = first.quadratic_coefficient_per_native_squared
    b = second.quadratic_coefficient_per_native_squared
    relative_difference = abs(a - b) / max(abs(a), abs(b), np.finfo(float).tiny)
    variance = first.parameter_covariance[1][1] + second.parameter_covariance[1][1]
    difference_z = abs(a - b) / max(math.sqrt(variance), np.finfo(float).tiny)
    passed = relative_difference <= rules.stability_relative_tolerance or difference_z <= rules.confidence_z
    return {
        "relative_difference": float(relative_difference),
        "difference_z": float(difference_z),
        "relative_tolerance": rules.stability_relative_tolerance,
        "passed": bool(passed),
    }


def fit_all_sweeps(sweeps: Sequence[SweepResult], rules: FitRules) -> tuple[SensitivityFit, ...]:
    names = [item.control_type for item in sweeps]
    if len(names) != len(set(names)):
        raise ValueError("duplicate control-type sweep")
    return tuple(fit_detector_sensitivity(item, rules) for item in sweeps)
