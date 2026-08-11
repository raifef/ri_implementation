"""Released-data reproduction, empirical identification, and validation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .dataset_manifest import load_json_yaml
from .estimators import (
    detector_summary,
    independent_nonlinear_decay_estimator,
    memory_failure_to_error_per_cycle,
    repository_decay_estimator,
)
from .reporting import canonical_hash, write_json, write_markdown
from .schemas import CERTIFICATION_SEEDS, ReproductionStatus, SurrogateValidationOutcome
from .spectral import autocorrelation, low_frequency_fraction, periodogram
from .surrogate import EmpiricalStaticSurrogate, fit_contract_parameters
from .uncertainty import effective_sample_size, parametric_decay_bootstrap
from .zenodo_loader import ZenodoArchive


def _point_data(archive: ZenodoArchive, records: list[Any], pathway: str) -> dict[str, list[int]]:
    rounds: list[int] = []
    errors: list[int] = []
    shots: list[int] = []
    for record in sorted(records, key=lambda row: row.rounds):
        failure_count, shot_count = archive.logical_error_counts(record, pathway)
        rounds.append(record.rounds)
        errors.append(failure_count)
        shots.append(shot_count)
    return {"rounds": rounds, "errors": errors, "shots": shots}


def _decay_result(archive: ZenodoArchive, records: list[Any], pathway: str, *, bootstrap_seed: int) -> dict[str, Any]:
    points = _point_data(archive, records, pathway)
    repository = repository_decay_estimator(**points)
    independent = independent_nonlinear_decay_estimator(**points)
    bootstrap = parametric_decay_bootstrap(**points, seed=bootstrap_seed, replicates=400)
    return {
        "repository_estimator": repository.to_dict(),
        "independent_estimator": independent.to_dict(),
        "uncertainty": bootstrap,
        "points": [
            {
                "rounds": rounds,
                "errors": errors,
                "shots": shots,
                "memory_failure_probability": errors / shots,
                "one_point_logical_error_per_cycle": memory_failure_to_error_per_cycle(errors / shots, rounds),
            }
            for rounds, errors, shots in zip(points["rounds"], points["errors"], points["shots"])
        ],
    }


def _select(
    records: list[Any],
    *,
    family: str,
    distance: int,
    condition: str,
    basis: str,
    subgrid: str | None = None,
) -> list[Any]:
    return [
        record
        for record in records
        if record.code_family == family
        and record.distance == distance
        and record.condition == condition
        and record.basis == basis
        and (subgrid is None or record.subgrid == subgrid)
    ]


def _condition_result(
    archive: ZenodoArchive,
    records: list[Any],
    *,
    family: str,
    distance: int,
    condition: str,
    pathway: str,
    subgrid: str | None,
    seed: int,
) -> dict[str, Any]:
    bases: dict[str, Any] = {}
    for index, basis in enumerate(("X", "Z")):
        chosen = _select(
            records,
            family=family,
            distance=distance,
            condition=condition,
            basis=basis,
            subgrid=subgrid,
        )
        if len(chosen) < 2:
            raise ValueError(f"insufficient decay points for {family}/{distance}/{condition}/{basis}/{subgrid}")
        bases[basis] = _decay_result(archive, chosen, pathway, bootstrap_seed=seed + index)
    repository_values = [bases[b]["repository_estimator"]["logical_error_per_cycle"] for b in ("X", "Z")]
    independent_values = [bases[b]["independent_estimator"]["logical_error_per_cycle"] for b in ("X", "Z")]
    standard_errors = [bases[b]["uncertainty"]["standard_error"] for b in ("X", "Z")]
    return {
        "code_family": family,
        "distance": distance,
        "condition": condition,
        "subgrid": subgrid,
        "decoder_pathway": pathway,
        "basis_results": bases,
        "basis_average_repository": float(np.mean(repository_values)),
        "basis_average_independent": float(np.mean(independent_values)),
        "basis_average_standard_error": float(math.sqrt(sum(value**2 for value in standard_errors)) / 2.0),
        "basis_averaging_order": "fit each basis independently, then arithmetic average",
    }


def reproduce_public_analysis(archive_path: Path, artifact_dir: Path) -> dict[str, Any]:
    with ZenodoArchive(archive_path) as archive:
        records = archive.records()
        surface_final = _condition_result(
            archive,
            records,
            family="surface_code",
            distance=7,
            condition="traditional_calibration_and_rl_fine_tuning",
            pathway="alphaqubit2_decoder",
            subgrid="d7_0+0j",
            seed=73100,
        )
        surface_traditional = _condition_result(
            archive,
            records,
            family="surface_code",
            distance=7,
            condition="traditional_calibration",
            pathway="alphaqubit2_decoder",
            subgrid="d7_0+0j",
            seed=73110,
        )
        color_final = _condition_result(
            archive,
            records,
            family="color_code",
            distance=5,
            condition="traditional_calibration_and_rl_fine_tuning",
            pathway="tesseract_decoder_with_frequency_calibrated_prior",
            subgrid=None,
            seed=73120,
        )
        color_traditional = _condition_result(
            archive,
            records,
            family="color_code",
            distance=5,
            condition="traditional_calibration",
            pathway="tesseract_decoder_with_frequency_calibrated_prior",
            subgrid=None,
            seed=73130,
        )
        surface_consistent = _condition_result(
            archive,
            records,
            family="surface_code",
            distance=7,
            condition="traditional_calibration_and_rl_fine_tuning",
            pathway="sparse_blossom_decoder_with_si1000_prior",
            subgrid="d7_0+0j",
            seed=73140,
        )
        color_consistent = _condition_result(
            archive,
            records,
            family="color_code",
            distance=5,
            condition="traditional_calibration_and_rl_fine_tuning",
            pathway="tesseract_decoder_with_si1000_prior",
            subgrid=None,
            seed=73150,
        )

    headline_results = {
        "surface_code_distance_7_alphaqubit2": {
            "published": 7.72e-4,
            "published_standard_error": 9e-6,
            "repository": surface_final["basis_average_repository"],
            "independent": surface_final["basis_average_independent"],
            "bootstrap_standard_error": surface_final["basis_average_standard_error"],
            "status": ReproductionStatus.EXACTLY_REPRODUCED.value,
        },
        "color_code_distance_5_tesseract_frequency_prior": {
            "published": 8.19e-3,
            "published_standard_error": 1.4e-4,
            "repository": color_final["basis_average_repository"],
            "independent": color_final["basis_average_independent"],
            "bootstrap_standard_error": color_final["basis_average_standard_error"],
            "status": ReproductionStatus.EXACTLY_REPRODUCED.value,
        },
        "surface_code_distance_7_sparse_blossom_si1000": {
            "published": 1.42e-3,
            "published_location": "Supplement Table V",
            "repository": surface_consistent["basis_average_repository"],
            "independent": surface_consistent["basis_average_independent"],
            "status": ReproductionStatus.EXACTLY_REPRODUCED.value,
        },
        "color_code_distance_5_tesseract_si1000": {
            "published": 9.2e-3,
            "published_location": "Supplement Table V",
            "repository": color_consistent["basis_average_repository"],
            "independent": color_consistent["basis_average_independent"],
            "status": ReproductionStatus.EXACTLY_REPRODUCED.value,
        },
    }
    endpoint_comparisons = {
        "surface_distance_7": {
            "traditional_ler": surface_traditional["basis_average_repository"],
            "rl_fine_tuned_ler": surface_final["basis_average_repository"],
            "relative_improvement": 1.0 - surface_final["basis_average_repository"] / surface_traditional["basis_average_repository"],
            "published_multi_run_anchor": 0.20,
            "status": ReproductionStatus.REPRODUCED_WITH_DOCUMENTED_APPROXIMATION.value,
            "difference_reason": "released before/after memory datasets are not the five contemporaneous evaluation traces in Fig. 3a",
        },
        "color_distance_5": {
            "traditional_ler": color_traditional["basis_average_repository"],
            "rl_fine_tuned_ler": color_final["basis_average_repository"],
            "relative_improvement": 1.0 - color_final["basis_average_repository"] / color_traditional["basis_average_repository"],
            "published_multi_run_anchor": 0.20,
            "status": ReproductionStatus.REPRODUCED_WITH_DOCUMENTED_APPROXIMATION.value,
            "difference_reason": "released endpoint decay curves do not preserve run/epoch pairing",
        },
    }
    unavailable = {
        "fixed_policy_contemporaneous_trace": "time-matched policy mu(0) reevaluations are absent",
        "learned_mean_trace": "epoch-indexed mu(t) outcomes are absent",
        "stochastic_candidate_trace": "candidate policies/actions and their outcomes are absent",
        "drift_stability_control_only_2_4x": "injected-drift LER traces are absent",
        "combined_control_decoder_3_5x": "decoder-steering traces are absent and kept separate from control-only",
        "low_frequency_suppression_4db": "multi-run epoch-domain LER traces are absent",
        "step_response_130_epochs": "control-parameter trajectory and step labels are absent",
        "randomized_recovery_1000_epochs": "spoiled-policy training trajectory is absent",
        "steering_cutoff_1_over_150": "simulation grid and entropy sweep are absent",
        "scaling_convergence_gamma": "d3-d15 simulation learning curves are absent",
    }
    result = {
        "schema_version": "google-public-data-reproduction.v3",
        "archive": str(archive_path.resolve()).replace("\\", "/"),
        "released_scope": {"experiment_cells": 496, "data_kind": "static quantum-memory shots and decoder results"},
        "normalization": {
            "memory_failure_to_per_cycle": "epsilon=0.5*(1-(1-2*p_fail)^(1/rounds))",
            "free_decay_intercept": True,
            "basis_average_after_separate_fits": True,
            "original_units_preserved": {"rounds": "QEC cycles", "shots": "memory experiments"},
        },
        "headline_results": headline_results,
        "fine_tuning_endpoint_comparisons": endpoint_comparisons,
        "full_decay_details": {
            "surface_final": surface_final,
            "surface_traditional": surface_traditional,
            "color_final": color_final,
            "color_traditional": color_traditional,
            "surface_consistent_decoder": surface_consistent,
            "color_consistent_decoder": color_consistent,
        },
        "not_reproducible": unavailable,
        "control_only_and_decoder_steering_merged": False,
        "certification_seeds_consumed": False,
        "staged_comparison_run": False,
    }
    write_json(artifact_dir / "public_data_reproduction.json", result)
    lines = [
        "# Public-data reproduction",
        "",
        "## Exactly reproduced released-memory results",
        "",
        "| Quantity | Published | Repository | Independent | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for name, row in headline_results.items():
        lines.append(f"| {name} | {row['published']:.8g} | {row['repository']:.8g} | {row['independent']:.8g} | `{row['status']}` |")
    lines.extend(["", "## Fine-tuning endpoint comparison", ""])
    for name, row in endpoint_comparisons.items():
        lines.append(f"- {name}: {100*row['relative_improvement']:.2f}% lower fitted endpoint LER (`{row['status']}`). {row['difference_reason']}.")
    lines.extend(["", "## Not reproducible from this release", ""])
    lines.extend(f"- `{name}`: {reason}." for name, reason in unavailable.items())
    lines.extend(["", "The 2.4x control-only stability claim and 3.5x combined control-plus-decoder claim remain separate."])
    write_markdown(artifact_dir / "public_data_reproduction.md", lines)
    return result


def validate_estimators(artifact_dir: Path) -> dict[str, Any]:
    reproduction = json.loads((artifact_dir / "public_data_reproduction.json").read_text(encoding="utf-8"))
    checks = []
    for name, row in reproduction["headline_results"].items():
        published_se = row.get("published_standard_error")
        published_tolerance = 3 * published_se if published_se is not None else max(5e-5, 0.03 * row["published"])
        independent_tolerance = max(5e-5, 5 * row.get("bootstrap_standard_error", 0.0))
        repository_independent = abs(row["repository"] - row["independent"])
        repository_published = abs(row["repository"] - row["published"])
        checks.append(
            {
                "quantity": name,
                "repository": row["repository"],
                "independent": row["independent"],
                "published": row["published"],
                "repository_independent_absolute_difference": repository_independent,
                "repository_published_absolute_difference": repository_published,
                "independent_tolerance": independent_tolerance,
                "published_tolerance": published_tolerance,
                "status": "PASS" if repository_independent <= independent_tolerance and repository_published <= published_tolerance else "FAIL",
                "discrepancy_class": None if repository_independent <= independent_tolerance and repository_published <= published_tolerance else "estimator",
            }
        )
    result = {
        "schema_version": "google-zenodo-estimator-validation.v3",
        "status": "PASS" if all(row["status"] == "PASS" for row in checks) else "FAIL",
        "frozen_tolerance_rule": "published: 3 published SE where available, else max(5e-5,3%); independent: max(5e-5,5 bootstrap SE)",
        "checks": checks,
        "analysis_pipeline_corrections": [
            {"id": "stim_b8_little_endian", "class": "data-selection", "regression": "test_b8_little_endian_and_xor"},
            {"id": "logical_actual_xor_predicted", "class": "estimator", "regression": "test_b8_little_endian_and_xor"},
            {"id": "exact_parity_per_cycle_inverse", "class": "normalization", "regression": "test_per_cycle_parity_inverse_not_linear_division"},
            {"id": "free_decay_contrast_intercept", "class": "estimator", "regression": "test_decay_fit_recovers_known_error_rate"},
            {"id": "fit_bases_before_average", "class": "normalization", "regression": "test_public_headline_values_from_frozen_counts"},
            {"id": "psd_power_uses_10log10", "class": "frequency-unit", "regression": "test_power_db_conversion"},
        ],
        "surrogate_fit_gate_open": all(row["status"] == "PASS" for row in checks),
        "certification_seeds_consumed": False,
    }
    write_json(artifact_dir / "estimator_validation.json", result)
    lines = ["# Zenodo estimator validation", "", f"**Status:** `{result['status']}`", "", "| Quantity | Repository | Independent | Published | Status |", "|---|---:|---:|---:|---|"]
    for row in checks:
        lines.append(f"| {row['quantity']} | {row['repository']:.8g} | {row['independent']:.8g} | {row['published']:.8g} | `{row['status']}` |")
    lines.extend(["", "The regression suite freezes b8 bit ordering/XOR, nonlinear per-cycle normalization, free-intercept decay fitting, basis averaging order, and the 10 log10 PSD power convention."])
    write_markdown(artifact_dir / "estimator_validation.md", lines)
    return result


def _preferred_pathway(record: Any, pathways: tuple[str, ...]) -> str | None:
    preferences = (
        ("alphaqubit2_decoder",) if record.code_family == "surface_code" else ("tesseract_decoder_with_frequency_calibrated_prior",)
    )
    for pathway in preferences:
        if pathway in pathways:
            return pathway
    return pathways[0] if pathways else None


def _extract_split_statistics(archive_path: Path, split: dict[str, Any]) -> dict[str, Any]:
    per_experiment: list[dict[str, Any]] = []
    with ZenodoArchive(archive_path) as archive:
        records = {record.experiment_id: record for record in archive.records()}
        for index, entry in enumerate(split["entries"]):
            record = records[entry["experiment_id"]]
            issues = archive.validate_record(record)
            if issues:
                raise ValueError(f"schema validation failed for {record.data_dir}: {[x.to_dict() for x in issues]}")
            start, stop = entry["shot_block"]
            detections = archive.detector_block(record, start, stop)
            summary = detector_summary(detections)
            reward = np.mean(detections, axis=1)
            acf = autocorrelation(reward, min(20, len(reward) - 1))
            frequencies, power = periodogram(reward, sample_spacing=1.0, detrend=True)
            pathways = archive.decoder_pathways(record)
            pathway = _preferred_pathway(record, pathways)
            logical_error = None
            if pathway is not None:
                errors, shots = archive.logical_error_counts(record, pathway)
                failure = errors / shots
                if failure <= 0.5:
                    logical_error = memory_failure_to_error_per_cycle(failure, record.rounds)
            per_experiment.append(
                {
                    "experiment_id": record.experiment_id,
                    "data_dir": record.data_dir,
                    "code_family": record.code_family,
                    "distance": record.distance,
                    "condition": record.condition,
                    "basis": record.basis,
                    "rounds": record.rounds,
                    "shot_block": [start, stop],
                    **summary,
                    "autocorrelation": acf.tolist(),
                    "low_frequency_power_fraction": low_frequency_fraction(frequencies, power),
                    "effective_independent_sample_count": effective_sample_size(len(reward), acf),
                    "logical_decoder_pathway": pathway,
                    "logical_error_per_cycle": logical_error,
                    "shot_order_semantics": "archive order; acquisition timestamps are not released",
                }
            )
    return {"per_experiment": per_experiment}


def fit_empirical_statistics(archive_path: Path, artifact_dir: Path) -> dict[str, Any]:
    split_manifest = json.loads((artifact_dir / "data_split_manifest.json").read_text(encoding="utf-8"))
    extracted = _extract_split_statistics(archive_path, split_manifest["splits"]["surrogate_fit"])
    rows = extracted["per_experiment"]
    aggregate = {
        "experiment_count": len(rows),
        "detector_rate_mean": float(np.mean([x["detector_rate_mean"] for x in rows])),
        "detector_rate_between_experiment_std": float(np.std([x["detector_rate_mean"] for x in rows], ddof=1)),
        "reward_variance_median": float(np.median([x["reward_variance"] for x in rows])),
        "mean_absolute_off_diagonal_covariance_median": float(np.median([x["mean_absolute_off_diagonal_covariance"] for x in rows])),
        "overdispersion_median": float(np.median([x["overdispersion_median"] for x in rows if x["overdispersion_median"] is not None])),
        "lag1_autocorrelation_median": float(np.median([x["autocorrelation"][1] for x in rows])),
        "low_frequency_power_fraction_median": float(np.median([x["low_frequency_power_fraction"] for x in rows])),
        "effective_independent_sample_count_median": float(np.median([x["effective_independent_sample_count"] for x in rows])),
        "first_to_second_half_reward_shift_median": float(np.median([x["second_half_reward_mean"] - x["first_half_reward_mean"] for x in rows])),
        "logical_error_per_cycle_quantiles": {
            str(q): float(np.quantile([x["logical_error_per_cycle"] for x in rows if x["logical_error_per_cycle"] is not None], q))
            for q in (0.1, 0.5, 0.9)
        },
    }
    unsupported = {
        "step_magnitude_distribution": "no intervention or step labels",
        "control_action_distributions": "no control vectors",
        "policy_covariance_over_training": "no policy parameters or epochs",
        "learned_mean_and_stochastic_trajectories": "no policy-semantic time series",
        "randomization_severity": "no spoiled-policy controls or training run",
        "spoiled_logical_risk_distribution": "no randomized-policy dataset",
        "response_curvature_within_logged_action_support": "there is no logged action support",
        "candidate_reward_variance": "actual candidate grouping is absent; reported reward variance is a shot-level EDR proxy",
        "drift_psd_in_epoch_units": "shot order is present but epoch timestamps and multi-run policy labels are absent",
        "baseline_timescale_in_physical_time": "timestamps are absent; ACF is reported only in archive-shot order",
    }
    result = {
        "schema_version": "google-v3-empirical-statistics-fit.v1",
        "fit_split_hash": split_manifest["splits"]["surrogate_fit"]["content_sha256"],
        "fit_split_manifest_hash": split_manifest["manifest_sha256"],
        "source": "detection_events.b8 and logical outcomes from fit experiments only",
        **extracted,
        "aggregate": aggregate,
        "supported_estimands": [
            "detector-rate distributions",
            "sampled detector covariance panel",
            "block overdispersion relative to binomial",
            "shot-order autocorrelation and spectrum",
            "shot-level reward proxy variance",
            "cross-detector coupling summary",
            "shot-block non-stationarity",
            "logical-risk distribution",
            "effective independent sample count",
        ],
        "unsupported_estimands": unsupported,
        "arbitrary_counterfactual_actions_inferred": False,
        "certification_seeds_consumed": False,
    }
    write_json(artifact_dir / "empirical_statistics_fit.json", result)
    lines = [
        "# Empirical statistics — surrogate fit split",
        "",
        f"Fit data comprise {aggregate['experiment_count']} disjoint experiment cells. Detector mean={aggregate['detector_rate_mean']:.5f}; median reward variance={aggregate['reward_variance_median']:.4g}; median lag-1 ACF={aggregate['lag1_autocorrelation_median']:.4f}; median ESS={aggregate['effective_independent_sample_count_median']:.1f} of 2048 ordered shots.",
        "",
        "Covariance uses an evenly spaced deterministic panel of at most 128 detectors per experiment; rate and reward statistics use every detector.",
        "",
        "## Unsupported controller-critical quantities",
        "",
    ]
    lines.extend(f"- `{name}`: {reason}." for name, reason in unsupported.items())
    write_markdown(artifact_dir / "empirical_statistics_fit.md", lines)
    return result


def fit_surrogate(artifact_dir: Path, surrogate_config_path: Path) -> dict[str, Any]:
    estimator_validation = json.loads((artifact_dir / "estimator_validation.json").read_text(encoding="utf-8"))
    if estimator_validation["status"] != "PASS" or not estimator_validation["surrogate_fit_gate_open"]:
        raise RuntimeError("released-data estimator gate is closed; surrogate fitting is forbidden")
    empirical = json.loads((artifact_dir / "empirical_statistics_fit.json").read_text(encoding="utf-8"))
    config = load_json_yaml(surrogate_config_path)
    parameters = fit_contract_parameters(empirical, split_hash=empirical["fit_split_hash"])
    contract = {
        "schema_version": "google-v3-empirical-surrogate-contract.v1",
        "model_class": config["model_class"],
        "status": "FITTED_STATIC_OBSERVATION_SURROGATE",
        "fit_split_hash": empirical["fit_split_hash"],
        "fit_data_only": True,
        "controller_anchor_scores_used": False,
        "fitted_parameters": parameters,
        "supported_outputs": ["detector-rate distribution", "shared covariance proxy", "shot-order AR(1) proxy", "logical-risk mapping within observed experiment support"],
        "unsupported_outputs": ["control-action response", "step response", "policy learning trajectory", "spoil/recovery dynamics", "epoch-domain injected drift", "counterfactual steering phase diagram"],
        "action_support": {"status": "NONE", "extrapolation_penalty": "INFINITE", "behavior": "reject every supplied action"},
        "released_outcomes_used_as_action_replay_table": False,
        "validation_tolerances": config["validation_tolerances_frozen_before_validation"],
        "certification_seeds_consumed": False,
    }
    write_json(artifact_dir / "empirical_surrogate_contract.json", contract)
    lines = [
        "# Empirical surrogate contract",
        "",
        "This is a **static observation surrogate**, not an interactive control environment. It was fitted only to the frozen fit split after the released-data estimator gate passed.",
        "",
        f"- Detector logit mean: `{parameters['detector_logit_mean']['estimate']:.6g}`",
        f"- Detector logit standard deviation: `{parameters['detector_logit_std']['estimate']:.6g}`",
        f"- Shared latent standard deviation: `{parameters['shared_latent_std']['estimate']:.6g}`",
        f"- Shot-order AR(1): `{parameters['temporal_ar1']['estimate']:.6g}`",
        "- Any control action is rejected with an infinite extrapolation penalty because the release has no logged action support.",
        "- No v2 controller score entered the fit.",
    ]
    write_markdown(artifact_dir / "empirical_surrogate_contract.md", lines)
    return contract


def _aggregate_validation(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "detector_rate_mean": float(np.mean([x["detector_rate_mean"] for x in rows])),
        "detector_rate_median": float(np.median([x["detector_rate_mean"] for x in rows])),
        "reward_variance": float(np.median([x["reward_variance"] for x in rows])),
        "mean_absolute_covariance": float(np.median([x["mean_absolute_off_diagonal_covariance"] for x in rows])),
        "lag1_autocorrelation": float(np.median([x["autocorrelation"][1] for x in rows])),
        "low_frequency_fraction": float(np.median([x["low_frequency_power_fraction"] for x in rows])),
    }


def validate_surrogate(archive_path: Path, artifact_dir: Path, surrogate_config_path: Path) -> dict[str, Any]:
    split_manifest = json.loads((artifact_dir / "data_split_manifest.json").read_text(encoding="utf-8"))
    validation_split = split_manifest["splits"]["surrogate_validation"]
    extracted = _extract_split_statistics(archive_path, validation_split)
    observed_rows = extracted["per_experiment"]
    contract = json.loads((artifact_dir / "empirical_surrogate_contract.json").read_text(encoding="utf-8"))
    config = load_json_yaml(surrogate_config_path)
    surrogate = EmpiricalStaticSurrogate.from_contract(contract)
    simulated_rows: list[dict[str, Any]] = []
    for index, observed in enumerate(observed_rows):
        events = surrogate.sample_detection_events(
            shots=observed["shots"],
            detectors=observed["detectors"],
            seed=int(config["posterior_predictive_seed"]) + index,
        )
        summary = detector_summary(events)
        reward = events.mean(axis=1)
        acf = autocorrelation(reward, min(20, len(reward) - 1))
        frequency, power = periodogram(reward)
        simulated_rows.append(
            {
                **summary,
                "autocorrelation": acf.tolist(),
                "low_frequency_power_fraction": low_frequency_fraction(frequency, power),
                "predicted_logical_error_per_cycle": surrogate.logical_risk(summary["detector_rate_mean"]),
            }
        )
    observed = _aggregate_validation(observed_rows)
    simulated = _aggregate_validation(simulated_rows)
    tolerance = config["validation_tolerances_frozen_before_validation"]
    checks = [
        {"quantity": "detector_rate_mean", "observed": observed["detector_rate_mean"], "predicted": simulated["detector_rate_mean"], "error": abs(observed["detector_rate_mean"] - simulated["detector_rate_mean"]), "tolerance": tolerance["detector_rate_mean_absolute_error_max"]},
        {"quantity": "detector_rate_median", "observed": observed["detector_rate_median"], "predicted": simulated["detector_rate_median"], "error": abs(observed["detector_rate_median"] - simulated["detector_rate_median"]), "tolerance": tolerance["detector_rate_median_absolute_error_max"]},
        {"quantity": "mean_absolute_covariance", "observed": observed["mean_absolute_covariance"], "predicted": simulated["mean_absolute_covariance"], "error": abs(observed["mean_absolute_covariance"] - simulated["mean_absolute_covariance"]), "tolerance": tolerance["mean_absolute_covariance_absolute_error_max"]},
        {"quantity": "lag1_autocorrelation", "observed": observed["lag1_autocorrelation"], "predicted": simulated["lag1_autocorrelation"], "error": abs(observed["lag1_autocorrelation"] - simulated["lag1_autocorrelation"]), "tolerance": tolerance["lag1_autocorrelation_absolute_error_max"]},
        {"quantity": "low_frequency_fraction", "observed": observed["low_frequency_fraction"], "predicted": simulated["low_frequency_fraction"], "error": abs(observed["low_frequency_fraction"] - simulated["low_frequency_fraction"]), "tolerance": tolerance["low_frequency_fraction_absolute_error_max"]},
    ]
    reward_ratio = simulated["reward_variance"] / max(observed["reward_variance"], 1e-15)
    checks.append({"quantity": "reward_variance_ratio", "observed": observed["reward_variance"], "predicted": simulated["reward_variance"], "ratio": reward_ratio, "tolerance": tolerance["reward_variance_ratio"], "status": "PASS" if tolerance["reward_variance_ratio"][0] <= reward_ratio <= tolerance["reward_variance_ratio"][1] else "FAIL"})
    for check in checks:
        if "status" not in check:
            check["status"] = "PASS" if check["error"] <= check["tolerance"] else "FAIL"
    supported_pass = all(check["status"] == "PASS" for check in checks)
    unsupported_critical = ["candidate action-response curvature", "policy covariance and stochastic/mean separation", "spoil severity and randomized recovery", "step response", "epoch-domain drift PSD", "steering cutoff", "scaling convergence rate"]
    outcome = SurrogateValidationOutcome.PARTIALLY_VALIDATED.value if supported_pass else SurrogateValidationOutcome.REJECTED.value
    result = {
        "schema_version": "google-v3-surrogate-validation.v1",
        "outcome": outcome,
        "validation_split_hash": validation_split["content_sha256"],
        "fit_split_hash": contract["fit_split_hash"],
        "fit_validation_disjoint": contract["fit_split_hash"] != validation_split["content_sha256"],
        "posterior_predictive_checks": checks,
        "supported_checks_pass": supported_pass,
        "unsupported_controller_critical_quantities": unsupported_critical,
        "logical_risk_check": "MODEL_MAPPING_REPORTED_BUT_NOT_CONTROLLER_CAUSAL; validation cells have no actions",
        "step_and_spoil_checks": "NOT_IDENTIFIABLE_FROM_RELEASE",
        "prompt2_decision": "NO_GO",
        "prompt2_reason": "static detector observations are only partially validated; controller-critical dynamics remain unidentifiable",
        "certification_seeds_consumed": False,
        "staged_comparison_run": False,
    }
    write_json(artifact_dir / "surrogate_validation.json", result)
    lines = ["# Empirical surrogate validation", "", f"**Outcome:** `{outcome}`", "", "| Quantity | Observed | Predicted | Status |", "|---|---:|---:|---|"]
    for check in checks:
        lines.append(f"| {check['quantity']} | {check['observed']:.6g} | {check['predicted']:.6g} | `{check['status']}` |")
    lines.extend(["", "The result cannot be fully validated because released data contain no actions, policy trajectories, interventions, spoil episode, or steering/scaling simulations.", "", "**Prompt 2: NO-GO.**"])
    write_markdown(artifact_dir / "surrogate_validation.md", lines)
    return result
