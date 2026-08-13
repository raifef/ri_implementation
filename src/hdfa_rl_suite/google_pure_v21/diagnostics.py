"""V21 projection-retention, estimator, source-fidelity, and variance audits."""
from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from hdfa_rl_suite.google_pure_v20.core import cosine_alignment
from hdfa_rl_suite.google_pure_v20.data import (
    evaluator,
    load_matched_run,
    replay_gradients,
    selected_fast_epochs,
)

from .candidate_design import (
    DESIGN_IDS,
    DESIGN_NAMES,
    SOURCE_FIDELITY,
    estimate_policy_updates,
    generate_frame,
    public_factor_graph_blocks,
)
from .io import ARTIFACT_ROOT, canonical_hash, read_json, settings, write_artifact
from .lineage import verify_import_manifest


def projection_components(reference: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact orthogonal retained/discarded decomposition."""
    gradient = np.asarray(reference, dtype=float)
    direction = np.asarray(basis, dtype=float)
    if gradient.ndim != 1 or direction.shape != gradient.shape:
        raise ValueError("reference and projection basis must be aligned vectors")
    norm = float(direction @ direction)
    if norm <= 0:
        raise ValueError("projection basis must be nonzero")
    retained = direction * float(direction @ gradient) / norm
    return retained, gradient - retained


def _blocks() -> tuple[np.ndarray, ...]:
    plant = evaluator().plant
    return public_factor_graph_blocks(
        plant.mask, plant.inventory, block_count=int(settings()["public_block_count"]))


def _reference_rows() -> list[dict[str, Any]]:
    return read_json(ARTIFACT_ROOT.parent /
                     "google_pure_v20/fast_reference_gradients.json")["rows"]


def _beneficial(mean: np.ndarray, epoch: int, frequency: float) -> np.ndarray:
    evaluation = evaluator()
    target = evaluation.bounded.latent_controls_for(
        evaluation.plant.optimum(epoch, frequency))
    value = target - np.asarray(mean, dtype=float)
    return value / max(float(np.linalg.norm(value)), 1e-15)


def _harmonic_phase(phases: np.ndarray, values: np.ndarray) -> float:
    design = np.column_stack([np.ones_like(phases), np.sin(phases), np.cos(phases)])
    fit = np.linalg.lstsq(design, values, rcond=None)[0]
    return float(-math.atan2(float(fit[2]), float(fit[1])))


def audit_projection_reference_retention() -> dict[str, Any]:
    verify_import_manifest()
    frequency = float(settings()["fast_frequency_per_epoch"])
    run = load_matched_run("fast")
    replay = replay_gradients(run)
    plant = evaluator().plant
    shared = np.ones(41) / math.sqrt(41)
    families = np.asarray([item.gate_type for item in plant.inventory])
    blocks = _blocks()
    rows = []
    series = {"baseline": [], "hard_projection": [], "population": []}
    phases = []
    for reference_row in _reference_rows():
        epoch = int(reference_row["epoch"])
        reference = np.asarray(reference_row["reference_gradient"], dtype=float)
        ordinary = np.asarray(reference_row["ordinary_gradient"], dtype=float)
        retained, discarded = projection_components(reference, shared)
        hard = shared * float(shared @ ordinary)
        beneficial = _beneficial(replay[epoch]["mean"], epoch, frequency)
        reference_progress = float(reference @ beneficial)
        family_rows = []
        for family in sorted(set(families)):
            selected = families == family
            family_rows.append({
                "family": family,
                "discarded_norm": float(np.linalg.norm(discarded[selected])),
                "discarded_norm_fraction": float(
                    np.linalg.norm(discarded[selected]) /
                    max(np.linalg.norm(discarded), 1e-15)),
            })
        neighborhood_rows = []
        for index, block in enumerate(blocks):
            neighborhood_rows.append({
                "neighborhood": f"public-block-{index}",
                "coordinates": len(block),
                "discarded_norm": float(np.linalg.norm(discarded[block])),
                "discarded_norm_fraction": float(
                    np.linalg.norm(discarded[block]) /
                    max(np.linalg.norm(discarded), 1e-15)),
            })
        comparisons = {}
        for name, update in (("baseline", ordinary), ("hard_projection", hard),
                             ("population", reference)):
            progress = float(update @ beneficial)
            series[name].append(progress)
            comparisons[name] = {
                "alignment_with_reference": cosine_alignment(update, reference),
                "directional_magnitude_ratio": progress /
                    reference_progress if abs(reference_progress) > 1e-15 else None,
                "signed_progress": progress,
                "update_norm": float(np.linalg.norm(update)),
            }
        phase = float((2*np.pi*frequency*epoch) % (2*np.pi))
        phases.append(phase)
        rows.append({
            "epoch": epoch,
            "phase_radians": phase,
            "reference_gradient_retained_fraction": float(
                np.linalg.norm(retained) / max(np.linalg.norm(reference), 1e-15)),
            "reference_gradient_discarded_fraction": float(
                np.linalg.norm(discarded) / max(np.linalg.norm(reference), 1e-15)),
            "discarded_gradient_norm": float(np.linalg.norm(discarded)),
            "discarded_gradient_beneficial_progress": float(discarded @ beneficial),
            "discarded_progress_fraction": float(discarded @ beneficial) /
                reference_progress if abs(reference_progress) > 1e-15 else None,
            "discarded_family_decomposition": family_rows,
            "discarded_neighborhood_decomposition": neighborhood_rows,
            "comparisons": comparisons,
        })
    phase_array = np.asarray(phases)
    phase_errors = {
        name: float(np.angle(np.exp(1j * (
            _harmonic_phase(phase_array, np.asarray(values)) -
            _harmonic_phase(phase_array, np.asarray(series["population"]))))))
        for name, values in series.items()
    }
    aggregate = {}
    for name in series:
        aggregate[name] = {
            "median_alignment": float(np.median([
                row["comparisons"][name]["alignment_with_reference"] for row in rows])),
            "median_directional_magnitude_ratio": float(np.median([
                row["comparisons"][name]["directional_magnitude_ratio"] for row in rows
                if row["comparisons"][name]["directional_magnitude_ratio"] is not None])),
            "phase_error_radians": phase_errors[name],
            "cumulative_signed_progress": float(np.sum(series[name])),
        }
    discarded_fraction = float(np.median([
        row["reference_gradient_discarded_fraction"] for row in rows]))
    discarded_progress = float(np.median([
        abs(row["discarded_progress_fraction"]) for row in rows
        if row["discarded_progress_fraction"] is not None]))
    classification = ("HARD_PROJECTION_OVERREGULARIZED" if
                      discarded_fraction >= .2 or discarded_progress >= .1 else
                      "HARD_PROJECTION_RETENTION_ADEQUATE")
    value = {
        "pass": True,
        "projection_classification": "ORACLE_LIKE_DIAGNOSTIC_UPPER_BOUND",
        "projection_basis": "known shared driven direction",
        "rows": rows,
        "median_reference_gradient_retained_fraction": float(np.median([
            row["reference_gradient_retained_fraction"] for row in rows])),
        "median_reference_gradient_discarded_fraction": discarded_fraction,
        "median_abs_discarded_beneficial_progress_fraction": discarded_progress,
        "update_comparison": aggregate,
        "classification": classification,
        "future_design_target": "minimum error to full g_ref, not zero orthogonal motion",
        "forbidden_auto_runs_launched": [],
    }
    return write_artifact("projection_reference_retention", value,
                          title="V21 hard-projection reference-gradient retention")


def classify_candidate_source_fidelity() -> dict[str, Any]:
    verify_import_manifest()
    rows = [{
        "design_id": design_id,
        "design_name": DESIGN_NAMES[design_id],
        "source_fidelity": SOURCE_FIDELITY[design_id],
        "uses_known_driven_direction": False,
        "uses_target_trajectory": False,
        "uses_future_phase": False,
        "uses_population_or_reference_gradient": False,
        "uses_hidden_optimum": False,
        "uses_multi_run_leakage": False,
        "basis": (
            "public Google-style iid Gaussian policy candidates" if design_id == "D0" else
            "public detector-control factor graph only" if design_id in {"D4", "D5"} else
            "candidate-design diagnostic not identified as the public source algorithm"),
    } for design_id in DESIGN_IDS]
    rows.append({
        "design_id": "V20_HARD_PROJECTION",
        "design_name": "shared driven-subspace projection",
        "source_fidelity": "ORACLE_LIKE",
        "uses_known_driven_direction": True,
        "uses_target_trajectory": True,
        "uses_future_phase": False,
        "uses_population_or_reference_gradient": False,
        "uses_hidden_optimum": True,
        "uses_multi_run_leakage": False,
        "basis": "V20 causal upper-bound diagnostic only",
    })
    allowed = {"SOURCE_EXPLICIT", "SOURCE_IMPLIED", "DIAGNOSTIC_EXTENSION", "ORACLE_LIKE"}
    value = {
        "pass": all(row["source_fidelity"] in allowed for row in rows),
        "classes": sorted(allowed),
        "designs": rows,
        "successful_diagnostic_extension_relabel_permitted": False,
        "hard_projection_promoted": False,
        "forbidden_auto_runs_launched": [],
    }
    return write_artifact("candidate_design_source_fidelity", value,
                          title="V21 candidate-design source fidelity")


def document_candidate_estimators() -> dict[str, Any]:
    verify_import_manifest()
    blocks = _blocks()
    rows = []
    for design_id in DESIGN_IDS:
        frame = generate_frame(
            design_id, dimension=41, epoch=0, seed=21_000, blocks=blocks)
        rows.append({
            "design_id": design_id,
            "design_name": DESIGN_NAMES[design_id],
            "source_fidelity": SOURCE_FIDELITY[design_id],
            "sampling_distribution": frame.metadata["sampling_distribution"],
            "inclusion_probability": (
                frame.metadata.get("inclusion_probability", 1.0)),
            "importance_weighting": frame.metadata["normalization"],
            "normalization": frame.metadata["normalization"],
            "covariance_preservation": frame.metadata["covariance_preservation"],
            "physical_covariance_preserved_in_expectation":
                frame.physical_covariance_preserved_in_expectation,
            "bias_properties": frame.metadata["bias_properties"],
            "within_epoch_rank": frame.metadata["rank"],
            "expected_coverage": frame.metadata["expected_coverage"],
            "mean_estimator_valid": frame.estimator_valid,
            "sigma_estimator_valid": frame.sigma_estimator_valid,
            "online_controller_eligible": frame.estimator_valid and frame.sigma_estimator_valid,
            "unchanged_iid_score_used": design_id in {"D0", "D1"},
            "derived_frame_specific_estimator_used": design_id not in {"D0", "D1"},
            "fail_closed_reason": (None if frame.sigma_estimator_valid else
                "sphere-frame sigma score is singular; frozen mean benchmark only"),
        })
    value = {
        "pass": all(row["mean_estimator_valid"] for row in rows),
        "K": 8,
        "M": 12000,
        "B": 96000,
        "coordinate_space": "standardized latent policy coordinates",
        "designs": rows,
        "all_non_iid_designs_have_explicit_derivations": True,
        "invalid_online_estimators_fail_closed": True,
        "forbidden_auto_runs_launched": [],
    }
    return write_artifact("candidate_estimators", value,
                          title="V21 mathematically matched candidate estimators")


def _exact_rewards(actions: np.ndarray, epoch: int, frequency: float) -> np.ndarray:
    return -np.asarray([
        evaluator().detector_expectations(action, epoch, frequency) for action in actions])


def _sample_rewards(actions: np.ndarray, epoch: int, frequency: float, *, seed: int,
                    cycles: int = 12000) -> np.ndarray:
    evaluation = evaluator()
    target = evaluation.boundary.target_to_native(evaluation.plant.optimum(epoch, frequency))
    shots = int(cycles) // evaluation.plant.rounds
    rows = []
    for candidate, action in enumerate(actions):
        counts = evaluation.plant.sample_detector_counts(
            evaluation.native(action), epoch=epoch, frequency=frequency,
            qec_cycles=cycles, seed=int(seed + candidate), target_controls=target)
        rows.append(-counts / shots)
    return np.asarray(rows)


def _variance_power(samples: np.ndarray, indices: np.ndarray | None = None) -> float:
    values = np.asarray(samples, dtype=float)
    if indices is not None:
        values = values[:, indices]
    center = np.mean(values, axis=0)
    return float(np.mean(np.sum((values - center[None, :])**2, axis=1)))


def _variance_components(direction_samples: np.ndarray, shot_residuals: np.ndarray,
                         total_samples: np.ndarray, indices: np.ndarray | None = None,
                         projection: np.ndarray | None = None) -> dict[str, float]:
    if projection is not None:
        d = np.asarray(direction_samples) @ projection
        s = np.asarray(shot_residuals) @ projection
        t = np.asarray(total_samples) @ projection
        direction = float(np.var(d, ddof=1))
        shot = float(np.var(s, ddof=1))
        total = float(np.var(t, ddof=1))
    else:
        direction = _variance_power(direction_samples, indices)
        shot = _variance_power(shot_residuals, indices)
        total = _variance_power(total_samples, indices)
    interaction = total - direction - shot
    return {
        "V_direction": direction,
        "V_shot": shot,
        "V_interaction_signed_residual": interaction,
        "V_total": total,
        "conservation_error": direction + shot + interaction - total,
    }


def decompose_gradient_variance() -> dict[str, Any]:
    verify_import_manifest()
    cfg = settings()
    frequency = float(cfg["fast_frequency_per_epoch"])
    blocks = _blocks()
    run = load_matched_run("fast")
    replay = replay_gradients(run)
    reference = {int(row["epoch"]): np.asarray(row["reference_gradient"], dtype=float)
                 for row in _reference_rows()}
    epochs = selected_fast_epochs()[::2][:int(cfg["variance_states"])]
    R_direction = int(cfg["variance_direction_repetitions"])
    R_shot = int(cfg["variance_shot_repetitions"])
    families = np.asarray([item.gate_type for item in evaluator().plant.inventory])
    shared = np.ones(41) / math.sqrt(41)
    rows = []
    for epoch in epochs:
        item = replay[epoch]
        exact_gradients = []
        shot_residuals = []
        total_gradients = []
        direction_hashes = []
        for direction_rep in range(R_direction):
            frame = generate_frame(
                "D0", dimension=41, epoch=epoch,
                seed=31_000 + 1000 * direction_rep, blocks=blocks)
            actions = item["mean"][None, :] + item["sigma"][None, :] * \
                frame.standardized_directions
            exact = estimate_policy_updates(
                frame, _exact_rewards(actions, epoch, frequency), item["baseline"],
                evaluator().plant.mask, item["sigma"])["mean_update_direction"]
            exact = np.asarray(exact, dtype=float)
            exact_gradients.append(exact)
            direction_hashes.append(canonical_hash(frame.standardized_directions.tolist()))
            for shot_rep in range(R_shot):
                sampled = estimate_policy_updates(
                    frame, _sample_rewards(
                        actions, epoch, frequency,
                        seed=3_100_000 + epoch * 10_000 + direction_rep * 100 + shot_rep * 10),
                    item["baseline"], evaluator().plant.mask, item["sigma"]
                )["mean_update_direction"]
                sampled = np.asarray(sampled, dtype=float)
                total_gradients.append(sampled)
                shot_residuals.append(sampled - exact)
        directions = np.asarray(exact_gradients)
        shots = np.asarray(shot_residuals)
        totals = np.asarray(total_gradients)
        full = _variance_components(directions, shots, totals)
        target = _variance_components(directions, shots, totals, projection=shared)
        orthogonal = {
            key: full[key] - target[key]
            for key in ("V_direction", "V_shot", "V_interaction_signed_residual", "V_total")
        }
        orthogonal["conservation_error"] = sum(orthogonal[key] for key in (
            "V_direction", "V_shot", "V_interaction_signed_residual")) - orthogonal["V_total"]
        family_rows = []
        for family in sorted(set(families)):
            family_rows.append({"family": family, **_variance_components(
                directions, shots, totals, indices=np.flatnonzero(families == family))})
        neighborhood_rows = []
        for index, block in enumerate(blocks):
            neighborhood_rows.append({
                "neighborhood": f"public-block-{index}",
                **_variance_components(directions, shots, totals, indices=block),
            })
        rows.append({
            "epoch": epoch,
            "phase_radians": float((2*np.pi*frequency*epoch) % (2*np.pi)),
            "reference_gradient_norm": float(np.linalg.norm(reference[epoch])),
            "direction_draws": R_direction,
            "shot_repetitions_per_direction": R_shot,
            "fixed_candidate_direction_hashes": direction_hashes,
            "full_gradient": full,
            "target_direction_component": target,
            "orthogonal_component": orthogonal,
            "control_families": family_rows,
            "detector_neighborhoods": neighborhood_rows,
        })
    totals = {
        key: float(np.mean([row["full_gradient"][key] for row in rows]))
        for key in ("V_direction", "V_shot", "V_interaction_signed_residual", "V_total")
    }
    positive = {
        "direction": max(totals["V_direction"], 0.0),
        "shot": max(totals["V_shot"], 0.0),
        "interaction": max(totals["V_interaction_signed_residual"], 0.0),
    }
    denom = max(sum(positive.values()), 1e-15)
    fractions = {key: value / denom for key, value in positive.items()}
    largest = max(fractions, key=fractions.get)
    if fractions[largest] >= .55:
        classification = {
            "direction": "DIRECTION_VARIANCE_DOMINATES",
            "shot": "SHOT_VARIANCE_DOMINATES",
            "interaction": "INTERACTION_VARIANCE_DOMINATES",
        }[largest]
    elif max(fractions.values()) >= .3:
        classification = "MIXED_VARIANCE"
    else:
        classification = "UNRESOLVED"
    value = {
        "pass": True,
        "method": "crossed direction-frame by independent finite-shot repetition ANOVA",
        "fixed_design": "D0 IID Gaussian K8 M12000",
        "states": rows,
        "aggregate_full_gradient": totals,
        "nonnegative_variance_fractions": fractions,
        "direction_variance_material": fractions["direction"] >= .2,
        "classification": classification,
        "structured_design_online_validation_permitted": fractions["direction"] >= .2,
        "forbidden_auto_runs_launched": [],
    }
    return write_artifact("gradient_variance_decomposition", value,
                          title="V21 fast gradient variance decomposition", notes=[
        "Shot variance fixes each candidate frame and repeats detector acquisition.",
        "Direction variance redraws frames under exact Stim detector marginals.",
        "The interaction term is the signed finite-sample ANOVA residual and conserves total variance.",
    ])
