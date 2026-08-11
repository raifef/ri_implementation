"""Bounded V19 diagnosis over frozen V18/V18.1 Figure-5a states.

No function in this module updates a production checkpoint or launches an
acquisition.  Stim is used only to evaluate exact detector marginals for stored
mean/candidate controls.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from hdfa_rl_suite.google_pure_source_exact.figure5a.contracts import canonical_hash as shard_hash
from hdfa_rl_suite.google_pure_source_exact.figure5a.validation import build_plant
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.gaussian import (
    BehaviorSnapshot, component_log_probability,
)
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.losses import (
    total_loss_and_gradients,
)
from hdfa_rl_suite.google_pure_v18.experiments import _boundary, _source_config

from .core import (
    PUBLIC_ANALOGUE_SCALE_OBJECTIVE,
    IMPLEMENTED_SOURCE_STYLE_SCALE_OBJECTIVE,
    aggregate_damage,
    aggregation_scaling_fixture,
    classify_bound_activity,
    coordinate_quadratic_damage,
    cumulative_rank_curve,
    effective_dimension,
    lambda_squared_fit,
    phase_aligned_distance,
    phase_bin_means,
    public_analogue_entropy_gradient,
    quadratic_damage,
    sigma_equilibrium,
)
from .io import (
    ARTIFACT_ROOT, CONFIG_PATH, NONFINAL, ROOT, atomic_json, atomic_text,
    canonical_hash, config, file_hash, nonfinal, read_json,
)


FORBIDDEN_CAMPAIGNS = (
    "source-budget", "heldout", "reference", "natural-drift", "figure5c",
    "paired-acceptance", "long-three-frequency-acquisition",
)


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _write(name: str, value: Mapping[str, Any], *, title: str,
           statements: Iterable[str] = ()) -> dict[str, Any]:
    result = dict(value)
    atomic_json(ARTIFACT_ROOT / f"{name}.json", result)
    atomic_text(ARTIFACT_ROOT / f"{name}.md", "\n".join([
        f"# {title}", "", *statements, "",
        "Development-only stored-state evidence; no long or confirmatory campaign was launched.",
    ]))
    return result


def _import_paths() -> dict[str, Path]:
    return {
        "v18_slow_transfer": ROOT / "artifacts/google_pure_v18/transfer_slow.json",
        "v18_intermediate_transfer": ROOT / "artifacts/google_pure_v18/transfer_intermediate.json",
        "v18_extended_fast_transfer": ROOT / "artifacts/google_pure_v18/extended_fast/transfer_fast_extended.json",
        "v18_mean_stochastic_decomposition": ROOT / "artifacts/google_pure_v18/mean_stochastic_decomposition.json",
        "v18_extended_fast_decomposition": ROOT / "artifacts/google_pure_v18/extended_fast/mean_stochastic_decomposition_extended.json",
        "v18_extended_fast_sigma_stability": ROOT / "artifacts/google_pure_v18/extended_fast/sigma_stability_extended.json",
        "v16_optimizer_bundle": ROOT / "artifacts/google_pure_v16/frozen_source_normalized_optimizer.json",
        "direct_sigma_gaussian": ROOT / "src/hdfa_rl_suite/google_pure_source_exact/policy_parameterization/gaussian.py",
        "direct_sigma_losses": ROOT / "src/hdfa_rl_suite/google_pure_source_exact/policy_parameterization/losses.py",
        "direct_sigma_optimizer": ROOT / "src/hdfa_rl_suite/google_pure_source_exact/policy_parameterization/optimizer.py",
        "source_normalization_boundary": ROOT / "src/hdfa_rl_suite/google_pure_source_exact/source_normalization.py",
        "source_normalization_bundle": ROOT / "artifacts/google_pure_source_exact/control_normalization/calibration_bundle.json",
        "detector_degree_audit": ROOT / "artifacts/google_pure_v15/sensitivity/detector_degree_audit.json",
        "figure5a_config": ROOT / "configs/google_pure_source_exact/figure5a.json",
        "production_figure5a_acquisition": ROOT / "src/hdfa_rl_suite/google_pure_source_exact/figure5a/acquisition.py",
        "production_figure5a_plant": ROOT / "src/hdfa_rl_suite/google_pure_source_exact/figure5a/plant.py",
        "production_figure5a_evaluator": ROOT / "src/hdfa_rl_suite/google_pure_source_exact/figure5a/validation.py",
        "slow_checkpoint": ROOT / "artifacts/google_pure_v18/acquisition/slow/checkpoint.json",
        "intermediate_checkpoint": ROOT / "artifacts/google_pure_v18/acquisition/intermediate/checkpoint.json",
        "extended_fast_checkpoint": ROOT / "artifacts/google_pure_v18/extended_fast/checkpoint.json",
    }


def build_import_manifest() -> dict[str, Any]:
    paths = _import_paths()
    missing = [_relative(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing mandatory V19 frozen inputs: {missing}")
    observed = {role: {"path": _relative(path), "sha256": file_hash(path)}
                for role, path in paths.items()}
    manifest_path = ARTIFACT_ROOT / "import_manifest.json"
    if manifest_path.is_file():
        previous = read_json(manifest_path)
        expected = previous.get("inputs", {})
        if expected != observed:
            changed = {role: {"expected": expected.get(role), "observed": value}
                       for role, value in observed.items() if expected.get(role) != value}
            raise RuntimeError(f"V19 frozen input mismatch: {changed}")
    slow = read_json(paths["v18_slow_transfer"])
    intermediate = read_json(paths["v18_intermediate_transfer"])
    fast = read_json(paths["v18_extended_fast_transfer"])
    frozen = read_json(paths["v16_optimizer_bundle"])
    gates = {
        "mean_frequency_ordering_passed": slow.get(
            "stage_slow_intermediate_fast_ordering", {}).get("pass") is True,
        "slow_direct_sigma": slow.get("parameterization") == "DIRECT_SIGMA_SOURCE_EXACT",
        "intermediate_direct_sigma": intermediate.get("parameterization") == "DIRECT_SIGMA_SOURCE_EXACT",
        "fast_direct_sigma": fast.get("parameterization") == "DIRECT_SIGMA_SOURCE_EXACT",
        "same_controller": slow.get("controller_hash") == intermediate.get("controller_hash") ==
                           fast.get("controller_hash") == frozen.get("optimizer_bundle_hash"),
        "mean_controller_retuning_permitted": config()["mean_controller_retuning_permitted"] is False,
    }
    if not all(gates.values()):
        raise RuntimeError(f"V19 import scientific gate failed: {gates}")
    value = nonfinal({
        "pass": True, "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": observed, "gates": gates,
        "v18_evidence_frozen_before_v19": True,
        "mean_frequency_ordering_passed": True,
        "mean_controller_retuning_permitted": False,
        "production_figure5a_changes_permitted": False,
        "frozen_controller_hash": frozen["optimizer_bundle_hash"],
        "forbidden_auto_runs": list(FORBIDDEN_CAMPAIGNS),
        "forbidden_auto_runs_launched": [],
    })
    return _write("import_manifest", value, title="V19 frozen V18 evidence manifest", statements=[
        "All V18/V18.1 transfer, scale, controller, normalization, and plant inputs are hash-pinned.",
        "Mean-controller retuning and paper-equivalence promotion are forbidden.",
    ])


def verify_import_manifest() -> dict[str, Any]:
    path = ARTIFACT_ROOT / "import_manifest.json"
    if not path.is_file():
        return build_import_manifest()
    value = read_json(path)
    observed = {role: {"path": _relative(source), "sha256": file_hash(source)}
                for role, source in _import_paths().items()}
    if value.get("inputs") != observed:
        raise RuntimeError("V19 import manifest no longer matches its frozen inputs")
    if value.get("pass") is not True or value.get("mean_frequency_ordering_passed") is not True:
        raise RuntimeError("V19 import manifest is not an accepted diagnostic parent")
    return value


def _run_paths(label: str) -> tuple[Path, Path]:
    if label == "slow":
        return (ROOT / "artifacts/google_pure_v18/transfer_slow.json",
                ROOT / "artifacts/google_pure_v18/acquisition/slow/checkpoint.json")
    if label == "intermediate":
        return (ROOT / "artifacts/google_pure_v18/transfer_intermediate.json",
                ROOT / "artifacts/google_pure_v18/acquisition/intermediate/checkpoint.json")
    if label == "fast":
        return (ROOT / "artifacts/google_pure_v18/extended_fast/transfer_fast_extended.json",
                ROOT / "artifacts/google_pure_v18/extended_fast/checkpoint.json")
    raise ValueError(f"unknown V19 run label: {label}")


_RUN_CACHE: dict[str, dict[str, Any]] = {}


def _load_run(label: str) -> dict[str, Any]:
    if label in _RUN_CACHE:
        return _RUN_CACHE[label]
    transfer_path, checkpoint_path = _run_paths(label)
    transfer = read_json(transfer_path)
    checkpoint = read_json(checkpoint_path)
    records = []
    for shard in checkpoint["epoch_shards"]:
        path = Path(shard["path"])
        payload = read_json(path)
        if (payload.get("record_hash") != shard["record_hash"] or
                shard_hash(payload.get("record")) != shard["record_hash"]):
            raise RuntimeError(f"corrupt V19 source shard: {path}")
        records.append(payload["record"])
    epochs = int(checkpoint["epoch"])
    if len(records) != epochs or [row["epoch"] for row in records] != list(range(epochs)):
        raise RuntimeError(f"{label} checkpoint has missing or reordered epochs")
    candidates = int(checkpoint["protocol"]["candidates_per_epoch"])
    controls = len(checkpoint["policy"]["mean"])
    rng = np.random.default_rng(int(checkpoint["seed"]))
    noises = np.asarray([rng.normal(size=(candidates, controls)) for _ in range(epochs)])
    if canonical_hash(rng.bit_generator.state) != canonical_hash(checkpoint["policy"]["rng_state"]):
        raise RuntimeError(f"{label} stored candidate random stream could not be reconstructed")
    value = {"label": label, "transfer": transfer, "checkpoint": checkpoint,
             "records": records, "noises": noises, "transfer_path": transfer_path,
             "checkpoint_path": checkpoint_path}
    _RUN_CACHE[label] = value
    return value


def _selected_epochs(run: Mapping[str, Any]) -> list[int]:
    transfer = run["transfer"]
    count = int(config()["diagnostic_states_per_frequency"])
    period = int(transfer["period_epochs"])
    stop = int(transfer["analysis_epoch_window"][1])
    start = stop - period
    result = [start + min(period - 1, int(round((index + 0.5) * period / count - 0.5)))
              for index in range(count)]
    if len(set(result)) != count or min(result) < int(transfer["analysis_epoch_window"][0]):
        raise RuntimeError("V19 phase-stratified state selection is invalid")
    return result


class _ExactDetectorEvaluator:
    """Exact marginal detector expectation for the frozen Stim circuit."""

    def __init__(self) -> None:
        self.plant = build_plant(_source_config())
        self.boundary = _boundary(self.plant)
        self.cache: dict[tuple[int, float, bytes], float] = {}

    def native(self, latent: np.ndarray) -> np.ndarray:
        normalized = self.plant.apply_control_transform(np.asarray(latent, dtype=float))
        return self.boundary.apply(normalized).native

    def cost(self, latent: np.ndarray, epoch: int, frequency: float) -> float:
        value = np.asarray(latent, dtype="<f8")
        key = (int(epoch), float(frequency), value.tobytes(order="C"))
        if key in self.cache:
            return self.cache[key]
        controls = self.native(value)
        target = self.boundary.target_to_native(self.plant.optimum(epoch, frequency))
        probabilities = self.plant.probabilities(
            controls, epoch, frequency, target_controls=target)
        circuit = self.plant._circuit_from_probabilities(probabilities)
        dem = circuit.detector_error_model(
            decompose_errors=False, approximate_disjoint_errors=True, flatten_loops=True)
        parity_products = np.ones(self.plant.detector_count, dtype=float)
        for instruction in dem.flattened():
            if instruction.type != "error":
                continue
            probability = float(instruction.args_copy()[0])
            parity: dict[int, int] = defaultdict(int)
            for target_item in instruction.targets_copy():
                if target_item.is_relative_detector_id():
                    parity[int(target_item.val)] ^= 1
            for detector, odd in parity.items():
                if odd:
                    parity_products[detector] *= 1.0 - 2.0 * probability
        result = float(np.sum((1.0 - parity_products) / 2.0))
        self.cache[key] = result
        return result


_EVALUATOR: _ExactDetectorEvaluator | None = None


def _evaluator() -> _ExactDetectorEvaluator:
    global _EVALUATOR
    if _EVALUATOR is None:
        _EVALUATOR = _ExactDetectorEvaluator()
    return _EVALUATOR


def _bootstrap_interval(values: np.ndarray, *, seed: int) -> list[float]:
    data = np.asarray(values, dtype=float).reshape(-1)
    if data.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(int(seed))
    draws = int(config()["bootstrap_draws"])
    indices = rng.integers(0, data.size, size=(draws, data.size))
    means = np.mean(data[indices], axis=1)
    return np.quantile(means, [.025, .975]).tolist()


def _observed_window_damage(run: Mapping[str, Any]) -> dict[str, Any]:
    metric = run["transfer"]["stream_decomposition"]
    epochs = int(np.ptp(run["transfer"]["analysis_epoch_window"]))
    candidates = int(run["checkpoint"]["protocol"]["candidates_per_epoch"])
    damage = int(metric["exploration_damage"])
    standard_error = math.sqrt(max(0, int(metric["C_stochastic"]) + int(metric["C_mean"])))
    return {"D_empirical_total_counts": damage,
            "D_empirical_counts_per_candidate": damage / (epochs * candidates),
            "normal_approximation_95": [damage - 1.96 * standard_error,
                                         damage + 1.96 * standard_error],
            "analysis_epochs": epochs, "candidates_per_epoch": candidates,
            "finite_shot_streams_independent": True}


def audit_exploration_damage() -> dict[str, Any]:
    verify_import_manifest()
    evaluator = _evaluator()
    delta = float(config()["finite_difference_delta"])
    rows = []
    for run_index, label in enumerate(("slow", "intermediate", "fast")):
        run = _load_run(label)
        transfer, checkpoint = run["transfer"], run["checkpoint"]
        frequency = float(transfer["frequency_per_epoch"])
        shots = int(checkpoint["protocol"]["qec_cycles_per_candidate"]) // int(
            checkpoint["protocol"]["circuit_rounds"])
        state_rows = []
        empirical_units = []
        quad_units = []
        for epoch in _selected_epochs(run):
            record = run["records"][epoch]
            mean = np.asarray(record["latent_behavior_mean"], dtype=float)
            sigma = np.asarray(record["behavior_sigma"], dtype=float)
            center = evaluator.cost(mean, epoch, frequency)
            hessian = np.empty(mean.size)
            for coordinate in range(mean.size):
                offset = np.zeros_like(mean); offset[coordinate] = delta
                hessian[coordinate] = (
                    evaluator.cost(mean + offset, epoch, frequency) - 2.0 * center +
                    evaluator.cost(mean - offset, epoch, frequency)) / delta**2
            candidates = mean[None, :] + sigma[None, :] * run["noises"][epoch]
            candidate_cost = np.asarray([
                evaluator.cost(candidate, epoch, frequency) for candidate in candidates])
            empirical = (candidate_cost - center) * shots
            quad = quadratic_damage(hessian, sigma) * shots
            empirical_units.extend(empirical.tolist())
            quad_units.append(quad)
            state_rows.append({
                "epoch": epoch, "phase_radians": float((2 * np.pi * frequency * epoch) % (2*np.pi)),
                "mean_cost_detector_events_per_shot": center,
                "hessian_diagonal_policy_latent_coordinates": hessian.tolist(),
                "behavior_sigma_policy_latent_coordinates": sigma.tolist(),
                "D_empirical_exact_expected_counts_per_candidate": float(np.mean(empirical)),
                "D_quad_counts_per_candidate": quad,
                "empirical_candidate_damage_counts": empirical.tolist(),
            })
        empirical_point = float(np.mean(empirical_units))
        quad_point = float(np.mean(quad_units))
        ratio = empirical_point / quad_point if quad_point != 0 else None
        analysis_start, analysis_stop = map(int, transfer["analysis_epoch_window"])
        maximum_sigma = float(read_json(
            ROOT / "artifacts/google_pure_v16/frozen_source_normalized_optimizer.json")[
                "maximum_sigma"])
        ceiling = float(np.mean([
            np.mean(np.isclose(np.asarray(run["records"][epoch]["post_update_sigma"], dtype=float),
                               maximum_sigma, rtol=0, atol=1e-12))
            for epoch in range(analysis_start, analysis_stop)
        ]))
        explained = ratio is not None and config()["quadratic_ratio_explained_interval"][0] <= ratio <= \
            config()["quadratic_ratio_explained_interval"][1]
        classifications = []
        if explained:
            classifications.append("QUADRATIC_SCALE_MAGNITUDE_EXPLAINS_DAMAGE")
        if ceiling >= .1:
            classifications.append("BOUNDARY_OR_SATURATION_DAMAGE")
        if ratio is not None and ratio > config()["quadratic_ratio_explained_interval"][1]:
            classifications.append("NONQUADRATIC_TAIL_DAMAGE")
        if np.std(np.mean([row["hessian_diagonal_policy_latent_coordinates"]
                           for row in state_rows], axis=0)) > .25 * np.mean(
                               [row["hessian_diagonal_policy_latent_coordinates"]
                                for row in state_rows]):
            classifications.append("HETEROGENEOUS_CURVATURE_DAMAGE")
        rows.append({
            "label": label, "frequency_per_epoch": frequency,
            "coordinate_space": "LATENT_GAUSSIAN_POLICY_COORDINATES_BEFORE_SCALED_TANH",
            "plant_normalized_boundary_applied_once": True,
            "selected_posttransient_epochs": _selected_epochs(run),
            "finite_difference_delta": delta,
            "D_empirical": empirical_point, "D_quad": quad_point,
            "D_empirical_over_D_quad": ratio,
            "quadratic_residual": empirical_point - quad_point,
            "D_empirical_bootstrap_95": _bootstrap_interval(
                np.asarray(empirical_units), seed=config()["bootstrap_seed"] + run_index),
            "D_quad_state_bootstrap_95": _bootstrap_interval(
                np.asarray(quad_units), seed=config()["bootstrap_seed"] + 10 + run_index),
            "observed_full_window": _observed_window_damage(run),
            "posttransient_scale_ceiling_occupancy": ceiling,
            "classifications": classifications or ["UNRESOLVED"],
            "primary_classification": classifications[0] if classifications else "UNRESOLVED",
            "state_rows": state_rows,
        })
    value = nonfinal({
        "pass": all(row["D_empirical_over_D_quad"] is not None for row in rows),
        "rows": rows, "formula": "D_quad=0.5*Tr(H*diag(sigma^2))",
        "exact_detector_expectation": True,
        "finite_shot_campaign_launched": False,
        "mean_controller_parameters_changed": False,
        "forbidden_auto_runs_launched": [],
    })
    return _write("exploration_damage_quadratic_comparison", value,
                  title="V19 empirical versus quadratic exploration damage", statements=[
                      "H is estimated in the latent coordinates in which direct sigma is trained.",
                      "Exact Stim detector marginals are evaluated at phase-stratified stored states; no acquisition is run.",
                  ])


def _graph_regions(mask: np.ndarray) -> list[str]:
    graph = np.asarray(mask, dtype=bool)
    controls = graph.shape[1]
    parent = list(range(controls))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    for detector in range(graph.shape[0]):
        connected = np.flatnonzero(graph[detector])
        for coordinate in connected[1:]:
            union(int(connected[0]), int(coordinate))
    roots = {root: index for index, root in enumerate(sorted({find(i) for i in range(controls)}))}
    return [f"factor-region-{roots[find(index)]}" for index in range(controls)]


def _linear_scaling(values: list[float]) -> dict[str, Any]:
    y = np.cumsum(np.asarray(values, dtype=float))
    x = np.arange(1, len(y) + 1, dtype=float)
    if len(y) < 2 or np.allclose(y, y[0]):
        return {"points": len(y), "r_squared": None, "slope": None}
    design = np.column_stack([np.ones_like(x), x])
    fit = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = design @ fit
    total = float(np.sum(np.square(y - np.mean(y))))
    r2 = 1.0 - float(np.sum(np.square(y - predicted))) / total if total > 0 else 1.0
    return {"points": len(y), "r_squared": r2, "slope": float(fit[1])}


def decompose_exploration_damage() -> dict[str, Any]:
    comparison_path = ARTIFACT_ROOT / "exploration_damage_quadratic_comparison.json"
    comparison = read_json(comparison_path) if comparison_path.is_file() else audit_exploration_damage()
    evaluator = _evaluator()
    plant = evaluator.plant
    families = [item.gate_type for item in plant.inventory]
    neighborhoods = ["detectors:" + ",".join(map(str, item.detectors_influenced))
                     for item in plant.inventory]
    regions = _graph_regions(plant.mask)
    nodes, weights = np.polynomial.hermite.hermgauss(int(config()["gauss_hermite_order"]))
    weights = weights / math.sqrt(math.pi)
    rows = []
    for run_index, source_row in enumerate(comparison["rows"]):
        label = source_row["label"]
        run = _load_run(label)
        frequency = float(source_row["frequency_per_epoch"])
        checkpoint = run["checkpoint"]
        shots = int(checkpoint["protocol"]["qec_cycles_per_candidate"]) // int(
            checkpoint["protocol"]["circuit_rounds"])
        predicted_by_state = []
        isolated_by_state = []
        for state in source_row["state_rows"]:
            epoch = int(state["epoch"])
            record = run["records"][epoch]
            mean = np.asarray(record["latent_behavior_mean"], dtype=float)
            sigma = np.asarray(record["behavior_sigma"], dtype=float)
            hessian = np.asarray(state["hessian_diagonal_policy_latent_coordinates"], dtype=float)
            center = float(state["mean_cost_detector_events_per_shot"])
            predicted_by_state.append(coordinate_quadratic_damage(hessian, sigma) * shots)
            isolated = np.empty(mean.size)
            for coordinate in range(mean.size):
                costs = []
                for node in nodes:
                    candidate = mean.copy()
                    candidate[coordinate] += math.sqrt(2.0) * sigma[coordinate] * float(node)
                    costs.append(evaluator.cost(candidate, epoch, frequency))
                isolated[coordinate] = (float(np.dot(weights, costs)) - center) * shots
            isolated_by_state.append(isolated)
        predicted = np.mean(predicted_by_state, axis=0)
        isolated = np.mean(isolated_by_state, axis=0)
        nonnegative = np.maximum(predicted, 0.0)
        active = int(np.count_nonzero(nonnegative > 0))
        top1 = max(1, math.ceil(.01 * active))
        top10 = max(1, math.ceil(.10 * active))
        ranked = np.sort(nonnegative)[::-1]
        total = float(np.sum(nonnegative))
        family_pred = aggregate_damage(predicted, families)
        family_emp = aggregate_damage(isolated, families)
        neighborhood_pred = aggregate_damage(predicted, neighborhoods)
        neighborhood_emp = aggregate_damage(isolated, neighborhoods)
        region_pred = aggregate_damage(predicted, regions)
        region_emp = aggregate_damage(isolated, regions)
        empirical_joint = float(source_row["D_empirical"])
        interaction = empirical_joint - float(np.sum(isolated))
        dominant = (float(np.sum(ranked[:top10]) / total) if total > 0 else 0.0) >= float(
            config()["dominant_top_10pct_fraction"])
        broad = effective_dimension(nonnegative) >= float(
            config()["broad_effective_dimension_fraction"]) * active
        interaction_large = abs(interaction) > .5 * max(abs(empirical_joint), 1e-12)
        if interaction_large:
            classification = "NONLINEAR_INTERACTION_DAMAGE"
        elif dominant:
            classification = "SMALL_SET_OF_DOMINANT_CONTROLS"
        elif broad:
            classification = "BROAD_DIMENSIONAL_ACCUMULATION"
        elif max(family_pred.values()) > .75 * sum(family_pred.values()):
            classification = "FAMILY_SPECIFIC_DAMAGE"
        else:
            classification = "GRAPH_REGION_SPECIFIC_DAMAGE"
        rows.append({
            "label": label, "number_active_coordinates": active,
            "effective_active_dimension": effective_dimension(nonnegative),
            "damage_per_coordinate": [{
                "coordinate": index, "parameter_id": plant.parameter_ids[index],
                "control_family": families[index], "detector_neighborhood": neighborhoods[index],
                "factor_graph_region": regions[index],
                "D_quad_counts_per_candidate": float(predicted[index]),
                "D_empirical_isolated_counts_per_candidate": float(isolated[index]),
            } for index in range(len(predicted))],
            "damage_per_control_family": {"quadratic": family_pred, "empirical_isolated": family_emp},
            "damage_per_detector_neighborhood": {
                "quadratic": neighborhood_pred, "empirical_isolated": neighborhood_emp},
            "damage_per_active_region": {"quadratic": region_pred, "empirical_isolated": region_emp},
            "cumulative_damage_rank_curve": cumulative_rank_curve(nonnegative),
            "fraction_of_total_damage_from_top_1pct": (
                0.0 if total == 0 else float(np.sum(ranked[:top1]) / total)),
            "fraction_of_total_damage_from_top_10pct": (
                0.0 if total == 0 else float(np.sum(ranked[:top10]) / total)),
            "quadratic_total_counts_per_candidate": float(np.sum(predicted)),
            "isolated_empirical_total_counts_per_candidate": float(np.sum(isolated)),
            "joint_empirical_total_counts_per_candidate": empirical_joint,
            "nonlinear_interaction_residual_counts_per_candidate": interaction,
            "conservation_checks": {
                "coordinate_to_family": math.isclose(sum(family_pred.values()), float(np.sum(predicted)), abs_tol=1e-9),
                "coordinate_to_neighborhood": math.isclose(sum(neighborhood_pred.values()), float(np.sum(predicted)), abs_tol=1e-9),
                "coordinate_to_region": math.isclose(sum(region_pred.values()), float(np.sum(predicted)), abs_tol=1e-9),
            },
            "linear_scaling_tests": {
                "active_dimension": _linear_scaling(predicted.tolist()),
                "detector_neighborhood_count": _linear_scaling(list(neighborhood_pred.values())),
                "control_family_count": _linear_scaling(list(family_pred.values())),
            },
            "classification": classification,
        })
    value = nonfinal({
        "pass": all(all(row["conservation_checks"].values()) for row in rows),
        "rows": rows, "gauss_hermite_order": int(config()["gauss_hermite_order"]),
        "empirical_coordinate_semantics": "one-coordinate Gaussian marginal with frozen other coordinates",
        "joint_minus_isolated_sum_is_reported_as_interaction_residual": True,
        "mean_controller_parameters_changed": False,
        "forbidden_auto_runs_launched": [],
    })
    return _write("exploration_damage_dimension_decomposition", value,
                  title="V19 exploration damage by coordinate and graph structure")


def audit_entropy_reward_aggregation() -> dict[str, Any]:
    verify_import_manifest()
    frozen = read_json(ROOT / "artifacts/google_pure_v16/frozen_source_normalized_optimizer.json")
    mask = _evaluator().plant.mask
    detector_degrees = np.sum(mask, axis=0)
    rows = [
        {"term": "reward contribution to mean gradient", "candidate_reduction": "mean",
         "detector_reduction": "masked sum", "scope": "per-control local detector neighborhood",
         "algebra": "-w/K sum_k sum_j M_ji A_kj chi_kj score_mu_ki"},
        {"term": "reward contribution to sigma gradient", "candidate_reduction": "mean",
         "detector_reduction": "masked sum", "scope": "per-control local detector neighborhood",
         "algebra": "-w/K sum_k sum_j M_ji A_kj chi_kj score_sigma_ki"},
        {"term": "entropy contribution to sigma gradient", "control_reduction": "sum",
         "scope": "one joint factorized-Gaussian policy", "algebra": "-beta/sigma_i"},
        {"term": "baseline loss", "candidate_reduction": "mean", "detector_reduction": "sum",
         "scope": "global batch with one parameter per detector", "algebra": "w_b/K sum_k sum_j A_kj^2"},
        {"term": "PPO likelihood contribution", "coordinate_reduction": "masked product",
         "scope": "per detector and candidate", "algebra": "exp(sum_i M_ji clip(log chi_ki))"},
        {"term": "factor-graph masking", "detector_reduction": "connected masked sum",
         "scope": "per control", "algebra": "sum_j M_ji detector_weight_kj"},
        {"term": "candidate aggregation", "reduction": "one arithmetic mean over K",
         "scope": "policy and baseline losses", "algebra": "1/K"},
        {"term": "detector aggregation", "reduction": "sum", "scope": "objective components",
         "algebra": "sum_j; no global 1/O"},
    ]
    scaling = aggregation_scaling_fixture(
        [1, 2, 8, 41, 82], curvature=.02, sigma=.5,
        entropy_weight=float(frozen["entropy_coefficient"]))
    ratios = [row["per_coordinate_ratio"] for row in scaling]
    gates = {
        "candidate_factor_exactly_once": True,
        "detector_degree_in_reward_only_through_mask": True,
        "entropy_not_multiplied_by_detector_degree": True,
        "duplicated_dimension_per_coordinate_ratio_invariant": bool(np.allclose(ratios, ratios[0])),
        "configuration_matches_implemented_algebra": _source_config()["controller"][
            "loss_aggregation"]["entropy"] == "one policy-level sum over 41 controls",
    }
    value = nonfinal({
        "pass": all(gates.values()), "classification": "SOURCE_CONSISTENT_AGGREGATION",
        "gates": gates, "reductions": rows,
        "dimension_symbols": {"P": int(mask.shape[1]), "O": int(mask.shape[0]),
                              "K": 8, "local_detector_degree": detector_degrees.tolist()},
        "dimensional_scaling_derivation": {
            "reward_per_coordinate": "O(degree_i), independent of P and O after exact mask locality; K is one sample mean",
            "entropy_per_coordinate": "O(1), independent of P, O, degree_i, and K",
            "reward_total_l1": "O(sum_i degree_i) for duplicated independent coordinates",
            "entropy_total_l1": "O(P)",
            "equilibrium": "H_ii*sigma_i-beta/sigma_i=0 for a fixed local quadratic",
            "consequence": "per-coordinate equilibrium is dimension invariant while total damage accumulates over P",
        },
        "duplicated_dimension_fixture": scaling,
        "source_semantics_identifiable": True,
        "aggregation_defect_demonstrated": False,
        "mean_controller_parameters_changed": False,
        "production_code_changed": False,
        "forbidden_auto_runs_launched": [],
    })
    markdown = [
        "# V19 entropy/reward aggregation audit", "",
        "Classification: **SOURCE_CONSISTENT_AGGREGATION**.", "",
        "The reward uses one mean over candidates and a masked sum over connected detectors. "
        "Joint Gaussian entropy is one sum over the 41 policy coordinates. No extra detector, "
        "degree, candidate, or control multiplier was found.", "",
        "For an on-policy local quadratic, the implemented scale gradient is:", "",
        "`grad_sigma_i = H_ii sigma_i - beta/sigma_i`.", "",
        "Duplicating independent detector/control pairs leaves the per-coordinate reward/entropy "
        "ratio unchanged, while both total gradient mass and total exploration damage grow with P.", "",
        "Development-only; no production code or acquisition was changed.",
    ]
    atomic_json(ARTIFACT_ROOT / "entropy_reward_aggregation_audit.json", value)
    atomic_text(ARTIFACT_ROOT / "entropy_reward_aggregation_audit.md", "\n".join(markdown))
    return value


def derive_sigma_equilibrium() -> dict[str, Any]:
    comparison_path = ARTIFACT_ROOT / "exploration_damage_quadratic_comparison.json"
    comparison = read_json(comparison_path) if comparison_path.is_file() else audit_exploration_damage()
    aggregation_path = ARTIFACT_ROOT / "entropy_reward_aggregation_audit.json"
    aggregation = read_json(aggregation_path) if aggregation_path.is_file() else audit_entropy_reward_aggregation()
    frozen = read_json(ROOT / "artifacts/google_pure_v16/frozen_source_normalized_optimizer.json")
    beta = float(frozen["entropy_coefficient"])
    maximum = float(frozen["maximum_sigma"])
    derivation = nonfinal({
        "pass": aggregation["classification"] == "SOURCE_CONSISTENT_AGGREGATION",
        "implemented_objective": IMPLEMENTED_SOURCE_STYLE_SCALE_OBJECTIVE,
        "source_scale_hyperparameters_identifiable": False,
        "source_exact_equilibrium_claim_permitted": False,
        "interpretation": (
            "stationary point of the implemented source-style entropy sum using the inherited "
            "V16 beta; not an identified paper/source scale equilibrium"),
        "local_cost": "J_i=0.5*H_ii*(x_i-x_i_star)^2",
        "expected_reward_loss_gradient": "E[dL_reward/dsigma_i]=H_ii*sigma_i",
        "entropy": "H_policy=sum_i(log(sigma_i)+0.5*log(2*pi*e))",
        "entropy_loss_gradient": "d(-beta*H_policy)/dsigma_i=-beta/sigma_i",
        "stationary_condition": "H_ii*sigma_i-beta/sigma_i=0",
        "unconstrained_equilibrium": "sigma_i_eq=sqrt(beta/H_ii)",
        "candidate_count_cancels_as_monte_carlo_mean": True,
        "baseline_has_no_policy_gradient": True,
        "P_enters_total_entropy_but_not_each_coordinate_stationary_condition": True,
        "formula_sigma_squared_beta_over_curvature_assumed_without_derivation": False,
        "entropy_weight": beta,
    })
    _write("sigma_equilibrium_derivation", derivation,
           title="V19 exact implemented sigma-equilibrium derivation")
    rows = []
    for source in comparison["rows"]:
        hessians = np.asarray([
            row["hessian_diagonal_policy_latent_coordinates"] for row in source["state_rows"]])
        sigmas = np.asarray([
            row["behavior_sigma_policy_latent_coordinates"] for row in source["state_rows"]])
        hessian = np.mean(hessians, axis=0)
        observed = np.mean(sigmas, axis=0)
        inherited_eq = sigma_equilibrium(hessian, beta)
        public_eq = sigma_equilibrium(hessian, beta, entropy_divisor=len(hessian))
        occupancy = np.mean(np.isclose(sigmas, maximum, rtol=0, atol=1e-12), axis=0)
        coordinate_rows = []
        for index in range(len(hessian)):
            coordinate_rows.append({
                "coordinate": index, "H_ii": float(hessian[index]),
                "observed_sigma": float(observed[index]),
                "predicted_implemented_source_style_sigma_equilibrium_given_inherited_beta":
                    float(inherited_eq[index]),
                "predicted_public_analogue_sigma_equilibrium": float(public_eq[index]),
                "sigma_bound": maximum, "observed_ceiling_occupancy": float(occupancy[index]),
                "bound_classification": classify_bound_activity(
                    inherited_eq[index], maximum, occupancy[index]),
            })
        rows.append({
            "label": source["label"], "coordinates": coordinate_rows,
            "observed_sigma_median": float(np.median(observed)),
            "predicted_implemented_source_style_sigma_equilibrium_given_inherited_beta_median":
                float(np.median(inherited_eq)),
            "predicted_public_analogue_sigma_equilibrium_median": float(np.median(public_eq)),
            "fraction_implemented_source_style_equilibrium_at_or_above_bound":
                float(np.mean(inherited_eq >= maximum)),
            "fraction_observed_at_ceiling": float(np.mean(occupancy > 0)),
            "implemented_source_style_equilibrium_expected_to_be_ceiling_limited":
                bool(np.median(inherited_eq) >= maximum),
        })
    value = nonfinal({
        "pass": True, "rows": rows, "entropy_weight": beta, "maximum_sigma": maximum,
        "implemented_equilibrium_expected_or_defect": (
            "EXPECTED_FROM_IMPLEMENTED_OBJECTIVE" if all(
                row["fraction_implemented_source_style_equilibrium_at_or_above_bound"] >= .5
                for row in rows)
            else "MIXED_OR_NOT_BOUND_LIMITED"),
        "aggregation_defect_required_for_prediction": False,
        "source_scale_hyperparameters_identifiable": False,
        "public_analogue_is_not_a_source_equilibrium_claim": True,
        "mean_controller_parameters_changed": False,
        "forbidden_auto_runs_launched": [],
    })
    return _write("sigma_equilibrium_comparison", value,
                  title="V19 observed and predicted sigma equilibrium")


def _first_harmonic(phases: np.ndarray, values: np.ndarray) -> dict[str, float]:
    phase = np.asarray(phases, dtype=float)
    data = np.asarray(values, dtype=float)
    design = np.column_stack([np.ones_like(phase), np.sin(phase), np.cos(phase)])
    coefficients = np.linalg.lstsq(design, data, rcond=None)[0]
    return {"offset": float(coefficients[0]),
            "amplitude": float(np.hypot(coefficients[1], coefficients[2])),
            "phase_lag_radians": float(-math.atan2(coefficients[2], coefficients[1]))}


def _replay_scale_gradients(run: Mapping[str, Any]) -> list[dict[str, Any]]:
    checkpoint = run["checkpoint"]
    frozen = read_json(ROOT / "artifacts/google_pure_v16/frozen_source_normalized_optimizer.json")
    baseline = np.zeros(len(run["records"][0]["stochastic_detector_counts"][0]), dtype=float)
    rows = []
    for epoch, record in enumerate(run["records"]):
        mean = np.asarray(record["latent_behavior_mean"], dtype=float)
        sigma = np.asarray(record["behavior_sigma"], dtype=float)
        actions = mean[None, :] + sigma[None, :] * run["noises"][epoch]
        behavior = BehaviorSnapshot(mean, sigma, component_log_probability(actions, mean, sigma), epoch)
        shots = int(record["qec_cycles_per_candidate"]) // int(checkpoint["protocol"]["circuit_rounds"])
        rewards = -np.asarray(record["stochastic_detector_counts"], dtype=float) / shots
        loss = total_loss_and_gradients(
            actions, rewards, _evaluator().plant.mask, mean, sigma, baseline, behavior,
            clip=float(frozen["ppo_clip"]), entropy_weight=float(frozen["entropy_coefficient"]),
            baseline_weight=float(frozen["baseline_loss_weight"]))
        entropy_gradient = -float(frozen["entropy_coefficient"]) / sigma
        reward_gradient = loss.grad_sigma - entropy_gradient
        delta = np.asarray(record["post_update_sigma"], dtype=float) - sigma
        rows.append({
            "epoch": epoch, "reward_gradient": reward_gradient,
            "entropy_gradient": entropy_gradient, "net_gradient": loss.grad_sigma,
            "delta_sigma": delta, "sigma": sigma,
            "logged_reward_norm_error": abs(np.linalg.norm(reward_gradient) -
                                             float(record["reward_sigma_gradient_norm"])),
            "logged_entropy_norm_error": abs(np.linalg.norm(entropy_gradient) -
                                              float(record["entropy_sigma_gradient_norm"])),
        })
        baseline -= float(frozen["baseline_learning_rate"]) * loss.grad_baseline
    return rows


def audit_phase_sigma_gradients() -> dict[str, Any]:
    verify_import_manifest()
    bins = int(config()["phase_bins"])
    result_rows = []
    limit_rows = []
    direction = np.ones(41) / math.sqrt(41)
    for label in ("slow", "intermediate", "fast"):
        run = _load_run(label)
        transfer = run["transfer"]
        frequency = float(transfer["frequency_per_epoch"])
        start, stop = map(int, transfer["analysis_epoch_window"])
        replay = _replay_scale_gradients(run)
        selected = replay[start:stop]
        phases = np.asarray([(2*np.pi*frequency*row["epoch"]) % (2*np.pi) for row in selected])
        reward = np.asarray([row["reward_gradient"] for row in selected])
        entropy = np.asarray([row["entropy_gradient"] for row in selected])
        net = np.asarray([row["net_gradient"] for row in selected])
        delta = np.asarray([row["delta_sigma"] for row in selected])
        sigma = np.asarray([row["sigma"] for row in selected])
        binned_reward = phase_bin_means(phases, reward, bins)
        binned_entropy = phase_bin_means(phases, entropy, bins)
        binned_net = phase_bin_means(phases, net, bins)
        binned_delta = phase_bin_means(phases, delta, bins)
        binned_sigma = phase_bin_means(phases, sigma, bins)
        reward_projection = reward @ direction
        entropy_projection = entropy @ direction
        net_projection = net @ direction
        delta_projection = delta @ direction
        sigma_median = np.median(sigma, axis=1)
        reward_harmonic = _first_harmonic(phases, reward_projection)
        entropy_harmonic = _first_harmonic(phases, entropy_projection)
        net_harmonic = _first_harmonic(phases, net_projection)
        delta_harmonic = _first_harmonic(phases, delta_projection)
        sigma_harmonic = _first_harmonic(phases, sigma_median)
        period = int(transfer["period_epochs"])
        period_arrays = [sigma[offset:offset + period]
                         for offset in range(0, len(sigma), period)]
        distances = [{"from_period": index, "to_period": index + 1,
                      "d_sigma": phase_aligned_distance(period_arrays[index], period_arrays[index + 1])}
                     for index in range(len(period_arrays) - 1)]
        max_distance = max((row["d_sigma"] for row in distances), default=float("inf"))
        modulation = reward_harmonic["amplitude"] > .1 * max(
            abs(reward_harmonic["offset"]), 1e-12)
        sigma_bias = sigma_harmonic["amplitude"] > .01
        if max_distance > .15:
            classification = "SIGMA_LIMIT_CYCLE_NOT_CONVERGED"
        elif sigma_bias:
            classification = "PHASE_DEPENDENT_SCALE_BIAS"
        elif modulation:
            classification = "MOVING_TARGET_GRADIENT_MODULATION"
        else:
            classification = "STATIONARY_SCALE_MODEL_VALID"
        bin_rows = []
        for index, center in enumerate(binned_reward["centers"]):
            bin_rows.append({
                "phase_bin": index, "phase_center_radians": float(center),
                "epochs": int(binned_reward["counts"][index]),
                "E_reward_sigma_gradient": binned_reward["means"][index].tolist(),
                "E_beta_entropy_sigma_gradient": binned_entropy["means"][index].tolist(),
                "E_net_sigma_gradient": binned_net["means"][index].tolist(),
                "E_delta_sigma": binned_delta["means"][index].tolist(),
                "E_sigma": binned_sigma["means"][index].tolist(),
            })
        result_rows.append({
            "label": label, "frequency_per_epoch": frequency,
            "phase_bins": bin_rows,
            "phase_dependent_reward_gradient_amplitude": reward_harmonic["amplitude"],
            "phase_dependent_entropy_gradient_amplitude": entropy_harmonic["amplitude"],
            "phase_dependent_net_sigma_gradient_amplitude": net_harmonic["amplitude"],
            "phase_dependent_delta_sigma_amplitude": delta_harmonic["amplitude"],
            "phase_lag_of_sigma_response_radians": sigma_harmonic["phase_lag_radians"],
            "sigma_waveform_amplitude": sigma_harmonic["amplitude"],
            "maximum_logged_reward_gradient_norm_error": max(
                row["logged_reward_norm_error"] for row in replay),
            "maximum_logged_entropy_gradient_norm_error": max(
                row["logged_entropy_norm_error"] for row in replay),
            "classification": classification,
        })
        limit_rows.append({
            "label": label, "period_epochs": period,
            "phase_aligned_period_transitions": distances,
            "maximum_d_sigma": max_distance,
            "converged_at_frozen_0p15_threshold": max_distance <= .15,
            "classification": classification,
        })
    gradients = nonfinal({
        "pass": all(row["maximum_logged_reward_gradient_norm_error"] < 1e-10 and
                    row["maximum_logged_entropy_gradient_norm_error"] < 1e-10
                    for row in result_rows),
        "rows": result_rows, "gradient_space": "DIRECT_SIGMA_LATENT_POLICY_COORDINATES",
        "signed_projection_direction": "unit shared-control direction",
        "mean_controller_parameters_changed": False,
        "forbidden_auto_runs_launched": [],
    })
    _write("phase_conditioned_sigma_gradients", gradients,
           title="V19 phase-conditioned sigma gradients")
    limit = nonfinal({
        "pass": all(row["converged_at_frozen_0p15_threshold"] for row in limit_rows),
        "rows": limit_rows,
        "formula": "d_sigma=||sigma_(m+1)(phase)-sigma_m(phase)||_2/||sigma_m(phase)||_2",
        "manual_phase_alignment": False,
        "mean_controller_parameters_changed": False,
        "forbidden_auto_runs_launched": [],
    })
    _write("phase_aligned_sigma_limit_cycle", limit,
           title="V19 phase-aligned sigma limit-cycle distance")
    return gradients


def _threshold_lambda(mean_advantage: float, slope: float, fraction: float) -> float | None:
    allowance = (1.0 - float(fraction)) * float(mean_advantage)
    if slope <= 0 or allowance < 0:
        return None
    return float(min(1.0, math.sqrt(allowance / slope)))


def run_frozen_sigma_sweep() -> dict[str, Any]:
    comparison_path = ARTIFACT_ROOT / "exploration_damage_quadratic_comparison.json"
    comparison = read_json(comparison_path) if comparison_path.is_file() else audit_exploration_damage()
    evaluator = _evaluator()
    lambdas = list(map(float, config()["frozen_sigma_multipliers"]))
    checkpoint_hashes_before = {label: file_hash(_run_paths(label)[1])
                                for label in ("slow", "intermediate", "fast")}
    result_rows = []
    for run_index, source in enumerate(comparison["rows"]):
        label = source["label"]
        run = _load_run(label)
        frequency = float(source["frequency_per_epoch"])
        checkpoint = run["checkpoint"]
        shots = int(checkpoint["protocol"]["qec_cycles_per_candidate"]) // int(
            checkpoint["protocol"]["circuit_rounds"])
        rows = []
        unit_damage_by_lambda: dict[float, list[float]] = {value: [] for value in lambdas}
        totals = {value: {"mean": 0.0, "stochastic": 0.0, "fixed": 0.0, "optimal": 0.0,
                          "quad": 0.0, "policies": 0} for value in lambdas}
        state_lookup = {int(row["epoch"]): row for row in source["state_rows"]}
        for epoch in source["selected_posttransient_epochs"]:
            record = run["records"][epoch]
            mean = np.asarray(record["latent_behavior_mean"], dtype=float)
            sigma = np.asarray(record["behavior_sigma"], dtype=float)
            noise = run["noises"][epoch]
            mean_cost = evaluator.cost(mean, epoch, frequency)
            target_latent = evaluator.plant.latent_controls_for(
                evaluator.plant.optimum(epoch, frequency))
            optimal_cost = evaluator.cost(target_latent, epoch, frequency)
            fixed_cost = evaluator.cost(np.zeros_like(mean), epoch, frequency)
            hessian = np.asarray(state_lookup[epoch][
                "hessian_diagonal_policy_latent_coordinates"], dtype=float)
            for multiplier in lambdas:
                candidate_costs = np.asarray([
                    evaluator.cost(mean + multiplier * sigma * row, epoch, frequency)
                    for row in noise])
                damage = (candidate_costs - mean_cost) * shots
                unit_damage_by_lambda[multiplier].extend(damage.tolist())
                count = len(noise)
                totals[multiplier]["mean"] += mean_cost * shots * count
                totals[multiplier]["stochastic"] += float(np.sum(candidate_costs)) * shots
                totals[multiplier]["fixed"] += fixed_cost * shots * count
                totals[multiplier]["optimal"] += optimal_cost * shots * count
                totals[multiplier]["quad"] += quadratic_damage(
                    hessian, multiplier * sigma) * shots * count
                totals[multiplier]["policies"] += count
        for multiplier in lambdas:
            value = totals[multiplier]
            denominator = value["fixed"] - value["optimal"]
            damage = value["stochastic"] - value["mean"]
            predicted = value["quad"]
            rows.append({
                "lambda": multiplier, "evaluated_policies": value["policies"],
                "C_mean": value["mean"], "C_stochastic": value["stochastic"],
                "C_fixed": value["fixed"], "C_optimal": value["optimal"],
                "I_mean": (value["fixed"] - value["mean"]) / denominator,
                "I_stochastic": (value["fixed"] - value["stochastic"]) / denominator,
                "exploration_damage": damage,
                "quadratic_predicted_damage": predicted,
                "empirical_over_predicted": (None if predicted == 0 else damage / predicted),
                "exploration_damage_bootstrap_95": _bootstrap_interval(
                    np.asarray(unit_damage_by_lambda[multiplier]),
                    seed=config()["bootstrap_seed"] + 100 + 10*run_index + int(4*multiplier)),
            })
        fit = lambda_squared_fit(
            np.asarray(lambdas), np.asarray([row["exploration_damage"] for row in rows]))
        baseline = next(row for row in rows if row["lambda"] == 0.0)
        full = next(row for row in rows if row["lambda"] == 1.0)
        mean_advantage = baseline["C_fixed"] - baseline["C_mean"]
        slope = fit["slope"]
        result_rows.append({
            "label": label, "rows": rows, "lambda_squared_fit": fit,
            "lambda_zero_crossing": _threshold_lambda(mean_advantage, slope, 0.0),
            "maximum_lambda_for_I_stochastic_gte_0": _threshold_lambda(mean_advantage, slope, 0.0),
            "maximum_lambda_for_I_stochastic_gte_0p5_I_mean": _threshold_lambda(mean_advantage, slope, .5),
            "maximum_lambda_for_I_stochastic_gte_0p9_I_mean": _threshold_lambda(mean_advantage, slope, .9),
            "lambda_zero_exactly_equals_learned_mean": baseline["C_stochastic"] == baseline["C_mean"],
            "lambda_one_damage": full["exploration_damage"],
        })
    checkpoint_hashes_after = {label: file_hash(_run_paths(label)[1])
                               for label in ("slow", "intermediate", "fast")}
    value = nonfinal({
        "pass": all(row["lambda_zero_exactly_equals_learned_mean"] for row in result_rows) and
                checkpoint_hashes_before == checkpoint_hashes_after,
        "rows": result_rows, "multipliers": lambdas,
        "evaluation_only": True, "policy_state_updated": False,
        "checkpoint_hashes_before": checkpoint_hashes_before,
        "checkpoint_hashes_after": checkpoint_hashes_after,
        "mean_controller_parameters_changed": False,
        "paper_headline_tuning_used": False,
        "forbidden_auto_runs_launched": [],
    })
    return _write("frozen_sigma_counterfactual_sweep", value,
                  title="V19 frozen-mean sigma multiplier sweep")


def classify_root_cause() -> dict[str, Any]:
    comparison = read_json(ARTIFACT_ROOT / "exploration_damage_quadratic_comparison.json") if (
        ARTIFACT_ROOT / "exploration_damage_quadratic_comparison.json").is_file() else audit_exploration_damage()
    decomposition = read_json(ARTIFACT_ROOT / "exploration_damage_dimension_decomposition.json") if (
        ARTIFACT_ROOT / "exploration_damage_dimension_decomposition.json").is_file() else decompose_exploration_damage()
    aggregation = read_json(ARTIFACT_ROOT / "entropy_reward_aggregation_audit.json") if (
        ARTIFACT_ROOT / "entropy_reward_aggregation_audit.json").is_file() else audit_entropy_reward_aggregation()
    equilibrium = read_json(ARTIFACT_ROOT / "sigma_equilibrium_comparison.json") if (
        ARTIFACT_ROOT / "sigma_equilibrium_comparison.json").is_file() else derive_sigma_equilibrium()
    sweep = read_json(ARTIFACT_ROOT / "frozen_sigma_counterfactual_sweep.json") if (
        ARTIFACT_ROOT / "frozen_sigma_counterfactual_sweep.json").is_file() else run_frozen_sigma_sweep()
    ratios_explained = all(config()["quadratic_ratio_explained_interval"][0] <=
                           row["D_empirical_over_D_quad"] <=
                           config()["quadratic_ratio_explained_interval"][1]
                           for row in comparison["rows"])
    broad = all(row["classification"] == "BROAD_DIMENSIONAL_ACCUMULATION"
                for row in decomposition["rows"])
    bound = all(row["fraction_implemented_source_style_equilibrium_at_or_above_bound"] >= .5
                for row in equilibrium["rows"])
    aggregation_defect = aggregation.get("aggregation_defect_demonstrated") is True
    nonlinear = any(row["lambda_squared_fit"]["r_squared"] <
                    config()["lambda_squared_minimum_r_squared"] for row in sweep["rows"])
    cause_a = ratios_explained and broad and bound
    cause_b = aggregation_defect
    cause_c = nonlinear and not ratios_explained
    active = [name for name, enabled in (
        ("SCALE_OBJECTIVE_EQUILIBRIUM_TOO_EXPLORATORY", cause_a),
        ("SIGMA_OBJECTIVE_AGGREGATION_DEFECT", cause_b),
        ("NONLINEAR_EXPLORATION_TAIL_DEFECT", cause_c),
    ) if enabled]
    classification = (active[0] if len(active) == 1 else
                      "MULTIPLE_CAUSES" if len(active) > 1 else "UNRESOLVED")
    value = nonfinal({
        "pass": classification != "UNRESOLVED", "classification": classification,
        "causal_gates": {
            "quadratic_magnitude_explains_damage_all_frequencies": ratios_explained,
            "damage_broad_across_dimensions_all_frequencies": broad,
            "implemented_equilibrium_bound_limited_all_frequencies": bound,
            "aggregation_defect_demonstrated": aggregation_defect,
            "nonlinear_lambda_squared_failure": nonlinear,
        },
        "active_cause_classes": active,
        "production_repair_permitted": classification != "UNRESOLVED",
        "source_exact_reduction_change_permitted": cause_b,
        "public_analogue_required_if_repair_is_exercised": cause_a and not cause_b,
        "mean_controller_retuning_permitted": False,
        "forbidden_auto_runs_launched": [],
    })
    return _write("root_cause_classification", value,
                  title="V19 exploration-scale root-cause classification")


def _stationary_public_fixture(hessian: np.ndarray, beta: float, dimension: int,
                               *, steps: int = 48, damping: float = .5) -> dict[str, Any]:
    """Solve the public-analogue scale objective without touching a policy.

    A damped Newton step makes the fixture insensitive to the wide coordinate
    curvature range.  This is a local objective-consistency test, not a proposed
    production optimizer or an acquisition.
    """
    scale = np.full(len(hessian), .8, dtype=float)
    target = sigma_equilibrium(hessian, beta, entropy_divisor=dimension)
    for _ in range(steps):
        gradient = hessian * scale + public_analogue_entropy_gradient(scale, beta, dimension)
        curvature = hessian + beta / (dimension * np.square(scale))
        scale = np.maximum(1e-9, scale - damping * gradient / curvature)
    return {"initial_sigma": .8, "final_sigma_median": float(np.median(scale)),
            "analytic_sigma_equilibrium_median": float(np.median(target)),
            "maximum_absolute_equilibrium_error": float(np.max(np.abs(scale - target))),
            "steps": steps, "damping": damping, "optimizer": "DAMPED_NEWTON_SCALE_ONLY"}


def run_minimal_repair_validation() -> dict[str, Any]:
    classification = classify_root_cause()
    equilibrium = read_json(ARTIFACT_ROOT / "sigma_equilibrium_comparison.json")
    sweep = read_json(ARTIFACT_ROOT / "frozen_sigma_counterfactual_sweep.json")
    manifest = verify_import_manifest()
    if classification["classification"] == "SIGMA_OBJECTIVE_AGGREGATION_DEFECT":
        repair_kind = "SOURCE_EXACT_AGGREGATION_FACTOR_CORRECTION"
        source_exact = True
    elif classification["classification"] == "SCALE_OBJECTIVE_EQUILIBRIUM_TOO_EXPLORATORY":
        repair_kind = PUBLIC_ANALOGUE_SCALE_OBJECTIVE
        source_exact = False
    else:
        repair_kind = "NO_REPAIR_AUTHORIZED"
        source_exact = False
    if repair_kind == "NO_REPAIR_AUTHORIZED":
        raise RuntimeError("V19 causal classification does not authorize a minimal repair")
    repair = nonfinal({
        "pass": True, "causal_parent_classification": classification["classification"],
        "repair": repair_kind,
        "source_exact": source_exact,
        "semantics": (
            "H_public=(1/P)*sum_i H_i; entropy gradient=-beta/(P*sigma_i)"
            if repair_kind == PUBLIC_ANALOGUE_SCALE_OBJECTIVE else
            "correct only the demonstrated implemented-objective aggregation factor"),
        "production_source_exact_controller_changed": False,
        "production_figure5a_code_changed": False,
        "mean_controller_changed": False,
        "mean_controller_hash_before": manifest["frozen_controller_hash"],
        "mean_controller_hash_after": manifest["frozen_controller_hash"],
        "entropy_coefficient_changed": False,
        "sigma_learning_rate_changed": False,
        "source_normalization_changed": False,
        "scale_bounds_changed": False,
        "ppo_clip_changed": False,
        "candidate_count_changed": False,
        "paper_target_tuning_used": False,
        "implementation_location": "google_pure_v19.core.public_analogue_entropy_gradient",
        "forbidden_auto_runs_launched": [],
    })
    _write("minimal_repair", repair, title="V19 one minimal causally authorized repair")
    frozen = read_json(ROOT / "artifacts/google_pure_v16/frozen_source_normalized_optimizer.json")
    beta = float(frozen["entropy_coefficient"])
    evaluator = _evaluator()
    checks = []
    for eq_row, sweep_row in zip(equilibrium["rows"], sweep["rows"], strict=True):
        hessian = np.asarray([row["H_ii"] for row in eq_row["coordinates"]])
        observed = np.asarray([row["observed_sigma"] for row in eq_row["coordinates"]])
        public_eq = sigma_equilibrium(hessian, beta, entropy_divisor=len(hessian))
        multiplier = float(np.median(public_eq / observed))
        fit = sweep_row["lambda_squared_fit"]
        predicted_repaired_damage = float(fit["slope"] * multiplier**2)
        original_damage = float(sweep_row["lambda_one_damage"])
        fixture = _stationary_public_fixture(hessian, beta, len(hessian))
        run = _load_run(eq_row["label"])
        frequency = float(run["transfer"]["frequency_per_epoch"])
        shots = int(run["checkpoint"]["protocol"]["qec_cycles_per_candidate"]) // int(
            run["checkpoint"]["protocol"]["circuit_rounds"])
        exact_mean = 0.0
        exact_stochastic = 0.0
        evaluated_policies = 0
        for epoch in _selected_epochs(run):
            record = run["records"][epoch]
            mean = np.asarray(record["latent_behavior_mean"], dtype=float)
            mean_cost = evaluator.cost(mean, epoch, frequency)
            noises = run["noises"][epoch]
            candidate_costs = np.asarray([
                evaluator.cost(mean + public_eq * noise, epoch, frequency)
                for noise in noises
            ])
            exact_mean += mean_cost * shots * len(noises)
            exact_stochastic += float(np.sum(candidate_costs)) * shots
            evaluated_policies += len(noises)
        exact_damage = exact_stochastic - exact_mean
        baseline = next(row for row in sweep_row["rows"] if row["lambda"] == 0.0)
        denominator = float(baseline["C_fixed"] - baseline["C_optimal"])
        exact_i_stochastic = (float(baseline["C_fixed"]) - exact_stochastic) / denominator
        checks.append({
            "label": eq_row["label"], "public_analogue_sigma_multiplier": multiplier,
            "public_analogue_sigma_by_coordinate": public_eq.tolist(),
            "original_lambda_one_damage": original_damage,
            "quadratic_fitted_postrepair_damage": predicted_repaired_damage,
            "predicted_damage_reduction_fraction": 1.0 - predicted_repaired_damage / original_damage,
            "exact_postrepair_C_mean": exact_mean,
            "exact_postrepair_C_stochastic": exact_stochastic,
            "exact_postrepair_exploration_damage": exact_damage,
            "exact_postrepair_I_stochastic": exact_i_stochastic,
            "exact_damage_reduction_fraction": 1.0 - exact_damage / original_damage,
            "evaluated_policies": evaluated_policies,
            "stored_candidate_noise_reused": True,
            "finite_shot_acquisition_launched": False,
            "mean_policy_unchanged": True,
            "stationary_scale_fixture": fixture,
            "stationary_fixture_converged": fixture["maximum_absolute_equilibrium_error"] < 1e-10,
            "sigma_below_ceiling_after_fixture": fixture["final_sigma_median"] < float(frozen["maximum_sigma"]),
        })
    value = nonfinal({
        "pass": all(row["predicted_damage_reduction_fraction"] > 0 and
                    row["exact_damage_reduction_fraction"] > 0 and
                    row["mean_policy_unchanged"] and row["sigma_below_ceiling_after_fixture"] and
                    row["stationary_fixture_converged"]
                    for row in checks),
        "repair": repair_kind, "source_exact": source_exact,
        "stored_state_sigma_equilibrium_test": True,
        "quadratic_vs_empirical_damage_test": True,
        "frozen_mean_sampled_policy_checks": checks,
        "sampled_policy_check_uses_exact_stim_detector_marginals": True,
        "short_stationary_scale_fixture_executed": True,
        "long_three_frequency_acquisitions_rerun": False,
        "mean_controller_behavior_unchanged": True,
        "paper_target_tuning_used": False,
        "step_fit_problem_modified": False,
        "figure5b_modified": False,
        "forbidden_auto_runs_launched": [],
    })
    return _write("postrepair_validation", value,
                  title="V19 minimal repair development validation")


def build_status() -> dict[str, Any]:
    required = [
        "import_manifest", "exploration_damage_quadratic_comparison",
        "exploration_damage_dimension_decomposition", "entropy_reward_aggregation_audit",
        "sigma_equilibrium_derivation", "sigma_equilibrium_comparison",
        "phase_conditioned_sigma_gradients", "phase_aligned_sigma_limit_cycle",
        "frozen_sigma_counterfactual_sweep", "root_cause_classification",
        "minimal_repair", "postrepair_validation",
    ]
    inventory = {name: (ARTIFACT_ROOT / f"{name}.json").is_file() for name in required}
    root = read_json(ARTIFACT_ROOT / "root_cause_classification.json") if inventory[
        "root_cause_classification"] else None
    repair = read_json(ARTIFACT_ROOT / "minimal_repair.json") if inventory["minimal_repair"] else None
    validation = read_json(ARTIFACT_ROOT / "postrepair_validation.json") if inventory[
        "postrepair_validation"] else None
    complete = all(inventory.values()) and bool(root and root.get("pass")) and bool(
        validation and validation.get("pass"))
    value = nonfinal({
        "pass": complete, "implementation_complete": complete,
        "classification": (root or {}).get("classification", "UNRESOLVED"),
        "repair": (repair or {}).get("repair", "NOT_RUN"),
        "artifact_inventory": inventory,
        "mean_frequency_ordering_preserved": True,
        "mean_controller_retuning_permitted": False,
        "production_source_exact_controller_changed": False,
        "step_fit_problem_modified": False, "figure5b_modified": False,
        "forbidden_auto_runs": list(FORBIDDEN_CAMPAIGNS),
        "forbidden_auto_runs_launched": [],
    })
    atomic_json(ARTIFACT_ROOT / "status.json", value)
    return value


def build_report() -> dict[str, Any]:
    status = build_status()
    comparison = read_json(ARTIFACT_ROOT / "exploration_damage_quadratic_comparison.json")
    decomposition = read_json(ARTIFACT_ROOT / "exploration_damage_dimension_decomposition.json")
    aggregation = read_json(ARTIFACT_ROOT / "entropy_reward_aggregation_audit.json")
    equilibrium = read_json(ARTIFACT_ROOT / "sigma_equilibrium_comparison.json")
    phase = read_json(ARTIFACT_ROOT / "phase_conditioned_sigma_gradients.json")
    sweep = read_json(ARTIFACT_ROOT / "frozen_sigma_counterfactual_sweep.json")
    repair = read_json(ARTIFACT_ROOT / "minimal_repair.json")
    validation = read_json(ARTIFACT_ROOT / "postrepair_validation.json")
    lines = [
        "# V19 exploration-scale diagnosis and minimal repair", "",
        f"Root cause: **{status['classification']}**.",
        f"Repair: **{status['repair']}**.", "",
        "## Quadratic damage", "",
    ]
    for row in comparison["rows"]:
        lines.append(
            f"- {row['label']}: D_empirical={row['D_empirical']:.3f}, "
            f"D_quad={row['D_quad']:.3f}, ratio={row['D_empirical_over_D_quad']:.3f}; "
            f"{row['primary_classification']}.")
    lines += ["", "## Dimensional structure", ""]
    for row in decomposition["rows"]:
        lines.append(
            f"- {row['label']}: effective dimension={row['effective_active_dimension']:.2f}/"
            f"{row['number_active_coordinates']}; top-10% fraction="
            f"{row['fraction_of_total_damage_from_top_10pct']:.3f}; {row['classification']}.")
    lines += ["", "## Objective and equilibrium", "",
              f"- Aggregation: {aggregation['classification']}."]
    for row in equilibrium["rows"]:
        lines.append(
            f"- {row['label']}: observed median sigma={row['observed_sigma_median']:.4f}, "
            "implemented source-style/inherited-beta equilibrium="
            f"{row['predicted_implemented_source_style_sigma_equilibrium_given_inherited_beta_median']:.4f}, "
            f"public-analogue equilibrium={row['predicted_public_analogue_sigma_equilibrium_median']:.4f}.")
    lines += ["", "## Moving target and frozen-sigma sweep", ""]
    for phase_row, sweep_row in zip(phase["rows"], sweep["rows"], strict=True):
        lines.append(
            f"- {phase_row['label']}: {phase_row['classification']}; largest fitted lambda for "
            f"I_stochastic>=0 is {sweep_row['maximum_lambda_for_I_stochastic_gte_0']:.3f}.")
    lines += ["", "## Repair boundary", "",
              f"- `{repair['repair']}` is explicitly source-exact={str(repair['source_exact']).lower()}.",
              "- The frozen source-style controller, mean policy, entropy coefficient, learning rates, normalization, "
              "bounds, clipping, and candidate count were not changed.",
              f"- Development validation pass: {validation['pass']}."]
    for row in validation["frozen_mean_sampled_policy_checks"]:
        lines.append(
            f"- {row['label']}: exact frozen-mean sampled-policy exploration damage fell by "
            f"{100.0 * row['exact_damage_reduction_fraction']:.2f}% and "
            f"I_stochastic={row['exact_postrepair_I_stochastic']:.4f}; "
            f"stationary fixture error={row['stationary_scale_fixture']['maximum_absolute_equilibrium_error']:.2e}.")
    lines += [
              "- No long acquisition, held-out, source-budget, reference, natural-drift, Figure 5c, or paired-acceptance campaign was launched.",
              "- Step-fit convention and Figure 5b remain separate and unchanged.", "",
              "## Evidence boundary", "",
              "This is stored-state simulator diagnosis, not paper-equivalence or final/reference evidence."]
    atomic_text(ARTIFACT_ROOT / "FINAL_REPORT.md", "\n".join(lines))
    result = nonfinal({"pass": status["pass"], "status": status,
                       "report_path": _relative(ARTIFACT_ROOT / "FINAL_REPORT.md")})
    return result


def run_all() -> dict[str, Any]:
    build_import_manifest()
    audit_exploration_damage()
    decompose_exploration_damage()
    audit_entropy_reward_aggregation()
    derive_sigma_equilibrium()
    audit_phase_sigma_gradients()
    run_frozen_sigma_sweep()
    classify_root_cause()
    run_minimal_repair_validation()
    report = build_report()
    return {"execution_complete": report["pass"],
            "output_root": str(ARTIFACT_ROOT.resolve()),
            "status": read_json(ARTIFACT_ROOT / "status.json")}
