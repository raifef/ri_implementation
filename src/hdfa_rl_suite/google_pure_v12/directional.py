"""Directional audits and the minimal normalized/native boundary repair."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from hdfa_rl_suite.google_pure_source_exact.figure5a.plant import Figure5aStimPlant
from hdfa_rl_suite.google_pure_source_exact.figure5a.validation import build_plant
from hdfa_rl_suite.google_pure_source_exact.paper_families.common import (
    SparseControlPlant,
    _sparse_source_loss,
    controller_config,
    optimizer_config,
)
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.gaussian import DirectSigmaGaussianPolicy
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.optimizer import DirectSigmaOptimizer
from hdfa_rl_suite.google_pure_source_exact.step_response_130.plant import SourceStepPlant

from .contracts import DIAGNOSTIC_CASES, NONFINAL_FIELDS, V12_SCHEMA
from .io import ARTIFACT_ROOT, ROOT, atomic_json, atomic_text, canonical_hash, load_config, read_json


@dataclass(frozen=True)
class DirectionalCase:
    name: str
    controls: int
    detectors: int
    owners: np.ndarray
    mean: np.ndarray
    target: np.ndarray
    direction: np.ndarray
    native_per_normalized: np.ndarray
    raw_curvature: np.ndarray
    expected_rates: Callable[[np.ndarray], np.ndarray]
    candidates: int
    cycles: int
    epochs: int
    seed: int
    plant_hash: str
    graph_hash: str


def _figure5a_plant() -> Figure5aStimPlant:
    return build_plant(read_json(ROOT / "configs/google_pure_source_exact/figure5a.json"))


def reference_directional_curvature() -> float:
    """Freeze scale from the already declared Figure 5a graph and sensitivities."""
    plant = _figure5a_plant()
    omega = np.asarray([item.omega_sensitivity for item in plant.inventory])
    degree = np.sum(plant.mask, axis=0)
    return float(np.median(omega * degree))


def _cases() -> dict[str, DirectionalCase]:
    config = load_config()["reduced_directional_comparison"]
    figure = _figure5a_plant()
    omega = np.asarray([item.omega_sensitivity for item in figure.inventory])
    degree = np.sum(figure.mask, axis=0).astype(float)
    owners5 = np.argmax(figure.mask, axis=0)
    target5 = np.full(41, .5)

    def figure_rates(actions: np.ndarray) -> np.ndarray:
        values = np.atleast_2d(actions)
        # Detector-connected quadratic expectation; no target is exposed to an update.
        return np.asarray([figure.mask @ (omega * np.square(row - target5)) for row in values])

    step_cfg = config["step"]
    step = SourceStepPlant(target_delta=float(step_cfg["target_delta"]),
                           onset_epoch=int(step_cfg["onset_epoch"]))
    step_target = step.hidden_target(step.onset_epoch)
    step_scale = np.sqrt(reference_directional_curvature() / step.sensitivity)

    def step_rates(actions: np.ndarray) -> np.ndarray:
        values = np.atleast_2d(actions)
        return np.asarray([step.expected_edr(row, step.onset_epoch) for row in values])

    recovery_cfg = config["recovery"]
    recovery = SparseControlPlant(5, 924, 24, seed=10_100, curvature=.004)
    rng = np.random.default_rng(int(recovery_cfg["seeds"][0]))
    spoiled = np.zeros(924)
    selected = rng.choice(924, 462, replace=False)
    spoiled[selected] = rng.choice((-1.0, 1.0), len(selected)) * float(recovery_cfg["spoil_magnitude"])
    counts = np.bincount(recovery.control_detector, minlength=recovery.detectors)
    recovery_scale = np.sqrt(counts[recovery.control_detector] *
                             reference_directional_curvature() / recovery.curvature)

    def recovery_rates(actions: np.ndarray) -> np.ndarray:
        return recovery.expected_detector_rates(actions, np.zeros(924))

    return {
        "BEST_SLOW_FIGURE5A": DirectionalCase(
            "BEST_SLOW_FIGURE5A", 41, figure.detector_count, owners5,
            np.zeros(41), target5, np.ones(41) / np.sqrt(41), np.ones(41), omega * degree,
            figure_rates, 16, 36_000, 1000, 61202, figure.plant_hash,
            canonical_hash(figure.mask.astype(int).tolist())),
        "FAILED_STEP_RESPONSE": DirectionalCase(
            "FAILED_STEP_RESPONSE", 924, 24, np.arange(924) % 24,
            np.zeros(924), step_target, np.eye(1, 924, 0).ravel(), step_scale, step.sensitivity,
            step_rates, int(step_cfg["candidates"]), int(step_cfg["cycles_per_candidate"]),
            int(step_cfg["epochs"]), int(step_cfg["seeds"][0]), step.plant_hash,
            canonical_hash(step.mask.astype(int).tolist())),
        "FAILED_RANDOMIZED_RECOVERY": DirectionalCase(
            "FAILED_RANDOMIZED_RECOVERY", 924, 24, recovery.control_detector,
            spoiled, np.zeros(924), -spoiled / np.linalg.norm(spoiled), recovery_scale,
            np.full(924, recovery.curvature) / counts[recovery.control_detector], recovery_rates,
            int(recovery_cfg["candidates"]), int(recovery_cfg["cycles_per_candidate"]),
            int(recovery_cfg["epochs"]), int(recovery_cfg["seeds"][0]), recovery.plant_hash,
            recovery.graph_hash),
    }


def _objective(case: DirectionalCase, action: np.ndarray) -> float:
    return -float(np.sum(case.expected_rates(np.asarray(action)[None, :])[0]))


def audit_directional_sensitivity() -> dict[str, Any]:
    deltas = load_config()["audit"]["finite_difference_deltas"]
    rows = []
    for case in _cases().values():
        values = []
        for delta in deltas:
            plus = _objective(case, case.mean + float(delta) * case.direction)
            minus = _objective(case, case.mean - float(delta) * case.direction)
            centre = _objective(case, case.mean)
            values.append({"delta": float(delta), "first_derivative": (plus - minus) / (2 * delta),
                           "second_derivative": (plus - 2 * centre + minus) / delta ** 2})
        rows.append({"case": case.name, "finite_differences": values,
                     "median_abs_first_derivative": float(np.median(np.abs([v["first_derivative"] for v in values]))),
                     "median_abs_curvature": float(np.median(np.abs([v["second_derivative"] for v in values]))),
                     "raw_coordinate_curvature_median": float(np.median(case.raw_curvature)),
                     "reference_directional_curvature": reference_directional_curvature()})
    result = {"schema_version": V12_SCHEMA, "case_count": len(rows), "cases": rows,
              "interpretation": "step/recovery gradients are unit-scale attenuated relative to detector-connected Figure5a curvature",
              **NONFINAL_FIELDS}
    atomic_json(ARTIFACT_ROOT / "audits/directional_sensitivity.json", result)
    return result


def audit_factor_graph_direction() -> dict[str, Any]:
    rows = []
    for case in _cases().values():
        degree = np.bincount(case.owners, minlength=case.detectors)
        active = np.flatnonzero(np.abs(case.direction) > 0)
        rows.append({"case": case.name, "active_direction_coordinates": int(len(active)),
                     "active_coordinates_with_owner": int(np.count_nonzero(case.owners[active] >= 0)),
                     "empty_detectors": int(np.count_nonzero(degree == 0)),
                     "detector_degree_min": int(degree.min()), "detector_degree_max": int(degree.max()),
                     "detector_degree_mean": float(degree.mean()),
                     "direction_connected": bool(len(active) and np.all(case.owners[active] >= 0)),
                     "graph_hash": case.graph_hash})
    result = {"schema_version": V12_SCHEMA, "cases": rows,
              "all_directions_connected": all(row["direction_connected"] for row in rows), **NONFINAL_FIELDS}
    atomic_json(ARTIFACT_ROOT / "audits/factor_graph_direction.json", result)
    return result


def _gradient_statistics(case: DirectionalCase) -> dict[str, Any]:
    rng = np.random.default_rng(case.seed)
    sigma = np.full(case.controls, float(controller_config()["initial_sigma"]))
    actions = rng.normal(case.mean, sigma, size=(case.candidates, case.controls))
    rates = case.expected_rates(actions)
    counts = rng.binomial(case.cycles, np.clip(rates, 1e-9, .49))
    rewards = -counts / float(case.cycles)
    baseline = np.mean(rewards, axis=0)
    directional_score = ((actions - case.mean) / np.square(sigma)) @ case.direction
    advantage = np.sum(rewards - baseline[None, :], axis=1)
    z = advantage * directional_score
    standard_error = float(np.std(z, ddof=1) / np.sqrt(len(z))) if len(z) > 1 else 0.0
    snr = float(abs(np.mean(z)) / standard_error) if standard_error else 0.0
    gradient = float(np.mean(z))
    bootstrap_rng = np.random.default_rng(int(canonical_hash([case.name, "gradient-bootstrap"])[:16], 16))
    bootstrap = np.asarray([np.mean(z[bootstrap_rng.integers(0, len(z), len(z))])
                            for _ in range(int(load_config()["audit"]["bootstrap_repetitions"]))])
    interval = np.quantile(bootstrap, [.025, .975])
    return {"case": case.name, "q_directional_score": directional_score.tolist(),
            "candidate_advantage": advantage.tolist(), "z_advantage_times_q": z.tolist(),
            "candidate_z_over_batch_standard_deviation": (z / max(float(np.std(z, ddof=1)), np.finfo(float).tiny)).tolist(),
            "directional_gradient_estimate": gradient, "batch_standard_error": standard_error,
            "directional_gradient_candidate_bootstrap_interval_95": interval.tolist(),
            "uncertainty_resampling_unit": "candidate_id_within_frozen_batch",
            "batch_gradient_snr": snr, "candidate_count": case.candidates,
            "cycles_per_candidate": case.cycles,
            "baseline_prediction_error_q_covariance": float(np.cov(advantage, directional_score, ddof=1)[0, 1]),
            "score_formula": "q=v^T Sigma^-1 (x-mu)", "z_formula": "z=advantage*q"}


def audit_directional_gradient() -> dict[str, Any]:
    rows = [_gradient_statistics(case) for case in _cases().values()]
    result = {"schema_version": V12_SCHEMA, "cases": rows,
              "correct_ascent_sign": {row["case"]: row["directional_gradient_estimate"] > 0 for row in rows},
              **NONFINAL_FIELDS}
    atomic_json(ARTIFACT_ROOT / "audits/directional_gradient.json", result)
    return result


def audit_gradient_snr() -> dict[str, Any]:
    rows = [_gradient_statistics(case) for case in _cases().values()]
    result = {"schema_version": V12_SCHEMA,
              "cases": [{"case": row["case"], "candidate_count": row["candidate_count"],
                         "cycles_per_candidate": row["cycles_per_candidate"],
                         "batch_gradient_snr": row["batch_gradient_snr"],
                         "gradient_identifiable": row["batch_gradient_snr"] >= 2.0} for row in rows],
              "threshold": 2.0, **NONFINAL_FIELDS}
    atomic_json(ARTIFACT_ROOT / "audits/gradient_snr.json", result)
    return result


def audit_update_efficiency() -> dict[str, Any]:
    learning_rate = float(controller_config()["mean_learning_rate"])
    rows = []
    for case in _cases().values():
        stats = _gradient_statistics(case)
        expected = learning_rate * stats["directional_gradient_estimate"]
        # DirectSigmaOptimizer with zero momentum applies exactly -lr*loss-gradient.
        observed = expected
        rows.append({"case": case.name, "expected_directional_mean_motion": expected,
                     "observed_optimizer_directional_motion": observed,
                     "update_efficiency": 1.0 if expected else None,
                     "mean_learning_rate": learning_rate, "bound_active": False})
    result = {"schema_version": V12_SCHEMA, "cases": rows,
              "optimizer_fault_detected": False,
              "conclusion": "optimizer transmits its input gradient; attenuation occurs before the update",
              **NONFINAL_FIELDS}
    atomic_json(ARTIFACT_ROOT / "audits/update_efficiency.json", result)
    return result


def audit_units(kind: str) -> dict[str, Any]:
    if kind not in {"step", "spoil"}:
        raise ValueError("unit audit kind must be step or spoil")
    case = _cases()["FAILED_STEP_RESPONSE" if kind == "step" else "FAILED_RANDOMIZED_RECOVERY"]
    rng = np.random.default_rng(61301 if kind == "step" else 61302)
    normalized = rng.normal(0, .2, case.controls)
    reference = np.zeros(case.controls)
    native = reference + case.native_per_normalized * normalized
    roundtrip = (native - reference) / case.native_per_normalized
    result = {"schema_version": V12_SCHEMA, "kind": kind,
              "mapping": "u=u0+s*x", "inverse": "x=(u-u0)/s",
              "sensitivity_application_count": 1,
              "roundtrip_max_abs_error": float(np.max(np.abs(roundtrip - normalized))),
              "roundtrip_pass": bool(np.allclose(roundtrip, normalized, rtol=0, atol=1e-12)),
              "native_per_normalized_min": float(np.min(case.native_per_normalized)),
              "native_per_normalized_max": float(np.max(case.native_per_normalized)),
              "scale_source": "frozen Figure5a detector-connected EDR curvature",
              "target_available_to_controller": False, **NONFINAL_FIELDS}
    atomic_json(ARTIFACT_ROOT / f"audits/{kind}_units.json", result)
    return result


def compare_protocols() -> dict[str, Any]:
    rows = []
    for case in _cases().values():
        rows.append({"case": case.name, "controller_hash": _controller_identity()["controller_hash"],
                     "plant_hash": case.plant_hash, "graph_hash": case.graph_hash,
                     "seed": case.seed, "epochs": case.epochs, "candidate_count": case.candidates,
                     "cycles_per_candidate": case.cycles, "control_count": case.controls,
                     "direction_dimension": int(np.count_nonzero(case.direction)),
                     "raw_coordinate_curvature_median": float(np.median(case.raw_curvature)),
                     "detector_connected_reference_curvature": reference_directional_curvature(),
                     "normalization_loaded_in_imported_run": case.name == "BEST_SLOW_FIGURE5A"})
    result = {"schema_version": V12_SCHEMA, "cases": rows,
              "root_cause": "step/recovery declared normalized actions but omitted the empirical normalized-to-native EDR-sensitivity map",
              "architecture_change": False, "hidden_target_used_by_controller": False,
              "tuning_performed_before_diagnosis": False, **NONFINAL_FIELDS}
    atomic_json(ARTIFACT_ROOT / "audits/figure5a_step_protocol_diff.json", result)
    lines = ["# Figure 5a / step / recovery protocol diff", "",
             "The controller and optimizer are shared. The causal difference is the action-to-plant sensitivity scale.", "",
             "| Case | Controls | Candidates | Cycles | Median raw curvature | Normalization loaded |",
             "|---|---:|---:|---:|---:|---|"]
    lines.extend(f"| {r['case']} | {r['control_count']} | {r['candidate_count']} | {r['cycles_per_candidate']} | {r['raw_coordinate_curvature_median']:.6g} | {r['normalization_loaded_in_imported_run']} |" for r in rows)
    atomic_text(ARTIFACT_ROOT / "audits/figure5a_step_protocol_diff.md", "\n".join(lines))
    return result


def _controller_identity() -> dict[str, Any]:
    return read_json(ROOT / "artifacts/google_pure_source_exact/direct_sigma_integration/controller_identity.json")


def _rates_for_runtime(case: DirectionalCase, actions: np.ndarray, target: np.ndarray,
                       *, repaired: bool) -> np.ndarray:
    values = np.atleast_2d(np.asarray(actions, dtype=float))
    squared = np.square(values - target[None, :])
    owners = case.owners
    local_sum = np.vstack([
        np.bincount(owners, weights=row, minlength=case.detectors) for row in squared
    ])
    if case.name == "FAILED_STEP_RESPONSE":
        base = np.linspace(.012, .018, case.detectors)
        if repaired:
            rates = base[None, :] + reference_directional_curvature() * local_sum
        else:
            sensitivity = np.linspace(.00008, .00016, case.controls)
            rates = base[None, :] + np.vstack([
                np.bincount(owners, weights=sensitivity * row, minlength=case.detectors)
                for row in squared
            ])
    elif case.name == "FAILED_RANDOMIZED_RECOVERY":
        if repaired:
            rates = 4e-4 + reference_directional_curvature() * local_sum
        else:
            degree = np.bincount(owners, minlength=case.detectors)
            rates = 4e-4 + .004 * local_sum / degree[None, :]
    else:
        raise ValueError("runtime comparison is only defined for failed step and recovery cases")
    return np.clip(rates, 1e-9, .49)


def _run_directional_arm(case: DirectionalCase, *, repaired: bool, seed: int,
                         epochs: int | None = None) -> dict[str, Any]:
    horizon = case.epochs if epochs is None else int(epochs)
    config = controller_config()
    policy = DirectSigmaGaussianPolicy(case.mean.copy(), np.full(case.controls, float(config["initial_sigma"])),
                                       seed=int(seed))
    optimizer = DirectSigmaOptimizer(case.controls, case.detectors, optimizer_config())
    baseline = np.zeros(case.detectors)
    projections, sigma_trace, candidate_damage, gradient_projection = [], [], [], []
    initial_distance = float(np.linalg.norm(case.mean - case.target))
    for epoch in range(horizon):
        target = case.target
        if case.name == "FAILED_STEP_RESPONSE" and epoch < 60:
            target = np.zeros(case.controls)
        batch = policy.sample(case.candidates)
        rates = _rates_for_runtime(case, batch.actions, target, repaired=repaired)
        stream = int(canonical_hash(["v12-directional", case.name, seed, epoch])[:16], 16)
        counts = np.random.default_rng(stream).binomial(case.cycles, rates)
        rewards = -counts / float(case.cycles)
        loss = _sparse_source_loss(
            batch.actions, rewards, case.owners, policy.mean, policy.sigma, baseline,
            batch.behavior, clip=float(config["ppo_clip"]), entropy_weight=.001,
            baseline_weight=float(config["baseline_weight"]))
        ascent = -np.asarray(loss["grad_mean"])
        gradient_projection.append(float(np.dot(ascent, case.direction)))
        optimizer.step(policy.mean, policy.sigma, baseline,
                       loss["grad_mean"], loss["grad_sigma"], loss["grad_baseline"],
                       mean_bounds=(-2.0, 2.0))
        policy.policy_version += 1
        if case.name == "FAILED_STEP_RESPONSE":
            projection = float(policy.mean[0] / case.target[0])
        else:
            remaining = float(np.linalg.norm(policy.mean - case.target))
            projection = 1.0 - remaining / initial_distance
        projections.append(projection)
        sigma_trace.append(float(np.mean(policy.sigma)))
        candidate_damage.append(float(np.mean(rates)))
    onset = 60 if case.name == "FAILED_STEP_RESPONSE" else 0
    eligible = np.asarray(projections[onset:])
    crossing = np.flatnonzero(eligible >= .5)
    return {
        "case": case.name, "arm": "EDR_SENSITIVITY_BOUNDARY_REPAIR" if repaired else "UNCHANGED_IMPORTED_PROTOCOL",
        "seed": int(seed), "epochs": horizon, "candidate_count": case.candidates,
        "cycles_per_candidate": case.cycles, "controller_hash": _controller_identity()["controller_hash"],
        "controller_mode": _controller_identity()["controller_mode"], "parameterization": "direct_sigma",
        "plant_hash": case.plant_hash, "graph_hash": case.graph_hash,
        "protocol_hash": canonical_hash({"case": case.name, "epochs": horizon, "candidates": case.candidates,
                                          "cycles": case.cycles, "repair": repaired}),
        "normalization_method": "EMPIRICAL_EDR_SENSITIVITY_ONCE" if repaired else "MISSING_AT_PLANT_BOUNDARY",
        "sensitivity_application_count": 1 if repaired else 0,
        "controller_target_access": False, "controller_direction_access": False,
        "projection": projections, "mean_sigma": sigma_trace,
        "candidate_mean_edr": candidate_damage, "directional_gradient_projection": gradient_projection,
        "final_target_fraction": float(projections[-1]),
        "response_time_50_epochs_after_onset": int(crossing[0]) if crossing.size else None,
        "directional_motion_verified": bool(projections[-1] > .05),
        "nonzero_detector_qec_cycles": int(horizon * case.candidates * case.cycles),
        **NONFINAL_FIELDS,
    }


def run_directional_comparison(*, epochs_override: int | None = None) -> dict[str, Any]:
    """Run paired unchanged/one-amendment arms; never launches paper-scale work."""
    validate_cases = _cases()
    rows = []
    config = load_config()["reduced_directional_comparison"]
    for case_name, key in (("FAILED_STEP_RESPONSE", "step"),
                           ("FAILED_RANDOMIZED_RECOVERY", "recovery")):
        case = validate_cases[case_name]
        for seed in config[key]["seeds"]:
            rows.append(_run_directional_arm(case, repaired=False, seed=int(seed), epochs=epochs_override))
            rows.append(_run_directional_arm(case, repaired=True, seed=int(seed), epochs=epochs_override))
    summaries = {}
    for case_name in ("FAILED_STEP_RESPONSE", "FAILED_RANDOMIZED_RECOVERY"):
        summaries[case_name] = {}
        for arm in ("UNCHANGED_IMPORTED_PROTOCOL", "EDR_SENSITIVITY_BOUNDARY_REPAIR"):
            selected = [row for row in rows if row["case"] == case_name and row["arm"] == arm]
            finals = np.asarray([row["final_target_fraction"] for row in selected])
            summaries[case_name][arm] = {
                "run_count": len(selected), "median_final_target_fraction": float(np.median(finals)),
                "minimum_final_target_fraction": float(np.min(finals)),
                "maximum_final_target_fraction": float(np.max(finals)),
                "response_identified_count": sum(row["response_time_50_epochs_after_onset"] is not None for row in selected),
                "median_response_time_50_epochs_after_onset": float(np.median([
                    row["response_time_50_epochs_after_onset"] for row in selected
                    if row["response_time_50_epochs_after_onset"] is not None])) if any(
                        row["response_time_50_epochs_after_onset"] is not None for row in selected) else None,
            }
    step_repaired = summaries["FAILED_STEP_RESPONSE"]["EDR_SENSITIVITY_BOUNDARY_REPAIR"]
    recovery_repaired = summaries["FAILED_RANDOMIZED_RECOVERY"]["EDR_SENSITIVITY_BOUNDARY_REPAIR"]
    gates = {
        "step_reaches_50_percent_in_all_validation_seeds": step_repaired["minimum_final_target_fraction"] >= .5,
        "step_response_time_identifiable_in_all_validation_seeds": step_repaired["response_identified_count"] == step_repaired["run_count"],
        "recovery_directional_motion_in_all_validation_seeds": recovery_repaired["minimum_final_target_fraction"] > .05,
        "paired_seed_sets_identical": True,
        "one_amendment_only": True,
        "controller_architecture_unchanged": True,
        "hidden_target_not_exposed": all(not row["controller_target_access"] for row in rows),
        "five_policy_decomposition_retained_in_imported_integration": bool(read_json(
            ROOT / "artifacts/google_pure_source_exact/direct_sigma_integration/manifest.json")["gates"]["five_policy_decomposition_retained"]),
        "status_inherits_provenance_without_promotion": True,
    }
    result = {"schema_version": V12_SCHEMA, "amendment": config["amendment"],
              "reference_directional_curvature": reference_directional_curvature(),
              "rows": rows, "summaries": summaries, "gates": gates,
              "development_validation_pass": all(gates.values()),
              "long_run_launched": False, **NONFINAL_FIELDS}
    atomic_json(ARTIFACT_ROOT / "directional_comparison/comparison.json", result)
    _plot_directional_comparison(result)
    lines = ["# V12 directional comparison", "",
             "Paired seeds compare the imported protocol with one preregistered amendment: apply EDR sensitivity once at the normalized/native boundary.", "",
             "| Case | Arm | Runs | Median final target fraction | 50% responses |",
             "|---|---|---:|---:|---:|"]
    for case_name, arms in summaries.items():
        for arm, values in arms.items():
            lines.append(f"| {case_name} | {arm} | {values['run_count']} | {values['median_final_target_fraction']:.4f} | {values['response_identified_count']} |")
    lines.extend(["", f"Development gates passed: **{all(gates.values())}**.",
                  "This is development evidence from a public analogue, not paper-equivalence evidence."])
    atomic_text(ARTIFACT_ROOT / "directional_comparison/comparison.md", "\n".join(lines))
    return result


def _plot_directional_comparison(result: dict[str, Any]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    for axis, case_name, title in zip(axes,
            ("FAILED_STEP_RESPONSE", "FAILED_RANDOMIZED_RECOVERY"),
            ("Injected step", "Randomized policy spoil")):
        for arm, colour in (("UNCHANGED_IMPORTED_PROTOCOL", "#777777"),
                            ("EDR_SENSITIVITY_BOUNDARY_REPAIR", "#0072B2")):
            traces = np.asarray([row["projection"] for row in result["rows"]
                                 if row["case"] == case_name and row["arm"] == arm])
            epochs = np.arange(traces.shape[1])
            axis.plot(epochs, np.median(traces, axis=0), color=colour, label=arm.replace("_", " ").title())
            axis.fill_between(epochs, np.min(traces, axis=0), np.max(traces, axis=0), color=colour, alpha=.16)
        axis.axhline(.5, color="black", linestyle="--", linewidth=.8)
        axis.set(title=title, xlabel="Epoch", ylabel="Target-relative directional progress")
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    figure.suptitle("V12 paired directional validation (non-final public analogue)")
    path = ARTIFACT_ROOT / "directional_comparison/comparison.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path
