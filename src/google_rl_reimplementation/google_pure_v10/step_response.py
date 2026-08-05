"""Target-relative injected-step acquisition, fitting, and causal ablations."""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, Mapping

import numpy as np

from google_rl_reimplementation.google_pure_v6.plant import PureQuadraticPlant, default_spec
from google_rl_reimplementation.google_pure_v6.reference_agent import PureGoogleV6Agent, evidence_from_counts
from google_rl_reimplementation.google_pure_v7.config import canonical_hash
from google_rl_reimplementation.google_pure_v9.common import guard_seed
from google_rl_reimplementation.google_pure_v9.contracts import ControllerConfig, five_policy_decomposition

from .common import artifact_root, load_config, read_json, write_artifact
from .contracts import ExperimentFamily, evidence_envelope, validate_provenance


def piecewise_constant_optimum(
    epochs: int,
    onset_epoch: int,
    pre_optimum: np.ndarray,
    post_optimum: np.ndarray,
) -> np.ndarray:
    pre = np.asarray(pre_optimum, dtype=float)
    post = np.asarray(post_optimum, dtype=float)
    if epochs < 4 or not 0 < onset_epoch < epochs or pre.shape != post.shape or pre.ndim != 1:
        raise ValueError("valid horizon, interior onset, and aligned optimum vectors are required")
    tape = np.repeat(pre[None, :], epochs, axis=0)
    tape[onset_epoch:] = post
    if not np.allclose(tape[:onset_epoch], pre) or not np.allclose(tape[onset_epoch:], post):
        raise RuntimeError("step optimum is not piecewise constant")
    return tape


def normalized_response(
    learned_mean: np.ndarray,
    optimum: np.ndarray,
    *,
    onset_epoch: int,
    weighting_matrix: np.ndarray,
) -> dict[str, np.ndarray]:
    mean = np.asarray(learned_mean, dtype=float)
    tape = np.asarray(optimum, dtype=float)
    weight = np.asarray(weighting_matrix, dtype=float)
    if mean.shape != tape.shape or mean.ndim != 2 or weight.shape != (mean.shape[1], mean.shape[1]):
        raise ValueError("mean, optimum, and weighting matrix shapes are inconsistent")
    if not np.allclose(weight, weight.T) or np.min(np.linalg.eigvalsh(weight)) <= 0:
        raise ValueError("weighting matrix must be symmetric positive definite")
    if not 0 < onset_epoch < len(mean):
        raise ValueError("step onset must be interior")
    delta = tape[onset_epoch] - tape[onset_epoch - 1]
    denominator = float(delta @ weight @ delta)
    if denominator <= 0:
        raise ValueError("step displacement must be nonzero under the frozen weighting matrix")
    reference_mean = mean[onset_epoch - 1].copy()
    response = (mean - reference_mean[None, :]) @ weight @ delta / denominator
    optimum_response = (tape - tape[onset_epoch - 1][None, :]) @ weight @ delta / denominator
    return {
        "response": response,
        "error": 1.0 - response,
        "optimum_response": optimum_response,
        "delta_optimum": delta,
        "reference_mean": reference_mean,
        "denominator": np.asarray(denominator),
    }

def _first_crossing(values: np.ndarray, threshold: float) -> int | None:
    hits = np.flatnonzero(values >= threshold)
    return int(hits[0]) if len(hits) else None


def fit_step_response(
    response: np.ndarray,
    *,
    onset_epoch: int,
    sustained_epochs: int = 12,
    settling_tolerance: float = 0.05,
) -> dict[str, Any]:
    values = np.asarray(response, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or not 0 < onset_epoch < len(values) - 4:
        raise ValueError("finite scalar response with an interior onset is required")
    post = values[onset_epoch:]
    time = np.arange(len(post), dtype=float)
    r0 = float(values[onset_epoch - 1])
    final_width = max(8, len(post) // 10)
    final = float(np.mean(post[-final_width:]))
    response_times = {
        "response_time_50_epochs": _first_crossing(post, 0.5),
        "response_time_63_2_epochs": _first_crossing(post, 0.632),
        "response_time_90_epochs": _first_crossing(post, 0.9),
    }
    settled = None
    for index in range(max(0, len(post) - sustained_epochs + 1)):
        if np.all(np.abs(post[index:index + sustained_epochs] - 1.0) <= settling_tolerance):
            settled = index
            break
    tau_grid = np.geomspace(0.5, max(10.0, 3.0 * len(post)), 700)
    fits: list[tuple[float, float, np.ndarray]] = []
    for tau in tau_grid:
        design = np.column_stack((np.ones(len(time)), np.exp(-time / tau)))
        beta, _, rank, _ = np.linalg.lstsq(design, post, rcond=None)
        if rank == 2:
            residual = post - design @ beta
            fits.append((float(residual @ residual), float(tau), beta))
    if not fits:
        raise RuntimeError("step exponential fit is rank deficient")
    sse, tau, beta = min(fits, key=lambda item: item[0])
    variance = sse / max(1, len(post) - 3)
    accepted = [item[1] for item in fits if item[0] <= sse + 3.841458820694124 * variance]
    total = float(np.sum((post - np.mean(post)) ** 2))
    r_squared = 1.0 - sse / total if total > 0 else 1.0
    r_infinity = float(beta[0])
    fitted_r0 = float(beta[0] + beta[1])
    fit_valid = bool(r_squared >= 0.5 and tau > tau_grid[0] * 1.001 and tau < tau_grid[-1] / 1.001)
    return {
        "R_0_observed": r0,
        "R_0_fitted": fitted_r0,
        "R_infinity": r_infinity,
        "tau_epochs": tau,
        "tau_profile_confidence_interval_95_epochs": [float(min(accepted)), float(max(accepted))],
        "fit_r_squared": float(r_squared),
        "fit_sse": sse,
        "fit_valid": fit_valid,
        **response_times,
        "settling_time_95_epochs": settled,
        "overshoot": float(max(0.0, np.max(post) - 1.0)),
        "final_response": final,
        "final_residual": float(1.0 - final),
        "integrated_absolute_error": float(np.sum(np.abs(1.0 - post))),
        "response_classification": "SETTLED" if settled is not None else "NO_SETTLING_WITHIN_HORIZON",
        "fit_model": "R(t)=R_infinity-(R_infinity-R_0)*exp(-(t-t0)/tau)",
    }


def _base_controller() -> ControllerConfig:
    return ControllerConfig(
        initial_scale=0.04,
        minimum_scale=0.001,
        maximum_scale=0.25,
        scale_learning_rate=0.01,
        entropy_coefficient=0.01,
        mean_learning_rate=0.02,
        replay_capacity_epochs=1,
        update_passes=1,
        ppo_clip=0.2,
    )


def _trace(
    controller: ControllerConfig,
    *,
    epochs: int,
    onset_epoch: int,
    candidates: int,
    cycles: int,
    seed: int,
    step_amplitude: float,
) -> dict[str, Any]:
    guard_seed(seed)
    spec = default_spec(6)
    plant = PureQuadraticPlant(spec)
    direction = np.linspace(1.0, 0.45, spec.control_count)
    direction /= np.linalg.norm(direction)
    pre = np.zeros(spec.control_count)
    post = step_amplitude * direction
    tape = piecewise_constant_optimum(epochs, onset_epoch, pre, post)
    agent = PureGoogleV6Agent(
        plant.mask,
        spec.base_optimum_normalized,
        spec.coordinates,
        controller.to_agent_choices(),
        seed=seed,
        objective_mode="source_literal_ppo",
    )
    training_rng = np.random.default_rng(seed + 100_000)
    evaluation_rng = np.random.default_rng(seed + 200_000)
    means = []
    candidates_mean = []
    scales = []
    diagnostics = []
    action_clipping = []
    epoch_costs = {name: [] for name in ("fixed", "oracle", "oracle_with_scale", "learned_mean", "sampled_candidates")}
    for optimum in tape:
        optimum_native = spec.coordinates.to_native(optimum)
        mean = agent.mean.copy()
        scale = agent.scale.copy()
        batch = agent.sample(candidates)
        fixed = np.repeat(spec.base_optimum_normalized[None, :], candidates, axis=0)
        oracle = np.repeat(optimum[None, :], candidates, axis=0)
        learned = np.repeat(mean[None, :], candidates, axis=0)
        oracle_scaled = spec.coordinates.apply_bounds(optimum[None, :] + scale[None, :] * evaluation_rng.normal(size=(candidates, spec.control_count)))
        policy_actions = {
            "fixed": fixed,
            "oracle": oracle,
            "oracle_with_scale": oracle_scaled,
            "learned_mean": learned,
            "sampled_candidates": batch.applied_normalized_actions,
        }
        for name, actions in policy_actions.items():
            counts = plant.acquire_counts(spec.coordinates.to_native(actions), optimum_native, cycles=cycles, rng=evaluation_rng)
            epoch_costs[name].append(float(np.sum(counts) / (candidates * cycles)))
        training_counts = plant.acquire_counts(batch.applied_native_actions, optimum_native, cycles=cycles, rng=training_rng)
        diagnostic = agent.update(batch, evidence_from_counts(batch, training_counts, cycles))
        diagnostics.append(diagnostic)
        means.append(mean)
        candidates_mean.append(np.mean(batch.applied_normalized_actions, axis=0))
        scales.append(scale)
        action_clipping.append(float(np.mean(batch.latent_normalized_actions != batch.applied_normalized_actions)))
    means_array = np.asarray(means)
    candidate_array = np.asarray(candidates_mean)
    scale_array = np.asarray(scales)
    weighting = np.eye(spec.control_count)
    normalized = normalized_response(means_array, tape, onset_epoch=onset_epoch, weighting_matrix=weighting)
    candidate_normalized = normalized_response(candidate_array, tape, onset_epoch=onset_epoch, weighting_matrix=weighting)
    fit = fit_step_response(normalized["response"], onset_epoch=onset_epoch, sustained_epochs=max(8, epochs // 20))
    post_slice = slice(onset_epoch, epochs)
    costs = {name: float(np.mean(np.asarray(values)[post_slice])) for name, values in epoch_costs.items()}
    five_policy = five_policy_decomposition(costs)
    ppo_clip_fraction = float(np.mean([row["clip_fraction"] for row in diagnostics]))
    ratio_min = float(min(row["ratio_min"] for row in diagnostics))
    ratio_max = float(max(row["ratio_max"] for row in diagnostics))
    replay_batches = int(max(row["replay_batches_used"] for row in diagnostics))
    clipping_causally_testable = bool(ppo_clip_fraction > 0 or replay_batches > 0 or ratio_min < 0.8 or ratio_max > 1.2)
    return {
        "controller": controller.to_dict(),
        "controller_hash": canonical_hash(controller.to_dict()),
        "plant_hash": canonical_hash({"mask": plant.mask.tolist(), "curvature": spec.normalized_curvature.tolist()}),
        "graph_hash": canonical_hash(plant.mask.tolist()),
        "seed": seed,
        "epochs": epochs,
        "onset_epoch": onset_epoch,
        "candidates": candidates,
        "cycles_per_candidate": cycles,
        "qec_cycle_budget": epochs * candidates * cycles,
        "candidate_budget": epochs * candidates,
        "optimum": tape,
        "learned_mean": means_array,
        "candidate_mean": candidate_array,
        "policy_scale": scale_array,
        "response": normalized["response"],
        "error": normalized["error"],
        "optimum_response": normalized["optimum_response"],
        "candidate_response": candidate_normalized["response"],
        "weighting_matrix": weighting,
        "fit": fit,
        "five_policy": five_policy,
        "action_clipping_fraction": float(np.mean(action_clipping)),
        "ppo_clip_fraction": ppo_clip_fraction,
        "importance_ratio_range": [ratio_min, ratio_max],
        "replay_batches_used": replay_batches,
        "clipping_causally_testable": clipping_causally_testable,
        "piecewise_constant_optimum_verified": bool(np.allclose(tape[:onset_epoch], pre) and np.allclose(tape[onset_epoch:], post)),
        "coordinates": "normalized",
        "observable_definition": "target-relative weighted projection of learned mean from the pre-step mean",
    }


def plan_step_response(mode: str = "smoke") -> dict[str, Any]:
    if mode not in {"smoke", "reference"}:
        raise ValueError("step-response mode must be smoke or reference")
    config = load_config("step_response.json")
    row = config[mode]
    ablation_runs = 1 + len(config["ppo_clip_grid"]) + len(config["mean_learning_rate_grid"]) + len(config["update_passes_grid"]) + len(config["replay_capacity_grid"])
    payload = {
        "schema_version": "google-pure-v10-step-plan.v1",
        "experiment_family": ExperimentFamily.STEP_RESPONSE_INJECTED_DRIFT_V10.value,
        "mode": mode,
        "runs": ablation_runs,
        "epochs": row["epochs"],
        "onset_epoch": row["onset_epoch"],
        "candidates": row["candidates"],
        "cycles_per_candidate": row["cycles_per_candidate"],
        "estimated_qec_cycles": ablation_runs * int(row["epochs"]) * int(row["candidates"]) * int(row["cycles_per_candidate"]),
        "estimated_runtime": "under two minutes smoke; long explicit user-run reference acquisition",
        "estimated_memory_storage": "under 40 MiB smoke",
        "seed": row["seed"],
        "controller_hash": canonical_hash(_base_controller().to_dict()),
        "protocol_hash": canonical_hash({"row": row, "grids": {key: config[key] for key in ("ppo_clip_grid", "mean_learning_rate_grid", "update_passes_grid", "replay_capacity_grid")}}),
        "qec_cycle_budget_matched": config["qec_cycle_budget_matched"],
        **evidence_envelope(complete=True, mechanism_valid=True, claim_supported=False, paper_comparable=False, blocking_reasons=["ACQUISITION_NOT_EXECUTED"]),
    }
    return write_artifact("step_response/run_plan", payload, "Injected-step Run Plan")


def _save_trace_files(trace: Mapping[str, Any]) -> None:
    target = artifact_root() / "step_response"
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target / "raw_trajectories.npz",
        optimum=trace["optimum"],
        learned_mean=trace["learned_mean"],
        candidate_mean=trace["candidate_mean"],
        policy_scale=trace["policy_scale"],
        response=trace["response"],
        candidate_response=trace["candidate_response"],
        error=trace["error"],
        optimum_response=trace["optimum_response"],
        weighting_matrix=trace["weighting_matrix"],
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7), sharex=True, constrained_layout=True)
    axes[0].plot(trace["optimum_response"], "k--", label="optimum")
    axes[0].plot(trace["response"], label="learned mean")
    axes[0].plot(trace["candidate_response"], alpha=0.7, label="candidate stream")
    axes[0].legend()
    axes[0].set(ylabel="normalized response", title="Target-relative injected-step response")
    axes[1].plot(trace["error"], label="1-R(t)")
    axes[1].axhline(0.1, color="tab:red", linestyle="--", label="90% response")
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].legend()
    axes[1].set(xlabel="epoch", ylabel="target residual")
    fig.savefig(target / "figure.png", dpi=170)
    plt.close(fig)


def run_step_response(*, mode: str = "smoke", execute: bool = False) -> dict[str, Any]:
    if mode == "reference" and not execute:
        raise RuntimeError("reference step-response acquisition requires --execute")
    plan = plan_step_response(mode)
    config = load_config("step_response.json")
    row = config[mode]
    trace = _trace(
        _base_controller(),
        epochs=int(row["epochs"]),
        onset_epoch=int(row["onset_epoch"]),
        candidates=int(row["candidates"]),
        cycles=int(row["cycles_per_candidate"]),
        seed=int(row["seed"]),
        step_amplitude=float(config["step_amplitude"]),
    )
    _save_trace_files(trace)
    fits = {
        "schema_version": "google-pure-v10-step-fits.v1",
        "fit": trace["fit"],
        "observable_definition": trace["observable_definition"],
        "piecewise_constant_optimum_verified": trace["piecewise_constant_optimum_verified"],
        **evidence_envelope(complete=True, mechanism_valid=trace["fit"]["fit_valid"], claim_supported=False, paper_comparable=False, blocking_reasons=[] if trace["fit"]["fit_valid"] else ["EXPONENTIAL_FIT_CREDIBILITY_GATE_FAILED"]),
    }
    fits = write_artifact("step_response/fits", fits, "Injected-step Fits")
    provenance = {
        "experiment_family": ExperimentFamily.STEP_RESPONSE_INJECTED_DRIFT_V10.value,
        "controller_hash": trace["controller_hash"],
        "decoder_hash": None,
        "plant_hash": trace["plant_hash"],
        "graph_hash": trace["graph_hash"],
        "protocol_hash": plan["protocol_hash"],
        "seed": trace["seed"],
        "drift_tape_hash": canonical_hash(np.asarray(trace["optimum"]).tolist()),
        "mode": mode,
        "qec_cycle_budget": trace["qec_cycle_budget"],
        "candidate_budget": trace["candidate_budget"],
        "observable_definition": trace["observable_definition"],
        "analysis_contract": "piecewise-constant optimum, target-relative W projection, profile exponential fit",
    }
    validate_provenance(provenance)
    blockers = [] if mode == "reference" else ["SMOKE_NOT_HELD_OUT_REFERENCE_EVIDENCE"]
    payload = {
        "schema_version": "google-pure-v10-step-results.v1",
        **provenance,
        "run_plan_hash": plan["artifact_hash"],
        "fit_hash": fits["artifact_hash"],
        "response": trace["fit"],
        "five_policy": trace["five_policy"],
        "piecewise_constant_optimum_verified": trace["piecewise_constant_optimum_verified"],
        "optimum_trajectory_stored": True,
        "action_clipping_fraction": trace["action_clipping_fraction"],
        "ppo_clip_fraction": trace["ppo_clip_fraction"],
        "importance_ratio_range": trace["importance_ratio_range"],
        "clipping_causally_testable": trace["clipping_causally_testable"],
        "raw_trajectory_file": "raw_trajectories.npz",
        "figure_file": "figure.png",
        **evidence_envelope(complete=True, mechanism_valid=True, claim_supported=mode == "reference", paper_comparable=False, blocking_reasons=blockers),
    }
    return write_artifact("step_response/report", payload, "Injected-step Response", markdown_relative="step_response/report.md")


def _ablation_config(base: ControllerConfig, dimension: str, value: float | int) -> ControllerConfig:
    mapping = {
        "ppo_clip": {"ppo_clip": float(value)},
        "mean_learning_rate": {"mean_learning_rate": float(value)},
        "update_passes": {"update_passes": int(value)},
        "replay_capacity_epochs": {"replay_capacity_epochs": int(value)},
    }
    if dimension not in mapping:
        raise ValueError("unknown one-factor step ablation")
    return replace(base, **mapping[dimension])


def run_step_ablation(*, mode: str = "smoke", execute: bool = False) -> dict[str, Any]:
    if mode == "reference" and not execute:
        raise RuntimeError("reference step ablation requires --execute")
    config = load_config("step_response.json")
    row = config[mode]
    base = _base_controller()
    baseline = _trace(
        base,
        epochs=int(row["epochs"]),
        onset_epoch=int(row["onset_epoch"]),
        candidates=int(row["candidates"]),
        cycles=int(row["cycles_per_candidate"]),
        seed=int(row["seed"]) + 100,
        step_amplitude=float(config["step_amplitude"]),
    )
    records = [
        {
            "dimension": "baseline",
            "value": None,
            "controller": base.to_dict(),
            "fit": baseline["fit"],
            "five_policy": baseline["five_policy"],
            "action_clipping_fraction": baseline["action_clipping_fraction"],
            "ppo_clip_fraction": baseline["ppo_clip_fraction"],
            "importance_ratio_range": baseline["importance_ratio_range"],
            "qec_cycle_budget": baseline["qec_cycle_budget"],
        }
    ]
    dimensions = [
        ("mean_learning_rate", config["mean_learning_rate_grid"]),
        ("update_passes", config["update_passes_grid"]),
        ("replay_capacity_epochs", config["replay_capacity_grid"]),
    ]
    clipping_operational = baseline["clipping_causally_testable"]
    if clipping_operational:
        dimensions.insert(0, ("ppo_clip", config["ppo_clip_grid"]))
    seed = int(row["seed"]) + 101
    for dimension, values in dimensions:
        for value in values:
            candidate = _ablation_config(base, dimension, value)
            trace = _trace(
                candidate,
                epochs=int(row["epochs"]),
                onset_epoch=int(row["onset_epoch"]),
                candidates=int(row["candidates"]),
                cycles=int(row["cycles_per_candidate"]),
                seed=seed,
                step_amplitude=float(config["step_amplitude"]),
            )
            records.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "controller": candidate.to_dict(),
                    "fit": trace["fit"],
                    "five_policy": trace["five_policy"],
                    "action_clipping_fraction": trace["action_clipping_fraction"],
                    "ppo_clip_fraction": trace["ppo_clip_fraction"],
                    "importance_ratio_range": trace["importance_ratio_range"],
                    "qec_cycle_budget": trace["qec_cycle_budget"],
                }
            )
            seed += 1
    baseline_tau = float(baseline["fit"]["tau_epochs"])
    causal = []
    for record in records[1:]:
        tau = float(record["fit"]["tau_epochs"])
        causal.append(
            {
                "dimension": record["dimension"],
                "value": record["value"],
                "tau_change_fraction": (tau - baseline_tau) / max(baseline_tau, 1e-12),
                "settling_speed_materially_changed": abs(tau - baseline_tau) / max(baseline_tau, 1e-12) >= 0.1,
                "held_out_causal_claim": False,
            }
        )
    budgets = {record["qec_cycle_budget"] for record in records}
    payload = {
        "schema_version": "google-pure-v10-step-ablation.v1",
        "mode": mode,
        "records": records,
        "causal_diagnostics": causal,
        "clipping_operational": clipping_operational,
        "clipping_classification": "PPO_CLIPPING_CAUSAL_ROLE_TESTED" if clipping_operational else "CLIP_COEFFICIENT_NOT_OPERATIONAL",
        "learning_rate_classification": "LEARNING_RATE_CAUSAL_ROLE_TESTED_AT_DEVELOPMENT_SCALE",
        "one_factor_at_a_time": True,
        "paper_time_constant_used_as_optimization_label": False,
        "qec_cycle_budgets_matched": len(budgets) == 1,
        **evidence_envelope(complete=True, mechanism_valid=len(budgets) == 1, claim_supported=False, paper_comparable=False, blocking_reasons=["HELD_OUT_ABLATION_REQUIRED"]),
    }
    return write_artifact("step_response/ablation_results", payload, "Step-response Causal Ablations")


def analyse_step_response() -> dict[str, Any]:
    result_path = artifact_root() / "step_response" / "report.json"
    ablation_path = artifact_root() / "step_response" / "ablation_results.json"
    if not result_path.is_file() or not ablation_path.is_file():
        raise RuntimeError("step response and ablation artifacts are both required")
    result = read_json(result_path)
    ablation = read_json(ablation_path)
    return {
        "response": result["response"],
        "clipping_classification": ablation["clipping_classification"],
        "learning_rate_classification": ablation["learning_rate_classification"],
        "qec_cycle_budgets_matched": ablation["qec_cycle_budgets_matched"],
        "held_out_causal_claim": False,
    }
