"""Development-only learning-rate and gradient-clipping ladder for Figure 5a."""
from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.gaussian import (
    DirectSigmaGaussianPolicy,
)
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.losses import (
    total_loss_and_gradients,
)
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.optimizer import (
    DirectSigmaOptimizer,
    GradientClippingMode,
    OptimizerConfig,
)

from .contracts import atomic_json, canonical_hash
from .validation import build_plant


def gradient_stability_conditions(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    ladder = config["gradient_stability_ladder"]
    controller = config["controller"]
    clipping = controller["gradient_clipping"]
    conditions = []
    for cycles, seed, rates, mode, threshold in product(
        ladder["qec_cycles_per_candidate"], ladder["development_seeds"],
        controller["learning_rate_candidates"], clipping["candidate_modes"],
        clipping["candidate_thresholds"],
    ):
        condition = {
            "qec_cycles_per_candidate": int(cycles), "seed": int(seed),
            "mean_learning_rate": float(rates["mean"]),
            "sigma_learning_rate": float(rates["sigma"]),
            "learning_rate_label": str(rates["label"]),
            "gradient_clipping_mode": str(mode),
            "gradient_clip_threshold": float(threshold),
        }
        condition["condition_id"] = canonical_hash(condition)[:20]
        conditions.append(condition)
    return conditions


def plan_gradient_stability(config: Mapping[str, Any]) -> dict[str, Any]:
    ladder = config["gradient_stability_ladder"]
    conditions = gradient_stability_conditions(config)
    epochs = int(ladder["stationary_epochs"]) + int(ladder["step_epochs"])
    candidates = int(ladder["candidates_per_epoch"])
    rows = [{
        "condition_index": index, **condition,
        "candidate_qec_cycles": epochs * candidates * condition["qec_cycles_per_candidate"],
    } for index, condition in enumerate(conditions)]
    return {
        "schema_version": "figure5a-gradient-stability-plan.v1",
        "scientific_status": ladder["scientific_status"],
        "selection_rule": ladder["selection_rule"],
        "candidates_per_epoch": candidates, "stationary_epochs": int(ladder["stationary_epochs"]),
        "step_epochs": int(ladder["step_epochs"]), "condition_count": len(rows),
        "conditions": rows, "certification_seeds_consumed": False,
        "source_candidate_count_preserved": candidates == 50,
        "long_runs_not_launched_by_plan": True,
        "warning": "Run a preregistered subset at 2k first, promote survivors to 10k, then promote finalists to 36k; do not select on certification seeds.",
        "plan_hash": canonical_hash({"ladder": ladder, "controller_candidates": {
            "learning_rates": config["controller"]["learning_rate_candidates"],
            "clipping": config["controller"]["gradient_clipping"]}}),
    }


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return None if denominator == 0 else float(np.dot(left, right) / denominator)


def _gradient_agreement(finite: Any, exact: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, finite_value, exact_value in (
        ("mu", finite.grad_mean, exact.grad_mean), ("sigma", finite.grad_sigma, exact.grad_sigma)):
        finite_array, exact_array = np.asarray(finite_value), np.asarray(exact_value)
        result[f"grad_{name}_cosine"] = _cosine(finite_array, exact_array)
        result[f"grad_{name}_relative_l2_error"] = float(
            np.linalg.norm(finite_array - exact_array) /
            max(np.linalg.norm(exact_array), np.finfo(float).tiny))
        result[f"finite_grad_{name}_l2_norm"] = float(np.linalg.norm(finite_array))
        result[f"exact_grad_{name}_l2_norm"] = float(np.linalg.norm(exact_array))
    return result


def run_gradient_stability_condition(
    config: Mapping[str, Any], *, condition_index: int, checkpoint_path: Path,
    resume: bool = False, max_epochs: int | None = None,
) -> dict[str, Any]:
    """Run one 50-candidate stationary-then-step development trajectory."""
    plan = plan_gradient_stability(config)
    if not 0 <= condition_index < len(plan["conditions"]):
        raise ValueError("gradient-stability condition index is outside the frozen grid")
    condition = dict(plan["conditions"][condition_index])
    plant = build_plant(config)
    ladder = config["gradient_stability_ladder"]
    controller = config["controller"]
    candidates = int(ladder["candidates_per_epoch"])
    cycles = int(condition["qec_cycles_per_candidate"])
    if cycles % plant.rounds:
        raise ValueError("gradient-stability QEC cycles must divide into whole Stim shots")
    optimizer_config = OptimizerConfig(
        condition["mean_learning_rate"], condition["sigma_learning_rate"],
        float(controller["baseline_learning_rate"]),
        minimum_sigma=float(controller["minimum_sigma"]),
        maximum_sigma=float(controller["maximum_sigma"]),
        positivity_guard=controller["positivity_guard"],
        gradient_clipping_mode=GradientClippingMode(condition["gradient_clipping_mode"]),
        gradient_clip_threshold=float(condition["gradient_clip_threshold"]))
    identity = {
        "plan_hash": plan["plan_hash"], "condition": condition,
        "plant_hash": plant.plant_hash,
    }
    if checkpoint_path.exists():
        if not resume:
            raise RuntimeError(f"checkpoint exists; pass resume=True: {checkpoint_path}")
        state = __import__("json").loads(checkpoint_path.read_text(encoding="utf-8"))
        if state["identity"] != identity:
            raise RuntimeError("gradient-stability checkpoint identity changed")
        policy = DirectSigmaGaussianPolicy.from_state_dict(state["policy"])
        optimizer = DirectSigmaOptimizer.from_state_dict(state["policy"]["optimizer_state"])
        baseline = np.asarray(state["policy"]["baseline"], dtype=float)
    else:
        policy = DirectSigmaGaussianPolicy(
            np.zeros(plant.control_count),
            np.full(plant.control_count, float(controller["initial_sigma"])),
            seed=int(condition["seed"]))
        optimizer = DirectSigmaOptimizer(plant.control_count, plant.detector_count, optimizer_config)
        baseline = np.zeros(plant.detector_count)
        state = {"schema_version": "figure5a-gradient-stability-checkpoint.v1",
                 "identity": identity, "condition_index": condition_index,
                 "next_epoch": 0, "records": [],
                 "policy": policy.state_dict(optimizer_state=optimizer.state_dict(), baseline=baseline)}
        atomic_json(checkpoint_path, state)
    stationary_epochs = int(ladder["stationary_epochs"])
    total_epochs = stationary_epochs + int(ladder["step_epochs"])
    diagnostic_epochs = {stationary_epochs - 1, stationary_epochs, total_epochs - 1}
    completed_this_call = 0
    while int(state["next_epoch"]) < total_epochs:
        epoch = int(state["next_epoch"])
        phase = "stationary" if epoch < stationary_epochs else "step"
        target_value = float(ladder["stationary_target"] if phase == "stationary"
                             else ladder["step_target"])
        target = np.full(plant.control_count, target_value)
        batch = policy.sample(candidates)
        observations = [plant.sample_detector_observation(
            action, epoch=epoch, frequency=float(config["anchor"]["frequency"]),
            qec_cycles=cycles,
            seed=plant.stream_seed(int(condition["seed"]), "gradient-stability", epoch, index),
            target_controls=target) for index, action in enumerate(batch.actions)]
        rewards = -np.asarray([item.reward_rates for item in observations])
        loss = total_loss_and_gradients(
            batch.actions, rewards, plant.mask, policy.mean, policy.sigma, baseline, batch.behavior,
            clip=float(controller["ppo_clip"]), baseline_weight=float(controller["baseline_weight"]),
            entropy_weight=float(ladder["entropy_weight"]))
        diagnostic = None
        if epoch in diagnostic_epochs:
            exact_rewards = -np.asarray([plant.expected_reward_rates(
                action, epoch=epoch, frequency=float(config["anchor"]["frequency"]),
                target_controls=target) for action in batch.actions])
            finite_gradient = total_loss_and_gradients(
                batch.actions, rewards, plant.mask, policy.mean, policy.sigma, baseline, batch.behavior,
                clip=float(controller["ppo_clip"]), baseline_weight=0.0, entropy_weight=0.0)
            exact_gradient = total_loss_and_gradients(
                batch.actions, exact_rewards, plant.mask, policy.mean, policy.sigma, baseline, batch.behavior,
                clip=float(controller["ppo_clip"]), baseline_weight=0.0, entropy_weight=0.0)
            diagnostic = _gradient_agreement(finite_gradient, exact_gradient)
        before_edr = plant.expected_global_edr(
            policy.mean, epoch=epoch, frequency=float(config["anchor"]["frequency"]),
            target_controls=target)
        update = optimizer.step(
            policy.mean, policy.sigma, baseline,
            loss.grad_mean, loss.grad_sigma, loss.grad_baseline)
        policy.policy_version += 1
        after_edr = plant.expected_global_edr(
            policy.mean, epoch=epoch, frequency=float(config["anchor"]["frequency"]),
            target_controls=target)
        fixed_edr = plant.expected_global_edr(
            np.zeros(plant.control_count), epoch=epoch,
            frequency=float(config["anchor"]["frequency"]), target_controls=target)
        oracle_edr = plant.expected_global_edr(
            target, epoch=epoch, frequency=float(config["anchor"]["frequency"]),
            target_controls=target)
        denominator = fixed_edr - oracle_edr
        record = {
            "epoch": epoch, "phase": phase, "target": target_value,
            "candidate_qec_cycles": candidates * cycles,
            "candidate_EDR": float(sum(item.raw_total for item in observations) /
                                   ((candidates * cycles // plant.rounds) * plant.raw_detector_count)),
            "mean_EDR_before_update": before_edr, "mean_EDR_after_update": after_edr,
            "update_reduced_current_target_EDR": after_edr <= before_edr,
            "post_update_mean_r": None if denominator <= 0 else float((fixed_edr - after_edr) / denominator),
            "post_update_mean_l2_error": float(np.linalg.norm(policy.mean - target)),
            "post_update_mean_sigma": float(np.mean(policy.sigma)),
            "post_update_max_abs_mean": float(np.max(np.abs(policy.mean))),
            "gradient_diagnostic": diagnostic,
            "gradient_clipping": {key: update[key] for key in update if key.startswith("gradient_")
                                  or key.startswith("raw_") or key.startswith("applied_")},
            "fraction_at_positivity_guard": update["fraction_at_positivity_guard"],
        }
        state["records"].append(record)
        state["next_epoch"] = epoch + 1
        state["policy"] = policy.state_dict(optimizer_state=optimizer.state_dict(), baseline=baseline)
        atomic_json(checkpoint_path, state)
        completed_this_call += 1
        if max_epochs is not None and completed_this_call >= max_epochs:
            return {"complete": False, "checkpoint_path": str(checkpoint_path.resolve()),
                    "next_epoch": state["next_epoch"], "condition": condition}
    records = state["records"]
    result = {
        "schema_version": "figure5a-gradient-stability-condition.v1", "complete": True,
        "scientific_status": ladder["scientific_status"], "condition_index": condition_index,
        "condition": condition, "plant_hash": plant.plant_hash,
        "source_candidate_count_preserved": candidates == 50,
        "certification_seed_used": int(condition["seed"]) in config["seed_registry"]["certification_reserved"],
        "records": records,
        "summary": {
            "all_gradients_finite": all(np.isfinite([
                row["post_update_mean_l2_error"], row["post_update_mean_sigma"],
                row["gradient_clipping"]["gradient_global_l2_norm_before_clipping"]]).all()
                for row in records),
            "harmful_update_fraction": float(np.mean([
                not row["update_reduced_current_target_EDR"] for row in records])),
            "clipped_epoch_fraction": float(np.mean([
                row["gradient_clipping"]["gradient_global_clip_scale"] < 1.0 or
                row["gradient_clipping"]["gradient_clipped_component_count"] > 0
                for row in records])),
            "final_step_mean_r": records[-1]["post_update_mean_r"],
            "final_mean_sigma": records[-1]["post_update_mean_sigma"],
            "candidate_qec_cycles": sum(row["candidate_qec_cycles"] for row in records),
        },
    }
    result["result_hash"] = canonical_hash(result)
    return result


def summarize_gradient_stability(config: Mapping[str, Any], result_directory: Path) -> dict[str, Any]:
    plan = plan_gradient_stability(config)
    rows = []
    for path in sorted(result_directory.glob("*.json")):
        value = __import__("json").loads(path.read_text(encoding="utf-8"))
        if value.get("schema_version") == "figure5a-gradient-stability-condition.v1" and value.get("complete"):
            rows.append({"path": str(path.resolve()), "condition_index": value["condition_index"],
                         "condition": value["condition"], "summary": value["summary"]})
    return {
        "schema_version": "figure5a-gradient-stability-summary.v1",
        "scientific_status": "DEVELOPMENT_ONLY_NOT_REFERENCE_EVIDENCE",
        "plan_hash": plan["plan_hash"], "completed_condition_count": len(rows),
        "planned_condition_count": plan["condition_count"], "rows": rows,
        "automatic_hyperparameter_selection_performed": False,
        "selection_instruction": config["gradient_stability_ladder"]["selection_rule"],
        "certification_seeds_consumed": any(
            row["condition"]["seed"] in config["seed_registry"]["certification_reserved"] for row in rows),
    }
