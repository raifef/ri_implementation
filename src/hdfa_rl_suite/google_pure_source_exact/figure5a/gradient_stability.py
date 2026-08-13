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


_PHYSICAL_DOMAIN_ERROR = "gate depolarization probability left the frozen physical range"


def _is_physical_domain_error(error: ValueError) -> bool:
    return _PHYSICAL_DOMAIN_ERROR in str(error)


def gradient_stability_conditions(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    ladder = config["gradient_stability_ladder"]
    controller = config["controller"]
    clipping = controller["gradient_clipping"]
    conditions = []
    derived_thresholds = clipping.get("derived_candidate_thresholds")
    if derived_thresholds is None:
        rates = {
            "mean": controller["mean_learning_rate"],
            "sigma": controller["sigma_learning_rate"],
            "label": "current_provisional_for_unclipped_pilot",
        }
        iterator = (
            (ladder["qec_cycles_per_candidate"][0], seed, rates, entropy_weight,
             clipping["pilot_mode"], None)
            for seed, entropy_weight in product(
                ladder["development_seeds"], ladder["entropy_weights"])
        )
        stage = "shadow_unclipped_gradient_norm_pilot"
    else:
        iterator = product(
            [ladder["qec_cycles_per_candidate"][0]],
            [ladder["development_seeds"][0]],
            controller["learning_rate_candidates"], ladder["entropy_weights"],
            clipping["candidate_modes"], derived_thresholds)
        stage = "two_thousand_cycle_instability_screen"
    for cycles, seed, rates, entropy_weight, mode, threshold in iterator:
        condition = {
            "successive_elimination_stage": stage,
            "qec_cycles_per_candidate": int(cycles), "seed": int(seed),
            "mean_learning_rate": float(rates["mean"]),
            "sigma_learning_rate": float(rates["sigma"]),
            "entropy_weight": float(entropy_weight),
            "learning_rate_label": str(rates["label"]),
            "gradient_clipping_mode": str(mode),
            "gradient_clip_threshold": None if threshold is None else float(threshold),
        }
        condition["condition_id"] = canonical_hash(condition)[:20]
        conditions.append(condition)
    return conditions


def plan_gradient_stability(config: Mapping[str, Any]) -> dict[str, Any]:
    ladder = config["gradient_stability_ladder"]
    clipping = config["controller"]["gradient_clipping"]
    conditions = gradient_stability_conditions(config)
    epochs = int(ladder["stationary_epochs"]) + int(ladder["step_epochs"])
    candidates = int(ladder["candidates_per_epoch"])
    rows = [{
        "condition_index": index, **condition,
        "candidate_qec_cycles": epochs * candidates * condition["qec_cycles_per_candidate"],
    } for index, condition in enumerate(conditions)]
    initial_sigma = float(config["controller"]["initial_sigma"])
    dimension = 41
    entropy_audit = []
    audit_thresholds = (
        clipping["legacy_absolute_thresholds_for_comparison_only"]
        if clipping.get("derived_candidate_thresholds") is None
        else clipping["derived_candidate_thresholds"])
    for entropy_weight, mode, threshold in product(
            ladder["entropy_weights"],
            config["controller"]["gradient_clipping"]["candidate_modes"],
            audit_thresholds):
        per_coordinate = float(entropy_weight) / initial_sigma
        norm = np.sqrt(dimension) * per_coordinate
        if mode == GradientClippingMode.GLOBAL_L2.value:
            sigma_scale = min(1.0, float(threshold) / norm)
            mean_scale_when_joint_gradient_present = sigma_scale
        elif mode == GradientClippingMode.PER_BLOCK_GLOBAL_L2.value:
            sigma_scale = min(1.0, float(threshold) / norm)
            mean_scale_when_joint_gradient_present = 1.0
        else:
            sigma_scale = min(1.0, float(threshold) / per_coordinate)
            mean_scale_when_joint_gradient_present = 1.0
        entropy_audit.append({
            "entropy_weight": float(entropy_weight), "gradient_clipping_mode": str(mode),
            "gradient_clip_threshold": float(threshold),
            "entropy_only_sigma_gradient_l2_norm": float(norm),
            "entropy_only_sigma_clip_scale": float(sigma_scale),
            "mean_gradient_scale_due_to_entropy_only_sigma_block":
                float(mean_scale_when_joint_gradient_present),
        })
    plant = build_plant(config)
    target = np.zeros(plant.control_count)
    base_edr = plant.exact_global_edr(
        target, epoch=0, frequency=1 / 1000, target_controls=target)
    fixed_peak_target = np.ones(plant.control_count)
    fixed_peak_edr = plant.exact_global_edr(
        target, epoch=0, frequency=1 / 1000,
        target_controls=fixed_peak_target)
    curvature_probe = float(config["figure_s8_plant_calibration"][
        "group_curvature_probe_sigma"])
    curvatures = []
    for coordinate in range(plant.control_count):
        plus = target.copy(); plus[coordinate] = curvature_probe
        minus = target.copy(); minus[coordinate] = -curvature_probe
        curvatures.append((
            plant.exact_global_edr(
                plus, epoch=0, frequency=1 / 1000, target_controls=target)
            + plant.exact_global_edr(
                minus, epoch=0, frequency=1 / 1000, target_controls=target)
            - 2.0 * base_edr) / (2.0 * curvature_probe**2))
    curvature_sum = float(np.sum(curvatures))
    fixed_gap = float(fixed_peak_edr - base_edr)
    exploration = config["controller"]["initial_exploration"]
    physical_sigma_candidates = [{
        "fixed_to_oracle_gap_fraction": float(fraction),
        "derived_initial_sigma": float(np.sqrt(
            float(fraction) * fixed_gap / curvature_sum)),
    } for fraction in exploration["candidate_gap_fractions"]]
    baseline = config["controller"]["baseline_dynamics"]
    baseline_weight = float(config["controller"]["baseline_weight"])
    baseline_candidates = [{
        "effective_update_rate_alpha_b": float(alpha),
        "derived_baseline_learning_rate_at_fixed_weight":
            float(alpha) / (2.0 * baseline_weight),
        "nominal_response_time_epochs": 1.0 / float(alpha),
    } for alpha in baseline["candidate_effective_update_rates"]]
    return {
        "schema_version": "figure5a-gradient-stability-plan.v1",
        "scientific_status": ladder["scientific_status"],
        "selection_rule": ladder["selection_rule"],
        "candidates_per_epoch": candidates, "stationary_epochs": int(ladder["stationary_epochs"]),
        "step_epochs": int(ladder["step_epochs"]), "condition_count": len(rows),
        "conditions": rows, "certification_seeds_consumed": False,
        "successive_elimination_stage": rows[0]["successive_elimination_stage"],
        "derived_candidate_thresholds_available":
            clipping.get("derived_candidate_thresholds") is not None,
        "entropy_weights": [float(value) for value in ladder["entropy_weights"]],
        "initial_entropy_only_clipping_audit": entropy_audit,
        "physical_initial_sigma_candidates": physical_sigma_candidates,
        "current_initial_sigma_implied_gap_fraction": float(
            curvature_sum * initial_sigma**2 / fixed_gap),
        "exact_coordinate_curvature_sum": curvature_sum,
        "exact_fixed_to_oracle_edr_gap": fixed_gap,
        "baseline_effective_dynamics_candidates": baseline_candidates,
        "current_baseline_effective_update_rate": float(
            2.0 * config["controller"]["baseline_learning_rate"] * baseline_weight),
        "source_candidate_count_preserved": candidates == 50,
        "long_runs_not_launched_by_plan": True,
        "warning": ("Run only the nine shadow-mode unclipped pilot conditions, summarize their gradient-norm quantiles, then preregister derived thresholds before any geometry screen. No optimizer updates are applied in this stage."
                    if clipping.get("derived_candidate_thresholds") is None else
                    "Run the 2k screen, preregister survivors, promote them to 10k, and promote only finalists to 36k; do not select on certification seeds."),
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
        gradient_clip_threshold=(
            None if condition["gradient_clip_threshold"] is None
            else float(condition["gradient_clip_threshold"])))
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
    shadow_gradient_pilot = condition["successive_elimination_stage"] == \
        "shadow_unclipped_gradient_norm_pilot"
    terminal_failure = state.get("terminal_failure")

    def record_physical_domain_failure(kind: str, epoch: int, error: ValueError) -> None:
        nonlocal terminal_failure
        terminal_failure = {
            "kind": kind,
            "epoch": int(epoch),
            "message": str(error),
            "classification": "SCIENTIFIC_INSTABILITY_NOT_INFRASTRUCTURE_FAILURE",
        }
        state["terminal_failure"] = terminal_failure
        state["policy"] = policy.state_dict(
            optimizer_state=optimizer.state_dict(), baseline=baseline)
        atomic_json(checkpoint_path, state)

    completed_this_call = 0
    while int(state["next_epoch"]) < total_epochs and terminal_failure is None:
        epoch = int(state["next_epoch"])
        phase = "stationary" if epoch < stationary_epochs else "step"
        target_value = float(ladder["stationary_target"] if phase == "stationary"
                             else ladder["step_target"])
        target = np.full(plant.control_count, target_value)
        batch = policy.sample(candidates)
        try:
            observations = [plant.sample_detector_observation(
                action, epoch=epoch, frequency=float(config["anchor"]["frequency"]),
                qec_cycles=cycles,
                seed=plant.stream_seed(
                    int(condition["seed"]), "gradient-stability", epoch, index),
                target_controls=target) for index, action in enumerate(batch.actions)]
        except ValueError as error:
            if not _is_physical_domain_error(error):
                raise
            record_physical_domain_failure(
                "sampled_candidate_left_physical_domain", epoch, error)
            break
        rewards = -np.asarray([item.reward_rates for item in observations])
        loss = total_loss_and_gradients(
            batch.actions, rewards, plant.mask, policy.mean, policy.sigma, baseline, batch.behavior,
            clip=float(controller["ppo_clip"]), baseline_weight=float(controller["baseline_weight"]),
            entropy_weight=float(condition["entropy_weight"]))
        diagnostic = None
        if epoch in diagnostic_epochs:
            try:
                exact_rewards = -np.asarray([plant.expected_reward_rates(
                    action, epoch=epoch, frequency=float(config["anchor"]["frequency"]),
                    target_controls=target) for action in batch.actions])
            except ValueError as error:
                if not _is_physical_domain_error(error):
                    raise
                record_physical_domain_failure(
                    "diagnostic_candidate_left_physical_domain", epoch, error)
                break
            finite_gradient = total_loss_and_gradients(
                batch.actions, rewards, plant.mask, policy.mean, policy.sigma, baseline, batch.behavior,
                clip=float(controller["ppo_clip"]), baseline_weight=0.0, entropy_weight=0.0)
            exact_gradient = total_loss_and_gradients(
                batch.actions, exact_rewards, plant.mask, policy.mean, policy.sigma, baseline, batch.behavior,
                clip=float(controller["ppo_clip"]), baseline_weight=0.0, entropy_weight=0.0)
            diagnostic = _gradient_agreement(finite_gradient, exact_gradient)
        try:
            before_edr = plant.expected_global_edr(
                policy.mean, epoch=epoch, frequency=float(config["anchor"]["frequency"]),
                target_controls=target)
        except ValueError as error:
            if not _is_physical_domain_error(error):
                raise
            record_physical_domain_failure(
                "pre_update_mean_left_physical_domain", epoch, error)
            break
        if shadow_gradient_pilot:
            update = {
                **optimizer.diagnose_gradients(
                    loss.grad_mean, loss.grad_sigma, loss.grad_baseline),
                "fraction_at_positivity_guard": 0.0,
                "fraction_at_sigma_min": float(np.mean(
                    policy.sigma <= optimizer_config.minimum_sigma)),
                "fraction_at_sigma_max": float(np.mean(
                    policy.sigma >= optimizer_config.maximum_sigma)),
                "unclipped_sigma_min": float(np.min(policy.sigma)),
                "unclipped_sigma_max": float(np.max(policy.sigma)),
                "backtracks": 0,
                "optimized_scale_variable": "sigma",
            }
        else:
            update = optimizer.step(
                policy.mean, policy.sigma, baseline,
                loss.grad_mean, loss.grad_sigma, loss.grad_baseline)
            policy.policy_version += 1
        try:
            after_edr = plant.expected_global_edr(
                policy.mean, epoch=epoch, frequency=float(config["anchor"]["frequency"]),
                target_controls=target)
        except ValueError as error:
            if not _is_physical_domain_error(error):
                raise
            record_physical_domain_failure(
                "post_update_mean_left_physical_domain", epoch, error)
            break
        fixed_edr = plant.expected_global_edr(
            np.zeros(plant.control_count), epoch=epoch,
            frequency=float(config["anchor"]["frequency"]), target_controls=target)
        oracle_edr = plant.expected_global_edr(
            target, epoch=epoch, frequency=float(config["anchor"]["frequency"]),
            target_controls=target)
        denominator = fixed_edr - oracle_edr
        record = {
            "epoch": epoch, "phase": phase, "target": target_value,
            "optimizer_update_applied": not shadow_gradient_pilot,
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
            "fraction_at_sigma_min": update["fraction_at_sigma_min"],
            "fraction_at_sigma_max": update["fraction_at_sigma_max"],
            "unclipped_sigma_min": update["unclipped_sigma_min"],
            "unclipped_sigma_max": update["unclipped_sigma_max"],
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
    scientifically_invalid = terminal_failure is not None
    result = {
        "schema_version": "figure5a-gradient-stability-condition.v1", "complete": True,
        "scientific_status": ladder["scientific_status"], "condition_index": condition_index,
        "condition": condition, "plant_hash": plant.plant_hash,
        "scientifically_invalid": scientifically_invalid,
        "terminal_failure": terminal_failure,
        "optimizer_updates_applied": not shadow_gradient_pilot,
        "shadow_gradient_observation_only": shadow_gradient_pilot,
        "source_candidate_count_preserved": candidates == 50,
        "certification_seed_used": int(condition["seed"]) in config["seed_registry"]["certification_reserved"],
        "records": records,
        "summary": {
            "all_gradients_finite": bool(records) and all(np.isfinite([
                row["post_update_mean_l2_error"], row["post_update_mean_sigma"],
                row["gradient_clipping"]["gradient_global_l2_norm_before_clipping"]]).all()
                for row in records),
            "harmful_update_fraction": (None if shadow_gradient_pilot or not records else float(np.mean([
                not row["update_reduced_current_target_EDR"] for row in records]))),
            "clipped_epoch_fraction": (None if not records else float(np.mean([
                row["gradient_clipping"]["gradient_global_clip_scale"] < 1.0 or
                row["gradient_clipping"]["gradient_clipped_component_count"] > 0
                for row in records]))),
            "maximum_epoch_sigma_min_fraction": (None if not records else float(max(
                row["fraction_at_sigma_min"] for row in records))),
            "maximum_epoch_sigma_max_fraction": (None if not records else float(max(
                row["fraction_at_sigma_max"] for row in records))),
            "minimum_unclipped_sigma": (None if not records else float(min(
                row["unclipped_sigma_min"] for row in records))),
            "maximum_unclipped_sigma": (None if not records else float(max(
                row["unclipped_sigma_max"] for row in records))),
            "final_step_mean_r": None if not records else records[-1]["post_update_mean_r"],
            "final_mean_sigma": None if not records else records[-1]["post_update_mean_sigma"],
            "candidate_qec_cycles": sum(row["candidate_qec_cycles"] for row in records),
            "physical_domain_exit": scientifically_invalid,
        },
    }
    result["result_hash"] = canonical_hash(result)
    return result


def summarize_gradient_stability(config: Mapping[str, Any], result_directory: Path) -> dict[str, Any]:
    plan = plan_gradient_stability(config)
    expected_conditions = {
        str(row["condition_id"]): dict(row)
        for row in plan["conditions"]}
    rows = []
    for path in sorted(result_directory.glob("*.json")):
        value = __import__("json").loads(path.read_text(encoding="utf-8"))
        condition = value.get("condition", {})
        condition_id = str(condition.get("condition_id", ""))
        if (value.get("schema_version") == "figure5a-gradient-stability-condition.v1"
                and value.get("complete")
                and condition_id in expected_conditions
                and condition == expected_conditions[condition_id]):
            rows.append({"path": str(path.resolve()), "condition_index": value["condition_index"],
                         "condition": value["condition"], "summary": value["summary"]})
    pilot_values: dict[str, list[float]] = {
        "mean": [], "sigma": [], "baseline": []}
    pilot_values_by_entropy: dict[str, dict[str, list[float]]] = {}
    completed_pilot_ids: set[str] = set()
    for path in sorted(result_directory.glob("*.json")):
        value = __import__("json").loads(path.read_text(encoding="utf-8"))
        condition = value.get("condition", {})
        condition_id = str(condition.get("condition_id", ""))
        if (value.get("schema_version") != "figure5a-gradient-stability-condition.v1"
                or not value.get("complete") or
                condition_id not in expected_conditions
                or condition != expected_conditions[condition_id]
                or
                condition.get("successive_elimination_stage") !=
                "shadow_unclipped_gradient_norm_pilot"):
            continue
        completed_pilot_ids.add(str(condition["condition_id"]))
        entropy = str(condition["entropy_weight"])
        entropy_values = pilot_values_by_entropy.setdefault(
            entropy, {"mean": [], "sigma": [], "baseline": []})
        for record in value["records"]:
            clipping = record["gradient_clipping"]
            for block in pilot_values:
                norm = float(clipping[f"raw_{block}_gradient_l2_norm"])
                pilot_values[block].append(norm)
                entropy_values[block].append(norm)
    quantiles = [float(value) for value in config["controller"][
        "gradient_clipping"]["pilot_quantiles"]]

    def quantile_table(values: dict[str, list[float]]) -> dict[str, dict[str, float]]:
        return {
            block: {str(quantile): float(np.quantile(samples, quantile))
                    for quantile in quantiles}
            for block, samples in values.items() if samples
        }

    expected_pilot_ids = {
        str(row["condition_id"]) for row in plan["conditions"]
        if row["successive_elimination_stage"] ==
        "shadow_unclipped_gradient_norm_pilot"}
    pilot_complete = bool(expected_pilot_ids) and \
        completed_pilot_ids == expected_pilot_ids
    result = {
        "schema_version": "figure5a-gradient-stability-summary.v1",
        "scientific_status": "DEVELOPMENT_ONLY_NOT_REFERENCE_EVIDENCE",
        "plan_hash": plan["plan_hash"], "completed_condition_count": len(rows),
        "planned_condition_count": plan["condition_count"], "rows": rows,
        "automatic_hyperparameter_selection_performed": False,
        "unclipped_pilot_complete_across_entropy_anchors": pilot_complete,
        "observed_gradient_norm_quantiles": quantile_table(pilot_values),
        "observed_gradient_norm_quantiles_by_entropy": {
            entropy: quantile_table(values)
            for entropy, values in pilot_values_by_entropy.items()},
        "threshold_preregistration_instruction": (
            "Use the observed per-block 90th/95th/99th percentiles to preregister "
            "separate block thresholds. Do not automatically select a threshold "
            "from the same trajectories or certification seeds."),
        "selection_instruction": config["gradient_stability_ladder"]["selection_rule"],
        "certification_seeds_consumed": any(
            row["condition"]["seed"] in config["seed_registry"]["certification_reserved"] for row in rows),
    }
    return result
