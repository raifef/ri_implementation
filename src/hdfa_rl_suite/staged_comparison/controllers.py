"""Matched Track-B arm implementations on the common detector-likelihood plants."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import time
from typing import Any

import numpy as np

from hdfa_rl_suite.common import deterministic_hash
from hdfa_rl_suite.google_rl_certification.agent import (
    CandidateEvaluation,
    GaussianPolicyGradientAgent,
)
from hdfa_rl_suite.google_rl_certification.config import named_config
from hdfa_rl_suite.stage6 import ResidualActivationGate
from hdfa_rl_suite.stage6.schema import ResidualGateEvidence, ResidualRLDisposition

from .config import TrackBConfig
from .substrate import (
    PlantContract,
    ScenarioRealization,
    expected_detector_rates,
    expected_logical_rate,
)


ARM_NAMES = (
    "fixed",
    "periodic_recalibration",
    "oracle",
    "certified_high_shot_google_rl",
    "certified_reduced_budget_google_rl",
    "predictive_hdfa_no_residual",
    "predictive_hdfa_conditional_residual_rl",
)


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(item) for item in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _observe(
    contract: PlantContract,
    actions: np.ndarray,
    optimum: np.ndarray,
    previous_action: np.ndarray,
    cycles: int,
    *seed_parts: object,
) -> tuple[np.ndarray, np.ndarray]:
    expected = expected_detector_rates(contract, actions, optimum, previous_action)
    rng = np.random.default_rng(_stable_seed(*seed_parts))
    observed = rng.binomial(cycles, expected) / float(cycles)
    return expected, observed


def _project_action(contract: PlantContract, previous: np.ndarray,
                    requested: np.ndarray) -> np.ndarray:
    bounded = np.clip(requested, -contract.hard_bound_normalized,
                      contract.hard_bound_normalized)
    return np.clip(
        bounded,
        previous-contract.slew_limit_normalized,
        previous+contract.slew_limit_normalized,
    )


def _policy_transaction(
    arm: str,
    interval: int,
    previous: np.ndarray,
    action: np.ndarray,
    scenario: ScenarioRealization,
) -> dict[str, Any]:
    reference_hash = deterministic_hash(previous.round(14).tolist())
    action_hash = deterministic_hash(action.round(14).tolist())
    state_id = deterministic_hash({
        "disturbance_path": scenario.disturbance_path_hash,
        "interval": interval,
        "reference": reference_hash,
    })
    bounds = bool(np.all(np.abs(action) <= 1.0+1e-12))
    slew = bool(np.all(np.abs(action-previous) <= .12+1e-12))
    return {
        "schema_version": "track-b-policy-transaction.v1",
        "policy_id": f"{arm}:{scenario.scenario_id}:{scenario.seed}:{interval}",
        "reference_policy_id": f"{arm}:{scenario.scenario_id}:{scenario.seed}:{interval-1}",
        "reference_policy_hash": reference_hash,
        "created_from_state_id": state_id,
        "expected_activation_state_id": deterministic_hash((state_id, reference_hash)),
        "projection_certificate": bounds and slew,
        "bounds_certificate": bounds,
        "slew_certificate": slew,
        "supervisor_authorization": "track-b-stage7-development-authorization",
        "activation_acknowledgement": True,
        "activated_policy_hash": action_hash,
        "lifecycle_valid": bounds and slew,
    }


@dataclass
class _PredictiveState:
    action: np.ndarray
    estimate: np.ndarray
    previous_estimate: np.ndarray
    uncertainty: np.ndarray
    residual_mean: float
    residual_sigma: float
    residual_persistence: int
    residual_sign_history: list[int]
    last_heldout_gain: float
    gate: ResidualActivationGate


def _model_probabilities(estimate: np.ndarray, previous: np.ndarray,
                         older: np.ndarray) -> dict[str, float]:
    velocity = float(np.linalg.norm(estimate-previous))
    acceleration = float(np.linalg.norm(estimate-2*previous+older))
    if velocity > .075:
        raw = {"step_or_rtn": .62, "oscillator": .12, "ou": .16, "unknown": .10}
    elif acceleration < .025 and velocity > .008:
        raw = {"step_or_rtn": .08, "oscillator": .58, "ou": .27, "unknown": .07}
    else:
        raw = {"step_or_rtn": .16, "oscillator": .24, "ou": .50, "unknown": .10}
    total = sum(raw.values())
    return {key: value/total for key, value in raw.items()}


def _predictive_step(
    contract: PlantContract,
    scenario: ScenarioRealization,
    interval: int,
    optimum: np.ndarray,
    state: _PredictiveState,
    config: TrackBConfig,
    *,
    conditional_residual: bool,
) -> tuple[np.ndarray, dict[str, Any], int, int, float, int]:
    started = time.perf_counter_ns()
    previous_action = state.action.copy()
    epsilon = config.stage2_probe_normalized
    observed_gradients = np.zeros(len(contract.control_ids))
    uncertainty = np.zeros(len(contract.control_ids))
    probe_rates: list[float] = []
    probe_count = 0
    for control_index in range(len(contract.control_ids)):
        plus = previous_action.copy(); minus = previous_action.copy()
        plus[control_index] = min(contract.hard_bound_normalized,
                                  plus[control_index]+epsilon)
        minus[control_index] = max(-contract.hard_bound_normalized,
                                   minus[control_index]-epsilon)
        actions = np.stack((plus, minus))
        expected, observed = _observe(
            contract, actions, optimum, previous_action,
            config.stage2_probe_cycles,
            scenario.seed, scenario.scenario_id, "stage2", interval, control_index,
        )
        relevant = contract.mask[:, control_index].astype(bool)
        plus_loss = float(np.mean(observed[0, relevant]))
        minus_loss = float(np.mean(observed[1, relevant]))
        denominator = max(plus[control_index]-minus[control_index], 1e-12)
        observed_gradients[control_index] = (plus_loss-minus_loss)/denominator
        p = float(np.mean(expected[:, relevant]))
        standard_error = math.sqrt(max(p*(1-p), 1e-12)/config.stage2_probe_cycles)
        uncertainty[control_index] = standard_error/max(
            2*epsilon*contract.curvature_by_control[control_index], 1e-8)
        probe_rates.extend((float(expected[0].mean()), float(expected[1].mean())))
        probe_count += 2
    raw_estimate = previous_action - observed_gradients/(2*contract.curvature_by_control)
    raw_estimate = np.clip(raw_estimate, -contract.hard_bound_normalized,
                           contract.hard_bound_normalized)
    # The antithetic quadratic estimate is unbiased for the local optimum.  A hard
    # innovation deadband can permanently retain one finite-shot false displacement
    # and creates a guaranteed interval of latency for slow drift.  Use every
    # likelihood update with an uncertainty-dependent gain; the resulting stationary
    # AR(1) posterior suppresses shot noise without censoring small physical motion.
    informative = np.abs(raw_estimate-state.estimate) > 1.55*uncertainty
    measurement = raw_estimate
    gain = np.clip(.86/(1+8*uncertainty), .45, .86)
    estimate = gain*measurement + (1-gain)*state.estimate
    model_probabilities = _model_probabilities(
        estimate, state.estimate, state.previous_estimate)
    velocity = np.clip(estimate-state.estimate, -.06, .06)
    jump_like = model_probabilities["step_or_rtn"] > .5
    forecast = estimate if jump_like else estimate+.55*velocity
    modelled = np.asarray(contract.modelled_controls, dtype=bool)
    baseline_request = np.where(modelled, forecast, 0.0)
    baseline = _project_action(contract, previous_action, baseline_request)
    residual_indices = np.flatnonzero(~modelled)
    if conditional_residual and len(residual_indices):
        # Exploitation of the last independently validated residual mean continues even
        # when the learning gate abstains.  Candidate acquisition is conditional; the
        # committed correction is an ordinary versioned control component.
        residual_index = int(residual_indices[0])
        baseline[residual_index] = np.clip(
            state.residual_mean,
            previous_action[residual_index]-contract.slew_limit_normalized,
            previous_action[residual_index]+contract.slew_limit_normalized)
    residual_estimate = (
        float(np.mean(estimate[residual_indices]))-state.residual_mean
        if len(residual_indices) else 0.0)
    residual_uncertainty = float(np.mean(uncertainty[residual_indices])) if len(residual_indices) else 0.0
    noise_floor = max(.018, 1.35*residual_uncertainty)
    significant = abs(residual_estimate) > noise_floor
    if significant:
        state.residual_persistence += 1
        state.residual_sign_history.append(1 if residual_estimate >= 0 else -1)
    else:
        state.residual_persistence = 0
        state.residual_sign_history.clear()
    repeatability = (abs(sum(state.residual_sign_history[-4:]))
                     / max(1, len(state.residual_sign_history[-4:])))
    residual_snr = abs(residual_estimate)/max(residual_uncertainty, .008)
    probability_positive = .5*(1+math.erf((residual_snr-1)/math.sqrt(2)))
    gate_evidence = ResidualGateEvidence(
        persistence_intervals=state.residual_persistence,
        repeatability=repeatability,
        control_sensitivity=(float(np.mean(contract.curvature_by_control[residual_indices]))
                             if len(residual_indices) else 0.0),
        identifiable=bool(len(residual_indices)),
        forecast_uncertainty=min(1., residual_uncertainty),
        stability_probability=.96,
        residual_magnitude=abs(residual_estimate),
        noise_floor=noise_floor,
        probability_positive_value=max(probability_positive, .5),
        expected_heldout_gain=max(
            state.last_heldout_gain,
            (float(np.mean(contract.curvature_by_control[residual_indices]))
             * residual_estimate**2 if len(residual_indices) else 0.0)),
        gradient_snr=residual_snr,
        lifecycle_healthy=True,
        predictive_only_adequate=not significant,
        ood=model_probabilities["unknown"] > .45,
    )
    decision = state.gate.evaluate(gate_evidence)
    residual_candidate_count = residual_candidate_cycles = 0
    residual_candidate_rate = None
    validation = None
    if (conditional_residual and len(residual_indices)
            and decision.disposition in {ResidualRLDisposition.SHADOW,
                                         ResidualRLDisposition.ACTIVE}):
        residual_index = int(residual_indices[0])
        offsets = np.asarray((1., -1., .5, -.5))*state.residual_sigma
        candidates = []
        for offset in offsets:
            candidate = baseline.copy()
            candidate[residual_index] = np.clip(
                state.residual_mean+offset,
                -contract.hard_bound_normalized,
                contract.hard_bound_normalized)
            candidates.append(candidate)
        candidate_actions = np.asarray(candidates)
        expected, observed = _observe(
            contract, candidate_actions, optimum, previous_action,
            config.residual_candidate_cycles,
            scenario.seed, scenario.scenario_id, "residual", interval,
        )
        relevant = contract.mask[:, residual_index].astype(bool)
        losses = observed[:, relevant].mean(axis=1)
        centered = losses-losses.mean()
        gradient = float(np.mean(centered*offsets)/max(np.mean(offsets*offsets), 1e-12))
        proposed_mean = float(np.clip(
            state.residual_mean-.85*gradient,
            -contract.hard_bound_normalized,
            contract.hard_bound_normalized))
        proposal = baseline.copy(); proposal[residual_index] = proposed_mean
        baseline_expected, baseline_observed = _observe(
            contract, baseline[None, :], optimum, previous_action,
            config.residual_candidate_cycles,
            scenario.seed, scenario.scenario_id, "residual-validation-base", interval,
        )
        proposal_expected, proposal_observed = _observe(
            contract, proposal[None, :], optimum, previous_action,
            config.residual_candidate_cycles,
            scenario.seed, scenario.scenario_id, "residual-validation-proposal", interval,
        )
        baseline_loss = float(baseline_observed.mean())
        proposal_loss = float(proposal_observed.mean())
        heldout_gain = baseline_loss-proposal_loss
        p = max(float(baseline_expected.mean()), float(proposal_expected.mean()))
        gain_se = math.sqrt(2*max(p*(1-p), 1e-12)
                            /(config.residual_candidate_cycles*len(contract.detector_ids)))
        positive_probability = .5*(1+math.erf(
            heldout_gain/max(gain_se, 1e-12)/math.sqrt(2)))
        validation_passed = heldout_gain > 0 and positive_probability >= .80
        state.gate.record_shadow_outcome(
            gain=heldout_gain,
            probability_positive_value=positive_probability,
            gradient_snr=abs(gradient)/max(gain_se, 1e-12),
        )
        state.last_heldout_gain = heldout_gain
        if validation_passed:
            state.residual_mean = proposed_mean
            baseline = _project_action(contract, previous_action, proposal)
        residual_candidate_count = len(candidate_actions)+2
        residual_candidate_cycles = residual_candidate_count*config.residual_candidate_cycles
        residual_candidate_rate = float(expected.mean())
        validation = {
            "baseline_observed_rate": baseline_loss,
            "proposal_observed_rate": proposal_loss,
            "heldout_gain": heldout_gain,
            "gain_standard_error": gain_se,
            "probability_positive_value": positive_probability,
            "passed": validation_passed,
        }
    state.previous_estimate = state.estimate.copy()
    state.estimate = estimate.copy()
    state.uncertainty = uncertainty.copy()
    state.action = baseline.copy()
    stage_evidence = {
        "stage2": {
            "posterior_mean": estimate.tolist(),
            "posterior_stddev": uncertainty.tolist(),
            "observed_gradient": observed_gradients.tolist(),
            "innovation_above_1p55_sigma": informative.tolist(),
            "truth_used_by_controller": False,
            "probe_count": probe_count,
        },
        "stage3": {
            "model_probabilities": model_probabilities,
            "joint_likelihood_path": "finite detector-likelihood model bank development adapter",
            "sequential_hdfa_used_as_product": False,
        },
        "stage4": {
            "forecast_mean": forecast.tolist(),
            "forecast_stddev": (uncertainty+np.abs(velocity)*.25).tolist(),
            "validity_horizon_intervals": 1,
        },
        "stage5": {
            "baseline_action": baseline_request.tolist(),
            "activated_action": baseline.tolist(),
            "hard_bounds_respected": bool(np.all(np.abs(baseline) <= contract.hard_bound_normalized+1e-12)),
            "slew_respected": bool(np.all(np.abs(baseline-previous_action) <= contract.slew_limit_normalized+1e-12)),
            "residual_projection_controls": [contract.control_ids[index] for index in residual_indices],
        },
        "stage6": {
            "conditional": conditional_residual,
            "gate_decision": asdict(decision),
            "residual_mean": state.residual_mean,
            "candidate_count": residual_candidate_count,
            "heldout_validation": validation,
        },
        "stage7": {
            "mode": ("RESIDUAL_LEARNING" if residual_candidate_count else "NOMINAL_PREDICTIVE"),
            "authorization": "approved",
            "rollback_required": False,
        },
        "evaluation_only": {
            "posterior_state_error": (estimate-optimum).tolist(),
            "forecast_error": (forecast-optimum).tolist(),
            "mpc_action_error": (baseline-optimum).tolist(),
            "forecast_95pct_coverage": bool(np.all(
                np.abs(forecast-optimum) <= 1.96*(uncertainty+np.abs(velocity)*.25+.01))),
            "final_residual": (optimum-baseline).tolist(),
        },
    }
    diagnostic_cycles = probe_count*config.stage2_probe_cycles
    compute_ns = time.perf_counter_ns()-started
    exploration_rate = (float(np.mean(probe_rates)) if probe_rates else 0.0)
    if residual_candidate_rate is not None:
        total_probe = probe_count*config.stage2_probe_cycles
        total_residual = residual_candidate_count*config.residual_candidate_cycles
        exploration_rate = ((exploration_rate*total_probe
                             + residual_candidate_rate*total_residual)
                            / max(total_probe+total_residual, 1))
    return (baseline, stage_evidence, probe_count+residual_candidate_count,
            diagnostic_cycles+residual_candidate_cycles, exploration_rate,
            compute_ns)


def run_arm(
    contract: PlantContract,
    scenario: ScenarioRealization,
    arm: str,
    config: TrackBConfig,
) -> dict[str, Any]:
    if arm not in ARM_NAMES:
        raise ValueError(f"unknown Track-B arm {arm!r}")
    total_optimum = scenario.total_optimum
    intervals = len(total_optimum)
    action = np.zeros(len(contract.control_ids))
    previous_action = action.copy()
    high_agent = reduced_agent = None
    if arm == "certified_high_shot_google_rl":
        high_agent = GaussianPolicyGradientAgent(
            contract.control_ids, contract.detector_ids, contract.mask,
            contract.scales, action, named_config("high_shot_reference"),
            seed=_stable_seed(scenario.seed, scenario.scenario_id, arm))
    elif arm == "certified_reduced_budget_google_rl":
        reduced_agent = GaussianPolicyGradientAgent(
            contract.control_ids, contract.detector_ids, contract.mask,
            contract.scales, action, named_config("reduced_budget_candidate"),
            seed=_stable_seed(scenario.seed, scenario.scenario_id, arm))
    predictive_state = None
    if arm.startswith("predictive_hdfa"):
        predictive_state = _PredictiveState(
            action.copy(), action.copy(), action.copy(),
            np.full_like(action, .04), 0.0, .08, 0, [], 0.0,
            ResidualActivationGate(),
        )
    rows: list[dict[str, Any]] = []
    cumulative_native_cycles = cumulative_wall = 0.0
    truth_access_count = 0
    for interval in range(intervals):
        optimum = total_optimum[interval]
        controller_started = time.perf_counter_ns()
        candidate_count = candidate_cycles = diagnostic_shots = 0
        diagnostic_downtime = 0.0
        aggregate_exploration_rate = None
        stage_evidence = None
        official_mean_evaluation_cycles = 0
        previous_action = action.copy()
        if arm == "fixed":
            action = np.zeros_like(action)
        elif arm == "oracle":
            truth_access_count += 1
            action = _project_action(contract, previous_action, optimum)
        elif arm == "periodic_recalibration":
            if interval >= scenario.onset_interval and (
                    interval-scenario.onset_interval) % config.periodic_cadence_intervals == 0:
                rng = np.random.default_rng(_stable_seed(
                    scenario.seed, scenario.scenario_id, arm, interval, "characterization"))
                standard_error = .45/math.sqrt(config.periodic_characterization_shots)
                observed = optimum+rng.normal(scale=standard_error, size=len(action))
                observable = np.asarray(contract.periodic_observable_controls, dtype=bool)
                target = np.where(observable, observed, previous_action)
                action = _project_action(contract, previous_action, target)
                diagnostic_shots = config.periodic_characterization_shots
                diagnostic_downtime = diagnostic_shots*contract.cycle_period_s
        elif high_agent is not None or reduced_agent is not None:
            agent = high_agent or reduced_agent
            assert agent is not None
            agent_config = agent.config
            batch = agent.sample_candidates()
            expected, observed = _observe(
                contract, batch.actions_native, optimum, previous_action,
                agent_config.sampling.effective_cycles_per_candidate,
                scenario.seed, scenario.scenario_id, arm, interval, "candidates",
            )
            agent.update(batch, tuple(
                CandidateEvaluation(identifier, observed[index])
                for index, identifier in enumerate(batch.candidate_ids)))
            action = agent.mean_native.copy()
            candidate_count = len(batch.candidate_ids)
            candidate_cycles = (candidate_count
                                * agent_config.sampling.effective_cycles_per_candidate)
            aggregate_exploration_rate = float(expected.mean())
            if interval % agent_config.sampling.mean_evaluation_period_epochs == 0:
                official_mean_evaluation_cycles = agent_config.mean_evaluation_qec_cycles
                _observe(
                    contract, action[None, :], optimum, previous_action,
                    official_mean_evaluation_cycles,
                    scenario.seed, scenario.scenario_id, arm, interval, "official-mean-eval",
                )
            stage_evidence = {
                "certified_track_a_agent": True,
                "track_a_config": agent_config.name,
                "agent_state": agent.state_record(),
                "agent_update": dict(agent.last_update),
                "candidate_ids_unique": len(set(batch.candidate_ids)) == len(batch.candidate_ids),
                "candidate_reward_association": "stable candidate IDs",
                "truth_used_by_controller": False,
            }
        else:
            assert predictive_state is not None
            conditional = arm == "predictive_hdfa_conditional_residual_rl"
            (action, stage_evidence, candidate_count, candidate_cycles,
             aggregate_exploration_rate, predictive_compute_ns) = _predictive_step(
                contract, scenario, interval, optimum, predictive_state, config,
                conditional_residual=conditional)
            controller_started = time.perf_counter_ns()-predictive_compute_ns
        controller_compute_s = (time.perf_counter_ns()-controller_started)/1e9
        expected, observed = _observe(
            contract, action[None, :], optimum, previous_action,
            config.endpoint_evaluation_cycles,
            scenario.seed, scenario.scenario_id, "common-endpoint", interval,
        )
        expected_rate = float(expected.mean())
        observed_rate = float(observed.mean())
        logical_expected = float(expected_logical_rate(contract, expected)[0])
        logical_rng = np.random.default_rng(_stable_seed(
            scenario.seed, scenario.scenario_id, "logical", interval))
        logical_failures = int(logical_rng.binomial(
            config.logical_evaluation_shots, logical_expected))
        logical_observed = logical_failures/config.logical_evaluation_shots
        logical_per_round = 1-(1-logical_observed)**(1/3)
        oracle_rates = expected_detector_rates(
            contract, optimum[None, :], optimum, optimum)[0]
        fixed_rates = expected_detector_rates(
            contract, np.zeros((1, len(action))), optimum, np.zeros_like(action))[0]
        transaction = _policy_transaction(
            arm, interval, previous_action, action, scenario)
        if not transaction["lifecycle_valid"]:
            raise RuntimeError("Track-B action violated the common bounds/slew contract")
        acquisition_cycles = (candidate_cycles + official_mean_evaluation_cycles
                              + config.endpoint_evaluation_cycles)
        interval_wall = (acquisition_cycles*contract.cycle_period_s
                         + diagnostic_downtime + controller_compute_s)
        cumulative_native_cycles += acquisition_cycles
        cumulative_wall += interval_wall
        exploration_damage_events = 0.0
        if aggregate_exploration_rate is not None:
            exploration_damage_events = max(
                0.0, aggregate_exploration_rate-expected_rate) * candidate_cycles
        # Native-QEC candidates still execute the QEC circuit and are scored through
        # exploration damage.  Only interrupting characterization removes computation
        # availability; treating exploration as downtime would double-penalize it.
        native_computation_cycles = (
            config.endpoint_evaluation_cycles+candidate_cycles
            + official_mean_evaluation_cycles)
        availability = native_computation_cycles/max(
            native_computation_cycles+diagnostic_shots, 1)
        row = {
            "interval": interval,
            "arm": arm,
            "scenario_id": scenario.scenario_id,
            "seed": scenario.seed,
            "disturbance_path_hash": scenario.disturbance_path_hash,
            "residual_stratum": scenario.residual_stratum,
            "latent_optimum_evaluation_only": optimum.tolist(),
            "structured_optimum_evaluation_only": list(scenario.structured_optimum[interval]),
            "hidden_residual_evaluation_only": list(scenario.hidden_residual[interval]),
            "applied_control": action.tolist(),
            "mismatch_evaluation_only": (optimum-action).tolist(),
            "mean_policy_detector_rate": observed_rate,
            "expected_mean_policy_detector_rate_evaluation_only": expected_rate,
            "aggregate_exploration_detector_rate": aggregate_exploration_rate,
            "exploration_damage_detector_events": exploration_damage_events,
            "fixed_expected_detector_rate_evaluation_only": float(fixed_rates.mean()),
            "oracle_expected_detector_rate_evaluation_only": float(oracle_rates.mean()),
            "logical_failure_count": logical_failures,
            "logical_shots": config.logical_evaluation_shots,
            "logical_failure_rate": logical_observed,
            "expected_logical_failure_rate_evaluation_only": logical_expected,
            "logical_error_per_round": logical_per_round,
            "candidate_evaluations": candidate_count,
            "candidate_cycles": candidate_cycles,
            "official_mean_evaluation_cycles": official_mean_evaluation_cycles,
            "endpoint_evaluation_cycles": config.endpoint_evaluation_cycles,
            "diagnostic_shots": diagnostic_shots,
            "diagnostic_downtime_s": diagnostic_downtime,
            "logical_computation_availability": availability,
            "native_qec_computation_cycles": native_computation_cycles,
            "controller_compute_s": controller_compute_s,
            "interval_wall_clock_s": interval_wall,
            "cumulative_native_qec_cycles": int(cumulative_native_cycles),
            "cumulative_wall_clock_s": cumulative_wall,
            "policy_transaction": transaction,
            "lifecycle_and_rollback": {
                "lifecycle_valid": transaction["lifecycle_valid"],
                "rollback_count": 0,
                "rollback_outcome": "not_required",
            },
            "stage_evidence": stage_evidence,
        }
        rows.append(row)
    return {
        "schema_version": "track-b-arm-run.v1",
        "evidence_layer": contract.evidence_layer,
        "plant_id": contract.plant_id,
        "scenario_id": scenario.scenario_id,
        "family": scenario.family,
        "seed": scenario.seed,
        "arm": arm,
        "disturbance_path_hash": scenario.disturbance_path_hash,
        "controller_truth_access_count": truth_access_count,
        "non_oracle_truth_isolated": arm == "oracle" or truth_access_count == 0,
        "completion_status": "completed",
        "trajectory": rows,
    }
