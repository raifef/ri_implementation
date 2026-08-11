"""Scenario-based local MPC with independent safety validation and certified fallbacks."""
from __future__ import annotations

from dataclasses import dataclass, replace
import heapq
import math
from typing import Callable, Mapping, Sequence

try:  # NumPy is the declared runtime accelerator; the scalar reference remains executable.
    import numpy as _np
except ImportError:  # pragma: no cover - exercised by minimal dependency installations.
    _np = None

from hdfa_rl_suite.stage0.schema import HardwareLimits, PolicySnapshot, stable_hash
from hdfa_rl_suite.stage2.inference import QuadraticLogitObservationModel
from hdfa_rl_suite.stage3.schema import FamiliarProcessState
from hdfa_rl_suite.stage4.schema import ForecastBundle, ResponseMap

from .schema import (
    InfeasibilityCertificate, PredictedCostDistribution, PredictiveControlPackage,
    ResidualAllocation, SharedResourceConstraint, SolverStatus,
)


@dataclass(frozen=True, slots=True)
class _ObjectiveScenario:
    weight: float
    risk_cost: float
    optimum_values: tuple[float | None, ...]
    prepared_responses: tuple[tuple[str, object], ...]


@dataclass(frozen=True, slots=True)
class _VectorizedDetectorObjective:
    detector_id: str
    prefix: object
    state_quadratic: object
    control_linear: tuple[tuple[int, float], ...]
    control_quadratic: tuple[tuple[int, int, float], ...]
    state_control_coefficients: object
    state_control_indices: object


@dataclass(frozen=True, slots=True)
class _VectorizedObjective:
    controls: tuple[str, ...]
    weights: object
    risk_costs: object
    optimum_values: object
    optimum_mask: object
    detectors: tuple[_VectorizedDetectorObjective, ...]


@dataclass(frozen=True)
class MPCConfig:
    control_penalty: float = 0.05
    worst_case_weight: float = 0.25
    chance_violation_limit: float = 0.05
    forecast_expiry_slack_s: float = 0.0
    maximum_coordinate_steps: int = 24
    cvar_weight: float = .25
    detector_threshold: float = .10
    distributional_ambiguity_radius: float = .02
    independent_validation_tolerance: float = 1e-9
    vectorized_objective: bool = True


class PredictiveController:
    """MVP scenario MPC; every numerical solution is rechecked independently before use."""

    def __init__(self, limits: HardwareLimits, observation_model: QuadraticLogitObservationModel,
                 config: MPCConfig = MPCConfig(), shared_constraints: Sequence[SharedResourceConstraint] = (),
                 safety_validator: Callable[[Mapping[str, float]], tuple[bool, str]] | None = None) -> None:
        self.limits, self.observation_model, self.config = limits, observation_model, config
        self.shared_constraints, self.safety_validator = tuple(shared_constraints), safety_validator

    def _prepare_vectorized_objective(self, scenarios, controls: Sequence[str]) -> _VectorizedObjective:
        controls = tuple(controls)
        control_indices = {control: index for index, control in enumerate(controls)}
        count, control_count = len(scenarios), len(controls)
        weights = _np.fromiter((scenario.weight for scenario in scenarios), dtype=float, count=count)
        risk_costs = _np.fromiter((
            getattr(scenario, "logical_risk", 0.) + getattr(scenario, "correlation_risk", 0.)
            + getattr(scenario, "leakage_risk", 0.) for scenario in scenarios), dtype=float, count=count)
        optimum_values = _np.zeros((count, control_count), dtype=float)
        optimum_mask = _np.zeros((count, control_count), dtype=bool)
        for row, scenario in enumerate(scenarios):
            for column, control in enumerate(controls):
                if control in scenario.optimum_controls:
                    optimum_values[row, column] = scenario.optimum_controls[control]
                    optimum_mask[row, column] = True
        detectors: list[_VectorizedDetectorObjective] = []
        for detector_id, response in self.observation_model.responses.items():
            prefix = _np.empty(count, dtype=float)
            state_quadratic = _np.empty(count, dtype=float)
            state_control_terms = tuple(response.state_control.items())
            state_control_coefficients = _np.empty((count, len(state_control_terms)), dtype=float)
            state_control_indices = _np.fromiter(
                (control_indices.get(control, -1) for (_, control), _ in state_control_terms),
                dtype=int, count=len(state_control_terms))
            for row, scenario in enumerate(scenarios):
                state = scenario.state
                value = response.intercept + response.context_intercepts.get("", 0.0)
                value += sum(coefficient * state.get(variable, 0.0)
                             for variable, coefficient in response.state_linear.items())
                prefix[row] = value
                state_quadratic[row] = sum(
                    coefficient * state.get(left, 0.0) * state.get(right, 0.0)
                    for (left, right), coefficient in response.state_quadratic.items())
                for column, ((variable, _), coefficient) in enumerate(state_control_terms):
                    state_control_coefficients[row, column] = coefficient * state.get(variable, 0.0)
            detectors.append(_VectorizedDetectorObjective(
                detector_id, prefix, state_quadratic,
                tuple((control_indices.get(control, -1), coefficient)
                      for control, coefficient in response.control_linear.items()),
                tuple((control_indices.get(left, -1), control_indices.get(right, -1), coefficient)
                      for (left, right), coefficient in response.control_quadratic.items()),
                state_control_coefficients, state_control_indices,
            ))
        return _VectorizedObjective(controls, weights, risk_costs, optimum_values,
                                    optimum_mask, tuple(detectors))

    def _prepare_objective(self, scenarios, controls: Sequence[str]
                           ) -> tuple[_ObjectiveScenario, ...] | _VectorizedObjective:
        if self.config.vectorized_objective and _np is not None:
            return self._prepare_vectorized_objective(scenarios, controls)
        return tuple(_ObjectiveScenario(
            scenario.weight,
            getattr(scenario, "logical_risk", 0.) + getattr(scenario, "correlation_risk", 0.)
            + getattr(scenario, "leakage_risk", 0.),
            tuple(scenario.optimum_controls.get(control) for control in controls),
            self.observation_model.prepare_state_for_controls(scenario.state, controls),
        ) for scenario in scenarios)

    def _objective_vectorized(self, action: Mapping[str, float], prepared: _VectorizedObjective,
                              *, violations: bool) -> tuple[float, float, dict[str, float]]:
        values = _np.fromiter((action.get(control, 0.) for control in prepared.controls),
                              dtype=float, count=len(prepared.controls))
        detector_costs = _np.zeros(len(prepared.weights), dtype=float)
        detector_violations = {detector: 0. for detector in self.observation_model.responses}
        for detector in prepared.detectors:
            logit = detector.prefix.copy()
            if detector.control_linear:
                logit += sum(coefficient * values[index] if index >= 0 else 0.
                             for index, coefficient in detector.control_linear)
            logit += detector.state_quadratic
            if detector.control_quadratic:
                logit += sum(coefficient * (values[left] if left >= 0 else 0.)
                             * (values[right] if right >= 0 else 0.)
                             for left, right, coefficient in detector.control_quadratic)
            if detector.state_control_indices.size:
                valid = detector.state_control_indices >= 0
                control_values = _np.zeros(len(detector.state_control_indices), dtype=float)
                control_values[valid] = values[detector.state_control_indices[valid]]
                logit += _np.sum(detector.state_control_coefficients * control_values, axis=1)
            probability = 1. / (1. + _np.exp(-_np.clip(logit, -35., 35.)))
            probability = _np.clip(probability, 1e-9, 1-1e-9)
            detector_costs += probability
            if violations:
                detector_violations[detector.detector_id] = float(
                    _np.sum(prepared.weights[probability > self.config.detector_threshold]))
        difference = values[None, :] - prepared.optimum_values
        movement = _np.sum(_np.where(prepared.optimum_mask, difference * difference, 0.), axis=1)
        costs = detector_costs + prepared.risk_costs + self.config.control_penalty * movement
        expected = float(_np.sum(costs * prepared.weights))
        order = _np.lexsort((prepared.weights, costs))[::-1]
        tail = mass = 0.
        for index in order:
            take = min(float(prepared.weights[index]) + self.config.distributional_ambiguity_radius,
                       max(0., .1 - mass))
            tail += float(costs[index]) * take
            mass += take
            if mass >= .1 - 1e-12:
                break
        cvar = tail / max(mass, 1e-12)
        worst = float(costs[order[0]])
        return (expected + self.config.worst_case_weight * worst + self.config.cvar_weight * cvar,
                worst, detector_violations)

    def _objective_prepared(self, action: Mapping[str, float], controls: Sequence[str],
                            scenarios: Sequence[_ObjectiveScenario] | _VectorizedObjective, *,
                            violations: bool) -> tuple[float, float, dict[str, float]]:
        if isinstance(scenarios, _VectorizedObjective):
            return self._objective_vectorized(action, scenarios, violations=violations)
        costs, detector_violations = [], {detector: 0. for detector in self.observation_model.responses}
        action_values = tuple(action.get(control, 0.) for control in controls)
        for scenario in scenarios:
            detector_cost = 0.
            for detector, response in scenario.prepared_responses:
                probability = self.observation_model.probability_prepared_values(response, action_values)
                detector_cost += probability
                if violations and probability > self.config.detector_threshold:
                    detector_violations[detector] += scenario.weight
            movement = sum((value - optimum) ** 2
                           for value, optimum in zip(action_values, scenario.optimum_values)
                           if optimum is not None)
            costs.append((detector_cost + scenario.risk_cost + self.config.control_penalty * movement,
                          scenario.weight))
        expected = sum(cost * weight for cost, weight in costs)
        if self.config.distributional_ambiguity_radius > 0.:
            tail_items = min(len(costs), math.ceil(.1 / self.config.distributional_ambiguity_radius))
            ordered = heapq.nlargest(tail_items, costs)
        else:
            ordered = sorted(costs, reverse=True)
        tail, mass = 0., 0.
        for cost, weight in ordered:
            take = min(weight + self.config.distributional_ambiguity_radius, max(0., .1 - mass))
            tail, mass = tail + cost * take, mass + take
            if mass >= .1 - 1e-12:
                break
        cvar = tail / max(mass, 1e-12)
        worst = ordered[0][0]
        return expected + self.config.worst_case_weight * worst + self.config.cvar_weight * cvar, worst, detector_violations

    def _objective(self, action: Mapping[str, float], scenarios) -> tuple[float, float, dict[str, float]]:
        """Compatibility wrapper for direct objective evaluation and tests."""
        controls = tuple(action)
        prepared = self._prepare_objective(scenarios, controls)
        return self._objective_prepared(action, controls, prepared, violations=True)

    def _project_hard_constraints(self, candidate: Mapping[str, float], current: Mapping[str, float]) -> tuple[dict[str, float], list[str]]:
        projected, active = dict(candidate), []
        for control, value in candidate.items():
            if control not in self.limits.controls:
                continue
            bound = self.limits.controls[control]
            limited = min(bound.maximum, max(bound.minimum, value))
            if limited != value:
                active.append(f"bound:{control}")
            previous = current.get(control, limited)
            slew_limited = min(previous + bound.max_slew, max(previous - bound.max_slew, limited))
            if slew_limited != limited:
                active.append(f"slew:{control}")
            # Stage-2/0 response trust radius constrains a single predictive control move.
            trusted = min(previous + bound.trust_radius, max(previous - bound.trust_radius, slew_limited))
            if trusted != slew_limited:
                active.append(f"trust:{control}")
            projected[control] = trusted
        for constraint in self.shared_constraints:
            usage = sum(constraint.coefficients.get(control, 0.) * projected.get(control, 0.) for control in constraint.coefficients)
            if usage > constraint.maximum:
                positive = {control: coefficient * projected.get(control, 0.) for control, coefficient in constraint.coefficients.items()
                            if coefficient * projected.get(control, 0.) > 0}
                total = sum(positive.values())
                if total > 0:
                    scale = max(0., (constraint.maximum - (usage - total)) / total)
                    for control in positive:
                        projected[control] *= min(1., scale)
                active.append(f"shared:{constraint.constraint_id}")
        # Conservative normalized simultaneous-drive duty proxy; hardware backends may replace it.
        duty = sum((projected.get(control, 0.) / max(abs(bound.maximum), abs(bound.minimum), 1e-12)) ** 2
                   for control, bound in self.limits.controls.items()) / max(1, len(self.limits.controls))
        if duty > self.limits.max_thermal_duty:
            scale = math.sqrt(self.limits.max_thermal_duty / duty)
            projected = {control: value * scale if control in self.limits.controls else value for control, value in projected.items()}
            active.append("thermal-duty")
        return projected, active

    def _independent_validate(self, candidate: Mapping[str, float], previous: Mapping[str, float]) -> tuple[bool, tuple[str, ...], float]:
        violations: list[str] = []
        margins: list[float] = []
        for control, value in candidate.items():
            bound = self.limits.controls.get(control)
            if not bound:
                violations.append(f"unknown:{control}")
                continue
            margins.extend((value - bound.minimum, bound.maximum - value,
                            bound.max_slew - abs(value - previous.get(control, value)),
                            bound.trust_radius - abs(value - previous.get(control, value))))
        for constraint in self.shared_constraints:
            usage = sum(constraint.coefficients.get(control, 0.) * candidate.get(control, 0.) for control in constraint.coefficients)
            margins.append(constraint.maximum - usage)
            if usage > constraint.maximum + self.config.independent_validation_tolerance:
                violations.append(f"shared:{constraint.constraint_id}")
        if self.safety_validator:
            valid, reason = self.safety_validator(candidate)
            if not valid:
                violations.append(f"nonlinear:{reason}")
        if any(margin < -self.config.independent_validation_tolerance for margin in margins):
            violations.append("hard-bound/slew/trust")
        return not violations, tuple(dict.fromkeys(violations)), min(margins, default=0.)

    def _residual_allocation(self, action: Mapping[str, float], scenarios) -> ResidualAllocation:
        model_controls = set()
        for response in self.observation_model.responses.values():
            model_controls.update(response.control_linear)
            model_controls.update(left for pair in response.control_quadratic for left in pair)
            model_controls.update(control for _, control in response.state_control)
        controls = tuple(key for key in action if key in self.limits.controls and key in model_controls)
        bounds, rationale = {}, {}
        for control in controls:
            mean = sum(item.weight * item.optimum_controls.get(control, action[control]) for item in scenarios)
            variance = sum(item.weight * (item.optimum_controls.get(control, action[control]) - mean) ** 2 for item in scenarios)
            hardware = self.limits.controls[control]
            uncertainty_bound = hardware.trust_radius * (.05 + .75 * variance / (1 + variance))
            bound = max(0., min(uncertainty_bound, action[control] - hardware.minimum,
                                hardware.maximum - action[control], hardware.max_slew))
            bounds[control] = bound
            rationale[control] = "residual authority widened only for forecast-disagreement directions"
        return ResidualAllocation(controls, bounds, controls, rationale)

    def _fallback(self, status: SolverStatus, reason: str, current: PolicySnapshot, activation_time_s: float, expiry_time_s: float,
                  scenarios=(), violations: tuple[str, ...] = ()) -> PredictiveControlPackage:
        action = dict(current.values)
        allocation = self._residual_allocation(action, scenarios) if scenarios else ResidualAllocation((), {}, (), {})
        cost = PredictedCostDistribution(math.inf, math.inf, {})
        certificate = InfeasibilityCertificate(reason, violations, current.policy_hash)
        return PredictiveControlPackage("stage5.v2", status, action, (action,), {}, allocation, violations, cost, current,
            current.policy_hash, activation_time_s, expiry_time_s, current, certificate, 0., action, (reason,))

    def familiar_feedforward(self, package: PredictiveControlPackage,
                             forecast: ForecastBundle,
                             current_policy: PolicySnapshot,
                             response_map: ResponseMap,
                             familiar: FamiliarProcessState | None
                             ) -> PredictiveControlPackage:
        """Select an immediate familiar-model correction only when it is no worse.

        The fast path reuses the already propagated model-conditioned scenarios and is
        independently projected/validated.  It cannot bypass the full scenario objective,
        chance constraints, or Stage-7 authorization.
        """
        if (package.status is not SolverStatus.OPTIMAL or familiar is None
                or not familiar.immediate_feedforward_safe):
            return package
        horizon = min(forecast.scenarios_by_horizon, default=None)
        if horizon is None:
            return package
        scenarios = forecast.scenarios(horizon)
        family_tokens = {
            "oscillator": ("oscillator",),
            "random_telegraph": ("rtn",),
            "semi_markov_telegraph": ("semi-markov",),
            "ornstein_uhlenbeck": ("ou",),
            "step": ("step",),
            "additive_composite": ("plus", "composite"),
        }.get(familiar.family, (familiar.family,))
        selected = [item for item in scenarios
                    if any(token in item.model_id for token in family_tokens)]
        if not selected:
            return package
        mass = sum(item.weight for item in selected)
        target = dict(package.action)
        for control in response_map.reference_controls:
            values = [(item.optimum_controls.get(control), item.weight)
                      for item in selected if control in item.optimum_controls]
            if values:
                target[control] = sum(value*weight for value, weight in values)/max(
                    sum(weight for _, weight in values), 1e-12)
        target, active = self._project_hard_constraints(target, current_policy.values)
        valid, violations, margin = self._independent_validate(
            target, current_policy.values)
        if not valid:
            return package
        prepared = self._prepare_objective(scenarios, tuple(package.action))
        reference_cost = self._objective_prepared(
            package.action, tuple(package.action), prepared, violations=False)[0]
        fast_cost = self._objective_prepared(
            target, tuple(package.action), prepared, violations=False)[0]
        if fast_cost > reference_cost+self.config.independent_validation_tolerance:
            return package
        trajectory = list(package.trajectory)
        trajectory[0] = target
        return replace(
            package, action=target, trajectory=tuple(trajectory),
            feedforward_component={
                control: target[control]-current_policy.values.get(control, 0.0)
                for control in target},
            active_constraints=package.active_constraints+(
                f"familiar-fast-path:{familiar.family}", *active),
            robustness_margin=min(package.robustness_margin, margin),
            policy_hash=stable_hash({
                "familiar": familiar.regime_id, "action": target,
                "trajectory": trajectory, "base": current_policy.policy_hash}),
        )

    def cached_regime_policy(self, package: PredictiveControlPackage,
                             forecast: ForecastBundle,
                             current_policy: PolicySnapshot,
                             cached_controls: Mapping[str, float] | None,
                             cache_key: str) -> PredictiveControlPackage:
        """Reuse a detector-validated regime policy subject to current evidence."""
        if package.status is not SolverStatus.OPTIMAL or not cached_controls:
            return package
        target = dict(package.action)
        target.update({control: value for control, value in cached_controls.items()
                       if control in target})
        target, active = self._project_hard_constraints(target, current_policy.values)
        valid, violations, margin = self._independent_validate(
            target, current_policy.values)
        if not valid:
            return package
        horizon = min(forecast.scenarios_by_horizon, default=None)
        if horizon is None:
            return package
        scenarios = forecast.scenarios(horizon)
        prepared = self._prepare_objective(scenarios, tuple(package.action))
        reference_cost = self._objective_prepared(
            package.action, tuple(package.action), prepared, violations=False)[0]
        cached_cost = self._objective_prepared(
            target, tuple(package.action), prepared, violations=False)[0]
        if cached_cost > reference_cost+self.config.independent_validation_tolerance:
            return package
        trajectory = list(package.trajectory)
        trajectory[0] = target
        return replace(
            package, action=target, trajectory=tuple(trajectory),
            feedforward_component={
                control: target[control]-current_policy.values.get(control, 0.0)
                for control in target},
            active_constraints=package.active_constraints+(
                f"validated-regime-policy-cache:{cache_key}", *active),
            robustness_margin=min(package.robustness_margin, margin),
            policy_hash=stable_hash({
                "cache_key": cache_key, "action": target,
                "trajectory": trajectory, "base": current_policy.policy_hash}),
        )

    def solve(self, forecast: ForecastBundle, horizon_s: float, current_policy: PolicySnapshot,
              *, now_s: float | None = None) -> PredictiveControlPackage:
        horizons = tuple(value for value in sorted(forecast.scenarios_by_horizon) if value <= horizon_s)
        if horizon_s in forecast.scenarios_by_horizon and horizon_s not in horizons:
            horizons += (horizon_s,)
        return self.solve_trajectory(forecast, horizons or (horizon_s,), current_policy, now_s=now_s)

    def solve_trajectory(self, forecast: ForecastBundle, horizons_s: Sequence[float], current_policy: PolicySnapshot,
                         *, now_s: float | None = None) -> PredictiveControlPackage:
        now = forecast.issued_at_s if now_s is None else now_s
        horizons = tuple(sorted(dict.fromkeys(horizons_s)))
        horizon_s = max(horizons, default=0.)
        first_horizon = min(horizons, default=horizon_s)
        activation = forecast.issued_at_s + first_horizon + forecast.latency.mean_s
        expiry = forecast.issued_at_s + forecast.validity_horizon_s + self.config.forecast_expiry_slack_s
        if not horizons or horizon_s > forecast.validity_horizon_s or now > expiry or forecast.invalidity_reasons:
            return self._fallback(SolverStatus.EXPIRED_FORECAST, "forecast is invalid or beyond its demonstrated horizon", current_policy, activation, expiry)
        if any(horizon not in forecast.scenarios_by_horizon for horizon in horizons):
            return self._fallback(SolverStatus.INFEASIBLE, "requested horizon is absent from forecast", current_policy, activation, expiry)
        all_scenarios = tuple(item for horizon in horizons for item in forecast.scenarios(horizon))
        if not all_scenarios:
            return self._fallback(SolverStatus.INFEASIBLE, "forecast has no scenarios", current_policy, activation, expiry)
        trajectory: list[dict[str, float]] = []
        active: list[str] = []
        previous = dict(current_policy.values)
        aggregate_violations = {detector: 0. for detector in self.observation_model.responses}
        aggregate_expected = aggregate_worst = 0.
        robustness = math.inf
        for horizon in horizons:
            scenarios = forecast.scenarios(horizon)
            objective_controls = tuple(current_policy.values)
            prepared_scenarios = self._prepare_objective(scenarios, objective_controls)
            target = {control: sum(item.weight * item.optimum_controls.get(control, previous.get(control, 0.)) for item in scenarios)
                      for control in current_policy.values}
            candidate, stage_active = self._project_hard_constraints(target, previous)
            active.extend(stage_active)
            base = self._objective_prepared(candidate, objective_controls, prepared_scenarios,
                                            violations=False)[0]
            for _ in range(self.config.maximum_coordinate_steps):
                improved = False
                for control in tuple(candidate):
                    if control not in self.limits.controls:
                        continue
                    step = self.limits.controls[control].trust_radius / 8
                    alternatives = []
                    for direction in (-1., 1.):
                        proposal = dict(candidate)
                        proposal[control] += direction * step
                        proposal, _ = self._project_hard_constraints(proposal, previous)
                        alternatives.append((self._objective_prepared(
                            proposal, objective_controls, prepared_scenarios, violations=False)[0], proposal))
                    best_cost, best = min(alternatives, key=lambda item: item[0])
                    if best_cost + 1e-12 < base:
                        candidate, base, improved = best, best_cost, True
                if not improved:
                    break
            valid, violations, margin = self._independent_validate(candidate, previous)
            if not valid:
                return self._fallback(SolverStatus.INFEASIBLE, "independent nonlinear/safety validation failed",
                                      current_policy, activation, expiry, scenarios, violations)
            expected, worst, stage_violations = self._objective_prepared(
                candidate, objective_controls, prepared_scenarios, violations=True)
            aggregate_expected += expected / len(horizons)
            aggregate_worst = max(aggregate_worst, worst)
            for detector, probability in stage_violations.items():
                aggregate_violations[detector] = max(aggregate_violations[detector], probability)
            robustness = min(robustness, margin)
            trajectory.append(candidate)
            previous = candidate
        unsafe = tuple(detector for detector, probability in aggregate_violations.items() if probability > self.config.chance_violation_limit)
        if unsafe:
            return self._fallback(SolverStatus.INFEASIBLE, "scenario chance constraint violated", current_policy,
                                  activation, expiry, all_scenarios, tuple(f"chance:{item}" for item in unsafe))
        action = trajectory[0]
        action_hash = stable_hash({"action": action, "trajectory": trajectory, "activation": activation, "base": current_policy.policy_hash})
        allocation = self._residual_allocation(action, forecast.scenarios(first_horizon))
        logical = sum(item.weight * getattr(item, "logical_risk", 0.) for item in all_scenarios) / len(horizons)
        correlation = sum(item.weight * getattr(item, "correlation_risk", 0.) for item in all_scenarios) / len(horizons)
        return PredictiveControlPackage("stage5.v2", SolverStatus.OPTIMAL, action, tuple(trajectory),
            {key: action[key] - current_policy.values.get(key, 0.) for key in action}, allocation, tuple(dict.fromkeys(active)),
            PredictedCostDistribution(aggregate_expected, aggregate_worst, aggregate_violations, aggregate_worst, logical, correlation),
            current_policy, action_hash, activation, expiry, current_policy, None, robustness, dict(current_policy.values), ())
