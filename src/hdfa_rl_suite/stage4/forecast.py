"""Activation-aligned particle/mixture propagation and forecast calibration."""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Mapping, Sequence

from hdfa_rl_suite.stage2.inference import QuadraticLogitObservationModel
from hdfa_rl_suite.stage3.schema import DynamicsModelKind, DynamicsModelSpec, DynamicsParticle, DynamicsPosterior

from .schema import ForecastBundle, ForecastCalibration, ForecastRisk, ForecastScenario, LatencyModel, ResponseMap


@dataclass(frozen=True)
class ForecastConfig:
    seed: int = 0
    maximum_validity_horizon_s: float = 10.0
    unknown_probability_limit: float = .35
    detector_probability_limit: float = .20
    maximum_scenarios: int = 256
    minimum_model_probability: float = 1e-5
    logical_risk_power: float = 2.0
    calibration_coverage_floor: float = .75


class ForecastScorer:
    """Rolling proper-score tracker; call only after a forecasted outcome becomes observable."""
    def __init__(self) -> None:
        self._log_scores: list[float] = []
        self._brier_scores: list[float] = []
        self._coverage: list[bool] = []
        self._crps: list[float] = []
        self._energy: list[float] = []

    def update_binary(self, probability: float, outcome: int, interval: tuple[float, float] | None = None, realised_value: float | None = None) -> None:
        probability = min(1 - 1e-12, max(1e-12, probability))
        self._log_scores.append(math.log(probability if outcome else 1 - probability))
        self._brier_scores.append((probability - outcome) ** 2)
        if interval is not None and realised_value is not None:
            self._coverage.append(interval[0] <= realised_value <= interval[1])

    def summary(self) -> ForecastCalibration:
        return ForecastCalibration(len(self._log_scores),
            sum(self._log_scores) / len(self._log_scores) if self._log_scores else None,
            sum(self._brier_scores) / len(self._brier_scores) if self._brier_scores else None,
            sum(self._coverage) / len(self._coverage) if self._coverage else None,
            sum(self._crps) / len(self._crps) if self._crps else None,
            sum(self._energy) / len(self._energy) if self._energy else None)

    def update_continuous(self, samples: Sequence[float], realised_value: float) -> None:
        """Empirical CRPS/energy score for one-dimensional posterior samples."""
        if not samples:
            return
        first = sum(abs(value - realised_value) for value in samples) / len(samples)
        second = sum(abs(left - right) for left in samples for right in samples) / (2 * len(samples) ** 2)
        self._crps.append(first - second)
        self._energy.append(first - second)


class ForecastEngine:
    def __init__(self, observation_model: QuadraticLogitObservationModel, model_bank: Sequence[DynamicsModelSpec],
                 response_map: ResponseMap, config: ForecastConfig = ForecastConfig(), scorer: ForecastScorer | None = None) -> None:
        self.observation_model, self.models, self.response_map, self.config = observation_model, {item.model_id: item for item in model_bank}, response_map, config
        self.scorer = scorer or ForecastScorer()
        self._rng = random.Random(config.seed)

    def _propagate(self, particle: DynamicsParticle, duration_s: float) -> dict[str, float]:
        """One sampled transition from the model held by a Stage-3 particle."""
        state, auxiliary = dict(particle.state), particle.auxiliary_state
        model = self.models[particle.model_id]
        variable, value, params = model.variable_id, state[model.variable_id], model.parameters
        params = {**params, **particle.parameter_state}
        if model.kind is DynamicsModelKind.RANDOM_WALK:
            value += self._rng.gauss(0., params.get("diffusion", .04) * math.sqrt(duration_s))
        elif model.kind is DynamicsModelKind.ORNSTEIN_UHLENBECK:
            kappa, mean, sigma = params.get("kappa", .3), params.get("mean", 0.), params.get("sigma", .06)
            value = mean + (value - mean) * math.exp(-kappa * duration_s) + self._rng.gauss(0., sigma * math.sqrt(duration_s))
        elif model.kind is DynamicsModelKind.OSCILLATOR:
            omega, damping = params.get("omega", 2.), params.get("damping", .02)
            velocity = auxiliary.get("velocity", 0.)
            value = (math.cos(omega * duration_s) * value + math.sin(omega * duration_s) * velocity / max(omega, 1e-9)) * math.exp(-damping * duration_s)
        elif model.kind in {DynamicsModelKind.RANDOM_TELEGRAPH, DynamicsModelKind.SEMI_MARKOV_TELEGRAPH}:
            rate = params.get("switch_rate", .2)
            if self._rng.random() < 1 - math.exp(-rate * duration_s):
                value = -value
            value = math.copysign(params.get("amplitude", abs(value) or .6), value or 1.)
        elif model.kind is DynamicsModelKind.STEP:
            if self._rng.random() < 1 - math.exp(-params.get("hazard", .04) * duration_s):
                value += self._rng.gauss(0., params.get("jump_sigma", .5))
        elif model.kind is DynamicsModelKind.UNKNOWN:
            scale = params.get("scale", .25) * math.sqrt(duration_s)
            value += self._rng.gauss(0., scale) / max(abs(self._rng.gauss(0., 1.)), .15)
        elif model.kind is DynamicsModelKind.ADDITIVE_COMPOSITE:
            value = 0.0
            for component in model.components:
                component_value = particle.component_state.get(component.model_id, 0.)
                component_params = component.parameters
                if component.kind is DynamicsModelKind.OSCILLATOR:
                    omega, damping = component_params.get("omega", 2.), component_params.get("damping", .02)
                    velocity = particle.auxiliary_state.get(f"{component.model_id}:velocity", 0.)
                    component_value = (math.cos(omega * duration_s) * component_value
                        + math.sin(omega * duration_s) * velocity / max(omega, 1e-9)) * math.exp(-damping * duration_s)
                    component_value += self._rng.gauss(0., component_params.get("noise", .02) * math.sqrt(duration_s))
                elif component.kind is DynamicsModelKind.ORNSTEIN_UHLENBECK:
                    kappa, mean, sigma = component_params.get("kappa", .3), component_params.get("mean", 0.), component_params.get("sigma", .06)
                    component_value = mean + (component_value - mean) * math.exp(-kappa * duration_s)
                    component_value += self._rng.gauss(0., sigma * math.sqrt(duration_s))
                elif component.kind in {DynamicsModelKind.RANDOM_TELEGRAPH,
                                       DynamicsModelKind.SEMI_MARKOV_TELEGRAPH}:
                    rate = max(0.0, component_params.get("switch_rate", .2))
                    if component.kind is DynamicsModelKind.SEMI_MARKOV_TELEGRAPH:
                        dwell = particle.auxiliary_state.get(
                            f"{component.model_id}:run_length", 0.0)
                        rate *= min(3.0, 1+dwell/max(
                            component_params.get("mean_dwell", 5.0), 1e-9))
                    if self._rng.random() < 1-math.exp(-rate*duration_s):
                        component_value = -component_value
                    component_value = math.copysign(
                        component_params.get("amplitude", abs(component_value) or .6),
                        component_value or 1.0)
                elif component.kind is DynamicsModelKind.STEP:
                    if self._rng.random() < 1-math.exp(-max(
                            0.0, component_params.get("hazard", .04))*duration_s):
                        component_value += self._rng.gauss(
                            0.0, abs(component_params.get("jump_sigma", .5)))
                value += component_value
        state[variable] = value
        return state

    def _optimum(self, state: Mapping[str, float]) -> dict[str, float]:
        controls = dict(self.response_map.reference_controls)
        for (control, variable), gain in self.response_map.correction_gain.items():
            controls[control] = controls.get(control, 0.) - gain * state.get(variable, 0.)
        return controls

    def _risk(self, scenarios: Sequence[ForecastScenario], unknown_probability: float) -> ForecastRisk:
        detectors = sorted({key for item in scenarios for key in item.detector_probabilities})
        variables = sorted({key for item in scenarios for key in item.state})
        controls = sorted({key for item in scenarios for key in item.optimum_controls})
        probability = {detector: sum(item.weight for item in scenarios if item.detector_probabilities[detector] > self.response_map.detector_threshold) for detector in detectors}
        state_means = {variable: sum(item.weight * item.state[variable] for item in scenarios) for variable in variables}
        control_means = {control: sum(item.weight * item.optimum_controls[control] for item in scenarios) for control in controls}
        variance = {variable: sum(item.weight * (item.state[variable] - state_means[variable]) ** 2 for item in scenarios) for variable in variables}
        optimum_variance = {control: sum(item.weight * (item.optimum_controls[control] - control_means[control]) ** 2 for item in scenarios) for control in controls}
        # Weighted spread of model-specific means flags cancellation by a model-averaged action.
        model_sums: dict[str, float] = {}
        model_weights: dict[str, float] = {}
        for item in scenarios:
            model_sums[item.model_id] = model_sums.get(item.model_id, 0.) + item.weight * next(iter(item.state.values()))
            model_weights[item.model_id] = model_weights.get(item.model_id, 0.) + item.weight
        model_means = {model: total / max(model_weights[model], 1e-12) for model, total in model_sums.items()}
        disagreement = max(model_means.values()) - min(model_means.values()) if model_means else 0.
        ordered_risk = sorted(((item.logical_risk, item.weight) for item in scenarios), reverse=True)
        tail, mass = 0., 0.
        for risk, weight in ordered_risk:
            take = min(weight, max(0., .1 - mass))
            tail += risk * take
            mass += take
            if mass >= .1 - 1e-12:
                break
        return ForecastRisk(probability, disagreement, variance, optimum_variance, unknown_probability,
            sum(item.weight * item.logical_risk for item in scenarios), tail / max(mass, 1e-12),
            max(probability.values(), default=0.), sum(item.weight * item.correlation_risk for item in scenarios))

    @staticmethod
    def _weighted_quantiles(scenarios: Sequence[ForecastScenario], variable: str,
                            quantiles: Sequence[float]) -> tuple[float, ...]:
        ordered = sorted((item.state[variable], item.weight) for item in scenarios)
        output: list[float] = []
        cumulative, index = 0.0, 0
        for value, weight in ordered:
            cumulative += weight
            while index < len(quantiles) and cumulative >= quantiles[index]:
                output.append(value)
                index += 1
        if index < len(quantiles):
            output.extend([ordered[-1][0]] * (len(quantiles) - index))
        return tuple(output)

    def _reduce(self, scenarios: Sequence[ForecastScenario]) -> tuple[tuple[ForecastScenario, ...], float]:
        if len(scenarios) <= self.config.maximum_scenarios:
            return tuple(scenarios), 0.0
        # Keep both control-risk tails, then stratify the remainder by cumulative mass.
        ranked = sorted(scenarios, key=lambda item: (item.logical_risk, tuple(sorted(item.optimum_controls.items()))))
        keep_indices = {0, len(ranked) - 1}
        for index in range(self.config.maximum_scenarios - 2):
            keep_indices.add(round((index + .5) * (len(ranked) - 1) / max(1, self.config.maximum_scenarios - 2)))
        kept = [ranked[index] for index in sorted(keep_indices)][:self.config.maximum_scenarios]
        total = sum(item.weight for item in kept)
        reduced = tuple(ForecastScenario(item.horizon_s, item.activation_offset_s, item.model_id, item.state,
            item.optimum_controls, item.detector_probabilities, item.weight / total, item.logical_risk,
            item.correlation_risk, item.leakage_risk, item.context_id) for item in kept)
        omitted = max(0., 1 - total)
        control_span = max((max((abs(value) for value in item.optimum_controls.values()), default=0.) for item in scenarios), default=0.)
        return reduced, omitted * control_span

    def forecast(self, dynamics: DynamicsPosterior, controls: Mapping[str, float], horizons_s: Sequence[float], latency: LatencyModel,
                 *, context_id: str = "default") -> ForecastBundle:
        if not horizons_s or any(horizon < 0 for horizon in horizons_s):
            raise ValueError("at least one non-negative horizon is required")
        invalid: list[str] = []
        validity = min(max(horizons_s), self.config.maximum_validity_horizon_s)
        if dynamics.unknown_model_probability >= self.config.unknown_probability_limit:
            invalid.append("unknown dynamics probability exceeds forecast authority")
            validity = 0.
        scenarios_by_horizon: dict[float, tuple[ForecastScenario, ...]] = {}
        risks: dict[float, ForecastRisk] = {}
        quantiles: dict[float, dict[str, tuple[float, float, float]]] = {}
        count_moments: dict[float, dict[str, tuple[float, float]]] = {}
        reduction_bound = 0.0
        for horizon in sorted(set(horizons_s)):
            scenarios = []
            for particle in dynamics.particles:
                activation = max(0., horizon + latency.mean_s + self._rng.gauss(0., latency.jitter_s))
                state = self._propagate(particle, max(activation, 0.))
                optimum = self._optimum(state)
                detector = {detector_id: self.observation_model.probability(detector_id, state, optimum) for detector_id in self.observation_model.responses}
                mean_detector = sum(detector.values()) / max(1, len(detector))
                correlation = max(detector.values(), default=0.) - min(detector.values(), default=0.)
                logical = min(1., mean_detector ** self.config.logical_risk_power)
                scenarios.append(ForecastScenario(horizon, activation, particle.model_id, state, optimum,
                    detector, particle.weight, logical, correlation, 0., context_id))
            reduced, bound = self._reduce(scenarios)
            reduction_bound = max(reduction_bound, bound)
            scenarios_by_horizon[horizon] = reduced
            risks[horizon] = self._risk(reduced, dynamics.unknown_model_probability)
            variables = sorted({name for item in reduced for name in item.state})
            quantiles[horizon] = {name: self._weighted_quantiles(reduced, name, (.05, .5, .95))
                                  for name in variables}
            count_moments[horizon] = {detector_id: (
                sum(item.weight * item.detector_probabilities[detector_id] for item in reduced),
                sum(item.weight * item.detector_probabilities[detector_id] * (1-item.detector_probabilities[detector_id]) for item in reduced))
                for detector_id in self.observation_model.responses}
            if horizon > validity:
                invalid.append(f"horizon {horizon} exceeds calibrated validity horizon")
        calibration = self.scorer.summary()
        if calibration.interval_coverage is not None and calibration.interval_coverage < self.config.calibration_coverage_floor:
            invalid.append("empirical interval coverage is below forecast authority threshold")
            validity = 0.
        return ForecastBundle("stage4.v2", dynamics.region_id, dynamics.timestamp_s, latency, scenarios_by_horizon,
            risks, validity, calibration, tuple(dict.fromkeys(invalid)), quantiles, count_moments, reduction_bound, context_id)
