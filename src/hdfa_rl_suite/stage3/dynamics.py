"""Finite-bank joint switching state-space inference using the Stage-2 emission likelihood."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import random
from typing import Mapping, Sequence

from hdfa_rl_suite.stage1.schema import TelemetryRegionView
from hdfa_rl_suite.stage2.inference import QuadraticLogitObservationModel
from hdfa_rl_suite.stage2.schema import PhysicalStatePosterior, StateSchema

from .schema import (
    ChangeAlarm, DynamicsModelKind, DynamicsModelSpec, DynamicsParticle, DynamicsPosterior,
    FamiliarProcessState, HierarchicalComponent, ModelEvidence,
    RecurringRegimeMatch,
)


def default_model_bank(variable_id: str) -> tuple[DynamicsModelSpec, ...]:
    """MVP bank: every claimed forecast family has an explicit alternative and unknown fallback."""
    oscillator = DynamicsModelSpec("oscillator-component", DynamicsModelKind.OSCILLATOR, variable_id, 0., {"omega": 2.0, "damping": .02, "noise": .02})
    slow_ou = DynamicsModelSpec("ou-component", DynamicsModelKind.ORNSTEIN_UHLENBECK, variable_id, 0., {"kappa": .2, "sigma": .04, "mean": 0.})
    return (
        DynamicsModelSpec("constant", DynamicsModelKind.CONSTANT, variable_id, .08),
        DynamicsModelSpec("random-walk", DynamicsModelKind.RANDOM_WALK, variable_id, .10, {"diffusion": .04}),
        DynamicsModelSpec("ou", DynamicsModelKind.ORNSTEIN_UHLENBECK, variable_id, .10, {"kappa": .3, "sigma": .06, "mean": 0.}),
        DynamicsModelSpec("oscillator", DynamicsModelKind.OSCILLATOR, variable_id, .15, {"omega": 2.0, "damping": .02, "noise": .02}),
        DynamicsModelSpec("rtn", DynamicsModelKind.RANDOM_TELEGRAPH, variable_id, .10, {"amplitude": .6, "switch_rate": .2}),
        DynamicsModelSpec("semi-markov-rtn", DynamicsModelKind.SEMI_MARKOV_TELEGRAPH, variable_id, .08, {"amplitude": .6, "switch_rate": .2, "mean_dwell": 5.}),
        DynamicsModelSpec("step", DynamicsModelKind.STEP, variable_id, .08, {"hazard": .04, "jump_sigma": .5}),
        DynamicsModelSpec("oscillator-plus-ou", DynamicsModelKind.ADDITIVE_COMPOSITE, variable_id, .12, {}, .08, (oscillator, slow_ou)),
        DynamicsModelSpec("unknown", DynamicsModelKind.UNKNOWN, variable_id, .09, {"scale": .25}),
    )


def extended_structured_model_bank(variable_id: str) -> tuple[DynamicsModelSpec, ...]:
    """Default bank plus explicit nested-switching and OU/step factorizations."""
    oscillator = DynamicsModelSpec(
        "oscillator-component", DynamicsModelKind.OSCILLATOR, variable_id, 0.,
        {"omega": 2.0, "damping": .02, "noise": .02})
    slow_ou = DynamicsModelSpec(
        "ou-component", DynamicsModelKind.ORNSTEIN_UHLENBECK, variable_id, 0.,
        {"kappa": .2, "sigma": .04, "mean": 0.})
    local_switch = DynamicsModelSpec(
        "semi-markov-component", DynamicsModelKind.SEMI_MARKOV_TELEGRAPH,
        variable_id, 0., {"amplitude": .35, "switch_rate": .3,
                           "mean_dwell": 3.0})
    abrupt_step = DynamicsModelSpec(
        "step-component", DynamicsModelKind.STEP, variable_id, 0.,
        {"hazard": .04, "jump_sigma": .4})
    base = default_model_bank(variable_id)
    unknown = base[-1]
    return (*base[:-1],
            DynamicsModelSpec(
                "nested-oscillator-semi-markov", DynamicsModelKind.ADDITIVE_COMPOSITE,
                variable_id, .08, {}, .10, (oscillator, local_switch)),
            DynamicsModelSpec(
                "ou-plus-step", DynamicsModelKind.ADDITIVE_COMPOSITE,
                variable_id, .08, {}, .10, (slow_ou, abrupt_step)),
            unknown)


@dataclass(frozen=True)
class DynamicsConfig:
    seed: int = 0
    particle_count: int = 384
    selected_window: int | None = None
    fixed_lag: int = 8
    change_alarm_probability: float = .5
    unknown_alarm_probability: float = .35
    parameter_jitter_fraction: float = .10
    recurrence_distance: float = .20
    model_probability_floor: float = 1e-6


def _normalise(log_weights: Sequence[float]) -> list[float]:
    maximum = max(log_weights)
    raw = [math.exp(value - maximum) for value in log_weights]
    total = sum(raw)
    return [value / total for value in raw]


class JointDynamicsEngine:
    """Jointly update state, regime and model particles from raw detector-count emissions.

    The engine deliberately does *not* consume Stage-2 posterior means as observations.
    Stage-2 can seed proposals, while every particle is weighted by p(D | x, u, c).
    """

    def __init__(self, schema: StateSchema, observation_model: QuadraticLogitObservationModel,
                 model_bank: Sequence[DynamicsModelSpec], config: DynamicsConfig = DynamicsConfig()) -> None:
        self.schema, self.observation_model, self.models, self.config = schema, observation_model, tuple(model_bank), config
        if not self.models or any(model.variable_id not in {item.variable_id for item in schema.variables} for model in self.models):
            raise ValueError("model bank must target state-schema variables")
        if not any(model.kind is DynamicsModelKind.UNKNOWN for model in self.models):
            raise ValueError("model bank requires an explicit unknown model")
        self._rng = random.Random(config.seed)
        self._particles: list[DynamicsParticle] = []
        self._history: list[DynamicsPosterior] = []
        self._regime_memory: list[tuple[str, str, float]] = []
        self._last_familiar: FamiliarProcessState | None = None

    def _counts(self, view: TelemetryRegionView) -> dict[str, tuple[int, int]]:
        window = self.config.selected_window or max(item.window_size for item in view.count_factors)
        return {item.detector_id: (item.events, item.exposures) for item in view.count_factors if item.window_size == window}

    def _initialise(self, state_prior: PhysicalStatePosterior | None) -> None:
        choices = []
        for model in self.models:
            choices.extend([model] * max(1, round(model.prior_probability * 100)))
        proposal = list(state_prior.samples) if state_prior and state_prior.samples else []
        variables = self.schema.variables
        self._particles = []
        for index in range(self.config.particle_count):
            model = choices[index % len(choices)]
            state = dict(proposal[index % len(proposal)].state) if proposal else {
                item.variable_id: self._rng.uniform(item.lower, item.upper) for item in variables
            }
            auxiliary = {"velocity": self._rng.gauss(0., .1), "run_length": 0.}
            parameter_state = {}
            for name, value in model.parameters.items():
                if isinstance(value, (int, float)):
                    scale = self.config.parameter_jitter_fraction * max(abs(value), 1e-3)
                    parameter_state[name] = value + self._rng.gauss(0., scale)
            component_state = {component.model_id: state[model.variable_id] / max(1, len(model.components)) for component in model.components}
            regime = "high" if state[model.variable_id] >= 0 else "low"
            self._particles.append(DynamicsParticle(model.model_id, state, auxiliary, regime, False,
                1 / self.config.particle_count, component_state, parameter_state, f"init:{index}"))

    def _propagate(self, particle: DynamicsParticle, model: DynamicsModelSpec, dt: float) -> DynamicsParticle:
        state, auxiliary = dict(particle.state), dict(particle.auxiliary_state)
        variable, change = model.variable_id, False
        parameter_state = dict(particle.parameter_state)

        def transition(value: float, spec: DynamicsModelSpec, prefix: str = "") -> tuple[float, bool]:
            nonlocal auxiliary
            parameters = dict(spec.parameters)
            if not prefix:
                parameters.update(parameter_state)
            changed = False
            if spec.kind is DynamicsModelKind.RANDOM_WALK:
                value += self._rng.gauss(0., max(0., parameters.get("diffusion", .04)) * math.sqrt(dt))
            elif spec.kind is DynamicsModelKind.ORNSTEIN_UHLENBECK:
                kappa, mean, sigma = max(0., parameters.get("kappa", .3)), parameters.get("mean", 0.), max(0., parameters.get("sigma", .06))
                value += -kappa * (value - mean) * dt + self._rng.gauss(0., sigma * math.sqrt(dt))
            elif spec.kind is DynamicsModelKind.OSCILLATOR:
                omega, damping, noise = max(1e-6, abs(parameters.get("omega", 2.))), max(0., parameters.get("damping", .02)), max(0., parameters.get("noise", .02))
                velocity_key = f"{prefix}velocity"
                velocity = auxiliary.get(velocity_key, auxiliary.get("velocity", 0.))
                cosine, sine = math.cos(omega * dt), math.sin(omega * dt)
                next_value = cosine * value + sine * velocity / omega
                auxiliary[velocity_key] = (cosine * velocity - omega * sine * value) * math.exp(-damping * dt) + self._rng.gauss(0., noise * math.sqrt(dt))
                value = next_value * math.exp(-damping * dt) + self._rng.gauss(0., noise * math.sqrt(dt))
            elif spec.kind in {DynamicsModelKind.RANDOM_TELEGRAPH, DynamicsModelKind.SEMI_MARKOV_TELEGRAPH}:
                rate = max(0., parameters.get("switch_rate", .2))
                run_key = f"{prefix}run_length"
                if spec.kind is DynamicsModelKind.SEMI_MARKOV_TELEGRAPH:
                    rate *= min(3., 1 + auxiliary.get(run_key, 0.) / max(parameters.get("mean_dwell", 5.), 1e-9))
                if self._rng.random() < 1 - math.exp(-rate * dt):
                    value, changed, auxiliary[run_key] = -value if value else parameters.get("amplitude", .6), True, 0.
                else:
                    auxiliary[run_key] = auxiliary.get(run_key, 0.) + dt
                amplitude = abs(parameters.get("amplitude", .6))
                value = math.copysign(amplitude, value if value else 1.)
            elif spec.kind is DynamicsModelKind.STEP:
                if self._rng.random() < 1 - math.exp(-max(0., parameters.get("hazard", .04)) * dt):
                    value += self._rng.gauss(0., abs(parameters.get("jump_sigma", .5)))
                    changed = True
            elif spec.kind is DynamicsModelKind.UNKNOWN:
                scale = abs(parameters.get("scale", .25)) * math.sqrt(dt)
                value += self._rng.gauss(0., scale) / max(abs(self._rng.gauss(0., 1.)), .15)
            return value, changed

        component_state = dict(particle.component_state)
        if model.kind is DynamicsModelKind.ADDITIVE_COMPOSITE:
            total = 0.0
            for component in model.components:
                value, changed = transition(component_state.get(component.model_id, 0.), component, f"{component.model_id}:")
                component_state[component.model_id] = value
                total += value
                change = change or changed
            value = total
        else:
            value, change = transition(state[variable], model)
        state[variable] = value
        regime = "high" if value >= 0 else "low"
        return DynamicsParticle(model.model_id, state, auxiliary, regime, change, particle.weight,
                                component_state, parameter_state, particle.lineage_id)

    def _resample(self, particles: Sequence[DynamicsParticle], weights: Sequence[float]) -> list[DynamicsParticle]:
        step, cursor, index, cumulative = 1 / len(particles), self._rng.random() / len(particles), 0, weights[0]
        output: list[DynamicsParticle] = []
        for _ in particles:
            while cursor > cumulative and index < len(particles) - 1:
                index += 1
                cumulative += weights[index]
            selected = particles[index]
            output.append(DynamicsParticle(selected.model_id, selected.state, selected.auxiliary_state,
                selected.regime, selected.changepoint, 1 / len(particles), selected.component_state,
                selected.parameter_state, selected.lineage_id))
            cursor += step
        return output

    def update(self, view: TelemetryRegionView, applied_controls: Mapping[str, float], timestamp_s: float,
               *, state_prior: PhysicalStatePosterior | None = None) -> DynamicsPosterior:
        if view.region_id != self.schema.region_id:
            raise ValueError("telemetry region and dynamics schema must match")
        if not self._particles:
            self._initialise(state_prior)
        previous_time = self._history[-1].timestamp_s if self._history else timestamp_s
        dt = max(timestamp_s - previous_time, 1e-6)
        counts = self._counts(view)
        model_by_id = {model.model_id: model for model in self.models}
        proposed = [self._propagate(item, model_by_id[item.model_id], dt) for item in self._particles]
        log_weights = [self.observation_model.log_likelihood(counts, item.state, applied_controls) - model_by_id[item.model_id].complexity_penalty for item in proposed]
        weights = _normalise(log_weights)
        weighted = [DynamicsParticle(item.model_id, item.state, item.auxiliary_state, item.regime,
            item.changepoint, weight, item.component_state, item.parameter_state,
            f"{item.lineage_id}>{len(self._history)}:{index}") for index, (item, weight) in enumerate(zip(proposed, weights))]
        evidence = {model.model_id: sum(item.weight for item in weighted if item.model_id == model.model_id) for model in self.models}
        unknown_particles = [item for item in weighted if model_by_id[item.model_id].kind is DynamicsModelKind.UNKNOWN]
        unknown_probability = 1.0 if len(unknown_particles) == len(weighted) else sum(item.weight for item in unknown_particles)
        change_probability = sum(item.weight for item in weighted if item.changepoint)
        state_mean = {variable.variable_id: sum(item.weight * item.state[variable.variable_id] for item in weighted) for variable in self.schema.variables}
        variable_id = self.models[0].variable_id
        previous_value = (self._history[-1].current_state_mean.get(variable_id, 0.0)
                          if self._history else state_mean[variable_id])
        abrupt_delta = state_mean[variable_id]-previous_value
        posterior_scale = 0.0
        if state_prior is not None and state_prior.covariance:
            posterior_scale = math.sqrt(max(state_prior.covariance[0][0], 0.0))
        abrupt = abs(abrupt_delta) > max(.06, 2.5*posterior_scale)
        if abrupt:
            change_probability = max(change_probability, .95)
        self._particles = self._resample(weighted, weights)
        components_list: list[HierarchicalComponent] = []
        for item in self.models:
            if evidence[item.model_id] <= .01:
                continue
            timescale = (2 * math.pi / max(abs(item.parameters.get("omega", 0.)), 1e-9)
                         if "omega" in item.parameters else 1 / max(item.parameters.get("switch_rate", item.parameters.get("kappa", .2)), 1e-9))
            parent = f"{item.model_id}:{item.variable_id}" if item.components else None
            components_list.append(HierarchicalComponent(f"{item.model_id}:{item.variable_id}", item.model_id,
                                                          item.variable_id, timescale, evidence[item.model_id]))
            for child in item.components:
                child_scale = 2 * math.pi / max(abs(child.parameters.get("omega", 0.)), 1e-9) if "omega" in child.parameters else 1 / max(child.parameters.get("kappa", .2), 1e-9)
                components_list.append(HierarchicalComponent(f"{item.model_id}/{child.model_id}", child.model_id,
                                                              child.variable_id, child_scale, evidence[item.model_id], parent))
        components = tuple(components_list)
        invalidity: list[str] = []
        if unknown_probability >= self.config.unknown_alarm_probability:
            invalidity.append("unknown model posterior exceeds safe forecast threshold")
        dominant = max(evidence, key=evidence.get)
        current_value = state_mean[self.models[0].variable_id]
        candidates = [(identifier, value) for identifier, model_id, value in self._regime_memory if model_id == dominant]
        match = min(candidates, key=lambda item: abs(item[1] - current_value), default=None)
        recurrence = None
        if match and abs(match[1] - current_value) <= self.config.recurrence_distance:
            recurrence = RecurringRegimeMatch(match[0], max(0., 1 - abs(match[1] - current_value) / self.config.recurrence_distance),
                                              {self.models[0].variable_id: match[1]}, True)
        elif change_probability >= self.config.change_alarm_probability:
            self._regime_memory.append((f"regime:{len(self._regime_memory)}", dominant, current_value))
        model_by_kind: dict[DynamicsModelKind, float] = {}
        for model in self.models:
            model_by_kind[model.kind] = model_by_kind.get(model.kind, 0.0)+evidence[model.model_id]
        familiar_kinds = (
            DynamicsModelKind.OSCILLATOR, DynamicsModelKind.RANDOM_TELEGRAPH,
            DynamicsModelKind.SEMI_MARKOV_TELEGRAPH,
            DynamicsModelKind.ORNSTEIN_UHLENBECK, DynamicsModelKind.STEP,
            DynamicsModelKind.ADDITIVE_COMPOSITE,
        )
        family_kind = max(familiar_kinds, key=lambda kind: model_by_kind.get(kind, 0.0))
        family_confidence = model_by_kind.get(family_kind, 0.0)
        dominant_particles = [item for item in weighted
                              if model_by_id[item.model_id].kind is family_kind]
        dominant_mass = sum(item.weight for item in dominant_particles)
        velocity = (sum(item.weight*item.auxiliary_state.get("velocity", 0.0)
                        for item in dominant_particles)/max(dominant_mass, 1e-12))
        regime = ("high" if current_value >= 0 else "low")
        regime_id = f"{family_kind.value}:{regime}"
        amplitude = period = phase = dwell = transition_probability = None
        common_component = local_component = smooth_component = abrupt_component = 0.0
        if family_kind is DynamicsModelKind.OSCILLATOR:
            omega = sum(item.weight*abs(item.parameter_state.get(
                "omega", model_by_id[item.model_id].parameters.get("omega", 2.0)))
                for item in dominant_particles)/max(dominant_mass, 1e-12)
            period = 2*math.pi/max(omega, 1e-9)
            amplitude = math.sqrt(current_value**2+(velocity/max(omega, 1e-9))**2)
            phase = math.atan2(current_value*omega, velocity)
        elif family_kind in {DynamicsModelKind.RANDOM_TELEGRAPH,
                             DynamicsModelKind.SEMI_MARKOV_TELEGRAPH}:
            dwell = sum(item.weight*item.auxiliary_state.get("run_length", 0.0)
                        for item in dominant_particles)/max(dominant_mass, 1e-12)
            rate = sum(item.weight*max(0.0, item.parameter_state.get(
                "switch_rate", model_by_id[item.model_id].parameters.get("switch_rate", .2)))
                for item in dominant_particles)/max(dominant_mass, 1e-12)
            transition_probability = 1-math.exp(-rate*dt)
        elif family_kind is DynamicsModelKind.ORNSTEIN_UHLENBECK:
            smooth_component = current_value
        elif family_kind is DynamicsModelKind.ADDITIVE_COMPOSITE:
            for item in dominant_particles:
                specification = model_by_id[item.model_id]
                component_kinds = {component.model_id: component.kind
                                   for component in specification.components}
                for component_id, value in item.component_state.items():
                    contribution = item.weight*value/max(dominant_mass, 1e-12)
                    kind = component_kinds.get(component_id)
                    if kind is DynamicsModelKind.OSCILLATOR:
                        common_component += contribution
                    elif kind in {DynamicsModelKind.RANDOM_TELEGRAPH,
                                  DynamicsModelKind.SEMI_MARKOV_TELEGRAPH}:
                        local_component += contribution
                    elif kind is DynamicsModelKind.STEP:
                        abrupt_component += contribution
                    else:
                        smooth_component += contribution
            residual = current_value-common_component-local_component-smooth_component-abrupt_component
            local_component += residual
        familiar_invalid: list[str] = []
        if family_confidence < .35:
            familiar_invalid.append("familiar model posterior below fast-path threshold")
        if unknown_probability >= self.config.unknown_alarm_probability:
            familiar_invalid.append("unknown-model probability revokes familiar fast path")
        phase_coherent = (family_kind is DynamicsModelKind.OSCILLATOR
                          and family_confidence >= .55 and amplitude is not None
                          and amplitude > 1e-4)
        warm_started = bool(
            self._last_familiar is not None
            and self._last_familiar.family == family_kind.value
            and (family_kind not in {DynamicsModelKind.RANDOM_TELEGRAPH,
                                     DynamicsModelKind.SEMI_MARKOV_TELEGRAPH}
                 or self._last_familiar.regime_id == regime_id))
        familiar = FamiliarProcessState(
            family_kind.value, family_confidence, regime_id, current_value,
            velocity, amplitude, period, phase, dwell, transition_probability,
            common_component, local_component, smooth_component,
            abrupt_component if family_kind is DynamicsModelKind.ADDITIVE_COMPOSITE
            else (abrupt_delta if abrupt else 0.0), phase_coherent, warm_started,
            not familiar_invalid and (warm_started or abrupt or recurrence is not None),
            tuple(familiar_invalid))
        self._last_familiar = familiar
        posterior = DynamicsPosterior("stage3.v2", view.region_id, timestamp_s, tuple(weighted), state_mean,
            ModelEvidence(evidence, unknown_probability, math.log(sum(math.exp(value - max(log_weights)) for value in log_weights)) + max(log_weights) - math.log(len(log_weights))),
            ChangeAlarm(change_probability, timestamp_s, "high" if change_probability >= self.config.change_alarm_probability else "low", view.region_id),
            components, {item.model_id: {name: sum(p.weight * p.parameter_state.get(name, value)
                for p in weighted if p.model_id == item.model_id) / max(evidence[item.model_id], 1e-12)
                for name, value in item.parameters.items()} for item in self.models}, unknown_probability, tuple(invalidity),
            "Causal likelihood-weighted particle filter; smooth_history() applies a non-causal fixed-lag reference.",
            recurrence, None, familiar)
        self._history.append(posterior)
        return posterior

    def smooth_history(self) -> tuple[DynamicsPosterior, ...]:
        """Non-causal fixed-lag reference using future likelihood-weighted state summaries."""
        causal = self._history[-self.config.fixed_lag:]
        output: list[DynamicsPosterior] = []
        for index, posterior in enumerate(causal):
            future = causal[index:]
            weights = [math.exp(-.5 * offset) for offset in range(len(future))]
            total = sum(weights)
            smoothed = {name: sum(weight * item.current_state_mean[name] for weight, item in zip(weights, future)) / total
                        for name in posterior.current_state_mean}
            divergence = math.sqrt(sum((smoothed[name] - posterior.current_state_mean[name]) ** 2 for name in smoothed))
            output.append(replace(posterior, current_state_mean=smoothed, offline_divergence=divergence,
                                  online_approximation_note="Non-causal exponential fixed-lag smoother reference."))
        return tuple(output)
