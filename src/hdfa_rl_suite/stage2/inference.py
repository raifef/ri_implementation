"""Control-conditioned count likelihoods with Gaussian and particle posterior paths."""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Mapping, Sequence

from hdfa_rl_suite.stage1.schema import TelemetryRegionView

from .schema import (
    DetectorResponse, InferenceValidity, InterventionDesignRequest, LatentVariable,
    ObservabilityReport, PhysicalStatePosterior, PosteriorPredictiveCheck, PosteriorSample,
    StateSchema,
)


@dataclass(frozen=True, slots=True)
class _StateConditionedResponse:
    """State-only terms cached for repeated control evaluations of one scenario."""

    prefix: float
    state_quadratic: float
    control_linear: tuple[tuple[str, float], ...]
    control_quadratic: tuple[tuple[str, str, float], ...]
    state_control: tuple[tuple[float, str, float], ...]


@dataclass(frozen=True, slots=True)
class _IndexedStateConditionedResponse:
    """State-conditioned response indexed into a fixed local control vector."""

    prefix: float
    state_quadratic: float
    control_linear: tuple[tuple[int, float], ...]
    control_quadratic: tuple[tuple[int, int, float], ...]
    state_control: tuple[tuple[float, int, float], ...]


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, value))))


def _normalise_log_weights(log_weights: Sequence[float]) -> list[float]:
    maximum = max(log_weights)
    weights = [math.exp(item - maximum) for item in log_weights]
    total = sum(weights)
    return [item / total for item in weights]


def _jacobi_eigensystem(matrix: list[list[float]], tolerance: float = 1e-10) -> tuple[list[float], list[list[float]]]:
    """Small symmetric-matrix eigensolver to keep the MVP dependency-free."""
    size = len(matrix)
    values = [row[:] for row in matrix]
    vectors = [[float(i == j) for j in range(size)] for i in range(size)]
    for _ in range(64 * max(1, size * size)):
        p, q, magnitude = 0, 0, 0.0
        for i in range(size):
            for j in range(i + 1, size):
                if abs(values[i][j]) > magnitude:
                    p, q, magnitude = i, j, abs(values[i][j])
        if magnitude < tolerance:
            break
        angle = .5 * math.atan2(2 * values[p][q], values[q][q] - values[p][p])
        cosine, sine = math.cos(angle), math.sin(angle)
        old_pp, old_qq, old_pq = values[p][p], values[q][q], values[p][q]
        values[p][p] = cosine*cosine*old_pp - 2*sine*cosine*old_pq + sine*sine*old_qq
        values[q][q] = sine*sine*old_pp + 2*sine*cosine*old_pq + cosine*cosine*old_qq
        values[p][q] = values[q][p] = 0.0
        for i in range(size):
            if i not in (p, q):
                old_ip, old_iq = values[i][p], values[i][q]
                values[i][p] = values[p][i] = cosine*old_ip - sine*old_iq
                values[i][q] = values[q][i] = sine*old_ip + cosine*old_iq
            old_vp, old_vq = vectors[i][p], vectors[i][q]
            vectors[i][p] = cosine*old_vp - sine*old_vq
            vectors[i][q] = sine*old_vp + cosine*old_vq
    order = sorted(range(size), key=lambda i: values[i][i], reverse=True)
    return [values[i][i] for i in order], [[vectors[row][i] for row in range(size)] for i in order]


@dataclass(frozen=True)
class InferenceConfig:
    seed: int = 0
    particle_count: int = 256
    selected_window: int | None = None
    observability_eigenvalue_threshold: float = 1e-5
    ood_residual_threshold: float = 4.0
    likelihood: str = "correlated_count"
    particle_rejuvenation_fraction: float = .03
    nominal_validity_horizon_s: float = 1.0


class QuadraticLogitObservationModel:
    """Sparse empirical observation model p(D | x, u, context), with explicit discrepancy."""

    version = "quadratic-logit.v1"

    def __init__(self, schema: StateSchema, responses: Sequence[DetectorResponse],
                 pair_correlations: Mapping[tuple[str, str], float] | None = None) -> None:
        self.schema = schema
        self.responses = {response.detector_id: response for response in responses}
        self.pair_correlations = {tuple(sorted(key)): value for key, value in (pair_correlations or {}).items()}
        if set(self.responses) == set():
            raise ValueError("At least one detector response is required")

    def probability(self, detector_id: str, state: Mapping[str, float], controls: Mapping[str, float],
                    context_id: str | None = None) -> float:
        response = self.responses[detector_id]
        logit = response.intercept + response.context_intercepts.get(context_id or "", 0.0)
        logit += sum(coefficient * state.get(variable, 0.0) for variable, coefficient in response.state_linear.items())
        logit += sum(coefficient * controls.get(control, 0.0) for control, coefficient in response.control_linear.items())
        logit += sum(coefficient * state.get(left, 0.0) * state.get(right, 0.0) for (left, right), coefficient in response.state_quadratic.items())
        logit += sum(coefficient * controls.get(left, 0.0) * controls.get(right, 0.0) for (left, right), coefficient in response.control_quadratic.items())
        logit += sum(coefficient * state.get(variable, 0.0) * controls.get(control, 0.0) for (variable, control), coefficient in response.state_control.items())
        return min(1 - 1e-9, max(1e-9, _sigmoid(logit)))

    def prepare_state(self, state: Mapping[str, float],
                      context_id: str | None = None) -> Mapping[str, _StateConditionedResponse]:
        """Precompute state-only logit terms without changing the observation model.

        Stage 5 evaluates the same forecast scenario under many candidate controls.  The
        state is immutable during that coordinate search, so recomputing its polynomial
        terms is pure overhead.  Terms are kept in their original groups and insertion
        order to retain the numerical operation order of :meth:`probability`.
        """
        prepared: dict[str, _StateConditionedResponse] = {}
        for detector_id, response in self.responses.items():
            prefix = response.intercept + response.context_intercepts.get(context_id or "", 0.0)
            prefix += sum(coefficient * state.get(variable, 0.0)
                          for variable, coefficient in response.state_linear.items())
            state_quadratic = sum(coefficient * state.get(left, 0.0) * state.get(right, 0.0)
                                  for (left, right), coefficient in response.state_quadratic.items())
            prepared[detector_id] = _StateConditionedResponse(
                prefix,
                state_quadratic,
                tuple(response.control_linear.items()),
                tuple((left, right, coefficient)
                      for (left, right), coefficient in response.control_quadratic.items()),
                tuple((state.get(variable, 0.0), control, coefficient)
                      for (variable, control), coefficient in response.state_control.items()),
            )
        return prepared

    @staticmethod
    def probability_prepared(prepared: _StateConditionedResponse,
                             controls: Mapping[str, float]) -> float:
        """Evaluate a state-conditioned response at a candidate control."""
        logit = prepared.prefix
        if prepared.control_linear:
            logit += sum(coefficient * controls.get(control, 0.0)
                         for control, coefficient in prepared.control_linear)
        logit += prepared.state_quadratic
        if prepared.control_quadratic:
            logit += sum(coefficient * controls.get(left, 0.0) * controls.get(right, 0.0)
                         for left, right, coefficient in prepared.control_quadratic)
        if prepared.state_control:
            logit += sum(coefficient * state_value * controls.get(control, 0.0)
                         for state_value, control, coefficient in prepared.state_control)
        return min(1 - 1e-9, max(1e-9, _sigmoid(logit)))

    def prepare_state_for_controls(self, state: Mapping[str, float], controls: Sequence[str],
                                   context_id: str | None = None
                                   ) -> tuple[tuple[str, _IndexedStateConditionedResponse], ...]:
        """Compile a scenario for repeated evaluation on a fixed local control vector."""
        indices = {control: index for index, control in enumerate(controls)}
        return tuple((detector_id, _IndexedStateConditionedResponse(
            response.prefix,
            response.state_quadratic,
            tuple((indices.get(control, -1), coefficient)
                  for control, coefficient in response.control_linear),
            tuple((indices.get(left, -1), indices.get(right, -1), coefficient)
                  for left, right, coefficient in response.control_quadratic),
            tuple((state_value, indices.get(control, -1), coefficient)
                  for state_value, control, coefficient in response.state_control),
        )) for detector_id, response in self.prepare_state(state, context_id).items())

    @staticmethod
    def probability_prepared_values(prepared: _IndexedStateConditionedResponse,
                                    controls: Sequence[float]) -> float:
        """Evaluate an indexed state-conditioned response without mapping lookups."""
        logit = prepared.prefix
        if prepared.control_linear:
            logit += sum(coefficient * controls[index] if index >= 0 else 0.
                         for index, coefficient in prepared.control_linear)
        logit += prepared.state_quadratic
        if prepared.control_quadratic:
            logit += sum(coefficient * (controls[left] if left >= 0 else 0.)
                         * (controls[right] if right >= 0 else 0.)
                         for left, right, coefficient in prepared.control_quadratic)
        if prepared.state_control:
            logit += sum(coefficient * state_value * (controls[index] if index >= 0 else 0.)
                         for state_value, index, coefficient in prepared.state_control)
        return min(1 - 1e-9, max(1e-9, _sigmoid(logit)))

    def gradient(self, detector_id: str, state: Mapping[str, float], controls: Mapping[str, float],
                 context_id: str | None = None) -> dict[str, float]:
        response, q = self.responses[detector_id], self.probability(detector_id, state, controls, context_id)
        output: dict[str, float] = {}
        for variable in (item.variable_id for item in self.schema.variables):
            derivative = response.state_linear.get(variable, 0.0)
            for (left, right), coefficient in response.state_quadratic.items():
                if variable == left:
                    derivative += coefficient * state.get(right, 0.0)
                if variable == right:
                    derivative += coefficient * state.get(left, 0.0)
            derivative += sum(coefficient * controls.get(control, 0.0) for (name, control), coefficient in response.state_control.items() if name == variable)
            output[variable] = q * (1 - q) * derivative
        return output

    @staticmethod
    def _beta_binomial(events: int, exposures: int, probability: float, dispersion: float) -> float:
        concentration = max(2.0, 1.0 / max(dispersion * dispersion, 1e-9))
        alpha, beta = probability * concentration, (1 - probability) * concentration
        return (math.lgamma(exposures + 1) - math.lgamma(events + 1) - math.lgamma(exposures - events + 1)
                + math.lgamma(events + alpha) + math.lgamma(exposures - events + beta)
                - math.lgamma(exposures + alpha + beta) + math.lgamma(alpha + beta)
                - math.lgamma(alpha) - math.lgamma(beta))

    def log_likelihood(self, counts: Mapping[str, tuple[int, int]], state: Mapping[str, float],
                       controls: Mapping[str, float], context_id: str | None = None) -> float:
        score = 0.0
        for detector_id, (events, exposures) in counts.items():
            if not exposures or detector_id not in self.responses:
                continue
            q = self.probability(detector_id, state, controls, context_id)
            discrepancy = self.responses[detector_id].discrepancy_scale
            score += (self._beta_binomial(events, exposures, q, discrepancy) if discrepancy > 0
                      else events * math.log(q) + (exposures - events) * math.log(1 - q))
        return score

    def event_log_likelihood(self, view: TelemetryRegionView, state: Mapping[str, float],
                             controls: Mapping[str, float]) -> float:
        score = 0.0
        for event in view.events:
            if not event.exposure or event.ambiguous_policy or event.detector_id not in self.responses:
                continue
            active = event.active_controls or controls
            q = self.probability(event.detector_id, state, active, event.context_id)
            score += math.log(q if event.value else 1 - q)
        return score

    def pair_log_likelihood(self, view: TelemetryRegionView, state: Mapping[str, float],
                            controls: Mapping[str, float]) -> float:
        """Sparse correlated Bernoulli likelihood using valid Frechet-bounded p11."""
        score = 0.0
        for pair in view.pair_counts:
            key = tuple(sorted((pair.detector_a, pair.detector_b)))
            if key not in self.pair_correlations:
                continue
            p1 = self.probability(pair.detector_a, state, controls, pair.context_id)
            p2 = self.probability(pair.detector_b, state, controls, pair.context_id)
            covariance = self.pair_correlations[key] * math.sqrt(p1 * (1 - p1) * p2 * (1 - p2))
            p11 = min(min(p1, p2) - 1e-9, max(max(0., p1 + p2 - 1.) + 1e-9, p1 * p2 + covariance))
            probabilities = (1 - p1 - p2 + p11, p2 - p11, p1 - p11, p11)
            counts = (pair.n00, pair.n01, pair.n10, pair.n11)
            joint = sum(count * math.log(max(probability, 1e-12)) for count, probability in zip(counts, probabilities))
            # Marginal count terms are already present; add only the correlation likelihood ratio.
            independent = sum(count * math.log(max(probability, 1e-12)) for count, probability in zip(
                counts, ((1-p1)*(1-p2), (1-p1)*p2, p1*(1-p2), p1*p2)))
            score += joint - independent
        return score


class PhysicalInferenceEngine:
    def __init__(self, schema: StateSchema, observation_model: QuadraticLogitObservationModel,
                 config: InferenceConfig = InferenceConfig()) -> None:
        self.schema, self.model, self.config = schema, observation_model, config
        self._rng = random.Random(config.seed)

    def _counts(self, view: TelemetryRegionView) -> dict[str, tuple[int, int]]:
        if not view.count_factors:
            return {}
        selected = self.config.selected_window or max(factor.window_size for factor in view.count_factors)
        return {factor.detector_id: (factor.events, factor.exposures) for factor in view.count_factors if factor.window_size == selected}

    def _log_likelihood(self, view: TelemetryRegionView, counts: Mapping[str, tuple[int, int]],
                        state: Mapping[str, float], controls: Mapping[str, float]) -> float:
        if self.config.likelihood == "event":
            return self.model.event_log_likelihood(view, state, controls)
        score = self.model.log_likelihood(counts, state, controls, view.context.context_id)
        if self.config.likelihood == "correlated_count":
            score += self.model.pair_log_likelihood(view, state, controls)
        elif self.config.likelihood != "count":
            raise ValueError("likelihood must be 'event', 'count', or 'correlated_count'")
        return score

    def _observability(self, counts: Mapping[str, tuple[int, int]], state: Mapping[str, float], controls: Mapping[str, float]) -> ObservabilityReport:
        variables = [item.variable_id for item in self.schema.variables]
        fisher = [[0.0 for _ in variables] for _ in variables]
        contributions: dict[str, float] = {}
        for detector_id, (_, exposure) in counts.items():
            if not exposure or detector_id not in self.model.responses:
                continue
            q = self.model.probability(detector_id, state, controls)
            gradient = self.model.gradient(detector_id, state, controls)
            scale = exposure / max(q * (1 - q), 1e-12)
            contributions[detector_id] = sum(scale * gradient[name] ** 2 for name in variables)
            for i, left in enumerate(variables):
                for j, right in enumerate(variables):
                    fisher[i][j] += scale * gradient[left] * gradient[right]
        values, vectors = _jacobi_eigensystem(fisher)
        positive = [item for item in values if item > self.config.observability_eigenvalue_threshold]
        rank = len(positive)
        condition = positive[0] / positive[-1] if positive else math.inf
        unresolved_scores = {name: 0.0 for name in variables}
        for value, vector in zip(values, vectors):
            if value <= self.config.observability_eigenvalue_threshold:
                for name, loading in zip(variables, vector):
                    unresolved_scores[name] += loading * loading
        unresolved = tuple(name for name in variables if unresolved_scores[name] > 1e-6)
        return ObservabilityReport(tuple(tuple(row) for row in fisher), tuple(values), tuple(tuple(item) for item in vectors), rank, condition, unresolved, contributions)

    def _prior_samples(self, previous: PhysicalStatePosterior | None) -> list[dict[str, float]]:
        if previous and previous.samples:
            cumulative, source = [], []
            total = 0.0
            for sample in previous.samples:
                total += sample.weight
                cumulative.append(total)
                source.append(sample.state)
            output = []
            for _ in range(self.config.particle_count):
                cursor = self._rng.random() * max(total, 1e-12)
                index = next((i for i, value in enumerate(cumulative) if value >= cursor), len(source) - 1)
                output.append(dict(source[index]))
            return output
        return [{variable.variable_id: self._rng.uniform(variable.lower, variable.upper) for variable in self.schema.variables}
                for _ in range(self.config.particle_count)]

    def _particle_posterior(self, view: TelemetryRegionView, counts: Mapping[str, tuple[int, int]], controls: Mapping[str, float], previous: PhysicalStatePosterior | None) -> tuple[list[dict[str, float]], list[float]]:
        particles = self._prior_samples(previous)
        weights = _normalise_log_weights([self._log_likelihood(view, counts, item, controls) for item in particles])
        # Systematic resampling remains deterministic under the configured seed and preserves modes.
        step, cursor, index = 1 / len(particles), self._rng.random() / len(particles), 0
        cumulative = weights[0]
        selected: list[dict[str, float]] = []
        for _ in particles:
            while cursor > cumulative and index < len(particles) - 1:
                index += 1
                cumulative += weights[index]
            rejuvenated = dict(particles[index])
            for variable in self.schema.variables:
                scale = self.config.particle_rejuvenation_fraction * (variable.upper - variable.lower)
                rejuvenated[variable.variable_id] = max(variable.lower, min(variable.upper,
                    rejuvenated[variable.variable_id] + self._rng.gauss(0., scale)))
            selected.append(rejuvenated)
            cursor += step
        return selected, [1 / len(selected)] * len(selected)

    def _gaussian_posterior(self, view: TelemetryRegionView, counts: Mapping[str, tuple[int, int]], controls: Mapping[str, float], previous: PhysicalStatePosterior | None) -> tuple[list[dict[str, float]], list[float]]:
        variables = self.schema.variables
        if previous:
            centre = dict(previous.mean)
            variance = [max(previous.covariance[i][i], 1e-12) for i in range(len(variables))]
        else:
            centre = {item.variable_id: item.nominal for item in variables}
            variance = [((item.upper - item.lower) / 4) ** 2 for item in variables]
        sigma = [centre] + [{**centre, item.variable_id: max(item.lower, min(item.upper, centre[item.variable_id] + sign * math.sqrt(variance[i])))}
                              for i, item in enumerate(variables) for sign in (-1.0, 1.0)]
        weights = _normalise_log_weights([self._log_likelihood(view, counts, item, controls) for item in sigma])
        return sigma, weights

    def _summary(self, samples: Sequence[Mapping[str, float]], weights: Sequence[float]) -> tuple[dict[str, float], tuple[tuple[float, ...], ...]]:
        ids = [item.variable_id for item in self.schema.variables]
        mean = {name: sum(weight * sample[name] for sample, weight in zip(samples, weights)) for name in ids}
        covariance = tuple(tuple(sum(weight * (sample[left] - mean[left]) * (sample[right] - mean[right]) for sample, weight in zip(samples, weights))
                                 for right in ids) for left in ids)
        return mean, covariance

    def _predictive(self, view: TelemetryRegionView, counts: Mapping[str, tuple[int, int]], samples: Sequence[Mapping[str, float]], weights: Sequence[float], controls: Mapping[str, float]) -> PosteriorPredictiveCheck:
        expected, residuals = {}, {}
        for detector_id, (events, exposure) in counts.items():
            q = sum(weight * self.model.probability(detector_id, sample, controls) for sample, weight in zip(samples, weights))
            expected[detector_id] = q
            residuals[detector_id] = (events - exposure*q) / math.sqrt(max(exposure*q*(1-q), 1e-9)) if exposure else 0.0
        maximum = max((abs(item) for item in residuals.values()), default=0.0)
        pair_residuals: dict[str, float] = {}
        for pair in view.pair_counts:
            total = pair.n00 + pair.n01 + pair.n10 + pair.n11
            if not total:
                continue
            predicted = 0.0
            for sample, weight in zip(samples, weights):
                p1 = self.model.probability(pair.detector_a, sample, controls)
                p2 = self.model.probability(pair.detector_b, sample, controls)
                rho = self.model.pair_correlations.get(tuple(sorted((pair.detector_a, pair.detector_b))), 0.)
                predicted += weight * (p1 * p2 + rho * math.sqrt(p1*(1-p1)*p2*(1-p2)))
            pair_residuals[f"{pair.detector_a}|{pair.detector_b}"] = (pair.n11 - total * predicted) / math.sqrt(max(total * predicted * (1-predicted), 1e-9))
        pair_max = max((abs(item) for item in pair_residuals.values()), default=0.)
        maximum = max(maximum, pair_max)
        return PosteriorPredictiveCheck(expected, residuals, maximum, pair_max > self.config.ood_residual_threshold, pair_residuals)

    def _intervention(self, observability: ObservabilityReport, preferred_variable: str | None = None) -> InterventionDesignRequest | None:
        if not observability.unresolved_variable_ids and preferred_variable is None:
            return None
        variable_id = preferred_variable or observability.unresolved_variable_ids[0]
        variable = next(item for item in self.schema.variables if item.variable_id == variable_id)
        if not variable.intervention_control or variable.safe_intervention is None:
            return None
        epsilon = variable.safe_intervention
        return InterventionDesignRequest(variable.variable_id, variable.intervention_control,
            {variable.intervention_control: epsilon}, {variable.intervention_control: -epsilon},
            1 / (1 + max(observability.eigenvalues[-1], 0.0)), "control-relevant low-Fisher direction")

    def infer(self, view: TelemetryRegionView, applied_controls: Mapping[str, float], *, method: str = "particle",
              previous: PhysicalStatePosterior | None = None) -> PhysicalStatePosterior:
        if view.region_id != self.schema.region_id:
            raise ValueError("Telemetry region and state schema must match")
        counts = self._counts(view)
        if not any(exposure for _, exposure in counts.values()):
            report = self._observability(counts, {item.variable_id: item.nominal for item in self.schema.variables}, applied_controls)
            return PhysicalStatePosterior("stage2.v1", view.region_id, method, {}, (), (), report,
                PosteriorPredictiveCheck({}, {}, 0.0, False), {}, 0.0, InferenceValidity.INSUFFICIENT_DATA,
                ("no unambiguous detector exposures",), self.model.version, self._intervention(report))
        if method == "particle":
            samples, weights = self._particle_posterior(view, counts, applied_controls, previous)
        elif method == "gaussian":
            samples, weights = self._gaussian_posterior(view, counts, applied_controls, previous)
        else:
            raise ValueError("method must be 'particle' or 'gaussian'")
        mean, covariance = self._summary(samples, weights)
        observability = self._observability(counts, mean, applied_controls)
        predictive = self._predictive(view, counts, samples, weights, applied_controls)
        validity = InferenceValidity.VALID
        reasons: list[str] = []
        if observability.rank < len(self.schema.variables):
            validity, reasons = InferenceValidity.LOW_OBSERVABILITY, ["control-relevant unresolved state direction"]
        if predictive.max_abs_residual > self.config.ood_residual_threshold:
            validity, reasons = InferenceValidity.MODEL_MISMATCH, reasons + ["posterior predictive residual exceeds threshold"]
        sign_ambiguous = next((variable.variable_id for variable in self.schema.variables
            if sum(weight for sample, weight in zip(samples, weights) if sample[variable.variable_id] < 0) > .15
            and sum(weight for sample, weight in zip(samples, weights) if sample[variable.variable_id] > 0) > .15), None)
        if sign_ambiguous is not None and validity is InferenceValidity.VALID:
            validity, reasons = InferenceValidity.LOW_OBSERVABILITY, ["symmetric posterior retains control-relevant sign ambiguity"]
        attribution = {item.variable_id: abs(mean[item.variable_id]) / max(sum(abs(mean[v.variable_id]) for v in self.schema.variables), 1e-12) for item in self.schema.variables}
        radius = {variable.variable_id: max(0., min(variable.upper - mean[variable.variable_id],
                  mean[variable.variable_id] - variable.lower)) for variable in self.schema.variables}
        shared = {name: mean[name] for name in self.schema.shared_variables if name in mean}
        return PhysicalStatePosterior("stage2.v2", view.region_id, method, mean, covariance,
            tuple(PosteriorSample(dict(sample), weight) for sample, weight in zip(samples, weights)), observability,
            predictive, attribution, predictive.max_abs_residual, validity, tuple(reasons), self.model.version,
            self._intervention(observability, sign_ambiguous), radius,
            self.config.nominal_validity_horizon_s if validity is InferenceValidity.VALID else 0.,
            predictive.max_abs_residual / max(self.config.ood_residual_threshold, 1e-9), shared)
