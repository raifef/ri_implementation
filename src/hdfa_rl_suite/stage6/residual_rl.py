"""Graph-masked residual Gaussian policy gradients with antithetic safe exploration."""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Mapping, Sequence

from hdfa_rl_suite.stage0.schema import HardwareLimits, PolicySnapshot, stable_hash
from hdfa_rl_suite.stage5.schema import PredictedCostDistribution, PredictiveControlPackage, ResidualAllocation, SolverStatus

from .schema import (CandidateObservation, EmpiricalResponseEvidence, ReplayItem,
                     ResidualCandidate, ResidualGateDecision, ResidualRLResult,
                     ShadowValidation)


@dataclass(frozen=True)
class ExplorationBudget:
    per_candidate: float
    cumulative: float


@dataclass
class GaussianResidualPolicy:
    mean: dict[str, float]
    stddev: dict[str, float]
    version: int = 0
    covariance: dict[str, dict[str, float]] | None = None

    @classmethod
    def full_control_baseline(cls, controls: Sequence[str], stddev: float = .05) -> "GaussianResidualPolicy":
        """Reference Google-style full-policy parameter exploration (identity residual basis)."""
        ordered = tuple(controls)
        # Identity covariance is represented sparsely.  The previous dense representation
        # consumed O(P^2) memory even though every off-diagonal entry was exactly zero.
        covariance = {control: {control: stddev * stddev} for control in ordered}
        return cls({control: 0. for control in ordered}, {control: stddev for control in ordered}, 0, covariance)


@dataclass(frozen=True)
class ResidualRLConfig:
    seed: int = 0
    learning_rate: float = .15
    entropy_floor: float = .005
    minimum_candidates: int = 4
    maximum_candidates: int = 16
    covariance_learning_rate: float = .05
    natural_gradient: bool = False
    covariance_contraction: float = .97
    residual_bias_fraction: float = .8
    maximum_microbatch_age_s: float = 2.0
    maximum_dense_covariance_block: int = 64


class ResidualRLController:
    """Optimise detector-local residuals only inside the Stage-5 issued allocation."""

    def __init__(self, policy: GaussianResidualPolicy, detector_control_graph: Mapping[str, Sequence[str]],
                 budget: ExplorationBudget, config: ResidualRLConfig = ResidualRLConfig()) -> None:
        self.policy, self.graph, self.budget, self.config = policy, {key: tuple(value) for key, value in detector_control_graph.items()}, budget, config
        self._rng, self._cumulative_damage = random.Random(config.seed), 0.
        self._replay: list[ReplayItem] = []
        self._schedule: dict[str, ResidualCandidate] = {}
        self._detectors_by_control: dict[str, tuple[str, ...]] = {}
        self._control_neighbours: dict[str, set[str]] = {}
        detectors: dict[str, list[str]] = {}
        for detector, linked in self.graph.items():
            for control in linked:
                detectors.setdefault(control, []).append(detector)
                self._control_neighbours.setdefault(control, {control}).update(linked)
        self._detectors_by_control = {control: tuple(values) for control, values in detectors.items()}

    def _dense_sample(self, controls: Sequence[str]) -> dict[str, float]:
        """Exactly sample zero-centred covariance noise with a jittered Cholesky factor."""
        covariance = self.policy.covariance or {}
        matrix = [[covariance.get(left, {}).get(
            right, self.policy.stddev.get(left, self.config.entropy_floor) ** 2 if left == right else 0.)
            for right in controls] for left in controls]
        factor = [[0. for _ in controls] for _ in controls]
        for i in range(len(controls)):
            for j in range(i + 1):
                value = matrix[i][j] - sum(factor[i][k] * factor[j][k] for k in range(j))
                factor[i][j] = (math.sqrt(max(value, self.config.entropy_floor ** 2)) if i == j
                                else value / max(factor[j][j], self.config.entropy_floor))
        normal = [self._rng.gauss(0., 1.) for _ in controls]
        return {control: sum(factor[i][j] * normal[j] for j in range(i + 1))
                for i, control in enumerate(controls)}

    def _sample(self, controls: Sequence[str]) -> dict[str, float]:
        """Sample sparse covariance components without constructing a global dense matrix.

        Exact dependency-free Cholesky sampling is retained for bounded local components.
        A component larger than ``maximum_dense_covariance_block`` is sampled diagonally;
        this conservative fallback prevents an accidental O(P^3) controller path and never
        invents unvalidated long-range correlation.
        """
        covariance = self.policy.covariance or {}
        selected = tuple(controls)
        selected_set = set(selected)
        adjacency = {control: {other for other, value in covariance.get(control, {}).items()
                               if other in selected_set and other != control and abs(value) > 1e-18}
                     for control in selected}
        for control, neighbours in tuple(adjacency.items()):
            for neighbour in neighbours:
                adjacency.setdefault(neighbour, set()).add(control)
        output: dict[str, float] = {}
        unseen = set(selected)
        while unseen:
            # Set iteration is process-hash dependent.  Consume components in the
            # caller's declared control order so a seed has identical meaning in serial
            # runs and spawned scalability workers.
            root = next(control for control in selected if control in unseen)
            unseen.remove(root)
            component, frontier = {root}, [root]
            while frontier:
                current = frontier.pop()
                discovered = adjacency.get(current, set()) & unseen
                unseen.difference_update(discovered)
                component.update(discovered)
                frontier.extend(discovered)
            ordered = tuple(control for control in selected if control in component)
            if len(ordered) <= self.config.maximum_dense_covariance_block:
                output.update(self._dense_sample(ordered))
            else:
                for control in ordered:
                    variance = covariance.get(control, {}).get(
                        control, self.policy.stddev.get(control, self.config.entropy_floor) ** 2)
                    output[control] = self._rng.gauss(
                        0., math.sqrt(max(variance, self.config.entropy_floor ** 2)))
        return output

    def suggest_candidate_count(self, gradient_snr: float | None) -> int:
        """Reduce device cost when the residual gradient is decisive; grow only when it is ambiguous."""
        if gradient_snr is None or gradient_snr < 1.0:
            return self.config.maximum_candidates
        if gradient_snr < 2.5:
            return max(8, self.config.minimum_candidates)
        return self.config.minimum_candidates

    def graph_colours(self) -> Mapping[str, int]:
        """Greedily colour controls that share detector support; equal colours may run concurrently."""
        controls = sorted({control for linked in self.graph.values() for control in linked})
        conflicts = {control: set() for control in controls}
        for linked in self.graph.values():
            for left in linked:
                conflicts[left].update(right for right in linked if right != left)
        colours: dict[str, int] = {}
        for control in sorted(controls, key=lambda item: (-len(conflicts[item]), item)):
            occupied = {colours[other] for other in conflicts[control] if other in colours}
            colour = 0
            while colour in occupied:
                colour += 1
            colours[control] = colour
        return colours

    @staticmethod
    def orthogonal_directions(controls: Sequence[str]) -> tuple[Mapping[str, float], ...]:
        """Return a deterministic orthogonal basis for local antithetic microbatches."""
        return tuple({control: float(index == coordinate) for coordinate, control in enumerate(controls)} for index in range(len(controls)))

    @staticmethod
    def allocate_shots(candidate_uncertainty: Mapping[str, float], minimum_shots: int = 32, maximum_shots: int = 512) -> Mapping[str, int]:
        """Allocate more QEC cycles only to statistically unresolved candidate comparisons."""
        maximum = max(candidate_uncertainty.values(), default=0.)
        if maximum <= 0:
            return {candidate: minimum_shots for candidate in candidate_uncertainty}
        return {candidate: min(maximum_shots, max(minimum_shots, round(minimum_shots * (1 + 15 * uncertainty / maximum))))
                for candidate, uncertainty in candidate_uncertainty.items()}

    def propose(self, package: PredictiveControlPackage, *, candidate_count: int | None = None) -> tuple[ResidualCandidate, ...]:
        if package.status.value != "optimal" or self._cumulative_damage >= self.budget.cumulative:
            return ()
        allocation = package.residual_allocation
        count = candidate_count or self.config.minimum_candidates
        count = max(self.config.minimum_candidates, min(self.config.maximum_candidates, count))
        if count % 2:
            count -= 1
        candidates: list[ResidualCandidate] = []
        controls = tuple(allocation.projection_controls)
        for pair_index in range(count // 2):
            pair_id = f"v{self.policy.version}-pair{pair_index}"
            sampled_block = self._sample(controls)
            direction = {}
            orthogonal_control = controls[pair_index % len(controls)] if controls else None
            for control in controls:
                # Orthogonal coverage is blended with covariance exploration, preserving block couplings.
                sampled = sampled_block[control] + float(control == orthogonal_control) * self.policy.stddev.get(control, self.config.entropy_floor)
                bound = allocation.bounds.get(control, 0.)
                # Keep the exploration direction feasible around the learned mean on
                # both sides.  Silent asymmetric clipping biases the finite difference.
                mean = max(-bound, min(bound, self.policy.mean.get(control, 0.)))
                headroom = max(0., min(bound-mean, bound+mean))
                direction[control] = max(-headroom, min(headroom, sampled))
            for sign in (1, -1):
                residual = {control: max(-allocation.bounds.get(control, 0.),
                            min(allocation.bounds.get(control, 0.),
                                self.policy.mean.get(control, 0.) + sign * direction.get(control, 0.)))
                            for control in controls}
                predicted_damage = sum(value * value for value in residual.values())
                if predicted_damage > self.budget.per_candidate:
                    continue
                applied = {control: package.action.get(control, 0.) + residual.get(control, 0.) for control in package.action}
                identifier = f"{pair_id}:{'plus' if sign > 0 else 'minus'}"
                candidate = ResidualCandidate(
                    identifier, pair_id, sign, residual, applied, predicted_damage,
                    self.policy.version, dict(self.policy.mean),
                    {control: sign * value for control, value in direction.items()},
                )
                candidates.append(candidate)
                self._schedule[identifier] = candidate
        return tuple(candidates)

    def _masked_loss(self, observation: CandidateObservation, control: str) -> float:
        relevant = self._detectors_by_control.get(control, ())
        if not relevant:
            return 0.
        weights = [max(0, observation.exposures.get(detector, 0)) for detector in relevant]
        detector_loss = (sum(observation.detector_losses.get(detector, 0.) * weight for detector, weight in zip(relevant, weights)) / sum(weights)
                         if sum(weights) else sum(observation.detector_losses.get(detector, 0.) for detector in relevant) / len(relevant))
        return detector_loss + observation.logical_risk + observation.leakage_risk + observation.correlation_penalty

    def update(self, package: PredictiveControlPackage, observations: Sequence[CandidateObservation], *, current_regime: str = "unknown",
               current_context: str = "default", current_model_version: str = "unknown",
               commit: bool = True,
               gate_decision: ResidualGateDecision | None = None,
               shadow_validation: ShadowValidation | None = None) -> ResidualRLResult:
        prior_mean = dict(self.policy.mean)
        prior_stddev = dict(self.policy.stddev)
        prior_covariance = ({key: dict(value) for key, value in self.policy.covariance.items()}
                            if self.policy.covariance is not None else None)
        prior_version = self.policy.version
        by_pair: dict[str, dict[int, tuple[ResidualCandidate, CandidateObservation]]] = {}
        update_damage = 0.0
        for observation in observations:
            candidate = self._schedule.get(observation.candidate_id)
            if candidate is None or candidate.policy_version != self.policy.version:
                continue
            damage = candidate.predicted_damage + observation.total_damage
            update_damage += damage
            self._cumulative_damage += damage
            if observation.observed_at_s and observation.observed_at_s - package.activation_time_s > self.config.maximum_microbatch_age_s:
                continue
            by_pair.setdefault(candidate.pair_id, {})[candidate.sign] = (candidate, observation)
            self._replay.append(ReplayItem(candidate, observation, package.policy_hash))
        gradient = {control: 0. for control in package.residual_allocation.projection_controls}
        evidence: list[EmpiricalResponseEvidence] = []
        pairs_used = 0
        for pair_id, pair in by_pair.items():
            if 1 not in pair or -1 not in pair:
                continue
            pairs_used += 1
            plus_candidate, plus = pair[1]
            minus_candidate, minus = pair[-1]
            for control in gradient:
                perturbation = plus_candidate.residual.get(control, 0.) - minus_candidate.residual.get(control, 0.)
                if abs(perturbation) < 1e-12:
                    continue
                delta = self._masked_loss(plus, control) - self._masked_loss(minus, control)
                gradient[control] += -delta / perturbation
                for detector in self._detectors_by_control.get(control, ()):
                    evidence.append(EmpiricalResponseEvidence(control, detector, -delta / perturbation, pair_id))
        raw_gradient = dict(gradient)
        if pairs_used:
            for control, value in gradient.items():
                value /= pairs_used
                gradient[control] = value
            if self.config.natural_gradient and self.policy.covariance:
                gradient = {left: sum(value * raw_gradient.get(right, 0.) / pairs_used
                                      for right, value in self.policy.covariance.get(left, {}).items())
                            for left in gradient}
            for control, value in gradient.items():
                bound = package.residual_allocation.bounds.get(control, 0.)
                self.policy.mean[control] = max(-bound, min(bound, self.policy.mean.get(control, 0.) + self.config.learning_rate * value))
                self.policy.stddev[control] = max(
                    self.config.entropy_floor,
                    min(bound, self.policy.stddev.get(control, self.config.entropy_floor)
                        * self.config.covariance_contraction),
                )
            if self.policy.covariance is not None:
                controls = tuple(gradient)
                control_set = set(controls)
                norm = math.sqrt(sum(value * value for value in gradient.values()))
                unit = {control: gradient[control] / max(norm, 1e-12) for control in controls}
                for left in controls:
                    self.policy.covariance.setdefault(left, {})
                    neighbours = self._control_neighbours.get(left, {left}) & control_set
                    neighbours.add(left)
                    for right in neighbours:
                        target = unit[left] * unit[right] * self.policy.stddev[left] * self.policy.stddev[right]
                        old = self.policy.covariance[left].get(right, 0.)
                        self.policy.covariance[left][right] = ((1-self.config.covariance_learning_rate) * old
                                                              + self.config.covariance_learning_rate * target)
                    self.policy.covariance[left][left] = max(
                        min(self.policy.covariance[left][left], self.policy.stddev[left] ** 2),
                        self.config.entropy_floor ** 2,
                    )
            self.policy.version += 1
        bias = max((abs(self.policy.mean.get(control, 0.)) / max(package.residual_allocation.bounds.get(control, 1e-12), 1e-12)
                    for control in gradient), default=0.)
        fallback = (self._cumulative_damage >= self.budget.cumulative
                    or any(abs(value) > 10. * max(package.residual_allocation.bounds.get(control, 1e-12), 1e-12) for control, value in gradient.items())
                    or bias >= self.config.residual_bias_fraction)
        reason = ("exploration damage budget exhausted" if self._cumulative_damage >= self.budget.cumulative
                  else "persistent residual mean/gradient exceeds allocated physical subspace" if fallback else None)
        compatible = [item for item in self._replay if item.observation.regime_id == current_regime
                      and item.observation.context_id == current_context and item.observation.model_version == current_model_version]
        compatible_replay = sum(min(10., max(.1, item.observation.current_probability / max(item.observation.behaviour_probability, 1e-9))) for item in compatible)
        gradient_values = tuple(gradient.values())
        gradient_mean = sum(abs(value) for value in gradient_values) / max(1, len(gradient_values))
        gradient_spread = math.sqrt(sum((abs(value) - gradient_mean) ** 2 for value in gradient_values) / max(1, len(gradient_values)))
        proposed_mean = dict(self.policy.mean)
        proposed_stddev = dict(self.policy.stddev)
        proposed_version = self.policy.version
        proposed_covariance = {key: dict(value) for key, value in (self.policy.covariance or {}).items()}
        if not commit:
            self.policy.mean = prior_mean
            self.policy.stddev = prior_stddev
            self.policy.covariance = prior_covariance
            self.policy.version = prior_version
        return ResidualRLResult(proposed_mean, proposed_stddev, proposed_version, gradient,
            update_damage, self._cumulative_damage, tuple(evidence), fallback, reason,
            round(compatible_replay), proposed_covariance, bias,
            gradient_mean / max(gradient_spread, 1e-12), (reason,) if reason else (),
            gate_decision, shadow_validation, commit)

    def commit_shadow(self, result: ResidualRLResult) -> None:
        """Promote an independently validated shadow update atomically."""
        if result.committed:
            return
        self.policy.mean = dict(result.policy_mean)
        self.policy.stddev = dict(result.policy_stddev)
        self.policy.covariance = {
            key: dict(value) for key, value in result.policy_covariance.items()}
        self.policy.version = result.policy_version

    def deactivate(self) -> None:
        """Remove learned residual authority; Stage 5 remains the live policy."""
        self.policy.mean = {control: 0.0 for control in self.policy.mean}
        self.policy.stddev = {
            control: self.config.entropy_floor for control in self.policy.stddev}
        if self.policy.covariance is not None:
            self.policy.covariance = {
                control: {control: self.config.entropy_floor**2}
                for control in self.policy.mean}
        self.policy.version += 1

    def replay_for(self, regime_id: str, context_id: str, model_version: str) -> tuple[ReplayItem, ...]:
        """Never use stale experience merely to inflate a residual-RL batch."""
        return tuple(item for item in self._replay if item.observation.regime_id == regime_id
                     and item.observation.context_id == context_id and item.observation.model_version == model_version)

    @property
    def cumulative_damage(self) -> float:
        return self._cumulative_damage


class FullControlDetectorRL:
    """Legacy reduced-budget full-control approximation.

    This shares the antithetic finite-difference residual core and is not the paper's
    detector-baseline/PPO-style policy gradient.  The independent high-shot scientific
    reference lives in :mod:`hdfa_rl_suite.google_rl_certification`.
    """

    def __init__(self, limits: HardwareLimits, detector_control_graph: Mapping[str, Sequence[str]],
                 initial_policy: PolicySnapshot, budget: ExplorationBudget, *, seed: int = 0,
                 candidate_count: int = 40, stddev: float = .05) -> None:
        if candidate_count < 4 or candidate_count % 2:
            raise ValueError("full-control candidate count must be even and at least four")
        self.limits, self.current_policy = limits, initial_policy
        self.candidate_count = candidate_count
        config = ResidualRLConfig(seed=seed, minimum_candidates=self.candidate_count,
                                  maximum_candidates=self.candidate_count)
        self.core = ResidualRLController(GaussianResidualPolicy.full_control_baseline(tuple(limits.controls), stddev),
                                         detector_control_graph, budget, config)
        self._package: PredictiveControlPackage | None = None

    def _control_package(self) -> PredictiveControlPackage:
        bounds = {control: min(bound.trust_radius, bound.max_slew / 2,
                  self.current_policy.values[control] - bound.minimum,
                  bound.maximum - self.current_policy.values[control]) for control, bound in self.limits.controls.items()}
        allocation = ResidualAllocation(tuple(bounds), bounds, (), {control: "full-control RL baseline" for control in bounds})
        action = dict(self.current_policy.values)
        return PredictiveControlPackage("baseline-rl.v1", SolverStatus.OPTIMAL, action, (action,), {}, allocation, (),
            PredictedCostDistribution(0., 0., {}), self.current_policy, stable_hash(action),
            self.current_policy.timestamp_s, math.inf, self.current_policy)

    def propose(self) -> tuple[ResidualCandidate, ...]:
        self._package = self._control_package()
        return self.core.propose(self._package, candidate_count=self.candidate_count)

    @property
    def proposed_package(self) -> PredictiveControlPackage | None:
        """The exact baseline package against which the current candidates were drawn."""
        return self._package

    def update(self, observations: Sequence[CandidateObservation], **context) -> ResidualRLResult:
        if self._package is None:
            raise RuntimeError("propose must be called before update")
        result = self.core.update(self._package, observations, **context)
        committed_step = dict(result.policy_mean)
        values = {control: max(self.limits.controls[control].minimum,
                  min(self.limits.controls[control].maximum,
                      self.current_policy.values[control] + committed_step.get(control, 0.)))
                  for control in self.current_policy.values}
        self.current_policy = PolicySnapshot(values, stable_hash(values), self.current_policy.timestamp_s + 1.)
        # The full-control package for the next epoch is centred on the newly committed
        # policy.  Retaining the old relative mean would apply the same step twice.
        self.core.policy.mean = {control: 0. for control in self.core.policy.mean}
        return result
