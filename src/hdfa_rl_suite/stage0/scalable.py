"""Graph-scalable joint-block bootstrap for simulator and hardware-like adapters."""
from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import NormalDist
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from hdfa_rl_suite.simulator import ScalableQECDevice

from .schema import (
    BootstrapResult,
    CalibrationEstimate,
    CalibrationEvent,
    CalibrationNode,
    HealthStatus,
    NodeStatus,
    ParameterRecord,
    PolicySnapshot,
    StageHealthPacket,
    stable_hash,
)


@dataclass(frozen=True)
class ScalableBootstrapConfig:
    characterization_shots: int = 384
    validation_cycles: int = 512
    maximum_rounds: int = 6
    target_posterior_stddev: float = .035
    qec_detector_rate_limit: float = .10
    sensitivity_fraction: float = .15
    sensitivity_batching: bool = True
    sensitivity_max_batch_size: int = 32
    sensitivity_interference_alpha: float = 1e-4
    block_predictive_familywise_alpha: float = 1e-4

    def __post_init__(self) -> None:
        if self.characterization_shots <= 0 or self.validation_cycles <= 0 or self.maximum_rounds <= 0:
            raise ValueError("Stage-0 acquisition budgets must be positive")
        if (self.target_posterior_stddev <= 0
                or not 0 < self.block_predictive_familywise_alpha < 1
                or not 0 < self.sensitivity_interference_alpha < 1):
            raise ValueError("Stage-0 uncertainty and family-wise alpha must be physical")
        if self.sensitivity_max_batch_size <= 0:
            raise ValueError("sensitivity_max_batch_size must be positive")
        if not 0 < self.qec_detector_rate_limit < 1 or not 0 < self.sensitivity_fraction <= 1:
            raise ValueError("Stage-0 QEC and sensitivity thresholds must lie in (0, 1]")


class ScalableBootstrapCalibrator:
    """Joint regional calibration with graph-coloured scheduling and atomic rollback."""

    def __init__(self, device: ScalableQECDevice,
                 config: ScalableBootstrapConfig = ScalableBootstrapConfig()) -> None:
        self.device, self.config = device, config
        self._events: list[CalibrationEvent] = []
        self._estimates: dict[str, CalibrationEstimate] = {}
        self._nodes = self._build_nodes()
        self._status = {node_id: NodeStatus.PENDING for node_id in self._nodes}
        self._initial = PolicySnapshot(dict(device.confirmed_policy.controls),
                                       device.confirmed_policy.policy_hash, device.now_s)
        self._run_id = stable_hash({
            "stage": "stage0",
            "started_at_s": device.now_s,
            "reference_policy_id": device.confirmed_policy.policy_id,
            "controller_state_hash": device.controller_state_hash,
        })[:16]
        self._batches = self._schedule_batches()

    def _policy_id(self, local_id: str) -> str:
        return f"stage0:{self._run_id}:{local_id}"

    def _record(self, node: str, event: str, **payload: Any) -> None:
        self._events.append(CalibrationEvent(len(self._events), node, event, self.device.now_s, payload))

    def _build_nodes(self) -> dict[str, CalibrationNode]:
        nodes = {"timing": CalibrationNode("timing", "timing", (), (), (), ("controller",))}
        for detector in self.device.circuit.detectors:
            controls = self.device.detector_control_graph[detector.detector_id]
            node_id = f"block:{detector.region_id}"
            nodes[node_id] = CalibrationNode(node_id, "joint_control_block", tuple(controls), ("timing",),
                ("qec", "sensitivity", "final_validation"), tuple(self.device.topology.control_channels[c] for c in controls),
                max_attempts=self.config.maximum_rounds, validity_duration_s=120.)
        block_ids = tuple(node for node in nodes if node.startswith("block:"))
        nodes["qec"] = CalibrationNode("qec", "qec", (), block_ids, ("sensitivity", "final_validation"), ("qec-controller",))
        nodes["sensitivity"] = CalibrationNode("sensitivity", "sensitivity", (), ("qec",), ("final_validation",), tuple(self.device.topology.control_channels.values()))
        nodes["final_validation"] = CalibrationNode("final_validation", "final_validation", (), ("sensitivity",), (), ("qec-controller",))
        return nodes

    def _schedule_batches(self) -> tuple[tuple[str, ...], ...]:
        remaining = set(self._nodes)
        completed: set[str] = set()
        output: list[tuple[str, ...]] = []
        while remaining:
            ready = sorted(node_id for node_id in remaining if set(self._nodes[node_id].prerequisites) <= completed)
            if not ready:
                raise ValueError("calibration DAG contains a cycle")
            selected: list[str] = []
            occupied: set[str] = set()
            for node_id in ready:
                resources = set(self._nodes[node_id].resources)
                if resources.isdisjoint(occupied):
                    selected.append(node_id)
                    occupied.update(resources)
            output.append(tuple(selected))
            remaining.difference_update(selected)
            completed.update(selected)
        return tuple(output)

    def _apply_towards(self, target: Mapping[str, float], policy_id: str) -> None:
        current = dict(self.device.confirmed_policy.controls)
        for control, desired in target.items():
            bound = self.device.limits.controls[control]
            current[control] = min(bound.maximum, max(bound.minimum,
                min(current[control] + bound.max_slew, max(current[control] - bound.max_slew, desired))))
        self.device.apply_policy(current, policy_id=self._policy_id(policy_id))

    def _joint_block(self, node: CalibrationNode) -> CalibrationEstimate:
        means = {control: self.device.confirmed_policy.controls[control] for control in node.owned_parameters}
        variances = {control: math.inf for control in node.owned_parameters}
        total_shots = 0
        for round_index in range(node.max_attempts):
            # Active design targets the largest downstream control uncertainty first.
            ordered = sorted(node.owned_parameters, key=lambda control: variances[control], reverse=True)
            result = self.device.characterize_controls(ordered, shots=self.config.characterization_shots)
            total_shots += result.shots
            for control in ordered:
                if math.isinf(variances[control]):
                    means[control], variances[control] = result.estimates[control], result.variances[control]
                else:
                    prior_precision, new_precision = 1 / variances[control], 1 / result.variances[control]
                    means[control] = (prior_precision * means[control] + new_precision * result.estimates[control]) / (prior_precision + new_precision)
                    variances[control] = 1 / (prior_precision + new_precision)
            self._apply_towards(means, f"{node.node_id}:round:{round_index}")
            if max(math.sqrt(value) for value in variances.values()) <= self.config.target_posterior_stddev:
                break
        held = self.device.characterize_controls(node.owned_parameters, shots=self.config.characterization_shots)
        residual = max(abs(held.estimates[control] - means[control]) / math.sqrt(variances[control] + held.variances[control])
                       for control in node.owned_parameters)
        # Under the calibrated Gaussian observation model each standardized held-out
        # residual is N(0,1).  Use the exact max-|Z| tail within this block and split a
        # declared family-wise false-rejection budget across all regional blocks.  The
        # previous z<2 heuristic rejected a correctly calibrated many-region device with
        # high probability merely because many comparisons were made.
        comparisons = max(1, len(node.owned_parameters))
        within_probability = math.erf(residual / math.sqrt(2.0))
        familywise_tail = (1.0 if within_probability <= 0 else
                           -math.expm1(comparisons * math.log(within_probability)))
        block_count = max(1, sum(item.family == "joint_control_block" for item in self._nodes.values()))
        alpha_share = self.config.block_predictive_familywise_alpha / block_count
        confidence = min(.999, node.minimum_confidence * familywise_tail / alpha_share)
        return CalibrationEstimate(means, variances, {
            "joint_block_predictive_z": residual,
            "joint_block_familywise_tail_probability": familywise_tail,
            "experiment_familywise_alpha": self.config.block_predictive_familywise_alpha,
            "block_alpha_share": alpha_share,
        },
                                   -residual, confidence,
                                   {"active_design_shots": total_shots, "held_out_shots": held.shots,
                                    "risk_target": "downstream detector/logical uncertainty",
                                    "held_out_decision": "Bonferroni-controlled max-|Z| posterior-predictive test"})

    def _qec(self, final: bool = False) -> CalibrationEstimate:
        batch = self.device.acquire(self.config.validation_cycles)
        rate = batch.detector_rate
        se = math.sqrt(max(rate * (1-rate), 1e-12) / max(1, batch.detector_exposures))
        upper = rate + 2.326 * se
        confidence = .999 if upper <= self.config.qec_detector_rate_limit else max(0., 1-upper)
        return CalibrationEstimate({}, {}, {"detector_rate": rate, "upper_99": upper},
            1-rate, confidence, {"cycles": batch.cycles, "logical_failures": batch.logical_failures,
                                 "qec_cycles": batch.cycles, "independent": final})

    def _control_detector_supports(self) -> dict[str, tuple[str, ...]]:
        supports = {control: [] for control in self.device.limits.controls}
        for detector, controls in self.device.detector_control_graph.items():
            for control in controls:
                if control in supports:
                    supports[control].append(detector)
        return {control: tuple(detectors) for control, detectors in supports.items()}

    def _sensitivity_batches(self, supports: Mapping[str, tuple[str, ...]]) -> tuple[tuple[str, ...], ...]:
        """Greedily colour controls whose detector and hardware footprints are disjoint."""
        if not self.config.sensitivity_batching:
            return tuple((control,) for control in sorted(supports))
        batches: list[list[str]] = []
        occupied_detectors: list[set[str]] = []
        occupied_resources: list[set[str]] = []
        ordered = sorted(supports, key=lambda control: (-len(supports[control]), control))
        for control in ordered:
            detectors = set(supports[control])
            resources = {self.device.topology.control_channels[control]}
            for index, batch in enumerate(batches):
                if (len(batch) < self.config.sensitivity_max_batch_size
                        and detectors.isdisjoint(occupied_detectors[index])
                        and resources.isdisjoint(occupied_resources[index])):
                    batch.append(control)
                    occupied_detectors[index].update(detectors)
                    occupied_resources[index].update(resources)
                    break
            else:
                batches.append([control])
                occupied_detectors.append(set(detectors))
                occupied_resources.append(set(resources))
        return tuple(tuple(batch) for batch in batches)

    @staticmethod
    def _rate(count: tuple[int, int]) -> float:
        return count[0] / count[1] if count[1] else math.nan

    @staticmethod
    def _difference_standard_error(positive: tuple[int, int],
                                   negative: tuple[int, int]) -> float:
        p_plus = ScalableBootstrapCalibrator._rate(positive)
        p_minus = ScalableBootstrapCalibrator._rate(negative)
        if not math.isfinite(p_plus) or not math.isfinite(p_minus):
            return math.inf
        return math.sqrt(
            max(p_plus * (1-p_plus), 1e-12) / max(1, positive[1])
            + max(p_minus * (1-p_minus), 1e-12) / max(1, negative[1]))

    def _execute_sensitivity_pair(self, controls: tuple[str, ...],
                                  epsilons: Mapping[str, float],
                                  baseline: Mapping[str, float],
                                  cycles: int, pair_id: str):
        plus, minus = dict(baseline), dict(baseline)
        for control in controls:
            plus[control] += epsilons[control]
            minus[control] -= epsilons[control]
        self.device.apply_policy(plus, policy_id=self._policy_id(f"sensitivity:{pair_id}:plus"),
                                 perturbation={control: epsilons[control] for control in controls})
        positive = self.device.acquire(cycles, retain_records=False)
        self.device.apply_policy(minus, policy_id=self._policy_id(f"sensitivity:{pair_id}:minus"),
                                 perturbation={control: -epsilons[control] for control in controls})
        negative = self.device.acquire(cycles, retain_records=False)
        self.device.apply_policy(
            baseline, policy_id=self._policy_id(f"sensitivity:{pair_id}:restore"))
        return positive, negative

    def _interference_z(self, positive, negative, expected: set[str],
                        threshold: float) -> tuple[float, tuple[str, ...]]:
        unexpected: list[str] = []
        maximum = 0.0
        for detector in self.device.detector_control_graph:
            if detector in expected:
                continue
            pos = positive.detector_counts[detector]
            neg = negative.detector_counts[detector]
            se = self._difference_standard_error(pos, neg)
            z_score = (0.0 if not math.isfinite(se) or se <= 0 else
                       abs(self._rate(pos) - self._rate(neg)) / se)
            maximum = max(maximum, z_score)
            if z_score > threshold:
                unexpected.append(detector)
        return maximum, tuple(unexpected)

    def _record_sensitivity_contrasts(self, controls: tuple[str, ...],
                                      supports: Mapping[str, tuple[str, ...]],
                                      epsilons: Mapping[str, float], positive, negative,
                                      slopes: dict[str, float], variances: dict[str, float],
                                      jacobian: dict[str, dict[str, float]]) -> None:
        for control in controls:
            epsilon = epsilons[control]
            detector_values: dict[str, float] = {}
            local_events_plus = local_exposures_plus = 0
            local_events_minus = local_exposures_minus = 0
            for detector in supports[control]:
                pos = positive.detector_counts[detector]
                neg = negative.detector_counts[detector]
                detector_values[detector] = (self._rate(pos) - self._rate(neg)) / (2 * epsilon)
                local_events_plus += pos[0]
                local_exposures_plus += pos[1]
                local_events_minus += neg[0]
                local_exposures_minus += neg[1]
            pos_local = (local_events_plus, local_exposures_plus)
            neg_local = (local_events_minus, local_exposures_minus)
            slopes[control] = (self._rate(pos_local) - self._rate(neg_local)) / (2 * epsilon)
            standard_error = self._difference_standard_error(pos_local, neg_local) / (2 * epsilon)
            variances[control] = standard_error ** 2
            jacobian[control] = detector_values

    def _sensitivity(self) -> CalibrationEstimate:
        baseline = dict(self.device.confirmed_policy.controls)
        supports = self._control_detector_supports()
        epsilons: dict[str, float] = {}
        slopes: dict[str, float] = {}
        variances: dict[str, float] = {}
        jacobian: dict[str, dict[str, float]] = {}
        for control, value in baseline.items():
            bound = self.device.limits.controls[control]
            epsilon = min(bound.max_slew / 2, bound.trust_radius * self.config.sensitivity_fraction,
                          value - bound.minimum, bound.maximum - value)
            if epsilon <= 0:
                slopes[control], variances[control], jacobian[control] = 0., math.inf, {}
            else:
                epsilons[control] = epsilon
        cycles_per_arm = max(64, self.config.validation_cycles // 4)
        batches = self._sensitivity_batches({control: supports[control] for control in epsilons})
        maximum_tests = max(1, len(batches) * len(self.device.circuit.detectors))
        z_threshold = NormalDist().inv_cdf(
            1 - self.config.sensitivity_interference_alpha / (2 * maximum_tests))
        batch_diagnostics: list[dict[str, Any]] = []
        qec_cycles = 0
        unresolved_interference: list[str] = []
        maximum_sentinel_z = 0.0
        for batch_index, batch in enumerate(batches):
            expected = {detector for control in batch for detector in supports[control]}
            positive, negative = self._execute_sensitivity_pair(
                batch, epsilons, baseline, cycles_per_arm, f"batch:{batch_index}")
            qec_cycles += 2 * cycles_per_arm
            maximum_z, unexpected = self._interference_z(
                positive, negative, expected, z_threshold)
            maximum_sentinel_z = max(maximum_sentinel_z, maximum_z)
            fallback = bool(unexpected and len(batch) > 1)
            batch_diagnostics.append({
                "batch": batch,
                "expected_detectors": tuple(sorted(expected)),
                "sentinel_max_z": maximum_z,
                "unexpected_detectors": unexpected,
                "fallback_to_individual": fallback,
            })
            if fallback:
                for control in batch:
                    pos_single, neg_single = self._execute_sensitivity_pair(
                        (control,), epsilons, baseline, cycles_per_arm,
                        f"fallback:{batch_index}:{control}")
                    qec_cycles += 2 * cycles_per_arm
                    single_z, single_unexpected = self._interference_z(
                        pos_single, neg_single, set(supports[control]), z_threshold)
                    maximum_sentinel_z = max(maximum_sentinel_z, single_z)
                    if single_unexpected:
                        unresolved_interference.extend(
                            f"{control}->{detector}" for detector in single_unexpected)
                    self._record_sensitivity_contrasts(
                        (control,), supports, epsilons, pos_single, neg_single,
                        slopes, variances, jacobian)
            else:
                if unexpected:
                    unresolved_interference.extend(
                        f"{batch[0]}->{detector}" for detector in unexpected)
                self._record_sensitivity_contrasts(
                    batch, supports, epsilons, positive, negative,
                    slopes, variances, jacobian)
        confidence = 0.0 if unresolved_interference else .95
        return CalibrationEstimate({}, variances, slopes,
                                   -maximum_sentinel_z, confidence,
                                   {"nonlinearity": "central antithetic local response",
                                    "design": "conflict-free graph-coloured local Jacobian",
                                    "per_control_cycles_per_sign": cycles_per_arm,
                                    "sensitivity_batches": tuple(batch_diagnostics),
                                    "batch_count": len(batches),
                                    "maximum_batch_size": max((len(batch) for batch in batches), default=0),
                                    "interference_alpha": self.config.sensitivity_interference_alpha,
                                    "interference_z_threshold": z_threshold,
                                    "interference_passed": not unresolved_interference,
                                    "unresolved_interference": tuple(unresolved_interference),
                                    "local_jacobian": jacobian,
                                    "aggregate_only_acquisition": True,
                                    "qec_cycles": qec_cycles})

    def run(self) -> BootstrapResult:
        for batch in self._batches:
            self._record("scheduler", "resource_batch", nodes=batch)
            for node_id in batch:
                node = self._nodes[node_id]
                if self._status[node_id] is not NodeStatus.PENDING or not all(
                        self._status[prerequisite] is NodeStatus.PASSED for prerequisite in node.prerequisites):
                    continue
                self._status[node_id] = NodeStatus.RUNNING
                self._record(node_id, "started", resources=node.resources, prerequisites=node.prerequisites)
                if node.family == "timing":
                    estimate = CalibrationEstimate({}, {}, {"clock": 1.}, 1., .999,
                        {"controller_latency_s": self.device.config.controller_latency_s})
                elif node.family == "joint_control_block":
                    estimate = self._joint_block(node)
                elif node.family == "qec":
                    estimate = self._qec(False)
                elif node.family == "sensitivity":
                    estimate = self._sensitivity()
                else:
                    estimate = self._qec(True)
                accepted = estimate.confidence >= node.minimum_confidence
                if node.family == "joint_control_block":
                    accepted = accepted and (
                        estimate.model_scores["joint_block_familywise_tail_probability"]
                        >= estimate.model_scores["block_alpha_share"])
                if node.family in {"qec", "final_validation"}:
                    accepted = accepted and estimate.model_scores["upper_99"] <= self.config.qec_detector_rate_limit
                self._estimates[node_id] = estimate
                self._status[node_id] = NodeStatus.PASSED if accepted else NodeStatus.FAILED
                self._record(node_id, "accepted" if accepted else "rejected", estimate=estimate.__dict__)
                if not accepted:
                    for other_id, other in self._nodes.items():
                        if node_id in other.prerequisites and self._status[other_id] is NodeStatus.PENDING:
                            self._status[other_id] = NodeStatus.BLOCKED
                    break
        failed = tuple(node for node, status in self._status.items() if status in {NodeStatus.FAILED, NodeStatus.BLOCKED})
        values = dict(self.device.confirmed_policy.controls)
        policy = PolicySnapshot(values, self.device.confirmed_policy.policy_hash, self.device.now_s)
        sensitivity = dict(self._estimates.get("sensitivity", CalibrationEstimate({}, {}, {}, 0., 0.)).model_scores)
        sensitivity_diagnostics = self._estimates.get(
            "sensitivity", CalibrationEstimate({}, {}, {}, 0., 0.)).diagnostics
        local_jacobian = sensitivity_diagnostics.get("local_jacobian", {})
        variances: dict[str, float] = {}
        for node_id, estimate in self._estimates.items():
            if node_id.startswith("block:"):
                variances.update(estimate.variances)
        registry = {control: ParameterRecord(control, control.split(":")[0], self.device.topology.control_channels[control],
            bound.unit, values[control], bound, next(node_id for node_id, node in self._nodes.items() if control in node.owned_parameters),
            tuple(f"gate:{control}" for _ in (0,)),
            tuple(detector for detector, linked in self.device.detector_control_graph.items() if control in linked),
            next((detector.region_id for detector in self.device.circuit.detectors if control in self.device.detector_control_graph[detector.detector_id]), "global"),
            variances.get(control, math.inf), sensitivity.get(control),
            dict(local_jacobian.get(control, {})),
            self.device.now_s + self._nodes[next(
                node_id for node_id, node in self._nodes.items() if control in node.owned_parameters
            )].validity_duration_s,
            "stage0.v2") for control, bound in self.device.limits.controls.items()}
        health = StageHealthPacket(HealthStatus.FAILED if failed else HealthStatus.PASSED, failed,
            tuple(node for node, status in self._status.items() if status is not NodeStatus.PASSED), {}, True)
        return BootstrapResult("stage0.v2", policy, self.device.circuit, registry, dict(self._status),
            dict(self.device.detector_control_graph), sensitivity, self._initial, health, tuple(self._events), (),
            dict(self._nodes), dict(self._estimates), self._batches)
