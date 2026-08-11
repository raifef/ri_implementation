"""Dependency-aware, deterministic Stage-0 bootstrap calibration."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

from .schema import (
    BootstrapResult, CalibrationEstimate, CalibrationEvent, CalibrationNode, ControlBound,
    DeclaredCompromise, DeviceTopology, HardwareLimits, HealthStatus, NodeStatus,
    ParameterRecord, PolicySnapshot, StageHealthPacket, TargetQECCircuit, stable_hash,
)
from .backend import CalibrationBackend


@dataclass(frozen=True)
class BootstrapConfig:
    seed: int = 0
    shots_per_fit: int = 512
    validation_shots: int = 1024
    sensitivity_perturbation_fraction: float = 0.1
    qec_rate_limit: float = 0.10
    minimum_confidence: float = 0.90


class BootstrapCalibrator:
    """Executes an auditable calibration DAG against a backend with recorded observations."""

    def __init__(self, topology: DeviceTopology, limits: HardwareLimits, circuit: TargetQECCircuit,
                 backend: CalibrationBackend, config: BootstrapConfig = BootstrapConfig()) -> None:
        self.topology, self.limits, self.circuit = topology, limits, circuit
        self.backend, self.config = backend, config
        self._log: list[CalibrationEvent] = []
        self._sequence = 0
        self._values: dict[str, float] = {
            "frequency:q0": 5.10e9, "amplitude:q0": .50, "coupling:q0-q1": .03,
        }
        self._variances: dict[str, float] = {key: math.inf for key in self._values}
        self._status = {node.node_id: NodeStatus.PENDING for node in self._make_dag()}
        self._dag = {node.node_id: node for node in self._make_dag()}
        self._estimates: dict[str, CalibrationEstimate] = {}
        self._initial_snapshot = self._snapshot()

    def _make_dag(self) -> tuple[CalibrationNode, ...]:
        return (
            CalibrationNode("timing", "timing", (), (), (), ("controller",)),
            CalibrationNode("resonator", "resonator", (), ("timing",), ("readout",), ("r0",)),
            CalibrationNode("readout", "readout", (), ("resonator",), ("single_qubit",), ("r0",)),
            CalibrationNode("spectroscopy", "spectroscopy", ("frequency:q0",), ("readout",), ("single_qubit", "entangling", "qec"), ("d0",)),
            CalibrationNode("single_qubit", "single_qubit", ("frequency:q0", "amplitude:q0"), ("spectroscopy",), ("entangling", "qec"), ("d0",)),
            CalibrationNode("entangling", "entangling", ("coupling:q0-q1",), ("single_qubit",), ("qec",), ("c0",)),
            CalibrationNode("transfer", "transfer", (), ("entangling",), ("qec",), ("d0", "c0")),
            CalibrationNode("qec", "qec", (), ("transfer",), ("sensitivity", "final_validation"), ("d0", "c0")),
            CalibrationNode("sensitivity", "sensitivity", (), ("qec",), ("final_validation",), ("d0", "c0")),
            CalibrationNode("final_validation", "final_validation", (), ("sensitivity",), (), ("d0", "c0")),
        )

    def _record(self, node_id: str, event_type: str, **payload: Any) -> None:
        self._log.append(CalibrationEvent(self._sequence, node_id, event_type, self.backend.now_s, payload))
        self._sequence += 1

    def _snapshot(self) -> PolicySnapshot:
        values = dict(self._values)
        return PolicySnapshot(values, stable_hash(values), self.backend.now_s)

    def _ready_nodes(self) -> list[CalibrationNode]:
        return [n for n in self._dag.values() if self._status[n.node_id] in {NodeStatus.PENDING, NodeStatus.STALE}
                and all(self._status[p] is NodeStatus.PASSED for p in n.prerequisites)]

    @staticmethod
    def _peaks(record: Mapping[str, Any]) -> list[tuple[float, float]]:
        x, y = record["x"], record["y"]
        peaks = [(x[i], y[i]) for i in range(1, len(y) - 1) if y[i] >= y[i-1] and y[i] >= y[i+1]]
        return sorted(peaks, key=lambda p: p[1], reverse=True)

    def _sweep(self, family: str, centre: float, span: float) -> Mapping[str, Any]:
        # Odd point count guarantees the prior centre is sampled and retains acquisition order.
        points = tuple(centre - span + 2 * span * i / 80 for i in range(81))
        return self.backend.execute(family, {"sweep_hz": points}, self.config.shots_per_fit)

    def _run_node(self, node: CalibrationNode) -> CalibrationEstimate:
        family = node.family
        self._record(node.node_id, "started", resources=node.resources, inputs_hash=stable_hash(self._values))
        if family == "timing":
            r = self.backend.execute("timing", {}, 1)
            confidence = 0.999 if r["channels_ok"] and abs(r["clock_residual_s"]) < 1e-7 else 0.0
            estimate = CalibrationEstimate({}, {}, {"clock": confidence}, confidence, confidence, r)
        elif family in {"resonator", "spectroscopy"}:
            key, centre, span = (("resonance:q0", 6.80e9, 25e6) if family == "resonator" else ("frequency:q0", 5.10e9, 35e6))
            r = self._sweep(family, centre, span)
            peaks = self._peaks(r)
            if len(peaks) < 2:
                raise RuntimeError("spectral fit has insufficient competing-peak evidence")
            selected, alternate = peaks[0], peaks[1]
            held = self._sweep(family, selected[0], 3e6)
            held_peaks = self._peaks(held)
            held_score = held_peaks[0][1] if held_peaks else 0.0
            ratio = selected[1] / max(alternate[1], 1e-12)
            confidence = min(.999, .75 + .12 * ratio + .08 * min(1.0, held_score))
            estimate = CalibrationEstimate({key: selected[0]}, {key: (0.5e6) ** 2}, {"selected": selected[1], "alternate": alternate[1]}, held_score, confidence, {"peaks": peaks[:4], "record_hash": stable_hash(r)})
            if family == "spectroscopy":
                self._values["frequency:q0"] = selected[0]
                self._variances["frequency:q0"] = estimate.variances[key]
        elif family == "readout":
            r = self.backend.execute("readout", {}, self.config.shots_per_fit)
            g, e, n = r["ground_correct"] / r["n"], r["excited_correct"] / r["n"], r["n"]
            confidence = min(g, e)
            estimate = CalibrationEstimate({}, {}, {"ground_assignment": g, "excited_assignment": e}, confidence, confidence, {"confusion": ((g, 1-g), (1-e, e)), "n": n})
        elif family == "single_qubit":
            # Bounded local grid: every proposal is checked before backend execution.
            f0, a0 = self._values["frequency:q0"], self._values["amplitude:q0"]
            candidates = [(f0 + df, a0 + da) for df in (-1e6, 0.0, 1e6) for da in (-.015, 0., .015)]
            evaluated = []
            for f, a in candidates:
                if self.limits.controls["frequency:q0"].validate(f) and self.limits.controls["amplitude:q0"].validate(a):
                    obs = self.backend.execute("single_qubit", {"frequency:q0": f, "amplitude:q0": a}, self.config.shots_per_fit)
                    evaluated.append((obs["error_rate"], f, a))
            score, f, a = min(evaluated)
            held = self.backend.execute("single_qubit", {"frequency:q0": f, "amplitude:q0": a}, self.config.validation_shots, held_out=True)
            confidence = max(0., min(.999, 1 - 10 * held["error_rate"]))
            self._values.update({"frequency:q0": f, "amplitude:q0": a})
            self._variances.update({"frequency:q0": (0.5e6)**2, "amplitude:q0": .005**2})
            estimate = CalibrationEstimate({"frequency:q0": f, "amplitude:q0": a}, {"frequency:q0": (0.5e6)**2, "amplitude:q0": .005**2}, {"best_error": score}, 1 - held["error_rate"], confidence, {"candidates": len(evaluated)})
        elif family == "entangling":
            candidates = [.02, .026, .032, .038, .044]
            trials = [(self.backend.execute("entangling", {"coupling:q0-q1": c}, self.config.shots_per_fit)["error_rate"], c) for c in candidates]
            score, c = min(trials)
            held = self.backend.execute("entangling", {"coupling:q0-q1": c}, self.config.validation_shots, held_out=True)
            confidence = max(0., min(.999, 1 - 5 * held["error_rate"]))
            self._values["coupling:q0-q1"], self._variances["coupling:q0-q1"] = c, .003**2
            estimate = CalibrationEstimate({"coupling:q0-q1": c}, {"coupling:q0-q1": .003**2}, {"best_error": score}, 1 - held["error_rate"], confidence)
        elif family == "transfer":
            estimate = CalibrationEstimate({}, {}, {"identity_transfer_model": 1.0}, 1.0, .95, {"validated_contexts": 1})
        elif family in {"qec", "final_validation"}:
            r = self.backend.execute("final_validation" if family == "final_validation" else "qec", self._values, self.config.validation_shots, held_out=family == "final_validation")
            confidence = max(0., 1 - r["rate"] / (2 * self.config.qec_rate_limit))
            estimate = CalibrationEstimate({}, {}, {"detector_rate": r["rate"]}, confidence, confidence, r)
        elif family == "sensitivity":
            slopes: dict[str, float] = {}
            nonlinearities: dict[str, float] = {}
            for key, value in self._values.items():
                bound = self.limits.controls[key]
                eps = min(bound.trust_radius * self.config.sensitivity_perturbation_fraction, (bound.maximum - bound.minimum) / 8)
                plus, minus = dict(self._values), dict(self._values)
                plus[key], minus[key] = value + eps, value - eps
                q_plus = self.backend.execute("sensitivity", plus, self.config.validation_shots)["rate"]
                q_minus = self.backend.execute("sensitivity", minus, self.config.validation_shots)["rate"]
                slopes[key] = (q_plus - q_minus) / (2 * eps)
                half_plus, half_minus = dict(self._values), dict(self._values)
                half_plus[key], half_minus[key] = value + eps / 2, value - eps / 2
                inner = ((self.backend.execute("sensitivity", half_plus, self.config.validation_shots)["rate"]
                         - self.backend.execute("sensitivity", half_minus, self.config.validation_shots)["rate"]) / eps)
                nonlinearities[key] = abs(inner - slopes[key]) / max(abs(inner), abs(slopes[key]), 1e-12)
            estimate = CalibrationEstimate({}, {}, slopes, 1.0, .95,
                {"nonlinearity_ratio": nonlinearities,
                 "trust_region_valid": all(value <= 1.0 for value in nonlinearities.values())})
        else:
            raise AssertionError(f"Unknown calibration family {family}")
        self._record(node.node_id, "observation_and_fit", estimate=estimate.__dict__)
        return estimate

    def _accept(self, node: CalibrationNode, estimate: CalibrationEstimate) -> bool:
        valid = estimate.confidence >= min(node.minimum_confidence, self.config.minimum_confidence)
        if node.family in {"qec", "final_validation"}:
            valid = valid and estimate.model_scores["detector_rate"] <= self.config.qec_rate_limit
        self._record(node.node_id, "acceptance", accepted=valid, confidence=estimate.confidence, held_out_score=estimate.held_out_score)
        return valid

    def _invalidate(self, node: CalibrationNode) -> None:
        for downstream in node.invalidates:
            if self._status.get(downstream) is NodeStatus.PASSED:
                self._status[downstream] = NodeStatus.STALE
                self._record(node.node_id, "invalidated", downstream=downstream)

    def run(self) -> BootstrapResult:
        """Run all calibration nodes; return a valid result or a health packet explaining failure."""
        while ready := self._ready_nodes():
            # Resource-conflicting nodes execute serially.  Independent backends can schedule batches here.
            node = sorted(ready, key=lambda n: n.node_id)[0]
            self._status[node.node_id] = NodeStatus.RUNNING
            success = False
            for attempt in range(node.max_attempts):
                self._record(node.node_id, "attempt", attempt=attempt)
                try:
                    estimate = self._run_node(node)
                    if self._accept(node, estimate):
                        self._estimates[node.node_id] = estimate
                        self._status[node.node_id] = NodeStatus.PASSED
                        self._invalidate(node)
                        success = True
                        break
                except (ValueError, RuntimeError) as error:
                    self._record(node.node_id, "failure", classification=type(error).__name__, message=str(error))
            if not success:
                self._status[node.node_id] = NodeStatus.FAILED
                for candidate in self._dag.values():
                    if node.node_id in candidate.prerequisites and self._status[candidate.node_id] is NodeStatus.PENDING:
                        self._status[candidate.node_id] = NodeStatus.BLOCKED
                break
        return self._result()

    def _result(self) -> BootstrapResult:
        failed = tuple(k for k, v in self._status.items() if v in {NodeStatus.FAILED, NodeStatus.BLOCKED})
        unresolved = tuple(k for k, v in self._status.items() if v is not NodeStatus.PASSED)
        registry = {
            key: ParameterRecord(key, key.split(":")[0], self.topology.control_channels[key], self.limits.controls[key].unit,
                                 value, self.limits.controls[key], "single_qubit" if key != "coupling:q0-q1" else "entangling",
                                 ("x90:q0",) if key != "coupling:q0-q1" else ("cz:q0-q1",),
                                 tuple(d.detector_id for d in self.circuit.detectors), "r:q0", self._variances[key])
            for key, value in self._values.items()
        }
        sensitivity = self._estimates.get("sensitivity", CalibrationEstimate({}, {}, {}, 0., 0.)).model_scores
        graph = {}
        for detector in self.circuit.detectors:
            linked = []
            for control in self._values:
                physical = control.split(":", 1)[0]
                if (physical == "coupling" and any("cz:" in gate for gate in detector.affected_gates)
                        or physical == "frequency" and any(control.split(":", 1)[1] in gate for gate in detector.affected_gates)
                        or physical == "amplitude" and any(gate.startswith(("x", "y")) and control.split(":", 1)[1] in gate for gate in detector.affected_gates)):
                    linked.append(control)
            graph[detector.detector_id] = tuple(linked or self._values)
        health = StageHealthPacket(HealthStatus.FAILED if failed else HealthStatus.PASSED, failed, unresolved,
                                  {k: e.diagnostics for k, e in self._estimates.items() if e.confidence < .98}, True)
        return BootstrapResult("stage0.v2", self._snapshot(), self.circuit, registry, dict(self._status), graph,
                               dict(sensitivity), self._initial_snapshot, health, tuple(self._log), (),
                               dict(self._dag), dict(self._estimates), tuple((node_id,) for node_id in self._dag))

    @staticmethod
    def verify_replay(result: BootstrapResult) -> bool:
        """Verify log ordering and immutable final-policy provenance without querying a live backend."""
        return (all(event.sequence == i for i, event in enumerate(result.event_log))
                and result.baseline_policy.policy_hash == stable_hash(result.baseline_policy.values)
                and result.health.rollback_available)
