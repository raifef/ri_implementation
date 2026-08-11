"""A deterministic, observation-only backend for Stage-0 integration tests and demos."""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Mapping

from .schema import (
    ControlBound, DetectorDefinition, DeviceTopology, HardwareLimits, TargetQECCircuit,
)


@dataclass
class SimulatedCalibrationBackend:
    """Synthetic device that exposes experiments, never latent parameters, to the calibrator."""

    topology: DeviceTopology
    limits: HardwareLimits
    seed: int = 0
    drift_per_experiment: float = 0.0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._time = 0.0
        # Intentionally private: bootstrap only receives noisy experimental records.
        self._truth = {
            "resonance:q0": 6.220e9,
            "readout:q0": 6.800e9,
            "frequency:q0": 5.100e9,
            "amplitude:q0": 0.505,
            "phase:q0": 0.015,
            "drag:q0": 0.08,
            "coupling:q0-q1": 0.032,
        }
        if len(self.topology.qubits) > 1:
            self._truth["frequency:q1"] = 5.230e9
            self._truth["amplitude:q1"] = 0.495

    @property
    def now_s(self) -> float:
        return self._time

    def execute(self, family: str, parameters: Mapping[str, float], shots: int, held_out: bool = False) -> dict[str, Any]:
        """Return timestamped noisy records.  No response contains a latent-truth field."""
        self._time += max(1, shots) * 1e-6
        noise = 0.003 if held_out else 0.005
        if family == "timing":
            return {"timestamp_s": self._time, "clock_residual_s": self._rng.gauss(0, 2e-9), "channels_ok": True}
        if family in {"resonator", "spectroscopy"}:
            key = "readout:q0" if family == "resonator" else "frequency:q0"
            centre = self._truth[key] + self.drift_per_experiment * self._time
            width = 2.5e6 if family == "resonator" else 1.2e6
            xs = list(parameters["sweep_hz"])
            ys = [
                sum(math.exp(-0.5 * ((x - p) / width) ** 2) * w for p, w in ((centre, 1.0), (centre + 5.5e6, 0.36)))
                + self._rng.gauss(0, noise)
                for x in xs
            ]
            return {"timestamp_s": self._time, "x": xs, "y": ys}
        if family == "readout":
            # counts of classified ground/excited outcomes with a modest confusion matrix
            n = shots
            return {"timestamp_s": self._time, "n": n, "ground_correct": sum(self._rng.random() < .975 for _ in range(n)), "excited_correct": sum(self._rng.random() < .960 for _ in range(n))}
        if family == "single_qubit":
            amp = parameters["amplitude:q0"]
            detuning = parameters["frequency:q0"] - self._truth["frequency:q0"]
            error = (amp - self._truth["amplitude:q0"]) ** 2 * 6 + (detuning / 2e6) ** 2 * .015
            return {"timestamp_s": self._time, "error_rate": max(0.0, .006 + error + self._rng.gauss(0, noise / 3)), "n": shots}
        if family == "entangling":
            c = parameters["coupling:q0-q1"]
            error = (c - self._truth["coupling:q0-q1"]) ** 2 * 8
            return {"timestamp_s": self._time, "error_rate": max(0.0, .012 + error + self._rng.gauss(0, noise / 2)), "n": shots}
        if family in {"qec", "sensitivity", "final_validation"}:
            policy = parameters
            detuning = abs(policy.get("frequency:q0", self._truth["frequency:q0"]) - self._truth["frequency:q0"]) / 2e6
            amp = abs(policy.get("amplitude:q0", self._truth["amplitude:q0"]) - self._truth["amplitude:q0"])
            coupling = abs(policy.get("coupling:q0-q1", self._truth["coupling:q0-q1"]) - self._truth["coupling:q0-q1"])
            rate = min(.49, .015 + .015 * detuning**2 + 3 * amp**2 + 2 * coupling**2)
            events = sum(self._rng.random() < rate for _ in range(shots))
            return {"timestamp_s": self._time, "events": events, "exposures": shots, "rate": events / shots}
        raise ValueError(f"Unsupported experiment family: {family}")


def demo_topology() -> tuple[DeviceTopology, HardwareLimits, TargetQECCircuit]:
    topology = DeviceTopology(
        device_id="sim-qec-2q", qubits=("q0", "q1"), couplers=(("q0", "q1"),),
        resonators={"q0": "r0", "q1": "r1"},
        control_channels={"frequency:q0": "d0", "amplitude:q0": "d0", "coupling:q0-q1": "c0"},
    )
    limits = HardwareLimits(controls={
        "frequency:q0": ControlBound(5.05e9, 5.15e9, 5e6, "Hz", 5e6),
        "amplitude:q0": ControlBound(.35, .65, .05, "arb", .06),
        "coupling:q0-q1": ControlBound(.0, .08, .02, "arb", .02),
    })
    detectors = (
        DetectorDefinition("d0", (0, 1), 0, ("x90:q0", "cz:q0-q1"), "r:q0"),
        DetectorDefinition("d1", (1, 2), 0, ("cz:q0-q1",), "r:q0"),
    )
    circuit = TargetQECCircuit("memory-2q", "demo-circuit-v1", ("x90:q0", "cz:q0-q1", "measure:q0"), detectors)
    return topology, limits, circuit
