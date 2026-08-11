"""Stim + PyMatching logical-memory evidence for controlled physical errors.

The adapter is intentionally evaluation-only.  It maps the simulator's declared
control-error vector into circuit-level Pauli/reset/measurement probabilities, samples a
named rotated surface-code memory circuit with Stim, and decodes the resulting detector
events using a fixed-noise PyMatching MWPM decoder.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib import metadata
import math
from typing import Mapping

import numpy as np

from hdfa_rl_suite.simulator import ScalableQECDevice


class LogicalStackUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class SurfaceCodeMemoryConfig:
    distance: int = 3
    rounds: int = 3
    shots: int = 256
    basis: str = "z"
    decoder_noise_probability: float = 0.002

    def __post_init__(self) -> None:
        if self.distance < 3 or self.distance % 2 == 0:
            raise ValueError("rotated surface-code distance must be an odd integer >= 3")
        if self.rounds <= 0 or self.shots <= 0:
            raise ValueError("rounds and shots must be positive")
        if self.basis.lower() not in {"x", "z"}:
            raise ValueError("basis must be x or z")
        if not 0 <= self.decoder_noise_probability < 1:
            raise ValueError("decoder noise probability must lie in [0, 1)")


@dataclass(frozen=True)
class ControlErrorNoiseMap:
    """Declared mapping from normalized control mismatch to circuit-level Pauli noise."""

    base_gate_depolarization: float = 0.001
    base_data_depolarization: float = 0.0005
    base_measurement_flip: float = 0.001
    base_reset_flip: float = 0.001
    gate_energy_gain: float = 0.08
    data_energy_gain: float = 0.04
    measurement_rms_gain: float = 0.01
    reset_rms_gain: float = 0.005
    maximum_probability: float = 0.2

    def map(self, control_errors: Mapping[str, float]) -> Mapping[str, float]:
        values = tuple(float(value) for value in control_errors.values())
        energy = sum(value * value for value in values) / max(1, len(values))
        rms = math.sqrt(energy)
        cap = self.maximum_probability
        return {
            "after_clifford_depolarization": min(cap, self.base_gate_depolarization + self.gate_energy_gain * energy),
            "before_round_data_depolarization": min(cap, self.base_data_depolarization + self.data_energy_gain * energy),
            "before_measure_flip_probability": min(cap, self.base_measurement_flip + self.measurement_rms_gain * rms),
            "after_reset_flip_probability": min(cap, self.base_reset_flip + self.reset_rms_gain * rms),
            "control_error_energy": energy,
            "control_error_rms": rms,
        }


@dataclass(frozen=True)
class LogicalPerformanceEvidence:
    schema_version: str
    stack_id: str
    circuit_task: str
    distance: int
    rounds: int
    decoder: str
    shots: int
    logical_failures: int
    logical_failure_probability: float
    logical_error_per_round: float
    confidence_interval_95: tuple[float, float]
    physical_noise: Mapping[str, float]
    sample_seed: int
    circuit_sha256: str
    stim_version: str
    pymatching_version: str
    physical_state_id: str = ""
    policy_hash: str = ""
    disturbance_state_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class RotatedSurfaceCodeEvaluator:
    """Reproducible rotated-memory circuit sampler and fixed MWPM decoder."""

    def __init__(self, config: SurfaceCodeMemoryConfig = SurfaceCodeMemoryConfig(),
                 noise_map: ControlErrorNoiseMap = ControlErrorNoiseMap()) -> None:
        self.config, self.noise_map = config, noise_map
        try:
            import stim
            import pymatching
        except ImportError as error:  # pragma: no cover - depends on optional system state.
            raise LogicalStackUnavailable(
                "logical evidence requires the declared stim and pymatching dependencies") from error
        self._stim, self._pymatching = stim, pymatching
        nominal = self._circuit({
            "after_clifford_depolarization": config.decoder_noise_probability,
            "before_round_data_depolarization": config.decoder_noise_probability,
            "before_measure_flip_probability": config.decoder_noise_probability,
            "after_reset_flip_probability": config.decoder_noise_probability,
        })
        detector_error_model = nominal.detector_error_model(decompose_errors=True)
        self._decoder = pymatching.Matching.from_detector_error_model(detector_error_model)

    @property
    def circuit_task(self) -> str:
        return f"surface_code:rotated_memory_{self.config.basis.lower()}"

    def _circuit(self, noise: Mapping[str, float]):
        return self._stim.Circuit.generated(
            self.circuit_task,
            distance=self.config.distance,
            rounds=self.config.rounds,
            after_clifford_depolarization=float(noise["after_clifford_depolarization"]),
            before_round_data_depolarization=float(noise["before_round_data_depolarization"]),
            before_measure_flip_probability=float(noise["before_measure_flip_probability"]),
            after_reset_flip_probability=float(noise["after_reset_flip_probability"]),
        )

    @staticmethod
    def _wilson(failures: int, shots: int) -> tuple[float, float]:
        z = 1.959963984540054
        p = failures / shots
        denominator = 1 + z * z / shots
        centre = (p + z * z / (2 * shots)) / denominator
        radius = z * math.sqrt(p * (1-p) / shots + z*z / (4*shots*shots)) / denominator
        return max(0.0, centre-radius), min(1.0, centre+radius)

    def evaluate(self, control_errors: Mapping[str, float], *, seed: int,
                 physical_state_id: str = "", policy_hash: str = "",
                 disturbance_state_id: str = "") -> LogicalPerformanceEvidence:
        noise = self.noise_map.map(control_errors)
        circuit = self._circuit(noise)
        sampler = circuit.compile_detector_sampler(seed=int(seed) & ((1 << 63) - 1))
        detection_events, actual_observables = sampler.sample(
            shots=self.config.shots, separate_observables=True)
        predicted_observables = self._decoder.decode_batch(detection_events)
        actual = np.asarray(actual_observables, dtype=np.bool_)
        predicted = np.asarray(predicted_observables, dtype=np.bool_)
        if actual.ndim == 1:
            actual = actual[:, None]
        if predicted.ndim == 1:
            predicted = predicted[:, None]
        failures = int(np.count_nonzero(np.any(predicted != actual, axis=1)))
        probability = failures / self.config.shots
        per_round = 1 - (1-probability) ** (1 / self.config.rounds)
        try:
            stim_version = metadata.version("stim")
        except metadata.PackageNotFoundError:  # pragma: no cover
            stim_version = str(getattr(self._stim, "__version__", "unknown"))
        try:
            matching_version = metadata.version("pymatching")
        except metadata.PackageNotFoundError:  # pragma: no cover
            matching_version = str(getattr(self._pymatching, "__version__", "unknown"))
        circuit_hash = sha256(str(circuit).encode("utf-8")).hexdigest()
        return LogicalPerformanceEvidence(
            "logical-evidence.v1", "stim+pymatching-mwpm.v1", self.circuit_task,
            self.config.distance, self.config.rounds, "PyMatching MWPM (fixed nominal DEM)",
            self.config.shots, failures, probability, per_round,
            self._wilson(failures, self.config.shots), dict(noise), int(seed),
            circuit_hash, stim_version, matching_version,
            physical_state_id, policy_hash, disturbance_state_id,
        )

    def evaluate_device(self, device: ScalableQECDevice, *, seed: int) -> LogicalPerformanceEvidence:
        view = device.oracle_evaluation_view("evaluation:stim-pymatching-logical-performance")
        diagnostic = view.physical_diagnostic()
        return self.evaluate(
            diagnostic.mismatch, seed=seed,
            physical_state_id=diagnostic.state_id,
            policy_hash=diagnostic.policy_hash,
            disturbance_state_id=diagnostic.disturbance_state_id,
        )
