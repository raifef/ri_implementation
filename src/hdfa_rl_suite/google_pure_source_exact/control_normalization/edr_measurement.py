"""Finite-shot detector-event evaluators used by the sensitivity sweeps.

The Stim evaluator constructs and samples a detector-generating circuit for
every policy candidate.  Synthetic gate sensitivities are used only to create
the circuit's physical noise probabilities; EDR is always obtained by counting
sampled detector events, never by returning the analytic sensitivity model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import math
from typing import Mapping, Protocol, Sequence

import numpy as np

from .contracts import ControlTypeSpec, FrozenReference, canonical_hash


@dataclass(frozen=True)
class EDRCountMeasurement:
    detector_events: int
    detector_opportunities: int
    candidates: int
    shots_per_candidate: int
    qec_cycles: int
    detector_count: int
    candidate_detector_events: tuple[int, ...]
    candidate_detector_opportunities: int

    @property
    def edr_fraction(self) -> float:
        return self.detector_events / self.detector_opportunities

    @property
    def edr_percentage_points(self) -> float:
        return 100.0 * self.edr_fraction


class DetectorEventEvaluator(Protocol):
    control_specs: tuple[ControlTypeSpec, ...]
    reference: FrozenReference
    plant_hash: str

    def measure_joint(self, sigma_native_by_type: Mapping[str, float], *, candidates: int,
                      shots_per_candidate: int, perturbation_seed: int,
                      detector_seed: int) -> EDRCountMeasurement: ...


def _derived_seed(seed: int, *parts: object) -> int:
    material = ":".join([str(int(seed)), *(str(part) for part in parts)])
    return int.from_bytes(sha256(material.encode("utf-8")).digest()[:8], "little") & ((1 << 63) - 1)


class StimSurfaceCodeEDREvaluator:
    """Actual Stim detector sampler for the source-exact synthetic plant.

    Each selected control type supplies independent Gaussian perturbations to
    every registered gate.  Their fixed synthetic mismatch-to-error map changes
    a Stim circuit noise channel.  The returned objective is the empirical
    detector-event fraction from ``compile_detector_sampler``.
    """

    def __init__(self, control_specs: Sequence[ControlTypeSpec], *, distance: int = 3,
                 rounds: int = 3, basis: str = "z",
                 base_noise: Mapping[str, float] | None = None,
                 maximum_probability: float = 0.12) -> None:
        try:
            import stim
        except ImportError as error:  # pragma: no cover - dependency is declared by pyproject.
            raise RuntimeError("Stim is required for source-exact detector evaluation") from error
        if distance < 3 or distance % 2 == 0 or rounds <= 0 or basis.lower() not in {"x", "z"}:
            raise ValueError("invalid rotated surface-code circuit specification")
        self._stim = stim
        self.control_specs = tuple(control_specs)
        if not self.control_specs:
            raise ValueError("at least one control type is required")
        names = [item.control_type for item in self.control_specs]
        if len(names) != len(set(names)):
            raise ValueError("control types must be unique")
        self._by_type = {item.control_type: item for item in self.control_specs}
        self.distance, self.rounds, self.basis = int(distance), int(rounds), basis.lower()
        defaults = {
            "after_clifford_depolarization": 0.0010,
            "before_round_data_depolarization": 0.0005,
            "before_measure_flip_probability": 0.0010,
            "after_reset_flip_probability": 0.0010,
        }
        if base_noise is not None:
            defaults.update({str(key): float(value) for key, value in base_noise.items()})
        if set(defaults) != set(StimSurfaceCodeEDREvaluator.noise_channels()):
            raise ValueError("base_noise must define exactly the supported Stim channels")
        if any(not 0 <= value < maximum_probability for value in defaults.values()):
            raise ValueError("base noise lies outside the declared safety range")
        self.base_noise = defaults
        self.maximum_probability = float(maximum_probability)
        reference_circuit = self._circuit(defaults).flattened()
        self._reference_circuit = reference_circuit
        self.detector_count = int(reference_circuit.num_detectors)
        detector_ids = tuple(f"D{index}" for index in range(self.detector_count))
        circuit_hash = sha256(str(reference_circuit).encode("utf-8")).hexdigest()
        registry = [
            {
                "control_type": item.control_type,
                "gate_ids": item.gate_ids,
                "native_unit": item.native_unit,
                "reference_value_native": item.reference_value_native,
            }
            for item in self.control_specs
        ]
        self.reference = FrozenReference(
            reference_policy_hash=canonical_hash(
                {item.control_type: item.reference_value_native for item in self.control_specs}),
            circuit_hash=circuit_hash,
            detector_set_hash=canonical_hash(list(detector_ids)),
            detector_ids=detector_ids,
            parameter_registry_hash=canonical_hash(registry),
        )
        self.plant_hash = canonical_hash({
            "engine": "stim.compile_detector_sampler with per-gate candidate noise injection",
            "stim_version": str(getattr(stim, "__version__", "unknown")),
            "distance": self.distance,
            "rounds": self.rounds,
            "basis": self.basis,
            "base_noise": self.base_noise,
            "maximum_probability": self.maximum_probability,
            "control_specs": [asdict(item) for item in self.control_specs],
        })

    @staticmethod
    def noise_channels() -> tuple[str, ...]:
        return (
            "after_clifford_depolarization",
            "before_round_data_depolarization",
            "before_measure_flip_probability",
            "after_reset_flip_probability",
        )

    @property
    def circuit_task(self) -> str:
        return f"surface_code:rotated_memory_{self.basis}"

    def _circuit(self, noise: Mapping[str, float]):
        return self._stim.Circuit.generated(
            self.circuit_task,
            distance=self.distance,
            rounds=self.rounds,
            after_clifford_depolarization=float(noise["after_clifford_depolarization"]),
            before_round_data_depolarization=float(noise["before_round_data_depolarization"]),
            before_measure_flip_probability=float(noise["before_measure_flip_probability"]),
            after_reset_flip_probability=float(noise["after_reset_flip_probability"]),
        )

    @staticmethod
    def _is_qubit_target(target) -> bool:
        return bool(getattr(target, "is_qubit_target", False))

    def _candidate_circuit(self, perturbations: Mapping[str, np.ndarray]):
        """Inject each registered gate's mismatch into an actual Stim operation.

        Gate parameters are cycled across repeated physical operation instances,
        so every registered gate contributes during every candidate.  The
        candidate circuit is then sampled; no analytic EDR shortcut exists.
        """
        injected = {
            name: self._by_type[name].synthetic_probability_gain * np.square(values)
            for name, values in perturbations.items()
        }
        for name, values in injected.items():
            spec = self._by_type[name]
            base = self.base_noise[spec.stim_error_channel]
            if np.any(values + base >= self.maximum_probability):
                raise ValueError("candidate exceeds the preregistered Stim noise safety envelope")
        counters = {name: 0 for name in injected}

        def next_probability(name: str) -> float:
            values = injected[name]
            index = counters[name] % len(values)
            counters[name] += 1
            return float(values[index])

        circuit = self._stim.Circuit()
        for instruction in self._reference_circuit:
            name = instruction.name
            targets = [target for target in instruction.targets_copy() if self._is_qubit_target(target)]
            # Measurement-channel control errors act immediately before the
            # physical measurement, while reset errors act after reset.
            if name in {"M", "MR", "MX", "MRX", "MY", "MRY"}:
                for control_name in injected:
                    if self._by_type[control_name].stim_error_channel == "before_measure_flip_probability":
                        for target in targets:
                            probability = next_probability(control_name)
                            if probability > 0:
                                circuit.append("X_ERROR", [target], probability)
            circuit.append(instruction)
            for control_name in injected:
                channel = self._by_type[control_name].stim_error_channel
                if channel == "after_clifford_depolarization" and name in {
                    "H", "H_XY", "H_XZ", "H_YZ", "RX", "RY", "X", "Y", "Z"
                }:
                    for target in targets:
                        probability = next_probability(control_name)
                        if probability > 0:
                            circuit.append("DEPOLARIZE1", [target], probability)
                elif channel == "before_round_data_depolarization" and name in {"CX", "CZ"}:
                    if len(targets) % 2:
                        raise RuntimeError("two-qubit Stim instruction has an odd target count")
                    for index in range(0, len(targets), 2):
                        probability = next_probability(control_name)
                        if probability > 0:
                            circuit.append("DEPOLARIZE2", targets[index:index + 2], probability)
                elif channel == "after_reset_flip_probability" and name in {
                    "R", "RX", "RY", "MR", "MRX", "MRY"
                }:
                    for target in targets:
                        probability = next_probability(control_name)
                        if probability > 0:
                            circuit.append("X_ERROR", [target], probability)
        unused = [name for name, count in counters.items() if count < len(injected[name])]
        if unused:
            raise RuntimeError(f"not every registered gate was applied to the Stim circuit: {unused}")
        return circuit

    def measure_control_type(self, control_type: str, sigma_native: float, *, candidates: int,
                             shots_per_candidate: int, perturbation_seed: int,
                             detector_seed: int) -> EDRCountMeasurement:
        return self.measure_joint(
            {control_type: sigma_native}, candidates=candidates,
            shots_per_candidate=shots_per_candidate,
            perturbation_seed=perturbation_seed, detector_seed=detector_seed)

    def measure_joint(self, sigma_native_by_type: Mapping[str, float], *, candidates: int,
                      shots_per_candidate: int, perturbation_seed: int,
                      detector_seed: int) -> EDRCountMeasurement:
        if candidates <= 0 or shots_per_candidate <= 0:
            raise ValueError("finite-shot budgets must be positive")
        unknown = set(sigma_native_by_type) - set(self._by_type)
        if unknown:
            raise ValueError(f"unknown control types: {sorted(unknown)}")
        sigmas = {name: float(value) for name, value in sigma_native_by_type.items()}
        if any(not math.isfinite(value) or value < 0 for value in sigmas.values()):
            raise ValueError("perturbation sigma must be finite and non-negative")
        perturbation_rng = np.random.default_rng(int(perturbation_seed))
        events = 0
        candidate_events: list[int] = []
        for candidate in range(candidates):
            perturbations: dict[str, np.ndarray] = {}
            for name, sigma in sigmas.items():
                spec = self._by_type[name]
                # Every gate is perturbed in the same candidate.  Independent
                # draws reproduce a factorized Gaussian policy over instances.
                perturbations[name] = perturbation_rng.normal(0.0, sigma, len(spec.gate_ids))
            circuit = self._candidate_circuit(perturbations)
            sampler = circuit.compile_detector_sampler(
                seed=_derived_seed(detector_seed, candidate, canonical_hash(sigmas)))
            sample = sampler.sample(shots=shots_per_candidate, bit_packed=False)
            if sample.shape != (shots_per_candidate, self.detector_count):
                raise RuntimeError("Stim detector shape changed under the frozen circuit")
            candidate_count = int(np.count_nonzero(sample))
            candidate_events.append(candidate_count)
            events += candidate_count
        opportunities = candidates * shots_per_candidate * self.detector_count
        return EDRCountMeasurement(
            detector_events=events,
            detector_opportunities=opportunities,
            candidates=candidates,
            shots_per_candidate=shots_per_candidate,
            qec_cycles=candidates * shots_per_candidate * self.rounds,
            detector_count=self.detector_count,
            candidate_detector_events=tuple(candidate_events),
            candidate_detector_opportunities=shots_per_candidate * self.detector_count,
        )


class QuadraticSyntheticEDREvaluator:
    """Fast exact-response evaluator for statistical and unit tests."""

    def __init__(self, control_specs: Sequence[ControlTypeSpec],
                 sigma0_native_by_type: Mapping[str, float], *,
                 edr0_percentage_points: float = 2.0, detector_count: int = 24,
                 qec_rounds_per_shot: int = 3,
                 quartic_by_type: Mapping[str, float] | None = None,
                 linear_variance_by_type: Mapping[str, float] | None = None) -> None:
        self.control_specs = tuple(control_specs)
        self._sigma0 = {str(key): float(value) for key, value in sigma0_native_by_type.items()}
        names = {item.control_type for item in self.control_specs}
        if names != set(self._sigma0) or any(value <= 0 for value in self._sigma0.values()):
            raise ValueError("one positive true sigma0 is required per control type")
        self.edr0_percentage_points = float(edr0_percentage_points)
        self.detector_count = int(detector_count)
        self.rounds = int(qec_rounds_per_shot)
        self._quartic = {name: float(value) for name, value in (quartic_by_type or {}).items()}
        self._linear_variance = {
            name: float(value) for name, value in (linear_variance_by_type or {}).items()
        }
        detector_ids = tuple(f"synthetic-D{index}" for index in range(self.detector_count))
        self.reference = FrozenReference(
            reference_policy_hash=canonical_hash({"synthetic_reference": 0}),
            circuit_hash=canonical_hash({"synthetic_circuit": 1}),
            detector_set_hash=canonical_hash(list(detector_ids)),
            detector_ids=detector_ids,
            parameter_registry_hash=canonical_hash(
                [(item.control_type, item.gate_ids, item.native_unit) for item in self.control_specs]),
        )
        self.plant_hash = canonical_hash({
            "kind": "quadratic_synthetic_edr",
            "sigma0": self._sigma0,
            "edr0_percentage_points": self.edr0_percentage_points,
            "quartic": self._quartic,
            "linear_variance": self._linear_variance,
        })

    def expected_edr_percentage_points(self, sigma_native_by_type: Mapping[str, float]) -> float:
        result = self.edr0_percentage_points
        for name, sigma in sigma_native_by_type.items():
            x = (float(sigma) / self._sigma0[name]) ** 2
            result += x + self._quartic.get(name, 0.0) * x * x
            result += self._linear_variance.get(name, 0.0) * float(sigma)
        return result

    def measure_control_type(self, control_type: str, sigma_native: float, *, candidates: int,
                             shots_per_candidate: int, perturbation_seed: int,
                             detector_seed: int) -> EDRCountMeasurement:
        return self.measure_joint(
            {control_type: sigma_native}, candidates=candidates,
            shots_per_candidate=shots_per_candidate,
            perturbation_seed=perturbation_seed, detector_seed=detector_seed)

    def measure_joint(self, sigma_native_by_type: Mapping[str, float], *, candidates: int,
                      shots_per_candidate: int, perturbation_seed: int,
                      detector_seed: int) -> EDRCountMeasurement:
        del perturbation_seed  # The exact test plant integrates the Gaussian analytically.
        if candidates <= 0 or shots_per_candidate <= 0:
            raise ValueError("finite-shot budgets must be positive")
        unknown = set(sigma_native_by_type) - set(self._sigma0)
        if unknown:
            raise ValueError(f"unknown control types: {sorted(unknown)}")
        rate = self.expected_edr_percentage_points(sigma_native_by_type) / 100.0
        if not 0 <= rate <= 1:
            raise ValueError("synthetic EDR is not a probability")
        opportunities = candidates * shots_per_candidate * self.detector_count
        rng = np.random.default_rng(int(detector_seed))
        candidate_opportunities = shots_per_candidate * self.detector_count
        candidate_events = tuple(int(value) for value in rng.binomial(
            candidate_opportunities, rate, size=candidates))
        events = sum(candidate_events)
        return EDRCountMeasurement(
            detector_events=events,
            detector_opportunities=opportunities,
            candidates=candidates,
            shots_per_candidate=shots_per_candidate,
            qec_cycles=candidates * shots_per_candidate * self.rounds,
            detector_count=self.detector_count,
            candidate_detector_events=candidate_events,
            candidate_detector_opportunities=candidate_opportunities,
        )
