"""Distance-3, 41-gate-parameter Stim plant from Supplement VI.A."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import math
from typing import Any, Iterable

import numpy as np

from .contracts import canonical_hash


@dataclass(frozen=True)
class GateParameter:
    parameter_id: str
    gate_type: str
    qubits: tuple[int, ...]
    circuit_locations: tuple[str, ...]
    detectors_influenced: tuple[int, ...]
    irreducible_error: float
    omega_sensitivity: float


@dataclass(frozen=True)
class DetectorObservation:
    """One Stim acquisition with source-level and raw-detector views.

    ``reward_component_counts`` is the mean count within each
    time-translation equivalence class.  Dividing it by ``shots`` therefore
    produces the detector-rate vector used by the learner without making its
    scale depend on the number of circuit rounds.  ``raw_total`` is retained
    separately because the Figure 5a performance ratio is defined from the
    total number of detection events, not from the reduced reward vector.
    """

    raw_counts: np.ndarray
    reward_component_counts: np.ndarray
    shots: int

    def __post_init__(self) -> None:
        raw = np.asarray(self.raw_counts, dtype=np.int64)
        reduced = np.asarray(self.reward_component_counts, dtype=float)
        if raw.ndim != 1 or reduced.ndim != 1 or self.shots <= 0:
            raise ValueError("invalid detector observation")
        raw = raw.copy(); reduced = reduced.copy()
        raw.setflags(write=False); reduced.setflags(write=False)
        object.__setattr__(self, "raw_counts", raw)
        object.__setattr__(self, "reward_component_counts", reduced)

    @property
    def raw_total(self) -> int:
        return int(self.raw_counts.sum())

    @property
    def reward_rates(self) -> np.ndarray:
        return self.reward_component_counts / float(self.shots)


def _derived_seed(seed: int, *parts: object) -> int:
    material = ":".join([str(int(seed)), *(str(part) for part in parts)])
    return int.from_bytes(sha256(material.encode("utf-8")).digest()[:8], "little") & ((1 << 63) - 1)


class Figure5aStimPlant:
    """A gate-local Stim circuit with 17 one-qubit and 24 two-qubit controls."""

    def __init__(self, *, rounds: int, basis: str, ensemble_seed: int,
                 one_qubit_irreducible: tuple[float, float],
                 two_qubit_irreducible: tuple[float, float],
                 one_qubit_omega: tuple[float, float],
                 two_qubit_omega: tuple[float, float],
                 maximum_probability: float = 0.08,
                 action_probability_margin_fraction: float = 1e-6) -> None:
        try:
            import stim
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("Stim is required") from error
        if rounds <= 0 or basis not in {"x", "z"}:
            raise ValueError("invalid distance-3 memory circuit")
        self._stim = stim
        self.rounds, self.basis = int(rounds), str(basis)
        self.maximum_probability = float(maximum_probability)
        self.action_probability_margin_fraction = float(action_probability_margin_fraction)
        if not 0.0 < self.action_probability_margin_fraction < 1.0:
            raise ValueError("action probability margin fraction must lie in (0,1)")
        self._reference = stim.Circuit.generated(
            f"surface_code:rotated_memory_{basis}", distance=3, rounds=rounds).flattened()
        self.raw_detector_count = int(self._reference.num_detectors)
        qubits, edges, one_locations, two_locations = self._enumerate_topology()
        if len(qubits) != 17 or len(edges) != 24:
            raise RuntimeError(f"distance-3 inventory mismatch: {len(qubits)} one-qubit, {len(edges)} two-qubit")
        rng = np.random.default_rng(int(ensemble_seed))
        inventory: list[GateParameter] = []
        for q in qubits:
            inventory.append(GateParameter(
                f"sq-q{q}", "single_qubit", (q,), tuple(one_locations[q]), (),
                float(rng.uniform(*one_qubit_irreducible)), float(rng.uniform(*one_qubit_omega))))
        for left, right in edges:
            inventory.append(GateParameter(
                f"tq-q{left}-q{right}", "two_qubit", (left, right), tuple(two_locations[(left, right)]), (),
                float(rng.uniform(*two_qubit_irreducible)), float(rng.uniform(*two_qubit_omega))))
        self._inventory = tuple(inventory)
        raw_parameter_detectors = [self._dem_detectors(index) for index in range(len(inventory))]
        raw_mask = np.zeros((self.raw_detector_count, len(inventory)), dtype=bool)
        for parameter, detectors in enumerate(raw_parameter_detectors):
            raw_mask[list(detectors), parameter] = True
        self.reward_component_raw_detectors, self.reward_component_keys = \
            self._time_translation_equivalence_classes(raw_mask)
        self.detector_count = len(self.reward_component_raw_detectors)
        self.mask = np.zeros((self.detector_count, len(inventory)), dtype=bool)
        for component, raw_detectors in enumerate(self.reward_component_raw_detectors):
            self.mask[component] = np.any(raw_mask[list(raw_detectors)], axis=0)
        self.inventory = tuple(replace(
            item,
            detectors_influenced=tuple(np.flatnonzero(self.mask[:, index]).astype(int).tolist()),
        ) for index, item in enumerate(inventory))
        irreducible = np.asarray([item.irreducible_error for item in self.inventory])
        omega = np.asarray([item.omega_sensitivity for item in self.inventory])
        probability_ceiling = self.maximum_probability * (
            1.0 - self.action_probability_margin_fraction)
        self._maximum_mismatch = np.sqrt((probability_ceiling - irreducible) / omega)
        self._control_limits = self._maximum_mismatch - 1.0
        if (not np.all(np.isfinite(self._control_limits)) or
                np.any(self._control_limits <= 1.0)):
            raise ValueError(
                "frozen Figure 5a plant has no symmetric action domain containing the full optimum range")
        self._control_limits.setflags(write=False)
        self._maximum_mismatch.setflags(write=False)
        if not self.mask.any(axis=0).all() or not self.mask.any(axis=1).all():
            raise RuntimeError("Stim-derived detector mask has an empty parameter or detector")
        self.parameter_ids = tuple(item.parameter_id for item in self.inventory)
        self.plant_hash = canonical_hash({
            "stim_version": getattr(stim, "__version__", "unknown"), "distance": 3,
            "rounds": rounds, "basis": basis, "maximum_probability": maximum_probability,
            "action_probability_margin_fraction": self.action_probability_margin_fraction,
            "inventory": [asdict(item) for item in self.inventory], "mask": self.mask.astype(int).tolist(),
            "raw_detector_count": self.raw_detector_count,
            "reward_representation": "time_translation_equivalence_class_mean_edr",
            "reward_component_raw_detectors": [list(group) for group in self.reward_component_raw_detectors],
            "reward_component_keys": list(self.reward_component_keys),
            "ensemble_seed": int(ensemble_seed),
            "action_execution": "plant_derived_per_coordinate_scaled_tanh",
            "control_limits": self._control_limits.tolist(),
        })

    def _time_translation_equivalence_classes(
        self, raw_mask: np.ndarray,
    ) -> tuple[tuple[tuple[int, ...], ...], tuple[str, ...]]:
        """Combine only raw detectors with identical translated semantics.

        Stim's generated memory circuits encode detector coordinates as
        ``(space..., time)``.  Equal spatial coordinates alone are insufficient
        at the initial/final boundaries, whose detecting regions differ.  The
        class key therefore combines the spatial coordinate with the exact
        control-dependency row.  Interior copies collapse across time while
        non-equivalent boundaries remain distinct; the resulting reward size is
        independent of the number of repeated interior rounds.
        """
        coordinates = self._reference.get_detector_coordinates()
        grouped: dict[tuple[tuple[float, ...], tuple[int, ...]], list[int]] = {}
        for detector in range(self.raw_detector_count):
            coordinate = tuple(float(value) for value in coordinates.get(detector, ()))
            if len(coordinate) < 2:
                raise RuntimeError("Figure 5a detector lacks space-time coordinates")
            spatial = coordinate[:-1]
            dependencies = tuple(np.flatnonzero(raw_mask[detector]).astype(int).tolist())
            grouped.setdefault((spatial, dependencies), []).append(detector)
        groups = tuple(tuple(values) for values in grouped.values())
        keys = tuple(canonical_hash({"space": key[0], "controls": key[1]}) for key in grouped)
        if sorted(detector for group in groups for detector in group) != list(range(self.raw_detector_count)):
            raise RuntimeError("time-translation reduction lost or duplicated a detector")
        return groups, keys

    def _enumerate_topology(self):
        qubits: set[int] = set()
        edges: set[tuple[int, int]] = set()
        one_locations: dict[int, list[str]] = {}
        two_locations: dict[tuple[int, int], list[str]] = {}
        for index, instruction in enumerate(self._reference):
            targets = [int(target.value) for target in instruction.targets_copy() if target.is_qubit_target]
            qubits.update(targets)
            if instruction.name == "CX":
                for offset in range(0, len(targets), 2):
                    edge = tuple(sorted((targets[offset], targets[offset + 1])))
                    edges.add(edge)
                    two_locations.setdefault(edge, []).append(f"instruction:{index}:CX")
            elif instruction.name in {"R", "RX", "H", "M", "MX", "MR", "MRX"}:
                for q in targets:
                    one_locations.setdefault(q, []).append(f"instruction:{index}:{instruction.name}")
        ordered_qubits, ordered_edges = tuple(sorted(qubits)), tuple(sorted(edges))
        if set(one_locations) != set(ordered_qubits) or set(two_locations) != set(ordered_edges):
            raise RuntimeError("not every gate-site has a circuit location")
        return ordered_qubits, ordered_edges, one_locations, two_locations

    @property
    def control_count(self) -> int:
        return len(self.inventory)

    @property
    def control_limits(self) -> np.ndarray:
        """Symmetric public action limits that are safe for every possible optimum."""
        return self._control_limits

    def normalized_control_limits(self, native_scale: np.ndarray | None = None) -> np.ndarray:
        scale = (np.ones(self.control_count, dtype=float) if native_scale is None
                 else np.asarray(native_scale, dtype=float))
        if scale.shape != (self.control_count,) or np.any(scale <= 0) or not np.all(np.isfinite(scale)):
            raise ValueError("native scale must be a positive 41-coordinate vector")
        native_absolute_limit = self._maximum_mismatch - scale
        normalized_limit = native_absolute_limit / scale
        if np.any(normalized_limit <= 1.0):
            raise ValueError(
                "empirical normalization leaves no safe action domain containing the full sinusoidal optimum")
        return normalized_limit

    def apply_control_transform(self, latent_controls: np.ndarray,
                                *, native_scale: np.ndarray | None = None) -> np.ndarray:
        """Map Gaussian latent actions into the plant domain without clipping.

        The scaled-tanh map is fixed, one-to-one, and independent of the hidden
        optimum.  Its Jacobian therefore cancels from PPO ratios evaluated for the
        same sampled action, so the Gaussian scores remain those of latent space.
        """
        latent = np.asarray(latent_controls, dtype=float)
        if latent.shape[-1:] != (self.control_count,) or not np.all(np.isfinite(latent)):
            raise ValueError("latent Figure 5a controls must end in 41 finite coordinates")
        limits = self.normalized_control_limits(native_scale)
        return limits * np.tanh(latent / limits)

    def latent_controls_for(self, applied_controls: np.ndarray,
                            *, native_scale: np.ndarray | None = None) -> np.ndarray:
        """Invert the action transform for feasible deterministic plant controls."""
        applied = np.asarray(applied_controls, dtype=float)
        if applied.shape[-1:] != (self.control_count,) or not np.all(np.isfinite(applied)):
            raise ValueError("applied Figure 5a controls must end in 41 finite coordinates")
        limits = self.normalized_control_limits(native_scale)
        ratio = applied / limits
        if np.any(np.abs(ratio) >= 1.0):
            raise ValueError("applied controls must lie strictly inside the frozen action domain")
        return limits * np.arctanh(ratio)

    @staticmethod
    def optimum(epoch: int, frequency: float) -> np.ndarray:
        value = math.sin(2.0 * math.pi * float(frequency) * int(epoch))
        return np.full(41, value, dtype=float)

    def probabilities(self, controls: np.ndarray, epoch: int, frequency: float,
                      *, target_controls: np.ndarray | None = None) -> np.ndarray:
        policy = np.asarray(controls, dtype=float)
        if policy.shape != (41,) or not np.all(np.isfinite(policy)):
            raise ValueError("Figure 5a policy must contain 41 finite coordinates")
        optimum = (self.optimum(epoch, frequency) if target_controls is None
                   else np.asarray(target_controls, dtype=float))
        if optimum.shape != (41,) or not np.all(np.isfinite(optimum)):
            raise ValueError("Figure 5a target must contain 41 finite native coordinates")
        irreducible = np.asarray([item.irreducible_error for item in self.inventory])
        omega = np.asarray([item.omega_sensitivity for item in self.inventory])
        probabilities = irreducible + omega * (policy - optimum) ** 2
        if np.any(probabilities < 0) or np.any(probabilities >= self.maximum_probability):
            raise ValueError("gate depolarization probability left the frozen physical range")
        return probabilities

    def _circuit_from_probabilities(self, probabilities: np.ndarray):
        values = np.asarray(probabilities, dtype=float)
        if values.shape != (41,):
            raise ValueError("probability vector must align with 41 parameters")
        one = {item.qubits[0]: values[index] for index, item in enumerate(self._inventory[:17])}
        two = {item.qubits: values[index + 17] for index, item in enumerate(self._inventory[17:])}
        circuit = self._stim.Circuit()
        for instruction in self._reference:
            targets = [int(target.value) for target in instruction.targets_copy() if target.is_qubit_target]
            if instruction.name in {"M", "MX", "MR", "MRX"}:
                for q in targets:
                    if one[q] > 0:
                        circuit.append("DEPOLARIZE1", [q], float(one[q]))
            circuit.append(instruction)
            if instruction.name in {"R", "RX", "H"}:
                for q in targets:
                    if one[q] > 0:
                        circuit.append("DEPOLARIZE1", [q], float(one[q]))
            elif instruction.name == "CX":
                for offset in range(0, len(targets), 2):
                    edge = tuple(sorted((targets[offset], targets[offset + 1])))
                    if two[edge] > 0:
                        circuit.append("DEPOLARIZE2", targets[offset:offset + 2], float(two[edge]))
        return circuit

    def _dem_detectors(self, parameter: int) -> set[int]:
        values = np.zeros(41)
        values[parameter] = 1e-3
        dem = self._circuit_from_probabilities(values).detector_error_model(
            decompose_errors=False, approximate_disjoint_errors=True, flatten_loops=True)
        detectors: set[int] = set()
        for instruction in dem.flattened():
            if instruction.type != "error":
                continue
            for target in instruction.targets_copy():
                if target.is_relative_detector_id():
                    detectors.add(int(target.val))
        return detectors

    def _reduce_raw_counts(self, raw_counts: np.ndarray) -> np.ndarray:
        values = np.asarray(raw_counts, dtype=np.int64)
        if values.shape != (self.raw_detector_count,):
            raise ValueError("raw detector counts do not match the Stim circuit")
        return np.asarray([
            float(values[list(group)].sum()) / len(group)
            for group in self.reward_component_raw_detectors
        ], dtype=float)

    def _raw_detector_marginals(self, controls: np.ndarray, *, epoch: int, frequency: float,
                                target_controls: np.ndarray | None = None) -> np.ndarray:
        """Exact detector marginals of the actual Stim detector error model."""
        circuit = self._circuit_from_probabilities(self.probabilities(
            controls, epoch, frequency, target_controls=target_controls))
        dem = circuit.detector_error_model(
            decompose_errors=False, approximate_disjoint_errors=True, flatten_loops=True).flattened()
        parity_complement = np.ones(self.raw_detector_count, dtype=float)
        for instruction in dem:
            if instruction.type != "error":
                continue
            probability = float(instruction.args_copy()[0])
            affected: set[int] = set()
            for target in instruction.targets_copy():
                if not target.is_relative_detector_id():
                    continue
                detector = int(target.val)
                if detector in affected:
                    affected.remove(detector)
                else:
                    affected.add(detector)
            for detector in affected:
                parity_complement[detector] *= 1.0 - 2.0 * probability
        return 0.5 * (1.0 - parity_complement)

    def expected_reward_rates(self, controls: np.ndarray, *, epoch: int, frequency: float,
                              target_controls: np.ndarray | None = None) -> np.ndarray:
        raw = self._raw_detector_marginals(
            controls, epoch=epoch, frequency=frequency, target_controls=target_controls)
        return np.asarray([
            float(np.mean(raw[list(group)])) for group in self.reward_component_raw_detectors
        ], dtype=float)

    def expected_global_edr(self, controls: np.ndarray, *, epoch: int, frequency: float,
                            target_controls: np.ndarray | None = None) -> float:
        """Mean EDR over raw detector opportunities, as in Figure S3."""
        return float(np.mean(self._raw_detector_marginals(
            controls, epoch=epoch, frequency=frequency, target_controls=target_controls)))

    def sample_detector_observation(self, controls: np.ndarray, *, epoch: int, frequency: float,
                                    qec_cycles: int, seed: int,
                                    target_controls: np.ndarray | None = None) -> DetectorObservation:
        if qec_cycles <= 0 or qec_cycles % self.rounds:
            raise ValueError("QEC cycles must be positive and divisible by circuit rounds")
        circuit = self._circuit_from_probabilities(self.probabilities(
            controls, epoch, frequency, target_controls=target_controls))
        sampler = circuit.compile_detector_sampler(seed=int(seed))
        shots = qec_cycles // self.rounds
        sample = sampler.sample(shots=shots, bit_packed=False)
        if sample.shape != (shots, self.raw_detector_count):
            raise RuntimeError("Stim raw-detector shape changed")
        raw_counts = np.count_nonzero(sample, axis=0).astype(np.int64)
        return DetectorObservation(raw_counts, self._reduce_raw_counts(raw_counts), shots)

    def sample_detector_counts(self, controls: np.ndarray, *, epoch: int, frequency: float,
                               qec_cycles: int, seed: int,
                               target_controls: np.ndarray | None = None) -> np.ndarray:
        """Return count-equivalents for the reduced source reward vector."""
        return self.sample_detector_observation(
            controls, epoch=epoch, frequency=frequency, qec_cycles=qec_cycles,
            seed=seed, target_controls=target_controls).reward_component_counts

    def stream_seed(self, base_seed: int, stream: str, epoch: int, candidate: int) -> int:
        return _derived_seed(base_seed, stream, epoch, candidate, self.plant_hash)

    def inventory_rows(self) -> list[dict[str, Any]]:
        return [asdict(item) for item in self.inventory]
