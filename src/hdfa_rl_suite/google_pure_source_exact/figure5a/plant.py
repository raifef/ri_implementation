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
        self.detector_count = int(self._reference.num_detectors)
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
        masks = [self._dem_detectors(index) for index in range(len(inventory))]
        self.inventory = tuple(replace(item, detectors_influenced=tuple(sorted(masks[index])))
                               for index, item in enumerate(inventory))
        irreducible = np.asarray([item.irreducible_error for item in self.inventory])
        omega = np.asarray([item.omega_sensitivity for item in self.inventory])
        probability_ceiling = self.maximum_probability * (
            1.0 - self.action_probability_margin_fraction)
        self._control_limits = np.sqrt((probability_ceiling - irreducible) / omega) - 1.0
        if (not np.all(np.isfinite(self._control_limits)) or
                np.any(self._control_limits <= 1.0)):
            raise ValueError(
                "frozen Figure 5a plant has no symmetric action domain containing the full optimum range")
        self._control_limits.setflags(write=False)
        self.mask = np.zeros((self.detector_count, len(self.inventory)), dtype=bool)
        for parameter, detectors in enumerate(masks):
            self.mask[list(detectors), parameter] = True
        if not self.mask.any(axis=0).all() or not self.mask.any(axis=1).all():
            raise RuntimeError("Stim-derived detector mask has an empty parameter or detector")
        self.parameter_ids = tuple(item.parameter_id for item in self.inventory)
        self.plant_hash = canonical_hash({
            "stim_version": getattr(stim, "__version__", "unknown"), "distance": 3,
            "rounds": rounds, "basis": basis, "maximum_probability": maximum_probability,
            "action_probability_margin_fraction": self.action_probability_margin_fraction,
            "inventory": [asdict(item) for item in self.inventory], "mask": self.mask.astype(int).tolist(),
            "ensemble_seed": int(ensemble_seed),
            "action_execution": "plant_derived_per_coordinate_scaled_tanh",
            "control_limits": self._control_limits.tolist(),
        })

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

    def apply_control_transform(self, latent_controls: np.ndarray) -> np.ndarray:
        """Map Gaussian latent actions into the plant domain without clipping.

        The scaled-tanh map is fixed, one-to-one, and independent of the hidden
        optimum.  Its Jacobian therefore cancels from PPO ratios evaluated for the
        same sampled action, so the Gaussian scores remain those of latent space.
        """
        latent = np.asarray(latent_controls, dtype=float)
        if latent.shape[-1:] != (self.control_count,) or not np.all(np.isfinite(latent)):
            raise ValueError("latent Figure 5a controls must end in 41 finite coordinates")
        return self._control_limits * np.tanh(latent / self._control_limits)

    def latent_controls_for(self, applied_controls: np.ndarray) -> np.ndarray:
        """Invert the action transform for feasible deterministic plant controls."""
        applied = np.asarray(applied_controls, dtype=float)
        if applied.shape[-1:] != (self.control_count,) or not np.all(np.isfinite(applied)):
            raise ValueError("applied Figure 5a controls must end in 41 finite coordinates")
        ratio = applied / self._control_limits
        if np.any(np.abs(ratio) >= 1.0):
            raise ValueError("applied controls must lie strictly inside the frozen action domain")
        return self._control_limits * np.arctanh(ratio)

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

    def sample_detector_counts(self, controls: np.ndarray, *, epoch: int, frequency: float,
                               qec_cycles: int, seed: int,
                               target_controls: np.ndarray | None = None) -> np.ndarray:
        if qec_cycles <= 0 or qec_cycles % self.rounds:
            raise ValueError("QEC cycles must be positive and divisible by circuit rounds")
        circuit = self._circuit_from_probabilities(self.probabilities(
            controls, epoch, frequency, target_controls=target_controls))
        sampler = circuit.compile_detector_sampler(seed=int(seed))
        sample = sampler.sample(shots=qec_cycles // self.rounds, bit_packed=False)
        if sample.shape != (qec_cycles // self.rounds, self.detector_count):
            raise RuntimeError("Stim detector shape changed")
        return np.count_nonzero(sample, axis=0).astype(np.int64)

    def stream_seed(self, base_seed: int, stream: str, epoch: int, candidate: int) -> int:
        return _derived_seed(base_seed, stream, epoch, candidate, self.plant_hash)

    def inventory_rows(self) -> list[dict[str, Any]]:
        return [asdict(item) for item in self.inventory]
