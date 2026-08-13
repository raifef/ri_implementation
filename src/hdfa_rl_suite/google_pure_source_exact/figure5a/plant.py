"""Distance-3, 41-gate-parameter Stim plant from Supplement VI.A."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from itertools import product
import math
from typing import Any

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
class PauliChannelOccurrence:
    """One independent physical depolarizing-channel occurrence."""

    occurrence_id: str
    parameter_index: int
    instruction_index: int
    timing: str
    qubits: tuple[int, ...]


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

    STIM_DEPOLARIZE1_PROBABILITY_CEILING = 3.0 / 4.0
    STIM_DEPOLARIZE2_PROBABILITY_CEILING = 15.0 / 16.0
    ONE_QUBIT_INJECTION_MAPPINGS = (
        "per_qubit_operation_aggregate",
        "one_location_per_cycle",
    )
    EXACT_MARGINAL_EVALUATOR_VERSION = "channel-wise-pauli-parity.v1"
    _ONE_QUBIT_PAULI_BRANCHES = tuple((pauli,) for pauli in "XYZ")
    _TWO_QUBIT_PAULI_BRANCHES = tuple(
        branch for branch in product("IXYZ", repeat=2) if branch != ("I", "I"))
    _FAULT_SIGNATURE_CACHE: dict[
        str, tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray, str]] = {}

    def __init__(self, *, rounds: int, basis: str, ensemble_seed: int,
                 one_qubit_irreducible: tuple[float, float],
                 two_qubit_irreducible: tuple[float, float],
                 one_qubit_omega: tuple[float, float],
                 two_qubit_omega: tuple[float, float],
                 irreducible_global_scale: float = 1.0,
                 one_qubit_omega_global_scale: float = 1.0,
                 two_qubit_omega_global_scale: float = 1.0,
                 omega_coordinate_scales: tuple[float, ...] | None = None,
                 one_qubit_injection_mapping: str =
                 "per_qubit_operation_aggregate") -> None:
        try:
            import stim
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("Stim is required") from error
        scales = np.asarray([
            irreducible_global_scale,
            one_qubit_omega_global_scale,
            two_qubit_omega_global_scale,
        ], dtype=float)
        if (rounds <= 0 or basis not in {"x", "z"}
                or one_qubit_injection_mapping not in self.ONE_QUBIT_INJECTION_MAPPINGS
                or not np.all(np.isfinite(scales)) or np.any(scales <= 0)):
            raise ValueError("invalid distance-3 memory circuit")
        self._stim = stim
        self.rounds, self.basis = int(rounds), str(basis)
        self.one_qubit_injection_mapping = str(one_qubit_injection_mapping)
        self.irreducible_global_scale = float(irreducible_global_scale)
        self.one_qubit_omega_global_scale = float(one_qubit_omega_global_scale)
        self.two_qubit_omega_global_scale = float(two_qubit_omega_global_scale)
        coordinate_scales = (np.ones(41, dtype=float) if omega_coordinate_scales is None
                             else np.asarray(omega_coordinate_scales, dtype=float))
        if (coordinate_scales.shape != (41,) or not np.all(np.isfinite(coordinate_scales))
                or np.any(coordinate_scales <= 0)):
            raise ValueError("omega coordinate scales must contain 41 positive finite values")
        coordinate_scales = coordinate_scales.copy()
        coordinate_scales.setflags(write=False)
        self.omega_coordinate_scales = coordinate_scales
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
                float(rng.uniform(*one_qubit_irreducible) * self.irreducible_global_scale),
                float(rng.uniform(*one_qubit_omega) * self.one_qubit_omega_global_scale *
                      self.omega_coordinate_scales[len(inventory)])))
        for left, right in edges:
            inventory.append(GateParameter(
                f"tq-q{left}-q{right}", "two_qubit", (left, right), tuple(two_locations[(left, right)]), (),
                float(rng.uniform(*two_qubit_irreducible) * self.irreducible_global_scale),
                float(rng.uniform(*two_qubit_omega) * self.two_qubit_omega_global_scale *
                      self.omega_coordinate_scales[len(inventory)])))
        self._inventory = tuple(inventory)
        self.channel_occurrences = self._enumerate_physical_channel_occurrences()
        self.fault_signatures, self._channel_detector_branch_flip_fractions, \
            self._channel_parameter_indices, self.fault_signature_hash = \
            self._build_exact_fault_signatures()
        raw_parameter_detectors = [set() for _ in inventory]
        for occurrence, signatures in zip(
                self.channel_occurrences, self.fault_signatures, strict=True):
            raw_parameter_detectors[occurrence.parameter_index].update(
                np.flatnonzero(np.any(signatures, axis=0)).astype(int).tolist())
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
        self.probability_ceilings = np.asarray([
            self.STIM_DEPOLARIZE1_PROBABILITY_CEILING
            if item.gate_type == "single_qubit"
            else self.STIM_DEPOLARIZE2_PROBABILITY_CEILING
            for item in self.inventory
        ], dtype=float)
        self.probability_ceilings.setflags(write=False)
        if (not np.all(np.isfinite(irreducible)) or np.any(irreducible < 0)
                or np.any(irreducible >= self.probability_ceilings)
                or not np.all(np.isfinite(omega)) or np.any(omega <= 0)):
            raise ValueError("invalid Figure 5a irreducible errors or quadratic sensitivities")
        if not self.mask.any(axis=0).all() or not self.mask.any(axis=1).all():
            raise RuntimeError("Stim-derived detector mask has an empty parameter or detector")
        self.parameter_ids = tuple(item.parameter_id for item in self.inventory)
        self.plant_hash = canonical_hash({
            "stim_version": getattr(stim, "__version__", "unknown"), "distance": 3,
            "rounds": rounds, "basis": basis,
            "stim_probability_ceilings": self.probability_ceilings.tolist(),
            "inventory": [asdict(item) for item in self.inventory], "mask": self.mask.astype(int).tolist(),
            "raw_detector_count": self.raw_detector_count,
            "reward_representation": "time_translation_equivalence_class_mean_edr",
            "reward_component_raw_detectors": [list(group) for group in self.reward_component_raw_detectors],
            "reward_component_keys": list(self.reward_component_keys),
            "ensemble_seed": int(ensemble_seed),
            "irreducible_global_scale": self.irreducible_global_scale,
            "one_qubit_omega_global_scale": self.one_qubit_omega_global_scale,
            "two_qubit_omega_global_scale": self.two_qubit_omega_global_scale,
            "omega_coordinate_scales": self.omega_coordinate_scales.tolist(),
            "one_qubit_injection_mapping": self.one_qubit_injection_mapping,
            "physical_channel_occurrences": [
                asdict(occurrence) for occurrence in self.channel_occurrences],
            "exact_marginal_evaluator_version": self.EXACT_MARGINAL_EVALUATOR_VERSION,
            "fault_signature_hash": self.fault_signature_hash,
            "canonical_action_execution": "identity_applied_gaussian",
            "constructor_requires_bounded_action_domain": False,
        })

    def _time_translation_equivalence_classes(
        self, raw_mask: np.ndarray,
    ) -> tuple[tuple[tuple[int, ...], ...], tuple[str, ...]]:
        """Combine only raw detectors with identical translated semantics.

        Stim's generated memory circuits encode detector coordinates as
        ``(space..., time)``.  Equal spatial coordinates alone are insufficient
        at boundaries and the first repeated layer, whose detecting regions
        differ.  The initial key combines spatial coordinate with the exact
        control-dependency row; the first member of every repeated key is then
        retained separately from its steady translated copies.  The resulting
        reward size is independent of the number of repeated steady rounds.
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
        groups_list: list[tuple[int, ...]] = []
        keys_list: list[str] = []
        for key, values in grouped.items():
            # The first repeated detector layer is a circuit-start transient:
            # it has the same spatial coordinate and binary control support as
            # later layers, but not the same exact detector marginal.  Keep it
            # separate and combine only the steady translated copies.  This
            # produces a round-invariant representation without over-merging.
            partitions = ((values[0],), tuple(values[1:])) if len(values) > 1 else (tuple(values),)
            for role, partition in enumerate(partitions):
                if not partition:
                    continue
                groups_list.append(partition)
                keys_list.append(canonical_hash({
                    "space": key[0], "controls": key[1],
                    "temporal_role": "first_repeated_layer" if role == 0 and len(values) > 1
                    else "steady_time_translation_class",
                }))
        groups = tuple(groups_list)
        keys = tuple(keys_list)
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
        if self.one_qubit_injection_mapping == "one_location_per_cycle":
            one_locations = {
                q: [f"cycle:{cycle}:synthetic_1q_layer" for cycle in range(self.rounds)]
                for q in ordered_qubits
            }
        if set(one_locations) != set(ordered_qubits) or set(two_locations) != set(ordered_edges):
            raise RuntimeError("not every gate-site has a circuit location")
        return ordered_qubits, ordered_edges, one_locations, two_locations

    @property
    def control_count(self) -> int:
        return len(self._inventory)

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
        if np.any(probabilities < 0) or np.any(probabilities >= self.probability_ceilings):
            raise ValueError("gate depolarization probability left the frozen physical range")
        return probabilities

    def _circuit_from_probabilities(self, probabilities: np.ndarray):
        values = np.asarray(probabilities, dtype=float)
        if values.shape != (41,):
            raise ValueError("probability vector must align with 41 parameters")
        one = {item.qubits[0]: values[index] for index, item in enumerate(self._inventory[:17])}
        two = {item.qubits: values[index + 17] for index, item in enumerate(self._inventory[17:])}
        circuit = self._stim.Circuit()
        awaiting_cycle_start = True
        for instruction in self._reference:
            targets = [int(target.value) for target in instruction.targets_copy() if target.is_qubit_target]
            aggregate = self.one_qubit_injection_mapping == "per_qubit_operation_aggregate"
            if aggregate and instruction.name in {"M", "MX", "MR", "MRX"}:
                for q in targets:
                    if one[q] > 0:
                        circuit.append("DEPOLARIZE1", [q], float(one[q]))
            circuit.append(instruction)
            if (self.one_qubit_injection_mapping == "one_location_per_cycle"
                    and instruction.name == "TICK" and awaiting_cycle_start):
                for q, probability in one.items():
                    if probability > 0:
                        circuit.append("DEPOLARIZE1", [q], float(probability))
                awaiting_cycle_start = False
            if instruction.name in {"MR", "MRX"}:
                awaiting_cycle_start = True
            if aggregate and instruction.name in {"R", "RX", "H"}:
                for q in targets:
                    if one[q] > 0:
                        circuit.append("DEPOLARIZE1", [q], float(one[q]))
            elif instruction.name == "CX":
                for offset in range(0, len(targets), 2):
                    edge = tuple(sorted((targets[offset], targets[offset + 1])))
                    if two[edge] > 0:
                        circuit.append("DEPOLARIZE2", targets[offset:offset + 2], float(two[edge]))
        return circuit

    def _enumerate_physical_channel_occurrences(
        self,
    ) -> tuple[PauliChannelOccurrence, ...]:
        """Mirror every channel inserted by ``_circuit_from_probabilities``."""
        one_parameter = {
            item.qubits[0]: index for index, item in enumerate(self._inventory[:17])}
        two_parameter = {
            item.qubits: index + 17 for index, item in enumerate(self._inventory[17:])}
        occurrences: list[PauliChannelOccurrence] = []
        awaiting_cycle_start = True
        cycle = 0

        def add(parameter: int, instruction: int, timing: str,
                qubits: tuple[int, ...], label: str) -> None:
            occurrences.append(PauliChannelOccurrence(
                occurrence_id=f"{label}:{timing}:instruction:{instruction}",
                parameter_index=int(parameter), instruction_index=int(instruction),
                timing=timing, qubits=qubits))

        for instruction_index, instruction in enumerate(self._reference):
            targets = tuple(
                int(target.value) for target in instruction.targets_copy()
                if target.is_qubit_target)
            aggregate = self.one_qubit_injection_mapping == \
                "per_qubit_operation_aggregate"
            if aggregate and instruction.name in {"M", "MX", "MR", "MRX"}:
                for qubit in targets:
                    add(one_parameter[qubit], instruction_index, "before", (qubit,),
                        f"q{qubit}:{instruction.name}")
            if (self.one_qubit_injection_mapping == "one_location_per_cycle"
                    and instruction.name == "TICK" and awaiting_cycle_start):
                for qubit, parameter in one_parameter.items():
                    add(parameter, instruction_index, "after", (qubit,),
                        f"q{qubit}:cycle:{cycle}:synthetic_1q_layer")
                cycle += 1
                awaiting_cycle_start = False
            if instruction.name in {"MR", "MRX"}:
                awaiting_cycle_start = True
            if aggregate and instruction.name in {"R", "RX", "H"}:
                for qubit in targets:
                    add(one_parameter[qubit], instruction_index, "after", (qubit,),
                        f"q{qubit}:{instruction.name}")
            elif instruction.name == "CX":
                for offset in range(0, len(targets), 2):
                    ordered = targets[offset:offset + 2]
                    edge = tuple(sorted(ordered))
                    add(two_parameter[edge], instruction_index, "after", ordered,
                        f"q{edge[0]}-q{edge[1]}:CX")
        if (self.one_qubit_injection_mapping == "one_location_per_cycle"
                and cycle != self.rounds):
            raise RuntimeError(
                f"expected {self.rounds} synthetic 1Q layers, found {cycle}")
        actual_counts = np.bincount(
            [item.parameter_index for item in occurrences], minlength=self.control_count)
        expected_counts = np.asarray(
            [len(item.circuit_locations) for item in self._inventory], dtype=int)
        if not np.array_equal(actual_counts, expected_counts):
            raise RuntimeError("physical channel occurrences do not match the inventory")
        return tuple(occurrences)

    @classmethod
    def _pauli_branches(cls, arity: int) -> tuple[tuple[str, ...], ...]:
        if arity == 1:
            return cls._ONE_QUBIT_PAULI_BRANCHES
        if arity == 2:
            return cls._TWO_QUBIT_PAULI_BRANCHES
        raise ValueError("only one- and two-qubit Pauli channels are supported")

    def _build_exact_fault_signatures(
        self,
    ) -> tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray, str]:
        """Propagate every mutually exclusive Pauli branch in one bit-packed batch."""
        cache_key = canonical_hash({
            "evaluator": self.EXACT_MARGINAL_EVALUATOR_VERSION,
            "stim_version": getattr(self._stim, "__version__", "unknown"),
            "reference_circuit": str(self._reference),
            "occurrences": [asdict(item) for item in self.channel_occurrences],
        })
        cached = self._FAULT_SIGNATURE_CACHE.get(cache_key)
        if cached is not None:
            return cached
        branch_counts = [
            len(self._pauli_branches(len(occurrence.qubits)))
            for occurrence in self.channel_occurrences]
        offsets = np.cumsum([0, *branch_counts])
        simulator = self._stim.FlipSimulator(
            batch_size=int(offsets[-1]), num_qubits=int(self._reference.num_qubits),
            disable_stabilizer_randomization=True)
        before: dict[int, list[int]] = {}
        after: dict[int, list[int]] = {}
        for occurrence_index, occurrence in enumerate(self.channel_occurrences):
            destination = before if occurrence.timing == "before" else after
            destination.setdefault(occurrence.instruction_index, []).append(occurrence_index)

        def inject(occurrence_indices: list[int]) -> None:
            for occurrence_index in occurrence_indices:
                occurrence = self.channel_occurrences[occurrence_index]
                branches = self._pauli_branches(len(occurrence.qubits))
                for branch_index, branch in enumerate(branches):
                    instance = int(offsets[occurrence_index] + branch_index)
                    for qubit, pauli in zip(occurrence.qubits, branch, strict=True):
                        if pauli != "I":
                            simulator.set_pauli_flip(
                                pauli, qubit_index=qubit, instance_index=instance)

        for instruction_index, instruction in enumerate(self._reference):
            inject(before.get(instruction_index, []))
            simulator.do(instruction)
            inject(after.get(instruction_index, []))
        if simulator.num_detectors != self.raw_detector_count:
            raise RuntimeError("fault-signature detector count changed")
        all_signatures = np.asarray(simulator.get_detector_flips(), dtype=bool).T
        signatures: list[np.ndarray] = []
        fractions: list[np.ndarray] = []
        digest = sha256(self.EXACT_MARGINAL_EVALUATOR_VERSION.encode("utf-8"))
        digest.update(canonical_hash([
            asdict(occurrence) for occurrence in self.channel_occurrences]).encode("utf-8"))
        for occurrence_index in range(len(self.channel_occurrences)):
            value = all_signatures[offsets[occurrence_index]:offsets[occurrence_index + 1]].copy()
            value.setflags(write=False)
            signatures.append(value)
            fractions.append(np.mean(value, axis=0))
            digest.update(np.packbits(value, axis=None, bitorder="little").tobytes())
        fraction_array = np.asarray(fractions, dtype=float)
        parameter_indices = np.asarray(
            [item.parameter_index for item in self.channel_occurrences], dtype=int)
        fraction_array.setflags(write=False)
        parameter_indices.setflags(write=False)
        result = tuple(signatures), fraction_array, parameter_indices, digest.hexdigest()
        self._FAULT_SIGNATURE_CACHE[cache_key] = result
        return result

    def _reduce_raw_counts(self, raw_counts: np.ndarray) -> np.ndarray:
        values = np.asarray(raw_counts, dtype=np.int64)
        if values.shape != (self.raw_detector_count,):
            raise ValueError("raw detector counts do not match the Stim circuit")
        return np.asarray([
            float(values[list(group)].sum()) / len(group)
            for group in self.reward_component_raw_detectors
        ], dtype=float)

    def _validate_probability_vector(self, probabilities: np.ndarray) -> np.ndarray:
        values = np.asarray(probabilities, dtype=float)
        if (values.shape != (self.control_count,) or not np.all(np.isfinite(values))
                or np.any(values < 0) or np.any(values >= self.probability_ceilings)):
            raise ValueError("invalid physical probability vector")
        return values

    def exact_raw_detector_marginals(
        self, controls: np.ndarray, *, epoch: int, frequency: float,
        target_controls: np.ndarray | None = None,
    ) -> np.ndarray:
        """Exact marginals of independent physical, mutually exclusive Pauli channels."""
        return self._exact_raw_detector_marginals_from_probabilities(self.probabilities(
            controls, epoch, frequency, target_controls=target_controls))

    def _exact_raw_detector_marginals_from_probabilities(
        self, probabilities: np.ndarray,
    ) -> np.ndarray:
        values = self._validate_probability_vector(probabilities)
        occurrence_probabilities = values[self._channel_parameter_indices]
        channel_characteristics = 1.0 - 2.0 * occurrence_probabilities[:, None] * \
            self._channel_detector_branch_flip_fractions
        result = 0.5 * (1.0 - np.prod(channel_characteristics, axis=0))
        result.setflags(write=False)
        return result

    def approximate_dem_raw_detector_marginals(
        self, controls: np.ndarray, *, epoch: int, frequency: float,
        target_controls: np.ndarray | None = None,
    ) -> np.ndarray:
        """Legacy DEM-disjoint approximation retained only for explicit audits."""
        return self._approximate_dem_raw_detector_marginals_from_probabilities(
            self.probabilities(
                controls, epoch, frequency, target_controls=target_controls))

    def _approximate_dem_raw_detector_marginals_from_probabilities(
        self, probabilities: np.ndarray,
    ) -> np.ndarray:
        values = self._validate_probability_vector(probabilities)
        circuit = self._circuit_from_probabilities(values)
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
        result = 0.5 * (1.0 - parity_complement)
        result.setflags(write=False)
        return result

    # Compatibility aliases are exact and never route canonical code through
    # the explicitly named approximate DEM-disjoint evaluator.
    raw_detector_marginals = exact_raw_detector_marginals
    _raw_detector_marginals = raw_detector_marginals

    def exact_reward_rates(self, controls: np.ndarray, *, epoch: int, frequency: float,
                           target_controls: np.ndarray | None = None) -> np.ndarray:
        raw = self.exact_raw_detector_marginals(
            controls, epoch=epoch, frequency=frequency,
            target_controls=target_controls)
        return np.asarray([
            float(np.mean(raw[list(group)]))
            for group in self.reward_component_raw_detectors
        ], dtype=float)

    def exact_global_edr(self, controls: np.ndarray, *, epoch: int, frequency: float,
                         target_controls: np.ndarray | None = None) -> float:
        """Exact mean EDR over raw detector opportunities, as in Figure S3."""
        return float(np.mean(self.exact_raw_detector_marginals(
            controls, epoch=epoch, frequency=frequency,
            target_controls=target_controls)))

    def approximate_dem_reward_rates(
        self, controls: np.ndarray, *, epoch: int, frequency: float,
        target_controls: np.ndarray | None = None,
    ) -> np.ndarray:
        raw = self.approximate_dem_raw_detector_marginals(
            controls, epoch=epoch, frequency=frequency,
            target_controls=target_controls)
        return np.asarray([
            float(np.mean(raw[list(group)]))
            for group in self.reward_component_raw_detectors
        ], dtype=float)

    def approximate_dem_global_edr(
        self, controls: np.ndarray, *, epoch: int, frequency: float,
        target_controls: np.ndarray | None = None,
    ) -> float:
        return float(np.mean(self.approximate_dem_raw_detector_marginals(
            controls, epoch=epoch, frequency=frequency,
            target_controls=target_controls)))

    def expected_reward_rates(self, controls: np.ndarray, *, epoch: int, frequency: float,
                              target_controls: np.ndarray | None = None) -> np.ndarray:
        return self.exact_reward_rates(
            controls, epoch=epoch, frequency=frequency,
            target_controls=target_controls)

    def expected_global_edr(self, controls: np.ndarray, *, epoch: int, frequency: float,
                            target_controls: np.ndarray | None = None) -> float:
        return self.exact_global_edr(
            controls, epoch=epoch, frequency=frequency,
            target_controls=target_controls)

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
