"""Deterministic graph-local simulator for structured and unknown QEC drift.

The ordinary controller interface returns measurements, timestamps, policy metadata and
observable logical sentinels only.  Hidden physical state is available through the
explicitly named :class:`OracleEvaluationView`, which benchmark/oracle code must request.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import copy
from enum import Enum
import math
import random
from typing import Mapping, Sequence

import numpy as np

from hdfa_rl_suite.common import (
    PolicyCertificate,
    PolicyLifecycleState,
    PolicyTransaction,
    PolicyTransactionLedger,
    deterministic_hash,
)
from hdfa_rl_suite.stage0.schema import (
    ControlBound,
    DetectorDefinition,
    DeviceTopology,
    HardwareLimits,
    TargetQECCircuit,
    stable_hash,
)
from hdfa_rl_suite.stage1.schema import CircuitContext, PolicyActivation, RawMeasurementRecord


SIMULATOR_VERSION = "scalable-qec-simulator.v6"


class DriftKind(str, Enum):
    CONSTANT = "constant"
    SINUSOID = "sinusoid"
    RANDOM_TELEGRAPH = "random_telegraph"
    SEMI_MARKOV_TELEGRAPH = "semi_markov_telegraph"
    ORNSTEIN_UHLENBECK = "ornstein_uhlenbeck"
    RANDOM_WALK = "random_walk"
    STEP = "step"
    UNKNOWN_HEAVY_TAILED = "unknown_heavy_tailed"


@dataclass(frozen=True)
class LatentProcessSpec:
    process_id: str
    kind: DriftKind
    loadings: Mapping[str, float]
    amplitude: float = 0.25
    period_s: float = 16.0
    phase_rad: float = 0.0
    rate_hz: float = 0.15
    mean_dwell_s: float = 5.0
    ou_kappa: float = 0.2
    diffusion: float = 0.03
    step_time_s: float | None = None
    parent_process_id: str | None = None


@dataclass(frozen=True)
class SimulatorConfig:
    qubit_count: int = 9
    code_distance: int = 3
    cycle_period_s: float = 1e-3
    controller_latency_s: float = 2e-3
    base_detector_probability: float = 0.012
    response_curvature: float = 2.8
    cross_coupling_strength: float = 0.0
    maximum_detector_probability: float = 0.49
    validated_mismatch_radius: float = 0.35
    correlation_probability: float = 0.002
    measurement_dropout_probability: float = 0.0
    logical_scale: float = 0.15
    disturbance_resolution_s: float = 0.01
    disturbances_enabled_at_start: bool = True
    stationary_vectorized_acquisition: bool = True
    dynamic_vectorized_acquisition: bool = True
    seed: int = 0
    processes: tuple[LatentProcessSpec, ...] = ()

    def __post_init__(self) -> None:
        if self.qubit_count < 2:
            raise ValueError("qubit_count must be at least two")
        if self.cycle_period_s <= 0 or self.controller_latency_s < 0 or self.disturbance_resolution_s <= 0:
            raise ValueError("timing values must be physical")
        for probability in (self.base_detector_probability, self.correlation_probability, self.measurement_dropout_probability):
            if not 0 <= probability < 1:
                raise ValueError("probabilities must lie in [0, 1)")
        if self.response_curvature < 0 or self.cross_coupling_strength < 0:
            raise ValueError("response coefficients must be non-negative")
        if not self.base_detector_probability < self.maximum_detector_probability <= .5:
            raise ValueError("maximum detector probability must exceed the floor and be at most 0.5")
        if self.validated_mismatch_radius <= 0:
            raise ValueError("validated mismatch radius must be positive")


@dataclass(frozen=True)
class QECObservationBatch:
    records: tuple[RawMeasurementRecord, ...]
    policy_activation: PolicyActivation
    context: CircuitContext
    detector_events: int
    detector_exposures: int
    logical_failures: int
    cycles: int
    batch_id: str
    detector_counts: Mapping[str, tuple[int, int]] = field(default_factory=dict)
    physical_state_id: str = ""
    disturbance_state_id: str = ""
    simulator_state_hash: str = ""
    controller_state_hash: str = ""

    @property
    def detector_rate(self) -> float:
        return self.detector_events / self.detector_exposures if self.detector_exposures else math.nan


class OracleEvaluationView:
    """Capability deliberately named and typed for oracle/evaluation-only truth access."""

    def __init__(self, device: "ScalableQECDevice", purpose: str) -> None:
        if not (purpose.startswith("evaluation:") or purpose.startswith("oracle:")):
            raise PermissionError("hidden truth is restricted to evaluation/oracle purposes")
        self._device = device
        self.purpose = purpose

    def latent_state(self) -> Mapping[str, float]:
        return dict(self._device._latent_state)

    def optimum_policy(self) -> Mapping[str, float]:
        return dict(self._device._latent_state)

    def process_state(self) -> Mapping[str, float]:
        return dict(self._device._process_state)

    def disturbance_realization_id(self) -> str:
        return self._device.disturbance_realization_id

    def physical_diagnostic(self, policy: Mapping[str, float] | None = None) -> "PhysicalStateDiagnostic":
        """Return an evaluation-only snapshot of the explicit plant response chain."""
        return self._device._physical_diagnostic(policy)


@dataclass(frozen=True)
class PhysicalStateDiagnostic:
    """One immutable, evaluation-only view of the plant's causal response chain."""

    state_id: str
    disturbance_state_id: str
    timestamp_s: float
    policy_id: str
    policy_hash: str
    latent_optimum: Mapping[str, float]
    applied_control: Mapping[str, float]
    mismatch: Mapping[str, float]
    detector_probabilities: Mapping[str, float]
    detector_controllable_excess: Mapping[str, float]
    detector_cross_coupling: Mapping[str, float]
    expected_global_detector_rate: float
    expected_logical_failure_proxy: float
    irreducible_detector_floor: float
    validated_local_range: float
    saturated_detectors: tuple[str, ...]


@dataclass(frozen=True)
class CounterfactualStateFingerprint:
    """Stable pre-action state used to prove matched scientific counterfactuals."""

    physical_state_id: str
    disturbance_state_id: str
    disturbance_realization_id: str
    simulator_state_hash: str
    controller_state_hash: str
    process_rng_state_hash: str
    characterization_rng_state_hash: str
    detector_evaluator_config_hash: str
    policy_id: str
    policy_hash: str
    policy_controls: Mapping[str, float]
    timestamp_s: float


@dataclass(frozen=True)
class CharacterizationResult:
    estimates: Mapping[str, float]
    variances: Mapping[str, float]
    shots: int
    downtime_s: float
    timestamp_s: float


class ScalableQECDevice:
    """Sparse line-graph QEC device supporting hundreds of local control variables."""

    def __init__(self, config: SimulatorConfig = SimulatorConfig()) -> None:
        self.config = config
        # Physical dynamics, characterization noise, and circuit samples are separate
        # common-random-number streams.  Controller-dependent sampling can therefore not
        # alter the exogenous disturbance realization.
        self._process_rng = random.Random(config.seed ^ 0x4D595DF4D0F33173)
        self._characterization_rng = random.Random(config.seed ^ 0x14057B7EF767814F)
        self._time_s = 0.0
        self._sequence = 0
        self._batch_sequence = 0
        self.topology, self.limits, self.circuit, self.detector_control_graph = self._build_contracts(config)
        self.context = CircuitContext(self.circuit.circuit_hash, "memory", "Z", config.code_distance,
                                      "surface-memory", "active", "sim-decoder.v1")
        self._policy = {control: 0.0 for control in self.limits.controls}
        self._policy_activation = PolicyActivation("initial", stable_hash(self._policy), -1.0, -1.0, 0.0, dict(self._policy))
        self._pending_policy: tuple[dict[str, float], PolicyActivation] | None = None
        self._process_state = {process.process_id: 0.0 for process in self.processes}
        self._latent_state = {control: 0.0 for control in self.limits.controls}
        self._disturbance_tape = {process.process_id: [0.0] for process in self.processes}
        self._disturbance_dwell = {process.process_id: 0.0 for process in self.processes}
        self._oracle_access_log: list[tuple[float, str]] = []
        self._stationary_probability_cache: dict[str, np.ndarray] = {}
        self._measurement_channel_ids = tuple(
            f"m:{index}" for index in range(2 * len(self.circuit.detectors)))
        # Benchmarks may perform the dedicated Stage-0 calibration against a stationary
        # device and then arm a matched exogenous disturbance at a declared boundary.
        # The tape's time coordinate is relative to this epoch, so calibration duration
        # can never consume or shift the disturbance realization.
        self._disturbance_epoch_s: float | None = (0.0 if config.disturbances_enabled_at_start else None)
        initial_state_id = stable_hash({"time_s": self._time_s, "policy_hash": self._policy_activation.policy_hash,
                                        "disturbance": self.disturbance_realization_id})
        self._policy_ledger = PolicyTransactionLedger(
            "initial", self._policy, initial_state_id, self._time_s)
        initial_transaction = self._policy_ledger.confirmed
        self._policy_activation = replace(
            self._policy_activation,
            transaction_id=initial_transaction.transaction_id,
            reference_policy_id=initial_transaction.reference_policy_id,
            reference_policy_hash=initial_transaction.reference_policy_hash,
            created_from_state_id=initial_transaction.created_from_state_id,
            expected_activation_state_id=initial_transaction.expected_activation_state_id,
            supervisor_authorization=initial_transaction.supervisor_authorization,
            activation_acknowledgement=(
                initial_transaction.activation_acknowledgement.acknowledgement_id
                if initial_transaction.activation_acknowledgement else ""),
            lifecycle_state=PolicyLifecycleState.CONFIRMED,
        )

    @property
    def simulator_version(self) -> str:
        return SIMULATOR_VERSION

    @property
    def disturbance_realization_id(self) -> str:
        return stable_hash({
            "simulator": SIMULATOR_VERSION,
            "seed": self.config.seed,
            "resolution_s": self.config.disturbance_resolution_s,
            "time_basis": "relative-to-declared-disturbance-epoch",
            "processes": self.processes,
        })

    @property
    def disturbances_armed(self) -> bool:
        return self._disturbance_epoch_s is not None

    @property
    def disturbance_epoch_s(self) -> float | None:
        return self._disturbance_epoch_s

    @property
    def disturbance_elapsed_s(self) -> float | None:
        if self._disturbance_epoch_s is None:
            return None
        return max(0.0, self._time_s - self._disturbance_epoch_s)

    def arm_disturbances(self) -> float:
        """Start the configured exogenous tape at the current device time.

        This one-way boundary is intended for randomized experiments with a common
        stationary calibration phase.  It is deliberately not a controller action.
        """
        if self._disturbance_epoch_s is not None:
            raise RuntimeError("disturbances are already armed")
        self._disturbance_epoch_s = self._time_s
        self._process_state = {process.process_id: 0.0 for process in self.processes}
        self._latent_state = {control: 0.0 for control in self.limits.controls}
        return self._disturbance_epoch_s

    @property
    def processes(self) -> tuple[LatentProcessSpec, ...]:
        if self.config.processes:
            return self.config.processes
        local = tuple(LatentProcessSpec(f"ou:q{i}", DriftKind.ORNSTEIN_UHLENBECK,
                                       {f"drive:q{i}": 1.0}, diffusion=.025)
                      for i in range(self.config.qubit_count))
        common = LatentProcessSpec("common-sinusoid", DriftKind.SINUSOID,
                                   {f"drive:q{i}": 0.35 for i in range(self.config.qubit_count)},
                                   amplitude=.3, period_s=12.)
        return local + (common,)

    @staticmethod
    def _build_contracts(config: SimulatorConfig) -> tuple[DeviceTopology, HardwareLimits, TargetQECCircuit, dict[str, tuple[str, ...]]]:
        qubits = tuple(f"q{i}" for i in range(config.qubit_count))
        couplers = tuple((qubits[i], qubits[i + 1]) for i in range(len(qubits) - 1))
        controls = {f"drive:{qubit}": f"dac:{qubit}" for qubit in qubits}
        controls.update({f"coupling:{left}-{right}": f"flux:{left}-{right}" for left, right in couplers})
        topology = DeviceTopology(
            f"sim-line-{config.qubit_count}", qubits, couplers,
            {qubit: f"resonator:{qubit}" for qubit in qubits}, controls,
            sample_period_s=1e-9, controller_latency_s=config.controller_latency_s,
        )
        limits = HardwareLimits({control: ControlBound(-1., 1., .15, "normalized", .25) for control in controls},
                                max_thermal_duty=.65, max_leakage=.04)
        detectors: list[DetectorDefinition] = []
        graph: dict[str, tuple[str, ...]] = {}
        for index, qubit in enumerate(qubits):
            detector_id = f"d:{qubit}"
            affected = [f"drive:{qubit}"]
            if index:
                affected.append(f"coupling:{qubits[index - 1]}-{qubit}")
            if index < len(qubits) - 1:
                affected.append(f"coupling:{qubit}-{qubits[index + 1]}")
            detectors.append(DetectorDefinition(detector_id, (2 * index, 2 * index + 1), 0,
                                                tuple(f"gate:{item}" for item in affected), f"region:{qubit}"))
            graph[detector_id] = tuple(affected)
        circuit_hash = stable_hash({"qubits": qubits, "couplers": couplers, "distance": config.code_distance})
        circuit = TargetQECCircuit("surface-memory", circuit_hash,
                                   tuple(f"cycle:{qubit}" for qubit in qubits), tuple(detectors), config.code_distance)
        return topology, limits, circuit, graph

    @property
    def now_s(self) -> float:
        return self._time_s

    @property
    def confirmed_policy(self) -> PolicyActivation:
        return self._policy_activation

    @property
    def policy_transaction_log(self) -> tuple[PolicyTransaction, ...]:
        return self._policy_ledger.events

    @property
    def controller_state_hash(self) -> str:
        return deterministic_hash({
            "policy": dict(self._policy),
            "activation": self._policy_activation,
            "pending": self._pending_policy,
            "ledger": self._policy_ledger.state_hash,
        })

    @property
    def simulator_state_hash(self) -> str:
        return deterministic_hash({
            "time_s": self._time_s,
            "sequence": self._sequence,
            "batch_sequence": self._batch_sequence,
            "policy": dict(self._policy),
            "pending": self._pending_policy,
            "process_state": dict(self._process_state),
            "latent_state": dict(self._latent_state),
            "disturbance_tape": self._disturbance_tape,
            "disturbance_dwell": self._disturbance_dwell,
            "disturbance_epoch_s": self._disturbance_epoch_s,
            "process_rng": self._process_rng.getstate(),
            "characterization_rng": self._characterization_rng.getstate(),
            "controller": self.controller_state_hash,
        })

    def counterfactual_state_fingerprint(self) -> CounterfactualStateFingerprint:
        diagnostic = self._physical_diagnostic()
        return CounterfactualStateFingerprint(
            diagnostic.state_id, diagnostic.disturbance_state_id,
            self.disturbance_realization_id, self.simulator_state_hash,
            self.controller_state_hash,
            deterministic_hash(self._process_rng.getstate()),
            deterministic_hash(self._characterization_rng.getstate()),
            deterministic_hash({
                "simulator": SIMULATOR_VERSION,
                "context": self.context,
                "circuit": self.circuit,
                "detector_control_graph": self.detector_control_graph,
                "base_detector_probability": self.config.base_detector_probability,
                "response_curvature": self.config.response_curvature,
                "cross_coupling_strength": self.config.cross_coupling_strength,
            }),
            self._policy_activation.policy_id, self._policy_activation.policy_hash,
            dict(self._policy), self._time_s,
        )

    def shares_mutable_state_with(self, other: "ScalableQECDevice") -> bool:
        """Return true if clone isolation has been violated by object aliasing."""
        return any(left is right for left, right in (
            (self._policy, other._policy),
            (self._pending_policy, other._pending_policy),
            (self._disturbance_tape, other._disturbance_tape),
            (self._process_state, other._process_state),
            (self._latent_state, other._latent_state),
            (self._process_rng, other._process_rng),
            (self._characterization_rng, other._characterization_rng),
            (self._policy_ledger, other._policy_ledger),
        ) if left is not None and right is not None)

    def oracle_evaluation_view(self, purpose: str) -> OracleEvaluationView:
        view = OracleEvaluationView(self, purpose)
        self._oracle_access_log.append((self._time_s, purpose))
        return view

    def clone(self) -> "ScalableQECDevice":
        """Return an independent state clone with identical RNG and disturbance state.

        The clone is intended for matched scientific counterfactuals.  Deep copying is
        deliberate: no mutable policy, tape, cache, or random-number generator is shared.
        """
        return copy.deepcopy(self)

    def await_policy_acknowledgement(self) -> PolicyActivation:
        """Advance only through controller latency and return the confirmed version."""
        if self._pending_policy is not None:
            remaining = max(0.0, self._pending_policy[1].nominal_activation_s - self._time_s)
            if remaining > 1e-15:
                self._advance_processes(remaining)
            else:
                self._confirm_pending_policy()
        return self._policy_activation

    def advance_elapsed_time(self, duration_s: float) -> None:
        """Advance exogenous state without fabricating QEC observations.

        This represents declared controller/experiment idle time.  It is never counted
        as native-QEC acquisition and is used only where a protocol explicitly matches
        physical elapsed time rather than detector sample budget.
        """
        if duration_s < 0:
            raise ValueError("elapsed-time advance cannot be negative")
        if duration_s:
            self._advance_processes(duration_s)

    @property
    def oracle_access_log(self) -> tuple[tuple[float, str], ...]:
        """Immutable audit record used to detect truth leakage in experiments."""
        return tuple(self._oracle_access_log)

    def _confirm_pending_policy(self) -> None:
        if self._pending_policy is None:
            return
        candidate, activation = self._pending_policy
        acknowledgement_time = activation.nominal_activation_s
        reference = self._policy_ledger.confirmed
        active = self._policy_ledger.mark_active(
            activation.transaction_id,
            reference_policy_id=reference.policy_id,
            reference_policy_hash=reference.policy_hash,
            atomic=True,
        )
        observed_state_id = stable_hash({
            "expected_activation_state_id": active.expected_activation_state_id,
            "timestamp_s": acknowledgement_time,
            "policy_hash": active.policy_hash,
            "disturbance_realization_id": self.disturbance_realization_id,
        })
        confirmed = self._policy_ledger.acknowledge(
            active.transaction_id, observed_policy_hash=stable_hash(candidate),
            observed_activation_state_id=observed_state_id,
            acknowledged_at_s=acknowledgement_time, atomic=True,
        )
        acknowledgement = confirmed.activation_acknowledgement
        self._policy = candidate
        self._policy_activation = replace(
            activation,
            acknowledged_at_s=acknowledgement_time,
            activation_acknowledgement=(acknowledgement.acknowledgement_id
                                        if acknowledgement else ""),
            lifecycle_state=PolicyLifecycleState.CONFIRMED,
        )
        self._pending_policy = None

    def apply_policy(self, controls: Mapping[str, float], *, policy_id: str,
                     candidate_id: str | None = None,
                     perturbation: Mapping[str, float] | None = None,
                     reference_policy_id: str | None = None,
                     reference_policy_hash: str | None = None,
                     created_from_state_id: str | None = None,
                     expected_activation_state_id: str | None = None,
                     supervisor_authorization: str = "simulator-runtime-assurance:auto") -> PolicyActivation:
        """Validate, authorize, schedule, and later acknowledge one atomic policy.

        The optional reference fields are mandatory at hardware integration boundaries.
        Defaults bind direct simulator calls to the currently confirmed version, so even
        tests and baseline adapters retain explicit reference semantics.
        """
        if self._pending_policy is not None:
            # A hardware adapter would wait for/observe the acknowledgement.  The simulator
            # advances only through the outstanding atomic activation latency.
            remaining = max(0., self._pending_policy[1].nominal_activation_s - self._time_s)
            if remaining > 1e-15:
                self._advance_processes(remaining)
            else:
                self._confirm_pending_policy()
        confirmed_reference = self._policy_ledger.confirmed
        supplied_reference_id = reference_policy_id or confirmed_reference.policy_id
        supplied_reference_hash = reference_policy_hash or confirmed_reference.policy_hash
        if (supplied_reference_id != confirmed_reference.policy_id
                or supplied_reference_hash != confirmed_reference.policy_hash):
            raise ValueError("policy reference changed; reject and reproject before activation")
        candidate = dict(self._policy)
        candidate.update(controls)
        unknown = set(candidate) - set(self.limits.controls)
        if unknown:
            raise ValueError(f"unknown controls: {sorted(unknown)}")
        for control, value in candidate.items():
            bound = self.limits.controls[control]
            if not bound.validate(value):
                raise ValueError(f"hard bound violated for {control}")
            previous = self._policy[control]
            if abs(value - previous) > bound.max_slew + 1e-12:
                raise ValueError(f"slew bound violated for {control}")
        requested = self._time_s
        acknowledged = requested + self.config.controller_latency_s
        creation_state = created_from_state_id or self._physical_diagnostic().state_id
        expected_state = expected_activation_state_id or stable_hash({
            "created_from_state_id": creation_state,
            "reference_policy_id": supplied_reference_id,
            "reference_policy_hash": supplied_reference_hash,
            "policy_hash": stable_hash(candidate),
            "nominal_activation_s": acknowledged,
        })
        transaction = self._policy_ledger.propose(
            policy_id, candidate,
            reference_policy_id=supplied_reference_id,
            reference_policy_hash=supplied_reference_hash,
            created_from_state_id=creation_state,
            expected_activation_state_id=expected_state,
            created_at_s=requested,
        )
        projection = PolicyCertificate.issue(
            "projection", transaction.policy_hash, supplied_reference_id, True,
            "complete policy is projected from the confirmed reference")
        bounds = PolicyCertificate.issue(
            "bounds", transaction.policy_hash, supplied_reference_id, True,
            "all controls satisfy registered hard bounds")
        slew = PolicyCertificate.issue(
            "slew", transaction.policy_hash, supplied_reference_id, True,
            "all controls satisfy registered slew bounds")
        transaction = self._policy_ledger.pending_validation(
            transaction.transaction_id, projection=projection, bounds=bounds, slew=slew)
        transaction = self._policy_ledger.authorize(
            transaction.transaction_id, supervisor_authorization)
        activation = PolicyActivation(
            policy_id, stable_hash(candidate), requested, acknowledged, 0.0, dict(candidate),
            dict(perturbation or {}), candidate_id,
            transaction_id=transaction.transaction_id,
            reference_policy_id=supplied_reference_id,
            reference_policy_hash=supplied_reference_hash,
            created_from_state_id=creation_state,
            expected_activation_state_id=expected_state,
            projection_certificate=projection.certificate_id,
            bounds_certificate=bounds.certificate_id,
            slew_certificate=slew.certificate_id,
            supervisor_authorization=supervisor_authorization,
            lifecycle_state=PolicyLifecycleState.AUTHORIZED,
        )
        self._pending_policy = (candidate, activation)
        return activation

    def _extend_disturbance_tape(self, required_index: int) -> None:
        """Extend a fixed-time latent tape independent of controller call boundaries."""
        resolution = self.config.disturbance_resolution_s
        current_length = len(next(iter(self._disturbance_tape.values()), [0.0]))
        for index in range(current_length, required_index + 1):
            timestamp = index * resolution
            for process in self.processes:
                values = self._disturbance_tape[process.process_id]
                value = values[-1]
                if process.kind is DriftKind.CONSTANT:
                    value = process.amplitude
                elif process.kind is DriftKind.SINUSOID:
                    value = process.amplitude * math.sin(
                        2 * math.pi * timestamp / max(process.period_s, 1e-9) + process.phase_rad)
                elif process.kind in {DriftKind.RANDOM_TELEGRAPH, DriftKind.SEMI_MARKOV_TELEGRAPH}:
                    rate = process.rate_hz
                    dwell = self._disturbance_dwell[process.process_id]
                    if process.kind is DriftKind.SEMI_MARKOV_TELEGRAPH:
                        rate *= min(4., .25 + dwell / max(process.mean_dwell_s, 1e-9))
                    if self._process_rng.random() < 1 - math.exp(-rate * resolution):
                        value = -value if value else process.amplitude
                        dwell = 0.0
                    else:
                        dwell += resolution
                    self._disturbance_dwell[process.process_id] = dwell
                    value = math.copysign(process.amplitude, value or 1.0)
                elif process.kind is DriftKind.ORNSTEIN_UHLENBECK:
                    value += (-process.ou_kappa * value * resolution
                              + self._process_rng.gauss(0., process.diffusion * math.sqrt(resolution)))
                elif process.kind is DriftKind.RANDOM_WALK:
                    value += self._process_rng.gauss(0., process.diffusion * math.sqrt(resolution))
                elif process.kind is DriftKind.STEP:
                    if process.step_time_s is not None:
                        value = process.amplitude if timestamp >= process.step_time_s else 0.0
                    elif self._process_rng.random() < 1 - math.exp(-process.rate_hz * resolution):
                        value += self._process_rng.gauss(0., process.amplitude)
                elif process.kind is DriftKind.UNKNOWN_HEAVY_TAILED:
                    value += (self._process_rng.gauss(0., process.diffusion * math.sqrt(resolution))
                              / max(abs(self._process_rng.gauss(0., 1.)), .12))
                values.append(value)

    def _advance_processes(self, dt: float) -> None:
        if dt < 0:
            raise ValueError("time cannot run backwards")
        self._time_s += dt
        if self._pending_policy is not None and self._pending_policy[1].nominal_activation_s <= self._time_s + 1e-15:
            self._confirm_pending_policy()
        if self._disturbance_epoch_s is None:
            # Keep the device stationary and, critically, do not advance the latent RNG
            # while Stage 0 or a pre-disturbance baseline is being acquired.
            self._process_state = {process.process_id: 0.0 for process in self.processes}
            self._latent_state = {control: 0.0 for control in self.limits.controls}
            return
        resolution = self.config.disturbance_resolution_s
        coordinate = max(0.0, self._time_s - self._disturbance_epoch_s) / resolution
        left = max(0, int(math.floor(coordinate + 1e-12)))
        right = left + 1
        self._extend_disturbance_tape(right)
        fraction = min(1.0, max(0.0, coordinate - left))
        contributions = {control: 0.0 for control in self._latent_state}
        for process in self.processes:
            values = self._disturbance_tape[process.process_id]
            if process.kind in {DriftKind.RANDOM_TELEGRAPH, DriftKind.SEMI_MARKOV_TELEGRAPH, DriftKind.STEP}:
                value = values[left]
            else:
                value = values[left] + fraction * (values[right] - values[left])
            self._process_state[process.process_id] = value
            parent_scale = 1.0
            if process.parent_process_id:
                parent_scale += self._process_state.get(process.parent_process_id, 0.)
            for control, loading in process.loadings.items():
                if control in contributions:
                    contributions[control] += loading * parent_scale * value
        self._latent_state = contributions

    @staticmethod
    def _mix64(value: int) -> int:
        value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return value ^ (value >> 31)

    def _sample_uniform(self, stream: int, absolute_cycle: int, index: int = 0) -> float:
        key = ((self.config.seed & 0xFFFFFFFFFFFFFFFF)
               ^ ((stream + 1) * 0xD1B54A32D192ED03)
               ^ ((absolute_cycle + 1) * 0x9E3779B97F4A7C15)
               ^ ((index + 1) * 0x94D049BB133111EB))
        return self._mix64(key & 0xFFFFFFFFFFFFFFFF) / 2**64

    def _response_components(self, detector_id: str,
                             policy: Mapping[str, float] | None = None) -> tuple[float, float, float]:
        """Map active-optimum mismatch to floor, local excess, and cross coupling."""
        applied = policy or self._policy
        controls = self.detector_control_graph[detector_id]
        errors = [self._latent_state[control] - applied[control] for control in controls]
        energy = sum(error * error for error in errors) / max(1, len(errors))
        local_excess = self.config.response_curvature * energy
        pair_count = max(1, len(errors) * (len(errors) - 1) // 2)
        cross = (self.config.cross_coupling_strength
                 * sum(abs(errors[i] * errors[j]) for i in range(len(errors))
                       for j in range(i + 1, len(errors))) / pair_count)
        return self.config.base_detector_probability, local_excess, cross

    def _detector_probability(self, detector_id: str,
                              policy: Mapping[str, float] | None = None) -> float:
        floor, local_excess, cross = self._response_components(detector_id, policy)
        return min(self.config.maximum_detector_probability,
                   max(1e-6, floor + local_excess + cross))

    def _physical_diagnostic(self, policy: Mapping[str, float] | None = None) -> PhysicalStateDiagnostic:
        applied = dict(policy or self._policy)
        if set(applied) != set(self.limits.controls):
            raise ValueError("diagnostic policy must contain exactly the registered controls")
        mismatch = {control: self._latent_state[control] - applied[control]
                    for control in self.limits.controls}
        probabilities: dict[str, float] = {}
        controllable: dict[str, float] = {}
        cross: dict[str, float] = {}
        saturated: list[str] = []
        for detector in self.detector_control_graph:
            _, local_excess, coupling = self._response_components(detector, applied)
            probability = self._detector_probability(detector, applied)
            probabilities[detector] = probability
            controllable[detector] = local_excess
            cross[detector] = coupling
            if probability >= self.config.maximum_detector_probability - 1e-12:
                saturated.append(detector)
        expected_rate = sum(probabilities.values()) / max(1, len(probabilities))
        expected_logical = min(.5, self.config.logical_scale
                               * expected_rate ** ((self.config.code_distance + 1) / 2))
        disturbance_state_id = stable_hash({
            "realization": self.disturbance_realization_id,
            "elapsed": self.disturbance_elapsed_s,
            "process": self._process_state,
            "latent": self._latent_state,
        })
        state_id = stable_hash({
            "disturbance_state_id": disturbance_state_id,
            "policy_hash": stable_hash(applied),
            "timestamp_s": self._time_s,
        })
        return PhysicalStateDiagnostic(
            state_id, disturbance_state_id, self._time_s,
            self._policy_activation.policy_id, stable_hash(applied),
            dict(self._latent_state), applied, mismatch, probabilities,
            controllable, cross, expected_rate, expected_logical,
            self.config.base_detector_probability,
            self.config.validated_mismatch_radius, tuple(saturated),
        )

    def _stationary_detector_probabilities(self, policy: Mapping[str, float],
                                           policy_hash: str) -> np.ndarray:
        """Cache the exact detector probabilities for a disarmed, zero-latent device."""
        cached = self._stationary_probability_cache.get(policy_hash)
        if cached is not None:
            return cached
        probabilities_list = []
        for controls in self.detector_control_graph.values():
            errors = [-policy[control] for control in controls]
            energy = sum(error*error for error in errors)/max(1, len(errors))
            pair_count = max(1, len(errors)*(len(errors)-1)//2)
            cross = (self.config.cross_coupling_strength
                     * sum(abs(errors[i]*errors[j]) for i in range(len(errors))
                           for j in range(i+1, len(errors)))/pair_count)
            probabilities_list.append(min(
                self.config.maximum_detector_probability,
                max(1e-6, self.config.base_detector_probability
                    + self.config.response_curvature*energy + cross),
            ))
        probabilities = np.asarray(probabilities_list, dtype=np.float64)
        probabilities.setflags(write=False)
        self._stationary_probability_cache[policy_hash] = probabilities
        return probabilities

    def _sample_uniform_array(self, stream: int, absolute_cycles: np.ndarray,
                              count: int) -> np.ndarray:
        """Vectorized SplitMix64 samples exactly matching :meth:`_sample_uniform`."""
        mask = 0xFFFFFFFFFFFFFFFF
        seed = np.uint64(self.config.seed & mask)
        stream_term = np.uint64(((stream + 1) * 0xD1B54A32D192ED03) & mask)
        cycle_values = absolute_cycles.astype(np.uint64, copy=False)
        indices = np.arange(1, count + 1, dtype=np.uint64)
        with np.errstate(over="ignore"):
            keys = (seed ^ stream_term
                    ^ ((cycle_values[:, None] + np.uint64(1))
                       * np.uint64(0x9E3779B97F4A7C15))
                    ^ (indices[None, :] * np.uint64(0x94D049BB133111EB)))
            values = keys + np.uint64(0x9E3779B97F4A7C15)
            values = ((values ^ (values >> np.uint64(30)))
                      * np.uint64(0xBF58476D1CE4E5B9))
            values = ((values ^ (values >> np.uint64(27)))
                      * np.uint64(0x94D049BB133111EB))
            values = values ^ (values >> np.uint64(31))
        return values.astype(np.float64) / float(2**64)

    def _finish_batch(self, records: list[RawMeasurementRecord], events: int,
                      exposures: int, logical_failures: int, cycles: int,
                      detector_counts: Mapping[str, Sequence[int]]) -> QECObservationBatch:
        batch_id = f"batch:{self._batch_sequence}"
        self._batch_sequence += 1
        diagnostic = self._physical_diagnostic()
        return QECObservationBatch(tuple(records), self._policy_activation, self.context,
                                   events, exposures, logical_failures, cycles, batch_id,
                                   {key: tuple(value) for key, value in detector_counts.items()},
                                   diagnostic.state_id, diagnostic.disturbance_state_id,
                                   self.simulator_state_hash, self.controller_state_hash)

    def _acquire_scalar(self, cycles: int, *, shot: int,
                        retain_records: bool) -> QECObservationBatch:
        """Reference acquisition used for dynamic disturbances and equivalence tests."""
        records: list[RawMeasurementRecord] = []
        events = exposures = logical_failures = 0
        detector_counts = {detector.detector_id: [0, 0] for detector in self.circuit.detectors}
        for cycle in range(cycles):
            self._advance_processes(self.config.cycle_period_s)
            absolute_cycle = round(self._time_s / self.config.cycle_period_s)
            common_shock = self._sample_uniform(0, absolute_cycle) < self.config.correlation_probability
            measurements: list[int | None] | None = ([0] * (2 * len(self.circuit.detectors))
                                                      if retain_records else None)
            cycle_events = 0
            for index, detector in enumerate(self.circuit.detectors):
                if self._sample_uniform(1, absolute_cycle, index) < self.config.measurement_dropout_probability:
                    if measurements is not None:
                        measurements[2 * index] = None
                    continue
                event = int(common_shock or self._sample_uniform(2, absolute_cycle, index)
                            < self._detector_probability(detector.detector_id))
                if measurements is not None:
                    measurements[2 * index] = event
                    measurements[2 * index + 1] = 0
                events += event
                exposures += 1
                cycle_events += event
                detector_counts[detector.detector_id][0] += event
                detector_counts[detector.detector_id][1] += 1
            local_rate = cycle_events / max(1, len(self.circuit.detectors))
            logical_probability = min(.5, self.config.logical_scale
                                      * local_rate ** ((self.config.code_distance + 1) / 2))
            logical_failures += int(self._sample_uniform(3, absolute_cycle) < logical_probability)
            if measurements is not None:
                records.append(RawMeasurementRecord(
                    f"record:{self._sequence}", self._sequence, shot, cycle, self._time_s,
                    tuple(measurements), self.circuit.circuit_hash, self._measurement_channel_ids,
                ))
            self._sequence += 1
        return self._finish_batch(records, events, exposures, logical_failures, cycles,
                                  detector_counts)

    def _acquire_stationary_vectorized(self, cycles: int, *, shot: int,
                                       retain_records: bool) -> QECObservationBatch:
        """Exact counter-RNG acquisition for the stationary pre-disturbance phase."""
        detector_count = len(self.circuit.detectors)
        cursor = self._time_s
        timestamps: list[float] = []
        absolute_values: list[int] = []
        for _ in range(cycles):
            cursor += self.config.cycle_period_s
            timestamps.append(cursor)
            absolute_values.append(round(cursor / self.config.cycle_period_s))
        absolute_cycles = np.asarray(absolute_values, dtype=np.uint64)

        current_probabilities = self._stationary_detector_probabilities(
            self._policy, self._policy_activation.policy_hash)
        probabilities = np.broadcast_to(current_probabilities, (cycles, detector_count))
        pending = self._pending_policy
        if pending is not None:
            candidate, activation = pending
            activation_mask = np.asarray([
                timestamp + 1e-15 >= activation.nominal_activation_s
                for timestamp in timestamps
            ], dtype=bool)
            if activation_mask.any():
                probabilities = np.array(probabilities, copy=True)
                probabilities[activation_mask] = self._stationary_detector_probabilities(
                    candidate, activation.policy_hash)

        common_shocks = (self._sample_uniform_array(0, absolute_cycles, 1)[:, 0]
                         < self.config.correlation_probability)
        if self.config.measurement_dropout_probability:
            exposed = (self._sample_uniform_array(1, absolute_cycles, detector_count)
                       >= self.config.measurement_dropout_probability)
        else:
            exposed = np.ones((cycles, detector_count), dtype=bool)
        events_matrix = ((common_shocks[:, None]
                          | (self._sample_uniform_array(2, absolute_cycles, detector_count)
                             < probabilities)) & exposed)
        cycle_events = events_matrix.sum(axis=1, dtype=np.int64)
        local_rates = cycle_events.astype(np.float64) / max(1, detector_count)
        logical_probabilities = np.minimum(
            .5, self.config.logical_scale
            * local_rates ** ((self.config.code_distance + 1) / 2))
        logical_failures = int(np.count_nonzero(
            self._sample_uniform_array(3, absolute_cycles, 1)[:, 0]
            < logical_probabilities))

        records: list[RawMeasurementRecord] = []
        initial_sequence = self._sequence
        if retain_records:
            for cycle in range(cycles):
                measurements: list[int | None] = [0] * (2 * detector_count)
                for index in range(detector_count):
                    measurements[2 * index] = (int(events_matrix[cycle, index])
                                                 if exposed[cycle, index] else None)
                records.append(RawMeasurementRecord(
                    f"record:{initial_sequence + cycle}", initial_sequence + cycle,
                    shot, cycle, timestamps[cycle], tuple(measurements),
                    self.circuit.circuit_hash, self._measurement_channel_ids,
                ))
        self._sequence += cycles
        self._time_s = timestamps[-1]
        if pending is not None and pending[1].nominal_activation_s <= self._time_s + 1e-15:
            self._confirm_pending_policy()
        self._process_state = {process.process_id: 0.0 for process in self.processes}
        self._latent_state = {control: 0.0 for control in self.limits.controls}

        detector_events = events_matrix.sum(axis=0, dtype=np.int64)
        detector_exposures = exposed.sum(axis=0, dtype=np.int64)
        counts = {
            detector.detector_id: [int(detector_events[index]), int(detector_exposures[index])]
            for index, detector in enumerate(self.circuit.detectors)
        }
        return self._finish_batch(records, int(detector_events.sum()),
                                  int(detector_exposures.sum()), logical_failures,
                                  cycles, counts)

    def _acquire_dynamic_vectorized(self, cycles: int, *, shot: int) -> QECObservationBatch:
        """Counter-RNG dynamic acquisition with scalar-equivalent state evolution.

        This path is deliberately limited to aggregate acquisitions.  It constructs the
        same fixed-resolution latent tape and uses the same operation order as the scalar
        reference, but evaluates detector probabilities and counter-based random streams
        in arrays.  The final physical state is recomputed by the scalar state transition
        to keep lifecycle and counterfactual hashes exactly aligned.
        """
        detector_count = len(self.circuit.detectors)
        cursor = self._time_s
        timestamps: list[float] = []
        absolute_values: list[int] = []
        for _ in range(cycles):
            cursor += self.config.cycle_period_s
            timestamps.append(cursor)
            absolute_values.append(round(cursor / self.config.cycle_period_s))
        absolute_cycles = np.asarray(absolute_values, dtype=np.uint64)
        timestamp_array = np.asarray(timestamps, dtype=np.float64)

        resolution = self.config.disturbance_resolution_s
        epoch = float(self._disturbance_epoch_s or 0.0)
        coordinates = np.maximum(0.0, timestamp_array - epoch) / resolution
        left = np.maximum(0, np.floor(coordinates + 1e-12).astype(np.int64))
        right = left + 1
        fractions = np.clip(coordinates - left, 0.0, 1.0)
        self._extend_disturbance_tape(int(right.max()))

        process_paths: dict[str, np.ndarray] = {}
        latent_paths = {
            control: np.zeros(cycles, dtype=np.float64)
            for control in self.limits.controls
        }
        for process in self.processes:
            tape = np.asarray(self._disturbance_tape[process.process_id], dtype=np.float64)
            if process.kind in {
                DriftKind.RANDOM_TELEGRAPH,
                DriftKind.SEMI_MARKOV_TELEGRAPH,
                DriftKind.STEP,
            }:
                values = tape[left]
            else:
                values = tape[left] + fractions * (tape[right] - tape[left])
            process_paths[process.process_id] = values
            parent_scale: float | np.ndarray = 1.0
            if process.parent_process_id:
                parent_scale = 1.0 + process_paths.get(
                    process.parent_process_id, np.zeros(cycles, dtype=np.float64))
            for control, loading in process.loadings.items():
                if control in latent_paths:
                    latent_paths[control] += loading * parent_scale * values

        pending = self._pending_policy
        activation_mask = np.zeros(cycles, dtype=bool)
        pending_policy: Mapping[str, float] | None = None
        if pending is not None:
            pending_policy, activation = pending
            activation_mask = timestamp_array + 1e-15 >= activation.nominal_activation_s
        policy_paths: dict[str, np.ndarray] = {}
        for control in self.limits.controls:
            current = self._policy[control]
            if pending_policy is None:
                policy_paths[control] = np.full(cycles, current, dtype=np.float64)
            else:
                policy_paths[control] = np.where(
                    activation_mask, pending_policy[control], current).astype(np.float64)

        probabilities = np.empty((cycles, detector_count), dtype=np.float64)
        for detector_index, detector in enumerate(self.circuit.detectors):
            errors = [
                latent_paths[control] - policy_paths[control]
                for control in self.detector_control_graph[detector.detector_id]
            ]
            energy = np.zeros(cycles, dtype=np.float64)
            for error in errors:
                energy += error * error
            energy /= max(1, len(errors))
            pair_count = max(1, len(errors) * (len(errors) - 1) // 2)
            cross = np.zeros(cycles, dtype=np.float64)
            for first in range(len(errors)):
                for second in range(first + 1, len(errors)):
                    cross += np.abs(errors[first] * errors[second])
            cross *= self.config.cross_coupling_strength / pair_count
            probabilities[:, detector_index] = np.minimum(
                self.config.maximum_detector_probability,
                np.maximum(
                    1e-6,
                    self.config.base_detector_probability
                    + self.config.response_curvature * energy + cross,
                ),
            )

        common_shocks = (
            self._sample_uniform_array(0, absolute_cycles, 1)[:, 0]
            < self.config.correlation_probability
        )
        if self.config.measurement_dropout_probability:
            exposed = (
                self._sample_uniform_array(1, absolute_cycles, detector_count)
                >= self.config.measurement_dropout_probability
            )
        else:
            exposed = np.ones((cycles, detector_count), dtype=bool)
        events_matrix = (
            common_shocks[:, None]
            | (self._sample_uniform_array(2, absolute_cycles, detector_count) < probabilities)
        ) & exposed
        cycle_events = events_matrix.sum(axis=1, dtype=np.int64)
        local_rates = cycle_events.astype(np.float64) / max(1, detector_count)
        logical_probabilities = np.minimum(
            .5,
            self.config.logical_scale
            * local_rates ** ((self.config.code_distance + 1) / 2),
        )
        logical_failures = int(np.count_nonzero(
            self._sample_uniform_array(3, absolute_cycles, 1)[:, 0]
            < logical_probabilities
        ))

        self._sequence += cycles
        self._time_s = timestamps[-1]
        # Reuse the scalar transition at dt=0 for byte-for-byte final state/lifecycle
        # identity without consuming another disturbance or detector random number.
        self._advance_processes(0.0)
        detector_events = events_matrix.sum(axis=0, dtype=np.int64)
        detector_exposures = exposed.sum(axis=0, dtype=np.int64)
        counts = {
            detector.detector_id: [
                int(detector_events[index]), int(detector_exposures[index])
            ]
            for index, detector in enumerate(self.circuit.detectors)
        }
        return self._finish_batch(
            [], int(detector_events.sum()), int(detector_exposures.sum()),
            logical_failures, cycles, counts,
        )

    def acquire(self, cycles: int, *, shot: int = 0,
                retain_records: bool = True) -> QECObservationBatch:
        if cycles <= 0:
            raise ValueError("cycles must be positive")
        if self._disturbance_epoch_s is None and self.config.stationary_vectorized_acquisition:
            return self._acquire_stationary_vectorized(
                cycles, shot=shot, retain_records=retain_records)
        if self.config.dynamic_vectorized_acquisition and not retain_records:
            return self._acquire_dynamic_vectorized(cycles, shot=shot)
        return self._acquire_scalar(cycles, shot=shot, retain_records=retain_records)

    def characterize_controls(self, controls: Sequence[str] | None = None, *, shots: int = 256) -> CharacterizationResult:
        """Dedicated, interrupting characterization returning noisy estimates, never exact truth."""
        if shots <= 0:
            raise ValueError("shots must be positive")
        selected = tuple(controls or self.limits.controls)
        if any(control not in self.limits.controls for control in selected):
            raise ValueError("unknown characterization control")
        downtime = shots * self.config.cycle_period_s
        self._advance_processes(downtime)
        standard_error = .5 / math.sqrt(shots)
        estimates = {control: max(self.limits.controls[control].minimum,
                     min(self.limits.controls[control].maximum,
                         self._latent_state[control] + self._characterization_rng.gauss(0., standard_error))) for control in selected}
        return CharacterizationResult(estimates, {control: standard_error ** 2 for control in selected},
                                      shots, downtime, self._time_s)
