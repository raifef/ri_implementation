"""Reproducible scalability study matched to Sivak et al., Nature 655 (2026).

The paper's Figure 5 simulator and training code are proprietary.  This module therefore
separates three evidence layers instead of presenting a synthetic reproduction as source
data:

* immutable published protocol anchors and equations;
* a reduced, declared factor-graph surrogate used for matched sweeps;
* optional end-to-end probes of this repository's implemented controller stack.

Every output row carries its evidence layer and native-QEC budget so plots cannot silently
mix published claims, surrogate outcomes and implementation throughput measurements.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import threading
import time
from typing import Mapping, Sequence

from hdfa_rl_suite.common import deterministic_hash


PAPER_METHOD = "paper_sparse_policy_gradient"
HDFA_METHOD = "predictive_hdfa_residual_rl"


@dataclass(frozen=True)
class PaperProtocol:
    title: str = "Reinforcement learning control of quantum error correction"
    article_doi: str = "10.1038/s41586-026-10759-2"
    supplement_url: str = ("https://media.springernature.com/original/springer-static/esm/"
                           "art%3A10.1038%2Fs41586-026-10759-2/MediaObjects/"
                           "41586_2026_10759_MOESM1_ESM.pdf")
    source_data_concept_doi: str = "10.5281/zenodo.17566521"
    source_data_version_doi: str = "10.5281/zenodo.18896801"
    source_data_archive: str = "google_reinforcement_learning_qec.zip"
    source_data_archive_bytes: int = 7_786_791_716
    source_data_archive_md5: str = "ca54323082fcd0e3671d5b90ce45d85c"
    distances: tuple[int, ...] = (3, 5, 7, 9, 11, 13, 15)
    parameters_per_gate: tuple[int, ...] = (1, 10, 30)
    memory_cycles: int = 10
    scaling_epochs: int = 500
    realtime_epochs: int = 1_000
    realtime_candidates_per_epoch: int = 50
    realtime_cycles_per_candidate: int = 36_000
    realtime_total_qec_cycles: int = 1_800_000_000
    physical_error_threshold: float = 1.79e-3
    empirical_response_time_epochs: float = 130.
    empirical_steerability_frequency: float = 1 / 150
    distance_5_reward_components: int = 97
    distance_5_policy_parameters: int = 1_582
    distance_5_mean_parameters_per_detector: float = 302.
    distance_5_mean_detectors_per_parameter: float = 18.


@dataclass(frozen=True)
class ScalabilityConfig:
    profile: str = "smoke"
    distances: tuple[int, ...] = (3, 5)
    parameters_per_gate: tuple[int, ...] = (1, 10)
    epochs: int = 30
    seeds: tuple[int, ...] = (7, 19)
    steering_frequencies: tuple[float, ...] = (1e-3, 1 / 150, 2e-2)
    entropy_regularizations: tuple[float, ...] = (1e-3, 1e-2, 1e-1)
    steering_epochs: int = 240
    physical_error_threshold: float = 1.79e-3
    irreducible_physical_error: float = 4.0e-4
    initial_physical_error: float = 3.2e-3
    logical_error_at_threshold: float = 1.0e-2
    reference_learning_rate: float = .020
    hdfa_modeled_fraction: float = .75
    static_update_noise: float = .08
    paper_candidates_per_epoch: int = 50
    paper_cycles_per_candidate: int = 36_000
    hdfa_candidates_per_epoch: int = 4
    hdfa_cycles_per_candidate: int = 36_000
    exploration_noise_scale: float = 1.5
    run_pipeline_probe: bool = False
    pipeline_distances: tuple[int, ...] = (3, 5, 7, 9, 11, 13, 15)
    pipeline_epochs: int = 2
    pipeline_cycles_per_interval: int = 10
    pipeline_candidate_cycles: int = 2
    pipeline_candidates: int = 4
    pipeline_workers: int = 1
    pipeline_bootstrap_characterization_shots: int = 384
    pipeline_bootstrap_validation_cycles: int = 512
    pipeline_bootstrap_target_stddev: float = .035
    pipeline_bootstrap_qec_rate_limit: float = .10
    pipeline_bootstrap_block_familywise_alpha: float = 1e-4
    pipeline_bootstrap_sensitivity_max_batch_size: int = 32
    pipeline_bootstrap_sensitivity_interference_alpha: float = 1e-4
    pipeline_baseline_cycles: int = 64

    def __post_init__(self) -> None:
        if not self.distances or any(distance < 3 or distance % 2 == 0 for distance in self.distances):
            raise ValueError("surface-code distances must be non-empty odd integers >= 3")
        if not self.parameters_per_gate or any(value <= 0 for value in self.parameters_per_gate):
            raise ValueError("parameters_per_gate must be positive")
        if self.epochs < 2 or self.steering_epochs < 2 or not self.seeds:
            raise ValueError("experiments require at least two epochs and one seed")
        if not 0 <= self.hdfa_modeled_fraction < 1:
            raise ValueError("hdfa_modeled_fraction must lie in [0, 1)")
        if any(value <= 0 for value in self.steering_frequencies + self.entropy_regularizations):
            raise ValueError("steering axes must be strictly positive")
        if self.pipeline_workers < 1:
            raise ValueError("pipeline_workers must be at least one")
        if (self.pipeline_epochs <= 0 or self.pipeline_cycles_per_interval <= 0
                or self.pipeline_candidate_cycles <= 0 or self.pipeline_candidates <= 0
                or self.pipeline_bootstrap_characterization_shots <= 0
                or self.pipeline_bootstrap_validation_cycles <= 0
                or self.pipeline_bootstrap_sensitivity_max_batch_size <= 0
                or self.pipeline_baseline_cycles <= 0):
            raise ValueError("pipeline and bootstrap budgets must be positive")
        if (self.pipeline_bootstrap_target_stddev <= 0
                or not 0 < self.pipeline_bootstrap_qec_rate_limit < 1
                or not 0 < self.pipeline_bootstrap_block_familywise_alpha < 1
                or not 0 < self.pipeline_bootstrap_sensitivity_interference_alpha < 1):
            raise ValueError("pipeline bootstrap statistical thresholds must be physical")

    @classmethod
    def for_profile(cls, profile: str) -> "ScalabilityConfig":
        if profile == "smoke":
            return cls()
        if profile == "paper":
            return cls(
                profile="paper", distances=PaperProtocol().distances,
                parameters_per_gate=PaperProtocol().parameters_per_gate, epochs=500,
                seeds=(7, 19, 43), steering_frequencies=_logspace(-3., -1., 9),
                entropy_regularizations=_logspace(-4., -1., 10), steering_epochs=1_000,
            )
        if profile == "full":
            return cls(
                profile="full", distances=PaperProtocol().distances,
                parameters_per_gate=PaperProtocol().parameters_per_gate, epochs=500,
                seeds=(7, 19, 43, 71, 101), steering_frequencies=_logspace(-3., -1., 13),
                entropy_regularizations=_logspace(-4., -1., 13), steering_epochs=1_000,
                run_pipeline_probe=True, pipeline_workers=8,
            )
        raise ValueError(f"unknown scalability profile: {profile}")


@dataclass(frozen=True)
class ScalingPoint:
    evidence_layer: str
    method: str
    seed: int
    code_distance: int
    parameters_per_gate: int
    physical_qubits: int
    control_parameters: int
    epoch: int
    cumulative_qec_cycles: int
    physical_error_rate: float
    logical_error_rate: float
    lambda_ratio: float


@dataclass(frozen=True)
class ConvergencePoint:
    method: str
    seed: int
    code_distance: int
    parameters_per_gate: int
    control_parameters: int
    epoch: int
    distance_to_local_optimum: float
    normalized_speed: float


@dataclass(frozen=True)
class ConvergenceFit:
    method: str
    code_distance: int
    parameters_per_gate: int
    control_parameters: int
    gamma: float
    r_squared: float
    points: int


@dataclass(frozen=True)
class SteerabilityPoint:
    evidence_layer: str
    method: str
    seed: int
    drift_frequency: float
    entropy_regularization: float
    stochastic_improvement: float
    learned_mean_improvement: float
    fixed_cumulative_cost: float
    optimal_cumulative_cost: float
    stochastic_cumulative_cost: float
    learned_cumulative_cost: float
    qec_cycles: int


@dataclass(frozen=True)
class ResourcePoint:
    evidence_layer: str
    method: str
    code_distance: int
    parameters_per_gate: int
    physical_qubits: int
    one_qubit_gates: int
    two_qubit_gates: int
    detector_factors: int
    control_parameters: int
    sparse_factor_edges: int
    candidates_per_epoch: int
    qec_cycles_per_epoch: int
    estimated_policy_state_bytes: int


@dataclass(frozen=True)
class SampleEfficiencyPoint:
    method: str
    seed: int
    code_distance: int
    parameters_per_gate: int
    target_fraction: float
    achieved: bool
    epoch: int | None
    cumulative_qec_cycles: int | None


@dataclass(frozen=True)
class PipelineProbePoint:
    evidence_layer: str
    method: str
    seed: int
    code_distance: int
    physical_qubits: int
    suite_control_variables: int
    paper_p30_control_parameters: int
    epoch: int
    detector_event_rate: float
    logical_failure_rate: float
    candidate_evaluations: int
    qec_cycles: int
    elapsed_s: float
    process_memory_baseline_bytes: int
    peak_process_memory_bytes: int
    peak_incremental_process_memory_bytes: int
    memory_measurement: str
    worker_concurrency: int
    condition_process_isolation: str
    bootstrap_characterization_shots: int
    bootstrap_qec_cycles: int
    bootstrap_device_time_s: float
    bootstrap_wall_time_s: float
    bootstrap_process_memory_baseline_bytes: int
    bootstrap_peak_process_memory_bytes: int
    bootstrap_incremental_process_memory_bytes: int
    bootstrap_execution_id: str
    bootstrap_execution_share: float
    pre_disturbance_baseline_cycles: int
    pre_disturbance_baseline_rate: float
    pre_disturbance_observation_hash: str
    disturbance_realization_id: str
    disturbance_epoch_s: float
    lifecycle_violation_count: int


@dataclass(frozen=True)
class PipelineProbeFailure:
    evidence_layer: str
    method: str
    seed: int
    code_distance: int
    physical_qubits: int
    epoch: int | None
    phase: str
    status: str
    exception_type: str
    reason: str
    disturbance_realization_id: str
    disturbance_epoch_s: float | None
    bootstrap_replay_hash: str


@dataclass(frozen=True)
class PipelineConditionResult:
    points: tuple[PipelineProbePoint, ...]
    failures: tuple[PipelineProbeFailure, ...]


PIPELINE_CHECKPOINT_SCHEMA = "evaluation.scalability.condition.v3"


class PipelineCheckpointError(RuntimeError):
    """Raised when resume evidence is corrupt or belongs to another protocol."""


def _pipeline_condition_fingerprint(config: ScalabilityConfig, distance: int,
                                    seed: int) -> str:
    from hdfa_rl_suite import __version__
    from hdfa_rl_suite.simulator import SIMULATOR_VERSION

    scientific_config = asdict(config)
    # Worker count affects throughput context but not the controller trajectory or
    # disturbance realization.  Each point records its own concurrency so completed
    # conditions can be resumed with a resized pool without concealing mixed timings.
    scientific_config.pop("pipeline_workers", None)
    return deterministic_hash({
        "schema": PIPELINE_CHECKPOINT_SCHEMA,
        "package_version": __version__,
        "simulator_version": SIMULATOR_VERSION,
        "config": scientific_config,
        "distance": distance,
        "seed": seed,
    })


def _pipeline_checkpoint_path(directory: Path, distance: int, seed: int) -> Path:
    return directory / f"distance-{distance:02d}-seed-{seed}.json"


def _write_pipeline_checkpoint(directory: Path, config: ScalabilityConfig,
                               distance: int, seed: int,
                               result: PipelineConditionResult) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    result_payload = {
        "points": [asdict(row) for row in result.points],
        "failures": [asdict(row) for row in result.failures],
    }
    payload = {
        "schema_version": PIPELINE_CHECKPOINT_SCHEMA,
        "condition_fingerprint": _pipeline_condition_fingerprint(config, distance, seed),
        "distance": distance,
        "seed": seed,
        "worker_concurrency": config.pipeline_workers,
        "result_hash": deterministic_hash(result_payload),
        "result": result_payload,
    }
    path = _pipeline_checkpoint_path(directory, distance, seed)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path


def _read_pipeline_checkpoint(path: Path, config: ScalabilityConfig,
                              distance: int, seed: int) -> PipelineConditionResult:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PipelineCheckpointError(f"cannot read checkpoint {path}: {error}") from error
    expected = _pipeline_condition_fingerprint(config, distance, seed)
    if payload.get("schema_version") != PIPELINE_CHECKPOINT_SCHEMA:
        raise PipelineCheckpointError(f"checkpoint schema mismatch: {path}")
    if payload.get("condition_fingerprint") != expected:
        raise PipelineCheckpointError(
            f"checkpoint configuration/version mismatch for d={distance}, seed={seed}: {path}")
    result_payload = payload.get("result")
    if not isinstance(result_payload, dict) or payload.get("result_hash") != deterministic_hash(result_payload):
        raise PipelineCheckpointError(f"checkpoint payload hash mismatch: {path}")
    try:
        return PipelineConditionResult(
            tuple(PipelineProbePoint(**row) for row in result_payload.get("points", ())),
            tuple(PipelineProbeFailure(**row) for row in result_payload.get("failures", ())),
        )
    except (TypeError, ValueError) as error:
        raise PipelineCheckpointError(f"checkpoint record is invalid: {path}: {error}") from error


def _process_memory_bytes() -> int:
    """Return current resident process memory without allocation tracing."""
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCountersEx(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCountersEx()
            counters.cb = ctypes.sizeof(counters)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCountersEx), wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            handle = kernel32.GetCurrentProcess()
            if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return int(counters.WorkingSetSize)
        except (AttributeError, OSError, ValueError):
            pass
    try:
        statm = Path("/proc/self/statm").read_text(encoding="ascii").split()
        return int(statm[1]) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, AttributeError, IndexError, ValueError):
        return 0


class _ProcessMemorySampler:
    """Low-overhead RSS sampler kept outside the scientifically timed hot path."""

    def __init__(self, interval_s: float = .02) -> None:
        self._interval_s = interval_s
        self._baseline = self._peak = _process_memory_bytes()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="pipeline-memory-sampler",
                                        daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        value = _process_memory_bytes()
        with self._lock:
            self._peak = max(self._peak, value)

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            self._sample()

    def reset(self) -> None:
        with self._lock:
            self._baseline = self._peak = _process_memory_bytes()

    @property
    def baseline(self) -> int:
        with self._lock:
            return self._baseline

    @property
    def peak(self) -> int:
        self._sample()
        with self._lock:
            return self._peak

    @property
    def incremental_peak(self) -> int:
        peak = self.peak
        with self._lock:
            return max(0, peak - self._baseline)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(.1, 5 * self._interval_s))


@dataclass(frozen=True)
class ScalabilityGate:
    gate_id: str
    status: str
    measured: float | None
    required: float | None
    rationale: str


@dataclass(frozen=True)
class ScalabilityReport:
    schema_version: str
    protocol: PaperProtocol
    config: ScalabilityConfig
    scaling: tuple[ScalingPoint, ...]
    convergence: tuple[ConvergencePoint, ...]
    fits: tuple[ConvergenceFit, ...]
    steerability: tuple[SteerabilityPoint, ...]
    resources: tuple[ResourcePoint, ...]
    sample_efficiency: tuple[SampleEfficiencyPoint, ...]
    pipeline_probe: tuple[PipelineProbePoint, ...]
    pipeline_failures: tuple[PipelineProbeFailure, ...]
    gates: tuple[ScalabilityGate, ...]
    limitations: tuple[str, ...]
    environment: Mapping[str, str]
    report_hash: str

    def to_dict(self) -> dict:
        return asdict(self)


def _logspace(start: float, stop: float, count: int) -> tuple[float, ...]:
    if count == 1:
        return (10 ** start,)
    return tuple(10 ** (start + index * (stop-start)/(count-1)) for index in range(count))


def physical_qubits(distance: int) -> int:
    return 2 * distance * distance - 1


def one_qubit_gates(distance: int) -> int:
    return 2 * distance * distance - 1


def two_qubit_gates(distance: int) -> int:
    return 4 * distance * distance - 4 * distance


def paper_control_parameters(distance: int, parameters_per_gate: int) -> int:
    return (one_qubit_gates(distance) + two_qubit_gates(distance)) * parameters_per_gate


def time_reduced_detector_factors(distance: int) -> int:
    """O(d^2) reward-vector proxy anchored to the paper's 97 terms at d=5."""
    return 4 * distance * distance - 3


def normalized_improvement(candidate: float, fixed: float, optimal: float) -> float:
    """Paper/SI definition: 1 is optimal and 0 is fixed; negative is harmful."""
    denominator = optimal - fixed
    return (candidate - fixed) / denominator if abs(denominator) > 1e-15 else math.nan


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def _origin_fit(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    denominator = sum(x*x for x, _ in points)
    if denominator <= 0 or len(points) < 2:
        return math.nan, math.nan
    slope = sum(x*y for x, y in points) / denominator
    mean_y = _mean([y for _, y in points])
    residual = sum((y - slope*x) ** 2 for x, y in points)
    total = sum((y - mean_y) ** 2 for _, y in points)
    return slope, 1 - residual / total if total > 0 else 1.


def _run_pipeline_condition(payload: tuple[ScalabilityConfig, int, int]
                            ) -> PipelineConditionResult:
    """Run one independent distance/seed probe; safe for spawned worker processes."""
    config, distance, seed = payload
    from hdfa_rl_suite.baselines import FullControlRLArm, PredictiveHDFARLArm
    from hdfa_rl_suite.product import QECOperabilityError
    from hdfa_rl_suite.simulator import DriftKind, LatentProcessSpec, ScalableQECDevice, SimulatorConfig
    from hdfa_rl_suite.stage0 import ScalableBootstrapCalibrator, ScalableBootstrapConfig
    from hdfa_rl_suite.stage0.schema import HealthStatus

    qubits = physical_qubits(distance)
    loadings = {f"drive:q{index}": 1. for index in range(qubits)}
    process = LatentProcessSpec("shared-sinusoid", DriftKind.SINUSOID, loadings,
                                amplitude=.15, period_s=max(.1, config.pipeline_epochs
                                * config.pipeline_cycles_per_interval * .001))
    output: list[PipelineProbePoint] = []
    failures: list[PipelineProbeFailure] = []
    methods = (
        (PAPER_METHOD, lambda: FullControlRLArm(
            seed=seed, candidate_count=config.pipeline_candidates,
            candidate_cycles=config.pipeline_candidate_cycles,
            per_candidate_damage_budget=max(.25, (2*qubits-1) * .0025),
            damage_budget=max(10., config.pipeline_epochs * (2*qubits-1) * .01))),
        (HDFA_METHOD, lambda: PredictiveHDFARLArm(
            seed=seed, residual=True, candidate_count=config.pipeline_candidates,
            candidate_cycles=config.pipeline_candidate_cycles)),
    )
    sampler = _ProcessMemorySampler()
    bootstrap_replay_hash = ""
    try:
        # Stage 0 and the held-out baseline are pre-randomization evidence.  Execute
        # them once, then clone that immutable starting state for both counterfactual
        # controller arms.  Scientific resource accounting remains attached to each
        # arm, while actual experiment wall time does not pay for duplicate evidence.
        device = ScalableQECDevice(SimulatorConfig(
            qubit_count=qubits, code_distance=distance, cycle_period_s=.001,
            controller_latency_s=.002, disturbances_enabled_at_start=False,
            seed=seed, processes=(process,),
        ))
        sampler.reset()
        bootstrap_memory_baseline = sampler.baseline
        bootstrap_started_device = device.now_s
        bootstrap_started_wall = time.perf_counter()
        bootstrap = ScalableBootstrapCalibrator(device, ScalableBootstrapConfig(
            characterization_shots=config.pipeline_bootstrap_characterization_shots,
            validation_cycles=config.pipeline_bootstrap_validation_cycles,
            target_posterior_stddev=config.pipeline_bootstrap_target_stddev,
            qec_detector_rate_limit=config.pipeline_bootstrap_qec_rate_limit,
            block_predictive_familywise_alpha=config.pipeline_bootstrap_block_familywise_alpha,
            sensitivity_max_batch_size=config.pipeline_bootstrap_sensitivity_max_batch_size,
            sensitivity_interference_alpha=config.pipeline_bootstrap_sensitivity_interference_alpha,
        )).run()
        bootstrap_wall = time.perf_counter() - bootstrap_started_wall
        bootstrap_memory = sampler.peak
        bootstrap_incremental_memory = sampler.incremental_peak
        bootstrap_device = device.now_s - bootstrap_started_device
        bootstrap_replay_hash = bootstrap.replay_hash
        bootstrap_shots = sum(
            int(estimate.diagnostics.get("active_design_shots", 0))
            + int(estimate.diagnostics.get("held_out_shots", 0))
            for estimate in bootstrap.calibration_estimates.values())
        bootstrap_qec = sum(
            int(estimate.diagnostics.get("qec_cycles", 0))
            for estimate in bootstrap.calibration_estimates.values())
        if bootstrap.health.status is not HealthStatus.PASSED:
            for method, _ in methods:
                failures.append(PipelineProbeFailure(
                    "executed_suite_pipeline", method, seed, distance, qubits, None,
                    "stage0", "censored", "QECOperabilityError",
                    f"Stage-0 QEC-operability gate failed: {bootstrap.health.invalid_reasons}",
                    device.disturbance_realization_id, None, bootstrap_replay_hash))
            sampler.close()
            return PipelineConditionResult(tuple(output), tuple(failures))
        baseline = device.acquire(config.pipeline_baseline_cycles)
        baseline_hash = deterministic_hash(baseline)
        bootstrap_execution_id = deterministic_hash({
            "distance": distance,
            "seed": seed,
            "bootstrap_replay_hash": bootstrap_replay_hash,
            "baseline_hash": baseline_hash,
        })
        pre_randomization_device = copy.deepcopy(device)
    except Exception as error:
        disturbance_id = (device.disturbance_realization_id
                          if "device" in locals() else "")
        disturbance_epoch = (device.disturbance_epoch_s if "device" in locals() else None)
        for method, _ in methods:
            failures.append(PipelineProbeFailure(
                "executed_suite_pipeline", method, seed, distance, qubits, None,
                "pre_randomization", "missing", type(error).__name__, str(error),
                disturbance_id, disturbance_epoch, bootstrap_replay_hash))
        sampler.close()
        return PipelineConditionResult(tuple(output), tuple(failures))

    for method, factory in methods:
        device = copy.deepcopy(pre_randomization_device)
        arm = factory()
        try:
            prepare = getattr(arm, "prepare", None)
            if callable(prepare):
                prepare(device, bootstrap)
            disturbance_epoch = device.arm_disturbances()
            for epoch in range(config.pipeline_epochs):
                sampler.reset()
                memory_baseline = sampler.baseline
                started = time.perf_counter()
                try:
                    result = arm.run_interval(device, config.pipeline_cycles_per_interval, epoch)
                except QECOperabilityError as error:
                    failures.append(PipelineProbeFailure(
                        "executed_suite_pipeline", method, seed, distance, qubits, epoch,
                        "online_reentry", "censored", type(error).__name__, str(error),
                        device.disturbance_realization_id, disturbance_epoch,
                        bootstrap_replay_hash))
                    break
                except Exception as error:
                    failures.append(PipelineProbeFailure(
                        "executed_suite_pipeline", method, seed, distance, qubits, epoch,
                        "online_interval", "missing", type(error).__name__, str(error),
                        device.disturbance_realization_id, disturbance_epoch,
                        bootstrap_replay_hash))
                    break
                elapsed = time.perf_counter()-started
                peak = sampler.peak
                incremental_peak = sampler.incremental_peak
                output.append(PipelineProbePoint(
                    "executed_suite_pipeline", method, seed, distance, qubits,
                    len(device.limits.controls), paper_control_parameters(distance, 30), epoch,
                    result.observation.detector_rate,
                    result.observation.logical_failures / max(1, result.observation.cycles),
                    result.candidate_evaluations, result.total_qec_cycles, elapsed,
                    memory_baseline, peak, incremental_peak,
                    "sampled_process_resident_set",
                    config.pipeline_workers,
                    ("fresh_process_per_condition" if config.pipeline_workers > 1
                     else "in_process_sequential"),
                    bootstrap_shots if epoch == 0 else 0,
                    bootstrap_qec if epoch == 0 else 0,
                    bootstrap_device if epoch == 0 else 0.0,
                    bootstrap_wall if epoch == 0 else 0.0,
                    bootstrap_memory_baseline if epoch == 0 else 0,
                    bootstrap_memory if epoch == 0 else 0,
                    bootstrap_incremental_memory if epoch == 0 else 0,
                    bootstrap_execution_id,
                    1 / len(methods) if epoch == 0 else 0.0,
                    baseline.cycles if epoch == 0 else 0,
                    baseline.detector_rate, baseline_hash,
                    device.disturbance_realization_id, disturbance_epoch,
                    len(result.lifecycle_violations),
                ))
                if result.lifecycle_violations:
                    failures.append(PipelineProbeFailure(
                        "executed_suite_pipeline", method, seed, distance, qubits, epoch,
                        "lifecycle", "censored", "LifecycleViolation",
                        "; ".join(result.lifecycle_violations),
                        device.disturbance_realization_id, disturbance_epoch,
                        bootstrap_replay_hash))
                    break
        except Exception as error:
            failures.append(PipelineProbeFailure(
                "executed_suite_pipeline", method, seed, distance, qubits, None,
                "pre_randomization", "missing", type(error).__name__, str(error),
                device.disturbance_realization_id, device.disturbance_epoch_s,
                bootstrap_replay_hash))
    sampler.close()
    return PipelineConditionResult(tuple(output), tuple(failures))


class ScalabilityRunner:
    """Execute Figure-5-matched surrogate sweeps and optional real stack probes."""

    def __init__(self, config: ScalabilityConfig = ScalabilityConfig(),
                 protocol: PaperProtocol = PaperProtocol(), *,
                 checkpoint_directory: Path | None = None,
                 resume: bool = False) -> None:
        self.config, self.protocol = config, protocol
        self.checkpoint_directory = checkpoint_directory
        self.resume = resume
        if resume and checkpoint_directory is None:
            raise ValueError("resume requires a checkpoint directory")

    def _cycles_per_epoch(self, method: str) -> int:
        if method == PAPER_METHOD:
            return self.config.paper_candidates_per_epoch * self.config.paper_cycles_per_candidate
        return self.config.hdfa_candidates_per_epoch * self.config.hdfa_cycles_per_candidate

    def _gain(self, method: str, parameters_per_gate: int) -> float:
        if method == PAPER_METHOD:
            effective = float(parameters_per_gate)
            information = 1.
        else:
            effective = max(1., parameters_per_gate * (1-self.config.hdfa_modeled_fraction))
            information = math.sqrt(self._cycles_per_epoch(method) / self._cycles_per_epoch(PAPER_METHOD))
        return self.config.reference_learning_rate * information / math.sqrt(effective)

    def _static_trace(self, method: str, distance: int, parameters_per_gate: int,
                      seed: int) -> tuple[ScalingPoint, ...]:
        condition_seed = seed * 1_000_003 + distance * 10_007 + parameters_per_gate * 101
        rng = random.Random(condition_seed)
        irreducible = self.config.irreducible_physical_error * (1 + rng.uniform(-.04, .04))
        initial = self.config.initial_physical_error * (1 + rng.uniform(-.04, .04))
        excess = max(1e-12, initial - irreducible)
        gain = self._gain(method, parameters_per_gate)
        controls = paper_control_parameters(distance, parameters_per_gate)
        cycles = self._cycles_per_epoch(method)
        output: list[ScalingPoint] = []
        for epoch in range(self.config.epochs + 1):
            physical = irreducible + excess
            exponent = (distance + 1) / 2
            logical = min(.499, self.config.logical_error_at_threshold
                          * (physical / self.config.physical_error_threshold) ** exponent)
            output.append(ScalingPoint(
                "declared_factor_graph_surrogate", method, seed, distance,
                parameters_per_gate, physical_qubits(distance), controls, epoch,
                epoch * cycles, physical, logical, irreducible / physical,
            ))
            if epoch == self.config.epochs:
                continue
            if method == HDFA_METHOD and epoch == 0:
                # Stage 2--5 feedforward removes the explicitly modelled component; the
                # remaining fraction is left to detector-driven residual RL.
                excess *= 1-self.config.hdfa_modeled_fraction
            local_noise = rng.gauss(0., self.config.static_update_noise
                                    / math.sqrt(time_reduced_detector_factors(distance)))
            excess *= math.exp(-max(0., gain * (1 + local_noise)))
        return tuple(output)

    def _static(self) -> tuple[ScalingPoint, ...]:
        return tuple(point for method in (PAPER_METHOD, HDFA_METHOD)
                     for distance in self.config.distances
                     for parameters in self.config.parameters_per_gate
                     for seed in self.config.seeds
                     for point in self._static_trace(method, distance, parameters, seed))

    def _convergence(self, scaling: Sequence[ScalingPoint]) -> tuple[tuple[ConvergencePoint, ...], tuple[ConvergenceFit, ...]]:
        grouped: dict[tuple[str, int, int, int], list[ScalingPoint]] = {}
        for point in scaling:
            grouped.setdefault((point.method, point.seed, point.code_distance,
                                point.parameters_per_gate), []).append(point)
        convergence: list[ConvergencePoint] = []
        for (method, seed, distance, parameters), rows in grouped.items():
            ordered = sorted(rows, key=lambda item: item.epoch)
            for left, right in zip(ordered, ordered[1:]):
                convergence.append(ConvergencePoint(
                    method, seed, distance, parameters, left.control_parameters, left.epoch,
                    1-left.lambda_ratio, right.lambda_ratio-left.lambda_ratio,
                ))
        fit_groups: dict[tuple[str, int, int], list[ConvergencePoint]] = {}
        for point in convergence:
            if point.epoch >= 2 and point.distance_to_local_optimum > 1e-9:
                fit_groups.setdefault((point.method, point.code_distance,
                                       point.parameters_per_gate), []).append(point)
        fits = []
        for (method, distance, parameters), rows in fit_groups.items():
            gamma, r2 = _origin_fit([(row.distance_to_local_optimum, row.normalized_speed)
                                     for row in rows])
            fits.append(ConvergenceFit(method, distance, parameters,
                paper_control_parameters(distance, parameters), gamma, r2, len(rows)))
        return tuple(convergence), tuple(fits)

    def _steering_condition(self, method: str, frequency: float, entropy: float,
                            seed: int) -> SteerabilityPoint:
        rng = random.Random(seed * 1_000_003 + round(frequency * 1e9) * 101
                            + round(entropy * 1e9) * 17 + int(method == HDFA_METHOD))
        mean_policy = velocity = 0.
        fixed_cost = optimal_cost = stochastic_cost = learned_cost = 0.
        nominal_gain = 1 / self.protocol.empirical_response_time_epochs
        support = entropy / (entropy + 1e-3)
        reference_support = 1e-2 / (1e-2 + 1e-3)
        gain = nominal_gain * support / reference_support
        residual_fraction = 1-self.config.hdfa_modeled_fraction if method == HDFA_METHOD else 1.
        for epoch in range(self.config.steering_epochs):
            optimum = math.sin(2 * math.pi * frequency * epoch)
            if method == PAPER_METHOD:
                mean_policy += gain * (optimum-mean_policy)
            else:
                prediction = mean_policy + velocity
                innovation = optimum-prediction
                alpha = min(.35, gain * (1 + 3*self.config.hdfa_modeled_fraction))
                mean_policy = prediction + alpha * innovation
                velocity = .92 * velocity + .08 * alpha * innovation
            mean_policy += rng.gauss(0., 2e-4 / math.sqrt(max(support, 1e-3)))
            fixed_loss = (0.-optimum) ** 2
            learned_loss = (mean_policy-optimum) ** 2
            exploration_loss = self.config.exploration_noise_scale * entropy * residual_fraction
            fixed_cost += fixed_loss
            optimal_cost += 0.
            learned_cost += learned_loss
            stochastic_cost += learned_loss + exploration_loss
        cycles = self.config.steering_epochs * self._cycles_per_epoch(method)
        return SteerabilityPoint(
            "declared_first_order_tracking_surrogate", method, seed, frequency, entropy,
            normalized_improvement(stochastic_cost, fixed_cost, optimal_cost),
            normalized_improvement(learned_cost, fixed_cost, optimal_cost),
            fixed_cost, optimal_cost, stochastic_cost, learned_cost, cycles,
        )

    def _steerability(self) -> tuple[SteerabilityPoint, ...]:
        return tuple(self._steering_condition(method, frequency, entropy, seed)
                     for method in (PAPER_METHOD, HDFA_METHOD)
                     for frequency in self.config.steering_frequencies
                     for entropy in self.config.entropy_regularizations
                     for seed in self.config.seeds)

    def _resources(self) -> tuple[ResourcePoint, ...]:
        output = []
        for method in (PAPER_METHOD, HDFA_METHOD):
            for distance in self.config.distances:
                detectors = time_reduced_detector_factors(distance)
                for parameters in self.config.parameters_per_gate:
                    controls = paper_control_parameters(distance, parameters)
                    # The d=5 paper factor graph reports about 302 parameter neighbours per
                    # detector.  Scaling this local degree with P retains O(d^2 P) edges.
                    edges = round(detectors * (self.protocol.distance_5_mean_parameters_per_detector
                                               / 30.) * parameters)
                    if method == PAPER_METHOD:
                        candidates = self.config.paper_candidates_per_epoch
                        cycles = self._cycles_per_epoch(method)
                        state_bytes = 8 * (2*controls + 2*edges)
                    else:
                        candidates = self.config.hdfa_candidates_per_epoch
                        cycles = self._cycles_per_epoch(method)
                        residual_controls = math.ceil(controls * (1-self.config.hdfa_modeled_fraction))
                        particle_slots = detectors * (192 + 256)
                        state_bytes = 8 * (2*residual_controls + 2*round(edges * (1-self.config.hdfa_modeled_fraction))
                                           + particle_slots)
                    output.append(ResourcePoint(
                        "structural_count", method, distance, parameters, physical_qubits(distance),
                        one_qubit_gates(distance), two_qubit_gates(distance), detectors, controls,
                        edges, candidates, cycles, state_bytes,
                    ))
        return tuple(output)

    @staticmethod
    def _sample_efficiency(scaling: Sequence[ScalingPoint],
                           target_fractions: Sequence[float] = (.50, .75, .90)) -> tuple[SampleEfficiencyPoint, ...]:
        grouped: dict[tuple[str, int, int, int], list[ScalingPoint]] = {}
        for row in scaling:
            grouped.setdefault((row.method, row.seed, row.code_distance,
                                row.parameters_per_gate), []).append(row)
        output = []
        for (method, seed, distance, parameters), rows in grouped.items():
            ordered = sorted(rows, key=lambda item: item.epoch)
            start = ordered[0].lambda_ratio
            for target_fraction in target_fractions:
                target = start + target_fraction * (1-start)
                achieved = next((row for row in ordered if row.lambda_ratio >= target), None)
                output.append(SampleEfficiencyPoint(
                    method, seed, distance, parameters, target_fraction, achieved is not None,
                    achieved.epoch if achieved else None,
                    achieved.cumulative_qec_cycles if achieved else None,
                ))
        return tuple(output)

    def _pipeline_probe(self) -> tuple[tuple[PipelineProbePoint, ...], tuple[PipelineProbeFailure, ...]]:
        if not self.config.run_pipeline_probe:
            return (), ()
        conditions = [(self.config, distance, seed) for distance in self.config.pipeline_distances
                      for seed in self.config.seeds]
        chunks: list[PipelineConditionResult | None] = [None] * len(conditions)
        pending: list[tuple[int, tuple[ScalabilityConfig, int, int]]] = []
        for index, condition in enumerate(conditions):
            _, distance, seed = condition
            checkpoint = (_pipeline_checkpoint_path(self.checkpoint_directory, distance, seed)
                          if self.checkpoint_directory is not None else None)
            if self.resume and checkpoint is not None and checkpoint.exists():
                restored = _read_pipeline_checkpoint(
                    checkpoint, self.config, distance, seed)
                # A missing worker record is not completed evidence; retry it on resume.
                if not any(failure.status == "missing" for failure in restored.failures):
                    chunks[index] = restored
                    continue
            pending.append((index, condition))
        if self.config.pipeline_workers == 1:
            for index, condition in pending:
                result = _run_pipeline_condition(condition)
                chunks[index] = result
                if self.checkpoint_directory is not None:
                    _, distance, seed = condition
                    _write_pipeline_checkpoint(
                        self.checkpoint_directory, self.config, distance, seed, result)
        elif pending:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            # Schedule the largest independent conditions first, then restore canonical
            # distance/seed order so artifact layout is unchanged by worker count.
            scheduled = sorted(pending, key=lambda item: (-item[1][1], item[1][2]))
            # A fresh process for every condition prevents Python allocator/RSS state
            # retained by a previous (usually larger) condition from contaminating
            # the next condition's absolute process-memory measurement.
            with ProcessPoolExecutor(
                    max_workers=min(self.config.pipeline_workers, len(pending)),
                    max_tasks_per_child=1) as pool:
                futures = {pool.submit(_run_pipeline_condition, condition): index
                           for index, condition in scheduled}
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        chunks[index] = future.result()
                    except Exception as error:
                        _, distance, seed = conditions[index]
                        chunks[index] = PipelineConditionResult((), (PipelineProbeFailure(
                            "executed_suite_pipeline", "worker_process", seed, distance,
                            physical_qubits(distance), None, "worker_process", "missing",
                            type(error).__name__, str(error), "", None, ""),))
                    if self.checkpoint_directory is not None:
                        _, distance, seed = conditions[index]
                        _write_pipeline_checkpoint(
                            self.checkpoint_directory, self.config, distance, seed,
                            chunks[index])
        completed = [chunk for chunk in chunks if chunk is not None]
        points = tuple(row for chunk in completed for row in chunk.points)
        failures = tuple(row for chunk in completed for row in chunk.failures)
        return points, failures

    def _gates(self, fits: Sequence[ConvergenceFit], steering: Sequence[SteerabilityPoint],
               resources: Sequence[ResourcePoint],
               efficiency: Sequence[SampleEfficiencyPoint],
               pipeline: Sequence[PipelineProbePoint],
               pipeline_failures: Sequence[PipelineProbeFailure]) -> tuple[ScalabilityGate, ...]:
        gates = [
            ScalabilityGate("distance_15_parameter_count", "pass" if paper_control_parameters(15, 30) == 38_670 else "fail",
                            float(paper_control_parameters(15, 30)), 38_670., "Supplementary Eq. 7 anchor"),
            ScalabilityGate("hundreds_of_qubits", "pass" if max(physical_qubits(d) for d in self.config.distances) >= 100 else "not_evaluable",
                            float(max(physical_qubits(d) for d in self.config.distances)), 100.,
                            "rotated-surface-code physical-qubit count 2d^2-1"),
        ]
        for method in (PAPER_METHOD, HDFA_METHOD):
            for parameters in self.config.parameters_per_gate:
                values = [fit.gamma for fit in fits if fit.method == method
                          and fit.parameters_per_gate == parameters and math.isfinite(fit.gamma)]
                cv = statistics.pstdev(values) / abs(_mean(values)) if len(values) > 1 and _mean(values) else 0.
                gates.append(ScalabilityGate(
                    f"distance_independent_gamma:{method}:p{parameters}",
                    "pass" if len(values) > 1 and cv <= .10 else ("not_evaluable" if len(values) <= 1 else "fail"),
                    cv if values else None, .10,
                    "coefficient of variation of fitted gamma across code distance",
                ))
        baseline = [row for row in steering if row.method == PAPER_METHOD]
        aggregate: dict[tuple[float, float], list[float]] = {}
        for row in baseline:
            aggregate.setdefault((row.drift_frequency, row.entropy_regularization), []).append(row.stochastic_improvement)
        best_by_frequency: dict[float, float] = {}
        for (frequency, _), values in aggregate.items():
            best_by_frequency[frequency] = max(best_by_frequency.get(frequency, -math.inf), _mean(values))
        # A two-percent normalized advantage prevents floating-point/noise-scale changes
        # around the r=0 isoline from being misreported as a usable steering regime.
        steerable = [frequency for frequency, advantage in best_by_frequency.items() if advantage >= .02]
        measured = max(steerable, default=None)
        gates.append(ScalabilityGate(
            "published_steerability_anchor", "contrast_only", measured,
            self.protocol.empirical_steerability_frequency,
            "surrogate maximum scanned frequency with >=2% mean stochastic improvement at some entropy; not an equivalence gate",
        ))
        paired: dict[tuple[int, int, int], dict[str, int]] = {}
        for row in efficiency:
            if row.target_fraction == .75 and row.achieved and row.cumulative_qec_cycles is not None:
                paired.setdefault((row.seed, row.code_distance, row.parameters_per_gate), {})[row.method] = row.cumulative_qec_cycles
        ratios = [values[PAPER_METHOD] / values[HDFA_METHOD] for values in paired.values()
                  if PAPER_METHOD in values and HDFA_METHOD in values and values[HDFA_METHOD] > 0]
        ratio = _mean(ratios) if ratios else None
        gates.append(ScalabilityGate(
            "native_qec_cycles_to_75pct_local_optimum", "contrast_only", ratio, 1.,
            "mean paired paper-method/HDFA native-QEC cycle ratio at 75% local-optimum progress; 50/75/90% rows are retained",
        ))
        if self.config.run_pipeline_probe:
            expected = (len(self.config.pipeline_distances) * len(self.config.seeds)
                        * 2 * self.config.pipeline_epochs)
            fraction = len(pipeline) / max(1, expected)
            gates.append(ScalabilityGate(
                "executed_pipeline_completion",
                "pass" if len(pipeline) == expected and not pipeline_failures else "fail",
                fraction, 1.0,
                "fraction of predeclared method/distance/seed/epoch probes completed after common stationary Stage 0 and synchronized onset",
            ))
        return tuple(gates)

    def run(self) -> ScalabilityReport:
        scaling = self._static()
        convergence, fits = self._convergence(scaling)
        steering = self._steerability()
        resources = self._resources()
        efficiency = self._sample_efficiency(scaling)
        pipeline, pipeline_failures = self._pipeline_probe()
        gates = self._gates(fits, steering, resources, efficiency, pipeline, pipeline_failures)
        limitations_list = [
            "The Figure 5 simulation code and hyperparameters are proprietary; surrogate results are not published source data.",
            "The public 7.8 GB Zenodo archive contains experimental surface/color-code records and is referenced, not bundled.",
            "The reduced surrogate uses the published quadratic gate-error and logical-scaling equations but not a Stim circuit sampler.",
            "Pipeline probes use the suite's sparse line-graph device (2Q-1 controls), so paper-equivalent P=30 counts are reported separately.",
            "Executed methods receive one shared stationary Stage-0 calibration and held-out QEC baseline per distance/seed, followed by cloned counterfactual device states and synchronized disturbance onset; full resource costs and actual execution shares are both reported.",
            "Wall time is measured without Python allocation tracing; memory is sampled independently from process resident-set size. Parallel probes isolate every distance/seed condition in a fresh worker process. Both are implementation diagnostics, not QPU latency estimates.",
        ]
        worker_contexts = sorted({row.worker_concurrency for row in pipeline})
        if len(worker_contexts) > 1:
            limitations_list.append(
                "Resumed checkpoints contain heterogeneous worker-concurrency contexts; each row records its context and latency comparisons must be stratified by that field.")
        limitations = tuple(limitations_list)
        environment = {
            "python": platform.python_version(), "platform": platform.platform(),
            "implementation": platform.python_implementation(),
            "pipeline_worker_contexts": ",".join(str(value) for value in worker_contexts),
            "checkpoint_resume": str(self.resume).lower(),
        }
        payload = {
            "protocol": asdict(self.protocol), "config": asdict(self.config),
            "scaling": [asdict(row) for row in scaling],
            "convergence": [asdict(row) for row in convergence],
            "fits": [asdict(row) for row in fits],
            "steerability": [asdict(row) for row in steering],
            "resources": [asdict(row) for row in resources],
            "sample_efficiency": [asdict(row) for row in efficiency],
            "pipeline_probe": [asdict(row) for row in pipeline],
            "pipeline_failures": [asdict(row) for row in pipeline_failures],
            "gates": [asdict(row) for row in gates], "limitations": limitations,
            "environment": environment,
        }
        return ScalabilityReport(
            "evaluation.scalability.v3", self.protocol, self.config, scaling, convergence,
            fits, steering, resources, efficiency, pipeline, pipeline_failures,
            gates, limitations, environment,
            deterministic_hash(payload),
        )


def with_pipeline_probe(config: ScalabilityConfig, *, maximum_distance: int | None = None) -> ScalabilityConfig:
    distances = tuple(distance for distance in config.pipeline_distances
                      if maximum_distance is None or distance <= maximum_distance)
    return replace(config, run_pipeline_probe=True, pipeline_distances=distances)
