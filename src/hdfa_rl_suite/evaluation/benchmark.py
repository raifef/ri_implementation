"""Authoritative matched-trajectory acceptance benchmark.

Every arm first receives the same stationary Stage-0 protocol and native-QEC baseline,
then the same fixed-time disturbance realization is armed at a declared phase boundary.
Runs retain interval trajectories, explicit censoring, circuit-level logical evidence,
paired statistics, confidence intervals, and complete source/simulator provenance.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time
from typing import Callable, Mapping, Sequence

from hdfa_rl_suite import __version__
from hdfa_rl_suite.baselines import (
    FixedCalibrationArm,
    FullControlRLArm,
    GreedyCalibrationArm,
    OracleControlArm,
    PeriodicRecalibrationArm,
    PhysicalInferenceArm,
    PredictiveHDFARLArm,
)
from hdfa_rl_suite.baselines.controllers import ArmIntervalResult, BenchmarkArm
from hdfa_rl_suite.common import (
    OnlineTimingBreakdown, TimingEnvironment, deterministic_hash,
)
from hdfa_rl_suite.logical import (
    LogicalPerformanceEvidence,
    LogicalStackUnavailable,
    RotatedSurfaceCodeEvaluator,
    SurfaceCodeMemoryConfig,
)
from hdfa_rl_suite.product import (ProductLoopConfig, QECOperabilityError,
                                   RecoveryCertificationError)
from hdfa_rl_suite.simulator import (
    SIMULATOR_VERSION,
    DriftKind,
    LatentProcessSpec,
    ScalableQECDevice,
    SimulatorConfig,
)
from hdfa_rl_suite.stage0 import ScalableBootstrapCalibrator, ScalableBootstrapConfig
from hdfa_rl_suite.stage0.schema import BootstrapResult, HealthStatus

from .evidence import (EvidenceRecord, canonical_benchmark_evidence,
                       validate_report_payload)


PRIMARY_ARMS = (
    "full_control_detector_rl",
    "predictive_hdfa_no_residual",
    "predictive_hdfa_residual_rl",
    "fixed",
    "periodic_recalibration",
    "oracle",
)
RECOVERY_TARGETS = (0.50, 0.75, 0.90)


@dataclass(frozen=True)
class BenchmarkScenario:
    scenario_id: str
    processes: tuple[LatentProcessSpec, ...]
    structured: bool
    description: str


@dataclass(frozen=True)
class BenchmarkConfig:
    qubit_count: int = 5
    intervals: int = 8
    cycles_per_interval: int = 128
    seeds: tuple[int, ...] = (3, 11)
    code_distance: int = 3
    candidate_cycles: int = 2048
    steady_state_intervals: int = 2
    cycle_period_s: float = 0.001
    censoring_limit_intervals: int | None = None
    logical_shots_per_interval: int = 256
    logical_rounds: int = 3
    bootstrap_characterization_shots: int = 96
    bootstrap_validation_cycles: int = 128
    bootstrap_target_stddev: float = 0.06
    bootstrap_qec_rate_limit: float = 0.15
    bootstrap_block_familywise_alpha: float = 1e-4
    pre_disturbance_baseline_cycles: int = 128
    minimum_fit_r2: float = 0.80
    maximum_fit_residual_autocorrelation: float = 0.50
    maximum_gamma_relative_standard_error: float = 1.0
    final_rate_noninferiority_margin: float = 0.005
    e2e_rmst_horizon_s: float | None = None
    minimum_compute_independent_seeds: int = 2
    compute_bootstrap_replicates: int = 2000
    compute_bootstrap_seed: int = 20260802
    compute_one_sided_confidence: float = 0.95
    e2e_tail_quantile: float = 0.95
    e2e_tail_noninferiority_margin_s: float = 0.25
    minimum_e2e_followup_support_s: float | None = None
    rmst_support_margin_s: float = 0.5
    endpoint_followup_chunk_cycles: int = 8192
    estimator_schema_version: str = "estimators.v2"
    gate_reference_arm: str = "full_control_detector_rl"
    gate_treatment_arm: str = "predictive_hdfa_residual_rl"
    integrated_excess_required_ratio: float = 5.0
    exploration_damage_required_ratio: float = 2.0
    one_interval_required_fraction: float = 0.90
    extended_structured_models: bool = False
    parallel_regional_updates: bool = False
    logical_failure_noninferiority_margin: float | None = None
    residual_benefit_scenario_id: str | None = None
    candidate_elimination_z: float | None = None
    authoritative: bool = True

    def __post_init__(self) -> None:
        if self.qubit_count < 2 or self.intervals <= 0 or self.cycles_per_interval <= 0:
            raise ValueError("benchmark dimensions must be positive")
        if not self.seeds:
            raise ValueError("at least one paired seed is required")
        if self.censoring_limit_intervals is not None and self.censoring_limit_intervals <= 0:
            raise ValueError("censoring limit must be positive")
        if self.pre_disturbance_baseline_cycles <= 0:
            raise ValueError("pre-disturbance baseline cycles must be positive")
        if not 0 < self.bootstrap_block_familywise_alpha < 1:
            raise ValueError("bootstrap block family-wise alpha must lie in (0, 1)")
        if self.e2e_rmst_horizon_s is not None and self.e2e_rmst_horizon_s <= 0:
            raise ValueError("the E2E RMST horizon must be positive when declared")
        if self.minimum_compute_independent_seeds < 2:
            raise ValueError("compute-aware confidence requires at least two seeds")
        if self.compute_bootstrap_replicates < 100:
            raise ValueError("cluster bootstrap requires at least 100 replicates")
        if not 0.5 < self.compute_one_sided_confidence < 1:
            raise ValueError("one-sided compute confidence must lie in (0.5,1)")
        if not 0 < self.e2e_tail_quantile < 1:
            raise ValueError("tail quantile must lie in (0,1)")
        if self.e2e_tail_noninferiority_margin_s < 0:
            raise ValueError("tail noninferiority margin cannot be negative")
        if (self.minimum_e2e_followup_support_s is not None
                and self.minimum_e2e_followup_support_s <= 0):
            raise ValueError("minimum follow-up support must be positive")
        if self.rmst_support_margin_s < 0 or self.endpoint_followup_chunk_cycles <= 0:
            raise ValueError("RMST support margin/chunk must be non-negative/positive")
        if self.estimator_schema_version not in {"legacy.v1", "estimators.v2"}:
            raise ValueError("unsupported estimator schema")
        if self.gate_reference_arm not in PRIMARY_ARMS or self.gate_treatment_arm not in PRIMARY_ARMS:
            raise ValueError("gate arms must be registered benchmark arms")
        if (self.integrated_excess_required_ratio < 0
                or self.exploration_damage_required_ratio < 0
                or not 0 < self.one_interval_required_fraction <= 1):
            raise ValueError("gate thresholds must be non-negative and recovery fraction valid")
        if (self.logical_failure_noninferiority_margin is not None
                and self.logical_failure_noninferiority_margin < 0):
            raise ValueError("logical noninferiority margin cannot be negative")
        if self.candidate_elimination_z is not None and self.candidate_elimination_z <= 0:
            raise ValueError("candidate-elimination confidence threshold must be positive")


class BenchmarkPreflightError(RuntimeError):
    """Raised before acquisition when the scientific launch manifest is absent/stale."""


@dataclass(frozen=True)
class ConfidenceInterval:
    lower: float
    estimate: float
    upper: float
    confidence: float = 0.95


@dataclass(frozen=True)
class RecoveryEndpoint:
    target_fraction: float
    status: str
    detector_cycles: int | None
    candidate_evaluations: int | None
    intervals_after_peak: int | None
    censoring_cycles: int
    censoring_candidate_evaluations: int
    threshold_rate: float | None
    reason: str
    e2e_time_s: float | None = None
    censoring_e2e_time_s: float | None = None
    host_control_wall_time_s: float | None = None
    censoring_host_control_wall_time_s: float | None = None
    e2e_components_s: Mapping[str, float] = field(default_factory=dict)
    timing_status: str = "not_instrumented"


@dataclass(frozen=True)
class ExponentialRecoveryFit:
    gamma: float | None
    gamma_standard_error: float | None
    r_squared: float | None
    residual_mean: float | None
    residual_autocorrelation_lag1: float | None
    credible: bool
    reason: str


@dataclass(frozen=True)
class PreDisturbanceBaseline:
    """Held-out native-QEC evidence acquired after Stage 0 and before onset."""

    scenario_id: str
    seed: int
    arm: str
    cycles: int
    started_at_s: float
    ended_at_s: float
    detector_events: int
    detector_exposures: int
    detector_rate: float
    detector_rate_ci95: tuple[float, float]
    detector_counts: Mapping[str, tuple[int, int]]
    generic_logical_proxy_failures: int
    policy_hash: str
    batch_id: str
    observation_hash: str
    disturbances_armed_during_acquisition: bool
    disturbance_epoch_s: float
    disturbance_realization_id: str
    evaluation_only_max_abs_latent: float
    evaluation_only_max_abs_process: float
    initial_physical_state_id: str = ""
    initial_disturbance_state_id: str = ""
    initial_simulator_state_hash: str = ""
    initial_controller_state_hash: str = ""
    process_rng_state_hash: str = ""
    characterization_rng_state_hash: str = ""
    detector_evaluator_config_hash: str = ""
    logical_evaluator_config_hash: str = ""
    initial_policy_id: str = ""
    initial_policy_controls: Mapping[str, float] = field(default_factory=dict)
    clone_isolation_verified: bool = False


@dataclass(frozen=True)
class IntervalTrajectory:
    scenario_id: str
    seed: int
    arm: str
    interval: int
    elapsed_time_s: float
    detector_events: int
    detector_exposures: int
    detector_rate: float | None
    detector_counts: Mapping[str, tuple[int, int]]
    auxiliary_detector_events: int
    auxiliary_detector_exposures: int
    qec_cycles: int
    candidate_evaluations: int
    candidate_cycles: int
    diagnostic_shots: int
    diagnostic_downtime_s: float
    exploration_damage: float
    generic_logical_proxy_failures: int
    logical_evidence: LogicalPerformanceEvidence | None
    policy_hash: str
    policy_controls: Mapping[str, float]
    lifecycle_mode: str
    authorization: str
    lifecycle_violations: tuple[str, ...]
    bootstrap_reason: str | None
    bootstrap_count: int
    stage_path: tuple[str, ...]
    replay_hash: str
    bootstrap_evidence: Mapping[str, object] | None
    candidate_trajectories: tuple[Mapping[str, object], ...]
    stage_evidence: Mapping[str, object] | None
    disturbance_realization_id: str
    disturbance_epoch_s: float
    disturbance_elapsed_s: float
    controller_truth_accesses: tuple[str, ...]
    evaluation_only_latent_state: Mapping[str, float]
    evaluation_only_process_state: Mapping[str, float]
    physical_state_id: str = ""
    disturbance_state_id: str = ""
    mean_policy_detector_rate: float | None = None
    aggregate_exploration_detector_rate: float | None = None
    exploration_excess_detector_events: float = 0.0
    evaluation_policy_cycles: int = 0
    candidate_budget_class: str = "not_applicable"
    simulator_state_hash: str = ""
    controller_state_hash: str = ""
    physical_rollback_failures: tuple[str, ...] = ()
    rollback_outcomes: tuple[Mapping[str, object], ...] = ()
    reentry_request: Mapping[str, object] | None = None
    regional_recovery: Mapping[str, object] | None = None
    recovery_count: int = 0
    timing: OnlineTimingBreakdown | None = None


@dataclass(frozen=True)
class EndpointFollowupObservation:
    """Evaluation-only hold-policy evidence acquired after controller completion.

    Follow-up never invokes a controller or changes its completion classification.  It
    exists solely to make the fixed wall-clock endpoint support observable rather than
    silently treating the last controller interval as the censoring horizon.
    """

    observation_index: int
    qec_cycles: int
    detector_events: int
    detector_exposures: int
    detector_rate: float
    cumulative_e2e_support_s: float
    policy_hash: str
    batch_id: str


@dataclass(frozen=True)
class ArmMetrics:
    scenario_id: str
    seed: int
    arm: str
    qec_cycles: int
    candidate_evaluations: int
    diagnostic_shots: int
    diagnostic_downtime_s: float
    detector_event_rate: float
    integrated_excess_detector_events: float
    logical_failure_rate: float
    exploration_damage: float
    final_detector_event_rate: float
    recovery_interval: int | None
    rollback_count: int
    wall_time_s: float
    final_detector_events: int = 0
    final_detector_exposures: int = 0
    completion_status: str = "completed"
    censoring_reason: str | None = None
    missing_data_reasons: tuple[str, ...] = ()
    recovery_endpoints: tuple[RecoveryEndpoint, ...] = ()
    worst_region_recovery_cycles: Mapping[str, int | None] = field(default_factory=dict)
    exponential_fit: ExponentialRecoveryFit | None = None
    logical_circuit_failures: int = 0
    logical_circuit_shots: int = 0
    logical_circuit_failure_probability: float | None = None
    logical_error_per_round: float | None = None
    lifecycle_violation_count: int = 0
    bootstrap_count: int = 0
    disturbance_realization_id: str = ""
    disturbance_epoch_s: float | None = None
    pre_disturbance_baseline_rate: float | None = None
    pre_disturbance_baseline_ci95: tuple[float, float] | None = None
    pre_disturbance_observation_hash: str = ""
    controller_truth_access_count: int = 0
    mean_policy_detector_event_rate: float | None = None
    aggregate_exploration_detector_event_rate: float | None = None
    exploration_excess_detector_events: float = 0.0
    mean_policy_evaluation_cycles: int = 0
    candidate_cycles: int = 0
    candidate_budget_class: str = "not_applicable"
    physical_rollback_failure_count: int = 0
    recovery_count: int = 0
    timing_invalidity_reasons: tuple[str, ...] = ()
    controller_completion_e2e_s: float | None = None
    endpoint_followup_s: float = 0.0
    endpoint_followup_cycles: int = 0
    total_observation_support_s: float | None = None
    endpoint_followup: tuple[EndpointFollowupObservation, ...] = ()


@dataclass(frozen=True)
class MatchedPairStatistic:
    comparator_arm: str
    reference_arm: str
    outcome: str
    pair_count: int
    missing_pairs: int
    differences: tuple[float, ...]
    confidence_interval: ConfidenceInterval | None
    rationale: str


@dataclass(frozen=True)
class RecoverySummary:
    arm: str
    target_fraction: float
    run_count: int
    reached_count: int
    censored_count: int
    missing_count: int
    independent_seed_count: int
    reached_fraction: float
    reached_fraction_ci95: tuple[float, float] | None
    median_recovery_cycles: float | None
    restricted_mean_recovery_cycles: float | None


@dataclass(frozen=True)
class AcceptanceGate:
    gate_id: str
    status: str
    measured_ratio: float | None
    required_ratio: float | None
    rationale: str
    confidence_interval: ConfidenceInterval | None = None
    pair_count: int = 0
    primary: bool = True
    estimand: Mapping[str, object] = field(default_factory=dict)
    estimators: "GateEstimators | None" = None


@dataclass(frozen=True)
class GateEstimators:
    worst_matched_ratio: float | None = None
    median_matched_ratio: float | None = None
    cluster_aggregate_ratio: float | None = None
    cluster_aggregate_ci95: ConfidenceInterval | None = None
    rmst_difference: float | None = None
    rmst_ci95: ConfidenceInterval | None = None
    tail_difference: float | None = None
    tail_ci95: ConfidenceInterval | None = None
    gate_decision_statistic: float | None = None
    gate_threshold: float | None = None
    gate_status: str = "not_evaluable"


@dataclass(frozen=True)
class BenchmarkProvenance:
    configuration_hash: str
    vcs_revision: str
    source_tree_hash: str
    package_version: str
    simulator_version: str
    python_version: str
    logical_stack_versions: Mapping[str, str]
    timing_environment: TimingEnvironment


@dataclass(frozen=True)
class ExperimentalDesignAudit:
    protocol_id: str
    stationary_stage0: bool
    held_out_native_qec_baseline: bool
    matched_baseline_observations: bool
    matched_disturbance_realizations: bool
    synchronized_disturbance_onsets: bool
    controller_truth_isolation: bool
    claim_scope: str
    matched_initial_physical_state: bool = False
    matched_controller_state: bool = False
    clone_mutability_isolated: bool = False
    matched_evaluator_configuration: bool = False


@dataclass(frozen=True)
class _MatchedPreparation:
    device: ScalableQECDevice
    bootstrap: BootstrapResult
    accounting: tuple[int, int, float]
    baseline: PreDisturbanceBaseline


@dataclass(frozen=True)
class BenchmarkReport:
    schema_version: str
    config: BenchmarkConfig
    scenarios: tuple[BenchmarkScenario, ...]
    metrics: tuple[ArmMetrics, ...]
    pre_disturbance_baselines: tuple[PreDisturbanceBaseline, ...]
    trajectories: tuple[IntervalTrajectory, ...]
    matched_statistics: tuple[MatchedPairStatistic, ...]
    recovery_summaries: tuple[RecoverySummary, ...]
    gates: tuple[AcceptanceGate, ...]
    provenance: BenchmarkProvenance
    design_audit: ExperimentalDesignAudit
    required_arms: tuple[str, ...]
    authoritative: bool
    accepted: bool
    invalidity_reasons: tuple[str, ...]
    acceptance_failure_reasons: tuple[str, ...]
    report_hash: str
    evidence_records: tuple[EvidenceRecord, ...] = ()
    report_contract_issues: tuple[str, ...] = ()
    preflight_manifest_hash: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def assert_authoritative(self) -> None:
        if not self.authoritative:
            raise AssertionError("benchmark is not authoritative: " + "; ".join(self.invalidity_reasons))

    def assert_accepted(self) -> None:
        self.assert_authoritative()
        failed = [gate.gate_id for gate in self.gates
                  if gate.primary and gate.status != "pass"]
        if failed or self.acceptance_failure_reasons:
            reasons = (["acceptance gates failed: " + ", ".join(failed)] if failed else [])
            reasons.extend(self.acceptance_failure_reasons)
            raise AssertionError("; ".join(reasons))


def _loadings(qubit_count: int, scale: float = 1.0) -> dict[str, float]:
    return {f"drive:q{index}": scale for index in range(qubit_count)}


def default_benchmark_scenarios(qubit_count: int = 5) -> tuple[BenchmarkScenario, ...]:
    local = {"drive:q0": 1.0, "drive:q1": .45}
    return (
        BenchmarkScenario("sinusoid", (LatentProcessSpec("periodic", DriftKind.SINUSOID, local,
            amplitude=.35, period_s=1.6),), True, "familiar periodic local drift"),
        BenchmarkScenario("telegraph", (LatentProcessSpec("rtn", DriftKind.SEMI_MARKOV_TELEGRAPH, local,
            amplitude=.32, rate_hz=1.2, mean_dwell_s=.5),), True, "semi-Markov recurring discrete drift"),
        BenchmarkScenario("ou_step", (
            LatentProcessSpec("ou", DriftKind.ORNSTEIN_UHLENBECK, local, diffusion=.12, ou_kappa=.5),
            LatentProcessSpec("step", DriftKind.STEP, {"drive:q2": 1.}, amplitude=.4, step_time_s=.35),
        ), True, "smooth drift plus abrupt local change"),
        BenchmarkScenario("nested_common", (
            LatentProcessSpec("slow", DriftKind.SINUSOID, _loadings(qubit_count, .35), amplitude=.3, period_s=2.4),
            LatentProcessSpec("fast", DriftKind.RANDOM_TELEGRAPH, local, amplitude=.22, rate_hz=1.8,
                              parent_process_id="slow"),
        ), True, "nested local switching under a common-mode oscillation"),
        BenchmarkScenario("unknown", (LatentProcessSpec("unknown", DriftKind.UNKNOWN_HEAVY_TAILED,
            _loadings(qubit_count, .25), diffusion=.08),), False, "unstructured heavy-tailed OOD drift"),
    )


def confirmatory_benchmark_scenarios(
        qubit_count: int = 5) -> tuple[BenchmarkScenario, ...]:
    """Prospectively frozen scenario variants for the v2 held-out campaign.

    The v1 scenario definitions and seeds 101--105 are development evidence.  These
    variants retain the same scientific disturbance families while changing every
    process definition and identifier before any confirmatory tape is generated.
    Controller code must never branch on these identifiers.
    """
    local_a = {"drive:q0": 1.0, "drive:q2": .38}
    local_b = {"drive:q1": 1.0, "drive:q3": .41}
    return (
        BenchmarkScenario(
            "confirmatory_periodic_mixture",
            (LatentProcessSpec(
                "heldout-periodic", DriftKind.SINUSOID, local_a,
                amplitude=.31, period_s=1.83),),
            True, "held-out periodic local drift with unseen loadings and period"),
        BenchmarkScenario(
            "confirmatory_semi_markov",
            (LatentProcessSpec(
                "heldout-semi-markov", DriftKind.SEMI_MARKOV_TELEGRAPH, local_b,
                amplitude=.29, rate_hz=.95, mean_dwell_s=.63),),
            True, "held-out recurring semi-Markov local drift"),
        BenchmarkScenario(
            "confirmatory_ou_step",
            (LatentProcessSpec(
                "heldout-ou", DriftKind.ORNSTEIN_UHLENBECK, local_a,
                diffusion=.10, ou_kappa=.42),
             LatentProcessSpec(
                "heldout-step", DriftKind.STEP, {"drive:q1": 1.0},
                amplitude=.36, step_time_s=.47)),
            True, "held-out smooth drift plus an unseen abrupt local change"),
        BenchmarkScenario(
            "confirmatory_nested_common",
            (LatentProcessSpec(
                "heldout-common", DriftKind.SINUSOID,
                _loadings(qubit_count, .32), amplitude=.27, period_s=2.77),
             LatentProcessSpec(
                "heldout-nested", DriftKind.SEMI_MARKOV_TELEGRAPH, local_b,
                amplitude=.20, rate_hz=1.45, mean_dwell_s=.44,
                parent_process_id="heldout-common")),
            True, "held-out nested local switching under common-mode drift"),
        BenchmarkScenario(
            "confirmatory_heavy_tailed",
            (LatentProcessSpec(
                "heldout-heavy-tail", DriftKind.UNKNOWN_HEAVY_TAILED,
                _loadings(qubit_count, .22), diffusion=.07),),
            False, "held-out unstructured heavy-tailed OOD drift"),
    )


def confirmatory_v3_benchmark_scenarios(
        qubit_count: int = 5) -> tuple[BenchmarkScenario, ...]:
    """Unexecuted v3 definitions frozen for conditional-residual confirmation."""
    local_a = {"drive:q0": 1.0, "drive:q3": .33}
    local_b = {"drive:q1": 1.0, "drive:q4": .36}
    return (
        BenchmarkScenario("v3_familiar_sinusoid", (
            LatentProcessSpec("v3-periodic", DriftKind.SINUSOID, local_a,
                              amplitude=.30, period_s=2.11, phase_rad=.37),), True,
            "fresh periodic recurrence for persistent-state forecasting"),
        BenchmarkScenario("v3_semi_markov", (
            LatentProcessSpec("v3-semi", DriftKind.SEMI_MARKOV_TELEGRAPH, local_b,
                              amplitude=.30, rate_hz=1.07, mean_dwell_s=.58),), True,
            "fresh semi-Markov recurrence for regime-policy reuse"),
        BenchmarkScenario("v3_ou_step", (
            LatentProcessSpec("v3-ou", DriftKind.ORNSTEIN_UHLENBECK, local_a,
                              diffusion=.11, ou_kappa=.47),
            LatentProcessSpec("v3-step", DriftKind.STEP, {"drive:q2": 1.0},
                              amplitude=.37, step_time_s=.53)), True,
            "fresh smooth plus abrupt component"),
        BenchmarkScenario("v3_nested_common", (
            LatentProcessSpec("v3-common", DriftKind.SINUSOID,
                              _loadings(qubit_count, .30), amplitude=.28,
                              period_s=2.93, phase_rad=.21),
            LatentProcessSpec("v3-local", DriftKind.SEMI_MARKOV_TELEGRAPH, local_b,
                              amplitude=.21, rate_hz=1.31, mean_dwell_s=.49,
                              parent_process_id="v3-common")), True,
            "fresh nested common/local recurrence"),
        BenchmarkScenario("v3_unknown_heavy_tailed", (
            LatentProcessSpec("v3-unknown", DriftKind.UNKNOWN_HEAVY_TAILED,
                              _loadings(qubit_count, .24), diffusion=.075),), False,
            "fresh OOD heavy-tailed process"),
        BenchmarkScenario("v3_persistent_residual", (
            LatentProcessSpec("v3-residual", DriftKind.RANDOM_WALK, local_a,
                              diffusion=.055),), True,
            "learnable persistent local residual after predictive correction"),
        BenchmarkScenario("v3_no_residual", (), True,
            "stationary no-residual negative-control condition"),
    )


def benchmark_scenario_registry(
        qubit_count: int = 5) -> tuple[BenchmarkScenario, ...]:
    """Return development and prospectively frozen confirmatory definitions."""
    return (*default_benchmark_scenarios(qubit_count),
            *confirmatory_benchmark_scenarios(qubit_count),
            *confirmatory_v3_benchmark_scenarios(qubit_count))


def _mean_ci(values: Sequence[float]) -> ConfidenceInterval | None:
    if len(values) < 2:
        return None
    estimate = statistics.fmean(values)
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    # Two-sided 95% Student-t critical values.  Seed counts are deliberately small in
    # expensive controller experiments, so a normal critical value is anti-conservative.
    critical_by_df = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }
    critical = critical_by_df.get(len(values)-1, 1.959963984540054)
    radius = critical * standard_error
    return ConfidenceInterval(estimate-radius, estimate, estimate+radius)


def _wilson(events: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    z = 1.959963984540054
    p = events / total
    denominator = 1 + z*z/total
    centre = (p + z*z/(2*total)) / denominator
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return max(0.0, centre-radius), min(1.0, centre+radius)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = probability*(len(ordered)-1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position-lower
    return ordered[lower]*(1-weight)+ordered[upper]*weight


def _restricted_mean_time(observations: Sequence[tuple[float, bool]],
                          horizon_s: float) -> float:
    """Kaplan--Meier RMST with administrative truncation at one common horizon."""
    if horizon_s <= 0 or not observations:
        return math.nan
    rows = [(min(max(0.0, time_s), horizon_s), bool(event and time_s <= horizon_s))
            for time_s, event in observations]
    survival = 1.0
    area = previous = 0.0
    at_risk = len(rows)
    for timestamp in sorted({time_s for time_s, _ in rows if time_s <= horizon_s}):
        area += survival*(timestamp-previous)
        events = sum(1 for time_s, event in rows
                     if time_s == timestamp and event)
        censored = sum(1 for time_s, event in rows
                       if time_s == timestamp and not event)
        if events and at_risk:
            survival *= 1-events/at_risk
        at_risk -= events+censored
        previous = timestamp
    if previous < horizon_s:
        area += survival*(horizon_s-previous)
    return area


def _source_provenance(config: BenchmarkConfig, scenarios: Sequence[BenchmarkScenario],
                       logical_versions: Mapping[str, str]) -> BenchmarkProvenance:
    root = Path(__file__).resolve().parents[3]
    source_files = sorted((root / "src").rglob("*.py")) + [root / "pyproject.toml"]
    tree_payload = [(str(path.relative_to(root)).replace("\\", "/"), path.read_bytes().hex())
                    for path in source_files if path.exists()]
    tree_hash = deterministic_hash(tree_payload)
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True, timeout=2).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = f"unversioned:{tree_hash}"
    return BenchmarkProvenance(
        deterministic_hash({
            "config": asdict(config),
            "scenarios": [asdict(scenario) for scenario in scenarios],
        }), revision, tree_hash, __version__,
        SIMULATOR_VERSION, platform.python_version(), dict(logical_versions),
        TimingEnvironment.capture(__version__))


def benchmark_configuration_hash(config: BenchmarkConfig,
                                 scenarios: Sequence[BenchmarkScenario],
                                 arm_names: Sequence[str]) -> str:
    return deterministic_hash({
        "config": asdict(config),
        "scenarios": [asdict(item) for item in scenarios],
        "arm_names": tuple(sorted(arm_names)),
    })


class BenchmarkRunner:
    """Execute primary and ablation arms over common, replayable disturbance tapes."""

    def __init__(self, config: BenchmarkConfig = BenchmarkConfig(),
                 scenarios: Sequence[BenchmarkScenario] | None = None,
                 arm_factories: Mapping[str, Callable[[int], BenchmarkArm]] | None = None,
                 preflight_manifest: object | str | Path | None = None,
                 launch_binding_hash: str | None = None) -> None:
        self.config = config
        self.scenarios = tuple(scenarios or default_benchmark_scenarios(config.qubit_count))
        self.arm_factories = dict(arm_factories or self._default_arms())
        missing = set(PRIMARY_ARMS) - set(self.arm_factories)
        if config.authoritative and missing:
            raise ValueError(f"authoritative benchmark is missing primary arms: {sorted(missing)}")
        self.preflight_manifest = preflight_manifest
        self._launch_binding_hash = launch_binding_hash
        self._logical_error: str | None = None
        try:
            self._logical = RotatedSurfaceCodeEvaluator(SurfaceCodeMemoryConfig(
                distance=config.code_distance, rounds=config.logical_rounds,
                shots=config.logical_shots_per_interval))
        except (LogicalStackUnavailable, ValueError) as error:
            self._logical = None
            self._logical_error = str(error)

    @property
    def launch_configuration_hash(self) -> str:
        return self._launch_binding_hash or benchmark_configuration_hash(
            self.config, self.scenarios, tuple(self.arm_factories))

    def _require_preflight(self) -> str:
        if not self.config.authoritative:
            return ""
        if self.preflight_manifest is None:
            raise BenchmarkPreflightError(
                "authoritative benchmark refused: a fresh passing preflight manifest is required")
        from hdfa_rl_suite.validation.manifest import (
            PreflightManifest, load_preflight_manifest, validate_preflight_manifest,
        )
        manifest = (load_preflight_manifest(self.preflight_manifest)
                    if isinstance(self.preflight_manifest, (str, Path))
                    else self.preflight_manifest)
        if not isinstance(manifest, PreflightManifest):
            raise BenchmarkPreflightError("authoritative benchmark refused: invalid preflight manifest object")
        reasons = validate_preflight_manifest(
            manifest, expected_configuration_hash=self.launch_configuration_hash)
        if self.config.candidate_cycles < manifest.minimum_candidate_cycles:
            reasons = (*reasons,
                       "benchmark candidate cycles are below the validated preflight floor")
        if reasons:
            raise BenchmarkPreflightError(
                "authoritative benchmark refused: " + "; ".join(reasons))
        return manifest.manifest_hash

    def _bootstrap_config(self) -> ScalableBootstrapConfig:
        return ScalableBootstrapConfig(
            characterization_shots=self.config.bootstrap_characterization_shots,
            validation_cycles=self.config.bootstrap_validation_cycles,
            target_posterior_stddev=self.config.bootstrap_target_stddev,
            qec_detector_rate_limit=self.config.bootstrap_qec_rate_limit,
            block_predictive_familywise_alpha=self.config.bootstrap_block_familywise_alpha,
        )

    def _default_arms(self) -> Mapping[str, Callable[[int], BenchmarkArm]]:
        cycles = self.config.candidate_cycles
        bootstrap = self._bootstrap_config()
        product = ProductLoopConfig(
            extended_structured_models=self.config.extended_structured_models,
            parallel_regional_updates=self.config.parallel_regional_updates,
            candidate_elimination_z=self.config.candidate_elimination_z)
        return {
            "fixed": lambda seed: FixedCalibrationArm(),
            "periodic_recalibration": lambda seed: PeriodicRecalibrationArm(period=4, shots=128),
            "greedy_calibration": lambda seed: GreedyCalibrationArm(shots=64),
            "state_only": lambda seed: PhysicalInferenceArm("state_only", seed=seed),
            "sequential_hdfa": lambda seed: PhysicalInferenceArm("sequential_hdfa", seed=seed),
            "joint_hdfa_reactive": lambda seed: PhysicalInferenceArm("joint_hdfa_reactive", seed=seed),
            "full_control_detector_rl": lambda seed: FullControlRLArm(
                seed=seed, candidate_count=40, candidate_cycles=cycles),
            "predictive_hdfa_no_residual": lambda seed: PredictiveHDFARLArm(
                seed=seed, residual=False, candidate_cycles=cycles,
                bootstrap_config=bootstrap, product_config=product),
            "predictive_hdfa_residual_rl": lambda seed: PredictiveHDFARLArm(
                seed=seed, residual=True, candidate_count=4, candidate_cycles=cycles,
                bootstrap_config=bootstrap, product_config=product),
            "oracle": lambda seed: OracleControlArm(),
        }

    @staticmethod
    def _bootstrap_counts(result: BootstrapResult) -> tuple[int, int]:
        shots = qec_cycles = 0
        for estimate in result.calibration_estimates.values():
            shots += int(estimate.diagnostics.get("active_design_shots", 0))
            shots += int(estimate.diagnostics.get("held_out_shots", 0))
            qec_cycles += int(estimate.diagnostics.get("qec_cycles", 0))
        return shots, qec_cycles

    def _common_bootstrap(self, device: ScalableQECDevice) -> tuple[BootstrapResult, int, int, float]:
        started = device.now_s
        result = ScalableBootstrapCalibrator(device, self._bootstrap_config()).run()
        shots, qec_cycles = self._bootstrap_counts(result)
        return result, shots, qec_cycles, device.now_s-started

    def _pre_disturbance_baseline(self, scenario: BenchmarkScenario, seed: int,
                                  name: str, device: ScalableQECDevice) -> PreDisturbanceBaseline:
        if device.disturbances_armed:
            raise RuntimeError("pre-disturbance baseline requires a disarmed disturbance tape")
        started = device.now_s
        batch = device.acquire(self.config.pre_disturbance_baseline_cycles)
        view = device.oracle_evaluation_view("evaluation:randomized-phase-boundary")
        latent = view.latent_state()
        process = view.process_state()
        latent_max = max((abs(value) for value in latent.values()), default=0.0)
        process_max = max((abs(value) for value in process.values()), default=0.0)
        epoch = device.arm_disturbances()
        fingerprint = device.counterfactual_state_fingerprint()
        logical_config_hash = (deterministic_hash({
            "config": asdict(self._logical.config),
            "noise_map": asdict(self._logical.noise_map),
        }) if self._logical is not None else "unavailable")
        return PreDisturbanceBaseline(
            scenario.scenario_id, seed, name, batch.cycles, started, device.now_s,
            batch.detector_events, batch.detector_exposures, batch.detector_rate,
            _wilson(batch.detector_events, batch.detector_exposures),
            dict(batch.detector_counts), batch.logical_failures,
            batch.policy_activation.policy_hash, batch.batch_id,
            deterministic_hash(batch), False, epoch, device.disturbance_realization_id,
            latent_max, process_max,
            fingerprint.physical_state_id, fingerprint.disturbance_state_id,
            fingerprint.simulator_state_hash, fingerprint.controller_state_hash,
            fingerprint.process_rng_state_hash,
            fingerprint.characterization_rng_state_hash,
            fingerprint.detector_evaluator_config_hash, logical_config_hash,
            fingerprint.policy_id, dict(fingerprint.policy_controls), False,
        )

    def _prepare_matched_state(self, scenario: BenchmarkScenario,
                               seed: int) -> _MatchedPreparation:
        device = ScalableQECDevice(SimulatorConfig(
            qubit_count=self.config.qubit_count, code_distance=self.config.code_distance,
            cycle_period_s=self.config.cycle_period_s,
            controller_latency_s=min(.002, 2*self.config.cycle_period_s),
            disturbances_enabled_at_start=False, seed=seed, processes=scenario.processes,
        ))
        bootstrap, shots, qec_cycles, downtime = self._common_bootstrap(device)
        if bootstrap.health.status is not HealthStatus.PASSED:
            raise RuntimeError("matched Stage-0 preparation failed QEC-operability gate")
        baseline = self._pre_disturbance_baseline(
            scenario, seed, "__matched_template__", device)
        return _MatchedPreparation(
            device, bootstrap, (shots, qec_cycles + baseline.cycles, downtime), baseline)

    def _logical_seed(self, scenario: BenchmarkScenario, seed: int, interval: int) -> int:
        # Arm deliberately omitted: identical circuit random numbers are used for a
        # matched seed/interval, while physical error probabilities remain arm-specific.
        digest = deterministic_hash((scenario.scenario_id, seed, interval, "logical-crn-v1"))
        return int(digest[:15], 16)

    def _trajectory(self, scenario: BenchmarkScenario, seed: int, name: str,
                    interval: int, device: ScalableQECDevice, output: ArmIntervalResult,
                    logical: LogicalPerformanceEvidence | None,
                    external_bootstrap: BootstrapResult | None,
                    external_counts: tuple[int, int, float],
                    controller_truth_accesses: Sequence[str]) -> IntervalTrajectory:
        if logical is not None and (
            logical.physical_state_id != output.observation.physical_state_id
            or logical.policy_hash != output.observation.policy_activation.policy_hash
            or logical.disturbance_state_id != output.observation.disturbance_state_id
        ):
            raise RuntimeError("logical and detector evidence do not share one physical/policy state")
        view = device.oracle_evaluation_view("evaluation:matched-trajectory-provenance")
        ext_shots, ext_qec, ext_downtime = external_counts if interval == 0 else (0, 0, 0.0)
        stage_path = output.stage_path
        bootstrap_reason = output.bootstrap_reason
        bootstrap_count = output.bootstrap_count
        if external_bootstrap is not None and interval == 0:
            stage_path = ("stage0:cold_start",) + stage_path
            bootstrap_reason = "cold_start"
            bootstrap_count = 1
        bootstrap_evidence = (external_bootstrap.to_dict() if external_bootstrap is not None and interval == 0
                              else output.bootstrap_evidence)
        return IntervalTrajectory(
            scenario.scenario_id, seed, name, interval, device.now_s,
            output.observation.detector_events, output.observation.detector_exposures,
            output.observation.detector_rate if output.observation.detector_exposures else None,
            dict(output.observation.detector_counts), output.auxiliary_detector_events,
            output.auxiliary_detector_exposures,
            output.total_qec_cycles + ext_qec, output.candidate_evaluations,
            output.candidate_cycles, output.diagnostic_shots + ext_shots,
            output.diagnostic_downtime_s + ext_downtime, output.exploration_damage,
            output.observation.logical_failures + output.auxiliary_logical_failures,
            logical, output.policy_hash, dict(device.confirmed_policy.controls),
            output.lifecycle_mode, output.authorization, output.lifecycle_violations,
            bootstrap_reason, bootstrap_count, stage_path, output.replay_hash,
            bootstrap_evidence, output.candidate_trajectories, output.stage_evidence,
            device.disturbance_realization_id,
            float(device.disturbance_epoch_s) if device.disturbance_epoch_s is not None else math.nan,
            float(device.disturbance_elapsed_s) if device.disturbance_elapsed_s is not None else math.nan,
            tuple(controller_truth_accesses), view.latent_state(), view.process_state(),
            output.observation.physical_state_id,
            output.observation.disturbance_state_id,
            (output.mean_policy_detector_rate if output.mean_policy_detector_rate is not None
             else output.observation.detector_rate),
            output.aggregate_exploration_detector_rate,
            output.exploration_excess_detector_events,
            output.evaluation_policy_cycles or output.observation.cycles,
            output.candidate_budget_class,
            output.observation.simulator_state_hash,
            output.observation.controller_state_hash,
            output.physical_rollback_failures,
            output.rollback_outcomes,
            output.reentry_request,
            output.regional_recovery,
            output.recovery_count,
            output.timing,
        )

    @staticmethod
    def _endpoint(rates: Sequence[float], interval_cycles: Sequence[int],
                  interval_candidates: Sequence[int], target: float,
                  base_rate: float,
                  timings: Sequence[OnlineTimingBreakdown | None] = ()) -> RecoveryEndpoint:
        def timing_prefix(end_index: int) -> tuple[
                float | None, float | None, Mapping[str, float], str]:
            selected = tuple(timings[:end_index+1])
            if len(selected) != end_index+1 or any(item is None for item in selected):
                return None, None, {}, "missing"
            typed = tuple(item for item in selected if item is not None)
            invalid = tuple(reason for item in typed for reason in item.validate())
            if invalid:
                return None, None, {}, "invalid:"+";".join(dict.fromkeys(invalid))
            components = {
                "qec_acquisition_s": sum(item.qec_acquisition_s for item in typed),
                "diagnostic_downtime_s": sum(item.diagnostic_downtime_s for item in typed),
                "actuation_acknowledgement_s": sum(
                    item.actuation_acknowledgement_s for item in typed),
                "online_compute_critical_s": sum(
                    item.online_compute_critical_s for item in typed),
            }
            return (sum(components.values()),
                    sum(item.total_observed_host_wall_s for item in typed),
                    components, "valid")

        censor_cycles = sum(interval_cycles)
        censor_candidates = sum(interval_candidates)
        censor_e2e, censor_host, censor_components, censor_timing_status = (
            timing_prefix(len(rates)-1) if rates else (None, None, {}, "missing"))
        if not rates or any(not math.isfinite(rate) for rate in rates):
            return RecoveryEndpoint(target, "missing", None, None, None, censor_cycles,
                                    censor_candidates, None,
                                    "detector rate is missing or non-finite",
                                    None, censor_e2e, None, censor_host,
                                    censor_components,
                                    censor_timing_status)
        peak = max(rates)
        peak_index = rates.index(peak)
        if peak <= base_rate:
            return RecoveryEndpoint(target, "reached", 0, 0, 0, censor_cycles,
                                    censor_candidates, base_rate,
                                    "no excess degradation above the declared detector floor",
                                    0.0, censor_e2e, 0.0, censor_host,
                                    {"qec_acquisition_s": 0.0,
                                     "diagnostic_downtime_s": 0.0,
                                     "actuation_acknowledgement_s": 0.0,
                                     "online_compute_critical_s": 0.0},
                                    "valid" if censor_timing_status == "valid" else censor_timing_status)
        threshold = base_rate + (1-target) * (peak-base_rate)
        accumulated = 0
        accumulated_candidates = sum(interval_candidates[:peak_index + 1])
        for index in range(peak_index+1, len(rates)):
            accumulated += interval_cycles[index]
            accumulated_candidates += interval_candidates[index]
            if rates[index] <= threshold:
                e2e, host, components, timing_status = timing_prefix(index)
                return RecoveryEndpoint(target, "reached", accumulated,
                                        accumulated_candidates, index-peak_index,
                                        accumulated, accumulated_candidates, threshold,
                                        "observed nonparametric recovery",
                                        e2e, censor_e2e, host, censor_host,
                                        components, timing_status)
        return RecoveryEndpoint(target, "censored", None, None, None, accumulated,
                                accumulated_candidates, threshold,
                                "target was not reached before the declared censoring limit",
                                None, censor_e2e, None, censor_host,
                                censor_components,
                                censor_timing_status)

    def _fit(self, rates: Sequence[float], base_rate: float) -> ExponentialRecoveryFit:
        if len(rates) < 4 or any(not math.isfinite(rate) for rate in rates):
            return ExponentialRecoveryFit(None, None, None, None, None, False,
                                          "at least four finite intervals are required")
        peak_index = rates.index(max(rates))
        points = [(index-peak_index, rate-base_rate) for index, rate in enumerate(rates[peak_index:])
                  if rate > base_rate]
        if len(points) < 3:
            return ExponentialRecoveryFit(None, None, None, None, None, False,
                                          "fewer than three positive excess-rate points")
        x = [float(item[0]) for item in points]
        y = [math.log(item[1]) for item in points]
        xbar, ybar = statistics.fmean(x), statistics.fmean(y)
        sxx = sum((value-xbar)**2 for value in x)
        if sxx <= 0:
            return ExponentialRecoveryFit(None, None, None, None, None, False,
                                          "recovery times have zero spread")
        slope = sum((left-xbar)*(right-ybar) for left, right in zip(x, y)) / sxx
        intercept = ybar-slope*xbar
        residuals = [right-(intercept+slope*left) for left, right in zip(x, y)]
        sse = sum(value*value for value in residuals)
        sst = sum((value-ybar)**2 for value in y)
        r2 = 1-sse/sst if sst > 0 else 0.0
        slope_se = math.sqrt((sse/max(1, len(x)-2))/sxx)
        autocorrelation = None
        if len(residuals) >= 3:
            mean = statistics.fmean(residuals)
            denominator = sum((value-mean)**2 for value in residuals)
            autocorrelation = (sum((residuals[i]-mean)*(residuals[i-1]-mean)
                                   for i in range(1, len(residuals))) / denominator
                               if denominator > 0 else 0.0)
        gamma = -slope
        relative_error = slope_se/max(abs(gamma), 1e-12)
        residual_ok = autocorrelation is None or abs(autocorrelation) <= self.config.maximum_fit_residual_autocorrelation
        credible = (gamma > 0 and r2 >= self.config.minimum_fit_r2 and residual_ok
                    and math.isfinite(slope_se)
                    and relative_error <= self.config.maximum_gamma_relative_standard_error)
        reason = ("fit passed the predeclared R2, residual-correlation, positive-decay, and gamma-uncertainty gates"
                  if credible else
                  "fit failed at least one predeclared R2/residual/decay/uncertainty gate; no target is extrapolated")
        return ExponentialRecoveryFit(gamma, slope_se, r2, statistics.fmean(residuals),
                                      autocorrelation, credible, reason)

    def _summarize_run(self, scenario: BenchmarkScenario, seed: int, name: str,
                       trajectories: Sequence[IntervalTrajectory], status: str,
                       censor_reason: str | None, missing: Sequence[str],
                       rollback_count: int, baseline: PreDisturbanceBaseline | None,
                       disturbance_realization_id: str,
                       disturbance_epoch_s: float | None,
                       controller_truth_access_count: int,
                       followup: Sequence[EndpointFollowupObservation] = ()) -> ArmMetrics:
        exposures = sum(item.detector_exposures + item.auxiliary_detector_exposures for item in trajectories)
        events = sum(item.detector_events + item.auxiliary_detector_events for item in trajectories)
        rates = [item.detector_rate if item.detector_rate is not None else math.nan for item in trajectories]
        interval_cycles = [item.qec_cycles for item in trajectories]
        interval_candidates = [item.candidate_evaluations for item in trajectories]
        base = baseline.detector_rate if baseline is not None else 0.012
        excess = sum(max(0.0, item.detector_events + item.auxiliary_detector_events
                         - base * (item.detector_exposures + item.auxiliary_detector_exposures))
                     for item in trajectories)
        final_count = min(self.config.steady_state_intervals, len(trajectories))
        final = trajectories[-final_count:] if final_count else ()
        final_events = sum(item.detector_events for item in final)
        final_exposures = sum(item.detector_exposures for item in final)
        timings = [item.timing for item in trajectories]
        endpoints = tuple(self._endpoint(
            rates, interval_cycles, interval_candidates, target, base, timings)
                          for target in RECOVERY_TARGETS)
        controller_e2e = None
        if timings and all(item is not None and not item.validate() for item in timings):
            controller_e2e = sum(
                item.e2e_convergence_time_s for item in timings if item is not None)
        total_support = (followup[-1].cumulative_e2e_support_s if followup
                         else controller_e2e)
        amended_endpoints: list[RecoveryEndpoint] = []
        for endpoint in endpoints:
            censor_components = dict(endpoint.e2e_components_s)
            if (endpoint.status == "censored" and total_support is not None
                    and controller_e2e is not None):
                censor_components["qec_acquisition_s"] = (
                    float(censor_components.get("qec_acquisition_s", 0.0))
                    + max(0.0, total_support-controller_e2e))
            updated = replace(
                endpoint,
                censoring_e2e_time_s=(total_support
                                      if total_support is not None
                                      else endpoint.censoring_e2e_time_s),
                e2e_components_s=censor_components)
            if endpoint.status == "censored" and endpoint.threshold_rate is not None:
                reached = next((row for row in followup
                                if row.detector_rate <= endpoint.threshold_rate), None)
                if reached is not None:
                    followup_cycles_to_target = sum(
                        row.qec_cycles for row in followup
                        if row.observation_index <= reached.observation_index)
                    components = dict(endpoint.e2e_components_s)
                    components["qec_acquisition_s"] = (
                        float(components.get("qec_acquisition_s", 0.0))
                        + (reached.cumulative_e2e_support_s-(controller_e2e or 0.0)))
                    updated = replace(
                        updated, status="reached",
                        detector_cycles=(endpoint.censoring_cycles
                                         + followup_cycles_to_target),
                        candidate_evaluations=endpoint.censoring_candidate_evaluations,
                        intervals_after_peak=None,
                        reason="observed during evaluation-only hold-policy follow-up",
                        e2e_time_s=reached.cumulative_e2e_support_s,
                        e2e_components_s=components, timing_status="valid")
            amended_endpoints.append(updated)
        endpoints = tuple(amended_endpoints)
        by_detector: dict[str, list[float]] = {}
        for trajectory in trajectories:
            for detector, (detector_events, detector_exposures) in trajectory.detector_counts.items():
                by_detector.setdefault(detector, []).append(
                    detector_events / detector_exposures if detector_exposures else math.nan)
        worst_region = {}
        for detector, detector_rates in by_detector.items():
            detector_base = base
            if baseline is not None and detector in baseline.detector_counts:
                baseline_events, baseline_exposures = baseline.detector_counts[detector]
                if baseline_exposures:
                    detector_base = baseline_events / baseline_exposures
            endpoint = self._endpoint(
                detector_rates, interval_cycles, interval_candidates, .90,
                detector_base, timings)
            worst_region[detector] = endpoint.detector_cycles
        logical_rows = [item.logical_evidence for item in trajectories if item.logical_evidence is not None]
        logical_failures = sum(item.logical_failures for item in logical_rows)
        logical_shots = sum(item.shots for item in logical_rows)
        logical_probability = logical_failures/logical_shots if logical_shots else None
        per_round = (1-(1-logical_probability)**(1/self.config.logical_rounds)
                     if logical_probability is not None else None)
        generic_logical = sum(item.generic_logical_proxy_failures for item in trajectories)
        qec_cycles = sum(item.qec_cycles for item in trajectories)
        mean_policy_events = sum(item.detector_events for item in trajectories)
        mean_policy_exposures = sum(item.detector_exposures for item in trajectories)
        exploration_events = sum(item.auxiliary_detector_events for item in trajectories)
        exploration_exposures = sum(item.auxiliary_detector_exposures for item in trajectories)
        budget_classes = {item.candidate_budget_class for item in trajectories
                          if item.candidate_budget_class != "not_applicable"}
        budget_class = next(iter(budget_classes)) if len(budget_classes) == 1 else (
            "mixed" if budget_classes else "not_applicable")
        timing_invalid: list[str] = []
        if name in {"full_control_detector_rl", "predictive_hdfa_no_residual",
                    "predictive_hdfa_residual_rl"}:
            for trajectory in trajectories:
                if trajectory.timing is None:
                    timing_invalid.append(f"interval {trajectory.interval}: timing missing")
                else:
                    timing_invalid.extend(
                        f"interval {trajectory.interval}: {reason}"
                        for reason in trajectory.timing.validate())
        t90 = next(item for item in endpoints if item.target_fraction == .90)
        return ArmMetrics(
            scenario.scenario_id, seed, name, qec_cycles,
            sum(item.candidate_evaluations for item in trajectories),
            sum(item.diagnostic_shots for item in trajectories),
            sum(item.diagnostic_downtime_s for item in trajectories),
            mean_policy_events/max(1, mean_policy_exposures), excess,
            generic_logical/max(1, qec_cycles),
            sum(item.exploration_damage for item in trajectories),
            final_events/max(1, final_exposures), t90.intervals_after_peak,
            rollback_count, trajectories[-1].elapsed_time_s if trajectories else 0.0,
            final_events, final_exposures, status, censor_reason, tuple(missing), endpoints,
            worst_region, self._fit(rates, base), logical_failures, logical_shots,
            logical_probability, per_round,
            sum(len(item.lifecycle_violations) for item in trajectories),
            max((item.bootstrap_count for item in trajectories), default=0),
            disturbance_realization_id, disturbance_epoch_s,
            baseline.detector_rate if baseline is not None else None,
            baseline.detector_rate_ci95 if baseline is not None else None,
            baseline.observation_hash if baseline is not None else "",
            controller_truth_access_count,
            mean_policy_events/max(1, mean_policy_exposures),
            (exploration_events/exploration_exposures if exploration_exposures else None),
            sum(item.exploration_excess_detector_events for item in trajectories),
            sum(item.evaluation_policy_cycles for item in trajectories),
            sum(item.candidate_cycles for item in trajectories),
            budget_class,
            sum(len(item.physical_rollback_failures) for item in trajectories),
            max((item.recovery_count for item in trajectories), default=0),
            tuple(dict.fromkeys(timing_invalid)),
            controller_e2e,
            ((total_support-controller_e2e)
             if total_support is not None and controller_e2e is not None else 0.0),
            sum(item.qec_cycles for item in followup),
            total_support,
            tuple(followup),
        )

    def _run_arm(self, scenario: BenchmarkScenario, seed: int, name: str,
                 factory: Callable[[int], BenchmarkArm],
                 prepared: _MatchedPreparation | None = None) -> tuple[
                     ArmMetrics, tuple[IntervalTrajectory, ...], PreDisturbanceBaseline | None]:
        device = (prepared.device.clone() if prepared is not None else ScalableQECDevice(SimulatorConfig(
            qubit_count=self.config.qubit_count, code_distance=self.config.code_distance,
            cycle_period_s=self.config.cycle_period_s,
            controller_latency_s=min(.002, 2*self.config.cycle_period_s),
            disturbances_enabled_at_start=False, seed=seed, processes=scenario.processes,
        )))
        arm = factory(seed)
        external_bootstrap = None
        external_counts = (0, 0, 0.0)
        baseline = None
        missing: list[str] = []
        status, censor_reason = "completed", None
        # Every arm receives the same dedicated, stationary Stage 0.  Product arms attach
        # that validated result, so they execute the real Stage 1--7 product loop without
        # rerunning Stage 0 after the randomized disturbance phase has begun.
        try:
            if prepared is not None:
                expected = prepared.device.counterfactual_state_fingerprint()
                observed = device.counterfactual_state_fingerprint()
                clone_isolated = not device.shares_mutable_state_with(prepared.device)
                if expected != observed:
                    raise RuntimeError("matched clone state fingerprint differs before controller preparation")
                if not clone_isolated:
                    raise RuntimeError("matched clone shares mutable policy, RNG, or disturbance state")
                external_bootstrap = prepared.bootstrap
                external_counts = prepared.accounting
                baseline = replace(prepared.baseline, arm=name,
                                   clone_isolation_verified=True)
                prepare = getattr(arm, "prepare", None)
                if callable(prepare):
                    prepare(device, external_bootstrap)
            else:
                external_bootstrap, shots, qec_cycles, downtime = self._common_bootstrap(device)
                external_counts = (shots, qec_cycles, downtime)
                if external_bootstrap.health.status is not HealthStatus.PASSED:
                    status, censor_reason = "censored", "Stage-0 QEC-operability gate failed"
                else:
                    prepare = getattr(arm, "prepare", None)
                    if callable(prepare):
                        prepare(device, external_bootstrap)
                    baseline = self._pre_disturbance_baseline(scenario, seed, name, device)
                    external_counts = (shots, qec_cycles + baseline.cycles, downtime)
        except Exception as error:
            status, censor_reason = "missing", f"pre-randomization exception: {type(error).__name__}: {error}"
            missing.append(censor_reason)
        trajectories: list[IntervalTrajectory] = []
        controller_truth_access_count = 0
        rollback_count = 0
        limit = min(self.config.intervals, self.config.censoring_limit_intervals or self.config.intervals)
        declared_censoring = limit < self.config.intervals
        if status == "completed":
            for interval in range(limit):
                access_start = len(device.oracle_access_log)
                truth_counted = False
                try:
                    output = arm.run_interval(device, self.config.cycles_per_interval, interval)
                    controller_accesses = tuple(
                        purpose for _, purpose in device.oracle_access_log[access_start:])
                    if name != "oracle":
                        controller_truth_access_count += len(controller_accesses)
                    truth_counted = True
                    rollback_count += output.rollback_count
                    logical_started_ns = time.perf_counter_ns()
                    logical = (self._logical.evaluate_device(
                        device, seed=self._logical_seed(scenario, seed, interval))
                        if self._logical is not None else None)
                    logical_elapsed_s = (time.perf_counter_ns()-logical_started_ns)/1e9
                    if output.timing is not None:
                        output = replace(
                            output, timing=replace(
                                output.timing,
                                offline_logical_evaluation_s=logical_elapsed_s))
                    if logical is None:
                        missing.append(self._logical_error or "logical stack unavailable")
                    trajectories.append(self._trajectory(
                        scenario, seed, name, interval, device, output, logical,
                        external_bootstrap, external_counts, controller_accesses))
                except QECOperabilityError as error:
                    if name != "oracle" and not truth_counted:
                        controller_truth_access_count += len(device.oracle_access_log[access_start:])
                    status, censor_reason = "censored", str(error)
                    break
                except RecoveryCertificationError as error:
                    if name != "oracle" and not truth_counted:
                        controller_truth_access_count += len(
                            device.oracle_access_log[access_start:])
                    status, censor_reason = "censored", str(error)
                    break
                except Exception as error:
                    if name != "oracle" and not truth_counted:
                        controller_truth_access_count += len(device.oracle_access_log[access_start:])
                    status, censor_reason = "missing", f"interval exception: {type(error).__name__}: {error}"
                    missing.append(censor_reason)
                    break
        if status == "completed" and declared_censoring:
            status, censor_reason = "censored", "declared interval censoring limit reached"
        followup: list[EndpointFollowupObservation] = []
        if status != "missing" and self.config.minimum_e2e_followup_support_s is not None:
            valid_timings = tuple(item.timing for item in trajectories)
            if valid_timings and all(
                    item is not None and not item.validate() for item in valid_timings):
                support = sum(item.e2e_convergence_time_s
                              for item in valid_timings if item is not None)
                target_support = max(
                    self.config.minimum_e2e_followup_support_s,
                    self.config.compute_rmst_horizon_s + self.config.rmst_support_margin_s)
                index = 0
                while support < target_support:
                    remaining = target_support-support
                    cycles = min(
                        self.config.endpoint_followup_chunk_cycles,
                        max(1, math.ceil(remaining/self.config.cycle_period_s)))
                    batch = device.acquire(cycles, retain_records=False)
                    support += cycles*self.config.cycle_period_s
                    followup.append(EndpointFollowupObservation(
                        index, cycles, batch.detector_events, batch.detector_exposures,
                        batch.detector_rate, support,
                        batch.policy_activation.policy_hash, batch.batch_id))
                    index += 1
        metrics = self._summarize_run(
            scenario, seed, name, trajectories, status, censor_reason, missing, rollback_count,
            baseline, device.disturbance_realization_id, device.disturbance_epoch_s,
            controller_truth_access_count, followup)
        return metrics, tuple(trajectories), baseline

    @staticmethod
    def _paired_statistics(metrics: Sequence[ArmMetrics]) -> tuple[MatchedPairStatistic, ...]:
        reference = "predictive_hdfa_residual_rl"
        by_key = {(item.scenario_id, item.seed, item.arm): item for item in metrics}
        outcomes = (
            "integrated_excess_detector_events", "exploration_damage",
            "final_detector_event_rate", "logical_circuit_failure_probability",
        )
        output: list[MatchedPairStatistic] = []
        scenario_ids = sorted({item.scenario_id for item in metrics})
        seeds = sorted({item.seed for item in metrics})
        arms = sorted({item.arm for item in metrics if item.arm != reference})
        for arm in arms:
            for outcome in outcomes:
                differences: list[float] = []
                missing = 0
                # Seeds, not scenario/seed rows, are the independent experimental
                # units.  Scenarios are fixed repeated conditions and are averaged
                # within seed before constructing a confidence interval.
                for seed in seeds:
                    seed_differences = []
                    for scenario_id in scenario_ids:
                        comparator = by_key.get((scenario_id, seed, arm))
                        staged = by_key.get((scenario_id, seed, reference))
                        left = getattr(comparator, outcome, None) if comparator else None
                        right = getattr(staged, outcome, None) if staged else None
                        if (comparator is None or staged is None
                                or comparator.completion_status != "completed"
                                or staged.completion_status != "completed"
                                or left is None or right is None
                                or not math.isfinite(left) or not math.isfinite(right)):
                            missing += 1
                            seed_differences = []
                            break
                        seed_differences.append(left-right)
                    if seed_differences:
                        differences.append(statistics.fmean(seed_differences))
                output.append(MatchedPairStatistic(
                    arm, reference, outcome, len(differences), missing, tuple(differences),
                    _mean_ci(differences),
                    "paired seed-level mean across declared scenarios; positive values favour the staged reference for cost/risk outcomes"))
        return tuple(output)

    @staticmethod
    def _recovery_summaries(metrics: Sequence[ArmMetrics]) -> tuple[RecoverySummary, ...]:
        def kaplan_meier(endpoints: Sequence[RecoveryEndpoint]) -> tuple[float | None, float | None]:
            observations = sorted(
                (float(item.detector_cycles if item.detector_cycles is not None else item.censoring_cycles),
                 item.status == "reached")
                for item in endpoints if item.status != "missing")
            if not observations:
                return None, None
            survival, area, previous, at_risk, median = 1.0, 0.0, 0.0, len(observations), None
            for time in sorted({item[0] for item in observations}):
                area += survival * (time-previous)
                events = sum(1 for value, event in observations if value == time and event)
                censored = sum(1 for value, event in observations if value == time and not event)
                if events and at_risk:
                    survival *= 1-events/at_risk
                    if median is None and survival <= .5:
                        median = time
                at_risk -= events+censored
                previous = time
            return median, area

        output = []
        for arm in sorted({item.arm for item in metrics}):
            rows = [item for item in metrics if item.arm == arm]
            for target in RECOVERY_TARGETS:
                endpoints = [next(endpoint for endpoint in row.recovery_endpoints
                                  if endpoint.target_fraction == target) for row in rows]
                reached = [item.detector_cycles for item in endpoints
                           if item.status == "reached" and item.detector_cycles is not None]
                censored = [item for item in endpoints if item.status == "censored"]
                missing = [item for item in endpoints if item.status == "missing"]
                seed_fractions = []
                for seed in sorted({row.seed for row in rows}):
                    seed_endpoints = [next(endpoint for endpoint in row.recovery_endpoints
                                           if endpoint.target_fraction == target)
                                      for row in rows if row.seed == seed]
                    evaluable = [item for item in seed_endpoints if item.status != "missing"]
                    if evaluable:
                        seed_fractions.append(
                            sum(item.status == "reached" for item in evaluable) / len(evaluable))
                fraction_ci = _mean_ci(seed_fractions)
                fraction_bounds = (max(0.0, fraction_ci.lower), min(1.0, fraction_ci.upper)) \
                    if fraction_ci is not None else None
                survival_median, restricted_mean = kaplan_meier(endpoints)
                output.append(RecoverySummary(
                    arm, target, len(endpoints), len(reached), len(censored), len(missing),
                    len(seed_fractions), len(reached)/max(1, len(endpoints)-len(missing)),
                    fraction_bounds,
                    survival_median, restricted_mean,
                ))
        return tuple(output)

    @staticmethod
    def _endpoint_for(metrics: ArmMetrics, target: float) -> RecoveryEndpoint:
        return next(item for item in metrics.recovery_endpoints if item.target_fraction == target)

    def _compute_aware_gates(self, pairs: Sequence[tuple[
            ArmMetrics | None, ArmMetrics | None]]) -> tuple[AcceptanceGate, AcceptanceGate]:
        usable = [(left, right) for left, right in pairs
                  if left is not None and right is not None]
        missing_reasons: list[str] = []
        by_seed: dict[int, list[tuple[RecoveryEndpoint, RecoveryEndpoint]]] = {}
        supports: list[float] = []
        staged_safety_censor = False
        reached_components: dict[str, list[float]] = {
            "full_qec_s": [], "staged_qec_s": [],
            "full_compute_s": [], "staged_compute_s": [],
            "full_diagnostic_s": [], "staged_diagnostic_s": [],
            "full_actuation_s": [], "staged_actuation_s": [],
        }
        for left, right in usable:
            assert left is not None and right is not None
            left_endpoint = self._endpoint_for(left, .90)
            right_endpoint = self._endpoint_for(right, .90)
            if left.timing_invalidity_reasons or right.timing_invalidity_reasons:
                missing_reasons.extend((*left.timing_invalidity_reasons,
                                        *right.timing_invalidity_reasons))
            for label, endpoint in (("full", left_endpoint), ("staged", right_endpoint)):
                if endpoint.timing_status != "valid":
                    missing_reasons.append(
                        f"{left.scenario_id}/{left.seed}/{label}: {endpoint.timing_status}")
                if endpoint.censoring_e2e_time_s is None:
                    missing_reasons.append(
                        f"{left.scenario_id}/{left.seed}/{label}: censoring time missing")
                else:
                    supports.append(endpoint.censoring_e2e_time_s)
                if endpoint.status == "reached":
                    components = endpoint.e2e_components_s
                    reached_components[f"{label}_qec_s"].append(
                        float(components.get("qec_acquisition_s", 0.0)))
                    reached_components[f"{label}_compute_s"].append(
                        float(components.get("online_compute_critical_s", 0.0)))
                    reached_components[f"{label}_diagnostic_s"].append(
                        float(components.get("diagnostic_downtime_s", 0.0)))
                    reached_components[f"{label}_actuation_s"].append(
                        float(components.get("actuation_acknowledgement_s", 0.0)))
            if right.completion_status != "completed":
                staged_safety_censor = True
            by_seed.setdefault(left.seed, []).append((left_endpoint, right_endpoint))

        independent_seeds = tuple(sorted(by_seed))
        horizon = self.config.e2e_rmst_horizon_s
        if horizon is None and supports:
            horizon = min(supports)
        if horizon is None or horizon <= 0:
            missing_reasons.append("common RMST horizon cannot be established")
            horizon = 0.0
        if self.config.e2e_rmst_horizon_s is not None:
            for seed, rows in by_seed.items():
                for left_endpoint, right_endpoint in rows:
                    for label, endpoint in (("full", left_endpoint),
                                            ("staged", right_endpoint)):
                        support = endpoint.censoring_e2e_time_s
                        reached_before = (endpoint.status == "reached"
                                          and endpoint.e2e_time_s is not None
                                          and endpoint.e2e_time_s <= horizon)
                        if not reached_before and (support is None or support+1e-12 < horizon):
                            missing_reasons.append(
                                f"seed {seed}/{label}: observation support ends before frozen horizon")

        def observations(seed_sample: Sequence[int], arm_index: int
                         ) -> list[tuple[float, bool]]:
            output: list[tuple[float, bool]] = []
            for seed in seed_sample:
                for endpoints in by_seed[seed]:
                    endpoint = endpoints[arm_index]
                    if endpoint.status == "reached" and endpoint.e2e_time_s is not None:
                        output.append((endpoint.e2e_time_s, True))
                    elif endpoint.censoring_e2e_time_s is not None:
                        output.append((endpoint.censoring_e2e_time_s, False))
            return output

        estimand: dict[str, object] = {
            "estimand_id": "compute-aware-rmst-net-convergence-gain.v1",
            "formula": (f"RMST_e2e({self.config.gate_reference_arm})"
                        f"-RMST_e2e({self.config.gate_treatment_arm})"),
            "target_fraction": 0.90,
            "observed_only": True,
            "rmst_horizon_s": horizon,
            "confidence_method": "seed-cluster nonparametric bootstrap; one-sided lower 95% bound",
            "bootstrap_replicates": self.config.compute_bootstrap_replicates,
            "bootstrap_seed": self.config.compute_bootstrap_seed,
            "independent_seed_count": len(independent_seeds),
            "cluster_unit": "independent disturbance seed",
            "declared_pair_count": len(pairs),
            "included_pair_count": len(usable),
            "complete_case_deletion": False,
            "timing_formula": "QEC acquisition + diagnostic downtime + actuation/acknowledgement + online critical compute",
            "online_offline_boundary": "logical evaluation, serialization and report analysis excluded; simulator host overhead reported separately",
            "observed_reached_component_means_s": {
                key: statistics.fmean(values) if values else None
                for key, values in reached_components.items()},
            "invalidity_reasons": tuple(dict.fromkeys(missing_reasons)),
        }
        enough_seeds = len(independent_seeds) >= self.config.minimum_compute_independent_seeds
        complete_pairs = len(usable) == len(pairs)
        if missing_reasons or not enough_seeds or not complete_pairs or horizon <= 0:
            reason = (
                "compute-aware gate is non-evaluable: complete symmetric timing, all declared pairs, "
                "a common horizon and sufficient independent seed clusters are mandatory")
            primary = AcceptanceGate(
                "compute_aware_rmst_net_convergence_gain", "not_evaluable",
                None, 0.0, reason, None, len(usable), True, estimand)
            tail = AcceptanceGate(
                "compute_aware_e2e_tail_noninferiority", "not_evaluable",
                None, self.config.e2e_tail_noninferiority_margin_s,
                "tail safeguard requires the same complete timed risk set as the primary RMST estimand",
                None, len(usable), True, {
                    **estimand, "tail_quantile": self.config.e2e_tail_quantile})
            if self.config.estimator_schema_version == "estimators.v2":
                primary = replace(primary, estimators=GateEstimators(
                    gate_decision_statistic=None, gate_threshold=0.0,
                    gate_status="not_evaluable"))
                tail = replace(tail, estimators=GateEstimators(
                    gate_decision_statistic=None,
                    gate_threshold=self.config.e2e_tail_noninferiority_margin_s,
                    gate_status="not_evaluable"))
            return primary, tail

        full_obs = observations(independent_seeds, 0)
        staged_obs = observations(independent_seeds, 1)
        full_rmst = _restricted_mean_time(full_obs, horizon)
        staged_rmst = _restricted_mean_time(staged_obs, horizon)
        gain = full_rmst-staged_rmst
        rng = random.Random(self.config.compute_bootstrap_seed)
        gains: list[float] = []
        tail_differences: list[float] = []
        for _ in range(self.config.compute_bootstrap_replicates):
            sample = [rng.choice(independent_seeds) for _ in independent_seeds]
            full_sample = observations(sample, 0)
            staged_sample = observations(sample, 1)
            gains.append(
                _restricted_mean_time(full_sample, horizon)
                - _restricted_mean_time(staged_sample, horizon))
            full_times = [min(time_s, horizon) for time_s, event in full_sample]
            staged_times = [min(time_s, horizon) for time_s, event in staged_sample]
            tail_differences.append(
                _quantile(staged_times, self.config.e2e_tail_quantile)
                - _quantile(full_times, self.config.e2e_tail_quantile))
        alpha = 1-self.config.compute_one_sided_confidence
        lower = _quantile(gains, alpha)
        upper = _quantile(gains, 1-alpha)
        estimand.update({
            "rmst_full_s": full_rmst,
            "rmst_staged_s": staged_rmst,
            "net_convergence_gain_s": gain,
            "one_sided_lower_confidence_bound_s": lower,
            "staged_safety_censoring": staged_safety_censor,
        })
        primary_status = (
            "pass" if lower > 0 and not staged_safety_censor else "fail")
        primary_ci = ConfidenceInterval(
            lower, gain, upper, self.config.compute_one_sided_confidence)
        primary = AcceptanceGate(
            "compute_aware_rmst_net_convergence_gain", primary_status,
            gain, 0.0,
            "one-sided 95% lower confidence bound on seed-clustered matched RMST E2E gain must exceed zero; staged safety censoring fails separately and here",
            (primary_ci if self.config.estimator_schema_version == "legacy.v1" else None),
            len(usable), True, estimand)
        if self.config.estimator_schema_version == "estimators.v2":
            primary = replace(primary, estimators=GateEstimators(
                rmst_difference=gain,
                rmst_ci95=primary_ci,
                gate_decision_statistic=lower, gate_threshold=0.0,
                gate_status=primary_status))

        full_tail = _quantile(
            [min(time_s, horizon) for time_s, _ in full_obs],
            self.config.e2e_tail_quantile)
        staged_tail = _quantile(
            [min(time_s, horizon) for time_s, _ in staged_obs],
            self.config.e2e_tail_quantile)
        tail_difference = staged_tail-full_tail
        tail_upper = _quantile(
            tail_differences, self.config.compute_one_sided_confidence)
        tail_status = (
            "pass" if tail_upper <= self.config.e2e_tail_noninferiority_margin_s
            and not staged_safety_censor else "fail")
        tail_estimand = {
            **estimand, "tail_quantile": self.config.e2e_tail_quantile,
            "full_tail_s": full_tail, "staged_tail_s": staged_tail,
            "staged_minus_full_tail_s": tail_difference,
            "one_sided_upper_confidence_bound_s": tail_upper,
            "fixed_margin_s": self.config.e2e_tail_noninferiority_margin_s,
        }
        tail_ci = ConfidenceInterval(
            _quantile(tail_differences, alpha), tail_difference, tail_upper,
            self.config.compute_one_sided_confidence)
        tail = AcceptanceGate(
            "compute_aware_e2e_tail_noninferiority", tail_status,
            tail_difference, self.config.e2e_tail_noninferiority_margin_s,
            "seed-cluster bootstrap upper confidence bound for the frozen E2E tail difference must not exceed the fixed margin",
            (tail_ci if self.config.estimator_schema_version == "legacy.v1" else None),
            len(usable), True, tail_estimand)
        if self.config.estimator_schema_version == "estimators.v2":
            tail = replace(tail, estimators=GateEstimators(
                tail_difference=tail_difference,
                tail_ci95=tail_ci,
                gate_decision_statistic=tail_upper,
                gate_threshold=self.config.e2e_tail_noninferiority_margin_s,
                gate_status=tail_status))
        return primary, tail

    def _gates(self, metrics: Sequence[ArmMetrics]) -> tuple[AcceptanceGate, ...]:
        structured = {scenario.scenario_id for scenario in self.scenarios if scenario.structured}
        baseline = self.config.gate_reference_arm
        staged = self.config.gate_treatment_arm
        by_key = {(item.scenario_id, item.seed, item.arm): item for item in metrics}
        pairs = [(by_key.get((scenario, seed, baseline)), by_key.get((scenario, seed, staged)))
                 for scenario in structured for seed in self.config.seeds]
        observable_pairs = [(left, right) for left, right in pairs if left is not None and right is not None
                            and left.completion_status != "missing" and right.completion_status != "missing"]
        completed_pairs = [(left, right) for left, right in observable_pairs
                           if left.completion_status == "completed" and right.completion_status == "completed"]
        minimum_pairs = 2

        def seed_level(values: Sequence[tuple[int, float]]) -> list[float]:
            grouped: dict[int, list[float]] = {}
            for seed, value in values:
                grouped.setdefault(seed, []).append(value)
            return [statistics.fmean(grouped[seed]) for seed in sorted(grouped)]

        sample_ratios: list[float] = []
        staged_censored = 0
        sample_values: list[tuple[int, float]] = []
        for left, right in observable_pairs:
            left_endpoint, right_endpoint = self._endpoint_for(left, .90), self._endpoint_for(right, .90)
            if right_endpoint.status == "censored":
                staged_censored += 1
                continue
            if right_endpoint.status != "reached":
                continue
            # Candidate evaluations are the predeclared adaptation sample unit.  A
            # censored baseline contributes its conservative observed lower bound.
            left_samples = (left_endpoint.candidate_evaluations
                            if left_endpoint.status == "reached"
                            else left_endpoint.censoring_candidate_evaluations)
            right_samples = right_endpoint.candidate_evaluations
            if left_samples is None or right_samples is None:
                continue
            left_samples, right_samples = max(1, left_samples), max(1, right_samples)
            sample_ratios.append(left_samples/right_samples)
            sample_values.append((left.seed, left_samples/right_samples))
        sample_ci = _mean_ci(seed_level(sample_values))
        if len(pairs) < minimum_pairs or len(observable_pairs) != len(pairs):
            sample_status = "not_evaluable"
        elif staged_censored or not sample_ratios:
            sample_status = "fail"
        else:
            sample_status = "pass" if min(sample_ratios) >= 10.0 else "fail"
        compute_gate, tail_gate = self._compute_aware_gates(pairs)
        sample_gate = AcceptanceGate(
            "sample_efficiency_to_observed_90pct_recovery", sample_status,
            min(sample_ratios) if sample_ratios else 0.0, 10.0,
            "paired cumulative candidate evaluations through an observed (never extrapolated) 90% recovery target; staged censoring is failure",
            (sample_ci if self.config.estimator_schema_version == "legacy.v1" else None),
            len(observable_pairs), False, {
                "role": "secondary diagnostic retained for continuity; not an acceptance rule",
                "legacy_target_ratio": 10.0,
            })
        if self.config.estimator_schema_version == "estimators.v2":
            seed_ratios = seed_level(sample_values)
            sample_gate = replace(sample_gate, estimators=GateEstimators(
                worst_matched_ratio=min(sample_ratios) if sample_ratios else None,
                median_matched_ratio=(statistics.median(sample_ratios)
                                      if sample_ratios else None),
                cluster_aggregate_ratio=(statistics.fmean(seed_ratios)
                                         if seed_ratios else None),
                cluster_aggregate_ci95=sample_ci,
                gate_decision_statistic=(min(sample_ratios)
                                         if sample_ratios else None),
                gate_threshold=10.0, gate_status=sample_status))
        gates = [compute_gate, tail_gate, sample_gate]

        def paired_ratio_gate(identifier: str, field_name: str, required: float, rationale: str) -> AcceptanceGate:
            ratios = []
            ratio_values: list[tuple[int, float]] = []
            for left, right in completed_pairs:
                numerator, denominator = getattr(left, field_name), getattr(right, field_name)
                if denominator == 0:
                    ratios.append(numerator/1e-12 if numerator > 0 else 1.0)
                else:
                    ratios.append(numerator/denominator)
                ratio_values.append((left.seed, ratios[-1]))
            finite = [value for value in ratios if math.isfinite(value)]
            ci = _mean_ci(seed_level([(seed, value) for seed, value in ratio_values if math.isfinite(value)]))
            if len(pairs) < minimum_pairs or len(observable_pairs) != len(pairs):
                status = "not_evaluable"
                measured = None
            elif len(completed_pairs) != len(pairs):
                measured = min(finite) if finite else None
                status = "fail"
            else:
                measured = min(finite) if finite else None
                status = "pass" if measured is not None and measured >= required else "fail"
            gate = AcceptanceGate(
                identifier, status, measured, required, rationale,
                (ci if self.config.estimator_schema_version == "legacy.v1" else None),
                len(completed_pairs))
            if self.config.estimator_schema_version == "estimators.v2":
                clustered = seed_level([
                    (seed, value) for seed, value in ratio_values
                    if math.isfinite(value)])
                gate = replace(gate, estimators=GateEstimators(
                    worst_matched_ratio=measured,
                    median_matched_ratio=(statistics.median(finite)
                                          if finite else None),
                    cluster_aggregate_ratio=(statistics.fmean(clustered)
                                             if clustered else None),
                    cluster_aggregate_ci95=ci,
                    gate_decision_statistic=measured,
                    gate_threshold=required, gate_status=status))
            return gate

        gates.append(paired_ratio_gate(
            "integrated_excess_edr", "integrated_excess_detector_events",
            self.config.integrated_excess_required_ratio,
            "worst matched baseline/staged ratio of area under the excess-EDR curve"))
        gates.append(paired_ratio_gate(
            "exploration_damage", "exploration_damage",
            self.config.exploration_damage_required_ratio,
            "worst matched baseline/staged ratio under identical disturbance tapes"))

        one_interval = []
        for _, right in observable_pairs:
            endpoint = self._endpoint_for(right, .50)
            one_interval.append(
                endpoint.status == "reached"
                and endpoint.intervals_after_peak is not None
                and endpoint.intervals_after_peak <= 1)
        if len(pairs) < minimum_pairs or len(observable_pairs) != len(pairs):
            one_status, one_fraction = "not_evaluable", None
        else:
            one_fraction = sum(one_interval)/len(one_interval)
            one_status = ("pass" if one_fraction >= self.config.one_interval_required_fraction
                          else "fail")
        gates.append(AcceptanceGate(
            "one_interval_recurring_recovery", one_status, one_fraction,
            self.config.one_interval_required_fraction,
            "fraction of structured matched runs with observed 50% recovery within one control interval",
            None, len(observable_pairs)))
        if self.config.estimator_schema_version == "estimators.v2":
            gates[-1] = replace(gates[-1], estimators=GateEstimators(
                gate_decision_statistic=one_fraction,
                gate_threshold=self.config.one_interval_required_fraction,
                gate_status=one_status))

        final_values = [(left.seed, right.final_detector_event_rate-left.final_detector_event_rate)
                        for left, right in completed_pairs]
        final_deltas = seed_level(final_values)
        final_ci = _mean_ci(final_deltas)
        if len(pairs) < minimum_pairs or len(observable_pairs) != len(pairs):
            final_status = "not_evaluable"
            final_delta = None
        elif len(completed_pairs) != len(pairs):
            # A declared controller censor is part of the composite safety/performance
            # estimand.  It is an observed failure of noninferiority, not missing data.
            final_status = "fail"
            final_delta = final_ci.estimate if final_ci is not None else None
        elif len(final_deltas) < 2 or final_ci is None:
            final_status = "not_evaluable"
            final_delta = None
        else:
            final_delta = final_ci.estimate
            final_status = ("pass" if final_ci.upper <= self.config.final_rate_noninferiority_margin
                            else "fail")
        gates.append(AcceptanceGate(
            "no_final_performance_loss", final_status, final_delta,
            self.config.final_rate_noninferiority_margin,
            "paired 95% confidence interval for staged-minus-full-RL final detector rate",
            final_ci, len(final_deltas)))
        if self.config.estimator_schema_version == "estimators.v2":
            gates[-1] = replace(gates[-1], estimators=GateEstimators(
                gate_decision_statistic=(final_ci.upper if final_ci else None),
                gate_threshold=self.config.final_rate_noninferiority_margin,
                gate_status=final_status))

        if self.config.logical_failure_noninferiority_margin is not None:
            logical_values = [(left.seed,
                               right.logical_circuit_failure_probability
                               - left.logical_circuit_failure_probability)
                              for left, right in completed_pairs
                              if left.logical_circuit_failure_probability is not None
                              and right.logical_circuit_failure_probability is not None]
            logical_deltas = seed_level(logical_values)
            logical_ci = _mean_ci(logical_deltas)
            if len(logical_deltas) < 2 or logical_ci is None:
                logical_status, logical_delta = "not_evaluable", None
            else:
                logical_delta = logical_ci.estimate
                logical_status = ("pass" if logical_ci.upper <=
                                  self.config.logical_failure_noninferiority_margin
                                  else "fail")
            logical_gate = AcceptanceGate(
                "no_logical_performance_loss", logical_status, logical_delta,
                self.config.logical_failure_noninferiority_margin,
                "paired seed-cluster 95% CI for treatment-minus-reference circuit-level logical failure probability",
                logical_ci, len(logical_deltas))
            if self.config.estimator_schema_version == "estimators.v2":
                logical_gate = replace(logical_gate, estimators=GateEstimators(
                    gate_decision_statistic=(logical_ci.upper if logical_ci else None),
                    gate_threshold=self.config.logical_failure_noninferiority_margin,
                    gate_status=logical_status))
            gates.append(logical_gate)

        if self.config.residual_benefit_scenario_id is not None:
            benefit_values = [(left.seed,
                               right.integrated_excess_detector_events
                               - left.integrated_excess_detector_events)
                              for left, right in completed_pairs
                              if left.scenario_id == self.config.residual_benefit_scenario_id]
            benefit_deltas = seed_level(benefit_values)
            benefit_ci = _mean_ci(benefit_deltas)
            if len(benefit_deltas) < 2 or benefit_ci is None:
                benefit_status, benefit_delta = "not_evaluable", None
            else:
                benefit_delta = benefit_ci.estimate
                benefit_status = "pass" if benefit_ci.upper < 0.0 else "fail"
            benefit_gate = AcceptanceGate(
                "residual_benefit_in_predeclared_scenario", benefit_status,
                benefit_delta, 0.0,
                "upper paired seed-cluster 95% CI for treatment-minus-reference integrated excess EDR must be below zero in the frozen learnable-residual scenario",
                benefit_ci, len(benefit_deltas))
            if self.config.estimator_schema_version == "estimators.v2":
                benefit_gate = replace(benefit_gate, estimators=GateEstimators(
                    gate_decision_statistic=(benefit_ci.upper if benefit_ci else None),
                    gate_threshold=0.0, gate_status=benefit_status))
            gates.append(benefit_gate)
        return tuple(gates)

    def run(self) -> BenchmarkReport:
        preflight_manifest_hash = self._require_preflight()
        run_results = []
        for scenario in self.scenarios:
            for seed in self.config.seeds:
                prepared = self._prepare_matched_state(scenario, seed)
                for name, factory in self.arm_factories.items():
                    run_results.append(self._run_arm(
                        scenario, seed, name, factory, prepared))
        metrics = tuple(item[0] for item in run_results)
        trajectories = tuple(row for item in run_results for row in item[1])
        baselines = tuple(item[2] for item in run_results if item[2] is not None)
        matched = self._paired_statistics(metrics)
        recovery = self._recovery_summaries(metrics)
        gates = self._gates(metrics)
        logical_versions = {}
        for trajectory in trajectories:
            if trajectory.logical_evidence is not None:
                logical_versions = {
                    "stim": trajectory.logical_evidence.stim_version,
                    "pymatching": trajectory.logical_evidence.pymatching_version,
                    "stack": trajectory.logical_evidence.stack_id,
                }
                break
        provenance = _source_provenance(self.config, self.scenarios, logical_versions)
        invalidity: list[str] = []
        acceptance_failures: list[str] = []
        if len(self.config.seeds) < 2:
            invalidity.append("confidence intervals require at least two paired seeds")
        if self._logical is None:
            invalidity.append(self._logical_error or "logical stack unavailable")
        if any(item.completion_status == "missing" for item in metrics):
            invalidity.append("one or more arms contain missing run data")
        if any(item.missing_data_reasons for item in metrics):
            invalidity.append("one or more scientific metrics contain missing data")
        if any(gate.primary and gate.status == "not_evaluable" for gate in gates):
            invalidity.append("one or more acceptance metrics cannot be evaluated")
        primary_metrics = [item for item in metrics if item.arm in PRIMARY_ARMS]
        primary_baselines = [item for item in baselines if item.arm in PRIMARY_ARMS]
        expected_primary_runs = len(self.scenarios) * len(self.config.seeds) * len(PRIMARY_ARMS)
        stationary_stage0 = all(
            not item.disturbances_armed_during_acquisition
            and item.evaluation_only_max_abs_latent <= 1e-12
            and item.evaluation_only_max_abs_process <= 1e-12
            for item in primary_baselines)
        held_out_baseline = (
            len(primary_baselines) == expected_primary_runs
            and all(item.cycles == self.config.pre_disturbance_baseline_cycles
                    and item.detector_exposures > 0 for item in primary_baselines)
        )
        baseline_sets: dict[tuple[str, int], set[str]] = {}
        for item in primary_baselines:
            baseline_sets.setdefault((item.scenario_id, item.seed), set()).add(item.observation_hash)
        expected_pairs = {(scenario.scenario_id, seed)
                          for scenario in self.scenarios for seed in self.config.seeds}
        matched_baselines = (
            set(baseline_sets) == expected_pairs
            and all(len(values) == 1 and "" not in values for values in baseline_sets.values())
        )
        def matched_field(field_name: str) -> bool:
            grouped: dict[tuple[str, int], set[object]] = {}
            for item in primary_baselines:
                value = getattr(item, field_name)
                if isinstance(value, Mapping):
                    value = deterministic_hash(dict(value))
                grouped.setdefault((item.scenario_id, item.seed), set()).add(value)
            return (set(grouped) == expected_pairs
                    and all(len(values) == 1 and "" not in values
                            for values in grouped.values()))
        matched_initial_physical = (
            matched_field("initial_physical_state_id")
            and matched_field("initial_disturbance_state_id")
            and matched_field("process_rng_state_hash")
            and matched_field("characterization_rng_state_hash")
            and matched_field("initial_policy_controls")
            and matched_field("initial_policy_id"))
        matched_controller_state = (
            matched_field("initial_controller_state_hash")
            and matched_field("initial_simulator_state_hash"))
        clone_isolation = (len(primary_baselines) == expected_primary_runs
                           and all(item.clone_isolation_verified
                                   for item in primary_baselines))
        matched_evaluators = (
            matched_field("detector_evaluator_config_hash")
            and matched_field("logical_evaluator_config_hash"))
        disturbance_sets: dict[tuple[str, int], set[str]] = {}
        for item in primary_metrics:
            disturbance_sets.setdefault((item.scenario_id, item.seed), set()).add(item.disturbance_realization_id)
        matched_disturbances = (
            set(disturbance_sets) == expected_pairs
            and all(len(values) == 1 and "" not in values for values in disturbance_sets.values())
        )
        epoch_sets: dict[tuple[str, int], list[float]] = {}
        for item in primary_metrics:
            if item.disturbance_epoch_s is not None:
                epoch_sets.setdefault((item.scenario_id, item.seed), []).append(item.disturbance_epoch_s)
        synchronized_onsets = (
            set(epoch_sets) == expected_pairs
            and all(len(values) == len(PRIMARY_ARMS) and max(values)-min(values) <= 1e-12
                    for values in epoch_sets.values())
        )
        truth_isolation = all(
            item.arm == "oracle" or item.controller_truth_access_count == 0
            for item in primary_metrics)
        design_audit = ExperimentalDesignAudit(
            "stationary-stage0-heldout-baseline-synchronized-onset-compute-aware.v2",
            stationary_stage0, held_out_baseline, matched_baselines,
            matched_disturbances, synchronized_onsets, truth_isolation,
            "Internal simulator acceptance evidence only; it does not reproduce the paper's hardware or prove real-QPU superiority.",
            matched_initial_physical, matched_controller_state,
            clone_isolation, matched_evaluators,
        )
        if not stationary_stage0:
            invalidity.append("Stage 0 or the held-out baseline was exposed to an active latent disturbance")
        if not held_out_baseline:
            invalidity.append("a primary arm lacks the predeclared held-out native-QEC baseline")
        if not matched_baselines:
            invalidity.append("primary arms did not share one baseline observation per scenario/seed")
        if not matched_disturbances:
            invalidity.append("primary arms did not share one disturbance realization per scenario/seed")
        if not synchronized_onsets:
            invalidity.append("primary arms did not use a synchronized post-bootstrap disturbance onset")
        if not truth_isolation:
            invalidity.append("a non-oracle primary controller accessed evaluation-only latent truth")
        if not matched_initial_physical:
            invalidity.append("primary arms did not share identical latent, policy, and RNG state before onset")
        if not matched_controller_state:
            invalidity.append("primary arms did not share identical simulator/controller state hashes")
        if not clone_isolation:
            invalidity.append("one or more primary arms shared mutable state with the matched template")
        if not matched_evaluators:
            invalidity.append("primary arms did not share detector/logical evaluator configuration")
        central_arm_ids = {
            "full_control_detector_rl", self.config.gate_reference_arm,
            self.config.gate_treatment_arm}
        central_metrics = [item for item in primary_metrics
                           if item.arm in central_arm_ids]
        if any(item.lifecycle_violation_count for item in central_metrics):
            acceptance_failures.append(
                "the staged controller or faithful full-control RL comparator recorded a lifecycle violation")
        if any(item.physical_rollback_failure_count for item in central_metrics):
            acceptance_failures.append(
                "the staged controller or faithful full-control RL comparator recorded an uncontained physical rollback-validation failure")
        if any(item.completion_status != "completed" for item in central_metrics):
            acceptance_failures.append(
                "the staged controller or faithful full-control RL comparator did not complete")
        authoritative = self.config.authoritative and not invalidity
        accepted = (authoritative and not acceptance_failures
                    and all(gate.status == "pass" for gate in gates if gate.primary))
        payload = {
            "config": asdict(self.config), "scenarios": [asdict(item) for item in self.scenarios],
            "metrics": [asdict(item) for item in metrics],
            "pre_disturbance_baselines": [asdict(item) for item in baselines],
            "trajectories": [asdict(item) for item in trajectories],
            "matched": [asdict(item) for item in matched], "recovery": [asdict(item) for item in recovery],
            "gates": [asdict(item) for item in gates], "provenance": asdict(provenance),
            "design_audit": asdict(design_audit), "authoritative": authoritative,
            "accepted": accepted, "invalidity": invalidity,
            "acceptance_failures": acceptance_failures,
            "evidence_records": [asdict(item) for item in canonical_benchmark_evidence()],
            "preflight_manifest_hash": preflight_manifest_hash,
        }
        report_issues = validate_report_payload(payload)
        if report_issues:
            invalidity.extend(f"report contract {item.code}: {item.message}" for item in report_issues)
            authoritative = False
            accepted = False
            payload["authoritative"] = False
            payload["accepted"] = False
            payload["invalidity"] = invalidity
        return BenchmarkReport(
            "evaluation.v5", self.config, self.scenarios, metrics, baselines, trajectories,
            matched, recovery, gates, provenance, design_audit, PRIMARY_ARMS,
            authoritative, accepted, tuple(invalidity), tuple(acceptance_failures),
            deterministic_hash(payload), canonical_benchmark_evidence(),
            tuple(f"{item.code}: {item.message}" for item in report_issues),
            preflight_manifest_hash)
