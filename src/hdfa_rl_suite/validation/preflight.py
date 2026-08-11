"""Fail-closed physical and algorithmic preflight for long QEC benchmarks."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping

from hdfa_rl_suite.common import deterministic_hash
from hdfa_rl_suite.common import TimingEnvironment
from hdfa_rl_suite import __version__
from hdfa_rl_suite.logical import (
    LogicalStackUnavailable,
    RotatedSurfaceCodeEvaluator,
    SurfaceCodeMemoryConfig,
)
from hdfa_rl_suite.simulator import (
    SIMULATOR_VERSION, DriftKind, LatentProcessSpec, ScalableQECDevice, SimulatorConfig,
)

from .common import ValidationCheck, ValidationReport, all_passed, finalize_report
from .controller_sanity import CONTROLLER_VERSION, ControllerSanityConfig, run_controller_validation
from .development_cohort import run_development_cohort
from .fault_matrix import run_fault_matrix_validation
from .lifecycle_sanity import run_lifecycle_validation
from .performance import run_performance_validation
from .plant_sanity import PlantSanityConfig, run_plant_validation
from .report_sanity import run_report_validation
from .sample_budget import SampleBudgetConfig, run_sample_budget_validation
from .post_comparison import run_post_comparison_validation
from .compute_sanity import run_compute_accounting_validation


@dataclass(frozen=True)
class PreflightConfig:
    plant: PlantSanityConfig = PlantSanityConfig()
    controller: ControllerSanityConfig = ControllerSanityConfig()
    sample_budget: SampleBudgetConfig = SampleBudgetConfig()


def source_tree_hash() -> str:
    root = Path(__file__).resolve().parents[3]
    files = sorted((root/"src").rglob("*.py")) + [root/"pyproject.toml"]
    return deterministic_hash([
        (str(path.relative_to(root)).replace("\\", "/"), path.read_bytes().hex())
        for path in files if path.exists()
    ])


@lru_cache(maxsize=8)
def _validation_components(config: PreflightConfig, current_source_hash: str,
                           plant_faults: tuple[str, ...],
                           controller_faults: tuple[str, ...],
                           budget_faults: tuple[str, ...],
                           lifecycle_faults: tuple[str, ...],
                           report_faults: tuple[str, ...]):
    """Cache immutable validation results only for an identical source tree."""
    del current_source_hash  # It is intentionally part of the cache key.
    return (
        run_plant_validation(config.plant, injected_faults=plant_faults),
        run_controller_validation(config.controller, injected_faults=controller_faults),
        run_sample_budget_validation(config.sample_budget, injected_faults=budget_faults),
        run_lifecycle_validation(injected_faults=lifecycle_faults),
        run_report_validation(injected_faults=report_faults),
        run_development_cohort(generate_figures=False),
        run_performance_validation(),
        run_fault_matrix_validation(),
        run_post_comparison_validation(),
        run_compute_accounting_validation(),
    )


def run_preflight(config: PreflightConfig = PreflightConfig(), *,
                  injected_faults: Iterable[str] = ()) -> ValidationReport:
    faults = set(injected_faults)
    plant_faults = {fault.split(":", 1)[1] for fault in faults if fault.startswith("plant:")}
    controller_faults = {fault.split(":", 1)[1] for fault in faults if fault.startswith("controller:")}
    budget_faults = {fault.split(":", 1)[1] for fault in faults if fault.startswith("budget:")}
    lifecycle_faults = {fault.split(":", 1)[1] for fault in faults if fault.startswith("lifecycle:")}
    report_faults = {fault.split(":", 1)[1] for fault in faults if fault.startswith("report:")}
    source_hash = source_tree_hash()
    (plant, controller, budget, lifecycle, report_schema, development,
     performance, fault_matrix, post_comparison,
     compute_accounting) = _validation_components(
        config, source_hash, tuple(sorted(plant_faults)),
        tuple(sorted(controller_faults)), tuple(sorted(budget_faults)),
        tuple(sorted(lifecycle_faults)), tuple(sorted(report_faults)))
    logical_versions: Mapping[str, str] = {}
    logical_shared_state = False
    logical_details = ""
    try:
        logical_device = ScalableQECDevice(SimulatorConfig(
            qubit_count=3, cycle_period_s=.001, controller_latency_s=0., seed=991,
            processes=(LatentProcessSpec("stationary", DriftKind.CONSTANT, {}, amplitude=0.),),
        ))
        detector_batch = logical_device.acquire(16, retain_records=False)
        logical = RotatedSurfaceCodeEvaluator(SurfaceCodeMemoryConfig(
            distance=3, rounds=3, shots=16)).evaluate_device(logical_device, seed=991)
        logical_shared_state = (
            logical.physical_state_id == detector_batch.physical_state_id
            and logical.policy_hash == detector_batch.policy_activation.policy_hash
            and logical.disturbance_state_id == detector_batch.disturbance_state_id
        )
        logical_versions = {"stim": logical.stim_version, "pymatching": logical.pymatching_version}
        logical_details = "detector batch and Stim/PyMatching evaluation share state, policy, and disturbance IDs"
    except (LogicalStackUnavailable, ValueError, RuntimeError) as error:
        logical_details = f"logical shared-state validation unavailable: {error}"
    if "stale_logical_state" in faults:
        logical_shared_state = False
        logical_details = "injected logical evidence used a stale physical state"

    def check_passed(report: ValidationReport, check_id: str) -> bool:
        return any(item.check_id == check_id and item.passed for item in report.checks)

    checks = [
        ValidationCheck("plant_validation_current", plant.passed,
                        {"report_hash": plant.report_hash}, "all canonical plant checks pass",
                        "No controller comparison is permitted on an invalid plant."),
        ValidationCheck("full_rl_validation_current", controller.passed,
                        {"report_hash": controller.report_hash}, "all controller ladder checks pass",
                        "The reference full-control learner must work before it becomes a comparator."),
        ValidationCheck("sample_budget_current", budget.passed,
                        {"report_hash": budget.report_hash,
                         "selected_budget": budget.metadata.get("selected_validated_reduced_budget")},
                        "one reduced budget and the paper-scale reference satisfy the declared protocol",
                        "Candidate-cycle reductions require an explicit finite-shot adequacy result."),
        ValidationCheck("no_disturbance_plant_sanity",
                        check_passed(plant, "no_disturbance_stationarity"), True,
                        "stationary fixed and optimum policies remain statistically equivalent",
                        "No hidden policy or cloning drift is allowed."),
        ValidationCheck("fixed_oracle_step_sanity",
                        check_passed(plant, "persistent_step_degrades_fixed"), True,
                        "persistent controllable steps degrade fixed control and remain recoverable",
                        "A controller comparison requires visible controllable plant damage."),
        ValidationCheck("periodic_calibration_ordering",
                        check_passed(plant, "oracle_periodic_fixed_ordering"), True,
                        "oracle <= periodic <= fixed at the declared cadence",
                        "Baseline ordering is checked independently of the staged controller."),
        ValidationCheck("disturbance_persistence_and_matched_cloning",
                        check_passed(plant, "ou_persistence_and_clone_identity"), True,
                        "OU state persists and cloned arms receive identical independent paths",
                        "Matched-arm differences may arise only from controller actions."),
        ValidationCheck("full_rl_analytic_convergence",
                        check_passed(controller, "analytic_convergence_both_sides"), True,
                        "full-control RL converges from both sides of a known convex optimum",
                        "Reward sign and optimizer direction are independently checked."),
        ValidationCheck("full_rl_static_detector_convergence",
                        check_passed(controller, "randomized_policy_recovery"), True,
                        "full-control RL recovers a spoiled valid detector policy",
                        "The comparator must function before use in a six-arm benchmark."),
        ValidationCheck("positive_gradient_alignment",
                        check_passed(controller, "static_sparse_gradient_alignment"), True,
                        "finite-shot gradient alignment exceeds the predeclared margin",
                        "Masks, sensitivity units, and candidate/reward indexing remain testable."),
        ValidationCheck("calibrated_start_no_regression",
                        check_passed(controller, "calibrated_start_no_regression"), True,
                        "the learned mean does not regress from a calibrated stationary start",
                        "Exploration damage cannot be substituted for mean-policy performance."),
        ValidationCheck("sample_budget_adequacy",
                        check_passed(budget, "candidate_budget_adequacy"), True,
                        "the selected candidate budget passes ranking, gradient, harm, and convergence gates",
                        "Underpowered candidate acquisition is a scientific invalidity."),
        ValidationCheck("mean_exploration_separation", check_passed(
            controller, "bounds_and_metric_separation"), True,
            "controller validation contains a passing mean/candidate/exploration separation gate",
            "Exploration cannot be hidden in or substituted for learned-policy evaluation."),
        ValidationCheck("logical_detector_shared_state", logical_shared_state,
                        logical_versions, "logical and detector evidence carry identical physical/policy state IDs",
                        logical_details),
        ValidationCheck("policy_lifecycle_transactions", lifecycle.passed,
                        {"report_hash": lifecycle.report_hash},
                        "delayed acknowledgement, stale reference, concurrent proposal, and rollback tests pass",
                        "Every action uses the explicit confirmed-to-acknowledged transaction protocol."),
        ValidationCheck("report_schema_and_evidence_layers", report_schema.passed,
                        {"report_hash": report_schema.report_hash},
                        "scientific report and evidence-layer schema checks pass",
                        "Simulator, surrogate, circuit-logical, published, and deployment evidence remain distinct."),
        ValidationCheck("development_baseline_cohort", development.passed,
                        {"report_hash": development.report_hash},
                        "short held-out no-drift, step, sinusoid, and RTN cohort passes",
                        "The six-arm acquisition cannot begin until baseline family behavior is qualitatively valid."),
        ValidationCheck("stage2_6_numerical_equivalence", performance.passed,
                        {"report_hash": performance.report_hash},
                        "optimized Stage 2--6 kernels match their scalar references",
                        "Performance work cannot change the numerical scientific result."),
        ValidationCheck("failure_injection_coverage", fault_matrix.passed,
                        {"report_hash": fault_matrix.report_hash,
                         "fault_count": fault_matrix.metadata.get("fault_count")},
                        "all fifteen predeclared scientific faults are caught",
                        "An acceptance gate is authoritative only when deliberate invalid results fail closed."),
        ValidationCheck("post_comparison_recovery_regressions", post_comparison.passed,
                        {"report_hash": post_comparison.report_hash},
                        "nested_common/102 and unknown/105 exact recovery regressions pass",
                        "Regional/global re-entry and rollback repair must remain deterministic."),
        ValidationCheck("rollback_fault_separation", check_passed(
            post_comparison, "rollback_transaction_physical_fault_separation"),
            {"report_hash": post_comparison.report_hash},
            "rollback transaction and physical restoration faults are separately fail-closed",
            "Stale targets, acknowledgement/hash corruption and unsafe telemetry block authority."),
        ValidationCheck("ou_nested_development_tail_latency", check_passed(
            post_comparison, "ou_nested_development_tail_latency"),
            {"report_hash": post_comparison.report_hash},
            "OU-step and nested-common development tails satisfy the frozen release threshold",
            "Censoring is retained and safety failures cannot be offset by median latency."),
        ValidationCheck("compute_accounting_and_rmst", compute_accounting.passed,
                        {"report_hash": compute_accounting.report_hash},
                        "symmetric critical-path timing and censored RMST/tail checks pass",
                        "A compute-aware superiority claim requires complete, non-overlapping online timing."),
        ValidationCheck("compute_evidence_fault_matrix", check_passed(
            compute_accounting, "compute_evidence_fault_matrix"),
            {"report_hash": compute_accounting.report_hash,
             "fault_count": compute_accounting.metadata.get("fault_count")},
            "timing, censoring, clustering and inflated-compute evidence faults fail closed",
            "Missing online work, complete-case deletion and unsafe censoring cannot produce acceptance."),
    ]
    if "stale_manifest" in faults:
        checks.append(ValidationCheck(
            "source_hash_current", False, "injected stale hash", "manifest source hash equals current tree",
            "Stale validation artifacts fail closed.",
        ))
    timing_environment = TimingEnvironment.capture(__version__)
    metadata: Mapping[str, object] = {
        "validation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_tree_hash": source_hash,
        "configuration_hash": deterministic_hash(asdict(config)),
        "simulator_version": SIMULATOR_VERSION,
        "controller_version": CONTROLLER_VERSION,
        "logical_stack_versions": dict(logical_versions),
        "result_hashes": {
            "plant": plant.report_hash,
            "controller": controller.report_hash,
            "sample_budget": budget.report_hash,
            "policy_lifecycle": lifecycle.report_hash,
            "report_schema": report_schema.report_hash,
            "development_cohort": development.report_hash,
            "stage2_6_performance": performance.report_hash,
            "fault_matrix": fault_matrix.report_hash,
            "post_comparison_recovery": post_comparison.report_hash,
            "compute_accounting": compute_accounting.report_hash,
        },
        "timing_environment": asdict(timing_environment),
        "timing_environment_hash": timing_environment.environment_hash,
        "selected_validated_reduced_budget": budget.metadata.get(
            "selected_validated_reduced_budget"),
        "thresholds": {
            "gradient_cosine": config.controller.minimum_gradient_cosine,
            "ranking_accuracy": config.sample_budget.minimum_ranking_accuracy,
            "harmful_update_probability": config.sample_budget.maximum_harmful_update_probability,
            "convergence_probability": config.sample_budget.minimum_convergence_probability,
        },
        "injected_faults": sorted(faults),
        "long_acquisition_authorized": all_passed(checks),
        "evidence_layer": "benchmark preflight; not an acceptance result",
    }
    return finalize_report(ValidationReport(
        "benchmark-preflight.v1", "benchmark_preflight",
        all_passed(checks), tuple(checks), (), metadata,
    ))
