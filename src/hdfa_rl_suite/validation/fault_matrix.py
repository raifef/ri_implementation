"""Deliberate-failure matrix for benchmark scientific-integrity gates."""
from __future__ import annotations

from dataclasses import asdict

from hdfa_rl_suite.logical import RotatedSurfaceCodeEvaluator, SurfaceCodeMemoryConfig
from hdfa_rl_suite.simulator import (
    DriftKind, LatentProcessSpec, ScalableQECDevice, SimulatorConfig,
)

from .common import ValidationCheck, ValidationReport, all_passed, finalize_report
from .controller_sanity import run_controller_validation
from .lifecycle_sanity import run_lifecycle_validation
from .plant_sanity import run_plant_validation
from .report_sanity import run_report_validation
from .sample_budget import run_sample_budget_validation


FAULT_MATRIX_VERSION = "benchmark-fault-matrix.v1"


def _failed(report: ValidationReport, check_id: str) -> tuple[bool, object]:
    match = next((check for check in report.checks if check.check_id == check_id), None)
    return bool(match is not None and not match.passed), (
        asdict(match) if match is not None else {"missing_check": check_id})


def _hidden_fixed_update() -> tuple[bool, object]:
    device = ScalableQECDevice(SimulatorConfig(
        qubit_count=3, controller_latency_s=0.0, disturbances_enabled_at_start=False,
        seed=4101,
    ))
    declared = device.confirmed_policy
    target = dict(declared.controls)
    target["drive:q0"] = .05
    device.apply_policy(target, policy_id="injected-hidden-fixed-update")
    device.await_policy_acknowledgement()
    observed = device.confirmed_policy
    detected = (observed.policy_id != declared.policy_id
                or observed.policy_hash != declared.policy_hash)
    return detected, {
        "declared_fixed_policy_id": declared.policy_id,
        "observed_policy_id": observed.policy_id,
        "declared_policy_hash": declared.policy_hash,
        "observed_policy_hash": observed.policy_hash,
    }


def _stale_logical_evaluation() -> tuple[bool, object]:
    device = ScalableQECDevice(SimulatorConfig(
        qubit_count=3, controller_latency_s=0.0, seed=4102,
        processes=(LatentProcessSpec(
            "step", DriftKind.STEP, {"drive:q0": 1.0}, amplitude=.2,
            step_time_s=0.0),),
    ))
    stale_batch = device.acquire(8, retain_records=False)
    device.acquire(8, retain_records=False)
    logical = RotatedSurfaceCodeEvaluator(SurfaceCodeMemoryConfig(
        distance=3, rounds=3, shots=16)).evaluate_device(device, seed=4102)
    shared = (
        logical.physical_state_id == stale_batch.physical_state_id
        and logical.policy_hash == stale_batch.policy_activation.policy_hash
        and logical.disturbance_state_id == stale_batch.disturbance_state_id
    )
    return not shared, {
        "stale_detector_state_id": stale_batch.physical_state_id,
        "logical_state_id": logical.physical_state_id,
        "shared_state_gate": shared,
    }


def run_fault_matrix_validation() -> ValidationReport:
    """Inject all predeclared failure modes and require their named gate to fail."""
    specifications = (
        ("reversed_reward_sign", "analytic_convergence_both_sides", "controller"),
        ("shuffled_candidate_rewards", "static_sparse_gradient_alignment", "controller"),
        ("transposed_parameter_masks", "static_sparse_gradient_alignment", "controller:transposed_mask"),
        ("cumulative_perturbations", "candidate_centring_and_no_cumulative_error", "controller"),
        ("wrong_sensitivity_units", "static_sparse_gradient_alignment", "controller"),
        ("oversized_covariance", "covariance_contraction", "controller"),
        ("noncontracting_covariance", "covariance_contraction", "controller"),
        ("hidden_fixed_arm_policy_update", "fixed_policy_immutability", "fixed"),
        ("disturbance_reset_during_cloning", "ou_persistence_and_clone_identity", "plant:ou_clone_mismatch"),
        ("stale_logical_evaluation", "logical_detector_shared_state", "logical"),
        ("wrong_policy_activation_reference", "pending_mpc_probe_reference_race", "lifecycle"),
        ("underpowered_candidate_cycles", "candidate_budget_adequacy", "budget"),
        ("mean_candidate_metric_conflation", "report_schema_and_evidence_layers", "report"),
        ("invalid_convergence_extrapolation", "report_schema_and_evidence_layers", "report"),
        ("informative_censoring_complete_case", "report_schema_and_evidence_layers", "report"),
    )
    rows: list[dict[str, object]] = []
    checks: list[ValidationCheck] = []
    for fault, gate, route in specifications:
        if route == "fixed":
            detected, evidence = _hidden_fixed_update()
        elif route == "logical":
            detected, evidence = _stale_logical_evaluation()
        elif route == "lifecycle":
            detected, evidence = _failed(
                run_lifecycle_validation(
                    injected_faults=("wrong_policy_activation_reference",)), gate)
        elif route == "budget":
            detected, evidence = _failed(
                run_sample_budget_validation(
                    injected_faults=("underpowered_budget_accepted",)), gate)
        elif route == "report":
            detected, evidence = _failed(
                run_report_validation(injected_faults=(fault,)), gate)
        elif route.startswith("plant:"):
            injected = route.split(":", 1)[1]
            detected, evidence = _failed(
                run_plant_validation(injected_faults=(injected,)), gate)
        else:
            injected = (route.split(":", 1)[1]
                        if route.startswith("controller:") else fault)
            detected, evidence = _failed(
                run_controller_validation(injected_faults=(injected,)), gate)
        row = {
            "fault_id": fault,
            "detector_gate": gate,
            "detected": detected,
            "evidence": evidence,
        }
        rows.append(row)
        checks.append(ValidationCheck(
            f"fault_detected:{fault}", detected, row,
            f"injected fault must fail {gate}",
            "A long acquisition is authorized only when every declared scientific fault is detectable.",
        ))
    return finalize_report(ValidationReport(
        FAULT_MATRIX_VERSION, "scientific_failure_injection_matrix",
        all_passed(checks), tuple(checks), tuple(rows), {
            "fault_count": len(specifications),
            "all_faults_detected": all_passed(checks),
            "evidence_layer": "deliberate validation-gate fault injection",
        },
    ))
