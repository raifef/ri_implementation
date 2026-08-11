"""Dedicated V15 CLIs; no command auto-launches a long or held-out acquisition."""
from __future__ import annotations

import json
from typing import Any, Callable

from .decoder import run_decoder_steering_offline
from .dynamics import audit_mean_scale_conditioning, audit_scale_floor, classify_residual_decay
from .fidelity import (analyse_figure5c, audit_objective_alignment, audit_ppo_lifecycle,
                       fit_step_response, model_figure5a_latency,
                       plan_natural_drift_power, report_resource_semantics, verify_provenance)
from .gate import build_heldout_freeze, reference_gate_status
from .imports import build_import_manifest, verify_import_manifest
from .io import ARTIFACT_ROOT
from .reporting import build_report, build_status
from .scaling import (audit_curvature_distribution, audit_gradient_normalization,
                      decompose_figure5b, estimate_hessian_spectrum, project_slow_modes,
                      report_ess, run_information_ablation, verify_boundary_map)
from .sensitivity import (audit_detector_degree_normalization,
                          audit_source_sensitivity_definition,
                          calibrate_multi_point_sensitivity,
                          propagate_calibration_uncertainty,
                          verify_calibration_firewall)
from .immediate_execution import (
    audit_calibration_objective, audit_execution_path, audit_shard_freshness,
    compare_v12_v15_scales, freeze_execution_contract, run_abc_figure5b,
    run_abc_step, run_reduced_postfix_validation, trace_boundary_figure5b,
    trace_boundary_recovery, trace_boundary_step, verify_driver_integration,
)


def _prepare() -> None:
    if not (ARTIFACT_ROOT / "immutable_import_manifest.json").is_file():
        build_import_manifest()
    verify_import_manifest()


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    omitted = {"rows", "issues", "artifact_presence", "iterations", "state_chain", "candidate_lineage"}
    value = {key: item for key, item in result.items() if key not in omitted}
    if len(value) != len(result):
        value["full_artifact_root"] = str(ARTIFACT_ROOT.resolve())
    return value


def _run(command: str, function: Callable[[], dict[str, Any]]) -> int:
    _prepare()
    result = function()
    print(json.dumps({"command": command, "result": _compact(result),
                      "long_run_auto_launched": False, "heldout_auto_launched": False,
                      "output_root": str(ARTIFACT_ROOT.resolve())}, indent=2, sort_keys=True))
    return 0


def _run_immediate(command: str, function: Callable[[], dict[str, Any]]) -> int:
    result = function()
    print(json.dumps({"command": command, "result": _compact(result),
                      "long_run_auto_launched": False, "heldout_auto_launched": False,
                      "output_root": str((ARTIFACT_ROOT / "immediate_execution_audit").resolve())},
                     indent=2, sort_keys=True))
    return 0


def audit_source_sensitivity_definition_main() -> int: return _run("audit-source-sensitivity-definition", audit_source_sensitivity_definition)
def audit_detector_degree_normalization_main() -> int: return _run("audit-detector-degree-normalization", audit_detector_degree_normalization)
def calibrate_multi_point_sensitivity_main() -> int: return _run("calibrate-multi-point-sensitivity", calibrate_multi_point_sensitivity)
def propagate_calibration_uncertainty_main() -> int: return _run("propagate-calibration-uncertainty", propagate_calibration_uncertainty)
def verify_calibration_firewall_main() -> int: return _run("verify-calibration-firewall", verify_calibration_firewall)
def verify_boundary_map_main() -> int: return _run("verify-boundary-map", verify_boundary_map)
def decompose_figure5b_main() -> int: return _run("decompose-figure5b", decompose_figure5b)
def audit_gradient_normalization_main() -> int: return _run("audit-gradient-normalization", audit_gradient_normalization)
def audit_curvature_distribution_main() -> int: return _run("audit-curvature-distribution", audit_curvature_distribution)
def estimate_hessian_spectrum_main() -> int: return _run("estimate-hessian-spectrum", estimate_hessian_spectrum)
def project_slow_modes_main() -> int: return _run("project-slow-modes", project_slow_modes)
def run_information_ablation_main() -> int: return _run("run-information-ablation", run_information_ablation)
def report_ess_main() -> int: return _run("report-ess", report_ess)
def audit_mean_scale_conditioning_main() -> int: return _run("audit-mean-scale-conditioning", audit_mean_scale_conditioning)
def audit_scale_floor_main() -> int: return _run("audit-scale-floor", audit_scale_floor)
def classify_residual_decay_main() -> int: return _run("classify-residual-decay", classify_residual_decay)
def audit_objective_alignment_main() -> int: return _run("audit-objective-alignment", audit_objective_alignment)
def analyse_figure5c_main() -> int: return _run("analyse-figure5c", analyse_figure5c)
def model_figure5a_latency_main() -> int: return _run("model-figure5a-latency", model_figure5a_latency)
def fit_step_response_main() -> int: return _run("fit-step-response", fit_step_response)
def plan_natural_drift_power_main() -> int: return _run("plan-natural-drift-power", plan_natural_drift_power)
def audit_ppo_lifecycle_main() -> int: return _run("audit-ppo-lifecycle", audit_ppo_lifecycle)
def verify_state_chain_main() -> int: return _run("verify-state-chain", verify_provenance)
def verify_candidate_lineage_main() -> int: return _run("verify-candidate-lineage", verify_provenance)
def report_resource_semantics_main() -> int: return _run("report-resource-semantics", report_resource_semantics)
def run_decoder_steering_offline_main() -> int: return _run("run-decoder-steering-offline", run_decoder_steering_offline)
def build_heldout_freeze_main() -> int: return _run("build-heldout-freeze", build_heldout_freeze)
def reference_gate_status_main() -> int: return _run("reference-gate-status", reference_gate_status)
def status_main() -> int: return _run("status", build_status)
def report_main() -> int: return _run("report", build_report)
def audit_execution_path_main() -> int: return _run_immediate("audit-execution-path", audit_execution_path)
def trace_boundary_step_main() -> int: return _run_immediate("trace-boundary-step", trace_boundary_step)
def trace_boundary_recovery_main() -> int: return _run_immediate("trace-boundary-recovery", trace_boundary_recovery)
def trace_boundary_figure5b_main() -> int: return _run_immediate("trace-boundary-figure5b", trace_boundary_figure5b)
def run_abc_step_main() -> int: return _run_immediate("run-abc-step", run_abc_step)
def compare_v12_v15_scales_main() -> int: return _run_immediate("compare-v12-v15-scales", compare_v12_v15_scales)
def audit_calibration_objective_main() -> int: return _run_immediate("audit-calibration-objective", audit_calibration_objective)
def run_abc_figure5b_main() -> int: return _run_immediate("run-abc-figure5b", run_abc_figure5b)
def audit_shard_freshness_main() -> int: return _run_immediate("audit-shard-freshness", audit_shard_freshness)
def verify_driver_integration_main() -> int: return _run_immediate("verify-driver-integration", verify_driver_integration)
def freeze_execution_contract_main() -> int: return _run_immediate("freeze-execution-contract", freeze_execution_contract)
def run_reduced_postfix_validation_main() -> int: return _run_immediate("run-reduced-postfix-validation", run_reduced_postfix_validation)
