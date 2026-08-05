"""Claim-separated decoder, root-cause, status, and next-command reports."""
from __future__ import annotations

from typing import Any

from .common import artifact_root, read_json, write_artifact
from .contracts import evidence_envelope, enforce_family_separation


def _read(relative: str) -> dict[str, Any]:
    path = artifact_root() / relative
    if not path.is_file():
        raise RuntimeError(f"missing required v10 artifact: {relative}")
    return read_json(path)


def report_decoder() -> dict[str, Any]:
    contract = _read("decoder/decoder_contract.json")
    validation = _read("decoder/decoder_validation.json")
    control_only = _read("decoder/control_only_results.json")
    closed_loop = _read("decoder/closed_loop_results.json")
    steering = _read("decoder/decoder_steering_results.json")
    enforce_family_separation([{"experiment_family": control_only["result"]["experiment_family"]}])
    enforce_family_separation([{"experiment_family": closed_loop["experiment_family"]}])
    payload = {
        "schema_version": "google-pure-v10-decoder-report.v1",
        "contract_hash": contract["artifact_hash"],
        "validation_hash": validation["artifact_hash"],
        "control_only": {
            "experiment_family": control_only["result"]["experiment_family"],
            "control_metrics": control_only["result"]["control_metrics"],
            "decoder_executed": False,
        },
        "control_plus_fixed_decoder": {
            "experiment_family": closed_loop["experiment_family"],
            "control_metrics": closed_loop["result"]["control_metrics"],
            "decoder_metrics": closed_loop["result"]["decoder_metrics"],
        },
        "decoder_steering": {
            "source_defined": steering["source_defined"],
            "control_metrics": steering["control_metrics"],
            "decoder_metrics": steering["decoder_metrics"],
        },
        "cross_family_aggregate_reported": False,
        "analytic_scaling_labeled_decoder_coupled": False,
        "neural_decoder_claimed_trained": False,
        **evidence_envelope(complete=True, mechanism_valid=True, claim_supported=True, paper_comparable=False),
    }
    return write_artifact("decoder/report", payload, "Decoder Integration Report", markdown_relative="decoder/report.md")


def root_cause_update() -> dict[str, Any]:
    corrected = _read("corrected_fault_contract.json")
    controller = _read("controller/scale_entropy_results.json")
    temporal = _read("controller/temporal_validation.json")
    spectral = _read("natural_drift/report.json")
    decoder = _read("decoder/report.json")
    step = _read("step_response/report.json")
    ablation = _read("step_response/ablation_results.json")
    selected = _read("controller/selected_controller_contract.json")
    preflight = _read("preflight_manifest.json")
    blockers = []
    for artifact in (controller, temporal, spectral, decoder, step, ablation, selected):
        blockers.extend(artifact.get("blocking_reasons", []))
    payload = {
        "schema_version": "google-pure-v10-root-cause-update.v1",
        "corrected_fault_contract_hash": corrected["artifact_hash"],
        "entropy_operational": controller["entropy_operationality"]["operational"],
        "selected_controller": selected.get("controller"),
        "controller_selected": selected["selected"],
        "temporal_phase_window_stable": temporal["claim_supported"],
        "natural_drift": {
            "frequency_resolution": spectral["rows"][0]["frequency_resolution_from_duration"],
            "median_mean_suppression_db": spectral["median_mean_suppression_db"],
            "median_candidate_suppression_db": spectral["median_candidate_suppression_db"],
            "paper_comparable": spectral["paper_comparable"],
        },
        "decoder": {
            "validated": decoder["mechanism_valid"],
            "control_only_and_decoder_assisted_separate": not decoder["cross_family_aggregate_reported"],
        },
        "step_response": {
            "tau_epochs": step["response"]["tau_epochs"],
            "tau_ci_95_epochs": step["response"]["tau_profile_confidence_interval_95_epochs"],
            "classification": step["response"]["response_classification"],
            "clipping_classification": ablation["clipping_classification"],
            "learning_rate_classification": ablation["learning_rate_classification"],
        },
        "remaining_public_information_limits": [
            "entropy reward balance is not source identifiable",
            "decoder steering is not source defined",
            "reference-scale held-out acquisition requires explicit execution",
        ],
        "pure_controller_ready_for_external_comparison": bool(selected["selected"] and temporal["claim_supported"]),
        "comparative_benchmark_preflight_pass": preflight["preflight_gate_pass"],
        "full_benchmark_permitted": preflight["full_benchmark_permitted"],
        **evidence_envelope(
            complete=True,
            mechanism_valid=True,
            claim_supported=not blockers,
            paper_comparable=False,
            blocking_reasons=sorted(set(blockers)),
        ),
    }
    return write_artifact("root_cause_update", payload, "v10 Root-cause Update")


def next_commands() -> dict[str, Any]:
    commands = [
        "google-rl-v10-plan-scale-entropy --mode development",
        "google-rl-v10-run-scale-entropy --mode development --execute",
        "google-rl-v10-freeze-held-out",
        "google-rl-v10-run-held-out --mode validation --execute",
        "google-rl-v10-plan-natural-drift --mode reference",
        "google-rl-v10-run-natural-drift --mode reference --execute",
        "google-rl-v10-plan-step-response --mode reference",
        "google-rl-v10-run-step-response --mode reference --execute",
        "google-rl-v10-run-step-ablation --mode reference --execute",
        "google-rl-v10-report",
    ]
    payload = {
        "schema_version": "google-pure-v10-next-commands.v1",
        "commands": commands,
        "long_acquisitions_automatic": False,
        **evidence_envelope(complete=True, mechanism_valid=True, claim_supported=False, paper_comparable=False, blocking_reasons=["EXPLICIT_USER_EXECUTION_REQUIRED"]),
    }
    return write_artifact("next_commands", payload, "Exact v10 Next Commands", markdown_relative="next_commands.md")


def status() -> dict[str, Any]:
    required = (
        "import_manifest.json",
        "corrected_fault_contract.json",
        "preflight_manifest.json",
        "controller/scale_entropy_results.json",
        "controller/scale_entropy_report.md",
        "controller/temporal_validation.json",
        "controller/temporal_validation.md",
        "controller/selected_controller_contract.json",
        "natural_drift/run_plan.json",
        "natural_drift/raw_traces.npz",
        "natural_drift/psd_results.npz",
        "natural_drift/report.md",
        "natural_drift/figure.png",
        "decoder/decoder_contract.json",
        "decoder/decoder_validation.json",
        "decoder/closed_loop_results.json",
        "decoder/report.md",
        "step_response/run_plan.json",
        "step_response/raw_trajectories.npz",
        "step_response/fits.json",
        "step_response/ablation_results.json",
        "step_response/report.md",
        "step_response/figure.png",
        "root_cause_update.json",
        "root_cause_update.md",
        "next_commands.md",
    )
    files = {name: (artifact_root() / name).is_file() for name in required}
    payload = {
        "schema_version": "google-pure-v10-status.v1",
        "artifacts": files,
        "implementation_complete": all(files.values()),
        "reference_evidence_complete": False,
        "certification_seeds_consumed": False,
        **evidence_envelope(
            complete=all(files.values()),
            mechanism_valid=all(files.values()),
            claim_supported=False,
            paper_comparable=False,
            blocking_reasons=["REFERENCE_SCALE_HELD_OUT_RUNS_NOT_AUTOMATIC"],
        ),
    }
    return write_artifact("status", payload, "v10 Status")
