"""Certification orchestration and fail-closed JSON/Markdown artifacts."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

from .analytic_landscape import run_analytic_certification
from .audit import write_audit_artifacts
from .common import jsonable, write_json
from .config import GoogleRLConfig, named_config, repository_root
from .drift_tracking import run_drift_tracking_certification
from .sample_budget_equivalence import run_budget_equivalence
from .scaling_locality import run_scaling_locality
from .spoiled_policy_recovery import run_spoiled_policy_recovery
from .static_detector_landscape import run_static_detector_certification
from .steering_frequency import run_steering_frequency_sweep


def artifact_directory(output: str | Path | None = None) -> Path:
    return (Path(output) if output is not None else
            repository_root()/"artifacts"/"google_rl_certification")


def _gate_markdown(gates: Mapping[str, Any]) -> list[str]:
    return [f"- {'PASS' if bool(value) else 'FAIL'} - `{key}`"
            for key, value in gates.items()]


def run_high_shot_certification(config: GoogleRLConfig | None = None,
                                *, seed: int = 8801) -> dict[str, Any]:
    selected = config or named_config("high_shot_reference")
    started = perf_counter()
    analytic = run_analytic_certification(selected, seed=seed+1)
    static = run_static_detector_certification(selected, seed=seed+2)
    spoiled = run_spoiled_policy_recovery(selected, seed=seed+3)
    drift = run_drift_tracking_certification(selected, seed=seed+4)
    steering = run_steering_frequency_sweep(selected, seed=seed+5)
    scaling = run_scaling_locality(selected, seed=seed+6)
    conditions = {
        "analytic_known_gradient": analytic["passed"],
        "static_sparse_detector": static["passed"],
        "spoiled_policy_recovery": spoiled["passed"],
        "slow_drift_tracking": drift["slow_drift"]["passed"],
        "calibrated_no_drift_no_regression": drift["no_drift"]["passed"],
        "plausible_step_response": drift["step_response"]["passed"],
        "steering_frequency_transition": steering["passed"],
        "locality_and_scaling": scaling["passed"],
        "mean_better_than_exploration": (
            spoiled["gates"]["learned_mean_better_than_exploration"]
            and drift["slow_drift"]["gates"]["mean_better_than_exploratory_aggregate"]),
    }
    passed = all(conditions.values())
    return {
        "schema_version": "google-rl-high-shot-certification.v1",
        "status": "HIGH_SHOT_REFERENCE_CERTIFIED" if passed else "HIGH_SHOT_REFERENCE_FAILED",
        "evidence_layer": "executed repository surrogate certification; not Willow hardware evidence",
        "config": asdict(selected),
        "certification_seed": seed,
        "runtime_s": perf_counter()-started,
        "conditions": conditions,
        "passed": passed,
        "source_boundary": "See source_extraction.md; proprietary details remain approximations.",
        "limited_run_policy": "one certification seed per level plus deterministic subcases; no staged comparison",
        "results": {
            "analytic": analytic,
            "static_detector": static,
            "spoiled_recovery": spoiled,
            "drift_tracking": drift,
            "steering_frequency": steering,
            "scaling_locality": scaling,
        },
    }


def _high_markdown(payload: Mapping[str, Any]) -> str:
    spoiled = payload["results"]["spoiled_recovery"]
    drift = payload["results"]["drift_tracking"]
    steering = payload["results"]["steering_frequency"]
    lines = [
        "# High-shot Google-style RL certification", "",
        f"**Status:** `{payload['status']}`", "",
        "This is repository-surrogate evidence, not a Willow reproduction or hardware claim.", "",
        "## Gates", "", *_gate_markdown(payload["conditions"]), "",
        "## Key observed results", "",
        f"- Spoiled-policy 90% recovery: {next(item['epochs'] for item in spoiled['recovery_endpoints'] if item['target_fraction'] == .90)} epochs.",
        f"- Slow-drift mean/fixed EDR: {drift['slow_drift']['mean_policy_edr']:.6g} / {drift['slow_drift']['fixed_edr']:.6g}.",
        f"- Mean/exploratory EDR: {drift['slow_drift']['mean_policy_edr']:.6g} / {drift['slow_drift']['aggregate_exploration_edr']:.6g}.",
        f"- Step characteristic response: {drift['step_response']['characteristic_response_epochs']} epochs; public 130-epoch anchor is qualitative only.",
        f"- Steering transition: period {steering['critical_period_epochs']} epochs versus public anchor near 150; preregistered range 100-225.",
        "- No staged-controller tuning or staged comparison was run.", "",
    ]
    return "\n".join(lines)


def write_high_shot_artifacts(output: str | Path | None = None,
                              *, seed: int = 8801) -> dict[str, Any]:
    destination = artifact_directory(output)
    write_audit_artifacts(destination)
    payload = run_high_shot_certification(seed=seed)
    write_json(destination/"high_shot_certification.json", payload)
    (destination/"high_shot_certification.md").write_text(
        _high_markdown(payload), encoding="utf-8")
    steering = payload["results"]["steering_frequency"]
    write_json(destination/"steering_frequency.json", steering)
    (destination/"steering_frequency.md").write_text(
        "\n".join([
            "# Google-style RL steering-frequency sweep", "",
            f"**Status:** `{'PASS' if steering['passed'] else 'FAIL'}`", "",
            f"Observed stochastic-policy transition period: {steering['critical_period_epochs']} epochs.",
            "Public anchor: approximately 150 epochs; preregistered repository-surrogate range: 100-225 epochs.",
            "Learned-mean and aggregate exploratory policies are reported separately in the JSON artifact.", "",
        ]), encoding="utf-8")
    return payload


def write_budget_equivalence_artifacts(output: str | Path | None = None,
                                       *, seed: int = 9901) -> dict[str, Any]:
    destination = artifact_directory(output)
    payload = run_budget_equivalence(
        named_config("high_shot_reference"),
        named_config("reduced_budget_candidate"), seed=seed)
    write_json(destination/"reduced_budget_equivalence.json", payload)
    lines = [
        "# Reduced-budget equivalence", "",
        f"**Status:** `{payload['status']}`", "",
        "Equivalence is limited to the declared repository certification landscapes; native-QEC cost is intentionally allowed to differ.", "",
        "## Gates", "", *_gate_markdown(payload["gates"]), "",
        f"Reduced/high native-QEC cost per epoch: {payload['native_qec_cost_ratio_per_epoch']:.6g}.", "",
    ]
    (destination/"reduced_budget_equivalence.md").write_text(
        "\n".join(lines), encoding="utf-8")
    return payload


def _read(path: Path) -> Mapping[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, Mapping) else None


def write_final_certification(output: str | Path | None = None) -> dict[str, Any]:
    destination = artifact_directory(output)
    high = _read(destination/"high_shot_certification.json")
    reduced = _read(destination/"reduced_budget_equivalence.json")
    high_status = (high or {}).get("status", "HIGH_SHOT_REFERENCE_FAILED")
    reduced_status = (reduced or {}).get("status", "REDUCED_BUDGET_NOT_EQUIVALENT")
    if high_status == "HIGH_SHOT_REFERENCE_FAILED":
        status = "HIGH_SHOT_REFERENCE_FAILED"
    elif reduced_status == "REDUCED_BUDGET_EQUIVALENT":
        status = "REDUCED_BUDGET_EQUIVALENT"
    else:
        status = "REDUCED_BUDGET_NOT_EQUIVALENT"
    prerequisite = (high_status == "HIGH_SHOT_REFERENCE_CERTIFIED"
                    and reduced_status == "REDUCED_BUDGET_EQUIVALENT")
    payload = {
        "schema_version": "google-rl-final-certification.v1",
        "status": status,
        "high_shot_status": high_status,
        "reduced_budget_status": reduced_status,
        "evidence_layer": "repository surrogate certification; no Willow reproduction",
        "track_b_prerequisite_satisfied": prerequisite,
        "track_b_prerequisite_reason": (
            "high-shot reference certified and reduced budget matched on all declared Track-A environments"
            if prerequisite else "a required high-shot or reduced-budget gate failed or is missing"),
        "public_anchor_comparison": {
            "step_response": ((high or {}).get("results", {}).get("drift_tracking", {})
                              .get("step_response", {}).get("characteristic_response_epochs")),
            "public_step_anchor_epochs": 130,
            "step_claim": "qualitative_only_different_surrogate_plant",
            "steering_critical_period_epochs": ((high or {}).get("results", {})
                                                .get("steering_frequency", {})
                                                .get("critical_period_epochs")),
            "public_steering_anchor_epochs": 150,
        },
        "remaining_uncertainties": [
            "proprietary optimizer and hyperparameters",
            "hardware sensitivity coefficients and detector-control graph",
            "controller upload, compilation, latency, and safety implementation",
            "simulator-to-Willow and detector-to-logical validity",
            "equivalence outside the declared surrogate landscapes",
        ],
        "commands_not_run": [
            "final staged HDFA-versus-RL comparison",
            "broad confirmatory seed sweep",
            "Willow or other QPU acquisition",
        ],
    }
    write_json(destination/"final_certification.json", payload)
    lines = [
        "# Final Google-style RL certification", "",
        f"**Status:** `{status}`", "",
        f"High-shot: `{high_status}`", "",
        f"Reduced budget: `{reduced_status}`", "",
        f"Track B prerequisite: `{'SATISFIED' if prerequisite else 'NOT_SATISFIED'}`", "",
        "The result is limited to declared repository surrogates and is not a Willow reproduction. See `source_extraction.md` for unavailable details and explicit approximations.", "",
    ]
    (destination/"final_certification.md").write_text(
        "\n".join(lines), encoding="utf-8")
    return payload
