"""Artifact generation and guarded confirmatory preregistration for Track B."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping

from hdfa_rl_suite import __version__
from hdfa_rl_suite.common import deterministic_hash
from hdfa_rl_suite.google_rl_certification.config import repository_root

from .config import FUTURE_TRACK_B_CONFIRMATORY_SEEDS, TrackBConfig
from .development import (
    residual_stratified_analysis,
    resource_matched_analysis,
    run_plant_development,
    scientific_outcome,
)
from .substrate import build_common_substrate, track_a_freeze


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def estimate_development_cost(config: TrackBConfig = TrackBConfig()) -> dict[str, Any]:
    a_epochs = len(config.development_seeds)*4*config.plant_a_intervals
    b_epochs = len(config.development_seeds)*3*config.plant_b_intervals
    epochs = a_epochs+b_epochs
    high_candidates = epochs*40
    high_candidate_cycles = high_candidates*100_000
    reduced_candidate_cycles = epochs*40*2_048
    high_mean_evaluations = ((config.plant_a_intervals+4)//5*4
                             + (config.plant_b_intervals+4)//5*3)
    high_mean_evaluations *= len(config.development_seeds)
    high_mean_cycles = high_mean_evaluations*5_000_000
    endpoint_cycles = epochs*len((
        "fixed", "periodic", "oracle", "high", "reduced", "predictive", "conditional"))*config.endpoint_evaluation_cycles
    staged_probe_cycles = epochs*2*5*config.stage2_probe_cycles
    return {
        "development_epochs_per_arm": epochs,
        "high_shot_candidate_evaluations": high_candidates,
        "high_shot_candidate_qec_cycles": high_candidate_cycles,
        "high_shot_mean_evaluation_qec_cycles": high_mean_cycles,
        "reduced_candidate_qec_cycles": reduced_candidate_cycles,
        "common_endpoint_evaluation_qec_cycles_upper_bound": endpoint_cycles,
        "staged_probe_qec_cycles_upper_bound": staged_probe_cycles,
        "estimated_vectorized_runtime": "approximately 1-5 minutes on the bundled Python runtime",
        "evidence_layer": "simulated cycle-equivalents only; no QPU acquisition",
        "confirmatory_acquisition_included": False,
    }


def _plant_markdown(report: Mapping[str, Any], label: str) -> str:
    gates = report["baseline_gates"]
    lines = [
        f"# {label} development comparison", "",
        f"**Status:** `{'PASS' if report['baseline_gates_passed'] else 'FAIL'}`", "",
        f"Evidence layer: {report['evidence_layer']}", "",
        "This is development-only evidence. Confirmatory seeds were not used and no long acquisition was run.", "",
        "## Baseline and substrate gates", "",
    ]
    lines.extend(f"- {'PASS' if value else 'FAIL'} - `{key}`" for key, value in gates.items())
    lines.extend(["", "## Arms", ""])
    for arm in report["arms"]:
        rows = [item for item in report["run_summaries"] if item["arm"] == arm]
        lines.append(
            f"- `{arm}`: final EDR {_mean(item['final_detector_rate'] for item in rows):.6g}; "
            f"native cycles {_mean(item['total_native_qec_cycles'] for item in rows):.6g}; "
            f"exploration damage {_mean(item['exploration_damage_detector_events'] for item in rows):.6g}.")
    lines.append("")
    return "\n".join(lines)


def _mean(values) -> float:
    rows = list(values)
    return sum(rows)/len(rows) if rows else float("nan")


def _analysis_markdown(title: str, report: Mapping[str, Any]) -> str:
    lines = [f"# {title}", "",
             f"**Status:** `{'PASS' if report.get('passed') else 'FAIL'}`", ""]
    if "gates" in report:
        lines.extend(f"- {'PASS' if value else 'FAIL'} - `{key}`"
                     for key, value in report["gates"].items())
    lines.append("")
    return "\n".join(lines)


def preregister_track_b(
    outcome: Mapping[str, Any],
    substrate: Mapping[str, Any],
    config: TrackBConfig = TrackBConfig(),
    output: Path | None = None,
) -> dict[str, Any]:
    if not outcome.get("confirmatory_preregistration_justified"):
        raise RuntimeError(
            "Track-B preregistration refused: development did not satisfy Outcome C")
    root = repository_root()
    destination = output or root / "artifacts/staged_vs_certified_rl"
    config_dir = root / "configs/staged_vs_certified_rl"
    config_dir.mkdir(parents=True, exist_ok=True)
    prereg_config = {
        "schema_version": "staged-vs-certified-rl-confirmatory-config.v1",
        "status": "frozen_not_executed",
        "plant_contract_hashes": {
            item["plant_id"]: item["contract_hash"] for item in substrate["plants"]},
        "track_a_frozen_hash": substrate["track_a_freeze"]["aggregate_sha256"],
        "seed_list": FUTURE_TRACK_B_CONFIRMATORY_SEEDS,
        "development_seed_list": config.development_seeds,
        "resource_views": ["matched_native_qec", "matched_wall_clock", "matched_final_quality"],
        "primary_estimands": [
            "final detector-rate noninferiority",
            "logical-failure noninferiority",
            "structured 50/75/90-percent observed recovery",
            "integrated excess EDR",
            "exploration damage",
            "conditional residual value by stratum",
            "periodic end-to-end logical utility",
        ],
        "thresholds": {
            "detector_noninferiority_margin": config.detector_noninferiority_margin,
            "logical_noninferiority_margin": config.logical_noninferiority_margin,
            "minimum_residual_relative_benefit": config.minimum_residual_relative_benefit,
            "minimum_one_interval_recovery_fraction": config.minimum_one_interval_recovery_fraction,
            "minimum_candidate_efficiency_ratio": config.minimum_candidate_efficiency_ratio,
            "minimum_excess_edr_ratio": config.minimum_excess_edr_ratio,
            "minimum_exploration_damage_ratio": config.minimum_exploration_damage_ratio,
        },
        "censoring": "observed targets only; no fit extrapolation; safety termination is failure",
        "followup": "all arms retain the complete frozen interval horizon",
        "rollback": "transactional rollback states remain distinct from physical restoration evidence",
        "confirmatory_seed_consumption": False,
        "long_acquisition_executed": False,
    }
    config_path = config_dir / "confirmatory-v1.yaml"
    _write_json(config_path, prereg_config)
    prereg = {
        "schema_version": "staged-vs-certified-rl-preregistration.v1",
        "status": "FROZEN_NOT_EXECUTED",
        "configuration_path": str(config_path.relative_to(root)).replace("\\", "/"),
        "configuration_hash": deterministic_hash(prereg_config),
        "source_track_a_hash": track_a_freeze()["aggregate_sha256"],
        "development_outcome_hash": deterministic_hash(outcome),
        "package_version": __version__,
        "confirmatory_seeds": FUTURE_TRACK_B_CONFIRMATORY_SEEDS,
        "seed_sets_disjoint": not set(FUTURE_TRACK_B_CONFIRMATORY_SEEDS).intersection(
            config.development_seeds),
        "confirmatory_seeds_consumed": False,
        "final_long_acquisition_executed": False,
        "estimated_runtime": "not estimated until a Tier-C implementation binding passes a fresh preflight",
        "future_command": "hdfa-run-staged-vs-certified-rl-confirmatory --config configs/staged_vs_certified_rl/confirmatory-v1.yaml",
        "command_intentionally_not_implemented_for_automatic_execution": True,
    }
    prereg["artifact_hash"] = deterministic_hash(prereg)
    _write_json(destination / "confirmatory_preregistration.json", prereg)
    (destination / "confirmatory_preregistration.md").write_text(
        "# Track-B confirmatory preregistration\n\n"
        "**Status:** `FROZEN_NOT_EXECUTED`\n\n"
        f"Seeds `{FUTURE_TRACK_B_CONFIRMATORY_SEEDS[0]}-{FUTURE_TRACK_B_CONFIRMATORY_SEEDS[-1]}` are protected and unconsumed. "
        "No Tier-C acquisition command was executed; launch remains fail-closed pending a fresh implementation-bound preflight.\n",
        encoding="utf-8")
    return prereg


def run_track_b_development(
    config: TrackBConfig = TrackBConfig(),
    output: Path | None = None,
) -> dict[str, Any]:
    root = repository_root()
    destination = output or root / "artifacts/staged_vs_certified_rl"
    destination.mkdir(parents=True, exist_ok=True)
    substrate = build_common_substrate(config, destination)
    plant_a = run_plant_development("a", config)
    plant_b = run_plant_development("b", config)
    residual = residual_stratified_analysis(plant_b, config)
    resources = resource_matched_analysis(plant_a, plant_b, config)
    outcome = scientific_outcome(plant_a, plant_b, residual, resources, config)
    artifacts = (
        ("plant_a_development", plant_a, _plant_markdown(plant_a, "Plant A")),
        ("plant_b_development", plant_b, _plant_markdown(plant_b, "Plant B")),
        ("residual_stratified_analysis", residual,
         _analysis_markdown("Residual-stratified analysis", residual)),
        ("resource_matched_analysis", resources,
         _analysis_markdown("Resource-matched analysis", resources)),
    )
    for stem, payload, markdown in artifacts:
        _write_json(destination / f"{stem}.json", payload)
        (destination / f"{stem}.md").write_text(markdown, encoding="utf-8")
    _write_json(destination / "scientific_outcome.json", outcome)
    outcome_lines = [
        "# Track-B scientific outcome", "",
        f"**Classification:** `{outcome['classification']}`", "",
        outcome["interpretation"], "",
        f"Confirmatory preregistration justified: `{'YES' if outcome['confirmatory_preregistration_justified'] else 'NO'}`.", "",
        "No confirmatory seeds were consumed and no final long acquisition was run.", "",
        "## Acceptance gates", "",
    ]
    outcome_lines.extend(
        f"- {'PASS' if value else 'FAIL'} - `{key}`"
        for key, value in outcome["gates"].items())
    outcome_lines.append("")
    (destination / "scientific_outcome.md").write_text(
        "\n".join(outcome_lines), encoding="utf-8")
    prereg = None
    if outcome["confirmatory_preregistration_justified"]:
        prereg = preregister_track_b(outcome, substrate, config, destination)
    return {
        "substrate": substrate,
        "plant_a": plant_a,
        "plant_b": plant_b,
        "residual": residual,
        "resources": resources,
        "outcome": outcome,
        "preregistration": prereg,
    }

