"""Public Track-B commands. No command in this module launches Tier-C acquisition."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from hdfa_rl_suite.google_rl_certification.config import repository_root

from .config import TrackBConfig
from .development import (
    residual_stratified_analysis,
    resource_matched_analysis,
    run_plant_development,
    scientific_outcome,
)
from .report import (
    _write_json,
    estimate_development_cost,
    preregister_track_b,
    run_track_b_development,
)
from .substrate import build_common_substrate


def _destination(path: str | None) -> Path:
    return (Path(path) if path else
            repository_root()/"artifacts/staged_vs_certified_rl")


def _parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output", help="artifact directory")
    return parser


def _announce_cost(config: TrackBConfig, scope: str) -> None:
    print(json.dumps({"scope": scope, **estimate_development_cost(config)}, indent=2))
    print("No QPU acquisition and no confirmatory seeds will be used.")


def build_substrate_main(argv: Sequence[str] | None = None) -> int:
    args = _parser("Build the frozen common Track-B comparison substrate").parse_args(argv)
    report = build_common_substrate(TrackBConfig(), _destination(args.output))
    print(report["manifest_hash"])
    return 0 if report["all_common_substrate_checks_pass"] else 2


def plant_a_main(argv: Sequence[str] | None = None) -> int:
    args = _parser("Run bounded Plant-A development comparisons").parse_args(argv)
    config = TrackBConfig(); _announce_cost(config, "plant-a-development")
    report = run_plant_development("a", config)
    _write_json(_destination(args.output)/"plant_a_development.json", report)
    return 0 if report["baseline_gates_passed"] else 2


def plant_b_main(argv: Sequence[str] | None = None) -> int:
    args = _parser("Run bounded rich-Plant-B development comparisons").parse_args(argv)
    config = TrackBConfig(); _announce_cost(config, "plant-b-development")
    report = run_plant_development("b", config)
    _write_json(_destination(args.output)/"plant_b_development.json", report)
    return 0 if report["baseline_gates_passed"] else 2


def residual_main(argv: Sequence[str] | None = None) -> int:
    args = _parser("Analyse conditional residual-RL strata").parse_args(argv)
    destination = _destination(args.output)
    plant_b = json.loads((destination/"plant_b_development.json").read_text(encoding="utf-8"))
    report = residual_stratified_analysis(plant_b)
    _write_json(destination/"residual_stratified_analysis.json", report)
    return 0 if report["passed"] else 2


def resource_main(argv: Sequence[str] | None = None) -> int:
    args = _parser("Compare matched native-QEC, wall-clock, and final-quality views").parse_args(argv)
    destination = _destination(args.output)
    plant_a = json.loads((destination/"plant_a_development.json").read_text(encoding="utf-8"))
    plant_b = json.loads((destination/"plant_b_development.json").read_text(encoding="utf-8"))
    report = resource_matched_analysis(plant_a, plant_b)
    _write_json(destination/"resource_matched_analysis.json", report)
    return 0 if report["passed"] else 2


def outcome_main(argv: Sequence[str] | None = None) -> int:
    args = _parser("Classify the Track-B scientific outcome").parse_args(argv)
    destination = _destination(args.output)
    load = lambda name: json.loads((destination/name).read_text(encoding="utf-8"))
    report = scientific_outcome(
        load("plant_a_development.json"), load("plant_b_development.json"),
        load("residual_stratified_analysis.json"), load("resource_matched_analysis.json"))
    _write_json(destination/"scientific_outcome.json", report)
    print(report["classification"])
    return 0 if report["classification"].startswith("OUTCOME_C") else 2


def preregister_main(argv: Sequence[str] | None = None) -> int:
    args = _parser("Preregister Track-B confirmation only after Outcome C").parse_args(argv)
    destination = _destination(args.output)
    outcome = json.loads((destination/"scientific_outcome.json").read_text(encoding="utf-8"))
    substrate = json.loads((destination/"common_substrate_manifest.json").read_text(encoding="utf-8"))
    preregister_track_b(outcome, substrate, output=destination)
    print("FROZEN_NOT_EXECUTED")
    return 0


def all_main(argv: Sequence[str] | None = None) -> int:
    args = _parser("Run all bounded Track-B development analyses").parse_args(argv)
    config = TrackBConfig(); _announce_cost(config, "complete-track-b-development")
    bundle = run_track_b_development(config, _destination(args.output))
    print(bundle["outcome"]["classification"])
    return 0 if bundle["outcome"]["classification"].startswith("OUTCOME_C") else 2

