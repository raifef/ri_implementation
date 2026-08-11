"""Public Track-A commands with pre-execution runtime and QEC-cost estimates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .audit import write_audit_artifacts
from .config import named_config
from .report import (write_budget_equivalence_artifacts, write_final_certification,
                     write_high_shot_artifacts)
from .steering_frequency import DEFAULT_PERIODS, run_steering_frequency_sweep
from .common import write_json
from .report import artifact_directory


def _output_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def _print_cost(label: str, config_name: str, epoch_equivalents: int,
                runtime: str) -> None:
    config = named_config(config_name)
    cost = config.estimated_cost(epoch_equivalents)
    print(json.dumps({
        "command": label,
        "configuration": config_name,
        "expected_runtime": runtime,
        "simulated_epoch_equivalents": epoch_equivalents,
        "estimated_qec_cycle_cost": cost,
        "notice": "vectorized surrogate Monte Carlo only; no QPU acquisition",
    }, indent=2))


def audit_main(argv: Sequence[str] | None = None) -> int:
    parser = _output_parser("Audit the current Google-style RL implementation.")
    args = parser.parse_args(argv)
    payload = write_audit_artifacts(artifact_directory(args.output))
    print(json.dumps({"status": "AUDIT_COMPLETE", "defects": len(payload["defects"])}, indent=2))
    return 0


def high_shot_main(argv: Sequence[str] | None = None) -> int:
    parser = _output_parser("Run the limited high-shot Track-A certification.")
    args = parser.parse_args(argv)
    seed = 8801 if args.seed is None else args.seed
    _print_cost("hdfa-certify-google-rl-high-shot", "high_shot_reference", 7768,
                "approximately 10-30 seconds on the bundled runtime")
    payload = write_high_shot_artifacts(args.output, seed=seed)
    print(json.dumps({"status": payload["status"], "runtime_s": payload["runtime_s"]}, indent=2))
    return 0 if payload["passed"] else 2


def compare_budgets_main(argv: Sequence[str] | None = None) -> int:
    parser = _output_parser("Compare reduced-budget RL with the high-shot reference.")
    args = parser.parse_args(argv)
    seed = 9901 if args.seed is None else args.seed
    _print_cost("hdfa-compare-google-rl-budgets", "high_shot_reference", 8050,
                "approximately 20-60 seconds on the bundled runtime")
    _print_cost("hdfa-compare-google-rl-budgets", "reduced_budget_candidate", 8050,
                "approximately 20-60 seconds on the bundled runtime")
    payload = write_budget_equivalence_artifacts(args.output, seed=seed)
    print(json.dumps({"status": payload["status"]}, indent=2))
    return 0 if payload["passed"] else 2


def steering_main(argv: Sequence[str] | None = None) -> int:
    parser = _output_parser("Run the Google-RL steering-frequency sweep.")
    parser.add_argument("--configuration", choices=(
        "high_shot_reference", "reduced_budget_candidate"),
        default="high_shot_reference")
    args = parser.parse_args(argv)
    seed = 5501 if args.seed is None else args.seed
    epochs = sum(max(600, 2*period) for period in DEFAULT_PERIODS)
    _print_cost("hdfa-google-rl-steering-sweep", args.configuration, epochs,
                "approximately 5-15 seconds on the bundled runtime")
    payload = run_steering_frequency_sweep(named_config(args.configuration), seed=seed)
    output = artifact_directory(args.output)
    write_json(output/"steering_frequency.json", payload)
    print(json.dumps({"status": "PASS" if payload["passed"] else "FAIL",
                      "critical_period_epochs": payload["critical_period_epochs"]}, indent=2))
    return 0 if payload["passed"] else 2


def report_main(argv: Sequence[str] | None = None) -> int:
    parser = _output_parser("Build the fail-closed final Google-RL certification report.")
    args = parser.parse_args(argv)
    payload = write_final_certification(args.output)
    print(json.dumps({"status": payload["status"],
                      "track_b_prerequisite_satisfied": payload["track_b_prerequisite_satisfied"]}, indent=2))
    return 0 if payload["track_b_prerequisite_satisfied"] else 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Google-style RL certification commands")
    parser.add_argument("command", choices=("audit", "high-shot", "compare", "steering", "report"))
    args, remainder = parser.parse_known_args(argv)
    functions = {"audit": audit_main, "high-shot": high_shot_main,
                 "compare": compare_budgets_main, "steering": steering_main,
                 "report": report_main}
    return functions[args.command](remainder)


if __name__ == "__main__":
    raise SystemExit(main())
