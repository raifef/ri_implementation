"""Public development/preregistration commands from the post-v2 workflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .next_steps import (
    analyse_candidate_tail, analyse_recovery_latency, compare_periodic_end_to_end,
    create_estimator_artifacts, preregister_confirmatory_v3,
    run_all_post_amendment, run_one_interval_development,
    run_post_amendment_cohort, validate_report_estimators, validate_rmst_support,
)


def _output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/development"))


def recovery_latency_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Decompose familiar-process recovery latency")
    _output(parser); args = parser.parse_args(argv)
    report = analyse_recovery_latency(args.output)
    print((args.output/"recovery_latency_breakdown.json").resolve())
    print(f"slow_recovery_count={report['slow_recovery_count']}")
    return 0


def one_interval_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the development-only one-interval recurrence benchmark")
    _output(parser); args = parser.parse_args(argv)
    report = run_one_interval_development(args.output)
    print((args.output/"one_interval_recovery.json").resolve())
    print(f"passed={report['passed']} familiar_fraction={report['familiar_one_interval_fraction']:.6f}")
    return 0 if report["passed"] else 2


def periodic_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare periodic recalibration active-period and E2E utility")
    _output(parser); args = parser.parse_args(argv)
    report = compare_periodic_end_to_end(args.output)
    print((args.output/"periodic_end_to_end_comparison.json").resolve())
    print(f"periodic_wins_active={report['periodic_wins_active_detector_rate']}")
    return 0


def rmst_support_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail-closed validation of frozen RMST support")
    parser.add_argument("--config", type=Path,
                        default=Path("configs/acceptance/confirmatory-v3.yaml"))
    parser.add_argument("--report", type=Path)
    _output(parser); args = parser.parse_args(argv)
    report = validate_rmst_support(
        config_path=None if args.report else args.config, report_path=args.report)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/"rmst_support_validation.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path.resolve()); print(f"passed={report['passed']} rows={len(report['support_table'])}")
    return 0 if report["passed"] else 2


def candidate_tail_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyse candidate-efficiency tails")
    _output(parser); args = parser.parse_args(argv)
    report = analyse_candidate_tail(args.output)
    print((args.output/"candidate_tail_analysis.json").resolve())
    print(f"passed={report['passed']} saved_batches={report['total_saved_candidate_batches']}")
    return 0 if report["passed"] else 2


def estimator_validation_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate estimator/CI consistency in a benchmark report")
    parser.add_argument("--report", type=Path,
                        default=Path("artifacts/development/estimator_consistency.json"))
    _output(parser); args = parser.parse_args(argv)
    if not args.report.exists():
        cohort = run_post_amendment_cohort(args.output)
        create_estimator_artifacts(args.output, cohort)
    report = validate_report_estimators(args.report)
    path = args.output/"report_estimator_validation.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path.resolve()); print(f"passed={report['passed']} issues={len(report['issues'])}")
    return 0 if report["passed"] else 2


def post_amendment_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run all short post-amendment development gates")
    parser.add_argument("--root", type=Path, default=Path(".")); args = parser.parse_args(argv)
    report = run_all_post_amendment(args.root.resolve())
    print((args.root/"artifacts/development/post_amendment_cohort.json").resolve())
    print(f"passed={report['passed']}")
    return 0 if report["passed"] else 2


def preregister_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze confirmatory v3 without running it")
    parser.add_argument("--root", type=Path, default=Path(".")); args = parser.parse_args(argv)
    # Re-evaluate the short prerequisites so a stale/failing development bundle cannot
    # be preregistered by accident. This never instantiates a confirmatory-v3 tape.
    report = run_all_post_amendment(args.root.resolve())
    prereg = report["preregistration"]
    print((args.root/"artifacts/acceptance/confirmatory-v3-preregistration.json").resolve())
    print("status=frozen_not_executed")
    print(f"fresh_seeds={prereg['seed_sets_disjoint']} rmst_support={prereg['rmst_support_validated']}")
    print(prereg["future_command_powershell"])
    return 0 if report["passed"] else 2

