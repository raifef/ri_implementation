"""Public command-line entry points for physical validation."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Sequence

from .common import ValidationReport, write_report
from .controller_sanity import run_controller_validation
from .development_cohort import run_development_cohort
from .fault_matrix import run_fault_matrix_validation
from .lifecycle_sanity import run_lifecycle_validation
from .performance import run_performance_validation
from .plant_sanity import run_plant_validation
from .preflight import run_preflight
from .manifest import build_preflight_manifest, write_preflight_manifest
from .sample_budget import run_sample_budget_validation
from .report_sanity import run_report_validation


Runner = Callable[..., ValidationReport]


def _run(argv: Sequence[str] | None, *, runner: Runner, stem: str,
         description: str) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output", type=Path, default=Path("artifacts/validation"),
                        help="directory for JSON and Markdown artifacts")
    parser.add_argument("--inject-fault", action="append", default=[],
                        help="test-only named fault; repeatable")
    args = parser.parse_args(argv)
    report = runner(injected_faults=tuple(args.inject_fault))
    json_path, markdown_path = write_report(report, args.output, stem)
    print(json_path.resolve())
    print(markdown_path.resolve())
    print(f"passed={report.passed} report_hash={report.report_hash}")
    return 0 if report.passed else 2


def plant_main(argv: Sequence[str] | None = None) -> int:
    return _run(argv, runner=run_plant_validation, stem="plant-sanity",
                description="Validate canonical QEC calibration plant invariants.")


def full_rl_main(argv: Sequence[str] | None = None) -> int:
    return _run(argv, runner=run_controller_validation, stem="full-rl-sanity",
                description="Validate the Google-style full-control detector-RL baseline.")


def sample_budget_main(argv: Sequence[str] | None = None) -> int:
    return _run(argv, runner=run_sample_budget_validation, stem="sample-budget",
                description="Validate cycles-per-candidate against finite-shot gradient quality.")


def lifecycle_main(argv: Sequence[str] | None = None) -> int:
    return _run(argv, runner=run_lifecycle_validation, stem="policy-lifecycle",
                description="Validate transactional policy activation and rollback semantics.")


def report_main(argv: Sequence[str] | None = None) -> int:
    return _run(argv, runner=run_report_validation, stem="report-contract",
                description="Validate evidence layers, metrics, convergence, and censoring schemas.")


def performance_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Profile scalar and optimized Stage 2--6 kernels with equivalence gates.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/validation"))
    args = parser.parse_args(argv)
    report = run_performance_validation()
    json_path, markdown_path = write_report(report, args.output, "stage2-6-performance")
    print(json_path.resolve())
    print(markdown_path.resolve())
    print(f"passed={report.passed} report_hash={report.report_hash}")
    return 0 if report.passed else 2


def fault_matrix_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inject all fifteen predeclared benchmark-invalidating faults.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/validation"))
    args = parser.parse_args(argv)
    report = run_fault_matrix_validation()
    json_path, markdown_path = write_report(report, args.output, "fault-matrix")
    print(json_path.resolve())
    print(markdown_path.resolve())
    print(f"passed={report.passed} report_hash={report.report_hash}")
    return 0 if report.passed else 2


def development_cohort_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the short held-out matched baseline development cohort.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/validation"))
    args = parser.parse_args(argv)
    report = run_development_cohort(
        output_dir=args.output / "development-cohort-figures", generate_figures=True)
    json_path, markdown_path = write_report(
        report, args.output, "development-cohort")
    print(json_path.resolve())
    print(markdown_path.resolve())
    print(f"passed={report.passed} report_hash={report.report_hash}")
    return 0 if report.passed else 2


def preflight_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run preflight and bind a fresh manifest to one exact benchmark configuration.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/validation"))
    parser.add_argument("--benchmark-config", type=Path, required=True,
                        help="JSON launch definition used by the later benchmark command")
    parser.add_argument("--maximum-age-hours", type=float, default=24.0)
    parser.add_argument("--inject-fault", action="append", default=[])
    args = parser.parse_args(argv)
    from hdfa_rl_suite.evaluation.launch import load_launch_definition
    definition = load_launch_definition(args.benchmark_config)
    report = run_preflight(injected_faults=tuple(args.inject_fault))
    json_path, markdown_path = write_report(
        report, args.output, "benchmark-preflight")
    manifest = build_preflight_manifest(
        report, definition.configuration_hash,
        maximum_age_hours=args.maximum_age_hours)
    manifest_path = write_preflight_manifest(
        manifest, args.output / "benchmark-preflight-manifest.json")
    print(json_path.resolve())
    print(markdown_path.resolve())
    print(manifest_path.resolve())
    print(f"passed={manifest.passed} manifest_hash={manifest.manifest_hash}")
    return 0 if manifest.passed else 2


if __name__ == "__main__":
    raise SystemExit(preflight_main())
