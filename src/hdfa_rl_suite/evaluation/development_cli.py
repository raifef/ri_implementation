from __future__ import annotations

import argparse
from pathlib import Path

from .residual_ablation import run_residual_ablation, validate_residual_gating
from .rollback_reproduction import reproduce_rollbacks_v2, validate_rollback_semantics


def _parts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--part-1", type=Path, default=Path(
        "artifacts/acceptance/compute-aware-v2/authoritative-comparison-v2.part-1.json.gz"))
    parser.add_argument("--part-2", type=Path, default=Path(
        "artifacts/acceptance/compute-aware-v2/authoritative-comparison-v2.part-2.json.gz"))


def reproduce_rollbacks_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproduce the four retained v2 rollback anomalies")
    _parts(parser)
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/development/rollback_reproductions"))
    args = parser.parse_args(argv)
    report = reproduce_rollbacks_v2((args.part_1, args.part_2), args.output)
    print(args.output/"rollback_classification_report.md")
    print(f"all_four_reproduced={report['all_four_reproduced']}")
    return 0 if report["all_four_reproduced"] else 2


def rollback_semantics_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate four-state rollback semantics")
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/development/rollback_reproductions"))
    args = parser.parse_args(argv)
    report = validate_rollback_semantics(args.output)
    print(args.output/"rollback-semantics-validation.json")
    print(f"passed={report['passed']}")
    return 0 if report["passed"] else 2


def residual_gating_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate conditional residual-RL authority")
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/development"))
    args = parser.parse_args(argv)
    report = validate_residual_gating(args.output)
    print(args.output/"residual-rl-gating-validation.json")
    print(f"passed={report['passed']}")
    return 0 if report["passed"] else 2


def residual_ablation_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the development-only residual-RL ablation")
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/development"))
    args = parser.parse_args(argv)
    report = run_residual_ablation(args.output)
    print(args.output/"residual_rl_ablation.json")
    print(args.output/"residual_rl_ablation.md")
    print(f"claim_supported={report['claim_supported']}")
    return 0 if report["claim_supported"] else 1


if __name__ == "__main__":
    raise SystemExit(residual_gating_main())

