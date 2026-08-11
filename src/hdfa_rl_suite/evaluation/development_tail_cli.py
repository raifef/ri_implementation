"""CLI for the development-only recovery-tail/ablation cohort."""
from __future__ import annotations

import argparse
from pathlib import Path

from .development_tail import run_development_tail


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run censored development-tail profiling; never authoritative.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--launch", type=str, default=
                        "experiments/physical_validation/authoritative-comparison-v1.json")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-ablations", action="store_true")
    args = parser.parse_args()
    paths = run_development_tail(
        args.output, launch_path=args.launch, workers=max(1, args.workers),
        include_ablations=not args.no_ablations)
    for path in paths:
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
