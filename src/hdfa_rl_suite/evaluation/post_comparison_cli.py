"""CLI for compact development-only post-comparison replay."""
from __future__ import annotations

import argparse
from pathlib import Path

from .post_comparison import write_diagnostic_bundle


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay compact post-comparison failure timelines without loading the full report.")
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/development/post-comparison-v1/pre-repair"))
    parser.add_argument(
        "--launch", type=Path,
        default=Path("experiments/physical_validation/authoritative-comparison-v1.json"))
    args = parser.parse_args()
    timeline, report = write_diagnostic_bundle(args.output, launch_path=args.launch)
    print(timeline)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
