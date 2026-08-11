"""Run the labelled fault-injection safety suite."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .fault_injection import FaultInjectionRunner


def main() -> int:
    parser = argparse.ArgumentParser(description="Run HDFA-RL fault injection.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/fault-report.json"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = FaultInjectionRunner(args.seed).run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    print(args.output)
    return 0 if report.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
