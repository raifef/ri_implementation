"""Exercise the deterministic Stage-7 supervisory state machine."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from hdfa_rl_suite.stage0.schema import ControlBound, HardwareLimits

from .schema import DiagnosticOption, SupervisorInput
from .supervisor import SupervisoryController


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Stage-7 supervisory decision.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/stage7-decision.json"))
    args = parser.parse_args()
    supervisor = SupervisoryController(HardwareLimits({"drive": ControlBound(-1, 1, 1, "norm", 1)}), (DiagnosticOption("ramsey-sign", .5, .1, .01),))
    decision = supervisor.tick(SupervisorInput(1., (), forecast_valid=True, residual_small=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(decision), indent=2, default=str), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
