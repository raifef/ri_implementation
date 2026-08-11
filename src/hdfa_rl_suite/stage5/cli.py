"""Create a Stage-5 predictive control package from a small scenario set."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from hdfa_rl_suite.stage0.schema import ControlBound, HardwareLimits, PolicySnapshot
from hdfa_rl_suite.stage2 import LatentVariable, QuadraticLogitObservationModel, StateSchema
from hdfa_rl_suite.stage2.schema import DetectorResponse
from hdfa_rl_suite.stage4.schema import ForecastBundle, ForecastCalibration, ForecastRisk, ForecastScenario, LatencyModel

from .mpc import PredictiveController


def main() -> int:
    parser = argparse.ArgumentParser(description="Solve a deterministic scenario-MPC control package.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/stage5-control.json"))
    args = parser.parse_args()
    scenario = ForecastScenario(0., .04, "ou", {"detuning": .4}, {"drive": .4}, {"d0": .05}, 1.)
    forecast = ForecastBundle("stage4.v1", "r0", 1., LatencyModel(.01, .01, .01, .01), {0.: (scenario,)}, {0.: ForecastRisk({"d0": 0.}, 0., {"detuning": 0.}, {"drive": 0.}, 0.)}, 1., ForecastCalibration(0, None, None, None), ())
    limits = HardwareLimits({"drive": ControlBound(-1., 1., 1., "norm", 1.)})
    schema = StateSchema("r0", (LatentVariable("detuning", "detuning", "norm", -1., 1.),))
    observation = QuadraticLogitObservationModel(schema, (DetectorResponse("d0", -3., {"detuning": 4.}, {"drive": -4.}),))
    package = PredictiveController(limits, observation).solve(forecast, 0., PolicySnapshot({"drive": 0.}, "current", 1.))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(package), indent=2, default=str), encoding="utf-8")
    print(args.output)
    return 0 if package.status.value == "optimal" else 2


if __name__ == "__main__":
    raise SystemExit(main())
