"""Run a small Stage-4 forecast from a joint dynamics posterior."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from hdfa_rl_suite.stage0.schema import DetectorDefinition
from hdfa_rl_suite.stage1 import CircuitContext, PolicyActivation, RawMeasurementRecord, TelemetryProcessor
from hdfa_rl_suite.stage2 import LatentVariable, QuadraticLogitObservationModel, StateSchema
from hdfa_rl_suite.stage2.schema import DetectorResponse
from hdfa_rl_suite.stage3 import JointDynamicsEngine, default_model_bank

from .forecast import ForecastEngine
from .schema import LatencyModel, ResponseMap


def main() -> int:
    parser = argparse.ArgumentParser(description="Create activation-aligned Stage-4 forecast scenarios.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/stage4-forecast.json"))
    args = parser.parse_args()
    detector = DetectorDefinition("d0", (0, 1), 0, ("g0",), "r0")
    records = tuple(RawMeasurementRecord(f"r{i}", i, 0, i, i*.001, (int(i % 3 == 0), 0), "c", ("m0", "m1")) for i in range(96))
    view = TelemetryProcessor((detector,), {"d0": ("drive",)}).process(records, (PolicyActivation("p", "h", -1, -1, 0, {"drive": 0}),), CircuitContext("c", "demo", "Z", 3, "memory", "active")).regional_views["r0"]
    schema = StateSchema("r0", (LatentVariable("detuning", "detuning", "norm", -1, 1),))
    observation = QuadraticLogitObservationModel(schema, (DetectorResponse("d0", -2, {"detuning": 3}),))
    dynamics = JointDynamicsEngine(schema, observation, default_model_bank("detuning")).update(view, {"drive": 0.}, 1.)
    response = ResponseMap({"drive": 0.}, {("drive", "detuning"): 1.})
    bundle = ForecastEngine(observation, default_model_bank("detuning"), response).forecast(dynamics, {"drive": 0.}, (0., .1, 1.), LatencyModel(.01, .01, .01, .01))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(bundle), indent=2, default=str), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
