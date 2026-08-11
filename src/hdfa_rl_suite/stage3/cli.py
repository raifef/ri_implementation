"""Run a deterministic joint Stage-3 filtering demonstration."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from hdfa_rl_suite.stage0.schema import DetectorDefinition
from hdfa_rl_suite.stage1 import CircuitContext, PolicyActivation, RawMeasurementRecord, TelemetryProcessor
from hdfa_rl_suite.stage2 import LatentVariable, QuadraticLogitObservationModel, StateSchema
from hdfa_rl_suite.stage2.schema import DetectorResponse

from .dynamics import JointDynamicsEngine, default_model_bank


def main() -> int:
    parser = argparse.ArgumentParser(description="Run joint detector-likelihood dynamics inference.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/stage3-dynamics.json"))
    args = parser.parse_args()
    detector = DetectorDefinition("d0", (0, 1), 0, ("g0",), "r0")
    context = CircuitContext("c", "demo", "Z", 3, "memory", "active")
    policy = PolicyActivation("p", "hash", -1., -1., 0., {"drive": 0.})
    schema = StateSchema("r0", (LatentVariable("detuning", "effective detuning", "norm", -1., 1.),))
    model = QuadraticLogitObservationModel(schema, (DetectorResponse("d0", -2., {"detuning": 3.}),))
    engine = JointDynamicsEngine(schema, model, default_model_bank("detuning"))
    outputs = []
    for batch_index in range(5):
        records = tuple(RawMeasurementRecord(f"{batch_index}-{i}", i, batch_index, i, batch_index + i * .001,
            (int((i + batch_index) % 3 == 0), 0), "c", ("m0", "m1")) for i in range(96))
        view = TelemetryProcessor((detector,), {"d0": ("drive",)}).process(records, (policy,), context).regional_views["r0"]
        outputs.append(engine.update(view, {"drive": 0.}, batch_index + 1.))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps([asdict(item) for item in outputs], indent=2, default=str), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
