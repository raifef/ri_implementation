"""Run a deterministic Stage-2 inference example from Stage-1-compatible telemetry."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from hdfa_rl_suite.stage0.schema import DetectorDefinition
from hdfa_rl_suite.stage1 import CircuitContext, PolicyActivation, RawMeasurementRecord, TelemetryProcessor

from .inference import PhysicalInferenceEngine, QuadraticLogitObservationModel
from .schema import DetectorResponse, LatentVariable, StateSchema


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local detector-to-state inference.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/stage2-posterior.json"))
    args = parser.parse_args()
    definition = DetectorDefinition("d0", (0, 1), 0, ("g0",), "r0")
    records = tuple(RawMeasurementRecord(f"r{i}", i, 0, i, i * 1e-3, (int(i % 3 == 0), 0), "c", ("m0", "m1")) for i in range(96))
    context = CircuitContext("c", "demo", "Z", 3, "memory", "active")
    policy = PolicyActivation("p", "hash", -1, -1, 0, {"drive": 0.0})
    view = TelemetryProcessor((definition,), {"d0": ("drive",)}).process(records, (policy,), context).regional_views["r0"]
    schema = StateSchema("r0", (LatentVariable("detuning", "effective detuning", "normalized", -1, 1, intervention_control="drive", safe_intervention=.1),))
    model = QuadraticLogitObservationModel(schema, (DetectorResponse("d0", -2.0, {"detuning": 3.0}),))
    posterior = PhysicalInferenceEngine(schema, model).infer(view, {"drive": 0.0})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(posterior), indent=2, default=str), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
