"""Replay/demo CLI for Stage-1 telemetry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from hdfa_rl_suite.stage0.simulator import demo_topology

from .schema import CircuitContext, PolicyActivation, RawMeasurementRecord
from .telemetry import TelemetryProcessor


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a deterministic Stage-1 telemetry batch.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/stage1-telemetry.json"))
    args = parser.parse_args()
    _, _, circuit = demo_topology()
    rng = random.Random(args.seed)
    records = tuple(RawMeasurementRecord(f"r-{i}", i, i // 8, i % 8, i * 1e-3,
        tuple(int(rng.random() < .03) for _ in range(3)), circuit.circuit_hash, ("m0", "m1", "m2")) for i in range(64))
    policy = PolicyActivation("p0", "demo-policy", -1.0, -1.0, 0.0, {"frequency:q0": 5.1e9}, candidate_id="c0")
    context = CircuitContext(circuit.circuit_hash, "demo", "Z", circuit.code_distance, "memory", "active")
    graph = {detector.detector_id: ("frequency:q0", "amplitude:q0") for detector in circuit.detectors}
    batch = TelemetryProcessor(circuit.detectors, graph).process(records, (policy,), context)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(batch.to_dict(), indent=2, default=str), encoding="utf-8")
    print(args.output)
    return 0 if not batch.hard_invalid else 2


if __name__ == "__main__":
    raise SystemExit(main())
