"""CLI for deterministic Stage-0 bootstrap artifacts."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from .bootstrap import BootstrapCalibrator, BootstrapConfig
from .scalable import ScalableBootstrapCalibrator
from .simulator import SimulatedCalibrationBackend, demo_topology
from hdfa_rl_suite.simulator import ScalableQECDevice, SimulatorConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Run scalable Stage-0 bootstrap on the deterministic QEC simulator.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--qubits", type=int, default=5)
    parser.add_argument("--legacy-demo", action="store_true", help="run the original two-qubit regression fixture")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if args.legacy_demo:
        topology, limits, circuit = demo_topology()
        backend = SimulatedCalibrationBackend(topology, limits, seed=args.seed)
        result = BootstrapCalibrator(topology, limits, circuit, backend, BootstrapConfig(seed=args.seed)).run()
    else:
        result = ScalableBootstrapCalibrator(ScalableQECDevice(SimulatorConfig(qubit_count=args.qubits, seed=args.seed))).run()
    output = args.artifacts / f"stage0-{time.strftime('%Y%m%d-%H%M%S')}-{args.seed}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "bootstrap_result.json").write_text(json.dumps(asdict(result), indent=2, default=str), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps({"status": result.health.status.value, "policy_hash": result.baseline_policy.policy_hash, "replay_verified": BootstrapCalibrator.verify_replay(result)}, indent=2), encoding="utf-8")
    print(output)
    return 0 if result.health.status.value == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
