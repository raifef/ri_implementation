"""End-to-end simulator closed-loop CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .product import HDFAProductController, ProductLoopConfig, QECOperabilityError
from .simulator import ScalableQECDevice, SimulatorConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the complete supervised HDFA-RL inference/control loop.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/closed-loop.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--qubits", type=int, default=5)
    parser.add_argument("--intervals", type=int, default=8)
    parser.add_argument("--cycles", type=int, default=64)
    parser.add_argument("--candidate-cycles", type=int, default=16)
    parser.add_argument("--residual-candidates", type=int, default=4)
    args = parser.parse_args()
    device = ScalableQECDevice(SimulatorConfig(qubit_count=args.qubits, seed=args.seed))
    loop = HDFAProductController(device, seed=args.seed, config=ProductLoopConfig(
        residual_candidate_count=args.residual_candidates,
        residual_candidate_cycles=args.candidate_cycles,
    ))
    intervals = []
    for index in range(args.intervals):
        try:
            result = loop.run_interval(args.cycles, interval=index)
        except QECOperabilityError as error:
            intervals.append({"interval": index, "status": "censored",
                              "reason": str(error), "bootstrap_reason": error.reason.value})
            break
        control = result.control
        intervals.append({
            "interval": index,
            "status": "completed",
            "detector_rate": result.feedback_observation.detector_rate,
            "logical_failure_proxy": result.feedback_observation.logical_failures,
            "mode": control.supervisor.mode.value,
            "authorization": result.authorization_log[-1].authorization.value,
            "bootstrap_reason": result.bootstrap_reason.value if result.bootstrap_reason else None,
            "bootstrap_count": result.bootstrap_count,
            "stage_path": result.stage_path,
            "residual_candidates": len(result.residual_candidates),
            "residual_policy_version": result.residual_result.policy_version if result.residual_result else None,
            "lifecycle_violations": result.lifecycle_violations,
            "policy_hash": device.confirmed_policy.policy_hash,
            "replay_hash": result.replay_hash,
            "region_validity": {region.region_id: region.state.validity.value for region in control.regions},
            "model_probabilities": {region.region_id: dict(region.dynamics.model_evidence.model_probabilities)
                                    for region in control.regions},
        })
    report = {"schema_version": "closed-loop.v2", "seed": args.seed, "qubits": args.qubits,
              "cycles_per_interval": args.cycles, "bootstrap_count": loop.bootstrap_count,
              "completed_without_lifecycle_violations": all(not item.get("lifecycle_violations") for item in intervals),
              "intervals": intervals}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
