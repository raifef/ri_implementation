"""Run one safe residual-RL antithetic update against synthetic detector rewards."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from hdfa_rl_suite.stage0.schema import PolicySnapshot
from hdfa_rl_suite.stage5.schema import PredictedCostDistribution, PredictiveControlPackage, ResidualAllocation, SolverStatus

from .residual_rl import ExplorationBudget, GaussianResidualPolicy, ResidualRLController
from .schema import CandidateObservation


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a residual detector-RL microbatch.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/stage6-residual.json"))
    args = parser.parse_args()
    snapshot = PolicySnapshot({"drive": 0.}, "baseline", 1.)
    allocation = ResidualAllocation(("drive",), {"drive": .1}, ("drive",), {"drive": "demo"})
    package = PredictiveControlPackage("stage5.v1", SolverStatus.OPTIMAL, {"drive": 0.}, ({"drive": 0.},), {"drive": 0.}, allocation, (), PredictedCostDistribution(0., 0., {}), snapshot, "action", 1., 2., snapshot)
    controller = ResidualRLController(GaussianResidualPolicy.full_control_baseline(("drive",), .03), {"d0": ("drive",)}, ExplorationBudget(.05, 1.))
    candidates = controller.propose(package)
    observations = tuple(CandidateObservation(item.candidate_id, {"d0": .1 if item.sign > 0 else .2}, {"d0": 100}, regime_id="demo", context_id="demo", model_version="v1") for item in candidates)
    result = controller.update(package, observations, current_regime="demo", current_context="demo", current_model_version="v1")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(result), indent=2, default=str), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
