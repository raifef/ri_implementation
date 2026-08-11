"""CLI for the Nature-2026-matched scalability experiment."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

from .scalability import (
    PipelineCheckpointError,
    ScalabilityConfig,
    ScalabilityRunner,
    with_pipeline_probe,
)
from .scalability_artifacts import write_scalability_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Figure-5-matched scalability/steerability sweeps with explicit evidence layers.")
    parser.add_argument("--profile", choices=("smoke", "paper", "full"), default="smoke",
                        help="paper matches d=3..15/P=1,10,30/500 epochs; full adds five seeds and pipeline probes")
    parser.add_argument("--config", type=Path,
                        help="JSON overrides applied to the selected/profile field in the file")
    parser.add_argument("--output", type=Path, default=Path("artifacts/scalability/nature-2026"))
    parser.add_argument("--pipeline-probe", action="store_true",
                        help="execute the actual suite controllers in addition to the reduced surrogate")
    parser.add_argument("--no-pipeline-probe", action="store_true",
                        help="disable actual suite probes even for the full profile")
    parser.add_argument("--max-pipeline-distance", type=int,
                        help="cap expensive end-to-end probes without changing the Figure-5 surrogate sweep")
    parser.add_argument("--pipeline-epochs", type=int)
    parser.add_argument("--pipeline-workers", type=int,
                        help="independent distance/seed worker processes; timings record this contention context")
    parser.add_argument("--checkpoint-directory", type=Path,
                        help="atomically persist each completed distance/seed condition")
    parser.add_argument("--resume", action="store_true",
                        help="reuse valid completed condition checkpoints; worker count may change and is recorded per row")
    parser.add_argument("--seed", type=int, action="append",
                        help="override paired seeds; repeat this option for multiple seeds")
    args = parser.parse_args()
    profile = args.profile
    overrides = {}
    if args.config:
        overrides = json.loads(args.config.read_text(encoding="utf-8"))
        profile = overrides.pop("profile", profile)
        tuple_fields = {
            "distances", "parameters_per_gate", "seeds", "steering_frequencies",
            "entropy_regularizations", "pipeline_distances",
        }
        overrides = {key: tuple(value) if key in tuple_fields else value for key, value in overrides.items()}
    config = replace(ScalabilityConfig.for_profile(profile), **overrides)
    if args.seed:
        config = replace(config, seeds=tuple(args.seed))
    if args.pipeline_epochs is not None:
        config = replace(config, pipeline_epochs=args.pipeline_epochs)
    if args.pipeline_workers is not None:
        config = replace(config, pipeline_workers=args.pipeline_workers)
    if args.no_pipeline_probe:
        config = replace(config, run_pipeline_probe=False)
    elif args.pipeline_probe or config.run_pipeline_probe:
        config = with_pipeline_probe(config, maximum_distance=args.max_pipeline_distance)
    checkpoint_directory = args.checkpoint_directory
    if args.resume and checkpoint_directory is None:
        checkpoint_directory = args.output / "checkpoints"
    try:
        report = ScalabilityRunner(
            config, checkpoint_directory=checkpoint_directory, resume=args.resume).run()
    except PipelineCheckpointError as error:
        print(f"INVALID CHECKPOINT: {error}", file=sys.stderr)
        return 3
    paths = write_scalability_artifacts(report, args.output)
    print(paths["report"])
    print(paths["manifest"])
    for failure in report.pipeline_failures:
        print(f"PIPELINE {failure.status.upper()}: d={failure.code_distance} seed={failure.seed} "
              f"method={failure.method} phase={failure.phase} - {failure.reason}")
    if any(failure.status == "missing" for failure in report.pipeline_failures):
        return 3
    failures = [gate for gate in report.gates if gate.status == "fail"]
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
