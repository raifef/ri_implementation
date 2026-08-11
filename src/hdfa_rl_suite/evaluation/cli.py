"""Run the paired-seed benchmark suite and write a machine-readable report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hdfa_rl_suite import __version__
from hdfa_rl_suite.common import TimingEnvironment
from hdfa_rl_suite.simulator import SIMULATOR_VERSION
from hdfa_rl_suite.validation.controller_sanity import CONTROLLER_VERSION
from hdfa_rl_suite.validation.preflight import source_tree_hash

from .benchmark import (PRIMARY_ARMS, BenchmarkConfig, BenchmarkPreflightError,
                        BenchmarkRunner, default_benchmark_scenarios)
from .launch import load_launch_definition


def main() -> int:
    parser = argparse.ArgumentParser(description="Run falsifiable HDFA-RL comparison benchmarks.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark-report.json"))
    parser.add_argument("--config", type=Path,
                        help="exact JSON launch definition also bound into the preflight manifest")
    parser.add_argument("--preflight-manifest", type=Path,
                        help="fresh passing manifest; required for authoritative acquisition")
    parser.add_argument(
        "--validate-only", action="store_true",
        help="verify launch, source, timing environment and preflight hashes without acquiring a held-out tape")
    parser.add_argument("--qubits", type=int, default=5)
    parser.add_argument("--intervals", type=int, default=8)
    parser.add_argument("--cycles", type=int, default=64)
    parser.add_argument("--candidate-cycles", type=int, default=2048,
                        help="cycles per candidate; 2048 is a Track-A reduced-budget candidate, 100000 is the high-shot reference")
    parser.add_argument("--logical-shots", type=int, default=256)
    parser.add_argument("--censoring-intervals", type=int)
    parser.add_argument("--bootstrap-shots", type=int, default=96)
    parser.add_argument("--bootstrap-cycles", type=int, default=128)
    parser.add_argument("--bootstrap-target-stddev", type=float, default=.06)
    parser.add_argument("--bootstrap-edr-limit", type=float, default=.15)
    parser.add_argument("--bootstrap-block-familywise-alpha", type=float, default=1e-4,
                        help="experiment-wide false-rejection rate for held-out Stage-0 block checks")
    parser.add_argument("--baseline-cycles", type=int, default=128,
                        help="held-out native-QEC cycles acquired before synchronized disturbance onset")
    parser.add_argument("--primary-only", action="store_true",
                        help="run only the six mandatory primary comparison arms")
    parser.add_argument("--scenario", action="append",
                        help="scenario id to run (repeatable; default runs the full declared suite)")
    parser.add_argument("--seed", type=int, action="append")
    args = parser.parse_args()
    definition = load_launch_definition(args.config) if args.config else None
    config = (definition.config if definition is not None else BenchmarkConfig(
        qubit_count=args.qubits, intervals=args.intervals, cycles_per_interval=args.cycles,
        seeds=tuple(args.seed) if args.seed else (3, 11),
        candidate_cycles=args.candidate_cycles,
        logical_shots_per_interval=args.logical_shots,
        censoring_limit_intervals=args.censoring_intervals,
        bootstrap_characterization_shots=args.bootstrap_shots,
        bootstrap_validation_cycles=args.bootstrap_cycles,
        bootstrap_target_stddev=args.bootstrap_target_stddev,
        bootstrap_qec_rate_limit=args.bootstrap_edr_limit,
        bootstrap_block_familywise_alpha=args.bootstrap_block_familywise_alpha,
        pre_disturbance_baseline_cycles=args.baseline_cycles,
    ))
    scenarios = (definition.scenarios() if definition is not None
                 else default_benchmark_scenarios(config.qubit_count))
    if args.scenario:
        requested = set(args.scenario)
        scenarios = tuple(item for item in scenarios if item.scenario_id in requested)
        missing = requested - {item.scenario_id for item in scenarios}
        if missing:
            parser.error(f"unknown scenarios: {sorted(missing)}")
    runner = BenchmarkRunner(config, scenarios)
    primary_only = definition.primary_only if definition is not None else args.primary_only
    if primary_only:
        factories = {name: factory for name, factory in runner.arm_factories.items()
                     if name in PRIMARY_ARMS}
        runner = BenchmarkRunner(
            config, scenarios, factories,
            preflight_manifest=args.preflight_manifest,
            launch_binding_hash=(definition.configuration_hash
                                 if definition is not None else None))
    else:
        runner = BenchmarkRunner(
            config, scenarios, runner.arm_factories,
            preflight_manifest=args.preflight_manifest,
            launch_binding_hash=(definition.configuration_hash
                                 if definition is not None else None))
    if args.validate_only:
        try:
            manifest_hash = runner._require_preflight()
        except BenchmarkPreflightError as error:
            print(f"INVALID: {error}")
            return 3
        timing = TimingEnvironment.capture(__version__)
        print("VALID: no benchmark arm or disturbance tape was acquired")
        print(f"configuration_hash={runner.launch_configuration_hash}")
        print(f"preflight_manifest_hash={manifest_hash}")
        print(f"source_tree_hash={source_tree_hash()}")
        print(f"timing_environment_hash={timing.environment_hash}")
        print(f"package_version={__version__}")
        print(f"controller_version={CONTROLLER_VERSION}")
        print(f"simulator_version={SIMULATOR_VERSION}")
        return 0
    try:
        report = runner.run()
    except BenchmarkPreflightError as error:
        print(f"INVALID: {error}")
        return 3
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    print(args.output)
    print(f"authoritative={report.authoritative} accepted={report.accepted}")
    if not report.authoritative:
        for reason in report.invalidity_reasons:
            print(f"INVALID: {reason}")
        return 3
    if not report.accepted:
        for reason in report.acceptance_failure_reasons:
            print(f"REJECTED: {reason}")
        for gate in report.gates:
            if gate.status != "pass":
                print(f"REJECTED GATE: {gate.gate_id} ({gate.status}) - {gate.rationale}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
