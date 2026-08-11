"""CLI for auditable, resumable detector-sensitivity normalization."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

from .contracts import (
    CalibrationBundle,
    ControlTypeSpec,
    FitRules,
    IterationRecord,
    SweepProtocol,
    SweepResult,
    atomic_json,
    build_source_contract,
    canonical_hash,
    file_sha256,
)
from .edr_measurement import StimSurfaceCodeEDREvaluator
from .perturbation_sweeps import merge_sweep_shards, run_control_type_sweep, shard_sigmas
from .quadratic_fit import FitRejected, fit_all_sweeps
from .validation import (
    audit_no_arbitrary_scale,
    build_calibration_bundle_from_sweeps,
    run_independent_validation,
    validate_coefficient_remeasurement,
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_config_path() -> Path:
    return repository_root() / "configs" / "google_pure_source_exact" / "control_normalization.json"


def default_artifact_root() -> Path:
    return repository_root() / "artifacts" / "google_pure_source_exact" / "control_normalization"


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "google-pure-source-exact-control-normalization-config.v1":
        raise ValueError("unsupported normalization config schema")
    scale_audit = audit_no_arbitrary_scale(config)
    if not scale_audit["passed"]:
        raise ValueError(f"source-exact config contains an arbitrary old scale: {scale_audit['findings']}")
    return config


def build_specs(config: dict[str, Any]) -> tuple[ControlTypeSpec, ...]:
    return tuple(ControlTypeSpec.from_dict(item) for item in config["control_types"])


def build_evaluator(config: dict[str, Any], specs: Sequence[ControlTypeSpec]):
    circuit = config["circuit"]
    return StimSurfaceCodeEDREvaluator(
        specs,
        distance=int(circuit["distance"]),
        rounds=int(circuit["rounds"]),
        basis=str(circuit["basis"]),
        base_noise=circuit["base_noise"],
        maximum_probability=float(circuit["maximum_probability"]),
    )


def build_protocol(config: dict[str, Any], profile: str, *, stability: bool = False) -> SweepProtocol:
    values = config["profiles"][profile]
    multiplier = 2 if stability else 1
    offset = 500_000 if stability else 0
    reserved = set(config["seed_registry"]["certification_reserved"])
    seeds = (int(values["perturbation_seed"]) + offset, int(values["detector_seed"]) + offset)
    if any(seed in reserved for seed in seeds):
        raise RuntimeError("certification seed cannot be consumed by a development/profile run")
    return SweepProtocol(
        candidates_per_sigma=int(values["candidates_per_sigma"]),
        shots_per_candidate=int(values["shots_per_candidate"]) * multiplier,
        qec_rounds_per_shot=int(config["circuit"]["rounds"]),
        perturbation_seed=seeds[0],
        detector_seed=seeds[1],
    )


def code_hash() -> str:
    root = Path(__file__).resolve().parent
    return canonical_hash({path.name: file_sha256(path) for path in sorted(root.glob("*.py"))})


def source_commit(current_code_hash: str) -> str:
    # This workspace is intentionally usable without Git metadata.  The code
    # hash remains an immutable source identity when no commit is available.
    return f"NO_GIT_METADATA:{current_code_hash[:16]}"


def _checkpoint_path(output: Path, iteration_id: str, control_type: str,
                     shard_index: int, shard_count: int) -> Path:
    return output / "checkpoints" / iteration_id / (
        f"{control_type}.shard-{shard_index:03d}-of-{shard_count:03d}.json")


def resource_plan(config: dict[str, Any], profile: str, output: Path,
                  iteration_id: str, shard_count: int) -> dict[str, Any]:
    specs = build_specs(config)
    values = config["profiles"][profile]
    rounds = int(config["circuit"]["rounds"])
    point_count = sum(len(item.sweep_sigmas_native) for item in specs)
    primary_cycles = point_count * int(values["candidates_per_sigma"]) * int(
        values["shots_per_candidate"]) * rounds
    stability_cycles = 2 * primary_cycles
    validation_measurements = len(specs) + 3
    validation_cycles = validation_measurements * int(values["validation_candidates"]) * int(
        values["validation_shots_per_candidate"]) * rounds
    total_cycles = primary_cycles + stability_cycles + validation_cycles
    checkpoints = [
        str(_checkpoint_path(output, iteration_id, spec.control_type, index, shard_count))
        for index in range(shard_count) for spec in specs
    ]
    command = (
        f"$env:PYTHONPATH=\"{repository_root() / 'src'}\"\n"
        f"& \"{sys.executable}\" -X utf8 -m "
        "hdfa_rl_suite.google_pure_source_exact.control_normalization.cli "
        f"run --profile {profile} --iteration-id {iteration_id}"
    )
    start_commands = [
        command + (f" --shard-index {index} --shard-count {shard_count}" if shard_count > 1 else "")
        for index in range(shard_count)
    ]
    shard_commands = [
        command + f" --shard-index {index} --shard-count {shard_count} --resume"
        for index in range(shard_count)
    ]
    result = {
        "schema_version": "google-pure-source-exact-resource-plan.v1",
        "profile": profile,
        "iteration_id": iteration_id,
        "control_types": len(specs),
        "sweep_points": point_count,
        "primary_qec_cycles": primary_cycles,
        "stability_qec_cycles": stability_cycles,
        "validation_qec_cycles": validation_cycles,
        "total_qec_cycles": total_cycles,
        "estimated_runtime": {
            "lower_seconds": max(1, int(total_cycles / 4_000_000)),
            "upper_seconds": max(5, int(total_cycles / 250_000)),
            "basis": "preregistered planning range; actual runtime is recorded",
        },
        "estimated_peak_memory_bytes": int(values["validation_shots_per_candidate"]) * 64,
        "estimated_storage_bytes": point_count * 4096 + len(specs) * 16384,
        "seeds": {
            "primary_perturbation": values["perturbation_seed"],
            "primary_detector": values["detector_seed"],
            "stability_offset": 500000,
        },
        "checkpoint_paths": checkpoints,
        "shard_count": shard_count,
        "start_commands": start_commands,
        "resumable_commands": shard_commands if shard_count > 1 else [command + " --resume"],
    }
    result["plan_hash"] = canonical_hash(result)
    return result


def _write_failed_iteration(output: Path, iteration_id: str, config_hash: str,
                            evaluator, protocol: SweepProtocol, error: FitRejected,
                            elapsed: float, seed_registry_hash: str) -> None:
    current_code_hash = code_hash()
    record = IterationRecord(
        iteration_id=iteration_id,
        source_commit=source_commit(current_code_hash),
        code_hash=current_code_hash,
        config_hash=config_hash,
        plant_hash=evaluator.plant_hash,
        protocol_hash=protocol.protocol_hash,
        analysis_hash=canonical_hash(error.diagnostics.values),
        seed_registry_hash=seed_registry_hash,
        changes_from_previous_iteration=("first recorded normalization iteration",),
        failed_gates=error.diagnostics.reasons,
        numerical_results={"fit_diagnostics": error.diagnostics.values, "runtime_seconds": elapsed},
        next_diagnosis=(
            "inspect monotonicity, local fit interval, and finite-shot coefficient stability",
            "amend only in a new iteration; preserve this failed record",
        ),
    )
    destination = output / "iterations" / iteration_id / "iteration_record.json"
    if destination.exists():
        raise FileExistsError("failed iteration record already exists and cannot be erased")
    atomic_json(destination, record.to_dict())


def _write_runtime_failure_iteration(output: Path, iteration_id: str, config: dict[str, Any],
                                     evaluator, protocol: SweepProtocol, error: Exception,
                                     started: float, stage: str) -> None:
    destination = output / "iterations" / iteration_id / "iteration_record.json"
    if destination.exists():
        return
    current_code_hash = code_hash()
    reason = f"{stage}: {type(error).__name__}: {error}"
    record = IterationRecord(
        iteration_id=iteration_id,
        source_commit=source_commit(current_code_hash),
        code_hash=current_code_hash,
        config_hash=canonical_hash(config),
        plant_hash=evaluator.plant_hash,
        protocol_hash=protocol.protocol_hash,
        analysis_hash=canonical_hash({"stage": stage, "error": str(error)}),
        seed_registry_hash=canonical_hash(config["seed_registry"]),
        changes_from_previous_iteration=_changes_from_previous(
            output, current_code_hash, canonical_hash(config)),
        failed_gates=(reason,),
        numerical_results={"runtime_seconds": time.perf_counter() - started, "stage": stage},
        next_diagnosis=("diagnose this failure before starting a longer run",),
    )
    atomic_json(destination, record.to_dict())


def _changes_from_previous(output: Path, current_code_hash: str,
                           config_hash: str) -> tuple[str, ...]:
    paths = sorted((output / "iterations").glob("*/iteration_record.json"))
    if not paths:
        return ("first recorded normalization iteration",)
    previous = json.loads(paths[-1].read_text(encoding="utf-8"))
    changes = []
    if previous.get("code_hash") != current_code_hash:
        changes.append("code_hash changed")
    if previous.get("config_hash") != config_hash:
        changes.append("config_hash changed")
    return tuple(changes or ["independent rerun with unchanged code and config"])


def _finalize(config: dict[str, Any], profile: str, output: Path, iteration_id: str,
              evaluator, sweeps: Sequence[SweepResult], started: float) -> dict[str, Any]:
    rules = FitRules.from_dict(config["fit_rules"])
    config_hash = canonical_hash(config)
    source_contract = build_source_contract()
    atomic_json(output / "source_contract.json", source_contract)
    try:
        fits = fit_all_sweeps(sweeps, rules)
        stability_protocol = build_protocol(config, profile, stability=True)
        stability_sweeps = tuple(
            run_control_type_sweep(evaluator, spec, stability_protocol)
            for spec in build_specs(config)
        )
        stability_fits = fit_all_sweeps(stability_sweeps, rules)
    except FitRejected as error:
        _write_failed_iteration(
            output, iteration_id, config_hash, evaluator, build_protocol(config, profile),
            error, time.perf_counter() - started,
            canonical_hash(config["seed_registry"]))
        raise
    bundle = build_calibration_bundle_from_sweeps(
        build_specs(config), sweeps, fits, rules,
        config_hash=config_hash,
        source_contract_hash=source_contract["source_contract_hash"],
        full_scale_completed=profile == "full",
        quantitative_match=False,
    )
    stability = validate_coefficient_remeasurement(fits, stability_fits, rules)
    values = config["profiles"][profile]
    validation = run_independent_validation(
        evaluator, bundle,
        candidates=int(values["validation_candidates"]),
        shots_per_candidate=int(values["validation_shots_per_candidate"]),
        normalized_isotropy_sigma=float(values["normalized_isotropy_sigma"]),
        normalized_joint_sigma=float(values["normalized_joint_sigma"]),
        seed=int(values["detector_seed"]) + 700_000,
    )
    gates = dict(validation["gates"])
    gates["coefficient_stability"] = bool(stability["passed"])
    gates["no_arbitrary_old_scale"] = bool(audit_no_arbitrary_scale(config)["passed"])
    failed_gates = tuple(name for name, passed in gates.items() if not passed)
    if failed_gates:
        bundle = replace(
            bundle,
            protocol_contract_pass=False,
            blocking_reasons=tuple(dict.fromkeys([
                *bundle.blocking_reasons,
                *(f"validation gate failed: {name}" for name in failed_gates),
            ])),
        )
    runtime = time.perf_counter() - started
    current_code_hash = code_hash()
    iteration = IterationRecord(
        iteration_id=iteration_id,
        source_commit=source_commit(current_code_hash),
        code_hash=current_code_hash,
        config_hash=config_hash,
        plant_hash=evaluator.plant_hash,
        protocol_hash=build_protocol(config, profile).protocol_hash,
        analysis_hash=canonical_hash({"validation": validation, "stability": stability}),
        seed_registry_hash=canonical_hash(config["seed_registry"]),
        changes_from_previous_iteration=_changes_from_previous(output, current_code_hash, config_hash),
        failed_gates=failed_gates,
        numerical_results={
            "runtime_seconds": runtime,
            "sigma0_native": {item.control_type: item.sigma0_native for item in fits},
            "validation": validation,
            "coefficient_stability": stability,
        },
        next_diagnosis=tuple(
            ["diagnose failed gates before any longer run"] if failed_gates else
            ["run the reduced profile before considering full source-scale acquisition"]),
    )
    iteration_path = output / "iterations" / iteration_id / "iteration_record.json"
    if iteration_path.exists():
        raise FileExistsError("iteration record already exists and cannot be overwritten")
    atomic_json(output / "calibration_bundle.json", bundle.to_dict())
    atomic_json(output / "validation.json", validation)
    atomic_json(output / "coefficient_stability.json", stability)
    atomic_json(iteration_path, iteration.to_dict())
    plan = resource_plan(config, profile, output, iteration_id, 1)
    final_status = {
        "schema_version": "google-pure-source-exact-normalization-final-status.v1",
        "iteration_id": iteration_id,
        "profile": profile,
        "artifact_complete": bundle.artifact_complete,
        "mathematical_contract_pass": bundle.mathematical_contract_pass,
        "protocol_contract_pass": bundle.protocol_contract_pass and all(gates.values()),
        "source_structure_match": bundle.source_structure_match,
        "quantitative_match": bundle.quantitative_match,
        "paper_comparable": bundle.paper_comparable,
        "blocking_reasons": list(bundle.blocking_reasons),
        "failed_gates": list(failed_gates),
        "runtime_seconds": runtime,
        "qec_cycles": plan["total_qec_cycles"],
        "estimated_peak_memory_bytes": plan["estimated_peak_memory_bytes"],
        "estimated_storage_bytes": plan["estimated_storage_bytes"],
        "seeds": plan["seeds"],
        "checkpoint_paths": plan["checkpoint_paths"],
        "start_commands": plan["start_commands"],
        "resumable_commands": plan["resumable_commands"],
        "normalization_method": bundle.normalization_method,
        "legacy_branch_label": config["legacy_algebraic_branch_label"],
        "stim_detector_events_used": True,
        "analytic_synthetic_gain_used_as_edr": False,
    }
    final_status["artifact_hash"] = canonical_hash(final_status)
    atomic_json(output / "final_status.json", final_status)
    return final_status


def command_source_contract(args: argparse.Namespace) -> int:
    output = Path(args.output_root)
    contract = build_source_contract()
    atomic_json(output / "source_contract.json", contract)
    print(json.dumps(contract, indent=2))
    return 0


def command_plan(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    plan = resource_plan(config, args.profile, Path(args.output_root), args.iteration_id, args.shard_count)
    print(json.dumps(plan, indent=2))
    return 0


def command_run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    config = load_config(Path(args.config))
    specs = build_specs(config)
    evaluator = build_evaluator(config, specs)
    protocol = build_protocol(config, args.profile)
    output = Path(args.output_root)
    sweeps = []
    try:
        for spec in specs:
            sigmas = shard_sigmas(spec.sweep_sigmas_native, args.shard_index, args.shard_count)
            checkpoint = _checkpoint_path(
                output, args.iteration_id, spec.control_type, args.shard_index, args.shard_count)
            sweeps.append(run_control_type_sweep(
                evaluator, spec, protocol, sigmas_native=sigmas,
                checkpoint_path=checkpoint, resume=args.resume,
                shard_index=args.shard_index, shard_count=args.shard_count))
    except Exception as error:
        _write_runtime_failure_iteration(
            output, args.iteration_id, config, evaluator, protocol, error, started, "primary_sweep")
        raise
    manifest = {
        "schema_version": "google-pure-source-exact-shard-manifest.v1",
        "iteration_id": args.iteration_id,
        "profile": args.profile,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "config_hash": canonical_hash(config),
        "plant_hash": evaluator.plant_hash,
        "protocol_hash": protocol.protocol_hash,
        "sweep_hashes": {item.control_type: canonical_hash(item.to_dict()) for item in sweeps},
        "runtime_seconds": time.perf_counter() - started,
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    atomic_json(output / "shards" / args.iteration_id / f"shard-{args.shard_index:03d}.json", manifest)
    if args.shard_count > 1:
        print(json.dumps(manifest, indent=2))
        return 0
    try:
        status = _finalize(config, args.profile, output, args.iteration_id, evaluator, sweeps, started)
    except Exception as error:
        _write_runtime_failure_iteration(
            output, args.iteration_id, config, evaluator, protocol, error, started, "finalize")
        raise
    print(json.dumps(status, indent=2))
    return 0 if not status["failed_gates"] else 2


def command_merge(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    config = load_config(Path(args.config))
    specs = build_specs(config)
    evaluator = build_evaluator(config, specs)
    output = Path(args.output_root)
    merged = []
    for spec in specs:
        shards = []
        for index in range(args.shard_count):
            path = _checkpoint_path(output, args.iteration_id, spec.control_type, index, args.shard_count)
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not payload.get("complete"):
                raise RuntimeError(f"incomplete shard checkpoint: {path}")
            shards.append(SweepResult.from_dict(payload["sweep_result"]))
        merged.append(merge_sweep_shards(shards, spec))
    status = _finalize(config, args.profile, output, args.iteration_id, evaluator, merged, started)
    print(json.dumps(status, indent=2))
    return 0 if not status["failed_gates"] else 2


def command_status(args: argparse.Namespace) -> int:
    path = Path(args.output_root) / "final_status.json"
    if not path.exists():
        print(json.dumps({"artifact_complete": False, "blocking_reasons": ["no final status artifact"]}, indent=2))
        return 1
    print(path.read_text(encoding="utf-8"), end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(func=None)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=str(default_config_path()))
    common.add_argument("--output-root", default=str(default_artifact_root()))
    subparsers = parser.add_subparsers(dest="command")
    source = subparsers.add_parser("source-contract", parents=[common])
    source.set_defaults(func=command_source_contract)
    plan = subparsers.add_parser("plan", parents=[common])
    plan.add_argument("--profile", choices=("smoke", "reduced", "full"), default="smoke")
    plan.add_argument("--iteration-id", required=True)
    plan.add_argument("--shard-count", type=int, default=1)
    plan.set_defaults(func=command_plan)
    run = subparsers.add_parser("run", parents=[common])
    run.add_argument("--profile", choices=("smoke", "reduced", "full"), default="smoke")
    run.add_argument("--iteration-id", required=True)
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--shard-count", type=int, default=1)
    run.add_argument("--resume", action="store_true")
    run.set_defaults(func=command_run)
    merge = subparsers.add_parser("merge", parents=[common])
    merge.add_argument("--profile", choices=("smoke", "reduced", "full"), required=True)
    merge.add_argument("--iteration-id", required=True)
    merge.add_argument("--shard-count", type=int, required=True)
    merge.set_defaults(func=command_merge)
    status = subparsers.add_parser("status", parents=[common])
    status.set_defaults(func=command_status)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.func is None:
        build_parser().print_help()
        return 2
    try:
        return int(args.func(args))
    except (FitRejected, ValueError, RuntimeError, FileNotFoundError, FileExistsError) as error:
        print(json.dumps({"error": type(error).__name__, "message": str(error)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
