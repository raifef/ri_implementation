"""CLI for source-exact Figure 5a planning, acquisition, merging, and reporting."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.contracts import PositivityGuard
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.optimizer import OptimizerConfig

from .acquisition import run_cell, substitution_identity
from .contracts import (AcquisitionMode, Figure5aProtocol, atomic_json, build_source_contract,
                        canonical_hash, file_sha256)
from .entropy_scan import build_conditions, classify_anchor_rows, scan_contract
from .normalization import build_empirical_normalization
from .validation import build_plant, dependency_hashes, physical_preflight


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = ROOT / "configs" / "google_pure_source_exact" / "figure5a.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "google_pure_source_exact" / "figure5a"
SOURCE_DIR = Path(__file__).resolve().parent


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _protocol(config: dict[str, Any], mode: AcquisitionMode) -> Figure5aProtocol:
    profile = config["profiles"][mode.value]
    return Figure5aProtocol(mode, int(profile["epochs"]), int(profile["candidates_per_epoch"]),
                            int(profile["qec_cycles_per_candidate"]),
                            int(config["plant"]["circuit_rounds"]))


def _optimizer(config: dict[str, Any]) -> OptimizerConfig:
    value = config["controller"]
    return OptimizerConfig(float(value["mean_learning_rate"]), float(value["sigma_learning_rate"]),
                           float(value["baseline_learning_rate"]),
                           minimum_sigma=float(value["minimum_sigma"]),
                           maximum_sigma=float(value["maximum_sigma"]),
                           positivity_guard=PositivityGuard(value["positivity_guard"]))


def _controller_hash(config: dict[str, Any]) -> str:
    return canonical_hash(config["controller"])


def _code_hash() -> str:
    return canonical_hash({path.name: file_sha256(path) for path in sorted(SOURCE_DIR.glob("*.py"))})


def plan(config_path: Path, output: Path, *, mode: AcquisitionMode, scan: str) -> dict[str, Any]:
    config = _load(config_path); protocol = _protocol(config, mode); plant = build_plant(config)
    contract = scan_contract(config, mode=mode, scan=scan, protocol=protocol,
                             plant_hash=plant.plant_hash, controller_hash=_controller_hash(config))
    conditions = contract["condition_count"]
    total_cycles = conditions * protocol.four_stream_qec_cycles
    candidate_cycles = conditions * protocol.candidate_qec_cycles
    shots = total_cycles // protocol.circuit_rounds
    estimate = shots / float(config["planning"]["estimated_stim_shots_per_second"])
    compilation_count = conditions * protocol.epochs * protocol.candidates_per_epoch * 4
    payload = {**contract, "created_at": datetime.now(timezone.utc).isoformat(),
               "config_hash": file_sha256(config_path), "control_count": plant.control_count,
               "candidate_training_qec_cycles": candidate_cycles,
               "four_stream_total_qec_cycles": total_cycles, "Stim_shots": shots,
               "circuit_compilations": compilation_count, "estimated_sampling_seconds_lower_bound": estimate,
               "estimated_peak_memory_bytes": protocol.candidates_per_epoch * 41 * 8 * 12,
               "estimated_storage_bytes": conditions * protocol.epochs * protocol.candidates_per_epoch * 28 * 6,
               "checkpoint_directory": str((output / "checkpoints" / mode.value / scan).resolve()),
               "reference_launch_requires_explicit_allow": mode == AcquisitionMode.REFERENCE,
               "certification_seeds_consumed_by_plan": False,
               "warning": "sampling estimate excludes candidate-specific Stim compilation and JSON checkpoint I/O"}
    atomic_json(output / "plans" / f"{mode.value}_{scan}.json", payload)
    return payload


def run_condition(config_path: Path, output: Path, *, mode: AcquisitionMode, scan: str,
                  condition_index: int, resume: bool, allow_reference: bool,
                  max_candidate_boundaries: int | None) -> dict[str, Any]:
    config = _load(config_path); protocol = _protocol(config, mode); plant = build_plant(config)
    conditions = build_conditions(config, mode=mode, scan=scan)
    if not 0 <= condition_index < len(conditions):
        raise ValueError("condition index is outside the frozen scan")
    if mode == AcquisitionMode.REFERENCE:
        if not allow_reference:
            raise RuntimeError("reference acquisition requires --allow-reference after reviewing the plan")
        preflight_path = output / "preflight.json"
        if not preflight_path.exists() or not _load(preflight_path)["pass"]:
            raise RuntimeError("reference acquisition blocked until physical preflight passes")
        protocol.assert_reference()
    condition = conditions[condition_index]
    contract = scan_contract(config, mode=mode, scan=scan, protocol=protocol,
                             plant_hash=plant.plant_hash, controller_hash=_controller_hash(config))
    cell_id = canonical_hash({"scan_hash": contract["scan_hash"], "condition": condition})[:20]
    checkpoint = output / "checkpoints" / mode.value / scan / f"{cell_id}.json"
    start = time.perf_counter()
    result = run_cell(protocol=protocol, plant=plant, frequency=condition["frequency"],
                      entropy_weight=condition["entropy_weight"], seed=condition["seed"],
                      optimizer_config=_optimizer(config), initial_sigma=float(config["controller"]["initial_sigma"]),
                      checkpoint_path=checkpoint, dependency_hashes=dependency_hashes(ROOT, config),
                      controller_hash=_controller_hash(config),
                      clip=float(config["controller"]["ppo_clip"]),
                      baseline_weight=float(config["controller"]["baseline_weight"]),
                      resume=resume, max_candidate_boundaries=max_candidate_boundaries,
                      source_budget_profile=mode.value)
    result.update({"scan_hash": contract["scan_hash"], "cell_id": cell_id,
                   "condition_index": condition_index, "mode": mode.value, "scan": scan,
                   "validation_watermark": mode != AcquisitionMode.REFERENCE,
                   "wall_seconds_this_call": time.perf_counter() - start})
    if result.get("complete"):
        if result["finite_shot_denominator_nonzero"]:
            substitution_identity(result["stream_totals"])
        atomic_json(output / "shards" / mode.value / scan / f"{cell_id}.json", result)
    return result


def merge(config_path: Path, output: Path, *, mode: AcquisitionMode, scan: str,
          iteration_id: str) -> dict[str, Any]:
    config = _load(config_path); protocol = _protocol(config, mode); plant = build_plant(config)
    contract = scan_contract(config, mode=mode, scan=scan, protocol=protocol,
                             plant_hash=plant.plant_hash, controller_hash=_controller_hash(config))
    rows = []
    for path in sorted((output / "shards" / mode.value / scan).glob("*.json")):
        row = _load(path)
        if row.get("scan_hash") == contract["scan_hash"]:
            rows.append(row)
    identities = [(row["frequency"], row["entropy_weight"], row["seed"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise RuntimeError("duplicate condition shard rejected")
    expected = {(row["frequency"], row["entropy_weight"], row["seed"])
                for row in contract["conditions"]}
    if set(identities) != expected:
        raise RuntimeError(f"missing or mixed-protocol shards: observed {len(rows)} of {len(expected)}")
    anchor_classification = classify_anchor_rows(rows, config["anchor"]["classification"]) if scan == "anchors" else None
    reference_exact = mode == AcquisitionMode.REFERENCE and all(
        row["candidate_qec_cycles"] == 1_800_000_000 and row["no_candidates_dropped"] for row in rows)
    quantitative = bool(anchor_classification and anchor_classification["anchor_classification_pass"])
    preflight_path = output / "preflight.json"
    preflight = _load(preflight_path) if preflight_path.exists() else {"pass": False}
    mathematical_contract = bool(preflight.get("pass"))
    blockers = []
    if not reference_exact: blockers.append("exact reference acquisition has not been completed")
    if scan == "anchors" and not quantitative: blockers.append("the three entropy regimes are not yet correctly ordered across seeds")
    if scan == "dense": blockers.append("anchor acceptance must precede dense-surface evidence")
    blockers.extend(["epsilon_tilde and Omega distributions are not publicly identifiable",
                     "the original proprietary simulation and optimizer hyperparameters are unavailable"])
    payload = {"schema_version": "figure5a-merge.v1", "iteration_id": iteration_id,
               "mode": mode.value, "scan": scan, "scan_hash": contract["scan_hash"],
               "protocol_hash": protocol.protocol_hash, "plant_hash": plant.plant_hash,
               "config_hash": file_sha256(config_path), "code_hash": _code_hash(), "rows": rows,
               "anchor_classification": anchor_classification,
               "artifact_complete": True, "mathematical_contract_pass": mathematical_contract,
               "protocol_contract_pass": reference_exact,
               "source_structure_match": mathematical_contract,
               "preflight_hash": canonical_hash(preflight),
               "quantitative_match": quantitative and reference_exact,
               "paper_comparable": False, "blocking_reasons": blockers,
               "candidate_qec_cycles": sum(row["candidate_qec_cycles"] for row in rows),
               "four_stream_qec_cycles": sum(row["four_stream_qec_cycles"] for row in rows)}
    payload["analysis_hash"] = canonical_hash(payload)
    atomic_json(output / "iterations" / iteration_id / "merged.json", payload)
    iteration = {"iteration_id": iteration_id, "source_commit": "WORKSPACE_WITHOUT_GIT_METADATA",
                 "code_hash": payload["code_hash"], "config_hash": payload["config_hash"],
                 "plant_hash": payload["plant_hash"], "protocol_hash": payload["protocol_hash"],
                 "analysis_hash": payload["analysis_hash"],
                 "seed_registry_hash": canonical_hash(config["seed_registry"]),
                 "changes_from_previous_iteration": ["implemented 41-control Stim plant, exact budgets, direct-sigma entropy anchors, and candidate-boundary resume"],
                 "failed_gates": blockers, "numerical_results": {"condition_count": len(rows),
                    "candidate_qec_cycles": payload["candidate_qec_cycles"], "quantitative_match": payload["quantitative_match"]},
                 "next_diagnosis": ["complete reference anchor cells before launching the dense phase surface"]}
    atomic_json(output / "iterations" / iteration_id / "iteration_record.json", iteration)
    atomic_json(output / "final_status.json", payload)
    write_report(payload, output)
    return payload


def write_report(payload: dict[str, Any], output: Path) -> None:
    records = [_load(path) for path in sorted((output / "iterations").glob("*/iteration_record.json"))]
    anchor_plan_path = output / "plans" / "reference_anchors.json"
    dense_plan_path = output / "plans" / "reference_dense.json"
    anchor_plan = _load(anchor_plan_path) if anchor_plan_path.exists() else None
    dense_plan = _load(dense_plan_path) if dense_plan_path.exists() else None
    lines = ["# Figure 5a source-exact implementation", "",
             "Implemented the public 41-parameter distance-3 Stim structure, exact budget gate, four raw-count streams, direct-sigma entropy anchors, deterministic candidate-boundary resumption, and condition sharding.", "",
             "## Evidence", "",
             f"- Artifact complete: `{payload['artifact_complete']}`",
             f"- Mathematical contract: `{payload['mathematical_contract_pass']}`",
             f"- Protocol contract: `{payload['protocol_contract_pass']}`",
             f"- Source structure: `{payload['source_structure_match']}`",
             f"- Quantitative match: `{payload['quantitative_match']}`",
             f"- Paper comparable: `{payload['paper_comparable']}`", "",
             "The current merged evidence is a reduced validation run, not source-scale evidence. It preserves all failed gates instead of treating a smoke or validation profile as equivalent to the paper.", "",
             "## Source contract", "",
             "- Distance-3 Stim plant with 17 one-qubit and 24 two-qubit controls (41 total).",
             "- Shared optimum `sin(2*pi*f*t)` and per-control quadratic error law.",
             "- Exact reference budget: 1,000 epochs, 50 candidates per epoch, and 36,000 QEC cycles per candidate (1.8 billion candidate-training QEC cycles per condition).",
             "- Entropy anchors: `0.001`, `0.01`, and `0.1`; slow-frequency anchor: `f=1/1000`.",
             "- The public article states that the source simulation code is proprietary. Its epsilon/Omega ensemble and optimizer hyperparameters are therefore recorded as non-identifiable, preregistered choices—not claimed source values.", "",
             "Sources: [Nature article](https://www.nature.com/articles/s41586-026-10759-2) and [official supplementary information](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-026-10759-2/MediaObjects/41586_2026_10759_MOESM1_ESM.pdf).", "",
             "## Iterations", ""]
    lines.extend(f"- `{record['iteration_id']}`: failed gates `{record['failed_gates']}`; results `{record['numerical_results']}`."
                 for record in records)
    if anchor_plan and dense_plan:
        lines.extend(["", "## Preregistered reference cost", "",
                      f"- Anchor scan: `{anchor_plan['condition_count']}` conditions, `{anchor_plan['four_stream_total_qec_cycles']}` four-stream QEC cycles, and a sampling-only lower bound of `{anchor_plan['estimated_sampling_seconds_lower_bound']}` seconds.",
                      f"- Dense scan: `{dense_plan['condition_count']}` conditions, `{dense_plan['four_stream_total_qec_cycles']}` four-stream QEC cycles, and a sampling-only lower bound of `{dense_plan['estimated_sampling_seconds_lower_bound']}` seconds.",
                      "- These lower bounds exclude candidate-specific circuit compilation and checkpoint I/O; inspect the JSON plans before launch."])
    lines.extend(["", "## Remaining source-scale acquisition", "",
                  "No reference acquisition was launched automatically. The explicit launch is resumable and fail-closed:", "", "```powershell",
                  "$env:PYTHONPATH='src'",
                  "python -m hdfa_rl_suite.google_pure_source_exact.figure5a.cli plan --mode reference --scan anchors",
                  "0..8 | ForEach-Object { python -m hdfa_rl_suite.google_pure_source_exact.figure5a.cli run --mode reference --scan anchors --condition-index $_ --allow-reference --resume }",
                  "python -m hdfa_rl_suite.google_pure_source_exact.figure5a.cli merge --mode reference --scan anchors --iteration-id figure5a-reference-anchors-001",
                  "# Launch the dense reference scan only if the anchor acceptance gate passes.",
                  "```", ""])
    (output / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("source-contract"); sub.add_parser("calibrate-normalization")
    sub.add_parser("preflight")
    for name in ("plan", "run", "merge"):
        item = sub.add_parser(name); item.add_argument("--mode", choices=[value.value for value in AcquisitionMode], required=True); item.add_argument("--scan", choices=("anchors", "dense"), required=True)
        if name == "run": item.add_argument("--condition-index", type=int, required=True); item.add_argument("--resume", action="store_true"); item.add_argument("--allow-reference", action="store_true"); item.add_argument("--max-candidate-boundaries", type=int)
        if name == "merge": item.add_argument("--iteration-id", required=True)
    args = parser.parse_args(argv); config = _load(args.config)
    if args.command == "source-contract": result = build_source_contract(); atomic_json(args.output / "source_contract.json", result)
    elif args.command == "calibrate-normalization":
        plant = build_plant(config); result = build_empirical_normalization(plant)
        atomic_json(ROOT / config["ablations"]["empirical_relative_normalization_bundle"], result)
    elif args.command == "preflight": result = physical_preflight(ROOT, config); atomic_json(args.output / "preflight.json", result); plant = build_plant(config); atomic_json(args.output / "parameter_inventory.json", {"plant_hash": plant.plant_hash, "rows": plant.inventory_rows()}); atomic_json(args.output / "detector_mask.json", {"plant_hash": plant.plant_hash, "mask": plant.mask.astype(int).tolist()})
    else:
        mode = AcquisitionMode(args.mode)
        if args.command == "plan": result = plan(args.config, args.output, mode=mode, scan=args.scan)
        elif args.command == "run": result = run_condition(args.config, args.output, mode=mode, scan=args.scan, condition_index=args.condition_index, resume=args.resume, allow_reference=args.allow_reference, max_candidate_boundaries=args.max_candidate_boundaries)
        else: result = merge(args.config, args.output, mode=mode, scan=args.scan, iteration_id=args.iteration_id)
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
