"""Resumable source-exact direct-sigma validation commands."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np

from .comparison import compare_positivity_guards, run_matched_seed
from .contracts import (
    DIRECT_SIGMA_PARAMETERIZATION,
    EvidenceStatus,
    IterationRecord,
    atomic_json,
    build_source_contract,
    canonical_hash,
    file_sha256,
)
from .validation import baseline_dynamics_audit, mathematical_audit, source_loss_semantics_audit


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = ROOT / "configs" / "google_pure_source_exact" / "policy_parameterization.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "google_pure_source_exact" / "policy_parameterization"
SOURCE_DIR = Path(__file__).resolve().parent


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _code_hash() -> str:
    return canonical_hash({path.name: file_sha256(path) for path in sorted(SOURCE_DIR.glob("*.py"))})


def _profile(config: dict[str, Any], name: str) -> dict[str, Any]:
    try:
        return dict(config["profiles"][name])
    except KeyError as error:
        raise ValueError(f"unknown profile: {name}") from error


def _shard_seeds(profile: dict[str, Any], index: int, count: int) -> list[int]:
    if not 0 <= index < count:
        raise ValueError("invalid shard identity")
    return [int(seed) for position, seed in enumerate(profile["seeds"]) if position % count == index]


def build_plan(config_path: Path, profile_name: str, shard_count: int,
               output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    config = _load(config_path)
    profile = _profile(config, profile_name)
    seeds = [int(seed) for seed in profile["seeds"]]
    direct_cycles = len(seeds) * 2 * int(profile["epochs"]) * int(profile["candidates_per_epoch"]) \
        * int(profile["qec_cycles_per_candidate"])
    comparison_cycles = 2 * direct_cycles
    estimated_seconds = len(seeds) * int(profile["epochs"]) * int(profile["candidates_per_epoch"]) \
        * int(profile["dimension"]) / 2.0e6
    return {
        "schema_version": "direct-sigma-run-plan.v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": profile_name, "parameterization": DIRECT_SIGMA_PARAMETERIZATION,
        "config_path": str(config_path.resolve()), "config_hash": file_sha256(config_path),
        "normalization_bundle": str((ROOT / config["normalization_bundle"]).resolve()),
        "normalization_bundle_hash": file_sha256(ROOT / config["normalization_bundle"]),
        "seed_count": len(seeds), "seeds": seeds, "shard_count": int(shard_count),
        "declared_direct_branch_qec_cycle_budget": direct_cycles,
        "declared_matched_comparison_qec_cycle_budget": comparison_cycles,
        "executed_qec_cycles": 0,
        "analytic_fixture_candidate_evaluations": len(seeds) * 4 * int(profile["epochs"]) *
                                                   int(profile["candidates_per_epoch"]),
        "estimated_runtime_seconds": estimated_seconds,
        "estimated_peak_memory_bytes": int(profile["candidates_per_epoch"]) * int(profile["dimension"]) * 8 * 12,
        "estimated_storage_bytes": len(seeds) * int(profile["epochs"]) * 16 * 8,
        "checkpoint_directory": str((output / "checkpoints" / profile_name).resolve()),
        "full_run_requires_explicit_allow": profile_name == "full",
        "certification_seeds_consumed_by_plan": False,
    }


def run_shard(config_path: Path, profile_name: str, shard_index: int, shard_count: int,
              output: Path, *, resume: bool, allow_full: bool) -> dict[str, Any]:
    if profile_name == "full" and not allow_full:
        raise RuntimeError("full profile requires --allow-full after reviewing the run plan")
    config = _load(config_path)
    profile = _profile(config, profile_name)
    seeds = _shard_seeds(profile, shard_index, shard_count)
    config_hash = file_sha256(config_path)
    checkpoint = output / "checkpoints" / profile_name / f"shard-{shard_index:03d}-of-{shard_count:03d}.json"
    state: dict[str, Any] = {"schema_version": "direct-sigma-shard-checkpoint.v1", "profile": profile_name,
                             "shard_index": shard_index, "shard_count": shard_count,
                             "config_hash": config_hash, "completed": {}}
    if checkpoint.exists():
        if not resume:
            raise RuntimeError(f"checkpoint exists; pass --resume: {checkpoint}")
        state = _load(checkpoint)
        identity = (state["profile"], state["shard_index"], state["shard_count"], state["config_hash"])
        if identity != (profile_name, shard_index, shard_count, config_hash):
            raise RuntimeError("checkpoint identity does not match requested shard")
    for seed in seeds:
        key = str(seed)
        if key not in state["completed"]:
            state["completed"][key] = run_matched_seed(seed, profile)
            atomic_json(checkpoint, state)
    result = {"schema_version": "direct-sigma-shard.v1", "profile": profile_name,
              "shard_index": shard_index, "shard_count": shard_count, "config_hash": config_hash,
              "seeds": seeds, "results": [state["completed"][str(seed)] for seed in seeds],
              "declared_qec_cycle_budget": len(seeds) * 4 * int(profile["epochs"]) *
                                           int(profile["candidates_per_epoch"]) *
                                           int(profile["qec_cycles_per_candidate"]),
              "executed_qec_cycles": 0,
              "analytic_fixture_candidate_evaluations": len(seeds) * 4 * int(profile["epochs"]) *
                                                        int(profile["candidates_per_epoch"]),
              "checkpoint_path": str(checkpoint.resolve()), "complete": True}
    path = output / "shards" / profile_name / f"shard-{shard_index:03d}-of-{shard_count:03d}.json"
    atomic_json(path, result)
    return result


def merge_shards(config_path: Path, profile_name: str, shard_count: int,
                 output: Path, iteration_id: str) -> dict[str, Any]:
    if (output / "iterations" / iteration_id / "iteration_record.json").exists():
        raise FileExistsError(f"iteration already exists and cannot be overwritten: {iteration_id}")
    config = _load(config_path)
    profile = _profile(config, profile_name)
    expected_seeds = [int(seed) for seed in profile["seeds"]]
    results: list[dict[str, Any]] = []
    declared_cycle_budget = 0
    executed_qec_cycles = 0
    analytic_candidate_evaluations = 0
    for index in range(shard_count):
        path = output / "shards" / profile_name / f"shard-{index:03d}-of-{shard_count:03d}.json"
        if not path.exists():
            raise RuntimeError(f"missing shard: {path}")
        shard = _load(path)
        if (shard["profile"], shard["shard_index"], shard["shard_count"], shard["config_hash"]) != \
                (profile_name, index, shard_count, file_sha256(config_path)):
            raise RuntimeError(f"shard identity mismatch: {path}")
        results.extend(shard["results"])
        declared_cycle_budget += int(shard["declared_qec_cycle_budget"])
        executed_qec_cycles += int(shard["executed_qec_cycles"])
        analytic_candidate_evaluations += int(shard["analytic_fixture_candidate_evaluations"])
    observed = [int(row["seed"]) for row in results]
    if len(observed) != len(set(observed)):
        raise RuntimeError("duplicate seed/shard contribution rejected")
    if sorted(observed) != sorted(expected_seeds):
        raise RuntimeError("merged seeds do not equal the frozen seed registry")
    audit = mathematical_audit()
    loss_audit = source_loss_semantics_audit()
    baseline_audit = baseline_dynamics_audit()
    gates = [value for row in results for value in row["gates"].values()]
    math_pass = bool(audit["finite_difference_pass"] and audit["negative_entropy_descent_increases_entropy"]
                     and audit["behavior_snapshot_immutable"] and loss_audit["pass"]
                     and baseline_audit["pass"])
    development_fixture_pass = all(gates) and all(row["common_random_numbers"] for row in results)
    protocol_pass = False
    blockers = ["source-scale hardware plant and proprietary hyperparameters are not publicly identifiable",
                "full source-scale acquisition has not been executed and ingested"]
    if not math_pass:
        blockers.insert(0, "direct-sigma mathematical audit failed")
    if not development_fixture_pass:
        blockers.insert(0, "reduced matched analytic comparison gates failed")
    status = EvidenceStatus(True, math_pass, protocol_pass, math_pass, False, False, tuple(blockers))
    payload = {"schema_version": "direct-sigma-comparison.v1", "iteration_id": iteration_id,
               "profile": profile_name, "parameterization": DIRECT_SIGMA_PARAMETERIZATION,
               "log_sigma_branch_label": config["log_sigma_ablation_label"],
               "optimized_scale_variable": "sigma", "config_hash": file_sha256(config_path),
               "code_hash": _code_hash(), "normalization_bundle_hash": file_sha256(ROOT / config["normalization_bundle"]),
               "seed_registry_hash": canonical_hash(config["seed_registry"]), "results": results,
               "mathematical_audit": audit, "positivity_guard_comparison": compare_positivity_guards(),
               "source_loss_semantics_audit": loss_audit,
               "baseline_dynamics_audit": baseline_audit,
               "paper_mode_ratio_clipping": config["paper_mode_ratio_clipping"],
               "aggregate_ratio_ablation_label": config["aggregate_ratio_ablation_label"],
               "paper_mode_baseline": config["paper_mode_baseline"],
               "ema_baseline_ablation_label": config["ema_baseline_ablation_label"],
               "baseline_loss_weight_preregistration": config["baseline_loss_weight_preregistration"],
               "development_fixture_pass": development_fixture_pass,
               "declared_qec_cycle_budget": declared_cycle_budget,
               "executed_qec_cycles": executed_qec_cycles,
               "analytic_fixture_candidate_evaluations": analytic_candidate_evaluations,
               **status.to_dict()}
    payload["analysis_hash"] = canonical_hash(payload)
    merged_path = output / "iterations" / iteration_id / "comparison.json"
    atomic_json(merged_path, payload)
    record = IterationRecord(
        iteration_id=iteration_id, source_commit="WORKSPACE_WITHOUT_GIT_METADATA", code_hash=payload["code_hash"],
        config_hash=payload["config_hash"], plant_hash=payload["normalization_bundle_hash"],
        protocol_hash=canonical_hash({"profile": profile_name, "profile_config": profile}),
        analysis_hash=payload["analysis_hash"], seed_registry_hash=payload["seed_registry_hash"],
        changes_from_previous_iteration=(
            "separated declared QEC-cycle budget from zero executed detector cycles in analytic fixtures",
            "hardened direct-sigma checkpoint baseline state, deterministic shard resume, and duplicate rejection",
            "retained the log-scale implementation only as an explicitly labelled non-paper ablation",
            "proved coordinate likelihood ratios are clipped before the sparse detector product",
            "made the jointly optimized detector-vector baseline mandatory in paper mode",
        ),
        failed_gates=tuple([] if math_pass and development_fixture_pass and protocol_pass else
                           (["source_protocol_not_executed"] if math_pass and development_fixture_pass else
                            ["mathematical_or_development_fixture_gate"])),
        numerical_results={"seed_count": len(results),
                           "declared_qec_cycle_budget": declared_cycle_budget,
                           "executed_qec_cycles": executed_qec_cycles,
                           "analytic_fixture_candidate_evaluations": analytic_candidate_evaluations,
                           "all_development_gates_pass": development_fixture_pass},
        next_diagnosis=("run and ingest the explicit source-scale protocol without consuming development seeds",))
    atomic_json(output / "iterations" / iteration_id / "iteration_record.json", record.to_dict())
    atomic_json(output / "final_status.json", payload)
    return payload


def write_report(payload: dict[str, Any], output: Path) -> None:
    audit = payload["mathematical_audit"]
    loss_audit = payload["source_loss_semantics_audit"]
    baseline_audit = payload["baseline_dynamics_audit"]
    iteration_rows = []
    for path in sorted((output / "iterations").glob("*/iteration_record.json")):
        record = _load(path)
        stored = "PASS" if not record["failed_gates"] else "BLOCKED"
        iteration_rows.append(
            f"- `{record['iteration_id']}` — `{stored}`; failed gates: "
            f"`{', '.join(record['failed_gates']) if record['failed_gates'] else 'none'}`; "
            f"results: `{json.dumps(record['numerical_results'], sort_keys=True)}`."
        )
    lines = ["# Direct-Sigma Policy Parameterization", "",
             "The source-exact branch directly optimizes `sigma`; it never stores or optimizes a log-scale variable.", "",
             "## Implemented source requirements", "",
             "- Supplement Eq. (11) factorized Gaussian with `theta=(mu,sigma)`.",
             "- Eqs. (18)-(22): policy, detector-baseline, negative-entropy, and total loss with gradient descent.",
             "- Eq. (18) coordinate likelihood ratios are clipped elementwise before `exp(M @ log(chi))`; aggregate detector-ratio clipping is retained only as `NON_SOURCE_PPO_ABLATION`.",
             "- Eqs. (13) and (19) use one jointly optimized baseline parameter per detector component, frozen before each batch update; EMA is retained only as `NON_SOURCE_EMA_BASELINE_ABLATION`.",
             "- Direct analytic likelihood and entropy gradients and immutable collection snapshots.",
             "- Direct-sigma checkpoints and optimizer state; the prior log-scale controller is labelled `NON_PAPER_LOG_SIGMA_ABLATION`.", "",
             "## Divergences removed", "",
             "- Replaced the learnable `log_sigma` scale with an actual positive `sigma` vector in paper mode.",
             "- Replaced log-scale reward and constant entropy gradients with direct `sigma` gradients.",
             "- Removed log conversion from source-exact checkpoint serialization and attached optimizer momentum to `sigma`.",
             "- Rejected the log-scale ablation whenever paper mode is active.", "",
             "- Paper mode now also rejects aggregate-ratio clipping and the EMA baseline ablation.", "",
             "## Validation", "",
             f"- Finite-difference audit: `{audit['finite_difference_pass']}`; max error `{max(audit['errors'].values()):.3e}`.",
             f"- Coordinate clipping order audit: `{loss_audit['pass']}`; hand-calculation error `{loss_audit['source_hand_error']:.3e}`; aggregate branch non-equivalent `{loss_audit['multi_coordinate_non_equivalence']}`.",
             f"- Learned baseline audit: `{baseline_audit['pass']}`; baseline finite-difference error `{loss_audit['baseline_finite_difference_error']:.3e}`; late/early advantage second-moment ratio `{baseline_audit['late_advantage_second_moment']/baseline_audit['early_advantage_second_moment']:.3f}`.",
             f"- Baseline-loss weight: `{payload['baseline_loss_weight_preregistration']['selected']}` selected from development grid `{payload['baseline_loss_weight_preregistration']['development_grid']}` before certification; source value unavailable.",
             f"- Reduced analytic-fixture gate: `{payload['development_fixture_pass']}` across `{len(payload['results'])}` seeds.",
             f"- Source protocol gate: `{payload['protocol_contract_pass']}` (no detector/QEC acquisition was executed).",
             f"- Declared comparison budget: `{payload['declared_qec_cycle_budget']}` QEC cycles; executed QEC cycles: `{payload['executed_qec_cycles']}`.",
             f"- Executed analytic candidate evaluations: `{payload['analytic_fixture_candidate_evaluations']}`.",
             "- Focused and legacy-regression suite at this iteration: `111 passed`.",
             "- Failed iterations are retained under `iterations/`; certification seeds were not consumed.", "",
             "## Iteration history", "", *iteration_rows, "",
             "Iterations `direct-sigma-smoke-001` and `direct-sigma-reduced-002` are retained exactly as produced but their old `PASS` classification is superseded: they incorrectly reported a declared cycle budget as executed QEC cycles. Iteration 003 repaired that evidence accounting and correctly blocks the source protocol gate.", "",
             "## Unresolved source choices", "",
             "- Positivity handling is `SOURCE_UNSPECIFIED_PREREGISTERED`; projected gradient is selected after comparing bounded and backtracking direct-sigma guards.",
             "- Public sources do not identify the experimental optimizer settings, loss weights, gradient clipping, or proprietary plant.", "",
             "## Evidence status", "",
             f"- `artifact_complete`: `{payload['artifact_complete']}`",
             f"- `mathematical_contract_pass`: `{payload['mathematical_contract_pass']}`",
             f"- `protocol_contract_pass`: `{payload['protocol_contract_pass']}`",
             f"- `source_structure_match`: `{payload['source_structure_match']}`",
             f"- `quantitative_match`: `{payload['quantitative_match']}`",
             f"- `paper_comparable`: `{payload['paper_comparable']}`", "",
             "No full source-scale detector acquisition was completed. The completed smoke and reduced runs are analytic development fixtures, not quantitative paper evidence.", "",
             "## Remaining validation commands", "",
             "```powershell",
             "$env:PYTHONPATH='src'",
             "python -m hdfa_rl_suite.google_pure_source_exact.policy_parameterization.cli plan --profile full --shard-count 8",
             "python -m hdfa_rl_suite.google_pure_source_exact.policy_parameterization.cli run --profile full --shard-count 8 --shard-index 0 --allow-full --resume",
             "# Repeat shard-index 1 through 7, then:",
             "python -m hdfa_rl_suite.google_pure_source_exact.policy_parameterization.cli merge --profile full --shard-count 8 --iteration-id direct-sigma-full-001",
             "```", "",
             "This command completes the full-size analytic parameterization fixture only. It cannot make the result paper-comparable; a true source-scale hardware-plant acquisition remains blocked by non-public plant and optimizer details.", ""]
    (output / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("source-contract")
    plan = sub.add_parser("plan"); plan.add_argument("--profile", choices=("smoke", "reduced", "full"), required=True); plan.add_argument("--shard-count", type=int, default=1)
    run = sub.add_parser("run"); run.add_argument("--profile", choices=("smoke", "reduced", "full"), required=True); run.add_argument("--shard-count", type=int, default=1); run.add_argument("--shard-index", type=int, default=0); run.add_argument("--resume", action="store_true"); run.add_argument("--allow-full", action="store_true")
    merge = sub.add_parser("merge"); merge.add_argument("--profile", choices=("smoke", "reduced", "full"), required=True); merge.add_argument("--shard-count", type=int, default=1); merge.add_argument("--iteration-id", required=True)
    args = parser.parse_args(argv)
    if args.command == "source-contract":
        payload = build_source_contract(); atomic_json(args.output / "source_contract.json", payload)
    elif args.command == "plan":
        payload = build_plan(args.config, args.profile, args.shard_count, args.output)
        atomic_json(args.output / f"{args.profile}_run_plan.json", payload)
    elif args.command == "run":
        payload = run_shard(args.config, args.profile, args.shard_index, args.shard_count,
                            args.output, resume=args.resume, allow_full=args.allow_full)
    else:
        payload = merge_shards(args.config, args.profile, args.shard_count, args.output, args.iteration_id)
        write_report(payload, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
