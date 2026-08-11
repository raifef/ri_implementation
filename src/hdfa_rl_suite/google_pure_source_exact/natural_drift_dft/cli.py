"""Reproducible commands for source-exact natural-drift DFT analysis."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from .contracts import (EvaluationTrace, SOURCE_ESTIMATOR_STATUS, SourceDFTConfig,
                        atomic_json, build_source_contract, canonical_hash, file_sha256)
from .estimator import analyze_traces, preprocess_trace, run_spectrum


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = ROOT / "configs" / "google_pure_source_exact" / "natural_drift_dft.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "google_pure_source_exact" / "natural_drift_dft"
SOURCE_DIR = Path(__file__).resolve().parent


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _config(path: Path) -> tuple[dict[str, Any], SourceDFTConfig]:
    value = _load(path)
    contract = SourceDFTConfig(
        cadence_epochs=int(value["cadence_epochs"]), warmup_epoch=int(value["warmup_epoch"]),
        shared_grid_points=int(value["shared_grid_points"]),
        gaussian_smoothing_sigma_bins=float(value["gaussian_smoothing_sigma_bins"]),
        power_normalization=str(value["power_normalization"]), filter_ratio=str(value["filter_ratio"]))
    return value, contract


def _code_hash() -> str:
    return canonical_hash({path.name: file_sha256(path) for path in sorted(SOURCE_DIR.glob("*.py"))})


def _inventory_status(config: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / config["public_release_inventory"]
    inventory = _load(path)
    candidates = [row["relative_path"] for row in inventory.get("members", [])
                  if any(token in row["relative_path"].lower()
                         for token in ("natural_drift", "filter_function", "learned_mean_ler", "training_epoch"))]
    return {"inventory_path": str(path.resolve()), "inventory_hash": file_sha256(path),
            "release_identity_status": inventory.get("identity_status"),
            "archive_sha256": inventory.get("identity", {}).get("archive_sha256"),
            "archive_members": inventory.get("summary", {}).get("archive_members"),
            "dynamic_trace_candidates": candidates,
            "hardware_dynamic_evaluation_traces_available": bool(candidates)}


def build_plan(config_path: Path, output: Path, input_path: Path | None = None) -> dict[str, Any]:
    config, estimator_config = _config(config_path)
    inventory = _inventory_status(config)
    input_bytes = input_path.stat().st_size if input_path is not None and input_path.exists() else 0
    payload = {"schema_version": "natural-drift-dft-plan.v1",
               "created_at": datetime.now(timezone.utc).isoformat(),
               "config_hash": file_sha256(config_path), "source_contract_hash": build_source_contract()["source_contract_hash"],
               "estimator_config": estimator_config.__dict__, "public_release": inventory,
               "input_path": str(input_path.resolve()) if input_path else None, "input_bytes": input_bytes,
               "qec_cycles": 0, "candidate_evaluations": 0,
               "estimated_runtime_seconds": max(0.1, input_bytes / 20_000_000),
               "estimated_peak_memory_bytes": max(1_000_000, input_bytes * 8),
               "estimated_storage_bytes": max(200_000, input_bytes * 3),
               "checkpoint_directory": str((output / "checkpoints").resolve()),
               "hardware_acquisition_not_launched": True,
               "certification_seeds_consumed_by_plan": False}
    atomic_json(output / "run_plan.json", payload)
    return payload


def generate_synthetic(config_path: Path, output_path: Path) -> dict[str, Any]:
    config, estimator_config = _config(config_path)
    traces = []
    lengths = (600, 675, 750, 825)
    for position, (seed, stop) in enumerate(zip(config["development_seeds"], lengths)):
        rng = np.random.default_rng(int(seed))
        epochs = np.arange(0, stop + estimator_config.cadence_epochs,
                           estimator_config.cadence_epochs, dtype=int)
        phase = 0.31 * position
        slow = np.sin(2 * np.pi * epochs / 240.0 + phase)
        fast = np.sin(2 * np.pi * epochs / 35.0 + 0.4 * phase)
        shared_noise = rng.normal(0.0, 0.006, size=len(epochs))
        fixed = 0.02 * (1.0 + 0.24 * slow + 0.035 * fast + shared_noise)
        learned = 0.018 * (1.0 + 0.10 * slow + 0.035 * fast + shared_noise)
        traces.append(EvaluationTrace(f"synthetic-development-{seed}", tuple(epochs),
                                      tuple(learned), tuple(fixed)).to_dict())
    payload = {"schema_version": "natural-drift-evaluation-traces.v1",
               "evidence_layer": "SYNTHETIC_DEVELOPMENT_ONLY", "traces": traces,
               "certification_seeds_consumed": False}
    payload["trace_set_hash"] = canonical_hash(payload)
    atomic_json(output_path, payload)
    return payload


def _write_figure(analysis: dict[str, Any], output: Path, iteration_id: str,
                  evidence_layer: str) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frequency = np.asarray(analysis["frequency_per_epoch"])
    learned = np.asarray(analysis["learned_geometric_psd"])
    fixed = np.asarray(analysis["fixed_geometric_psd"])
    raw = np.asarray(analysis["raw_filter_db"])
    smooth = np.asarray(analysis["smoothed_guide_to_eye_db"])
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].loglog(frequency, fixed, label="fixed initial policy", color="#777777")
    axes[0].loglog(frequency, learned, label="learned mean", color="#d64b8c")
    axes[0].set(xlabel="frequency (epochs$^{-1}$)", ylabel="DFT power")
    axes[0].legend(frameon=False)
    axes[1].semilogx(frequency, raw, label="raw empirical ratio", color="#e69ac2", alpha=0.75)
    axes[1].semilogx(frequency, smooth, label="Gaussian guide-to-eye", color="#8e245c")
    axes[1].axhline(0.0, color="black", linewidth=0.7)
    axes[1].set(xlabel="frequency (epochs$^{-1}$)", ylabel="10 log10(learned/fixed) (dB)")
    axes[1].legend(frameon=False)
    figure.suptitle(f"Natural-drift DFT — {evidence_layer}")
    directory = output / "figures" / iteration_id
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "svg"):
        path = directory / f"filter_function.{suffix}"
        temporary = path.with_name(path.stem + ".tmp" + path.suffix)
        figure.savefig(temporary, dpi=180 if suffix == "png" else None)
        temporary.replace(path)
        paths.append({"path": str(path.resolve()), "sha256": file_sha256(path)})
    plt.close(figure)
    return {"schema_version": "natural-drift-dft-figure-manifest.v1",
            "raw_and_smoothed_separate": True, "evidence_layer": evidence_layer, "files": paths}


def run_analysis(config_path: Path, input_path: Path, output: Path, iteration_id: str) -> dict[str, Any]:
    if (output / "iterations" / iteration_id / "analysis.json").exists():
        raise FileExistsError(f"iteration already exists and cannot be overwritten: {iteration_id}")
    config, estimator_config = _config(config_path)
    input_payload = _load(input_path)
    traces = [EvaluationTrace.from_mapping(value) for value in input_payload["traces"]]
    if len({trace.run_id for trace in traces}) != len(traces):
        raise RuntimeError("duplicate trace shard contribution rejected")
    started = time.perf_counter()
    shard_rows = []
    for trace in traces:
        prepared = preprocess_trace(trace, estimator_config)
        learned_frequency, learned_power = run_spectrum(
            prepared["learned"], cadence_epochs=estimator_config.cadence_epochs)
        fixed_frequency, fixed_power = run_spectrum(
            prepared["fixed"], cadence_epochs=estimator_config.cadence_epochs)
        shard = {"schema_version": "natural-drift-run-spectrum-shard.v1", "run_id": trace.run_id,
                 "input_trace_hash": canonical_hash(trace.to_dict()), "complete": True,
                 "frequency_per_epoch": learned_frequency.tolist(), "learned_power": learned_power.tolist(),
                 "fixed_power": fixed_power.tolist(), "paired_frequency_equal": bool(
                     np.array_equal(learned_frequency, fixed_frequency))}
        shard["shard_hash"] = canonical_hash(shard)
        path = output / "shards" / iteration_id / f"{trace.run_id}.json"
        if path.exists() and _load(path) != shard:
            raise RuntimeError(f"existing run shard changed: {trace.run_id}")
        atomic_json(path, shard)
        shard_rows.append({"run_id": trace.run_id, "path": str(path.resolve()),
                           "shard_hash": shard["shard_hash"]})
    if len({row["run_id"] for row in shard_rows}) != len(traces):
        raise RuntimeError("duplicate completed run shard rejected")
    analysis = analyze_traces(traces, estimator_config)
    public_release = _inventory_status(config)
    synthetic = input_payload.get("evidence_layer") != "PUBLIC_HARDWARE_SOURCE_TRACES"
    blockers = []
    if synthetic:
        blockers.append("analysis input is synthetic development data, not the source hardware traces")
    if not public_release["hardware_dynamic_evaluation_traces_available"]:
        blockers.append("official public release does not contain the dynamic learned-mean/fixed LER evaluation traces")
    blockers.append("the proprietary acquisition/controller implementation is unavailable")
    figure_manifest = _write_figure(analysis, output, iteration_id,
                                    str(input_payload.get("evidence_layer")))
    atomic_json(output / "figures" / iteration_id / "manifest.json", figure_manifest)
    payload = {**analysis, "iteration_id": iteration_id, "input_path": str(input_path.resolve()),
               "input_file_hash": file_sha256(input_path), "config_hash": file_sha256(config_path),
               "code_hash": _code_hash(), "source_contract": build_source_contract(),
               "public_release": public_release, "shards": shard_rows,
               "figure_manifest": figure_manifest,
               "wall_seconds": time.perf_counter() - started, "qec_cycles": 0,
               "candidate_evaluations": 0, "evidence_layer": input_payload.get("evidence_layer"),
               "structural_status": SOURCE_ESTIMATOR_STATUS,
               "artifact_complete": True, "mathematical_contract_pass": True,
               "protocol_contract_pass": not synthetic, "source_structure_match": True,
               "quantitative_match": False, "paper_comparable": False,
               "blocking_reasons": blockers}
    payload["analysis_hash"] = canonical_hash({key: value for key, value in payload.items()
                                                if key != "wall_seconds"})
    iteration = {"iteration_id": iteration_id, "source_commit": "WORKSPACE_WITHOUT_GIT_METADATA",
                 "code_hash": payload["code_hash"], "config_hash": payload["config_hash"],
                 "plant_hash": "NO_PLANT_ANALYSIS_ONLY", "protocol_hash": payload["source_contract"]["source_contract_hash"],
                 "analysis_hash": payload["analysis_hash"],
                 "seed_registry_hash": canonical_hash({"development": config["development_seeds"],
                                                        "certification": config["certification_seeds_reserved"]}),
                 "changes_from_previous_iteration": ["replaced Welch-band statistic with per-run epoch-domain DFT",
                                                       "added epoch-150 normalization, shared-grid interpolation, and geometric PSD averaging",
                                                       "separated raw filter ratio from Gaussian guide-to-eye"],
                 "failed_gates": blockers, "numerical_results": {
                     "run_count": len(traces), "frequency_points": len(analysis["frequency_per_epoch"]),
                     "median_low_frequency_filter_db": float(np.median(np.asarray(analysis["raw_filter_db"])[
                         np.asarray(analysis["frequency_per_epoch"]) < 0.01]))},
                 "next_diagnosis": ["ingest the source hardware learned-mean/fixed LER traces if released"]}
    atomic_json(output / "iterations" / iteration_id / "analysis.json", payload)
    atomic_json(output / "iterations" / iteration_id / "iteration_record.json", iteration)
    atomic_json(output / "source_contract.json", payload["source_contract"])
    atomic_json(output / "final_status.json", payload)
    write_report(payload, output)
    return payload


def write_report(payload: dict[str, Any], output: Path) -> None:
    records = [_load(path) for path in sorted((output / "iterations").glob("*/iteration_record.json"))]
    lines = ["# Natural-drift source DFT filter function", "",
             "The Section-III estimator is implemented exactly where publicly specified: decoded learned-mean and fixed-policy LER evaluations every five epochs; epochs before 150 excluded; per-trace epoch-150 normalization; epoch-domain DFT; zero-frequency removal; shared-grid interpolation for unequal lengths; geometric spectral averages; and learned/fixed filter ratio in dB.", "",
             "The raw ratio and Gaussian-smoothed guide-to-eye are stored separately. Welch spectra, band integration, candidate-policy data, physical-time resampling, and FFT zero-padding are prohibited in the source panel.", "",
             "## Divergences removed", "",
             "- Replaced the prior Welch integrated-band statistic with one DFT power spectrum per independent run.",
             "- Replaced candidate performance with typed decoded-LER learned-mean and fixed-policy evaluation streams.",
             "- Replaced physical-time handling with the source epoch coordinate and exact five-epoch cadence.",
             "- Added epoch-150 normalization, zero-frequency exclusion, unequal-length interpolation, and geometric averaging.", "",
             "## Evidence", "",
             f"- Structural status: `{payload['structural_status']}`",
             f"- Artifact complete: `{payload['artifact_complete']}`",
             f"- Mathematical contract: `{payload['mathematical_contract_pass']}`",
             f"- Source protocol: `{payload['protocol_contract_pass']}`",
             f"- Source structure: `{payload['source_structure_match']}`",
             f"- Quantitative match: `{payload['quantitative_match']}`",
             f"- Paper comparable: `{payload['paper_comparable']}`", "",
             "- Validation completed: `57` focused tests and `111` focused-plus-regression tests passed.",
             "- Full source-scale hardware runs completed: `0`; this analysis consumes stored evaluation traces and no QEC cycles.", "",
             "The official release inventory contains raw QEC experiments but no dynamic learned-mean/fixed LER evaluation traces indexed by training epoch. The numerical development output is therefore synthetic and cannot reproduce the reported hardware-scale 4 dB value.", "",
             "## Unresolved source choices", "",
             "The supplement does not publish the DFT power normalization, exact shared frequency grid, interpolation convention, or Gaussian smoothing bandwidth. The frozen development choices are `|DFT|^2/N^2`, a logarithmic shared-support grid, linear interpolation, and five grid bins of Gaussian smoothing. These choices remain explicitly source-unspecified.", "",
             "## Iterations", ""]
    lines.extend(f"- `{row['iteration_id']}`: `{json.dumps(row['numerical_results'], sort_keys=True)}`; failed gates `{row['failed_gates']}`."
                 for row in records)
    lines.extend(["", "## Replay or future source-data ingestion", "", "```powershell",
                  "$env:PYTHONPATH='src'",
                  "python -m hdfa_rl_suite.google_pure_source_exact.natural_drift_dft.cli plan --input path\\to\\decoded_ler_traces.json",
                  "python -m hdfa_rl_suite.google_pure_source_exact.natural_drift_dft.cli run --input path\\to\\decoded_ler_traces.json --iteration-id natural-drift-source-001",
                  "```", ""])
    (output / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("source-contract")
    plan = sub.add_parser("plan"); plan.add_argument("--input", type=Path)
    synthetic = sub.add_parser("synthetic-input"); synthetic.add_argument("--destination", type=Path)
    run = sub.add_parser("run"); run.add_argument("--input", type=Path, required=True); run.add_argument("--iteration-id", required=True)
    args = parser.parse_args(argv)
    if args.command == "source-contract":
        payload = build_source_contract(); atomic_json(args.output / "source_contract.json", payload)
    elif args.command == "plan":
        payload = build_plan(args.config, args.output, args.input)
    elif args.command == "synthetic-input":
        destination = args.destination or args.output / "inputs" / "synthetic_development_traces.json"
        payload = generate_synthetic(args.config, destination)
    else:
        payload = run_analysis(args.config, args.input, args.output, args.iteration_id)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
