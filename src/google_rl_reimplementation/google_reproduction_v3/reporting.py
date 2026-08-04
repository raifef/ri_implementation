"""Artifact production, evidence maps, split freezing, and v2 causality review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .dataset_manifest import load_json_yaml
from .schemas import CERTIFICATION_SEEDS, ReproductionStatus
from .zenodo_loader import ZenodoArchive


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_source_to_dataset_map(artifact_dir: Path) -> dict[str, Any]:
    exact = ReproductionStatus.EXACTLY_REPRODUCED.value
    approximate = ReproductionStatus.REPRODUCED_WITH_DOCUMENTED_APPROXIMATION.value
    impossible = ReproductionStatus.NOT_REPRODUCIBLE_FROM_RELEASED_DATA.value
    ambiguous = ReproductionStatus.ANALYSIS_DEFINITION_AMBIGUOUS.value
    entries = [
        {
            "id": "headline_surface_d7_ler",
            "paper_location": "main text/Fig. 3b; Supplement Fig. S7f and Table II",
            "dataset_files": ["surface_code_distance_3_5_7/traditional_calibration_and_rl_fine_tuning/d7_0+0j/{X,Z}/r*/metadata.json", ".../obs_flips_actual.b8", ".../decoding_results/alphaqubit2_decoder/obs_flips_predicted.b8"],
            "keys": ["basis", "rounds", "shots"],
            "preprocessing": "XOR actual and predicted observable bits; fit logical-observable decay separately by basis; average basis LERs",
            "estimator": "weighted log expectation decay with free contrast",
            "uncertainty": "binomial parametric bootstrap of complete decay curves",
            "status": exact,
        },
        {
            "id": "headline_color_d5_ler",
            "paper_location": "main text/Fig. 3b",
            "dataset_files": ["color_code_distance_5/traditional_calibration_and_rl_fine_tuning/{X,Z}/r*/metadata.json", ".../obs_flips_actual.b8", ".../decoding_results/tesseract_decoder_with_frequency_calibrated_prior/obs_flips_predicted.b8"],
            "keys": ["basis", "rounds", "shots"],
            "preprocessing": "same memory-decay pipeline as surface code",
            "estimator": "weighted log expectation decay with free contrast",
            "uncertainty": "binomial parametric bootstrap",
            "status": exact,
        },
        {
            "id": "fine_tuning_result",
            "paper_location": "Fig. 3a; Supplement Fig. S4",
            "dataset_files": ["*/traditional_calibration/**", "*/traditional_calibration_and_rl_fine_tuning/**"],
            "keys": ["condition encoded in directory", "basis", "rounds", "shots"],
            "preprocessing": "compare separately acquired final memory-decay datasets",
            "estimator": "relative change of fitted LER",
            "uncertainty": "bootstrap of decay fits",
            "status": approximate,
            "limitation": "the five contemporaneous fixed/learned evaluation traces and epoch labels in Fig. S4 are not released; endpoint datasets are not the 20% multi-run estimand",
        },
        {
            "id": "fixed_policy_comparator",
            "paper_location": "Figs. 3a, 4c and S4",
            "dataset_files": ["*/traditional_calibration/**"],
            "keys": ["directory condition only"],
            "preprocessing": "treat traditional-calibration memory datasets as a non-contemporaneous endpoint comparator",
            "estimator": "memory-decay LER",
            "uncertainty": "binomial bootstrap",
            "status": approximate,
            "limitation": "no time-matched mu(0) reevaluations or injected-drift labels",
        },
        {
            "id": "learned_mean_comparator",
            "paper_location": "Figs. 3a, 4 and S7",
            "dataset_files": ["*/traditional_calibration_and_rl_fine_tuning/**"],
            "keys": ["directory condition only"],
            "preprocessing": "final RL-fine-tuned memory dataset",
            "estimator": "memory-decay LER",
            "uncertainty": "binomial bootstrap",
            "status": ambiguous,
            "limitation": "the directory does not encode policy-distribution mean versus sampled candidate semantics",
        },
        {
            "id": "stochastic_policy_evaluation",
            "paper_location": "Fig. 5a and Supplement Fig. S8",
            "dataset_files": [],
            "keys": [],
            "preprocessing": None,
            "estimator": "normalized cumulative EDR of stochastic policy",
            "uncertainty": "not recoverable",
            "status": impossible,
        },
        {
            "id": "drift_injection_traces",
            "paper_location": "Fig. 4a-c",
            "dataset_files": [],
            "keys": [],
            "preprocessing": None,
            "estimator": "epoch-indexed EDR/control/LER trajectories",
            "uncertainty": "not recoverable",
            "status": impossible,
        },
        {
            "id": "control_only_stability_ratio",
            "paper_location": "main text/Fig. 4c (2.4x); combined decoder result is 3.5x",
            "dataset_files": [],
            "keys": [],
            "preprocessing": "requires separate fixed, control-only, and decoder-steered time series",
            "estimator": "ratio of LER-distribution standard deviations",
            "uncertainty": "not recoverable",
            "status": impossible,
        },
        {
            "id": "low_frequency_psd_suppression",
            "paper_location": "Supplement Section III and Fig. S5 (~4 dB)",
            "dataset_files": [],
            "keys": [],
            "preprocessing": "warmup 150 epochs; normalize at epoch 150; DFT; interpolate spectra; geometric run average; exclude DC",
            "estimator": "10 log10 of fixed/steered PSD power ratio; low-frequency reading is described approximately",
            "uncertainty": "run traces absent",
            "status": impossible,
        },
        {
            "id": "step_response",
            "paper_location": "Fig. 4b (~130 epochs)",
            "dataset_files": [],
            "keys": [],
            "preprocessing": "exponential fit to learned control response after XY-amplitude step",
            "estimator": "characteristic exponential time",
            "uncertainty": "fit data absent",
            "status": impossible,
        },
        {
            "id": "randomized_policy_recovery",
            "paper_location": "Supplement Section IV and Fig. S6 (~1000 epochs)",
            "dataset_files": [],
            "keys": [],
            "preprocessing": "requires spoiled policy, training trajectory and calibrated-policy target",
            "estimator": "figure describes smoothed guides; no sustained-crossing rule is specified",
            "uncertainty": "not recoverable",
            "status": impossible,
        },
        {
            "id": "steering_phase_diagram",
            "paper_location": "Fig. 5a and Supplement Section VI A/S8 (~1/150 epoch^-1)",
            "dataset_files": [],
            "keys": [],
            "preprocessing": "requires simulated fixed/optimal/stochastic/learned policy trajectories and entropy sweep",
            "estimator": "zero isoline of normalized cumulative EDR improvement",
            "uncertainty": "simulation samples absent",
            "status": impossible,
        },
        {
            "id": "normalization_constants",
            "paper_location": "Supplement Section I B",
            "dataset_files": ["*/metadata.json"],
            "keys": ["rounds", "shots", "basis"],
            "preprocessing": "released memory units are exact; training units (40 candidates, 4000 shots, 25 cycles) are paper-only",
            "estimator": "per-cycle parity inversion for released memory; paper-defined cycles per epoch for RL",
            "uncertainty": "not applicable",
            "status": approximate,
        },
        {
            "id": "published_uncertainty",
            "paper_location": "headline values and Table II",
            "dataset_files": ["metadata.json", "obs_flips_actual.b8", "obs_flips_predicted.b8"],
            "keys": ["shots"],
            "preprocessing": "resample released shot-level logical outcomes",
            "estimator": "parametric binomial bootstrap",
            "uncertainty": "independently reproducible, but exact unpublished random seed/resampling convention is unknown",
            "status": approximate,
        },
        {
            "id": "scaling",
            "paper_location": "Fig. 5b-c and Supplement Section VI B",
            "dataset_files": [],
            "keys": [],
            "preprocessing": "paper formula gives Ptot=(2d^2-1)P+(4d^2-4d)P and 38,670 at d=15,P=30",
            "estimator": "convergence gamma from Lambda trajectories",
            "uncertainty": "simulation trajectories and sensitivity draws absent",
            "status": impossible,
            "limitation": "the 38,670 count is structurally checkable from the paper, not empirically from this archive",
        },
    ]
    result = {
        "schema_version": "google-source-to-dataset-map.v3",
        "release_scope": "496 static quantum-memory experiment cells; no RL training or control trajectories",
        "entries": entries,
    }
    write_json(artifact_dir / "source_to_dataset_map.json", result)
    lines = [
        "# Paper-to-Zenodo evidence map",
        "",
        "The release contains static QEC memory shots, circuits, detector events, and decoder predictions. It does not contain epoch-indexed control policies, candidate actions, injected-drift labels, or scaling-simulation trajectories.",
        "",
        "| Anchor | Paper location | Released support | Status |",
        "|---|---|---|---|",
    ]
    for item in entries:
        files = "; ".join(item["dataset_files"]) if item["dataset_files"] else "none"
        lines.append(f"| {item['id']} | {item['paper_location']} | {files} | `{item['status']}` |")
    lines.extend(["", "Control-only 2.4x and combined control-plus-decoder 3.5x are kept as distinct estimands."])
    write_markdown(artifact_dir / "source_to_dataset_map.md", lines)
    return result


def _digest_bucket(experiment_id: str, modulus: int) -> int:
    return int(experiment_id, 16) % modulus


def freeze_data_splits(archive_path: Path, split_config_path: Path, artifact_dir: Path) -> dict[str, Any]:
    config = load_json_yaml(split_config_path)
    with ZenodoArchive(archive_path) as archive:
        records = archive.records()
    splits: dict[str, Any] = {
        "analysis_reproduction": {
            **config["analysis_reproduction"],
            "experiment_ids": [record.experiment_id for record in records],
            "experiment_count": len(records),
        }
    }
    selected_ids: dict[str, set[str]] = {}
    for name in ("surrogate_fit", "surrogate_validation"):
        rule = config[name]
        candidates = [
            record
            for record in records
            if record.rounds in rule["allowed_rounds"]
            and _digest_bucket(record.experiment_id, rule["experiment_digest_modulus"]) == rule["experiment_digest_remainder"]
        ]
        chosen = sorted(candidates, key=lambda row: (row.code_family, row.condition, row.distance, row.experiment_id))[: rule["maximum_experiments"]]
        entries = [
            {
                "experiment_id": record.experiment_id,
                "data_dir": record.data_dir,
                "member": record.data_dir + rule["member"],
                "shot_block": rule["shot_block"],
            }
            for record in chosen
        ]
        selected_ids[name] = {x["experiment_id"] for x in entries}
        splits[name] = {**rule, "entries": entries, "experiment_count": len(entries)}
    if selected_ids["surrogate_fit"] & selected_ids["surrogate_validation"]:
        raise AssertionError("fit and validation experiment splits overlap")
    for name in ("controller_development", "final_certification"):
        rule = config[name]
        candidates = [
            record.experiment_id
            for record in records
            if _digest_bucket(record.experiment_id, rule["experiment_digest_modulus"]) == rule["experiment_digest_remainder"]
        ]
        splits[name] = {**rule, "reserved_experiment_ids": sorted(candidates), "data_consumed": False}
    for split in splits.values():
        split["content_sha256"] = canonical_hash(split)
    result = {
        "schema_version": "google-v3-data-split-manifest.v1",
        "partition_method": config["freeze_policy"],
        "archive_path": str(archive_path.resolve()).replace("\\", "/"),
        "splits": splits,
        "fit_validation_disjoint": True,
        "controller_development_eligible": False,
        "final_certification_locked": True,
        "certification_seeds": list(CERTIFICATION_SEEDS),
        "certification_seeds_consumed": False,
    }
    result["manifest_sha256"] = canonical_hash(result)
    write_json(artifact_dir / "data_split_manifest.json", result)
    return result


def snapshot_v2(workspace: Path, artifact_dir: Path) -> dict[str, Any]:
    roots = [
        workspace / "src/google_rl_reimplementation/google_reproduction",
        workspace / "artifacts/google_reproduction_v2",
        workspace / "configs/google_rl",
        workspace / "tests/test_google_reproduction_v2.py",
    ]
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(path for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    entries = []
    for path in sorted(files):
        relative = path.relative_to(workspace).as_posix()
        entries.append({"path": relative, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    result = {
        "schema_version": "google-v2-immutability-snapshot.v1",
        "scope": [str(root.relative_to(workspace)).replace("\\", "/") for root in roots],
        "files": entries,
        "aggregate_sha256": canonical_hash(entries),
        "v2_modified_by_v3": False,
    }
    write_json(artifact_dir / "v2_immutability_snapshot.json", result)
    return result


def reclassify_v2(workspace: Path, artifact_dir: Path) -> dict[str, Any]:
    scorecard = json.loads((workspace / "artifacts/google_reproduction_v2/development_anchor_scorecard.json").read_text(encoding="utf-8"))
    final = json.loads((workspace / "artifacts/google_reproduction_v2/final_certification.json").read_text(encoding="utf-8"))
    rows = {
        "fine_tuning": {
            "v2_value": scorecard["anchors"]["fine_tuning"]["summary"]["relative_improvement"],
            "classification": "ANALYSIS_DEFINITION_MISMATCH",
            "reason": "v2 synthetic fixed-versus-learned logical-risk improvement is not the paper's five-run contemporaneous one-point LER estimand; released endpoint comparisons yield different 25-33% changes",
        },
        "drift_stability": {
            "v2_value": scorecard["anchors"]["drift_stability"]["summary"]["control_only_stability_ratio"],
            "classification": "TASK_NOT_COMMENSURABLE",
            "reason": "the release has no injected-drift or time-matched fixed/control-only traces, and v2's synthetic drift/noise law cannot be empirically matched",
            "algorithm_underperformance_retained": False,
        },
        "low_frequency_suppression": {
            "v2_value_db": scorecard["anchors"]["drift_stability"]["summary"]["low_frequency_suppression_db"],
            "classification": "INSUFFICIENT_PUBLIC_INFORMATION",
            "reason": "the paper defines epoch-domain PSD processing, but the underlying multi-run LER traces are absent from the release",
        },
        "step_response": {
            "v2_value_epochs": scorecard["anchors"]["step_response"]["summary"]["characteristic_response_epochs"],
            "classification": "TASK_NOT_COMMENSURABLE",
            "reason": "numerical agreement with 130 epochs is encouraging but the physical step magnitude, action scaling, observable, and raw fit trajectory are not released",
        },
        "randomized_recovery": {
            "v2_value_epochs": scorecard["anchors"]["randomized_recovery"]["summary"]["recovery_epoch"],
            "classification": "TASK_NOT_COMMENSURABLE",
            "reason": "526 epochs is faster, not worse; the released data do not identify spoil distribution, severity, smoothing, or a sustained-crossing definition",
            "algorithm_underperformance_retained": False,
        },
        "steering_phase": {
            "v2_value_per_epoch": scorecard["anchors"]["steering_phase"]["summary"]["critical_frequency_per_epoch"],
            "classification": "TASK_NOT_COMMENSURABLE",
            "secondary_classification": "INSUFFICIENT_PUBLIC_INFORMATION",
            "reason": "the public threshold is a zero isoline of a particular Stim simulation and entropy sweep; those trajectories and sensitivity draws are absent",
            "algorithm_underperformance_retained": False,
        },
        "scaling": {
            "v2_distance15_controls": scorecard["anchors"]["scaling"]["summary"]["distance_15_control_count"],
            "classification": "INSUFFICIENT_PUBLIC_INFORMATION",
            "reason": "38,670 is structurally exact from the paper formula, but released data contain neither d15 learning curves nor gamma estimates",
        },
    }
    result = {
        "schema_version": "google-v2-failure-reclassification.v3",
        "previous_outcome": final["outcome"],
        "revised_overall_classification": "TASK_NOT_COMMENSURABLE",
        "algorithm_failure_supported_by_public_release": False,
        "anchors": rows,
        "certification_seeds_consumed": False,
        "standalone_reference_workflow": True,
        "prompt2_decision": "NO_GO",
        "prompt2_reason": "controller-critical action response, drift, reward noise under policy candidates, spoil severity, and steering/scaling dynamics are not identifiable from the static release",
    }
    write_json(artifact_dir / "v2_failure_reclassification.json", result)
    lines = [
        "# v2 failure reclassification",
        "",
        f"The prior `{final['outcome']}` label is narrowed to **`TASK_NOT_COMMENSURABLE`**. This does not demonstrate that the v2 algorithm is adequate; it says the public release cannot support that controller-level causal diagnosis.",
        "",
        "| Anchor | v2 result | Reclassification | Reason |",
        "|---|---:|---|---|",
    ]
    for name, row in rows.items():
        values = [str(value) for key, value in row.items() if key.startswith("v2_value") or key == "v2_distance15_controls"]
        lines.append(f"| {name} | {', '.join(values)} | `{row['classification']}` | {row['reason']} |")
    lines.extend(["", "## Prompt 2", "", "**NO-GO.** The empirical surrogate can validate static observation statistics only, not the controller-critical dynamics required for amendment."])
    write_markdown(artifact_dir / "v2_failure_reclassification.md", lines)
    return result
