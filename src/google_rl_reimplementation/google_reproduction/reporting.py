"""Evidence-layered artifact builders for the public reproduction programme."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .config import load_reference_config, load_surrogate_config, repository_root, sha256_file
from .surrogate import surface_code_gate_count, surface_code_parameter_count


ARTIFACT_SCHEMA = "google-public-reproduction-artifact.v2"


class _Encoder(json.JSONEncoder):
    def default(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return super().default(value)


def artifact_directory() -> Path:
    path = repository_root() / "artifacts/google_reproduction_v2"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(name: str, payload: Mapping[str, Any]) -> Path:
    path = artifact_directory() / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, cls=_Encoder) + "\n", encoding="utf-8")
    return path


def write_markdown(name: str, title: str, lines: Iterable[str]) -> Path:
    path = artifact_directory() / f"{name}.md"
    path.write_text("\n".join([f"# {title}", "", *lines, ""]), encoding="utf-8")
    return path


def public_anchor_registry() -> dict[str, Any]:
    nature = "https://doi.org/10.1038/s41586-026-10759-2"
    arxiv = "https://arxiv.org/abs/2511.08493"
    entries = [
        {
            "id": "fine_tuning_ler",
            "source_document": "Sivak et al., Nature 655, 879-884 (2026)",
            "source_location": "main text and Fig. 3a; Supplement III",
            "source_url": nature,
            "published_value": "approximately 20% additional LER suppression after expert calibration",
            "uncertainty": "no single aggregate interval reported; figure uses five independent runs per code type",
            "evidence_layer": "Willow hardware experiment",
            "repository_analogue": "relative frozen logical-risk improvement of independently evaluated learned mean over contemporaneous fixed policy",
            "preregistered_tolerance": {"minimum": 0.15, "maximum": 0.25},
            "comparability": "approximate",
        },
        {
            "id": "control_only_stability",
            "source_document": "Sivak et al., Nature 655, 879-884 (2026)",
            "source_location": "main text Fig. 4c",
            "source_url": nature,
            "published_value": "2.4-fold LER standard-deviation stability improvement",
            "uncertainty": "not reported as an interval",
            "evidence_layer": "Willow hardware control steering only",
            "repository_analogue": "ratio of contemporaneous fixed-policy to learned-mean logical-risk standard deviations",
            "preregistered_tolerance": {"minimum": 1.8, "maximum": 3.0},
            "comparability": "approximate",
        },
        {
            "id": "control_plus_decoder_stability",
            "source_document": "Sivak et al., Nature 655, 879-884 (2026)",
            "source_location": "main text Fig. 4c",
            "source_url": nature,
            "published_value": "3.5-fold LER stability improvement",
            "uncertainty": "not reported as an interval",
            "evidence_layer": "Willow control steering plus separate decoder steering",
            "repository_analogue": "none: decoder steering is not implemented or credited to the control agent",
            "preregistered_tolerance": None,
            "comparability": "not commensurable",
        },
        {
            "id": "low_frequency_suppression",
            "source_document": "Sivak et al. Supplement",
            "source_location": "Supplement III and Fig. S5",
            "source_url": arxiv,
            "published_value": "approximately 4 dB low-frequency LER fluctuation suppression",
            "uncertainty": "approximate; no interval reported",
            "evidence_layer": "Fourier analysis of Willow LER traces",
            "repository_analogue": "10 log10 ratio of fixed to learned low-frequency logical-risk PSD after 150-epoch warm-up",
            "preregistered_tolerance": {"minimum_db": 2.0, "maximum_db": 6.0},
            "comparability": "approximate",
        },
        {
            "id": "step_response",
            "source_document": "Sivak et al., Nature 655, 879-884 (2026)",
            "source_location": "main text Fig. 4b",
            "source_url": nature,
            "published_value": "approximately 130 epochs",
            "uncertainty": "described as about 130; no fit interval reported",
            "evidence_layer": "Willow injected step-drift experiment",
            "repository_analogue": "first 1/e decay crossing of RMS normalized learned-mean policy error",
            "preregistered_tolerance": {"minimum_epochs": 80, "maximum_epochs": 220},
            "comparability": "approximate",
        },
        {
            "id": "randomized_recovery",
            "source_document": "Sivak et al. Supplement",
            "source_location": "Supplement IV and Fig. S6",
            "source_url": arxiv,
            "published_value": "about 1,000 epochs to calibrated-policy performance",
            "uncertainty": "no interval reported",
            "evidence_layer": "Willow experiment with randomized controls",
            "repository_analogue": "first learned-mean epoch at or below the frozen calibrated-surrogate risk",
            "preregistered_tolerance": {"minimum_epochs": 700, "maximum_epochs": 1300},
            "comparability": "approximate",
        },
        {
            "id": "candidates_per_epoch",
            "source_document": "Sivak et al. Supplement",
            "source_location": "Supplement I.B",
            "source_url": arxiv,
            "published_value": 40,
            "uncertainty": None,
            "evidence_layer": "hardware training protocol",
            "repository_analogue": "exact independent policy samples per reference epoch",
            "preregistered_tolerance": {"exact": 40},
            "comparability": "exact",
        },
        {
            "id": "candidate_evidence_budget",
            "source_document": "Sivak et al. Supplement",
            "source_location": "Supplement I.B",
            "source_url": arxiv,
            "published_value": "4,000 shots x 25 cycles = 100,000 effective QEC cycles per candidate",
            "uncertainty": None,
            "evidence_layer": "hardware training protocol",
            "repository_analogue": "binomial sufficient statistics with exact native-cycle accounting",
            "preregistered_tolerance": {"exact_effective_cycles": 100000},
            "comparability": "exact protocol structure; synthetic outcomes",
        },
        {
            "id": "ideal_acquisition_time",
            "source_document": "Sivak et al. Supplement",
            "source_location": "Supplement I.B",
            "source_url": arxiv,
            "published_value": "approximately 0.1 s/candidate and 4 s/40-candidate epoch at 1 microsecond cycles",
            "uncertainty": "ideal projection; current experiments stated as 1-10 minutes",
            "evidence_layer": "protocol projection",
            "repository_analogue": "cost report only; host runtime is separately measured",
            "preregistered_tolerance": None,
            "comparability": "qualitative cost accounting",
        },
        {
            "id": "steering_threshold",
            "source_document": "Sivak et al., Nature and Supplement VI",
            "source_location": "main text Fig. 5a; Supplement Figs. S8-S9",
            "source_url": arxiv,
            "published_value": "approximately 1/150 epochs^-1",
            "uncertainty": "approximate visual/behavioral threshold",
            "evidence_layer": "paper simulation",
            "repository_analogue": "largest tested frequency with positive normalized stochastic steering advantage",
            "preregistered_tolerance": {"minimum": "1/225", "maximum": "1/100"},
            "comparability": "approximate",
        },
        {
            "id": "slow_drift_balanced_entropy",
            "source_document": "Sivak et al. Supplement",
            "source_location": "Supplement VI, Fig. S8",
            "source_url": arxiv,
            "published_value": "stochastic policy approaches optimum at slow drift with balanced entropy regularization",
            "uncertainty": None,
            "evidence_layer": "paper simulation",
            "repository_analogue": "normalized stochastic steering advantage >= 0.8 at 1/1000 epochs and entropy weight 0.01",
            "preregistered_tolerance": {"minimum_normalized_advantage": 0.8},
            "comparability": "qualitative plus declared quantitative operationalization",
        },
        {
            "id": "distance_15_parameter_count",
            "source_document": "Sivak et al. Supplement",
            "source_location": "Supplement II Table I and Supplement VI Eq. 7",
            "source_url": arxiv,
            "published_value": "38,670 controls at d=15 and P=30 controls per gate",
            "uncertainty": None,
            "evidence_layer": "paper simulation structure",
            "repository_analogue": "(6d^2-4d-1)P = 1,289 x 30 = 38,670 sparse controls",
            "preregistered_tolerance": {"exact": 38670},
            "comparability": "exact structural reproduction",
        },
    ]
    return {
        "schema_version": "google-public-anchor-registry.v2",
        "paper": "Sivak et al., 'Real-time quantum error correction beyond break-even', Nature 655, 879-884 (2026)",
        "paper_doi": "10.1038/s41586-026-10759-2",
        "preprint_version": "arXiv:2511.08493v4",
        "claim_boundary": "Targets anchor an open algorithmic surrogate reproduction; none certify Willow hardware equivalence.",
        "anchors": entries,
    }


def write_anchor_registry() -> dict[str, Any]:
    payload = public_anchor_registry()
    write_json("public_anchor_registry", payload)
    rows = [
        "| Anchor | Published value | Evidence | Analogue | Tolerance/class |",
        "|---|---|---|---|---|",
    ]
    for item in payload["anchors"]:
        tolerance = json.dumps(item["preregistered_tolerance"], sort_keys=True) if item["preregistered_tolerance"] is not None else "none"
        rows.append(
            f"| {item['id']} | {item['published_value']} | {item['evidence_layer']} | "
            f"{item['repository_analogue']} | {tolerance}; {item['comparability']} |"
        )
    write_markdown(
        "public_anchor_registry",
        "Public anchor registry",
        [
            "This registry separates Willow hardware, decoder, protocol, and paper-simulation evidence. The 2.4x control-only result is never merged with the 3.5x decoder-assisted result.",
            "",
            *rows,
            "",
            "Sources: [Nature article](https://doi.org/10.1038/s41586-026-10759-2), [arXiv v4 and Supplement](https://arxiv.org/abs/2511.08493).",
        ],
    )
    return payload


def surrogate_contract() -> dict[str, Any]:
    cfg = load_surrogate_config()
    rows = [
        {
            "distance": d,
            "gate_count": surface_code_gate_count(d),
            "controls_per_gate": 30,
            "control_count": surface_code_parameter_count(d, 30),
        }
        for d in cfg["layout"]["odd_distances"]
    ]
    return {
        "schema_version": "google-paper-anchored-surrogate-contract.v2",
        "configuration": cfg,
        "configuration_sha256": sha256_file(repository_root() / "configs/google_rl/paper_anchored_surrogate_v2.yaml"),
        "frozen_before_controller_results": bool(cfg["frozen_before_controller_tuning"]),
        "scaling": rows,
        "hard_sanity_requirements": [
            "no-drift fixed policy stationary",
            "oracle at irreducible floor",
            "persistent drift degrades fixed policy",
            "larger validated mismatch increases detector cost",
            "detector and frozen logical-risk mappings agree directionally",
            "high-shot learned mean moves toward optimum",
            "random policy is poor",
            "inactive controls have zero masked gradient",
        ],
        "differences_from_unavailable_google_simulator": [
            "No Willow device, pulse stack, proprietary calibration state, detector graph, or control sensitivities.",
            "Frozen random local factors replace proprietary control-to-detector coefficients.",
            "Independent conditional Bernoulli detector counts omit unknown hardware correlations and leakage dynamics.",
            "The logical metric is a monotone declared risk proxy, not decoder-estimated hardware LER.",
            "Stim validates public surface-code topology where installed; it does not make the quadratic plant a circuit/pulse simulator.",
            "Wall-clock protocol projections are accounting fields, not measured hardware latency.",
        ],
        "evidence_boundary": "Algorithmic reproduction on a paper-anchored surrogate only.",
    }


def write_surrogate_contract() -> dict[str, Any]:
    payload = surrogate_contract()
    write_json("surrogate_contract", payload)
    last = payload["scaling"][-1]
    write_markdown(
        "surrogate_contract",
        "Paper-anchored surrogate contract",
        [
            f"The surrogate was frozen before controller experiments. At distance 15 it has {last['gate_count']:,} gates x 30 controls = {last['control_count']:,} parameters.",
            "",
            "It uses sparse local detector factors, frozen sensitivities/floors/curvatures, quadratic mismatch, sinusoidal and step drift, and distinct surrogate/controller/certification splits.",
            "",
            "## Non-equivalences",
            "",
            *[f"- {item}" for item in payload["differences_from_unavailable_google_simulator"]],
        ],
    )
    return payload


def source_tree_hash() -> tuple[str, dict[str, str]]:
    roots = [
        repository_root() / "src/google_rl_reimplementation/google_reproduction",
        repository_root() / "configs/google_rl",
        repository_root() / "tests/test_google_reproduction_v2.py",
        repository_root() / "pyproject.toml",
    ]
    files = []
    for root in roots:
        candidates = [root] if root.is_file() else root.rglob("*")
        files.extend(path for path in candidates if path.is_file() and "__pycache__" not in path.parts)
    files = sorted(files)
    records = {str(path.relative_to(repository_root())).replace("\\", "/"): sha256_file(path) for path in files}
    joined = "\n".join(f"{key}:{value}" for key, value in records.items()).encode()
    return hashlib.sha256(joined).hexdigest(), records


def initial_gate_artifacts() -> None:
    """Create fail-closed placeholders; commands replace them only when gates permit."""
    prereg = {
        "schema_version": "google-public-certification-preregistration.v2",
        "status": "DRAFT_BLOCKED_PENDING_DEVELOPMENT_GATES",
        "frozen": False,
        "certification_seeds": list(load_reference_config().untouched_certification_seeds),
        "certification_seeds_consumed": False,
        "single_run_limit": 1,
        "anchor_tolerances": {item["id"]: item["preregistered_tolerance"] for item in public_anchor_registry()["anchors"]},
        "blocked_by": ["development scorecard not yet complete", "source-unspecified sensitivity results pending"],
    }
    write_json("certification_preregistration", prereg)
    write_markdown(
        "certification_preregistration",
        "Certification preregistration (draft)",
        ["Status: `DRAFT_BLOCKED_PENDING_DEVELOPMENT_GATES`.", "", "Certification seeds 8101-8112 remain untouched. Freezing is refused until all development and source-choice gates are resolved."],
    )
    final = {
        "schema_version": "google-public-final-certification.v2",
        "status": "FROZEN_CERTIFICATION_NOT_RUN",
        "outcome": None,
        "certification_executed": False,
        "certification_run_count": 0,
        "certification_seeds_consumed": False,
        "claim_boundary": "No certification or public-paper performance claim exists before the single-use run.",
    }
    write_json("final_certification", final)
    write_markdown(
        "final_certification",
        "Final certification",
        ["Status: `FROZEN_CERTIFICATION_NOT_RUN`.", "", "No allowed final outcome is assigned because held-out certification has not been executed. This fail-closed placeholder is not a result."],
    )
    reduced = {
        "schema_version": "google-public-reduced-budget-equivalence.v2",
        "status": "BLOCKED_REFERENCE_NOT_CERTIFIED",
        "evaluated": False,
        "reason": "Reduced-budget equivalence is forbidden until the paper-scale reference is certified.",
    }
    write_json("reduced_budget_equivalence", reduced)
    write_markdown(
        "reduced_budget_equivalence",
        "Reduced-budget equivalence",
        ["Status: `BLOCKED_REFERENCE_NOT_CERTIFIED`.", "", "The legacy 2,048-cycle equivalence label is historical and is not transferred to this programme."],
    )


def write_public_data_reproduction() -> dict[str, Any]:
    payload = {
        "schema_version": "google-public-data-reproduction.v2",
        "evidence_layer": "released data, kept separate from surrogate certification",
        "search_scope": [
            "repository tree",
            "top level of D:/Users/Raife/DDriveDownloads",
            "matching local QEC-RL archive manifest and source notes",
        ],
        "official_archives": {
            "v1": {
                "doi": "10.5281/zenodo.17566522",
                "expected_size": "127.8 MB",
                "expected_md5": "31de3a1130689afddc384e59fec4d1bd",
                "available_locally": False,
            },
            "v2": {
                "doi": "10.5281/zenodo.18896801",
                "expected_size": "7.8 GB",
                "expected_md5": "ca54323082fcd0e3671d5b90ce45d85c",
                "available_locally": False,
            },
        },
        "matching_nonofficial_archive": {
            "path": "D:/Users/Raife/DDriveDownloads/google_rl_qec_qgss_tutorial_v0_4.zip",
            "bytes": 1043120,
            "sha256": "94380e417ab672a9a79e0ea8e93749fcca1131b3c3d22938d0a259fb4ec1a109",
            "md5": "1a1095a909440cd9fe5a6da66b7e5cfa",
            "classification": "pedagogical archive containing a tiny explicitly synthetic demo_public_data tree",
            "excluded_reason": "its README says Lab 6 defaults to a tiny clearly-labelled synthetic archive; its checksum and size do not match either Zenodo release",
        },
        "published_statistics_recomputed": False,
        "normalization_recomputed": False,
        "ler_improvement_recomputed": False,
        "psd_recomputed": False,
        "status": "PUBLIC_DATA_UNAVAILABLE_LOCALLY",
        "reason": "Neither checksum-identified official Zenodo archive is local. No data result is fabricated and the 7.8 GB archive is not downloaded automatically.",
        "remote_record": "https://zenodo.org/records/18896801",
    }
    write_json("public_data_reproduction", payload)
    write_markdown(
        "public_data_reproduction",
        "Public-data reproduction",
        [
            "Status: `PUBLIC_DATA_UNAVAILABLE_LOCALLY`.",
            "",
            "The official 127.8 MB v1 and 7.8 GB v2 archives were not found locally. A 1.0 MB tutorial archive was found, but its own README labels the bundled demo as synthetic and its checksum matches neither official release. It is therefore excluded from public-data evidence.",
            "",
            "No fixed-versus-RL, normalization, LER, or PSD statistic is reported as a public-data reproduction. The official [Zenodo v2 record](https://zenodo.org/records/18896801) remains a separate optional acquisition.",
        ],
    )
    return payload


DEVELOPMENT_ANCHORS = (
    "fine_tuning",
    "drift_stability",
    "step_response",
    "randomized_recovery",
    "steering_phase",
    "scaling",
)


def _initial_scorecard() -> dict[str, Any]:
    return {
        "schema_version": "google-public-development-anchor-scorecard.v2",
        "evidence_layer": "controller-development seeds on frozen synthetic surrogate",
        "development_seeds": list(load_reference_config().development_seeds),
        "certification_seeds_consumed": False,
        "anchors": {name: {"status": "NOT_RUN"} for name in DEVELOPMENT_ANCHORS},
        "decoder_steering": {
            "status": "NOT_EVALUATED_NOT_IMPLEMENTED",
            "credit_to_control_agent": False,
        },
        "source_choice_sensitivity": {"status": "NOT_RUN"},
        "all_required_development_gates_pass": False,
    }


def write_development_scorecard(
    anchor: str | None = None,
    result: Mapping[str, Any] | None = None,
    *,
    sensitivity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = artifact_directory() / "development_anchor_scorecard.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else _initial_scorecard()
    if anchor is not None:
        if anchor not in DEVELOPMENT_ANCHORS or result is None:
            raise ValueError("unknown development anchor or missing result")
        summary = dict(result.get("summary", {}))
        if anchor == "scaling":
            summary = {
                "status": result["status"],
                "distance_15_control_count": result["distance_15_control_count"],
                "distances": result["distances"],
            }
        payload["anchors"][anchor] = {
            "status": summary.get("status", "FAIL"),
            "seed": result.get("seed"),
            "epochs": result.get("epochs", result.get("epochs_per_cell")),
            "summary": summary,
            "native_cost": result.get("native_cost"),
            "host_runtime_seconds": result.get("host_runtime_seconds"),
            "learned_mean_and_stochastic_separate": anchor in {"fine_tuning", "drift_stability", "step_response", "randomized_recovery", "steering_phase"},
        }
    if sensitivity is not None:
        payload["source_choice_sensitivity"] = {"status": "COMPLETE", **dict(sensitivity)}
    payload["all_required_development_gates_pass"] = all(
        payload["anchors"][name]["status"] == "PASS" for name in DEVELOPMENT_ANCHORS
    ) and payload["source_choice_sensitivity"].get("status") == "COMPLETE"
    write_json("development_anchor_scorecard", payload)
    lines = [
        "This is development evidence only. Seeds 8101-8112 remain unopened.",
        "",
        "| Anchor | Status | Key result |",
        "|---|---|---|",
    ]
    for name in DEVELOPMENT_ANCHORS:
        item = payload["anchors"][name]
        lines.append(f"| {name} | {item['status']} | `{json.dumps(item.get('summary', {}), sort_keys=True, cls=_Encoder)}` |")
    lines.extend([
        "",
        f"All required gates pass: `{payload['all_required_development_gates_pass']}`.",
        "",
        "Decoder steering is not implemented, is not evaluated, and is not credited to the control agent.",
    ])
    write_markdown("development_anchor_scorecard", "Development anchor scorecard", lines)
    return payload


def write_amendment_log(extra_entries: Iterable[Mapping[str, Any]] = ()) -> list[dict[str, Any]]:
    aggregate, files = source_tree_hash()
    entries = [
        {
            "iteration": 0,
            "source_hash_before": "3b38853adf4dafcc4cbcf2b0c223f9c0a1b0e44f47f26f23b260f1bb279be2e4",
            "source_hash_after": "9d794d586a92b6afea3588843258a8037f8e82ef3249f41f3737bd7c2b6a537d",
            "hypothesis": "Historical Track A evidence is not sufficient to establish public-paper algorithmic fidelity.",
            "files_changed": ["new google_reproduction package and v2 configs; standalone source files written"],
            "development_seeds_used": [],
            "metrics_before": {"evidence_label": "historical REDUCED_BUDGET_EQUIVALENT"},
            "metrics_after": {"evidence_label": "historical only; v2 certification fail-closed"},
            "retained": True,
            "affected_frozen_definition": False,
        },
        *[dict(item) for item in extra_entries],
    ]
    jsonl = artifact_directory() / "amendment_log.jsonl"
    jsonl.write_text("".join(json.dumps(item, sort_keys=True, cls=_Encoder) + "\n" for item in entries), encoding="utf-8")
    write_markdown(
        "amendment_log",
        "Controlled amendment log",
        [
            "All amendments precede certification; certification seeds were not inspected.",
            "",
            *[
                f"- Iteration {item['iteration']}: {item['hypothesis']} Retained: `{item['retained']}`; affected frozen definition: `{item['affected_frozen_definition']}`."
                for item in entries
            ],
            "",
            f"Current v2 source/config aggregate SHA-256: `{aggregate}`.",
        ],
    )
    return entries


def write_development_failure_outcome() -> dict[str, Any]:
    scorecard = json.loads((artifact_directory() / "development_anchor_scorecard.json").read_text(encoding="utf-8"))
    passed = [name for name in DEVELOPMENT_ANCHORS if scorecard["anchors"][name]["status"] == "PASS"]
    failed = [name for name in DEVELOPMENT_ANCHORS if scorecard["anchors"][name]["status"] != "PASS"]
    payload = {
        "schema_version": "google-public-final-certification.v2",
        "status": "TERMINATED_BEFORE_CERTIFICATION_DEVELOPMENT_GATES_FAILED",
        "outcome": "REPRODUCTION_FAILED_ALGORITHM",
        "certification_executed": False,
        "certification_run_count": 0,
        "certification_seeds_consumed": False,
        "development_anchors_passed": passed,
        "development_anchors_failed": failed,
        "passed_subanchors": ["low_frequency_suppression_db"] if scorecard["anchors"]["drift_stability"].get("summary", {}).get("low_frequency_suppression_db", -999) >= 2 else [],
        "reason": "Fast correctness and plant sanity passed, but not all preregistered development anchors passed. Opening held-out certification would be scientifically invalid.",
        "claim_boundary": "No frozen certification and no Willow hardware reproduction claim.",
    }
    write_json("final_certification", payload)
    write_markdown(
        "final_certification",
        "Final certification",
        [
            "Outcome: `REPRODUCTION_FAILED_ALGORITHM`.",
            "",
            "The workflow terminated before held-out certification because development gates failed. Certification seeds 8101-8112 remain untouched and the single-use run count is zero.",
            "",
            f"Development anchors passed: {', '.join(passed) or 'none'}.",
            "",
            f"Development anchors failed: {', '.join(failed) or 'none'}.",
            "",
            "This is an honest pre-certification failure on the declared surrogate, not evidence about Willow hardware.",
        ],
    )
    return payload
