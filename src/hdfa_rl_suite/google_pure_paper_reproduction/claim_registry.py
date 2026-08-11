"""Claim-level registry with explicit reproducibility and family boundaries."""
from __future__ import annotations

from typing import Any

from .experiment_families import EvidenceClass, ExperimentFamily
from .storage import atomic_json, atomic_text, initialise_layout

F = ExperimentFamily
E = EvidenceClass


def _claim(claim_id: str, statement: str, source: str, status: str, families: list[str], *, value: Any = None,
           unit: str | None = None, direct: bool = False, exclusion: str | None = None) -> dict[str, Any]:
    return {"claim_id": claim_id, "statement": statement, "source_location": source, "public_value": value, "unit": unit,
            "status": status, "experiment_families": families, "public_data_direct": direct,
            "must_not_be_combined_with": exclusion, "plot_match_is_evidence": False}


def claims() -> list[dict[str, Any]]:
    public = F.PUBLIC_ENDPOINT_DATA_REPRODUCTION.value
    table = F.PUBLIC_TABLE_REPRODUCTION.value
    return [
        _claim("headline.surface_d7_alphaqubit2", "Distance-7 surface-code LER", "article:Fig.3b; released memory data", E.PUBLIC_EXACT.value, [public, table], value=7.72e-4, unit="logical error per cycle", direct=True),
        _claim("headline.color_d5_tesseract", "Distance-5 colour-code LER", "article:Fig.3b; released memory data", E.PUBLIC_EXACT.value, [public, table], value=8.19e-3, unit="logical error per cycle", direct=True),
        _claim("table.surface_d7_sparse_blossom", "Distance-7 SI1000-prior sparse-blossom LER", "supplement:Table V; released memory data", E.PUBLIC_EXACT.value, [public, table], value=1.42e-3, unit="logical error per cycle", direct=True),
        _claim("table.color_d5_tesseract_si1000", "Distance-5 colour-code SI1000-prior Tesseract LER", "supplement:Table V; released memory data", E.PUBLIC_EXACT.value, [public, table], value=9.2e-3, unit="logical error per cycle", direct=True),
        _claim("finetuning.about_20_percent", "RL fine-tuning suppresses LER by about 20%", "article:Fig.3a", E.PUBLIC_EXACT.value, [public], value=.20, unit="relative suppression", direct=True, exclusion="not the same paired five-run trace; released endpoints are approximate support"),
        _claim("drift.control_only_stability", "Control steering improves LER stability 2.4-fold", "article:Fig.4", E.UNIDENTIFIABLE.value, [F.STEP_RESPONSE_INJECTED_DRIFT.value], value=2.4, unit="fold", exclusion="decoder steering 3.5-fold"),
        _claim("drift.control_plus_decoder_stability", "Combined control and decoder steering improves stability 3.5-fold", "article:Fig.4", E.UNIDENTIFIABLE.value, [F.STEP_RESPONSE_INJECTED_DRIFT.value], value=3.5, unit="fold", exclusion="control-only 2.4-fold"),
        _claim("drift.natural_low_frequency", "Natural low-frequency LER fluctuations are suppressed by about 4 dB", "article:Fig.4; supplement:III", E.SYNTHETIC.value, [F.NATURAL_DRIFT_SPECTRAL_SUPPRESSION.value], value=4.0, unit="dB"),
        _claim("drift.step_response", "Injected step response is about 130 epochs", "article:Fig.4b", E.SYNTHETIC.value, [F.STEP_RESPONSE_INJECTED_DRIFT.value], value=130, unit="epochs"),
        _claim("recovery.randomized_policy", "Recovery after randomized controls takes roughly 1000 epochs", "supplement:IV", E.SYNTHETIC.value, [F.RANDOMIZED_RECOVERY_AFTER_SPOIL.value], value=1000, unit="epochs"),
        _claim("fig5a.candidate_cycles", "Real-time-steering scan uses 1.8e9 candidate QEC cycles", "article:Fig.5a; supplement:VI.A", E.SYNTHETIC.value, [F.FIGURE5A_REAL_TIME_STEERING.value], value=1.8e9, unit="QEC cycles"),
        _claim("fig5a.steerability_cutoff", "Critical steerable drift frequency is about 1/150 epoch^-1", "article:Fig.5a", E.SYNTHETIC.value, [F.FIGURE5A_REAL_TIME_STEERING.value], value=1/150, unit="epoch^-1"),
        _claim("fig5a.geometry", "Figure 5a is a normalized frequency-by-entropy phase surface with a zero isoline", "article:Fig.5a", E.VISUAL.value, [F.FIGURE5A_REAL_TIME_STEERING.value]),
        _claim("fig5b.d15_controls", "Distance-15 P=30 policy has 38,670 controls", "article:Fig.5b; supplement:Table I", E.SYNTHETIC.value, [F.FIGURE5B_SPARSE_SCALING.value], value=38670, unit="controls"),
        _claim("fig5b.threshold", "Scaling simulation uses threshold physical error 1.79e-3", "supplement:VI.B", E.SYNTHETIC.value, [F.FIGURE5B_SPARSE_SCALING.value], value=1.79e-3, unit="physical error"),
        _claim("fig5c.distance_independence", "Local convergence rate is approximately independent of distance", "article:Fig.5c", E.SYNTHETIC.value, [F.FIGURE5C_CONVERGENCE_LAW.value]),
        _claim("algorithm.gaussian_policy", "Factorized Gaussian parameter-exploring policy", "supplement:VIII", E.SYNTHETIC.value, [x.value for x in F if x.value not in {public, table}]),
        _claim("algorithm.ppo_entropy_replay_mask", "PPO, entropy, replay and sparse gradient masking are used", "article:Methods; supplement:VIII", E.SYNTHETIC.value, [x.value for x in F if x.value not in {public, table}]),
        _claim("code.proprietary", "The original custom code is not public", "article:Code availability", E.UNIDENTIFIABLE.value, [x.value for x in F]),
    ]


def build_claim_registry() -> dict[str, Any]:
    rows = claims()
    result = {"schema_version": "google-paper-claim-registry.v1", "claim_count": len(rows), "claims": rows,
              "master_scalar_certification": False, "control_decoder_metrics_kept_separate": True}
    root = initialise_layout() / "claim_registry"; atomic_json(root / "claim_registry.json", result)
    lines = ["# Claim registry", "", "No plot or aggregate scalar certifies the whole paper.", "", "| Claim | Public value | Status | Family |", "|---|---:|---|---|"]
    for row in rows:
        lines.append(f"| `{row['claim_id']}` | {row['public_value'] if row['public_value'] is not None else '—'} | `{row['status']}` | {', '.join(row['experiment_families'])} |")
    atomic_text(root / "claim_registry.md", "\n".join(lines) + "\n")
    return result

