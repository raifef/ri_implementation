"""Natural-drift spectral suppression through paired decoded-LER DFT traces."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from hdfa_rl_suite.google_pure_source_exact.natural_drift_dft.contracts import EvaluationTrace, SourceDFTConfig
from hdfa_rl_suite.google_pure_source_exact.natural_drift_dft.estimator import analyze_traces
from hdfa_rl_suite.google_pure_source_exact.paper_families.natural import acquire_natural_condition


def acquire_condition(protocol: Mapping[str, Any], condition: Mapping[str, Any]) -> dict[str, Any]:
    return acquire_natural_condition(protocol, condition)


def validation(rows: list[dict[str, Any]], mode: str) -> tuple[bool, list[str], dict[str, Any]]:
    gains = [row["low_frequency_suppression_db_fixed_over_mean"] for row in rows]
    reasons = []
    if any(row["power_db_convention"] != "10*log10(learned/fixed)" for row in rows): reasons.append("wrong PSD dB convention")
    if mode in {"reference", "paper-scale"} and len(rows) != 6: reasons.append("reference natural-drift ensemble must contain all six frozen plants")
    if any(row.get("stream_kind") != "DECODED_LER_EVALUATION" for row in rows):
        reasons.append("source spectral claim requires paired decoded-LER learned-mean and fixed-policy traces")
    if any(not row.get("source_dft_estimator", False) for row in rows): reasons.append("source Section-III DFT estimator was not executed")
    if any(not row.get("warmup_epoch_excluded", False) or not row.get("source_epoch_150_normalization", False) for row in rows):
        reasons.append("source warm-up exclusion and epoch-150 normalization are absent")
    if any(row.get("spectral_aggregation") != "GEOMETRIC_MEAN" for row in rows): reasons.append("source geometric spectral aggregation is absent")
    if any(row.get("controller_mode") != "PAPER_DIRECT_SIGMA" or row.get("parameterization") != "direct_sigma" for row in rows):
        reasons.append("amended direct-sigma controller did not execute")
    source_ready = not reasons
    analysis = None
    if source_ready:
        config = SourceDFTConfig()
        analysis = analyze_traces([EvaluationTrace.from_mapping(row["trace"]) for row in rows], config)
    low_filter = None if analysis is None else float(np.median(np.asarray(analysis["raw_filter_db"])[:64]))
    return source_ready, reasons, {"median_suppression_db": None if low_filter is None else -low_filter,
        "source_filter_low_frequency_db_learned_over_fixed": low_filter,
        "paper_anchor_db": -4.0, "outcome": "SOURCE_SPECTRAL_ESTIMATOR_NOT_EXECUTED" if not source_ready else "SOURCE_DFT_COMPLETE",
        "source_dft_analysis": analysis, "source_structure_match": source_ready,
        "paper_comparable": False,
        "blocking_reasons": ["the experimental learned-mean/fixed LER traces are not publicly available"]}

def diagnostic_dft(rows: list[dict[str, Any]], *, warmup_epoch: int = 150, cadence: int = 5) -> dict[str, Any]:
    """Transparent DFT diagnostic for legacy proxy traces; never source evidence."""
    if not rows: raise ValueError("natural-drift diagnostic requires rows")
    run_filters=[]; learned_psd=[]; fixed_psd=[]; frequencies=None
    for row in rows:
        learned=np.asarray(row["trajectory"]["learned_mean"],dtype=float)
        fixed=np.asarray(row["trajectory"]["fixed_policy"],dtype=float)
        stop=min(len(learned),len(fixed)); indices=np.arange(warmup_epoch,stop,cadence)
        if len(indices)<8: raise ValueError("trace is too short for the declared warmup and DFT cadence")
        learned=learned[indices]/max(learned[indices][0],np.finfo(float).tiny)
        fixed=fixed[indices]/max(fixed[indices][0],np.finfo(float).tiny)
        frequency=np.fft.rfftfreq(len(indices),d=float(cadence))[1:]
        lp=np.abs(np.fft.rfft(learned))[1:]**2/len(indices)**2
        fp=np.abs(np.fft.rfft(fixed))[1:]**2/len(indices)**2
        if frequencies is not None and not np.array_equal(frequencies,frequency): raise ValueError("legacy traces lost a shared DFT grid")
        frequencies=frequency; learned_psd.append(lp); fixed_psd.append(fp)
        run_filters.append(10*np.log10(np.maximum(lp,np.finfo(float).tiny)/np.maximum(fp,np.finfo(float).tiny)))
    learned_geo=np.exp(np.mean(np.log(np.maximum(learned_psd,np.finfo(float).tiny)),axis=0))
    fixed_geo=np.exp(np.mean(np.log(np.maximum(fixed_psd,np.finfo(float).tiny)),axis=0))
    raw=10*np.log10(learned_geo/fixed_geo); kernel=np.ones(9)/9
    smooth=np.convolve(np.pad(raw,4,mode="edge"),kernel,mode="valid")
    intervals=np.quantile(np.asarray(run_filters),[.025,.975],axis=0)
    return {"frequency_per_epoch":frequencies.tolist(),"learned_geometric_psd":learned_geo.tolist(),
        "fixed_geometric_psd":fixed_geo.tolist(),"filter_db":raw.tolist(),"smoothed_filter_db":smooth.tolist(),
        "filter_db_interval_95":intervals.tolist(),"warmup_epoch":warmup_epoch,"cadence_epochs":cadence,
        "spectral_aggregation":"GEOMETRIC_MEAN","uncertainty":"RUN_QUANTILE_DIAGNOSTIC",
        "evidence_label":"LEGACY_LOGICAL_RISK_PROXY_DFT_DIAGNOSTIC_NOT_SOURCE_LER"}
