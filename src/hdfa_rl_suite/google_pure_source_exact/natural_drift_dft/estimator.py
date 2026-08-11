"""Literal Section-III DFT, shared-grid geometric averaging, and filter ratio."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .contracts import EvaluationTrace, SourceDFTConfig, canonical_hash


def preprocess_trace(trace: EvaluationTrace, config: SourceDFTConfig) -> dict[str, np.ndarray]:
    trace.validate(config)
    epochs = np.asarray(trace.epochs, dtype=int)
    selected = epochs >= config.warmup_epoch
    selected_epochs = epochs[selected]
    learned = np.asarray(trace.learned_mean_ler, dtype=float)[selected]
    fixed = np.asarray(trace.fixed_initial_ler, dtype=float)[selected]
    if selected_epochs[0] != config.warmup_epoch:
        raise AssertionError("warmup selection lost the normalization epoch")
    return {"epochs": selected_epochs,
            "learned": learned / learned[0], "fixed": fixed / fixed[0]}


def run_spectrum(values: np.ndarray, *, cadence_epochs: int) -> tuple[np.ndarray, np.ndarray]:
    trace = np.asarray(values, dtype=float)
    if trace.ndim != 1 or len(trace) < 3 or np.any(~np.isfinite(trace)):
        raise ValueError("DFT input must be a finite one-dimensional trace")
    transform = np.fft.rfft(trace)
    frequency = np.fft.rfftfreq(len(trace), d=float(cadence_epochs))
    power = np.abs(transform) ** 2 / float(len(trace) ** 2)
    return frequency[1:], power[1:]


def _geometric_mean(values: np.ndarray) -> np.ndarray:
    safe = np.maximum(np.asarray(values, dtype=float), np.finfo(float).tiny)
    return np.exp(np.mean(np.log(safe), axis=0))


def _gaussian_smooth(values: np.ndarray, sigma_bins: float) -> np.ndarray:
    radius = max(1, int(np.ceil(4.0 * sigma_bins)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(np.asarray(values, dtype=float), radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def analyze_traces(traces: Sequence[EvaluationTrace], config: SourceDFTConfig) -> dict[str, Any]:
    if len(traces) < 2:
        raise ValueError("multiple independent complete runs are required")
    if len({trace.run_id for trace in traces}) != len(traces):
        raise ValueError("duplicate run identifier rejected")
    spectra = []
    normalization = []
    for trace in traces:
        prepared = preprocess_trace(trace, config)
        learned_f, learned_p = run_spectrum(prepared["learned"], cadence_epochs=config.cadence_epochs)
        fixed_f, fixed_p = run_spectrum(prepared["fixed"], cadence_epochs=config.cadence_epochs)
        if not np.array_equal(learned_f, fixed_f):
            raise AssertionError("paired policies lost their shared per-run frequency grid")
        spectra.append((learned_f, learned_p, fixed_p))
        normalization.append({"run_id": trace.run_id, "epoch_150_learned": float(prepared["learned"][0]),
                              "epoch_150_fixed": float(prepared["fixed"][0]),
                              "post_warmup_points": len(prepared["epochs"])})
    lower = max(float(row[0][0]) for row in spectra)
    upper = min(float(row[0][-1]) for row in spectra)
    if not lower < upper:
        raise ValueError("unequal run spectra have no shared positive-frequency support")
    shared = np.geomspace(lower, upper, config.shared_grid_points)
    learned_rows = np.vstack([np.interp(shared, freq, learned) for freq, learned, _ in spectra])
    fixed_rows = np.vstack([np.interp(shared, freq, fixed) for freq, _, fixed in spectra])
    learned_geo, fixed_geo = _geometric_mean(learned_rows), _geometric_mean(fixed_rows)
    raw_db = 10.0 * np.log10(learned_geo / fixed_geo)
    smooth_db = _gaussian_smooth(raw_db, config.gaussian_smoothing_sigma_bins)
    result = {"schema_version": "natural-drift-dft-analysis.v1",
              "estimator": "SOURCE_SECTION_III_DFT", "time_coordinate": "EPOCH",
              "cadence_epochs": config.cadence_epochs, "warmup_epoch": config.warmup_epoch,
              "zero_frequency_excluded": True, "spectral_aggregation": "GEOMETRIC_MEAN",
              "filter_db_definition": "10*log10(learned_mean_policy_psd/fixed_initial_policy_psd)",
              "source_panel_uses_welch": False, "source_panel_uses_candidate_stream": False,
              "run_ids": [trace.run_id for trace in traces], "normalization_checks": normalization,
              "frequency_per_epoch": shared.tolist(), "learned_geometric_psd": learned_geo.tolist(),
              "fixed_geometric_psd": fixed_geo.tolist(), "raw_filter_db": raw_db.tolist(),
              "smoothed_guide_to_eye_db": smooth_db.tolist(),
              "input_trace_hash": canonical_hash([trace.to_dict() for trace in traces])}
    result["analysis_hash"] = canonical_hash(result)
    return result
