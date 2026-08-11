"""Dedicated unlabelled natural-drift low-frequency spectral experiment."""
from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np

from .accounting import acquisition_accounting
from .config import load_config, paper_scale, source_choices
from .experiments import percentile_interval, run_matched_trace
from .plant import PurePlantSpec, PureQuadraticPlant
from .reporting import write_report


def generate_natural_drift(
    family: Mapping[str, Any], horizon: int, control_count: int
) -> np.ndarray:
    """Frozen slow process ensemble with no labelled injected intervention."""
    rng = np.random.default_rng(int(family["seed"]))
    t = np.arange(horizon, dtype=float)
    amplitude = float(family["amplitude"])
    name = str(family["family"])
    if name == "slow_instrumental":
        scalar = 0.62 * np.sin(2 * np.pi * t / 510.0 + 0.3) + 0.38 * np.sin(2 * np.pi * t / 290.0 + 1.7)
    elif name == "smooth_common_mode":
        scalar = 0.70 * np.sin(2 * np.pi * t / 430.0 + 0.8) + 0.30 * np.cos(2 * np.pi * t / 210.0 + 0.2)
    elif name == "bounded_multi_sine":
        frequencies = rng.uniform(1.0 / 650.0, 1.0 / 120.0, 5)
        phases = rng.uniform(0.0, 2.0 * np.pi, 5)
        weights = rng.uniform(0.4, 1.0, 5)
        scalar = sum(w * np.sin(2 * np.pi * f * t + p) for w, f, p in zip(weights, frequencies, phases))
        scalar /= max(np.max(np.abs(scalar)), 1e-12)
    elif name == "low_frequency_coloured":
        frequencies = np.fft.rfftfreq(horizon)
        spectrum = np.zeros(len(frequencies), dtype=complex)
        positive = frequencies > 0
        cutoff = frequencies <= 0.02
        active = positive & cutoff
        spectrum[active] = (rng.normal(size=active.sum()) + 1j * rng.normal(size=active.sum())) / np.maximum(frequencies[active], 1.0 / horizon)
        scalar = np.fft.irfft(spectrum, n=horizon)
        scalar /= max(np.max(np.abs(scalar)), 1e-12)
    else:
        raise ValueError("unknown natural-drift family")
    tape = np.zeros((horizon, control_count), dtype=float)
    stride = int(family["affected_stride"])
    indices = np.arange(0, control_count, stride)
    phases = rng.uniform(0.75, 1.25, len(indices))
    signs = np.where(np.arange(len(indices)) % 2 == 0, 1.0, -1.0)
    tape[:, indices] = amplitude * scalar[:, None] * phases[None, :] * signs[None, :]
    return np.clip(tape, -0.5, 0.5)


def _detrend(values: np.ndarray, kind: str) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if kind == "constant":
        return x - x.mean()
    if kind == "linear":
        t = np.arange(len(x), dtype=float)
        slope, intercept = np.polyfit(t, x, 1)
        return x - (slope * t + intercept)
    raise ValueError("unknown detrending choice")


def welch_psd(
    values: np.ndarray,
    *,
    segment_length: int,
    overlap_fraction: float,
    taper: str,
    detrend: str,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=float)
    if segment_length > len(x) or segment_length < 16:
        raise ValueError("invalid Welch segment length")
    step = max(1, int(round(segment_length * (1.0 - overlap_fraction))))
    window = np.hanning(segment_length) if taper == "hann" else np.ones(segment_length)
    scale = float(np.sum(window * window))
    spectra = []
    for start in range(0, len(x) - segment_length + 1, step):
        segment = _detrend(x[start : start + segment_length], detrend) * window
        power = np.abs(np.fft.rfft(segment)) ** 2 / scale
        power[1:-1] *= 2.0
        spectra.append(power)
    if not spectra:
        raise ValueError("no complete Welch segment")
    return np.fft.rfftfreq(segment_length), np.mean(spectra, axis=0)


def _band_power(frequency: np.ndarray, power: np.ndarray, band: tuple[float, float]) -> float:
    selected = (frequency >= band[0]) & (frequency < band[1])
    if not selected.any():
        raise ValueError("spectral band contains no Fourier bin")
    return float(np.sum(power[selected]))


def _spectral_metrics(
    traces: Mapping[str, np.ndarray], config: Mapping[str, Any], *,
    segment_length: int | None = None, taper: str | None = None,
    low_upper: float | None = None, detrend: str | None = None,
) -> dict[str, Any]:
    segment = int(config["welch_segment_length"] if segment_length is None else segment_length)
    taper_name = str(config["taper"] if taper is None else taper)
    detrend_name = str(config["detrend"] if detrend is None else detrend)
    low = tuple(map(float, config["low_frequency_band"]))
    if low_upper is not None:
        low = (low[0], float(low_upper))
    mid = tuple(map(float, config["mid_frequency_band"]))
    high = tuple(map(float, config["high_frequency_band"]))
    psd = {}
    band_values = {}
    for policy_name, values in traces.items():
        frequency, power = welch_psd(
            values, segment_length=segment, overlap_fraction=float(config["welch_overlap_fraction"]),
            taper=taper_name, detrend=detrend_name,
        )
        psd[policy_name] = power
        band_values[policy_name] = {
            "low": _band_power(frequency, power, low),
            "mid": _band_power(frequency, power, mid),
            "high": _band_power(frequency, power, high),
            "total_variance": float(np.var(values, ddof=1)),
        }
    fixed = band_values["fixed_policy"]["low"]
    learned = band_values["learned_mean"]["low"]
    stochastic = band_values["stochastic_candidates"]["low"]
    return {
        "low_frequency_gain_db": float(10.0 * np.log10(max(fixed, 1e-30) / max(learned, 1e-30))),
        "stochastic_low_frequency_gain_db": float(10.0 * np.log10(max(fixed, 1e-30) / max(stochastic, 1e-30))),
        "band_power": band_values,
        "estimator": {"segment_length": segment, "taper": taper_name, "detrend": detrend_name, "low_band": list(low)},
    }


def run_natural_drift_spectral(epochs: int | None = None) -> dict[str, Any]:
    config = load_config("natural_drift_spectral.yaml")
    horizon = int(config["horizon_epochs"] if epochs is None else epochs)
    if horizon < 256:
        raise ValueError("natural spectral development requires at least 256 epochs")
    choices, paper = source_choices(), paper_scale()
    plants = []
    raw_hashes = []
    for index, family in enumerate(config["families"]):
        spec = PurePlantSpec(
            str(family["id"]), detector_count=int(config["detectors"]), control_count=int(config["controls"]),
            curvature=float(config["curvature"]), detector_floor=float(config["detector_floor"]),
            logical_floor=float(config["logical_floor"]), logical_gain=float(config["logical_gain"]),
            draw_seed=7600 + index,
        )
        plant = PureQuadraticPlant(spec)
        tape = generate_natural_drift(family, horizon, spec.control_count)
        raw_hash = hashlib.sha256(np.asarray(tape, dtype="<f8").tobytes()).hexdigest()
        raw_hashes.append(raw_hash)
        result = run_matched_trace(plant, tape, choices, paper, seed=int(family["seed"]))
        primary = _spectral_metrics(result["logical_risk"], config)
        noise_floor = float(
            config["logical_gain"] ** 2 * config["detector_floor"] * (1.0 - config["detector_floor"])
            / (int(config["detectors"]) * int(paper["effective_cycles_per_candidate"]))
        )
        plants.append({
            "plant_id": family["id"], "family": family["family"], "raw_trace_hash": raw_hash,
            "low_frequency_gain_db": primary["low_frequency_gain_db"],
            "stochastic_low_frequency_gain_db": primary["stochastic_low_frequency_gain_db"],
            "band_power": primary["band_power"], "total_variance": {name: float(np.var(values, ddof=1)) for name, values in result["logical_risk"].items()},
            "measurement_noise_floor_variance": noise_floor,
        })
    gains = [float(row["low_frequency_gain_db"]) for row in plants]
    robustness = []
    # Sensitivity checks change one frozen convention at a time, never select a winner.
    reference_family = config["families"][0]
    reference_plant = PureQuadraticPlant(PurePlantSpec("robustness-reference", draw_seed=7699))
    reference_tape = generate_natural_drift(reference_family, horizon, reference_plant.spec.control_count)
    reference_result = run_matched_trace(reference_plant, reference_tape, choices, paper, seed=7399)
    for segment in config["robustness"]["segment_lengths"]:
        if int(segment) <= horizon:
            metric = _spectral_metrics(reference_result["logical_risk"], config, segment_length=int(segment))
            robustness.append({"dimension": "segment_length", "value": segment, "gain_db": metric["low_frequency_gain_db"]})
    for taper in config["robustness"]["tapers"]:
        metric = _spectral_metrics(reference_result["logical_risk"], config, taper=str(taper))
        robustness.append({"dimension": "taper", "value": taper, "gain_db": metric["low_frequency_gain_db"]})
    for edge in config["robustness"]["low_band_upper_edges"]:
        metric = _spectral_metrics(reference_result["logical_risk"], config, low_upper=float(edge))
        robustness.append({"dimension": "low_band_upper_edge", "value": edge, "gain_db": metric["low_frequency_gain_db"]})
    for detrending in config["robustness"]["detrending_choices"]:
        metric = _spectral_metrics(reference_result["logical_risk"], config, detrend=str(detrending))
        robustness.append({"dimension": "detrending", "value": detrending, "gain_db": metric["low_frequency_gain_db"]})
    median_gain = float(np.median(gains))
    band = list(map(float, config["low_frequency_gain_db_band"]))
    checks = {
        "all_primary_families_positive_lf_suppression": all(value > 0.0 for value in gains),
        "aggregate_compatible_with_four_db": band[0] <= median_gain <= band[1],
        "robust_to_all_declared_psd_conventions": all(float(row["gain_db"]) > 0.0 for row in robustness),
        "all_bands_reported": all(set(row["band_power"]["learned_mean"]) == {"low", "mid", "high", "total_variance"} for row in plants),
        "stochastic_spectral_performance_separate": all("stochastic_low_frequency_gain_db" in row for row in plants),
    }
    accounting = acquisition_accounting(horizon * (len(plants) + 1), paper, mean_evaluations=horizon * len(plants), fixed_evaluations=horizon * len(plants), logical_evaluations=4 * horizon * len(plants))
    payload = {
        "schema_version": "google-pure-v5-natural-drift-spectral.v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "experiment_kind": "unlabelled frozen natural slow-drift ensemble; no injected profiles",
        "primary_metric": "10*log10(integrated fixed-policy LF power / learned-mean LF power)",
        "aggregate": {"median_low_frequency_gain_db": median_gain, "low_frequency_gain_95_percent_interval_across_plants": percentile_interval(gains)},
        "checks": checks,
        "plants": plants,
        "spectral_robustness": robustness,
        "raw_trace_hashes": raw_hashes,
        "accounting": accounting,
        "decoder_steering_included": False,
        "certification_seeds_consumed": False,
    }
    write_report("natural_drift_spectral", payload, "Natural-drift spectral suppression")
    return payload
