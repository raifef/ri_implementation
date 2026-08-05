"""Duration-first paired natural-drift spectral planning and analysis."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from google_rl_reimplementation.google_pure_evidence_v8.uncertainty import bootstrap_interval
from google_rl_reimplementation.google_pure_v7.config import canonical_hash
from google_rl_reimplementation.google_pure_v7.natural import FAMILIES, FrozenNaturalPlant, generate_natural_drift
from google_rl_reimplementation.google_pure_v6.experiments import run_matched_trace
from google_rl_reimplementation.google_pure_v9.contracts import ControllerConfig
from google_rl_reimplementation.google_pure_v9.common import guard_seed

from .common import artifact_root, load_config, write_artifact
from .contracts import ExperimentFamily, evidence_envelope, hash_analysis_contract, validate_provenance


class InsufficientSpectralDuration(ValueError):
    """Raised before acquisition when the low-frequency claim is not resolvable."""


@dataclass(frozen=True)
class SpectralPlan:
    duration_epochs: int
    segment_length: int
    overlap_fraction: float
    low_frequency_band: tuple[float, float]
    minimum_low_frequency_bins: int
    minimum_welch_segments: int
    detrending: str = "constant"
    window: str = "hann"

    def __post_init__(self) -> None:
        if self.duration_epochs < 8 or not 4 <= self.segment_length <= self.duration_epochs:
            raise InsufficientSpectralDuration("INSUFFICIENT_DURATION_FOR_LOW_FREQUENCY_CLAIM")
        if not 0 <= self.overlap_fraction < 1:
            raise ValueError("Welch overlap must lie in [0,1)")
        low, high = self.low_frequency_band
        if not 0 < low < high < 0.5:
            raise ValueError("low-frequency band must lie below Nyquist")
        if self.detrending not in {"constant", "linear"} or self.window not in {"hann", "boxcar"}:
            raise ValueError("unsupported frozen PSD estimator")
        diagnostics = self.diagnostics()
        if (
            diagnostics["frequency_resolution_from_duration"] > (high - low) / 4
            or diagnostics["number_of_low_frequency_bins"] < self.minimum_low_frequency_bins
            or diagnostics["number_of_independent_segments"] < self.minimum_welch_segments
        ):
            raise InsufficientSpectralDuration("INSUFFICIENT_DURATION_FOR_LOW_FREQUENCY_CLAIM")

    def diagnostics(self) -> dict[str, Any]:
        step = max(1, int(round(self.segment_length * (1 - self.overlap_fraction))))
        segments = 1 + (self.duration_epochs - self.segment_length) // step
        frequency = np.fft.rfftfreq(self.segment_length)
        low, high = self.low_frequency_band
        bins = int(np.sum((frequency >= low) & (frequency < high)))
        duration_modes = int(np.floor(self.duration_epochs * (high - low)))
        return {
            "target_low_frequency_band": list(self.low_frequency_band),
            "duration_epochs": self.duration_epochs,
            "frequency_resolution_from_duration": 1.0 / self.duration_epochs,
            "welch_bin_width": 1.0 / self.segment_length,
            "number_of_low_frequency_bins": bins,
            "number_of_independent_low_frequency_modes": duration_modes,
            "welch_segment_length": self.segment_length,
            "welch_overlap": self.overlap_fraction,
            "number_of_independent_segments": segments,
            "estimated_uncertainty": "complete plant/seed runs are resampling units; finite Welch segments reported",
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _plan_from_config(mode: str) -> tuple[SpectralPlan, dict[str, Any]]:
    if mode not in {"smoke", "reference"}:
        raise ValueError("natural-drift mode must be smoke or reference")
    config = load_config("natural_drift.json")
    row = config[mode]
    plan = SpectralPlan(
        duration_epochs=int(row["duration_epochs"]),
        segment_length=int(row["segment_length"]),
        overlap_fraction=float(config["analysis_contract"]["overlap_fraction"]),
        low_frequency_band=tuple(map(float, row["low_frequency_band"])),
        minimum_low_frequency_bins=int(row["minimum_low_frequency_bins"]),
        minimum_welch_segments=int(row["minimum_welch_segments"]),
        detrending=str(config["analysis_contract"]["detrending"]),
        window=str(config["analysis_contract"]["window"]),
    )
    return plan, {**row, "analysis_contract": config["analysis_contract"]}


def plan_natural_drift(mode: str = "smoke") -> dict[str, Any]:
    plan, config = _plan_from_config(mode)
    diagnostics = plan.diagnostics()
    payload = {
        "schema_version": "google-pure-v10-natural-plan.v1",
        "experiment_family": ExperimentFamily.NATURAL_DRIFT_SPECTRAL_SUPPRESSION_V10.value,
        "mode": mode,
        **diagnostics,
        "plant_indices": config["plant_indices"],
        "runs": len(config["plant_indices"]),
        "candidates": config["candidates"],
        "cycles_per_candidate": config["cycles_per_candidate"],
        "estimated_qec_cycles": len(config["plant_indices"]) * plan.duration_epochs * int(config["candidates"]) * int(config["cycles_per_candidate"]),
        "estimated_runtime": "under two minutes smoke; long explicit user-run reference acquisition",
        "estimated_memory_storage": "raw four-policy traces plus paired PSD arrays; under 50 MiB smoke",
        "analysis_contract": config["analysis_contract"],
        "protocol_hash": canonical_hash({"plan": plan.to_dict(), "config": config}),
        **evidence_envelope(complete=True, mechanism_valid=True, claim_supported=False, paper_comparable=False, blocking_reasons=["ACQUISITION_NOT_EXECUTED"]),
    }
    return write_artifact("natural_drift/run_plan", payload, "Natural-drift Spectral Run Plan")


def welch_psd(values: np.ndarray, plan: SpectralPlan) -> tuple[np.ndarray, np.ndarray, int]:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < plan.segment_length or not np.all(np.isfinite(x)):
        raise ValueError("finite scalar trace longer than one segment is required")
    step = max(1, int(round(plan.segment_length * (1 - plan.overlap_fraction))))
    taper = np.hanning(plan.segment_length) if plan.window == "hann" else np.ones(plan.segment_length)
    time = np.arange(plan.segment_length, dtype=float)
    powers = []
    for start in range(0, len(x) - plan.segment_length + 1, step):
        segment = x[start:start + plan.segment_length].copy()
        if plan.detrending == "constant":
            segment -= np.mean(segment)
        else:
            segment -= np.polyval(np.polyfit(time, segment, 1), time)
        power = np.abs(np.fft.rfft(segment * taper)) ** 2 / np.sum(taper ** 2)
        if len(power) > 2:
            power[1:-1] *= 2
        powers.append(power)
    if len(powers) < plan.minimum_welch_segments:
        raise InsufficientSpectralDuration("INSUFFICIENT_DURATION_FOR_LOW_FREQUENCY_CLAIM")
    return np.fft.rfftfreq(plan.segment_length), np.mean(powers, axis=0), len(powers)


def integrated_band_power(frequency: np.ndarray, power: np.ndarray, band: tuple[float, float]) -> float:
    frequency = np.asarray(frequency, dtype=float)
    power = np.asarray(power, dtype=float)
    selected = (frequency >= band[0]) & (frequency < band[1])
    if np.count_nonzero(selected) < 1 or len(frequency) < 2:
        raise InsufficientSpectralDuration("INSUFFICIENT_DURATION_FOR_LOW_FREQUENCY_CLAIM")
    return float(np.sum(power[selected]) * (frequency[1] - frequency[0]))


def positive_suppression_db(fixed_power: float, policy_power: float) -> float:
    if fixed_power <= 0 or policy_power <= 0:
        raise ValueError("band powers must be positive")
    return float(10 * np.log10(fixed_power / policy_power))


def analyse_policy_traces(traces: Mapping[str, np.ndarray], plan: SpectralPlan) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    required = {"fixed_policy", "learned_mean", "stochastic_candidates", "oracle_optimum"}
    if set(traces) != required:
        raise ValueError("paired spectral analysis requires fixed, mean, candidate, and oracle traces")
    frequency = None
    powers: dict[str, np.ndarray] = {}
    segments = 0
    for name in sorted(required):
        current_frequency, current_power, segments = welch_psd(np.asarray(traces[name]), plan)
        if frequency is not None and not np.array_equal(frequency, current_frequency):
            raise RuntimeError("paired policy PSD grids differ")
        frequency = current_frequency
        powers[name] = current_power
    assert frequency is not None
    band = plan.low_frequency_band
    band_powers = {name: integrated_band_power(frequency, power, band) for name, power in powers.items()}
    row = {
        "band_power": band_powers,
        "mean_suppression_db": positive_suppression_db(band_powers["fixed_policy"], band_powers["learned_mean"]),
        "candidate_suppression_db": positive_suppression_db(band_powers["fixed_policy"], band_powers["stochastic_candidates"]),
        "oracle_suppression_db": (
            positive_suppression_db(band_powers["fixed_policy"], band_powers["oracle_optimum"])
            if band_powers["oracle_optimum"] > 0
            else None
        ),
        "oracle_zero_band_power_after_detrending": band_powers["oracle_optimum"] <= 0,
        "positive_means_suppression": True,
        "welch_segments": segments,
        **plan.diagnostics(),
    }
    return row, {"frequency": frequency, **powers}


def _controller() -> ControllerConfig:
    return ControllerConfig(
        initial_scale=0.04,
        minimum_scale=0.001,
        maximum_scale=0.25,
        scale_learning_rate=0.01,
        entropy_coefficient=0.01,
        mean_learning_rate=0.02,
    )


def run_natural_drift(*, mode: str = "smoke", execute: bool = False) -> dict[str, Any]:
    if mode == "reference" and not execute:
        raise RuntimeError("reference natural-drift acquisition requires --execute")
    run_plan = plan_natural_drift(mode)
    plan, config = _plan_from_config(mode)
    controller = _controller()
    all_traces: dict[str, np.ndarray] = {}
    all_psd: dict[str, np.ndarray] = {}
    rows = []
    sensitivity = []
    for index in map(int, config["plant_indices"]):
        family = FAMILIES[index]
        plant = FrozenNaturalPlant(index, v5_compatible_units=False)
        tape = generate_natural_drift(family, plan.duration_epochs)
        seed = 21101 + index
        guard_seed(seed)
        result = run_matched_trace(
            plant,
            tape,
            controller.to_agent_choices(),
            seed=seed,
            candidates=int(config["candidates"]),
            cycles=int(config["cycles_per_candidate"]),
            objective_mode="source_literal_ppo",
        )
        traces = {name: np.asarray(values) for name, values in result["logical_risk"].items()}
        spectral, psd = analyse_policy_traces(traces, plan)
        key = str(family["id"])
        provenance = {
            "experiment_family": ExperimentFamily.NATURAL_DRIFT_SPECTRAL_SUPPRESSION_V10.value,
            "controller_hash": canonical_hash(controller.to_dict()),
            "decoder_hash": None,
            "plant_hash": canonical_hash({"family": family, "curvature": plant.curvature.tolist(), "floors": plant.floors.tolist()}),
            "graph_hash": canonical_hash(plant.mask.tolist()),
            "protocol_hash": run_plan["protocol_hash"],
            "seed": seed,
            "drift_tape_hash": canonical_hash(tape.tolist()),
            "mode": mode,
            "qec_cycle_budget": plan.duration_epochs * int(config["candidates"]) * int(config["cycles_per_candidate"]),
            "candidate_budget": plan.duration_epochs * int(config["candidates"]),
            "observable_definition": "paired logical-risk traces; positive dB is fixed band power divided by policy band power",
            "analysis_contract": config["analysis_contract"],
        }
        validate_provenance(provenance)
        rows.append({"plant_id": key, **provenance, **spectral})
        for name, values in traces.items():
            all_traces[f"{key}__{name}"] = values
        for name, values in psd.items():
            all_psd[f"{key}__{name}"] = values
        for detrending in ("constant", "linear"):
            for window in ("hann", "boxcar"):
                alternate = SpectralPlan(
                    duration_epochs=plan.duration_epochs,
                    segment_length=plan.segment_length,
                    overlap_fraction=plan.overlap_fraction,
                    low_frequency_band=plan.low_frequency_band,
                    minimum_low_frequency_bins=plan.minimum_low_frequency_bins,
                    minimum_welch_segments=plan.minimum_welch_segments,
                    detrending=detrending,
                    window=window,
                )
                alternate_row, _ = analyse_policy_traces(traces, alternate)
                sensitivity.append(
                    {
                        "plant_id": key,
                        "detrending": detrending,
                        "window": window,
                        "mean_suppression_db": alternate_row["mean_suppression_db"],
                        "candidate_suppression_db": alternate_row["candidate_suppression_db"],
                    }
                )
    target = artifact_root() / "natural_drift"
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target / "raw_traces.npz", **all_traces)
    np.savez_compressed(target / "psd_results.npz", **all_psd)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    first = rows[0]["plant_id"]
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7), constrained_layout=True)
    for name, label in (("fixed_policy", "fixed"), ("learned_mean", "mean"), ("stochastic_candidates", "candidate"), ("oracle_optimum", "oracle")):
        axes[0].plot(all_traces[f"{first}__{name}"], label=label, alpha=0.85)
    axes[0].legend()
    axes[0].set(title="Matched natural-drift policy streams", ylabel="logical-risk proxy")
    frequency = all_psd[f"{first}__frequency"]
    axes[1].loglog(frequency[1:], all_psd[f"{first}__fixed_policy"][1:], label="fixed")
    axes[1].loglog(frequency[1:], all_psd[f"{first}__learned_mean"][1:], label="mean")
    axes[1].axvspan(*plan.low_frequency_band, alpha=0.2)
    axes[1].legend()
    axes[1].set(xlabel="cycles per epoch", ylabel="PSD", title="Frozen Welch estimator and low-frequency band")
    fig.savefig(target / "figure.png", dpi=170)
    plt.close(fig)
    mean_values = [row["mean_suppression_db"] for row in rows]
    candidate_values = [row["candidate_suppression_db"] for row in rows]
    reference = mode == "reference" and len(rows) == 6
    blockers = [] if reference else ["SMOKE_NOT_REFERENCE_EVIDENCE", "FULL_SIX_PLANT_UNCERTAINTY_NOT_ACQUIRED"]
    payload = {
        "schema_version": "google-pure-v10-natural-results.v1",
        "experiment_family": ExperimentFamily.NATURAL_DRIFT_SPECTRAL_SUPPRESSION_V10.value,
        "mode": mode,
        "run_plan_hash": run_plan["artifact_hash"],
        "analysis_contract_hash": hash_analysis_contract(config["analysis_contract"]),
        "rows": rows,
        "sensitivity_records": sensitivity,
        "median_mean_suppression_db": float(np.median(mean_values)),
        "median_candidate_suppression_db": float(np.median(candidate_values)),
        "mean_suppression_ci_95": bootstrap_interval(mean_values),
        "candidate_suppression_ci_95": bootstrap_interval(candidate_values),
        "complete_plant_seed_runs_are_uncertainty_units": True,
        "raw_trace_file": "raw_traces.npz",
        "psd_file": "psd_results.npz",
        "figure_file": "figure.png",
        **evidence_envelope(
            complete=True,
            mechanism_valid=True,
            claim_supported=reference,
            paper_comparable=False,
            blocking_reasons=blockers,
        ),
    }
    return write_artifact("natural_drift/report", payload, "Natural-drift Spectral Suppression", markdown_relative="natural_drift/report.md")
