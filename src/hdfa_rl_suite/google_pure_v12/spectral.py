"""Natural-drift sign convention and run-level spectral uncertainty."""
from __future__ import annotations

from typing import Any

import numpy as np

from hdfa_rl_suite.google_pure_source_exact.natural_drift_dft.contracts import EvaluationTrace, SourceDFTConfig
from hdfa_rl_suite.google_pure_source_exact.natural_drift_dft.estimator import preprocess_trace, run_spectrum

from .contracts import NONFINAL_FIELDS, V12_SCHEMA
from .io import ARTIFACT_ROOT, ROOT, atomic_json, atomic_text, load_config, read_json


def power_ratio_db(learned_power: np.ndarray | float,
                   fixed_power: np.ndarray | float) -> np.ndarray:
    learned = np.asarray(learned_power, dtype=float)
    fixed = np.asarray(fixed_power, dtype=float)
    if np.any(learned <= 0) or np.any(fixed <= 0):
        raise ValueError("spectral powers must be strictly positive")
    return 10.0 * np.log10(learned / fixed)


def validate_natural_drift_sign() -> dict[str, Any]:
    expected = 10.0 * np.log10(2.0)
    cases = {
        "equal_power_db": float(power_ratio_db(1.0, 1.0)),
        "half_learned_power_db": float(power_ratio_db(.5, 1.0)),
        "double_learned_power_db": float(power_ratio_db(2.0, 1.0)),
    }
    passed = (abs(cases["equal_power_db"]) < 1e-14 and
              abs(cases["half_learned_power_db"] + expected) < 1e-12 and
              abs(cases["double_learned_power_db"] - expected) < 1e-12)
    result = {"schema_version": V12_SCHEMA, "convention": "10*log10(P_learned/P_fixed)",
              "negative_means": "suppression", "positive_means": "amplification",
              "cases": cases, "pass": bool(passed), **NONFINAL_FIELDS}
    atomic_json(ARTIFACT_ROOT / "spectral/sign_validation.json", result)
    if not passed:
        raise RuntimeError("natural-drift power-ratio sign validation failed")
    return result


def _natural_rows() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    family = "natural_drift_spectral_suppression"
    protocol = read_json(ROOT / f"artifacts/google_pure_paper_reproduction/experiment_protocols/{family}_validation.json")
    path = (ROOT / "artifacts/google_pure_paper_reproduction/synthetic_reproduction/natural" /
            protocol["protocol_hash"][:16] / "merged.json")
    merged = read_json(path)
    if not merged.get("complete") or merged["protocol_hash"] != protocol["protocol_hash"]:
        raise RuntimeError("natural-drift lineage is incomplete or stale")
    return protocol, merged["rows"]


def analyse_natural_drift_uncertainty() -> dict[str, Any]:
    protocol, rows = _natural_rows()
    config = load_config()
    bins = int(config["natural_drift"]["low_frequency_bins"])
    repetitions = int(config["audit"]["bootstrap_repetitions"])
    dft_config = SourceDFTConfig()
    spectra = []
    for row in rows:
        prepared = preprocess_trace(EvaluationTrace.from_mapping(row["trace"]), dft_config)
        learned_frequency, learned_power = run_spectrum(prepared["learned"], cadence_epochs=dft_config.cadence_epochs)
        fixed_frequency, fixed_power = run_spectrum(prepared["fixed"], cadence_epochs=dft_config.cadence_epochs)
        if not np.array_equal(learned_frequency, fixed_frequency):
            raise RuntimeError("paired natural-drift policies lost a shared frequency grid")
        spectra.append((learned_frequency, learned_power, fixed_power))
    lower = max(float(row[0][0]) for row in spectra)
    upper = min(float(row[0][-1]) for row in spectra)
    frequency = np.geomspace(lower, upper, dft_config.shared_grid_points)
    learned = np.vstack([np.interp(frequency, raw_frequency, power)
                         for raw_frequency, power, _ in spectra])
    fixed = np.vstack([np.interp(frequency, raw_frequency, power)
                       for raw_frequency, _, power in spectra])
    low = min(bins, learned.shape[1])
    run_filter = power_ratio_db(learned, fixed)
    run_low = np.median(run_filter[:, :low], axis=1)
    learned_geo = np.exp(np.mean(np.log(learned), axis=0))
    fixed_geo = np.exp(np.mean(np.log(fixed), axis=0))
    aggregate_filter = power_ratio_db(learned_geo, fixed_geo)
    aggregate_low = float(np.median(aggregate_filter[:low]))
    rng = np.random.default_rng(int(config["audit"]["bootstrap_seed"]))
    boot_low = np.empty(repetitions)
    boot_filter = np.empty((repetitions, learned.shape[1]))
    for index in range(repetitions):
        sample = rng.integers(0, len(rows), len(rows))
        learned_sample = np.exp(np.mean(np.log(learned[sample]), axis=0))
        fixed_sample = np.exp(np.mean(np.log(fixed[sample]), axis=0))
        boot_filter[index] = power_ratio_db(learned_sample, fixed_sample)
        boot_low[index] = np.median(boot_filter[index, :low])
    interval = np.quantile(boot_low, [.025, .975])
    point_interval = np.quantile(boot_filter, [.025, .975], axis=0)
    strong = float(config["natural_drift"]["strong_effect_db"])
    spans_strong_both = bool(interval[0] <= -strong and interval[1] >= strong)
    crosses_zero = bool(interval[0] <= 0 <= interval[1])
    classification = "UNRESOLVED_STRONG_AMPLIFICATION_AND_SUPPRESSION" if spans_strong_both else (
        "UNRESOLVED_SIGN" if crosses_zero else ("SUPPRESSION" if interval[1] < 0 else "AMPLIFICATION"))
    result = {"schema_version": V12_SCHEMA, "protocol_hash": protocol["protocol_hash"],
              "run_count": len(rows), "run_ids": [row["plant_id"] for row in rows],
              "resampling_unit": "run_id", "frequency_bins_are_replicates": False,
              "bootstrap_repetitions": repetitions,
              "power_db_convention": "10*log10(P_learned/P_fixed)",
              "negative_means": "suppression", "warmup_epoch_excluded": all(row["warmup_epoch_excluded"] for row in rows),
              "source_epoch_150_normalization": all(row["source_epoch_150_normalization"] for row in rows),
              "spectral_aggregation": "GEOMETRIC_MEAN_ACROSS_RUNS",
              "frequency_per_epoch": frequency.tolist(),
              "learned_geometric_power": learned_geo.tolist(), "fixed_geometric_power": fixed_geo.tolist(),
              "filter_db_learned_over_fixed": aggregate_filter.tolist(),
              "filter_db_run_bootstrap_interval_95": point_interval.tolist(),
              "run_low_frequency_filter_db": run_low.tolist(),
              "aggregate_low_frequency_filter_db": aggregate_low,
              "aggregate_low_frequency_filter_db_interval_95": interval.tolist(),
              "direction_identifiable": not crosses_zero, "classification": classification,
              "paper_anchor_db": -4.0, **NONFINAL_FIELDS}
    atomic_json(ARTIFACT_ROOT / "spectral/natural_drift_uncertainty.json", result)
    _plot_spectral(result)
    atomic_text(ARTIFACT_ROOT / "spectral/natural_drift_uncertainty.md",
                "# Natural-drift spectral uncertainty\n\n"
                f"Low-frequency learned/fixed result: **{aggregate_low:.3f} dB** "
                f"(run-bootstrap 95% interval {interval[0]:.3f} to {interval[1]:.3f} dB).\n\n"
                f"Classification: **{classification}**. Runs—not frequency bins—are the uncertainty replicates. "
                "Negative values mean suppression; this validation artifact is not paper-equivalence evidence.")
    return result


def _plot_spectral(result: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frequency = np.asarray(result["frequency_per_epoch"])
    learned = np.asarray(result["learned_geometric_power"])
    fixed = np.asarray(result["fixed_geometric_power"])
    filter_db = np.asarray(result["filter_db_learned_over_fixed"])
    interval = np.asarray(result["filter_db_run_bootstrap_interval_95"])
    figure, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, constrained_layout=True)
    axes[0].loglog(frequency, learned, label="Learned mean", color="#0072B2")
    axes[0].loglog(frequency, fixed, label="Fixed policy", color="#D55E00")
    axes[0].set_ylabel("Geometric mean DFT power")
    axes[0].legend()
    axes[1].plot(frequency, filter_db, color="#009E73", label="10 log10(learned/fixed)")
    axes[1].fill_between(frequency, interval[0], interval[1], color="#009E73", alpha=.2,
                         label="95% run bootstrap")
    axes[1].axhline(0, color="black", linewidth=.8)
    axes[1].set(xlabel="Frequency (epoch⁻¹)", ylabel="Filter function (dB)")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=.2)
    figure.suptitle("Natural drift after warm-up exclusion and epoch-150 normalization")
    path = ARTIFACT_ROOT / "spectral/natural_drift_uncertainty.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
