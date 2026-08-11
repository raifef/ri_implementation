"""Run-level power planning and fail-closed natural-drift spectral analysis."""
from __future__ import annotations

from math import ceil
from statistics import NormalDist
from typing import Any

import numpy as np

from hdfa_rl_suite.google_pure_source_exact.paper_families.natural import acquire_natural_condition

from .contracts import NONFINAL, V13_SCHEMA
from .io import ARTIFACT_ROOT, ROOT, atomic_json, atomic_text, canonical_hash, config, read_json


def plan_natural_drift_power() -> dict[str, Any]:
    pilot = read_json(ROOT / "artifacts/google_pure_v12/spectral/natural_drift_uncertainty.json")
    values = np.asarray(pilot["run_low_frequency_filter_db"], dtype=float)
    settings = config()["natural_drift_power"]
    effect = abs(float(settings["preregistered_suppressive_effect_db"]))
    alpha = float(settings["two_sided_alpha"])
    desired_power = float(settings["power"])
    z_alpha = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    z_power = NormalDist().inv_cdf(desired_power)
    pilot_sd = float(np.std(values, ddof=1))
    calculated = int(ceil(((z_alpha + z_power) * pilot_sd / effect) ** 2))
    planned = max(int(settings["minimum_complete_runs"]),
                  min(calculated, int(settings["maximum_planned_runs"])))
    result = {"schema_version": V13_SCHEMA, "pilot_run_count": len(values),
              "pilot_complete_run_values_db": values.tolist(), "pilot_standard_deviation_db": pilot_sd,
              "two_sided_alpha": alpha, "desired_power": desired_power,
              "preregistered_effect_db": -effect, "normal_approximation_required_runs": calculated,
              "planned_complete_runs": planned, "maximum_planned_runs": int(settings["maximum_planned_runs"]),
              "uncertainty_unit": "COMPLETE_PAIRED_RUN",
              "frequency_bins_are_replicates": False, "inferential_smoothing_used": False,
              "auto_execute": False, "explicit_execute_long_required": True,
              "power_plan_capped": calculated > int(settings["maximum_planned_runs"]), **NONFINAL}
    atomic_json(ARTIFACT_ROOT / "natural_drift/power_plan.json", result)
    atomic_text(ARTIFACT_ROOT / "natural_drift/power_plan.md", "\n".join([
        "# V13 natural-drift power plan", "",
        f"Pilot run-level SD: **{pilot_sd:.3f} dB**",
        f"Planned complete paired runs: **{planned}**",
        f"Uncapped normal-approximation requirement: **{calculated}**", "",
        "Complete runs are the uncertainty units. Frequency bins are not replicates, and no smoothing enters inference.",
        "The long run is never launched automatically.",
    ]))
    return result


def run_natural_drift(*, execute_long: bool = False, maximum_runs: int | None = None) -> dict[str, Any]:
    plan = plan_natural_drift_power()
    if not execute_long:
        value = {"schema_version": V13_SCHEMA, "executed": False,
                 "reason": "EXPLICIT_EXECUTE_LONG_REQUIRED", "planned_complete_runs": plan["planned_complete_runs"],
                 **NONFINAL}
        atomic_json(ARTIFACT_ROOT / "natural_drift/acquisition_status.json", value)
        return value
    source_protocol = read_json(ROOT / "artifacts/google_pure_paper_reproduction/experiment_protocols/"
                               "natural_drift_spectral_suppression_validation.json")
    planned = int(plan["planned_complete_runs"])
    count = planned if maximum_runs is None else min(planned, int(maximum_runs))
    if count <= 0:
        raise ValueError("maximum_runs must be positive")
    protocol = dict(source_protocol)
    protocol["schema_version"] = V13_SCHEMA
    protocol["config"] = dict(source_protocol["config"])
    protocol["config"]["evidence_scope"] = "V13_PREREGISTERED_RUN_LEVEL_POWER_PLAN"
    conditions = [{"plant_index": index % 6, "seed": 65_000 + index} for index in range(count)]
    protocol["conditions"] = conditions
    protocol["protocol_hash"] = canonical_hash({"v13_natural": protocol["config"], "conditions": conditions})
    rows = [acquire_natural_condition(protocol, condition) for condition in conditions]
    result = {"schema_version": V13_SCHEMA, "executed": True, "complete": len(rows) == count,
              "planned_complete_runs": planned, "executed_complete_runs": len(rows),
              "truncated_by_maximum_runs": count < planned, "protocol_hash": protocol["protocol_hash"],
              "rows": rows, "inferential_smoothing_used": False,
              "controller_target_access": False, **NONFINAL}
    atomic_json(ARTIFACT_ROOT / "natural_drift/acquisition.json", result)
    return result


def analyse_natural_drift() -> dict[str, Any]:
    acquisition_path = ARTIFACT_ROOT / "natural_drift/acquisition.json"
    if not acquisition_path.is_file():
        pilot = read_json(ROOT / "artifacts/google_pure_v12/spectral/natural_drift_uncertainty.json")
        result = {"schema_version": V13_SCHEMA, "analysis_scope": "INHERITED_V12_PILOT_ONLY",
                  "run_count": pilot["run_count"],
                  "aggregate_low_frequency_filter_db": pilot["aggregate_low_frequency_filter_db"],
                  "aggregate_low_frequency_filter_db_interval_95":
                      pilot["aggregate_low_frequency_filter_db_interval_95"],
                  "classification": "CENTRAL_SUPPRESSION_ESTIMATE_BUT_STATISTICALLY_UNRESOLVED",
                  "direction_identifiable": False, "uncertainty_unit": "COMPLETE_PAIRED_RUN",
                  "inferential_smoothing_used": False, "power_plan_complete": False, **NONFINAL}
        atomic_json(ARTIFACT_ROOT / "natural_drift/analysis.json", result)
        return result
    acquisition = read_json(acquisition_path)
    rows = acquisition["rows"]
    if not rows:
        raise RuntimeError("natural-drift acquisition contains no complete runs")
    spectra = []
    run_low = []
    for row in rows:
        spectrum = row["per_run_spectrum"]
        frequency = np.asarray(spectrum["frequency_per_epoch"], dtype=float)
        learned = np.asarray(spectrum["learned_power"], dtype=float)
        fixed = np.asarray(spectrum["fixed_power"], dtype=float)
        spectra.append((frequency, learned, fixed))
    lower = max(values[0][0] for values in spectra)
    upper = min(values[0][-1] for values in spectra)
    shared = np.geomspace(lower, upper, 256)
    learned = np.vstack([np.interp(shared, f, p) for f, p, _ in spectra])
    fixed = np.vstack([np.interp(shared, f, p) for f, _, p in spectra])
    ratio_db = 10.0 * np.log10(np.maximum(learned, np.finfo(float).tiny) /
                               np.maximum(fixed, np.finfo(float).tiny))
    low_bins = max(1, len(shared) // 4)
    run_low = np.median(ratio_db[:, :low_bins], axis=1)
    learned_geo = np.exp(np.mean(np.log(learned), axis=0))
    fixed_geo = np.exp(np.mean(np.log(fixed), axis=0))
    filter_db = 10.0 * np.log10(learned_geo / fixed_geo)
    point = float(np.median(filter_db[:low_bins]))
    rng = np.random.default_rng(65_901)
    bootstrap = []
    for _ in range(2000):
        sample = rng.integers(0, len(rows), len(rows))
        l = np.exp(np.mean(np.log(learned[sample]), axis=0))
        f = np.exp(np.mean(np.log(fixed[sample]), axis=0))
        bootstrap.append(float(np.median(10.0 * np.log10(l[:low_bins] / f[:low_bins]))))
    interval = np.quantile(bootstrap, [.025, .975])
    complete = len(rows) >= int(plan_natural_drift_power()["planned_complete_runs"])
    direction = bool(interval[1] < 0 or interval[0] > 0)
    result = {"schema_version": V13_SCHEMA, "analysis_scope": "V13_COMPLETE_RUNS",
              "run_count": len(rows), "run_low_frequency_filter_db": run_low.tolist(),
              "frequency_per_epoch": shared.tolist(), "learned_geometric_power": learned_geo.tolist(),
              "fixed_geometric_power": fixed_geo.tolist(), "filter_db_learned_over_fixed": filter_db.tolist(),
              "aggregate_low_frequency_filter_db": point,
              "aggregate_low_frequency_filter_db_interval_95": interval.tolist(),
              "direction_identifiable": direction, "power_plan_complete": complete,
              "classification": ("SUPPRESSION_IDENTIFIED" if interval[1] < 0 else
                                 "AMPLIFICATION_IDENTIFIED" if interval[0] > 0 else "UNRESOLVED_SIGN"),
              "uncertainty_unit": "COMPLETE_PAIRED_RUN", "frequency_bins_are_replicates": False,
              "warmup_epoch_excluded": all(row["warmup_epoch_excluded"] for row in rows),
              "source_epoch_150_normalization": all(row["source_epoch_150_normalization"] for row in rows),
              "spectral_aggregation": "GEOMETRIC_MEAN_ACROSS_RUNS",
              "inferential_smoothing_used": False, **NONFINAL}
    atomic_json(ARTIFACT_ROOT / "natural_drift/analysis.json", result)
    return result

