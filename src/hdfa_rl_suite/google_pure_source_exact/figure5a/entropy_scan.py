"""Frozen entropy anchors, dense phase scan, and quantitative classifications."""
from __future__ import annotations

from itertools import product
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import (DIAGNOSTIC_STREAM_ACQUISITION_CONTRACT, AcquisitionMode,
                        Figure5aProtocol, SOURCE_ENTROPY_ANCHORS, canonical_hash)


def build_conditions(config: Mapping[str, Any], *, mode: AcquisitionMode,
                     scan: str) -> list[dict[str, Any]]:
    profile = config["profiles"][mode.value]
    seeds = tuple(int(value) for value in profile["seeds"])
    if scan == "anchors":
        frequencies = (float(config["anchor"]["frequency"]),)
        entropies = tuple(float(value) for value in config["anchor"]["entropy_weights"])
        if entropies != SOURCE_ENTROPY_ANCHORS:
            raise ValueError("anchor scan must be exactly 0.001, 0.01, 0.1")
    elif scan == "dense":
        frequencies = tuple(float(value) for value in config["dense_scan"]["frequencies"])
        entropies = tuple(float(value) for value in config["dense_scan"]["entropy_weights"])
    else:
        raise ValueError("scan must be anchors or dense")
    return [{"frequency": frequency, "entropy_weight": entropy_weight, "seed": seed}
            for frequency, entropy_weight, seed in product(frequencies, entropies, seeds)]


def scan_contract(config: Mapping[str, Any], *, mode: AcquisitionMode, scan: str,
                  protocol: Figure5aProtocol, plant_hash: str,
                  controller_hash: str) -> dict[str, Any]:
    conditions = build_conditions(config, mode=mode, scan=scan)
    payload = {"schema_version": "figure5a-scan-contract.v1", "mode": mode.value,
               "scan": scan, "protocol_hash": protocol.protocol_hash,
               "plant_hash": plant_hash, "controller_hash": controller_hash,
               "diagnostic_stream_acquisition_contract":
                   DIAGNOSTIC_STREAM_ACQUISITION_CONTRACT,
               "analysis_contract": dict(
                   config["anchor"]["classification"] if scan == "anchors"
                   else config["dense_scan"]["classification"]),
               "conditions": conditions, "condition_count": len(conditions),
               "validation_watermark": mode != AcquisitionMode.REFERENCE,
               "one_frozen_controller_and_plant": True}
    payload["scan_hash"] = canonical_hash(payload)
    return payload


def classify_anchor_rows(rows: Sequence[Mapping[str, Any]],
                         criteria: Mapping[str, Any]) -> dict[str, Any]:
    by_seed: dict[int, dict[float, Mapping[str, Any]]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), {})[float(row["entropy_weight"])] = row
    seed_rows = []
    for seed, anchors in sorted(by_seed.items()):
        missing = set(SOURCE_ENTROPY_ANCHORS) - set(anchors)
        if missing:
            seed_rows.append({"seed": seed, "pass": False, "blocking_reasons": [f"missing anchors {sorted(missing)}"]})
            continue
        low, middle, high = (anchors[value] for value in SOURCE_ENTROPY_ANCHORS)
        ratios = [row[kind]["source_ratio"] for row in (low, middle, high)
                  for kind in ("stochastic_ratio", "learned_mean_ratio")]
        if any(value is None for value in ratios):
            seed_rows.append({"seed": seed, "pass": False,
                              "blocking_reasons": ["zero finite-shot fixed/optimal denominator"]})
            continue
        entropies = [float(np.mean([record["policy_entropy"] for record in row["epoch_records"]]))
                     for row in (low, middle, high)]
        stochastic = [float(row["stochastic_ratio"]["source_ratio"]) for row in (low, middle, high)]
        learned = [float(row["learned_mean_ratio"]["source_ratio"]) for row in (low, middle, high)]
        cap_fractions = []
        cap_telemetry_complete = True
        for row in (low, middle, high):
            values = [record.get("fraction_at_sigma_max") for record in row["epoch_records"]]
            if not values or any(value is None for value in values):
                cap_telemetry_complete = False
                cap_fractions.append(None)
            else:
                cap_fractions.append(float(max(values)))
        entropy_order = entropies[0] + float(criteria["minimum_entropy_separation"]) < entropies[1] \
            and entropies[1] + float(criteria["minimum_entropy_separation"]) < entropies[2]
        high_gap = learned[2] - stochastic[2] >= float(criteria["minimum_high_exploration_gap"])
        low_failure = learned[1] - learned[0] >= float(criteria["minimum_low_tracking_gap"])
        middle_best = (stochastic[1] >= max(stochastic[0], stochastic[2])
                       if criteria["middle_must_maximize_stochastic_ratio"] else True)
        gates = {"monotonic_entropy_order": entropy_order, "high_entropy_exploration_degradation": high_gap,
                 "low_entropy_tracking_failure": low_failure,
                 "middle_balanced_stochastic_best": middle_best,
                 "middle_stochastic_absolute_advantage":
                     stochastic[1] >= float(criteria["minimum_middle_stochastic_ratio"]),
                 "high_entropy_learned_mean_beats_fixed":
                     learned[2] > float(criteria["minimum_high_learned_mean_ratio"]),
                 "sigma_cap_telemetry_complete": cap_telemetry_complete,
                 "sigma_cap_not_dominant": cap_telemetry_complete and all(
                     value is not None and value <= float(criteria["maximum_sigma_cap_fraction"])
                     for value in cap_fractions)}
        seed_rows.append({"seed": seed, "pass": all(gates.values()), "gates": gates,
                          "average_policy_entropy": entropies, "stochastic_ratios": stochastic,
                          "learned_mean_ratios": learned,
                          "maximum_epoch_sigma_cap_fractions": cap_fractions,
                          "blocking_reasons": [name for name, value in gates.items() if not value]})
    return {"anchor_classification_pass": bool(seed_rows and all(row["pass"] for row in seed_rows)),
            "stable_over_seeds": bool(seed_rows and all(row["pass"] for row in seed_rows)),
            "seed_rows": seed_rows, "criteria": dict(criteria),
            "absolute_acceptance_scientific_status":
                "PREREGISTERED_CLEAN_ROOM_THRESHOLDS_NOT_SOURCE_NUMERICAL_VALUES"}


def reduce_dense_rows(rows: Sequence[Mapping[str, Any]],
                      criteria: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce r(f, entropy) and estimate the primary stochastic r_max=0 crossing."""
    grouped: dict[tuple[float, float], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (float(row["frequency"]), float(row["entropy_weight"])), []).append(row)
    if not grouped:
        return {"dense_acceptance_pass": False, "blocking_reasons": ["no dense rows"]}
    frequencies = sorted({key[0] for key in grouped})
    entropies = sorted({key[1] for key in grouped})
    expected_cells = {(frequency, entropy) for frequency in frequencies for entropy in entropies}
    if set(grouped) != expected_cells:
        return {"dense_acceptance_pass": False,
                "blocking_reasons": ["dense frequency-entropy surface is incomplete"]}
    seed_counts = {len(values) for values in grouped.values()}
    if len(seed_counts) != 1 or next(iter(seed_counts)) < 1:
        return {"dense_acceptance_pass": False,
                "blocking_reasons": ["dense cells have inconsistent seed replication"]}
    surface = []
    by_frequency: dict[float, list[dict[str, Any]]] = {value: [] for value in frequencies}
    for (frequency, entropy), values in sorted(grouped.items()):
        stochastic = [row["stochastic_ratio"]["source_ratio"] for row in values]
        learned = [row["learned_mean_ratio"]["source_ratio"] for row in values]
        if any(value is None for value in (*stochastic, *learned)):
            return {"dense_acceptance_pass": False,
                    "blocking_reasons": ["zero finite-shot denominator in dense surface"]}
        stochastic_array = np.asarray(stochastic, dtype=float)
        learned_array = np.asarray(learned, dtype=float)
        cell = {
            "frequency": frequency, "entropy_weight": entropy,
            "seed_count": len(values),
            "stochastic_r_mean": float(np.mean(stochastic_array)),
            "stochastic_r_standard_error": float(
                np.std(stochastic_array, ddof=1) / np.sqrt(len(values)))
                if len(values) > 1 else 0.0,
            "learned_mean_r_mean": float(np.mean(learned_array)),
            "learned_mean_r_standard_error": float(
                np.std(learned_array, ddof=1) / np.sqrt(len(values)))
                if len(values) > 1 else 0.0,
        }
        surface.append(cell)
        by_frequency[frequency].append(cell)
    envelope = []
    for frequency in frequencies:
        cells = by_frequency[frequency]
        best = max(cells, key=lambda value: value["stochastic_r_mean"])
        envelope.append({
            "frequency": frequency,
            "r_max": best["stochastic_r_mean"],
            "best_entropy_weight": best["entropy_weight"],
            "r_standard_error_at_max": best["stochastic_r_standard_error"],
        })
    positive = [row["r_max"] > 0.0 for row in envelope]
    first_nonpositive = next((index for index, value in enumerate(positive) if not value), None)
    contiguous = (first_nonpositive is not None and any(positive[:first_nonpositive])
                  and not any(positive[first_nonpositive:]))
    bracket = None
    estimate = None
    if first_nonpositive is not None and first_nonpositive > 0:
        left, right = envelope[first_nonpositive - 1], envelope[first_nonpositive]
        if left["r_max"] > 0.0 and right["r_max"] <= 0.0:
            bracket = [left["frequency"], right["frequency"]]
            delta = right["r_max"] - left["r_max"]
            estimate = (right["frequency"] if delta == 0.0 else
                        left["frequency"] - left["r_max"] *
                        (right["frequency"] - left["frequency"]) / delta)
    source_threshold = float(criteria["source_threshold_frequency_per_epoch"])
    gates = {
        "low_frequency_positive_region":
            envelope[0]["r_max"] > float(criteria["minimum_low_frequency_r_max"]),
        "high_frequency_nonpositive_region":
            envelope[-1]["r_max"] <= float(criteria["maximum_high_frequency_r_max"]),
        "single_contiguous_positive_region":
            contiguous if criteria["require_contiguous_positive_region"] else True,
        "zero_crossing_bracketed": estimate is not None,
        "crossing_near_source_order": estimate is not None and
            abs(float(estimate) - source_threshold) <=
            float(criteria["maximum_absolute_threshold_error_per_epoch"]),
    }
    return {
        "schema_version": "figure5a-dense-threshold.v1",
        "dense_acceptance_pass": all(gates.values()),
        "gates": gates,
        "surface": surface,
        "stochastic_r_max_envelope": envelope,
        "estimated_steerability_threshold_frequency_per_epoch": estimate,
        "estimated_steerability_threshold_period_epochs":
            None if estimate is None or estimate <= 0 else 1.0 / estimate,
        "crossing_bracket_frequency_per_epoch": bracket,
        "source_threshold_order_frequency_per_epoch": source_threshold,
        "seed_count_per_cell": next(iter(seed_counts)),
        "criteria": dict(criteria),
        "blocking_reasons": [name for name, value in gates.items() if not value],
        "scientific_status": "CLEAN_ROOM_REDUCER_AGAINST_SOURCE_APPROXIMATE_THRESHOLD",
    }
