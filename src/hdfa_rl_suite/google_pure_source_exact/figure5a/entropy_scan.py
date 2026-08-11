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
        entropy_order = entropies[0] + float(criteria["minimum_entropy_separation"]) < entropies[1] \
            and entropies[1] + float(criteria["minimum_entropy_separation"]) < entropies[2]
        high_gap = learned[2] - stochastic[2] >= float(criteria["minimum_high_exploration_gap"])
        low_failure = learned[1] - learned[0] >= float(criteria["minimum_low_tracking_gap"])
        middle_best = stochastic[1] >= max(stochastic[0], stochastic[2])
        gates = {"monotonic_entropy_order": entropy_order, "high_entropy_exploration_degradation": high_gap,
                 "low_entropy_tracking_failure": low_failure, "middle_balanced_stochastic_best": middle_best}
        seed_rows.append({"seed": seed, "pass": all(gates.values()), "gates": gates,
                          "average_policy_entropy": entropies, "stochastic_ratios": stochastic,
                          "learned_mean_ratios": learned, "blocking_reasons": [name for name, value in gates.items() if not value]})
    return {"anchor_classification_pass": bool(seed_rows and all(row["pass"] for row in seed_rows)),
            "stable_over_seeds": bool(seed_rows and all(row["pass"] for row in seed_rows)),
            "seed_rows": seed_rows, "criteria": dict(criteria)}
