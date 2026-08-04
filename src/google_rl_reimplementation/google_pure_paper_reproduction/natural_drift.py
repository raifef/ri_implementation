"""Natural-drift spectral suppression using the frozen pure v7 controller."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from google_rl_reimplementation.google_pure_v7.config import canonical_hash
from google_rl_reimplementation.google_pure_v7.experiments import run_production_trace
from google_rl_reimplementation.google_pure_v7.natural import FAMILIES, FrozenNaturalPlant, generate_natural_drift, welch_band_metrics


def acquire_condition(protocol: Mapping[str, Any], condition: Mapping[str, Any]) -> dict[str, Any]:
    index = int(condition["plant_index"]); family = FAMILIES[index]; horizon = int(protocol["config"]["epochs"])
    plant = FrozenNaturalPlant(index, v5_compatible_units=False); tape = generate_natural_drift(family, horizon)
    result = run_production_trace(plant, tape, seed=int(condition["seed"]), candidates=int(protocol["config"]["candidates"]), cycles=int(protocol["config"]["cycles_per_candidate"]))
    spectral = welch_band_metrics(result["logical_risk"])
    return {"plant_id": family["id"], "family": family["family"], "seed": int(condition["seed"]),
            "raw_trace_hash": canonical_hash(tape.tolist()), **spectral,
            "trajectory": {key: np.asarray(value).tolist() for key, value in result["logical_risk"].items()},
            "epoch_frequency_unit": "cycles_per_epoch", "power_db_convention": "10*log10(power ratio)"}


def validation(rows: list[dict[str, Any]], mode: str) -> tuple[bool, list[str], dict[str, Any]]:
    gains = [row["low_frequency_suppression_db_fixed_over_mean"] for row in rows]
    reasons = []
    if any(row["power_db_convention"] != "10*log10(power ratio)" for row in rows): reasons.append("wrong PSD dB convention")
    if mode in {"reference", "paper-scale"} and len(rows) != 6: reasons.append("reference natural-drift ensemble must contain all six frozen plants")
    return not reasons, reasons, {"median_suppression_db": float(np.median(gains)) if gains else None, "paper_anchor_db": 4.0}
