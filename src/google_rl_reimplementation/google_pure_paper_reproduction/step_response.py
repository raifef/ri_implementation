"""Injected optimum-step response, kept separate from policy spoil recovery."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from google_rl_reimplementation.google_pure_v6.plant import PureQuadraticPlant, default_spec, optimum_tape
from google_rl_reimplementation.google_pure_v7.experiments import run_production_trace
from google_rl_reimplementation.google_pure_v7.response import estimate_step_response


def acquire_condition(protocol: Mapping[str, Any], condition: Mapping[str, Any]) -> dict[str, Any]:
    config = protocol["config"]; seed = int(condition["seed"]); severity = float(condition["severity"]); epochs = int(config["epochs"])
    plant = PureQuadraticPlant(default_spec(int(config.get("controls", 6)))); tape = optimum_tape("step", epochs, severity, controls=plant.spec.control_count, seed=seed)
    result = run_production_trace(plant, tape, seed=seed, candidates=int(config["candidates"]), cycles=int(config["cycles_per_candidate"]))
    target = tape[-1]; norm = max(float(np.dot(target, target)), 1e-12); projected = np.asarray(result["learned_mean_vectors"])@target/norm
    onset = int(.25*epochs); diagnostic = estimate_step_response(projected, onset_epoch=onset, target=1.0, sustained_epochs=min(25, max(3, epochs//10)))
    return {"seed": seed, "severity": severity, "onset_epoch": onset, "response": diagnostic,
            "trajectory": {"normalized_projected_policy_response": projected.tolist(), "learned_mean_logical_risk": result["logical_risk"]["learned_mean"].tolist(),
                           "fixed_logical_risk": result["logical_risk"]["fixed_policy"].tolist()},
            "policy_spoil_applied": False}


def validation(rows: list[dict[str, Any]], mode: str) -> tuple[bool, list[str], dict[str, Any]]:
    reasons = []
    if any(row.get("policy_spoil_applied") for row in rows): reasons.append("step/policy-spoil family conflation")
    times = [row["response"]["response_time_90_epochs"] for row in rows if row["response"]["response_time_90_epochs"] is not None]
    return not reasons, reasons, {"median_response_time_90_epochs": float(np.median(times)) if times else None, "paper_anchor_epochs": 130}

