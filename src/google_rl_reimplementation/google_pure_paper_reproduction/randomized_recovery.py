"""Recovery from a deliberately spoiled policy, distinct from a drift step."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from google_rl_reimplementation.google_pure_v6.plant import PureQuadraticPlant, default_spec
from google_rl_reimplementation.google_pure_v6.reference_agent import evidence_from_counts
from google_rl_reimplementation.google_pure_v7.controller import build_production_agent


def acquire_condition(protocol: Mapping[str, Any], condition: Mapping[str, Any]) -> dict[str, Any]:
    config = protocol["config"]; seed = int(condition["seed"]); severity = float(condition["severity"])
    plant = PureQuadraticPlant(default_spec(int(config.get("controls", 6)))); rng = np.random.default_rng(seed)
    spoiled = np.zeros(plant.spec.control_count); selected = rng.choice(plant.spec.control_count, max(1, plant.spec.control_count//2), replace=False)
    spoiled[selected] = rng.choice([-1., 1.], len(selected))*severity
    agent = build_production_agent(plant.mask, plant.spec.base_optimum_normalized, plant.spec.coordinates, seed=seed); agent.mean[:] = spoiled
    costs, excess, scales = [], [], []
    optimum_native = plant.base_optimum_native
    oracle = float(plant.logical_risk_native(optimum_native[None, :], optimum_native)[0])
    acquisition_rng = np.random.default_rng(seed+100_000)
    for _ in range(int(config["epochs"])):
        batch = agent.sample(int(config["candidates"])); counts = plant.acquire_counts(batch.applied_native_actions, optimum_native, cycles=int(config["cycles_per_candidate"]), rng=acquisition_rng)
        costs.append(float(np.mean(counts)/int(config["cycles_per_candidate"]))); agent.update(batch, evidence_from_counts(batch, counts, int(config["cycles_per_candidate"])))
        mean_native = plant.spec.coordinates.to_native(agent.mean); excess.append(float(plant.logical_risk_native(mean_native[None, :], optimum_native)[0]-oracle)); scales.append(float(np.mean(agent.scale)))
    initial = max(excess[:max(3, len(excess)//20)]); window = min(25, max(3, len(excess)//10)); moving = np.convolve(excess, np.ones(window)/window, mode="valid")
    hits = np.flatnonzero(moving <= .1*initial); recovery = int(hits[0]+window-1) if len(hits) else None
    return {"seed": seed, "severity": severity, "randomized_fraction": .5, "spoil_vector": spoiled.tolist(),
            "initial_excess": float(initial), "recovery_epoch": recovery, "censored": recovery is None,
            "trajectory": {"candidate_detector_cost": costs, "learned_mean_excess_logical_risk": excess, "mean_policy_scale": scales},
            "not_a_step_response": True}


def validation(rows: list[dict[str, Any]], mode: str) -> tuple[bool, list[str], dict[str, Any]]:
    reasons = []
    if any(not row.get("not_a_step_response") or row["randomized_fraction"] <= 0 for row in rows): reasons.append("recovery/step-response family conflation")
    observed = [row["recovery_epoch"] for row in rows if row["recovery_epoch"] is not None]
    return not reasons, reasons, {"median_recovery_epoch": float(np.median(observed)) if observed else None, "paper_anchor_epochs": 1000, "censored_count": len(rows)-len(observed)}

