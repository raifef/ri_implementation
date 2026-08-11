"""Fast plant sanity ordering and optional Stim topology validation."""
from __future__ import annotations

from typing import Any

import numpy as np

from .config import load_reference_config, load_surrogate_config
from .experiments import _trajectory
from .reference_agent import DetectorEvidence, ReferenceAgent
from .reporting import write_surrogate_contract
from .surrogate import PaperAnchoredSurrogate, surface_code_parameter_count


def validate_surrogate() -> dict[str, Any]:
    plant = PaperAnchoredSurrogate(distance=3, controls_per_gate=1)
    fixed = plant.initial_mean_native
    no_drift_a = plant.evaluate_native(fixed, plant.optimum_at(0))
    no_drift_b = plant.evaluate_native(fixed, plant.optimum_at(1000))
    oracle = plant.evaluate_native(np.zeros(plant.control_count), plant.optimum_normalized)
    small = plant.evaluate_native(0.1 * plant.sensitivity)
    large = plant.evaluate_native(0.3 * plant.sensitivity)
    drifted = plant.evaluate_native(
        fixed,
        plant.optimum_at(300, step_epoch=100, step_amplitude=0.28),
    )
    random = plant.evaluate_native(0.85 * np.where(np.arange(plant.control_count) % 2, 1, -1) * plant.sensitivity)
    trace = _trajectory(
        7901,
        50,
        lambda _epoch, local_plant: local_plant.optimum_normalized,
        regime_id="surrogate-sanity",
    )
    checks = {
        "no_drift_fixed_stationary": bool(np.array_equal(no_drift_a.detector_rates, no_drift_b.detector_rates)),
        "oracle_at_irreducible_floor": bool(np.allclose(oracle.detector_rates[0], plant.floors)),
        "persistent_drift_degrades_fixed": bool(drifted.logical_risk[0] > no_drift_a.logical_risk[0]),
        "larger_mismatch_larger_detector_cost": bool(large.detector_rates.mean() > small.detector_rates.mean()),
        "detector_logical_direction_agrees": bool(
            (large.detector_rates.mean() > small.detector_rates.mean()) and (large.logical_risk[0] > small.logical_risk[0])
        ),
        "high_shot_moves_toward_optimum": bool(
            trace["mean_policy_distance_to_optimum"][-1] < trace["mean_policy_distance_to_optimum"][0]
        ),
        "random_control_is_poor": bool(random.logical_risk[0] > no_drift_a.logical_risk[0]),
    }
    config = load_reference_config()
    mask = np.zeros((1, 2), dtype=bool)
    mask[0, 0] = True
    local_agent = ReferenceAgent(
        ["active", "inactive"], ["detector"], mask, np.ones(2), np.array([0.2, 0.4]), config, seed=7901
    )
    batch = local_agent.sample_candidates(regime_id="inactive")
    before = local_agent.mean.copy()
    counts = np.where(batch.actions_normalized[:, 0:1] > 0, 20_000, 5_000)
    evidence = tuple(
        DetectorEvidence(batch.candidate_ids[i], batch.action_hashes[i], counts[i], 100_000, "inactive")
        for i in range(40)
    )
    local_agent.update(batch, evidence)
    checks["inactive_control_unchanged"] = bool(local_agent.mean[1] == before[1])
    stim_check: dict[str, Any]
    try:
        import stim

        topology = []
        for distance in (3, 5, 7, 15):
            circuit = stim.Circuit.generated(
                "surface_code:rotated_memory_x",
                distance=distance,
                rounds=distance,
                after_clifford_depolarization=0.001,
            )
            topology.append({
                "distance": distance,
                "stim_qubits": circuit.num_qubits,
                "stim_detectors": circuit.num_detectors,
                "declared_controls_30_per_gate": surface_code_parameter_count(distance, 30),
            })
        stim_check = {"available": True, "status": "PASS", "topology": topology}
    except (ImportError, ValueError) as error:
        stim_check = {"available": False, "status": "NOT_EVALUABLE", "reason": str(error)}
    passed = all(checks.values()) and stim_check["status"] == "PASS"
    payload = {
        "schema_version": "google-paper-anchored-surrogate-validation.v2",
        "checks": checks,
        "stim_topology_validation": stim_check,
        "status": "PASS" if passed else "FAIL",
        "certification_allowed": passed,
        "claim_boundary": "Stim topology plus synthetic plant sanity; not Willow validation.",
    }
    write_surrogate_contract()
    return payload
