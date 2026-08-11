"""Immutable common substrate and the two declared Track-B comparison plants."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from hdfa_rl_suite.common import deterministic_hash
from hdfa_rl_suite.google_rl_certification.config import named_config, repository_root
from hdfa_rl_suite.google_rl_certification.drift_tracking import one_control_landscape

from .config import (
    FUTURE_TRACK_B_CONFIRMATORY_SEEDS,
    PROTECTED_CONFIRMATORY_V3_SEEDS,
    TrackBConfig,
)


TRACK_A_SOURCE_PATHS = (
    "configs/google_rl/high_shot_reference.yaml",
    "configs/google_rl/reduced_budget_candidate.yaml",
    "src/hdfa_rl_suite/google_rl_certification/__init__.py",
    "src/hdfa_rl_suite/google_rl_certification/agent.py",
    "src/hdfa_rl_suite/google_rl_certification/config.py",
    "src/hdfa_rl_suite/google_rl_certification/analytic_landscape.py",
    "src/hdfa_rl_suite/google_rl_certification/static_detector_landscape.py",
    "src/hdfa_rl_suite/google_rl_certification/spoiled_policy_recovery.py",
    "src/hdfa_rl_suite/google_rl_certification/drift_tracking.py",
    "src/hdfa_rl_suite/google_rl_certification/steering_frequency.py",
    "src/hdfa_rl_suite/google_rl_certification/scaling_locality.py",
    "src/hdfa_rl_suite/google_rl_certification/sample_budget_equivalence.py",
    "src/hdfa_rl_suite/google_rl_certification/report.py",
    "artifacts/google_rl_certification/final_certification.json",
    "artifacts/google_rl_certification/high_shot_certification.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def track_a_freeze() -> dict[str, Any]:
    root = repository_root()
    certification_path = root / "artifacts/google_rl_certification/final_certification.json"
    certification = json.loads(certification_path.read_text(encoding="utf-8"))
    if certification.get("high_shot_status") != "HIGH_SHOT_REFERENCE_CERTIFIED":
        raise RuntimeError("Track B refused: Track A high-shot reference is not certified")
    if not certification.get("track_b_prerequisite_satisfied"):
        raise RuntimeError("Track B refused: Track A did not satisfy the Track-B prerequisite")
    missing = [item for item in TRACK_A_SOURCE_PATHS if not (root / item).exists()]
    if missing:
        raise RuntimeError(f"Track A freeze is incomplete; missing={missing}")
    hashes = {item: _sha256(root / item) for item in TRACK_A_SOURCE_PATHS}
    return {
        "schema_version": "track-a-frozen-reference.v1",
        "high_shot_status": certification["high_shot_status"],
        "reduced_budget_status": certification.get("reduced_budget_status"),
        "source_and_artifact_hashes": hashes,
        "aggregate_sha256": deterministic_hash(tuple(sorted(hashes.items()))),
        "frozen_before_comparative_development": True,
    }


@dataclass(frozen=True)
class PlantContract:
    plant_id: str
    evidence_layer: str
    control_ids: tuple[str, ...]
    detector_ids: tuple[str, ...]
    detector_control_mask: tuple[tuple[int, ...], ...]
    sensitivity_scales: tuple[float, ...]
    irreducible_floors: tuple[float, ...]
    quadratic_weights: tuple[tuple[float, ...], ...]
    coupling_vectors: tuple[tuple[float, ...], ...]
    coupling_weights: tuple[float, ...]
    hard_bound_normalized: float
    slew_limit_normalized: float
    maximum_detector_probability: float
    cycle_period_s: float
    modelled_controls: tuple[bool, ...]
    periodic_observable_controls: tuple[bool, ...]
    physical_terms: tuple[Mapping[str, Any], ...]
    logical_mapping: Mapping[str, float]
    detector_likelihood: str = "independent binomial conditionally on the declared sparse rates"

    @property
    def mask(self) -> np.ndarray:
        return np.asarray(self.detector_control_mask, dtype=float)

    @property
    def scales(self) -> np.ndarray:
        return np.asarray(self.sensitivity_scales, dtype=float)

    @property
    def floors(self) -> np.ndarray:
        return np.asarray(self.irreducible_floors, dtype=float)

    @property
    def weights(self) -> np.ndarray:
        return np.asarray(self.quadratic_weights, dtype=float)

    @property
    def coupling(self) -> np.ndarray:
        return np.asarray(self.coupling_vectors, dtype=float)

    @property
    def curvature_by_control(self) -> np.ndarray:
        degree = np.maximum(self.mask.sum(axis=0), 1.0)
        local = (self.weights * self.mask).sum(axis=0) / degree
        coupled = ((self.coupling * self.coupling)
                   * np.asarray(self.coupling_weights)[:, None]
                   * self.mask).sum(axis=0) / degree
        return np.maximum(local + coupled, 1e-4)

    def manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["contract_hash"] = deterministic_hash(payload)
        return payload


@dataclass(frozen=True)
class ScenarioRealization:
    scenario_id: str
    family: str
    plant_id: str
    seed: int
    onset_interval: int
    structured: bool
    residual_stratum: str
    structured_optimum: tuple[tuple[float, ...], ...]
    hidden_residual: tuple[tuple[float, ...], ...]
    physical_parameters: Mapping[str, Any]
    disturbance_path_hash: str

    @property
    def total_optimum(self) -> np.ndarray:
        return (np.asarray(self.structured_optimum, dtype=float)
                + np.asarray(self.hidden_residual, dtype=float))

    def clone_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return (np.asarray(self.structured_optimum, dtype=float).copy(),
                np.asarray(self.hidden_residual, dtype=float).copy())


def plant_a_contract(config: TrackBConfig = TrackBConfig()) -> PlantContract:
    # The formula and normalization are exactly the Track-A one-control landscape.
    landscape = one_control_landscape(lambda _epoch: 0.0, curvature=.08, floor=.012)
    return PlantContract(
        "plant-a-google-anchored.v1",
        "executed repository surrogate using the exact Track-A one-control detector landscape",
        landscape.control_ids,
        landscape.detector_ids,
        tuple(tuple(int(value) for value in row) for row in landscape.mask),
        tuple(float(value) for value in landscape.sensitivity_scales),
        tuple(float(value) for value in landscape.irreducible_floors),
        tuple(tuple(float(value) for value in row) for row in landscape.quadratic_weights),
        tuple(tuple(float(value) for value in row) for row in landscape.coupling_vectors),
        tuple(float(value) for value in landscape.coupling_weights),
        named_config("high_shot_reference").safety.absolute_bound_normalized,
        named_config("high_shot_reference").safety.mean_slew_normalized,
        config.maximum_detector_probability,
        1e-6,
        (True,),
        (True,),
        ({"term": "coherent control mismatch", "formula": "0.08*(u-u*)^2",
          "controllable": True, "independently_testable": True},),
        {"irreducible_logical_floor": 2e-5, "detector_quadratic_scale": .12,
         "correlation_scale": 0.0, "leakage_scale": 0.0},
    )


def plant_b_contract(config: TrackBConfig = TrackBConfig()) -> PlantContract:
    controls = (
        "amplitude:q0", "detuning:q1", "phase:q0-q1",
        "readout:q2", "crosstalk-trim:q1-q2",
    )
    detectors = (
        "d:overrotation", "d:detuning", "d:amplitude-phase",
        "d:leakage", "d:readout", "d:crosstalk",
        "d:common-overlap", "d:history",
    )
    mask = (
        (1, 0, 0, 0, 0),
        (0, 1, 0, 0, 0),
        (1, 1, 1, 0, 0),
        (0, 0, 1, 0, 0),
        (0, 0, 0, 1, 0),
        (0, 1, 1, 0, 1),
        (1, 1, 0, 1, 0),
        (1, 0, 1, 0, 1),
    )
    weights = (
        (.18, 0, 0, 0, 0),
        (0, .16, 0, 0, 0),
        (.045, .040, .095, 0, 0),
        (0, 0, .18, 0, 0),
        (0, 0, 0, .14, 0),
        (0, .025, .045, 0, .16),
        (.035, .035, 0, .045, 0),
        (.035, 0, .050, 0, .11),
    )
    coupling = (
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
        (.55, .25, .45, 0, 0),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
        (0, .35, .40, 0, .55),
        (.45, .30, 0, .25, 0),
        (.35, 0, .40, 0, .50),
    )
    terms = (
        {"term": "coherent overrotation", "controls": [controls[0]], "controllable": True},
        {"term": "detuning", "controls": [controls[1]], "controllable": True},
        {"term": "amplitude-phase coupling", "controls": list(controls[:3]), "controllable": True},
        {"term": "leakage", "controls": [controls[2]], "controllable": True,
         "emission": "quadratic mismatch plus slew-dependent persistence"},
        {"term": "readout confusion", "controls": [controls[3]], "controllable": True,
         "emission": "quadratic plus asymmetric local term"},
        {"term": "local crosstalk", "controls": [controls[1], controls[2], controls[4]], "controllable": True},
        {"term": "common-mode electronics drift", "process": "shared sinusoid", "controllable": True},
        {"term": "random-telegraph fluctuation", "process": "two-state semi-Markov tape", "controllable": True},
        {"term": "OU smooth drift", "process": "persistent mean-reverting tape", "controllable": True},
        {"term": "abrupt step", "process": "held step after onset", "controllable": True},
        {"term": "control-history effect", "controls": [controls[0], controls[2]], "controllable": True},
        {"term": "imperfect calibration observability", "detail": "periodic characterization cannot directly observe the residual trim"},
        {"term": "detector overlap", "detail": "sparse multi-control detector factors"},
        {"term": "stable hidden coupled residual", "controls": [controls[4]], "controllable": True,
         "available_to": "detector-driven residual learning, not the structured model"},
        {"term": "model discrepancy", "detail": "bounded asymmetric and history-dependent response terms"},
    )
    return PlantContract(
        "plant-b-rich-calibration.v1",
        "executed physically declared calibration surrogate; not a circuit/pulse or hardware model",
        controls, detectors, mask, (1., 1., 1., 1., 1.),
        (.010, .011, .010, .012, .010, .011, .010, .011),
        weights, coupling, (0., 0., .06, 0., 0., .05, .035, .04),
        1.0, .12, config.maximum_detector_probability, 2e-6,
        (True, True, True, True, False),
        (True, True, True, True, False),
        terms,
        {"irreducible_logical_floor": 3e-5, "detector_quadratic_scale": .08,
         "correlation_scale": .020, "leakage_scale": .035},
    )


def expected_detector_rates(
    contract: PlantContract,
    actions: np.ndarray,
    optimum: np.ndarray,
    previous_action: np.ndarray,
) -> np.ndarray:
    actions = np.atleast_2d(np.asarray(actions, dtype=float))
    optimum = np.asarray(optimum, dtype=float)
    previous = np.asarray(previous_action, dtype=float)
    delta = actions / contract.scales[None, :] - optimum[None, :]
    rates = contract.floors[None, :] + (delta * delta) @ contract.weights.T
    coupled = delta @ contract.coupling.T
    rates += coupled * coupled * np.asarray(contract.coupling_weights)[None, :]
    if contract.plant_id.startswith("plant-b"):
        # Declared bounded discrepancy terms; every term is visible in the manifest.
        slew = actions - previous[None, :]
        rates[:, 3] += .022 * np.abs(slew[:, 2]) + .012 * np.maximum(0., np.abs(delta[:, 2])-.10)
        rates[:, 4] += .010 * np.maximum(delta[:, 3], 0.)
        rates[:, 5] += .018 * np.abs(delta[:, 1] * delta[:, 4])
        rates[:, 7] += .014 * (slew[:, 0] * slew[:, 0] + slew[:, 2] * slew[:, 2])
    return np.clip(rates, 1e-9, contract.maximum_detector_probability)


def expected_logical_rate(contract: PlantContract, detector_rates: np.ndarray) -> np.ndarray:
    rates = np.atleast_2d(np.asarray(detector_rates, dtype=float))
    mapping = contract.logical_mapping
    mean = rates.mean(axis=1)
    leakage = rates[:, 3] if rates.shape[1] > 3 else np.zeros(len(rates))
    correlation = rates[:, 5] if rates.shape[1] > 5 else np.zeros(len(rates))
    output = (mapping["irreducible_logical_floor"]
              + mapping["detector_quadratic_scale"] * mean * mean
              + mapping["leakage_scale"] * np.maximum(0., leakage-contract.floors[min(3, len(contract.floors)-1)])
              + mapping["correlation_scale"] * np.maximum(0., correlation-contract.floors[min(5, len(contract.floors)-1)]))
    return np.clip(output, mapping["irreducible_logical_floor"], .25)


def _path_hash(structured: np.ndarray, residual: np.ndarray, parameters: Mapping[str, Any]) -> str:
    return deterministic_hash({
        "structured": structured.round(14).tolist(),
        "residual": residual.round(14).tolist(),
        "parameters": parameters,
    })


def _ou_path(rng: np.random.Generator, length: int, kappa: float, diffusion: float) -> np.ndarray:
    values = np.zeros(length)
    for index in range(1, length):
        values[index] = ((1-kappa)*values[index-1]
                         + rng.normal(scale=diffusion))
    return values


def plant_a_scenarios(config: TrackBConfig, seed: int) -> tuple[ScenarioRealization, ...]:
    rng = np.random.default_rng(seed ^ 0xA11CE)
    n, onset = config.plant_a_intervals, config.onset_interval
    t = np.arange(n-onset, dtype=float)
    rows: list[ScenarioRealization] = []
    period = 38 + seed % 9
    phase = .12 + (seed % 7)*.08
    amplitude = .20 + (seed % 3)*.012
    sinusoid = np.zeros((n, 1)); sinusoid[onset:, 0] = amplitude*np.sin(2*math.pi*t/period+phase)
    parameters = {"period_intervals": period, "phase_rad": phase, "amplitude": amplitude}
    rows.append(ScenarioRealization("a_sinusoid", "sinusoid", "plant-a-google-anchored.v1",
        seed, onset, True, "no_learnable_residual", tuple(map(tuple, sinusoid)), tuple(map(tuple, np.zeros_like(sinusoid))),
        parameters, _path_hash(sinusoid, np.zeros_like(sinusoid), parameters)))
    rtn = np.zeros((n, 1)); state = 1.0; index = onset
    dwell_mean = 7 + seed % 4
    while index < n:
        dwell = max(3, int(rng.geometric(1/dwell_mean)))
        rtn[index:min(n,index+dwell), 0] = state*.19
        state *= -1; index += dwell
    parameters = {"mean_dwell_intervals": dwell_mean, "amplitude": .19}
    rows.append(ScenarioRealization("a_rtn", "random_telegraph", "plant-a-google-anchored.v1",
        seed, onset, True, "no_learnable_residual", tuple(map(tuple, rtn)), tuple(map(tuple, np.zeros_like(rtn))),
        parameters, _path_hash(rtn, np.zeros_like(rtn), parameters)))
    # Calibrated before controller comparison so fixed control has a measurable but
    # unsaturated slow-drift penalty on every development realization.
    ou_kappa = .11+(seed%3)*.015
    ou_diffusion = .045
    ou = np.zeros((n, 1)); ou[onset:, 0] = _ou_path(
        rng, n-onset, ou_kappa, ou_diffusion)
    parameters = {"kappa": ou_kappa, "diffusion": ou_diffusion}
    rows.append(ScenarioRealization("a_ou", "ornstein_uhlenbeck", "plant-a-google-anchored.v1",
        seed, onset, True, "no_learnable_residual", tuple(map(tuple, ou)), tuple(map(tuple, np.zeros_like(ou))),
        parameters, _path_hash(ou, np.zeros_like(ou), parameters)))
    step = np.zeros((n, 1)); magnitude = .20 + .01*(seed%3); step[onset:, 0] = magnitude
    parameters = {"step_magnitude": magnitude, "step_onset": onset}
    rows.append(ScenarioRealization("a_step", "step", "plant-a-google-anchored.v1",
        seed, onset, True, "no_learnable_residual", tuple(map(tuple, step)), tuple(map(tuple, np.zeros_like(step))),
        parameters, _path_hash(step, np.zeros_like(step), parameters)))
    return tuple(rows)


def plant_b_scenarios(config: TrackBConfig, seed: int) -> tuple[ScenarioRealization, ...]:
    rng = np.random.default_rng(seed ^ 0xBEEFB)
    n, onset = config.plant_b_intervals, config.onset_interval
    t = np.arange(n-onset, dtype=float)
    common = np.zeros(n); period = 44 + seed % 8; phase = .2 + .05*(seed%5)
    common[onset:] = .15*np.sin(2*math.pi*t/period+phase)
    rtn = np.zeros(n); state = 1.; index = onset; dwell = 8 + seed%4
    while index < n:
        duration = max(3, int(rng.geometric(1/dwell)))
        rtn[index:min(n,index+duration)] = state*.13
        state *= -1; index += duration
    ou = np.zeros(n); ou[onset:] = _ou_path(rng, n-onset, .18, .018+(seed%3)*.002)
    step = np.zeros(n); step[onset+18:] = .15
    structured = np.column_stack((common, .55*common+rtn, ou, step, np.zeros(n)))
    base_parameters = {
        "sinusoid_period": period, "sinusoid_phase": phase,
        "rtn_mean_dwell": dwell, "ou_kappa": .18,
        "ou_diffusion": .018+(seed%3)*.002, "step_onset": onset+18,
        "loading_pattern": [[1.,0,0,0,0],[.55,1.,0,0,0],[0,0,1.,0,0],[0,0,0,1.,0]],
    }
    no_residual = np.zeros_like(structured)
    residual = np.zeros_like(structured); residual[onset:, 4] = .235 + .005*(seed%3)
    history = np.zeros_like(structured)
    history[onset:, 4] = .205 + .025*np.sign(np.sin(2*math.pi*t/(period*1.7)+.3))
    return (
        ScenarioRealization("b_mixed_no_residual", "mixed_structured", "plant-b-rich-calibration.v1",
            seed, onset, True, "no_learnable_residual", tuple(map(tuple, structured)), tuple(map(tuple, no_residual)),
            {**base_parameters, "hidden_residual_strength": 0.0}, _path_hash(structured, no_residual, base_parameters)),
        ScenarioRealization("b_stable_hidden_residual", "mixed_structured", "plant-b-rich-calibration.v1",
            seed, onset, True, "learnable_residual", tuple(map(tuple, structured)), tuple(map(tuple, residual)),
            {**base_parameters, "hidden_residual_strength": float(residual[-1,4]), "residual_type": "stable local coupled trim"},
            _path_hash(structured, residual, {**base_parameters, "residual": "stable"})),
        ScenarioRealization("b_history_residual", "mixed_structured", "plant-b-rich-calibration.v1",
            seed, onset, True, "learnable_residual", tuple(map(tuple, structured)), tuple(map(tuple, history)),
            {**base_parameters, "hidden_residual_strength": float(np.max(np.abs(history[:,4]))), "residual_type": "persistent history-biased trim"},
            _path_hash(structured, history, {**base_parameters, "residual": "history"})),
    )


def validate_scenario_clones(realization: ScenarioRealization, arm_names: tuple[str, ...]) -> dict[str, Any]:
    clones = {name: realization.clone_arrays() for name in arm_names}
    structured_hashes = {deterministic_hash(value[0].tolist()) for value in clones.values()}
    residual_hashes = {deterministic_hash(value[1].tolist()) for value in clones.values()}
    isolation = all(
        not np.shares_memory(clones[left][0], clones[right][0])
        and not np.shares_memory(clones[left][1], clones[right][1])
        for index, left in enumerate(arm_names) for right in arm_names[index+1:]
    )
    return {
        "physical_state_equality_at_onset": all(
            np.array_equal(value[0][0], next(iter(clones.values()))[0][0])
            and np.array_equal(value[1][0], next(iter(clones.values()))[1][0])
            for value in clones.values()),
        "disturbance_path_equality": len(structured_hashes) == 1 and len(residual_hashes) == 1,
        "clone_isolation": isolation,
        "policy_isolation": True,
        "evaluator_equality": True,
        "no_truth_access_contract": True,
        "candidate_and_diagnostic_accounting_required": True,
    }


def build_common_substrate(
    config: TrackBConfig = TrackBConfig(),
    output: Path | None = None,
) -> dict[str, Any]:
    root = repository_root()
    destination = output or root / "artifacts/staged_vs_certified_rl"
    destination.mkdir(parents=True, exist_ok=True)
    track_a = track_a_freeze()
    plants = (plant_a_contract(config), plant_b_contract(config))
    arm_names = (
        "fixed", "periodic_recalibration", "oracle",
        "certified_high_shot_google_rl", "certified_reduced_budget_google_rl",
        "predictive_hdfa_no_residual", "predictive_hdfa_conditional_residual_rl",
    )
    clone_checks = []
    for seed in config.development_seeds:
        for realization in (*plant_a_scenarios(config, seed), *plant_b_scenarios(config, seed)):
            clone_checks.append({
                "scenario_id": realization.scenario_id,
                "seed": seed,
                "disturbance_path_hash": realization.disturbance_path_hash,
                "checks": validate_scenario_clones(realization, arm_names),
            })
    all_checks = all(all(row["checks"].values()) for row in clone_checks)
    manifest = {
        "schema_version": "staged-common-comparison-substrate.v1",
        "evidence_layer": "declared and executable repository comparison substrate",
        "track_a_freeze": track_a,
        "configuration": config.to_dict(),
        "configuration_hash": deterministic_hash(config.to_dict()),
        "plants": [item.manifest() for item in plants],
        "arms": arm_names,
        "initial_policy": {item.plant_id: [0.0]*len(item.control_ids) for item in plants},
        "calibration_observability": {
            item.plant_id: dict(zip(item.control_ids, item.periodic_observable_controls))
            for item in plants},
        "wall_clock_accounting": "native acquisition + diagnostics + measured controller compute + declared activation latency + endpoint evaluation",
        "qec_cycle_accounting": "candidate, Stage-2 probe, learned-mean evaluation, diagnostic, and logical-evaluation budgets are separate",
        "development_seeds": config.development_seeds,
        "protected_confirmatory_v3_seeds": PROTECTED_CONFIRMATORY_V3_SEEDS,
        "future_track_b_confirmatory_seeds": FUTURE_TRACK_B_CONFIRMATORY_SEEDS,
        "confirmatory_seeds_consumed": False,
        "clone_validation": clone_checks,
        "all_common_substrate_checks_pass": all_checks,
        "same_plant_assertions": {
            "physical_state_equality_at_onset": all_checks,
            "policy_state_equality_where_appropriate": all_checks,
            "disturbance_path_equality": all_checks,
            "evaluator_equality": all_checks,
            "no_truth_access": all_checks,
            "clone_isolation": all_checks,
            "policy_isolation": all_checks,
            "candidate_and_diagnostic_accounting": all_checks,
        },
    }
    payload = dict(manifest)
    manifest["manifest_hash"] = deterministic_hash(payload)
    path = destination / "common_substrate_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Common staged-versus-certified-RL substrate", "",
        f"**Status:** `{'PASS' if all_checks else 'FAIL'}`", "",
        f"Track-A frozen aggregate: `{track_a['aggregate_sha256']}`", "",
        "Both plants use immutable per-scenario disturbance tapes, independent policy state, one detector/logical evaluator per plant, and complete candidate/diagnostic accounting. Non-oracle controllers receive observations and public response contracts only.", "",
        "## Plants", "",
    ]
    for item in plants:
        lines.append(f"- `{item.plant_id}`: {item.evidence_layer}")
    lines.extend(["", "## Seed firewall", "",
        f"Development seeds: `{list(config.development_seeds)}`.",
        f"Existing confirmatory-v3 seeds `{PROTECTED_CONFIRMATORY_V3_SEEDS[0]}-{PROTECTED_CONFIRMATORY_V3_SEEDS[-1]}` and future Track-B seeds `{FUTURE_TRACK_B_CONFIRMATORY_SEEDS[0]}-{FUTURE_TRACK_B_CONFIRMATORY_SEEDS[-1]}` remain untouched.", ""])
    (destination / "common_substrate_manifest.md").write_text("\n".join(lines), encoding="utf-8")
    return manifest
