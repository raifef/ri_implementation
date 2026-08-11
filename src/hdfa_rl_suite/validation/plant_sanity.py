"""Canonical physical-plant sanity scenarios and fail-closed assertions."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import statistics
from typing import Iterable, Mapping

from hdfa_rl_suite.simulator import (
    SIMULATOR_VERSION,
    DriftKind,
    LatentProcessSpec,
    ScalableQECDevice,
    SimulatorConfig,
)

from .common import ValidationCheck, ValidationReport, all_passed, finalize_report


@dataclass(frozen=True)
class PlantSanityConfig:
    seed: int = 271828
    qubit_count: int = 3
    cycles_per_interval: int = 16
    interval_count: int = 28
    step_amplitude: float = .25
    step_time_s: float = .112
    periodic_cadence: int = 8
    stationary_batches: int = 10
    stationary_cycles: int = 512


def _zero_process() -> tuple[LatentProcessSpec, ...]:
    return (LatentProcessSpec("stationary", DriftKind.CONSTANT, {}, amplitude=0.),)


def canonical_scenarios(config: PlantSanityConfig = PlantSanityConfig()) -> Mapping[str, tuple[LatentProcessSpec, ...]]:
    local = {"drive:q0": 1.0}
    return {
        "no_disturbance": _zero_process(),
        "persistent_step": (LatentProcessSpec(
            "step", DriftKind.STEP, local, amplitude=config.step_amplitude,
            step_time_s=config.step_time_s),),
        "sinusoid": (LatentProcessSpec(
            "sinusoid", DriftKind.SINUSOID, local, amplitude=.18,
            period_s=.64, phase_rad=.4),),
        "rtn": (LatentProcessSpec(
            "rtn", DriftKind.RANDOM_TELEGRAPH, local, amplitude=.16,
            rate_hz=8.0, mean_dwell_s=.125),),
        "ou": (LatentProcessSpec(
            "ou", DriftKind.ORNSTEIN_UHLENBECK, local, ou_kappa=.7,
            diffusion=.11),),
    }


def _device(config: PlantSanityConfig, processes: tuple[LatentProcessSpec, ...],
            *, seed_offset: int = 0, cycle_period_s: float = .001) -> ScalableQECDevice:
    return ScalableQECDevice(SimulatorConfig(
        qubit_count=config.qubit_count,
        code_distance=3,
        cycle_period_s=cycle_period_s,
        controller_latency_s=0.,
        base_detector_probability=.02,
        response_curvature=1.6,
        cross_coupling_strength=.08,
        maximum_detector_probability=.45,
        validated_mismatch_radius=.35,
        correlation_probability=0.,
        disturbance_resolution_s=min(.01, cycle_period_s),
        disturbances_enabled_at_start=True,
        stationary_vectorized_acquisition=False,
        seed=config.seed + seed_offset,
        processes=processes,
    ))


def _slew_to(device: ScalableQECDevice, target: Mapping[str, float], policy_id: str) -> None:
    current = device.confirmed_policy.controls
    action = {}
    for control, previous in current.items():
        desired = float(target.get(control, previous))
        bound = device.limits.controls[control]
        action[control] = max(bound.minimum, min(bound.maximum,
            max(previous-bound.max_slew, min(previous+bound.max_slew, desired))))
    device.apply_policy(action, policy_id=policy_id)


def _trajectory(config: PlantSanityConfig, processes: tuple[LatentProcessSpec, ...],
                arm: str) -> list[dict[str, object]]:
    device = _device(config, processes)
    view = device.oracle_evaluation_view("evaluation:plant-sanity")
    rows: list[dict[str, object]] = []
    for interval in range(config.interval_count):
        if arm == "oracle":
            _slew_to(device, view.optimum_policy(), f"oracle:{interval}")
        elif arm == "periodic" and interval % config.periodic_cadence == 0:
            estimate = device.characterize_controls(shots=256)
            _slew_to(device, estimate.estimates, f"periodic:{interval}")
        batch = device.acquire(config.cycles_per_interval, retain_records=False)
        diagnostic = view.physical_diagnostic()
        rows.append({
            "scenario": next((process.process_id for process in processes), "none"),
            "arm": arm,
            "interval": interval,
            "time_s": diagnostic.timestamp_s,
            "latent_optimum": dict(diagnostic.latent_optimum),
            "applied_control": dict(diagnostic.applied_control),
            "mismatch": dict(diagnostic.mismatch),
            "detector_probabilities": dict(diagnostic.detector_probabilities),
            "expected_edr": diagnostic.expected_global_detector_rate,
            "observed_edr": batch.detector_rate,
            "logical_metric": diagnostic.expected_logical_failure_proxy,
            "physical_state_id": batch.physical_state_id,
            "disturbance_state_id": batch.disturbance_state_id,
            "process_state": dict(view.process_state()),
            "policy_hash": batch.policy_activation.policy_hash,
        })
    return rows


def _mean(values: Iterable[float]) -> float:
    materialized = tuple(values)
    return statistics.fmean(materialized) if materialized else math.nan


def run_plant_validation(config: PlantSanityConfig = PlantSanityConfig(),
                         *, injected_faults: Iterable[str] = ()) -> ValidationReport:
    faults = set(injected_faults)
    scenarios = canonical_scenarios(config)
    trajectories: list[dict[str, object]] = []
    checks: list[ValidationCheck] = []

    fixed_stationary = _device(config, scenarios["no_disturbance"], cycle_period_s=.001)
    oracle_stationary = fixed_stationary.clone()
    fixed_view = fixed_stationary.oracle_evaluation_view("evaluation:no-disturbance-fixed")
    oracle_view = oracle_stationary.oracle_evaluation_view("evaluation:no-disturbance-oracle")
    fixed_rates, oracle_rates = [], []
    for index in range(config.stationary_batches):
        fixed_rates.append(fixed_stationary.acquire(config.stationary_cycles, retain_records=False).detector_rate)
        _slew_to(oracle_stationary, oracle_view.optimum_policy(), f"oracle-stationary:{index}")
        oracle_rates.append(oracle_stationary.acquire(config.stationary_cycles, retain_records=False).detector_rate)
    if "no_disturbance_drift" in faults:
        fixed_rates[-1] += .08
    expected_fixed = fixed_view.physical_diagnostic().expected_global_detector_rate
    expected_oracle = oracle_view.physical_diagnostic().expected_global_detector_rate
    stationary_spread = max(fixed_rates)-min(fixed_rates)
    no_disturbance_pass = (stationary_spread <= .025
                           and abs(_mean(fixed_rates)-_mean(oracle_rates)) <= .012
                           and abs(expected_fixed-expected_oracle) <= 1e-12)
    checks.append(ValidationCheck(
        "no_disturbance_stationarity", no_disturbance_pass,
        {"fixed_range": stationary_spread,
         "fixed_mean": _mean(fixed_rates), "oracle_mean": _mean(oracle_rates),
         "expected_difference": expected_fixed-expected_oracle},
        "fixed range <= 0.025, fixed/oracle observed means within 0.012, expected rates identical",
        "A stationary optimum must not create hidden policy or simulator drift.",
    ))

    fixed_step = _trajectory(config, scenarios["persistent_step"], "fixed")
    periodic_step = _trajectory(config, scenarios["persistent_step"], "periodic")
    oracle_step = _trajectory(config, scenarios["persistent_step"], "oracle")
    trajectories.extend(fixed_step + periodic_step + oracle_step)
    onset_index = next(i for i, row in enumerate(fixed_step)
                       if abs(float(row["latent_optimum"]["drive:q0"])) > .5*config.step_amplitude)
    post = slice(onset_index+2, None)
    fixed_post = _mean(float(row["expected_edr"]) for row in fixed_step[post])
    periodic_post = _mean(float(row["expected_edr"]) for row in periodic_step[post])
    oracle_post = _mean(float(row["expected_edr"]) for row in oracle_step[post])
    fixed_pre = _mean(float(row["expected_edr"]) for row in fixed_step[:onset_index])
    fixed_mismatch = _mean(abs(float(row["mismatch"]["drive:q0"])) for row in fixed_step[post])
    if "step_reset" in faults:
        fixed_mismatch = 0.
    step_pass = fixed_mismatch >= .9*config.step_amplitude and fixed_post >= fixed_pre+.01
    checks.append(ValidationCheck(
        "persistent_step_degrades_fixed", step_pass,
        {"pre_edr": fixed_pre, "post_edr": fixed_post,
         "post_mismatch": fixed_mismatch, "onset_interval": onset_index},
        "persistent mismatch >= 90% of step and post-step expected EDR exceeds pre-step by >= 0.01",
        "The fixed policy must remain displaced while the latent step persists.",
    ))
    if "oracle_bias" in faults:
        oracle_post = fixed_post
    ordering_pass = oracle_post < periodic_post < fixed_post
    checks.append(ValidationCheck(
        "oracle_periodic_fixed_ordering", ordering_pass,
        {"oracle": oracle_post, "periodic": periodic_post, "fixed": fixed_post},
        "oracle < periodic recalibration < fixed on sustained controllable step",
        "The oracle removes controllable error; periodic calibration pays cadence and measurement delay.",
    ))

    sweep_device = _device(config, scenarios["no_disturbance"])
    sweep_view = sweep_device.oracle_evaluation_view("evaluation:response-monotonicity")
    magnitudes = (0., .04, .08, .12, .18, .24, .30)
    sweep = []
    for magnitude in magnitudes:
        policy = dict(sweep_device.confirmed_policy.controls)
        policy["drive:q0"] = magnitude
        sweep.append(sweep_view.physical_diagnostic(policy).expected_global_detector_rate)
    if "nonmonotonic_response" in faults:
        sweep[4] = sweep[2]-.01
    monotone = all(right > left for left, right in zip(sweep, sweep[1:]))
    checks.append(ValidationCheck(
        "response_monotonicity", monotone,
        {"magnitudes": magnitudes, "expected_edr": sweep},
        "strictly increasing expected EDR over the declared local range",
        "The mismatch-to-error map is explicit, unsaturated, and monotone in the validated regime.",
    ))

    sinusoid = _trajectory(config, scenarios["sinusoid"], "fixed")
    trajectories.extend(sinusoid)
    process = scenarios["sinusoid"][0]
    expected_latent = [process.amplitude*math.sin(2*math.pi*float(row["time_s"])/process.period_s
                       + process.phase_rad) for row in sinusoid]
    if "sinusoid_phase" in faults:
        expected_latent = [process.amplitude*math.sin(2*math.pi*float(row["time_s"])/process.period_s
                           + process.phase_rad+.7) for row in sinusoid]
    observed_latent = [float(row["latent_optimum"]["drive:q0"]) for row in sinusoid]
    phase_error = max(abs(left-right) for left, right in zip(expected_latent, observed_latent))
    edr = [float(row["expected_edr"]) for row in sinusoid]
    squared = [value*value for value in observed_latent]
    xbar, ybar = _mean(squared), _mean(edr)
    covariance = sum((x-xbar)*(y-ybar) for x, y in zip(squared, edr))
    denominator = math.sqrt(sum((x-xbar)**2 for x in squared)*sum((y-ybar)**2 for y in edr))
    envelope_correlation = covariance/max(denominator, 1e-15)
    checks.append(ValidationCheck(
        "sinusoid_period_phase_and_envelope", phase_error < 1e-9 and envelope_correlation > .98,
        {"maximum_phase_path_error": phase_error, "edr_squared_mismatch_correlation": envelope_correlation},
        "path matches configured period/phase and corr(EDR, mismatch^2) > 0.98",
        "Interval-wise retention prevents time averaging from hiding the oscillatory envelope.",
    ))

    rtn_device = _device(config, scenarios["rtn"], cycle_period_s=.01)
    probe = dict(rtn_device.confirmed_policy.controls)
    probe["drive:q0"] = .05
    rtn_device.apply_policy(probe, policy_id="rtn-asymmetric-probe")
    rtn_view = rtn_device.oracle_evaluation_view("evaluation:rtn-sanity")
    labels, probabilities = [], []
    for _ in range(1000):
        rtn_device.acquire(1, retain_records=False)
        state = float(rtn_view.process_state()["rtn"])
        labels.append(1 if state > 0 else -1)
        probabilities.append(rtn_view.physical_diagnostic().detector_probabilities["d:q0"])
    runs: list[int] = []
    start = 0
    for index in range(1, len(labels)+1):
        if index == len(labels) or labels[index] != labels[start]:
            runs.append(index-start)
            start = index
    dwell = _mean(run*.01 for run in runs)
    positive = _mean(value for label, value in zip(labels, probabilities) if label > 0)
    negative = _mean(value for label, value in zip(labels, probabilities) if label < 0)
    if "rtn_alias" in faults:
        negative = positive
    rtn_pass = len(runs) >= 8 and .04 <= dwell <= .35 and abs(positive-negative) >= .015
    checks.append(ValidationCheck(
        "rtn_state_and_dwell_statistics", rtn_pass,
        {"run_count": len(runs), "mean_dwell_s": dwell,
         "positive_state_detector_probability": positive,
         "negative_state_detector_probability": negative},
        ">=8 runs, mean dwell in [0.04, 0.35] s, state-conditioned detector rates differ by >=0.015",
        "An asymmetric safe probe makes the two reproducible telegraph states statistically distinguishable.",
    ))

    ou = _device(config, scenarios["ou"], cycle_period_s=.01)
    ou.acquire(20, retain_records=False)
    before = ou.oracle_evaluation_view("evaluation:ou-before-clone").latent_state()["drive:q0"]
    clone = ou.clone()
    paths_equal = True
    ids_equal = True
    for _ in range(50):
        left = ou.acquire(1, retain_records=False)
        right = clone.acquire(1, retain_records=False)
        paths_equal &= (ou.oracle_evaluation_view("evaluation:ou-left").latent_state()
                        == clone.oracle_evaluation_view("evaluation:ou-right").latent_state())
        ids_equal &= left.disturbance_state_id == right.disturbance_state_id
    independent_policy = dict(clone.confirmed_policy.controls)
    independent_policy["drive:q0"] = .05
    clone.apply_policy(independent_policy, policy_id="clone-only")
    clone.acquire(1, retain_records=False)
    mutable_independent = ou.confirmed_policy.policy_hash != clone.confirmed_policy.policy_hash
    if "ou_clone_mismatch" in faults:
        paths_equal = False
    if "state_id_mismatch" in faults:
        ids_equal = False
    checks.append(ValidationCheck(
        "ou_persistence_and_clone_identity", abs(before) > 1e-9 and paths_equal and ids_equal and mutable_independent,
        {"preclone_state": before, "paths_equal": paths_equal,
         "state_ids_equal": ids_equal, "mutable_policy_independent": mutable_independent},
        "nonzero persistent state; cloned exogenous path/IDs identical; mutable policies independent",
        "Controller clones share the disturbance realization without sharing mutable policy state.",
    ))

    metadata = {
        "simulator_version": SIMULATOR_VERSION,
        "config": asdict(config),
        "injected_faults": sorted(faults),
        "evidence_layer": "executed repository simulation",
        "plant_mapping": "active optimum - applied control -> quadratic/cross-coupled detector probability -> logical proxy",
    }
    return finalize_report(ValidationReport(
        "plant-physical-validation.v1", "plant_sanity",
        all_passed(checks), tuple(checks), tuple(trajectories), metadata,
    ))
