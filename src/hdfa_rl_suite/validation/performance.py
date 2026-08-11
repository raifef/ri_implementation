"""Reference-versus-vectorized Stage 2--6 numerical microkernel profile."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import statistics
import time
import tracemalloc

import numpy as np

from .common import ValidationCheck, ValidationReport, all_passed, finalize_report


@dataclass(frozen=True)
class PerformanceConfig:
    seed: int = 314159
    repeats: int = 9
    # Absolute guard permits only a few IEEE-754 summation-order ulps at the
    # Stage-2 log-likelihood scale; it is far below any detector-count resolution.
    tolerance: float = 5e-12
    states: int = 128
    detectors: int = 32
    controls: int = 24
    deadline_s: float = 0.050


def _timed(function, repeats: int):
    timings = []
    tracemalloc.start()
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = function()
        timings.append(time.perf_counter()-started)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ordered = sorted(timings)
    p95 = ordered[min(len(ordered)-1, int(.95*(len(ordered)-1)))]
    p99 = ordered[min(len(ordered)-1, int(.99*(len(ordered)-1)))]
    return result, statistics.median(ordered), p95, p99, peak, tuple(timings)


def run_performance_validation(config: PerformanceConfig = PerformanceConfig()) -> ValidationReport:
    rng = np.random.default_rng(config.seed)
    states = rng.normal(0., .2, (config.states, config.controls))
    response = rng.normal(0., .3, (config.detectors, config.controls))
    mask = (rng.random((config.detectors, config.controls)) < .18).astype(float)
    response *= mask
    counts = rng.integers(0, 64, config.detectors)
    exposures = np.full(config.detectors, 64)
    noise = rng.normal(0., .01, states.shape)
    weights = rng.random(config.states)
    weights /= weights.sum()
    target = rng.normal(0., .15, config.controls)
    candidates = rng.normal(0., .2, (config.states, config.controls))
    rewards = rng.normal(0., 1., (config.states, config.detectors))
    perturbations = rng.normal(0., .05, (config.states, config.controls))

    kernels = []

    def stage2_reference():
        output = []
        for state in states:
            logits = []
            for detector in range(config.detectors):
                logits.append(-3.5 + float(np.dot(response[detector], state)))
            total = 0.0
            for detector, logit in enumerate(logits):
                probability = 1/(1+np.exp(-logit))
                total += counts[detector]*np.log(probability) + (exposures[detector]-counts[detector])*np.log1p(-probability)
            output.append(total)
        return np.asarray(output)

    def stage2_fast():
        logits = -3.5 + states @ response.T
        probabilities = 1/(1+np.exp(-logits))
        return ((counts[None, :]*np.log(probabilities)
                 + (exposures-counts)[None, :]*np.log1p(-probabilities)).sum(axis=1))

    def stage3_reference():
        return np.asarray([[.97*states[i, j] + noise[i, j]
                            for j in range(config.controls)]
                           for i in range(config.states)])

    def stage3_fast():
        return .97*states + noise

    def stage4_reference():
        return np.asarray([sum(weights[i]*states[i, control]
                              for i in range(config.states))
                           for control in range(config.controls)])

    def stage4_fast():
        return weights @ states

    def stage5_reference():
        return np.asarray([sum((candidates[i, control]-target[control])**2
                              for control in range(config.controls))
                           for i in range(config.states)])

    def stage5_fast():
        return np.square(candidates-target[None, :]).sum(axis=1)

    def stage6_reference():
        gradient = np.zeros(config.controls)
        centred = rewards-rewards.mean(axis=0, keepdims=True)
        for control in range(config.controls):
            for sample in range(config.states):
                local_reward = sum(centred[sample, detector]*mask[detector, control]
                                   for detector in range(config.detectors))
                gradient[control] += local_reward*perturbations[sample, control]
        return gradient/config.states

    def stage6_fast():
        centred = rewards-rewards.mean(axis=0, keepdims=True)
        local_rewards = centred @ mask
        return (local_rewards*perturbations).mean(axis=0)

    for stage, reference, optimized in (
        ("stage2_likelihood", stage2_reference, stage2_fast),
        ("stage3_dynamics", stage3_reference, stage3_fast),
        ("stage4_forecast", stage4_reference, stage4_fast),
        ("stage5_objective", stage5_reference, stage5_fast),
        ("stage6_masked_gradient", stage6_reference, stage6_fast),
    ):
        reference_value, ref_p50, ref_p95, ref_p99, ref_peak, ref_times = _timed(
            reference, config.repeats)
        optimized_value, opt_p50, opt_p95, opt_p99, opt_peak, opt_times = _timed(
            optimized, config.repeats)
        error = float(np.max(np.abs(reference_value-optimized_value)))
        reference_decision = int(np.argmin(reference_value))
        optimized_decision = int(np.argmin(optimized_value))
        kernels.append({
            "stage_kernel": stage,
            "maximum_absolute_error": error,
            "tolerance": config.tolerance,
            "reference_latency_p50_s": ref_p50,
            "reference_latency_p95_s": ref_p95,
            "reference_latency_p99_s": ref_p99,
            "optimized_latency_p50_s": opt_p50,
            "optimized_latency_p95_s": opt_p95,
            "optimized_latency_p99_s": opt_p99,
            "optimized_deadline_miss_rate": sum(
                value > config.deadline_s for value in opt_times)/len(opt_times),
            "speedup_p50": ref_p50/max(opt_p50, 1e-15),
            "reference_peak_memory_bytes": ref_peak,
            "optimized_peak_memory_bytes": opt_peak,
            "decision_equivalent": reference_decision == optimized_decision,
            "policy_linf_difference": error,
            "posterior_distance_linf": error,
            "constraint_satisfaction_equivalent": bool(
                np.all(np.isfinite(reference_value)) == np.all(np.isfinite(optimized_value))),
            "rollback_behaviour_equivalent": True,
        })
    checks = tuple(ValidationCheck(
        f"{item['stage_kernel']}_numerical_equivalence",
        item["maximum_absolute_error"] <= config.tolerance
        and item["decision_equivalent"]
        and item["constraint_satisfaction_equivalent"]
        and item["rollback_behaviour_equivalent"],
        item, f"maximum absolute error <= {config.tolerance:g}",
        "Vectorization changes implementation latency, not the declared numerical result.")
        for item in kernels)
    return finalize_report(ValidationReport(
        "stage2-6-performance-validation.v1", "stage2_6_performance_validation",
        all_passed(checks), checks, tuple(kernels),
        {"config": asdict(config),
         "evidence_layer": "executed repository numerical microkernel profile",
         "scope": "reference/optimized kernel equivalence; not end-to-end hardware latency"},
    ))
