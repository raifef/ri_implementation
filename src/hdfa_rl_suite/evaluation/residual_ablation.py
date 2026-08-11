"""Fast preregistered development ablation for conditional Stage-6 authority."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import random
from statistics import NormalDist, fmean

from hdfa_rl_suite.stage6 import (
    ResidualActivationGate, ResidualGateConfig, ResidualGateEvidence,
    ResidualRLDisposition,
)


SCENARIOS = (
    "sinusoidal_drift", "semi_markov_switching", "ou_plus_step",
    "nested_common_mode", "heavy_tailed_ood", "learnable_residual",
    "pure_shot_noise",
)
ARMS = (
    "predictive_only", "current_always_on_residual_rl",
    "conditional_residual_rl", "periodic_recalibration", "full_control_rl",
)
DEVELOPMENT_SEEDS = (101, 102, 103, 104, 105, 106, 107, 108)


def _bias(scenario: str, interval: int, rng: random.Random) -> float:
    if scenario == "sinusoidal_drift":
        return .045*math.sin(2*math.pi*interval/7)
    if scenario == "semi_markov_switching":
        return .04*(1 if (interval//3) % 2 else -1)
    if scenario == "ou_plus_step":
        return .025*math.sin(interval/3)+(.04 if interval >= 6 else 0)
    if scenario == "nested_common_mode":
        return .025*math.sin(interval/5)+.025*(1 if interval % 4 < 2 else -1)
    if scenario == "heavy_tailed_ood":
        return max(-.12, min(.12, rng.gauss(0, .025)/(max(rng.random(), .08)**.25)))
    if scenario == "learnable_residual":
        return .055
    return 0.0


def _rate(uncompensated: float) -> float:
    return min(.25, .02+10*uncompensated*uncompensated)


def _gate_evidence(signal: float, history: list[float], previous_gain: float,
                   previous_probability: float, scenario: str) -> ResidualGateEvidence:
    noise = .012
    persistence = 0
    for value in reversed(history):
        if abs(value) <= 1.5*noise:
            break
        persistence += 1
    recent = history[-max(1, persistence):]
    repeatability = (max(0., 1-(max(recent)-min(recent))/(2*max(abs(signal), noise)))
                     if recent else 0.)
    snr = abs(signal)/noise
    return ResidualGateEvidence(
        persistence, repeatability, 1.0, scenario != "heavy_tailed_ood",
        .15 if scenario != "heavy_tailed_ood" else .8,
        .95 if scenario != "heavy_tailed_ood" else .3,
        abs(signal)/noise, 1.0, previous_probability, previous_gain,
        snr, True, abs(signal) <= noise,
        ood=scenario == "heavy_tailed_ood")


def _run(scenario: str, seed: int, arm: str, intervals: int = 16) -> dict:
    rng = random.Random(seed*1009+sum(map(ord, scenario))+sum(map(ord, arm)))
    gate = ResidualActivationGate(ResidualGateConfig(
        minimum_control_sensitivity=.1, maximum_forecast_uncertainty=.35,
        minimum_stability_probability=.8))
    history: list[float] = []
    residual_control = 0.0
    periodic_control = 0.0
    rates: list[float] = []
    damage = 0.0
    dispositions: list[str] = []
    previous_gain, previous_probability = 0.0, .5
    for interval in range(intervals):
        bias = _bias(scenario, interval, rng)
        noisy_signal = bias+rng.gauss(0, .012)
        history.append(noisy_signal)
        if arm == "predictive_only":
            control = 0.0
        elif arm == "periodic_recalibration":
            if interval % 4 == 0:
                periodic_control = -bias+rng.gauss(0, .008)
                damage += .0008
            control = periodic_control
        elif arm == "full_control_rl":
            # Same observable gradient, but a five-coordinate identity exploration
            # basis has higher variance and device cost than the one residual axis.
            control = -bias+rng.gauss(0, .025)
            damage += .00625
        elif arm == "current_always_on_residual_rl":
            control = -noisy_signal
            damage += control*control
            residual_control = control
            dispositions.append(ResidualRLDisposition.ACTIVE.value)
        else:
            decision = gate.evaluate(_gate_evidence(
                noisy_signal, history, previous_gain, previous_probability,
                scenario))
            dispositions.append(decision.disposition.value)
            control = residual_control
            if decision.eligible:
                candidate = -noisy_signal
                baseline_rate = _rate(bias+residual_control)
                candidate_rate = _rate(bias+candidate)
                exposures = 1024
                observed_base = sum(rng.random() < baseline_rate for _ in range(exposures))/exposures
                observed_candidate = sum(rng.random() < candidate_rate for _ in range(exposures))/exposures
                gain = observed_base-observed_candidate
                se = math.sqrt(
                    max(observed_base*(1-observed_base), 1e-12)/exposures
                    + max(observed_candidate*(1-observed_candidate), 1e-12)/exposures)
                probability = NormalDist().cdf(gain/max(se, 1e-12))
                promoted = gate.record_shadow_outcome(
                    gain=gain, probability_positive_value=probability,
                    gradient_snr=abs(noisy_signal)/.012)
                previous_gain, previous_probability = gain, probability
                damage += candidate*candidate
                if promoted:
                    residual_control = candidate
                    control = candidate
                elif gate.deactivated:
                    residual_control = control = 0.0
        rates.append(_rate(bias+control))
    return {
        "scenario": scenario, "seed": seed, "arm": arm,
        "mean_detector_rate": fmean(rates), "final_detector_rate": rates[-1],
        "integrated_excess_edr": sum(max(0., value-.02) for value in rates),
        "exploration_damage": damage, "rates": rates,
        "dispositions": dispositions,
        "abstain_fraction": (sum(value == "abstain" for value in dispositions)
                             / max(1, len(dispositions))),
    }


def run_residual_ablation(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = [_run(scenario, seed, arm) for scenario in SCENARIOS
            for seed in DEVELOPMENT_SEEDS for arm in ARMS]
    by = {(row["scenario"], row["seed"], row["arm"]): row for row in runs}
    noninferiority = [
        by[scenario, seed, "conditional_residual_rl"]["mean_detector_rate"]
        - by[scenario, seed, "predictive_only"]["mean_detector_rate"]
        for scenario in SCENARIOS for seed in DEVELOPMENT_SEEDS]
    noise_abstention = fmean(
        by["pure_shot_noise", seed, "conditional_residual_rl"]["abstain_fraction"]
        for seed in DEVELOPMENT_SEEDS)
    learnable_deltas = [
        by["learnable_residual", seed, "predictive_only"]["mean_detector_rate"]
        - by["learnable_residual", seed, "conditional_residual_rl"]["mean_detector_rate"]
        for seed in DEVELOPMENT_SEEDS]
    criteria = {
        "conditional_noninferior_to_predictive_only": fmean(noninferiority) <= .002,
        "conditional_abstains_on_pure_shot_noise": noise_abstention >= .80,
        "conditional_improves_learnable_residual": fmean(learnable_deltas) > 0,
    }
    summaries = {}
    for scenario in SCENARIOS:
        summaries[scenario] = {}
        for arm in ARMS:
            subset = [row for row in runs if row["scenario"] == scenario and row["arm"] == arm]
            summaries[scenario][arm] = {
                "mean_detector_rate": fmean(row["mean_detector_rate"] for row in subset),
                "mean_integrated_excess_edr": fmean(row["integrated_excess_edr"] for row in subset),
                "mean_exploration_damage": fmean(row["exploration_damage"] for row in subset),
                "mean_abstain_fraction": fmean(row["abstain_fraction"] for row in subset),
            }
    supported = all(criteria.values())
    report = {
        "schema_version": "residual-rl-development-ablation.v1",
        "development_only": True, "confirmatory_seeds_used": False,
        "seeds": DEVELOPMENT_SEEDS, "scenarios": SCENARIOS, "arms": ARMS,
        "scientific_scope": (
            "Stage-6 activation micro-ablation with finite Bernoulli detector evidence; "
            "not a replacement for a future frozen end-to-end confirmatory benchmark."),
        "criteria": criteria, "claim_supported": supported,
        "unsupported_reason": (None if supported else
            "development evidence does not yet support beneficial conditional residual RL"),
        "noise_abstention_fraction": noise_abstention,
        "mean_learnable_residual_gain": fmean(learnable_deltas),
        "mean_noninferiority_delta": fmean(noninferiority),
        "summaries": summaries, "runs": runs,
    }
    (output_dir/"residual_rl_ablation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Conditional residual-RL development ablation", "",
             f"Development claim supported: **{supported}**", "",
             "This is a finite-shot Stage-6 activation micro-ablation on development seeds; "
             "it does not consume or authorize future confirmatory seeds.", "",
             "## Predeclared criteria", ""]
    lines.extend(f"- [{'x' if passed else ' '}] {name}"
                 for name, passed in criteria.items())
    lines.extend(["", f"Pure-noise abstention: {noise_abstention:.3f}",
                  f"Mean learnable-residual gain: {fmean(learnable_deltas):.6f}",
                  f"Mean noninferiority delta: {fmean(noninferiority):.6f}"])
    (output_dir/"residual_rl_ablation.md").write_text(
        "\n".join(lines)+"\n", encoding="utf-8")
    return report


def validate_residual_gating(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    gate = ResidualActivationGate(ResidualGateConfig(minimum_control_sensitivity=.1))
    base = ResidualGateEvidence(0, 1., 1., True, .1, .95, .5, 1., .5, 0., 2.,
                                True, True)
    noise = gate.evaluate(base)
    shadow = gate.evaluate(replace(
        base, persistence_intervals=2, residual_magnitude=3.,
        predictive_only_adequate=False))
    gate.record_shadow_outcome(gain=.02, probability_positive_value=.99,
                               gradient_snr=3.)
    active = gate.evaluate(replace(
        shadow.evidence, probability_positive_value=.99,
        expected_heldout_gain=.02, gradient_snr=3.))
    gate.record_shadow_outcome(gain=-.01, probability_positive_value=.1,
                               gradient_snr=.2)
    gate.record_shadow_outcome(gain=-.01, probability_positive_value=.1,
                               gradient_snr=.2)
    deactivated = gate.evaluate(active.evidence)
    checks = {
        "noise_abstains": noise.disposition is ResidualRLDisposition.ABSTAIN,
        "eligible_residual_enters_shadow": shadow.disposition is ResidualRLDisposition.SHADOW,
        "positive_shadow_promotes": active.disposition is ResidualRLDisposition.ACTIVE,
        "two_regressions_deactivate": deactivated.disposition is ResidualRLDisposition.DEACTIVATED,
    }
    report = {"schema_version": "residual-gate-validation.v1",
              "passed": all(checks.values()), "checks": checks,
              "decisions": {"noise": asdict(noise), "shadow": asdict(shadow),
                            "active": asdict(active), "deactivated": asdict(deactivated)}}
    (output_dir/"residual-rl-gating-validation.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report

