"""Common-tape synthetic controller experiments and stability diagnostics."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

import numpy as np

from .config import load_controller_choices, reject_certification_seed
from .controller import DetectorEvidence, MaskedGaussianPPO
from .plant import PlantSpec, SyntheticPlant


def controller_choices(amendment: Mapping[str, Any] | None = None, *, initial_std: float | None = None) -> dict[str, Any]:
    choices = dict(load_controller_choices()["baseline"])
    if amendment:
        choices.update(dict(amendment))
    if initial_std is not None:
        choices["initial_std"] = float(initial_std)
        if choices.get("target_std") is not None:
            choices["target_std"] = float(initial_std)
    return choices


def _acf1(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 3 or np.std(values) < 1e-15:
        return 0.0
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def _bands(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 4:
        return {"low":0.0,"mid":0.0,"high":0.0}
    centered = values - np.mean(values)
    power = np.abs(np.fft.rfft(centered)) ** 2 / len(values) ** 2
    freq = np.fft.rfftfreq(len(values))
    return {
        "low":float(power[(freq > 0) & (freq <= 0.02)].sum()),
        "mid":float(power[(freq > 0.02) & (freq <= 0.10)].sum()),
        "high":float(power[freq > 0.10].sum()),
    }


def simulate_trace(
    spec: PlantSpec, *, seed: int, epochs: int = 120, candidates: int = 40, cycles: int = 100_000,
    amendment: Mapping[str, Any] | None = None, no_drift: bool = False,
    family_override: str | None = None, frequency: float | None = None, amplitude: float | None = None,
    initial_std: float | None = None, spoil_severity: float | None = None, certification: bool = False,
) -> dict[str, Any]:
    reject_certification_seed(seed, certification=certification)
    plant = SyntheticPlant(spec)
    choices = controller_choices(amendment, initial_std=initial_std)
    initial = plant.initial_mean.copy()
    if spoil_severity is not None:
        spoil_rng = np.random.default_rng(spec.plant_draw_seed + 9_000_000)
        fraction = min(1.0, 0.35 + 0.65 * float(spoil_severity))
        selected = spoil_rng.random(spec.control_count) < fraction
        displacement = spoil_rng.choice([-1.0, 1.0], spec.control_count) * float(spoil_severity)
        initial[selected] = np.clip(plant.base_optimum[selected] + displacement[selected], -0.95, 0.95)
    detector_variance = plant.floors * (1 - plant.floors) * spec.overdispersion / cycles
    agent = MaskedGaussianPPO(plant.mask, initial, choices, seed=seed,
                              detector_noise_variance=detector_variance)
    fixed = initial.copy()
    rng = np.random.default_rng(seed + 100_000)
    trace: dict[str, list[float]] = {key:[] for key in (
        "fixed","learned_mean","stochastic","oracle","measurement_variance","candidate_variance",
        "mean_distance","update_jump","harmful_update","policy_std","reward_residual","baseline_residual",
        "replay_age","clip_fraction","scale_floor","scale_ceiling","gradient_norm")}
    previous_mean_risk: float | None = None
    for epoch in range(epochs):
        optimum = plant.optimum(epoch, family_override=family_override, frequency=frequency,
                                amplitude=amplitude, no_drift=no_drift)
        before_risk = float(plant.logical_risk(agent.mean, optimum)[0])
        batch = agent.sample(candidates, regime_id=f"{spec.plant_id}:{family_override or spec.family}:{'stationary' if no_drift else 'dynamic'}")
        counts, true_rates = plant.acquire_counts(batch.actions, optimum, cycles, rng)
        evidence = tuple(DetectorEvidence(cid, batch.action_hashes[i], counts[i], cycles, batch.regime_id)
                         for i, cid in enumerate(batch.candidate_ids))
        observed_rates = counts / cycles
        baseline_before = agent.baseline.copy()
        update = agent.update(batch, evidence)
        mean_risk = float(plant.logical_risk(agent.mean, optimum)[0])
        fixed_risk = float(plant.logical_risk(fixed, optimum)[0])
        candidate_risk = plant.logical_risk(batch.actions, optimum)
        oracle_risk = float(plant.logical_risk(optimum, optimum)[0])
        measurement_var = float(np.mean(true_rates * (1 - true_rates) * spec.overdispersion / cycles))
        reward_residual = float(np.mean(-observed_rates - baseline_before[None, :]))
        trace["fixed"].append(fixed_risk)
        trace["learned_mean"].append(mean_risk)
        trace["stochastic"].append(float(np.mean(candidate_risk)))
        trace["oracle"].append(oracle_risk)
        trace["measurement_variance"].append(measurement_var)
        trace["candidate_variance"].append(float(np.var(candidate_risk, ddof=1)))
        trace["mean_distance"].append(float(np.sqrt(np.mean((agent.mean - optimum) ** 2))))
        trace["update_jump"].append(0.0 if previous_mean_risk is None else mean_risk - previous_mean_risk)
        trace["harmful_update"].append(float(mean_risk > before_risk + 1e-12))
        trace["policy_std"].append(float(agent.std.mean()))
        trace["reward_residual"].append(reward_residual)
        trace["baseline_residual"].append(float(np.mean(-true_rates - baseline_before[None, :])))
        trace["replay_age"].append(update["mean_replay_age"])
        trace["clip_fraction"].append(update["clip_fraction"])
        trace["scale_floor"].append(update["scale_floor_fraction"])
        trace["scale_ceiling"].append(update["scale_ceiling_fraction"])
        trace["gradient_norm"].append(update["gradient_norm_before_clip"])
        previous_mean_risk = mean_risk
    return {
        "plant":asdict(spec), "seed":seed, "epochs":epochs, "candidates_per_epoch":candidates,
        "effective_cycles_per_candidate":cycles, "choices":choices, "trace":trace,
        "native_cost":{"candidate_count":epochs*candidates,"native_qec_cycles":epochs*candidates*cycles},
        "certification_seed_consumed":bool(certification),
    }


def trace_metrics(result: Mapping[str, Any], *, warmup_fraction: float = 0.25) -> dict[str, Any]:
    t = result["trace"]
    n = len(t["fixed"])
    start = min(n - 4, max(2, int(n * warmup_fraction)))
    fixed = np.asarray(t["fixed"][start:], dtype=float)
    mean = np.asarray(t["learned_mean"][start:], dtype=float)
    stochastic = np.asarray(t["stochastic"][start:], dtype=float)
    oracle = np.asarray(t["oracle"][start:], dtype=float)
    measurement = float(np.mean(t["measurement_variance"][start:])) * (0.55 ** 2) / max(result["plant"]["detector_count"], 1)
    var = {"fixed":float(np.var(fixed, ddof=1)), "learned_mean":float(np.var(mean, ddof=1)),
           "stochastic":float(np.var(stochastic, ddof=1)), "oracle":float(np.var(oracle, ddof=1))}
    corrected = {k:max(v-measurement, 0.0) for k,v in var.items()}
    fixed_bands, mean_bands, stochastic_bands = _bands(fixed), _bands(mean), _bands(stochastic)
    fixed_low = fixed_bands["low"]
    mean_low = mean_bands["low"]
    suppression = 10 * np.log10(max(fixed_low, 1e-30) / max(mean_low, 1e-30))
    jump = np.asarray(t["update_jump"][start:])
    slow_proxy = 2.0 * mean_bands["low"]
    candidate_component = float(np.var(stochastic-mean,ddof=1))
    jump_component = float(np.var(jump,ddof=1))
    mean_tracking = max(var["learned_mean"]-slow_proxy,0.0)
    closure_residual = var["stochastic"]-(slow_proxy+mean_tracking+candidate_component+jump_component+measurement)
    return {
        "window":{"warmup_excluded":start,"epochs_evaluated":len(fixed)},
        "raw_total_variance":var, "measurement_noise_variance_estimate":measurement,
        "measurement_corrected_variance":corrected,
        "psd_integrated_power":{"fixed":fixed_bands,"learned_mean":mean_bands,"stochastic":stochastic_bands},
        "mean_policy_stability_ratio":float(np.std(fixed,ddof=1)/max(np.std(mean,ddof=1),1e-15)),
        "operational_stochastic_stability_ratio":float(np.std(fixed,ddof=1)/max(np.std(stochastic,ddof=1),1e-15)),
        "low_frequency_suppression_db":float(suppression),
        "variance_decomposition":{"slow_residual_proxy":slow_proxy,"mean_tracking_proxy":mean_tracking,
                                  "candidate_exploration":candidate_component,"update_jumps":jump_component,
                                  "measurement":measurement,"covariance_and_nonorthogonal_closure_residual":closure_residual,
                                  "note":"PSD and jump components are non-orthogonal diagnostics; the signed closure residual contains their covariance and overlap."},
        "candidate_only_variance":candidate_component,
        "learned_mean_variance":var["learned_mean"],
        "mean_exploration_damage":float(np.mean(stochastic-mean)),
        "fine_tuning_benefit":float((np.mean(fixed)-np.mean(mean))/max(np.mean(fixed),1e-15)),
        "update_jump_quantiles":{"q05":float(np.quantile(jump,.05)),"q50":float(np.quantile(jump,.5)),"q95":float(np.quantile(jump,.95))},
        "harmful_update_rate":float(np.mean(t["harmful_update"][start:])),
        "policy_covariance_trajectory":{"initial_std":float(t["policy_std"][0]),"median_std":float(np.median(t["policy_std"][start:])),"final_std":float(t["policy_std"][-1])},
        "reward_residual_lag1_autocorrelation":_acf1(np.asarray(t["reward_residual"][start:])),
        "baseline_residual_lag1_autocorrelation":_acf1(np.asarray(t["baseline_residual"][start:])),
        "replay_age_mean":float(np.mean(t["replay_age"][start:])),
        "ppo_clip_fraction":float(np.mean(t["clip_fraction"][start:])),
        "mean_policy_lag_rms":float(np.mean(t["mean_distance"][start:])),
        "scale_floor_hit_rate":float(np.mean(t["scale_floor"][start:])),
        "scale_ceiling_hit_rate":float(np.mean(t["scale_ceiling"][start:])),
    }


def characteristic_step_epochs(result: Mapping[str, Any]) -> float | None:
    values = np.asarray(result["trace"]["mean_distance"], dtype=float)
    step = int(result["plant"]["step_epoch"])
    if step >= len(values)-8:
        return None
    post = values[step:]
    tail = float(np.mean(post[-max(5,len(post)//8):]))
    initial = float(post[0])
    target = tail + (initial-tail)/np.e
    hit = np.flatnonzero(post <= target)
    return float(hit[0]) if hit.size else None
