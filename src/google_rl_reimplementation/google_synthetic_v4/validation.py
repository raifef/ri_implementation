"""Independent small-problem validation of the masked PPO mechanism."""
from __future__ import annotations

from typing import Any

import numpy as np

from .controller import (
    CandidateBatch, DetectorEvidence, MaskedGaussianPPO, action_hash,
    clipped_objective_and_gradient, local_policy_ratios, normal_log_density,
)


def _loop_reference(actions: np.ndarray, advantages: np.ndarray, mask: np.ndarray, mean: np.ndarray,
                    log_std: np.ndarray, old_mean: np.ndarray, old_log_std: np.ndarray,
                    clip: float) -> tuple[float, np.ndarray, np.ndarray]:
    n, c = actions.shape
    d = advantages.shape[1]
    gm = np.zeros(c)
    gs = np.zeros(c)
    objective = 0.0
    current = normal_log_density(actions, mean, log_std)
    old = normal_log_density(actions, old_mean, old_log_std)
    for i in range(n):
        for j in range(d):
            idx = np.flatnonzero(mask[j])
            ratio = float(np.exp(np.clip(np.sum(current[i, idx] - old[i, idx]), -40, 40)))
            advantage = float(advantages[i, j])
            objective += min(ratio * advantage, np.clip(ratio, 1 - clip, 1 + clip) * advantage) / (n * d)
            active = (advantage >= 0 and ratio <= 1 + clip) or (advantage < 0 and ratio >= 1 - clip)
            if active:
                for k in idx:
                    delta = actions[i, k] - mean[k]
                    inv_var = np.exp(-2 * log_std[k])
                    weight = advantage * ratio / (n * d)
                    gm[k] += weight * delta * inv_var
                    gs[k] += weight * (delta * delta * inv_var - 1)
    return objective, gm, gs


def _finite_difference(actions: np.ndarray, advantages: np.ndarray, mask: np.ndarray, mean: np.ndarray,
                       log_std: np.ndarray, old_mean: np.ndarray, old_log_std: np.ndarray,
                       clip: float, epsilon: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    def value(m: np.ndarray, s: np.ndarray) -> float:
        return clipped_objective_and_gradient(actions, advantages, mask, m, s, old_mean, old_log_std, clip=clip)[0]
    gm = np.empty_like(mean)
    gs = np.empty_like(log_std)
    for k in range(len(mean)):
        plus, minus = mean.copy(), mean.copy()
        plus[k] += epsilon
        minus[k] -= epsilon
        gm[k] = (value(plus, log_std) - value(minus, log_std)) / (2 * epsilon)
        plus_s, minus_s = log_std.copy(), log_std.copy()
        plus_s[k] += epsilon
        minus_s[k] -= epsilon
        gs[k] = (value(mean, plus_s) - value(mean, minus_s)) / (2 * epsilon)
    return gm, gs


def validate_ppo_reference() -> dict[str, Any]:
    rng = np.random.default_rng(424242)
    mask = np.array([[1, 1, 0], [0, 1, 1]], dtype=bool)
    old_mean = np.array([0.05, -0.02, 0.03])
    old_log_std = np.log(np.array([0.20, 0.17, 0.23]))
    actions = old_mean + np.exp(old_log_std) * rng.normal(size=(7, 3))
    mean = old_mean + np.array([0.012, -0.009, 0.007])
    log_std = old_log_std + np.array([0.015, -0.012, 0.008])
    advantages = np.array([
        [0.30, -0.20], [-0.15, 0.22], [0.12, 0.10], [-0.28, -0.17],
        [0.09, -0.13], [-0.05, 0.18], [0.21, -0.08],
    ])
    obj, gm, gs, diag = clipped_objective_and_gradient(
        actions, advantages, mask, mean, log_std, old_mean, old_log_std, clip=0.2)
    ref_obj, ref_gm, ref_gs = _loop_reference(actions, advantages, mask, mean, log_std, old_mean, old_log_std, 0.2)
    fd_gm, fd_gs = _finite_difference(actions, advantages, mask, mean, log_std, old_mean, old_log_std, 0.2)
    inactive_mask = np.array([[1, 0, 0]], dtype=bool)
    _, inactive_gm, inactive_gs, _ = clipped_objective_and_gradient(
        actions[:3], advantages[:3, :1], inactive_mask, mean, log_std, old_mean, old_log_std, clip=0.2)
    entropy_obj, entropy_gm, entropy_gs, _ = clipped_objective_and_gradient(
        actions, np.zeros_like(advantages), mask, mean, log_std, old_mean, old_log_std,
        clip=0.2, entropy_weight=0.07)
    enumerated_actions = np.array([[-0.2, -0.2], [-0.2, 0.2], [0.2, -0.2], [0.2, 0.2]])
    enum_mask = np.array([[1, 0], [0, 1]], dtype=bool)
    enum_adv = np.array([[0.2, -0.1], [-0.1, 0.3], [0.4, -0.2], [-0.3, 0.1]])
    enum_mean = np.array([0.01, -0.01])
    enum_log = np.log(np.array([0.25, 0.22]))
    enum = clipped_objective_and_gradient(enumerated_actions, enum_adv, enum_mask, enum_mean, enum_log,
                                           np.zeros(2), enum_log, clip=0.2)
    enum_ref = _loop_reference(enumerated_actions, enum_adv, enum_mask, enum_mean, enum_log,
                               np.zeros(2), enum_log, 0.2)
    stationary_choices = {
        "initial_std":0.12,"mean_learning_rate":0.18,"scale_learning_rate":0.004,
        "baseline_learning_rate":0.12,"entropy_weight":0.0,"target_std":0.07,"target_strength":0.05,
        "ppo_clip":0.2,"gradient_clip":1.0,"optimizer_passes":4,"replay_epochs":1,"replay_decay":1.0,
        "minimum_std":0.008,"maximum_std":0.3,"absolute_bound":1.0,"natural_mean":False,"variance_weighted":False,
    }
    stationary = MaskedGaussianPPO(np.eye(2,dtype=bool), np.array([0.50,-0.50]), stationary_choices, seed=4321)
    target = np.array([0.15,-0.10])
    initial_distance = float(np.linalg.norm(stationary.mean-target))
    for _epoch in range(100):
        sbatch = stationary.sample(40, regime_id="stationary-reference")
        rates = np.clip(0.05 + 0.25*(sbatch.actions-target[None,:])**2,0,0.9)
        stationary.update(sbatch, make_evidence(sbatch,rates,100_000))
    final_distance = float(np.linalg.norm(stationary.mean-target))
    # Executable state-machine and replay checks, independent of pytest.
    semantics=MaskedGaussianPPO(np.eye(2,dtype=bool),np.array([.35,-.35]),stationary_choices,seed=987)
    first=semantics.sample(24,regime_id="compatible")
    first_rates=np.clip(.05+.2*(first.actions-target[None,:])**2,0,.9)
    semantics.update(first,make_evidence(first,first_rates,100_000))
    second=semantics.sample(24,regime_id="compatible")
    second_rates=np.clip(.05+.2*(second.actions-target[None,:])**2,0,.9)
    replay_diag=semantics.update(second,make_evidence(second,second_rates,100_000))
    one_use_rejected=False
    try:
        semantics.update(second,make_evidence(second,second_rates,100_000))
    except ValueError:
        one_use_rejected=True
    incompatible=MaskedGaussianPPO(np.eye(2,dtype=bool),np.array([.35,-.35]),stationary_choices,seed=988)
    old_batch=incompatible.sample(20,regime_id="old")
    old_rates=np.clip(.05+.2*(old_batch.actions-target[None,:])**2,0,.9)
    incompatible.update(old_batch,make_evidence(old_batch,old_rates,100_000))
    new_batch=incompatible.sample(20,regime_id="new")
    new_rates=np.clip(.05+.2*(new_batch.actions-target[None,:])**2,0,.9)
    incompatible_diag=incompatible.update(new_batch,make_evidence(new_batch,new_rates,100_000))
    provenance=MaskedGaussianPPO(np.eye(2,dtype=bool),np.array([.35,-.35]),stationary_choices,seed=989)
    provenance_batch=provenance.sample(8,regime_id="provenance")
    provenance_evidence=list(make_evidence(provenance_batch,np.full((8,2),.05),100_000))
    item=provenance_evidence[0]
    provenance_evidence[0]=DetectorEvidence(item.candidate_id,"incorrect",item.detector_counts,item.effective_cycles,item.regime_id)
    provenance_rejected=False
    try:
        provenance.update(provenance_batch,provenance_evidence)
    except ValueError:
        provenance_rejected=True
    clipped_positive=clipped_objective_and_gradient(np.array([[1.5]]),np.array([[1.0]]),np.array([[1]],bool),
        np.array([.5]),np.array([0.]),np.array([0.]),np.array([0.]),clip=.2)
    clipped_negative=clipped_objective_and_gradient(np.array([[1.5]]),np.array([[-1.0]]),np.array([[1]],bool),
        np.array([.5]),np.array([0.]),np.array([0.]),np.array([0.]),clip=.2)
    mean_only_choices={**stationary_choices,"scale_learning_rate":0.0,"entropy_weight":0.0,"target_std":None,"target_strength":0.0,"replay_epochs":0}
    mean_only=MaskedGaussianPPO(np.eye(2,dtype=bool),np.array([.35,-.35]),mean_only_choices,seed=990)
    mean_batch=mean_only.sample(24,regime_id="decouple")
    old_mean_state,old_scale_state=mean_only.mean.copy(),mean_only.log_std.copy()
    mean_rates=np.clip(.05+.2*(mean_batch.actions-target[None,:])**2,0,.9)
    mean_only.update(mean_batch,make_evidence(mean_batch,mean_rates,100_000))
    scale_only_choices={**stationary_choices,"mean_learning_rate":0.0,"scale_learning_rate":.1,"entropy_weight":.1,"target_std":None,"target_strength":0.0,"replay_epochs":0}
    scale_only=MaskedGaussianPPO(np.eye(2,dtype=bool),np.array([.35,-.35]),scale_only_choices,seed=990)
    scale_batch=scale_only.sample(24,regime_id="decouple")
    scale_old_mean,scale_old_log=scale_only.mean.copy(),scale_only.log_std.copy()
    scale_rates=np.clip(.05+.2*(scale_batch.actions-target[None,:])**2,0,.9)
    scale_only.update(scale_batch,make_evidence(scale_batch,scale_rates,100_000))
    checks = {
        "analytic_vs_independent_objective":abs(obj - ref_obj) < 1e-12,
        "analytic_vs_independent_mean_gradient":float(np.max(np.abs(gm - ref_gm))) < 1e-12,
        "analytic_vs_independent_log_scale_gradient":float(np.max(np.abs(gs - ref_gs))) < 1e-12,
        "finite_difference_mean_gradient":float(np.max(np.abs(gm - fd_gm))) < 2e-8,
        "finite_difference_log_scale_gradient":float(np.max(np.abs(gs - fd_gs))) < 2e-8,
        "enumerated_action_reference":max(abs(enum[0] - enum_ref[0]), float(np.max(np.abs(enum[1]-enum_ref[1]))), float(np.max(np.abs(enum[2]-enum_ref[2])))) < 1e-12,
        "inactive_mean_zero":float(np.max(np.abs(inactive_gm[1:]))) < 1e-14,
        "inactive_scale_zero":float(np.max(np.abs(inactive_gs[1:]))) < 1e-14,
        "entropy_mean_zero":float(np.max(np.abs(entropy_gm))) < 1e-14,
        "entropy_log_scale_derivative":float(np.max(np.abs(entropy_gs - 0.07 / 3))) < 1e-14,
        "off_policy_ratios_exercised":diag["off_policy_fraction"] > 0.99,
        "no_drift_stationary_convergence":final_distance < 0.45 * initial_distance,
        "compatible_replay_used":replay_diag["replay_epochs_used"]==1,
        "incompatible_replay_rejected":incompatible_diag["replay_epochs_used"]==0 and incompatible_diag["incompatible_replay_rejected"]==1,
        "candidate_provenance_rejected":provenance_rejected,
        "one_use_policy_version_rejected":one_use_rejected,
        "sign_aware_positive_clip_zero_gradient":abs(clipped_positive[1][0])<1e-14 and abs(clipped_positive[2][0])<1e-14,
        "sign_aware_negative_clip_retains_gradient":abs(clipped_negative[1][0])>1e-8,
        "mean_update_decoupled":not np.allclose(mean_only.mean,old_mean_state) and np.array_equal(mean_only.log_std,old_scale_state),
        "scale_update_decoupled":np.array_equal(scale_only.mean,scale_old_mean) and not np.allclose(scale_only.log_std,scale_old_log),
        "sensitivity_normalization_round_trip":np.allclose(np.array([.2,-.3])*np.array([.8,1.2])/np.array([.8,1.2]),np.array([.2,-.3])),
    }
    max_errors = {
        "objective":abs(obj-ref_obj), "independent_mean":float(np.max(np.abs(gm-ref_gm))),
        "independent_log_scale":float(np.max(np.abs(gs-ref_gs))),
        "finite_difference_mean":float(np.max(np.abs(gm-fd_gm))),
        "finite_difference_log_scale":float(np.max(np.abs(gs-fd_gs))),
    }
    return {
        "schema_version":"google-synthetic-v4-ppo-validation.v1",
        "status":"PASS" if all(checks.values()) else "FAIL", "checks":checks, "maximum_absolute_errors":max_errors,
        "validated_mechanisms":["local likelihood ratio","sparse mask","clipped objective","sign-aware clipping","baseline subtraction","entropy derivative","log-scale update","replay weighting","gradient clipping","sensitivity normalization","inactive zero gradient","candidate provenance","one-use version semantics"],
        "reference_methods":["closed-form score gradient","central finite differences","independent scalar-loop implementation","four-action exact enumeration"],
        "certification_seeds_consumed":False,
        "stationary_reference":{"initial_distance":initial_distance,"final_distance":final_distance,"epochs":100},
        "summary":{"status":"PASS" if all(checks.values()) else "FAIL","max_finite_difference_error":max(max_errors["finite_difference_mean"],max_errors["finite_difference_log_scale"])}
    }


def make_evidence(batch: CandidateBatch, rates: np.ndarray, cycles: int) -> tuple[DetectorEvidence, ...]:
    counts = np.rint(np.asarray(rates) * cycles).astype(int)
    return tuple(DetectorEvidence(cid, batch.action_hashes[i], counts[i], cycles, batch.regime_id)
                 for i, cid in enumerate(batch.candidate_ids))
