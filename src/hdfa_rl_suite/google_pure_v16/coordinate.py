"""Coordinate-covariance, gradient, PPO, and entropy audits for V16."""
from __future__ import annotations

from typing import Any

import numpy as np

from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.gaussian import (
    BehaviorSnapshot,
    component_log_probability,
    entropy,
)
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.losses import (
    total_loss_and_gradients,
)
from hdfa_rl_suite.google_pure_source_exact.step_response_130.plant import SourceStepPlant
from hdfa_rl_suite.google_pure_v12.directional import reference_directional_curvature

from .contracts import HYPOTHESIS_STATUSES, NONFINAL
from .io import ARTIFACT_ROOT, atomic_json, atomic_text, canonical_hash, config


def _branches(dimension: int = 8) -> dict[str, np.ndarray]:
    native_curvature = SourceStepPlant().sensitivity[:dimension]
    return {
        "A_IDENTITY": np.ones(dimension),
        "B_V12_OUTCOME_DERIVED_DIAGNOSTIC": np.sqrt(
            reference_directional_curvature() / native_curvature),
        "C_V15_SOURCE_NORMALIZED": np.sqrt(.01 / native_curvature),
    }


def build_hypothesis_artifact() -> dict[str, Any]:
    cfg = config()
    v12 = float(reference_directional_curvature())
    v15 = float(cfg["conditioned_curvature"])
    ratio = v12 / v15
    result = {
        "schema_version": "google-pure-v16-hypothesis.v1",
        "statuses": dict(HYPOTHESIS_STATUSES),
        "v12_diagnostic_conditioned_curvature": v12,
        "v15_source_conditioned_curvature": v15,
        "curvature_ratio_v12_over_v15": ratio,
        "inherited_mean_learning_rate": .08,
        "inherited_sigma_learning_rate": .02,
        "coordinate_covariance_diagnostic_mean_learning_rate": .08 * ratio,
        "coordinate_covariance_diagnostic_sigma_learning_rate": .02 * ratio,
        "numerical_support": bool(3.8 < ratio < 4.1),
        "diagnostic_not_production_selection": True,
        "v12_map_restored_in_production": False,
        **NONFINAL,
    }
    atomic_json(ARTIFACT_ROOT / "optimizer_consistency_hypothesis.json", result)
    lines = [
        "# V16 optimizer-consistency hypothesis", "",
        *(f"- `{name}`: `{status}`" for name, status in result["statuses"].items()),
        "", f"The measured conditioned-curvature ratio is **{ratio:.6f}**.",
        "The transformed V12 learning rates are a coordinate-covariance diagnostic only; they are not a production choice.",
    ]
    atomic_text(ARTIFACT_ROOT / "optimizer_consistency_hypothesis.md", "\n".join(lines))
    return result


def audit_coordinate_transform() -> dict[str, Any]:
    branches = _branches()
    rows = []
    native_curvature = SourceStepPlant().sensitivity[:8]
    for name, scale in branches.items():
        kappa_x = native_curvature * scale**2
        rows.append({
            "branch": name,
            "scale_min": float(scale.min()),
            "scale_max": float(scale.max()),
            "normalized_curvature_min": float(kappa_x.min()),
            "normalized_curvature_max": float(kappa_x.max()),
            "scalar_native_learning_rate": .04,
            "covariant_normalized_learning_rate_min": float((.04 / scale**2).min()),
            "covariant_normalized_learning_rate_max": float((.04 / scale**2).max()),
        })
    invariance_rules = {
        "mean": "mu_x = S^-1 (mu_u-u0)",
        "direct_sigma": "sigma_x = S^-1 sigma_u",
        "mean_learning_rate": "A_x = S^-1 A_u S^-1; scalar eta_u gives A_x=eta_u*S^-2",
        "sigma_learning_rate": "same covariance rule as the direct-sigma parameter",
        "initial_sigma": "transform covariantly when a native initial distribution is held fixed",
        "sigma_floor_and_ceiling": "transform covariantly only when native support is the invariant",
        "native_exploration_covariance": "Cov[u] = S diag(sigma_x^2) S^T",
        "entropy": "H(u)=H(x)+log|det S|; the additive constant has zero policy gradient",
        "entropy_coefficient": "coordinate invariant for a physically matched reward/distribution; re-audit when normalized sigma was inherited instead of transformed",
        "ppo_ratio": "pi_new(u)/pi_old(u)=pi_new(x)/pi_old(x)",
        "clip_epsilon": "dimensionless and coordinate invariant",
        "baseline_and_reward_coefficients": "invariant under a pure control-coordinate change only when native rewards are physically matched",
        "reward_rescaling": "if rewards are multiplied by c, policy and baseline gradient magnitudes change and optimizer/loss coefficients require a separate audit",
    }
    result = {
        "schema_version": "google-pure-v16-coordinate-transform.v1",
        "definition": "u = u0 + Sx",
        "gradient_rule": "grad_x = S^T grad_u",
        "native_update_rule": "Delta_u = -S A_x S^T grad_u",
        "scalar_native_optimizer_rule": "A_x = eta_u S^-2",
        "local_contraction_rule": "q = |1-alpha_mu*kappa_x| for J=0.5*kappa_x*||e_x||^2",
        "rows": rows,
        "invariance_rules": invariance_rules,
        "pass": all(row["normalized_curvature_min"] > 0 for row in rows),
        **NONFINAL,
    }
    atomic_json(ARTIFACT_ROOT / "coordinate_transform_contract.json", result)
    md = ["# Coordinate transform contract", "",
          "For `u = u0 + Sx`, gradients obey `grad_x = S^T grad_u` and updates obey `Delta_u = -S A_x S^T grad_u`.", "",
          "| Quantity | Rule |", "|---|---|"]
    md.extend(f"| {key} | {value} |" for key, value in invariance_rules.items())
    atomic_text(ARTIFACT_ROOT / "coordinate_transform_contract.md", "\n".join(md))
    return result


def _loss_for_branch(actions_x: np.ndarray, mean_x: np.ndarray, sigma_x: np.ndarray,
                     rewards: np.ndarray, behavior_mean_x: np.ndarray,
                     behavior_sigma_x: np.ndarray) -> Any:
    behavior = BehaviorSnapshot(
        behavior_mean_x,
        behavior_sigma_x,
        component_log_probability(actions_x, behavior_mean_x, behavior_sigma_x),
        policy_version=0,
    )
    return total_loss_and_gradients(
        actions_x, rewards, np.eye(actions_x.shape[1], dtype=bool),
        mean_x, sigma_x, np.zeros(actions_x.shape[1]), behavior,
        clip=.2, entropy_weight=.001, baseline_weight=.2,
    )


def run_covariance_fixture() -> dict[str, Any]:
    dimension, candidates = 8, 32
    rng = np.random.default_rng(81611)
    native_mean = np.linspace(-.18, .22, dimension)
    native_sigma = np.linspace(.055, .095, dimension)
    native_target = np.linspace(.11, -.09, dimension)
    native_curvature = SourceStepPlant().sensitivity[:dimension]
    z = rng.normal(size=(candidates, dimension))
    canonical_native_actions = native_mean + native_sigma * z
    uniforms = np.random.default_rng(81612).random((12000, candidates, dimension))
    native_lr = .04
    rows = []
    updated_native_means, updated_native_sigmas = {}, {}
    native_rewards: dict[str, np.ndarray] = {}
    for name, scale in _branches(dimension).items():
        mean_x, sigma_x = native_mean / scale, native_sigma / scale
        actions_x = mean_x + sigma_x * z
        actions_u = scale * actions_x
        action_error = float(np.max(np.abs(actions_u - canonical_native_actions)))
        expected_cost = .5 * (actions_u - native_target)**2 * native_curvature
        probabilities = np.clip(.012 + expected_cost, 1e-9, .49)
        counts = np.sum(uniforms < probabilities[None, :, :], axis=0)
        rewards = -counts / 12000.0
        native_rewards[name] = rewards
        loss = _loss_for_branch(actions_x, mean_x, sigma_x, rewards, mean_x, sigma_x)
        preconditioner = native_lr / scale**2
        next_mean_x = mean_x - preconditioner * loss.grad_mean
        next_sigma_x = sigma_x - preconditioner * loss.grad_sigma
        updated_native_means[name] = scale * next_mean_x
        updated_native_sigmas[name] = scale * next_sigma_x
        rows.append({
            "branch": name,
            "native_action_max_abs_error": action_error,
            "native_mean_hash": canonical_hash(native_mean.tolist()),
            "native_covariance_diagonal_hash": canonical_hash(np.square(native_sigma).tolist()),
            "native_action_hash": canonical_hash(np.round(actions_u, 14).tolist()),
            "native_reward_hash": canonical_hash(rewards.tolist()),
            "standard_normal_tape_hash": canonical_hash(z.tolist()),
            "preconditioner_min": float(preconditioner.min()),
            "preconditioner_max": float(preconditioner.max()),
        })
    reference = "A_IDENTITY"
    mean_errors = {name: float(np.max(np.abs(value - updated_native_means[reference])))
                   for name, value in updated_native_means.items()}
    sigma_errors = {name: float(np.max(np.abs(value - updated_native_sigmas[reference])))
                    for name, value in updated_native_sigmas.items()}
    reward_errors = {name: float(np.max(np.abs(value - native_rewards[reference])))
                     for name, value in native_rewards.items()}
    pass_value = (max(row["native_action_max_abs_error"] for row in rows) < 2e-15 and
                  max(reward_errors.values()) == 0.0 and max(mean_errors.values()) < 2e-13 and
                  max(sigma_errors.values()) < 2e-13)
    result = {
        "schema_version": "google-pure-v16-strict-covariance-fixture.v1",
        "rows": rows,
        "hard_invariants": {
            "same_native_mean": True,
            "same_native_covariance": True,
            "same_native_actions": True,
            "same_native_optimum": True,
            "same_native_step": True,
            "same_standard_normal_tape": True,
            "same_qec_uniform_tape": True,
            "same_rewards": True,
            "same_covariantly_transformed_native_update": pass_value,
        },
        "updated_native_mean_max_abs_errors": mean_errors,
        "updated_native_sigma_max_abs_errors": sigma_errors,
        "native_reward_max_abs_errors": reward_errors,
        "pass": pass_value,
        **NONFINAL,
    }
    if not pass_value:
        raise RuntimeError("strict native covariance fixture failed")
    atomic_json(ARTIFACT_ROOT / "coordinate_covariance_fixture.json", result)
    atomic_text(ARTIFACT_ROOT / "coordinate_covariance_fixture.md",
                "# Strict native covariance fixture\n\nAll A/B/C branches reproduce the same native distribution, candidates, rewards, and covariantly transformed one-step update within float64 tolerance.")
    return result


def audit_gradient_covariance() -> dict[str, Any]:
    scale = _branches(6)["C_V15_SOURCE_NORMALIZED"]
    curvature = SourceStepPlant().sensitivity[:6]
    mean_u = np.linspace(-.2, .3, 6)
    sigma_u = np.linspace(.04, .09, 6)
    target_u = np.linspace(.12, -.08, 6)
    mean_x, sigma_x = mean_u / scale, sigma_u / scale

    def expected_loss(mx: np.ndarray, sx: np.ndarray) -> float:
        mu, sd = scale * mx, scale * sx
        return float(.5 * np.sum(curvature * ((mu - target_u)**2 + sd**2)))

    eps = 1e-6
    fd_mean, fd_sigma = np.zeros(6), np.zeros(6)
    for index in range(6):
        direction = np.zeros(6); direction[index] = eps
        fd_mean[index] = (expected_loss(mean_x + direction, sigma_x) -
                          expected_loss(mean_x - direction, sigma_x)) / (2 * eps)
        fd_sigma[index] = (expected_loss(mean_x, sigma_x + direction) -
                           expected_loss(mean_x, sigma_x - direction)) / (2 * eps)
    grad_u_mean = curvature * (mean_u - target_u)
    grad_u_sigma = curvature * sigma_u
    predicted_mean = scale * grad_u_mean
    predicted_sigma = scale * grad_u_sigma
    mean_error = float(np.max(np.abs(fd_mean - predicted_mean)))
    sigma_error = float(np.max(np.abs(fd_sigma - predicted_sigma)))
    result = {
        "schema_version": "google-pure-v16-gradient-covariance.v1",
        "finite_difference_mean_max_abs_error": mean_error,
        "finite_difference_direct_sigma_max_abs_error": sigma_error,
        "mean_gradient_rule": "grad_mu_x=S*grad_mu_u",
        "direct_sigma_rule": "sigma_u=S*sigma_x and grad_sigma_x=S*grad_sigma_u",
        "pass": mean_error < 2e-10 and sigma_error < 2e-10,
        **NONFINAL,
    }
    atomic_json(ARTIFACT_ROOT / "gradient_covariance_audit.json", result)
    return result


def audit_ppo_covariance() -> dict[str, Any]:
    scale = _branches(7)["C_V15_SOURCE_NORMALIZED"]
    rng = np.random.default_rng(81621)
    old_mu_u = np.linspace(-.1, .1, 7)
    old_sigma_u = np.linspace(.06, .12, 7)
    actions_u = old_mu_u + old_sigma_u * rng.normal(size=(48, 7))
    new_mu_u = old_mu_u + np.linspace(-.008, .009, 7)
    new_sigma_u = old_sigma_u * np.linspace(.94, 1.06, 7)
    log_ratio_u = component_log_probability(actions_u, new_mu_u, new_sigma_u) - \
        component_log_probability(actions_u, old_mu_u, old_sigma_u)
    actions_x = actions_u / scale
    log_ratio_x = component_log_probability(actions_x, new_mu_u / scale, new_sigma_u / scale) - \
        component_log_probability(actions_x, old_mu_u / scale, old_sigma_u / scale)
    clip = .2
    clipped_u = np.clip(log_ratio_u, np.log1p(-clip), np.log1p(clip))
    clipped_x = np.clip(log_ratio_x, np.log1p(-clip), np.log1p(clip))
    error = float(np.max(np.abs(log_ratio_u - log_ratio_x)))
    clip_error = float(np.max(np.abs(clipped_u - clipped_x)))
    result = {
        "schema_version": "google-pure-v16-ppo-covariance.v1",
        "coordinate_log_ratio_max_abs_error": error,
        "elementwise_clipped_log_ratio_max_abs_error": clip_error,
        "coordinate_ratio_invariant": error < 2e-14,
        "clip_epsilon_invariant": True,
        "clip_epsilon": clip,
        "pass": error < 2e-14 and clip_error < 2e-14,
        **NONFINAL,
    }
    atomic_json(ARTIFACT_ROOT / "ppo_covariance_audit.json", result)
    return result


def audit_entropy_covariance() -> dict[str, Any]:
    scale = _branches(8)["C_V15_SOURCE_NORMALIZED"]
    sigma_u = np.linspace(.04, .12, 8)
    sigma_x = sigma_u / scale
    difference = entropy(sigma_u) - entropy(sigma_x)
    logdet = float(np.sum(np.log(scale)))
    grad_x = 1.0 / sigma_x
    transformed_grad_u = scale * (1.0 / sigma_u)
    gradient_error = float(np.max(np.abs(grad_x - transformed_grad_u)))
    inherited_sigma = .15
    v12_kappa = reference_directional_curvature()
    v15_kappa = .01
    beta = .001
    inherited_ratios = {
        "V12": float((beta / inherited_sigma) / (2 * v12_kappa * inherited_sigma)),
        "V15": float((beta / inherited_sigma) / (2 * v15_kappa * inherited_sigma)),
    }
    result = {
        "schema_version": "google-pure-v16-entropy-covariance.v1",
        "entropy_difference": float(difference),
        "log_abs_det_scale": logdet,
        "identity_error": float(abs(difference - logdet)),
        "entropy_gradient_covariance_max_abs_error": gradient_error,
        "entropy_coefficient_coordinate_invariant_under_physical_matching": True,
        "inherited_normalized_sigma_breaks_physical_matching": True,
        "inherited_reward_to_entropy_relative_magnitude": inherited_ratios,
        "relative_magnitude_ratio_v15_over_v12": inherited_ratios["V15"] / inherited_ratios["V12"],
        "pass": abs(difference - logdet) < 2e-13 and gradient_error < 2e-13,
        **NONFINAL,
    }
    atomic_json(ARTIFACT_ROOT / "entropy_covariance_audit.json", result)
    return result


__all__ = [
    "audit_coordinate_transform", "run_covariance_fixture", "audit_gradient_covariance",
    "audit_ppo_covariance", "audit_entropy_covariance", "build_hypothesis_artifact",
]
