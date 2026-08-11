from __future__ import annotations

import numpy as np
import pytest

from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.contracts import (
    JOINT_LEARNED_DETECTOR_BASELINE,
    NON_SOURCE_EMA_BASELINE_ABLATION,
    NON_SOURCE_PPO_ABLATION,
    SOURCE_ELEMENTWISE_COORDINATE_CLIPPING,
)
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.gaussian import (
    BehaviorSnapshot,
    DirectSigmaGaussianPolicy,
    component_log_probability,
)
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.losses import (
    ema_baseline_update,
    total_loss_and_gradients,
)
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.optimizer import (
    DirectSigmaOptimizer,
    OptimizerConfig,
)
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.validation import finite_difference
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.validation import (
    baseline_dynamics_audit,
    source_loss_semantics_audit,
)


def _fixture(*, controls: int = 2, detectors: int = 1, candidates: int = 3):
    mean = np.zeros(controls)
    sigma = np.ones(controls)
    actions = np.tile(np.linspace(0.1, 0.3, candidates)[:, None], (1, controls))
    current = component_log_probability(actions, mean, sigma)
    behavior = BehaviorSnapshot(mean, sigma, current.copy(), 7)
    rewards = np.ones((candidates, detectors))
    mask = np.ones((detectors, controls), dtype=bool)
    baseline = np.zeros(detectors)
    return actions, rewards, mask, mean, sigma, baseline, behavior


def test_one_coordinate_elementwise_and_aggregate_clipping_are_equivalent() -> None:
    args = _fixture(controls=1)
    behavior = BehaviorSnapshot(np.zeros(1), np.ones(1),
                                args[-1].component_log_probability - np.log(1.5), 7)
    common = dict(actions=args[0], rewards=args[1], mask=args[2], mean=args[3], sigma=args[4],
                  baseline=args[5], behavior=behavior, clip=0.2, baseline_weight=0, entropy_weight=0,
                  paper_mode=False)
    source = total_loss_and_gradients(**common, ratio_clipping_mode=SOURCE_ELEMENTWISE_COORDINATE_CLIPPING)
    ablation = total_loss_and_gradients(**common, ratio_clipping_mode=NON_SOURCE_PPO_ABLATION)
    assert source.policy == ablation.policy


def test_multi_coordinate_clipping_precedes_product_and_differs_from_aggregate() -> None:
    args = _fixture(controls=3)
    behavior = BehaviorSnapshot(np.zeros(3), np.ones(3),
                                args[-1].component_log_probability - np.log(1.15), 7)
    common = dict(actions=args[0], rewards=args[1], mask=args[2], mean=args[3], sigma=args[4],
                  baseline=args[5], behavior=behavior, clip=0.2, baseline_weight=0, entropy_weight=0,
                  paper_mode=False)
    source = total_loss_and_gradients(**common, ratio_clipping_mode=SOURCE_ELEMENTWISE_COORDINATE_CLIPPING)
    ablation = total_loss_and_gradients(**common, ratio_clipping_mode=NON_SOURCE_PPO_ABLATION)
    assert source.policy == pytest.approx(-(1.15 ** 3))
    assert ablation.policy == pytest.approx(-1.2)
    assert source.policy != ablation.policy
    assert source.diagnostics["coordinate_ratios_clipped_before_sparse_product"]


def test_current_equals_behavior_ratios_are_one_and_shapes_are_explicit() -> None:
    args = _fixture(controls=4, detectors=2)
    result = total_loss_and_gradients(actions=args[0], rewards=args[1], mask=args[2], mean=args[3],
                                      sigma=args[4], baseline=args[5], behavior=args[6], clip=0.2,
                                      baseline_weight=0, entropy_weight=0)
    assert result.diagnostics["ratio_mean"] == 1.0
    assert result.diagnostics["tensor_shapes"]["coordinate_ratios"] == [3, 4]
    assert result.diagnostics["tensor_shapes"]["masked_detector_ratios"] == [3, 2]


def test_mask_locality_gives_exact_zero_unrelated_reward_gradient() -> None:
    mean, sigma = np.zeros(2), np.ones(2)
    actions = np.asarray([[0.2, 0.4], [-0.1, 0.3]])
    current = component_log_probability(actions, mean, sigma)
    behavior = BehaviorSnapshot(mean, sigma, current - 0.05, 2)
    result = total_loss_and_gradients(actions, np.asarray([[1.0], [2.0]]),
                                      np.asarray([[True, False]]), mean, sigma, np.zeros(1), behavior,
                                      clip=0.2, baseline_weight=0, entropy_weight=0)
    assert result.grad_mean[1] == 0.0
    assert result.grad_sigma[1] == 0.0


def test_coordinate_clipping_boundary_uses_zero_outside_gradient() -> None:
    args = _fixture(controls=1, candidates=1)
    behavior = BehaviorSnapshot(np.zeros(1), np.ones(1),
                                args[-1].component_log_probability - np.log(1.2), 7)
    result = total_loss_and_gradients(actions=args[0], rewards=args[1], mask=args[2], mean=args[3],
                                      sigma=args[4], baseline=args[5], behavior=behavior, clip=0.2,
                                      baseline_weight=0, entropy_weight=0)
    assert result.grad_mean[0] == 0.0 and result.grad_sigma[0] == 0.0


def test_detector_degree_does_not_trigger_aggregate_saturation_in_source_branch() -> None:
    controls = 20
    args = _fixture(controls=controls, candidates=1)
    behavior = BehaviorSnapshot(np.zeros(controls), np.ones(controls),
                                args[-1].component_log_probability - np.log(1.01), 7)
    common = dict(actions=args[0], rewards=args[1], mask=args[2], mean=args[3], sigma=args[4],
                  baseline=args[5], behavior=behavior, clip=0.2, baseline_weight=0, entropy_weight=0,
                  paper_mode=False)
    source = total_loss_and_gradients(**common, ratio_clipping_mode=SOURCE_ELEMENTWISE_COORDINATE_CLIPPING)
    ablation = total_loss_and_gradients(**common, ratio_clipping_mode=NON_SOURCE_PPO_ABLATION)
    assert np.linalg.norm(source.grad_mean) > 0
    assert np.linalg.norm(ablation.grad_mean) == 0


def test_paper_mode_rejects_both_non_source_ablations() -> None:
    args = _fixture()
    common = dict(actions=args[0], rewards=args[1], mask=args[2], mean=args[3], sigma=args[4],
                  baseline=args[5], behavior=args[6], clip=0.2)
    with pytest.raises(ValueError, match="clipped before"):
        total_loss_and_gradients(**common, ratio_clipping_mode=NON_SOURCE_PPO_ABLATION)
    with pytest.raises(ValueError, match="jointly learned"):
        total_loss_and_gradients(**common, baseline_mode=NON_SOURCE_EMA_BASELINE_ABLATION)
    with pytest.raises(ValueError, match="jointly learned"):
        ema_baseline_update(np.zeros(1), np.ones((2, 1)), 0.1, paper_mode=True)


def test_replay_uses_immutable_collection_density_and_policy_version() -> None:
    policy = DirectSigmaGaussianPolicy(np.zeros(2), np.ones(2), seed=3)
    batch = policy.sample(4)
    policy.mean[:] = 0.4
    policy.policy_version = 1
    result = total_loss_and_gradients(batch.actions, np.ones((4, 1)), np.ones((1, 2), dtype=bool),
                                      policy.mean, policy.sigma, np.zeros(1), batch.behavior,
                                      clip=0.2, baseline_weight=0, entropy_weight=0)
    assert batch.behavior.policy_version == 0
    assert not batch.behavior.component_log_probability.flags.writeable
    assert result.diagnostics["behavior_snapshot_writeable"] is False


def test_learned_detector_baseline_gradient_optimum_and_finite_difference() -> None:
    args = _fixture(controls=2, detectors=2, candidates=4)
    rewards = np.asarray([[1.0, 4.0], [2.0, 6.0], [3.0, 8.0], [4.0, 10.0]])
    optimum = rewards.mean(axis=0)
    common = dict(actions=args[0], rewards=rewards, mask=args[2], mean=args[3], sigma=args[4],
                  behavior=args[6], clip=0.2, policy_weight=0, baseline_weight=0.7, entropy_weight=0)
    result = total_loss_and_gradients(baseline=optimum, **common)
    np.testing.assert_allclose(result.grad_baseline, 0, atol=1e-15)
    start = np.asarray([0.2, -0.1])
    analytic = total_loss_and_gradients(baseline=start, **common).grad_baseline
    numeric = finite_difference(lambda b: total_loss_and_gradients(baseline=b, **common).total, start)
    np.testing.assert_allclose(analytic, numeric, atol=2e-7)


def test_baseline_is_preupdate_batch_frozen_permutation_invariant_and_action_independent() -> None:
    rng = np.random.default_rng(4)
    actions = rng.normal(size=(30, 2)); rewards = rng.normal(size=(30, 3))
    mean = np.zeros(2); sigma = np.ones(2); base = np.asarray([0.2, -0.1, 0.4])
    logp = component_log_probability(actions, mean, sigma)
    behavior = BehaviorSnapshot(mean, sigma, logp, 4)
    mask = np.ones((3, 2), dtype=bool)
    common = dict(mask=mask, mean=mean, sigma=sigma, baseline=base, clip=0.2,
                  policy_weight=0, entropy_weight=0, baseline_weight=0.5)
    original = total_loss_and_gradients(actions=actions, rewards=rewards, behavior=behavior, **common)
    order = rng.permutation(len(actions))
    permuted_behavior = BehaviorSnapshot(mean, sigma, logp[order], 4)
    permuted = total_loss_and_gradients(actions=actions[order], rewards=rewards[order],
                                        behavior=permuted_behavior, **common)
    np.testing.assert_allclose(original.grad_baseline, permuted.grad_baseline, atol=1e-15)
    changed_actions = actions + 10.0
    changed_behavior = BehaviorSnapshot(mean, sigma,
                                        component_log_probability(changed_actions, mean, sigma), 4)
    action_changed = total_loss_and_gradients(actions=changed_actions, rewards=rewards,
                                              behavior=changed_behavior, **common)
    np.testing.assert_array_equal(original.grad_baseline, action_changed.grad_baseline)


def test_joint_optimizer_has_one_baseline_parameter_per_detector() -> None:
    optimizer = DirectSigmaOptimizer(2, 3, OptimizerConfig(0.1, 0.1, 0.1))
    assert optimizer.baseline_velocity.shape == (3,)
    assert JOINT_LEARNED_DETECTOR_BASELINE == "JOINT_LEARNED_DETECTOR_BASELINE"


def test_machine_readable_loss_and_baseline_audits_pass() -> None:
    loss = source_loss_semantics_audit()
    baseline = baseline_dynamics_audit()
    assert loss["pass"] and loss["coordinate_clip_before_product"]
    assert loss["multi_coordinate_non_equivalence"]
    assert baseline["pass"] and baseline["causal_preupdate_baseline_logged"]
    assert baseline["source_consistent_variance"]
