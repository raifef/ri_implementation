import numpy as np
import pytest

from google_rl_reimplementation.google_rl_certification.agent import (
    CandidateEvaluation,
    GaussianPolicyGradientAgent,
)
from google_rl_reimplementation.google_rl_certification.analytic_landscape import (
    run_analytic_certification,
)
from google_rl_reimplementation.google_rl_certification.config import named_config
from google_rl_reimplementation.google_rl_certification.static_detector_landscape import (
    make_static_landscape,
)


def _agent(configuration: str = "high_shot_reference", seed: int = 7):
    landscape = make_static_landscape()
    agent = GaussianPolicyGradientAgent(
        landscape.control_ids,
        landscape.detector_ids,
        landscape.mask,
        landscape.sensitivity_scales,
        np.zeros(len(landscape.control_ids)),
        named_config(configuration),
        seed=seed,
    )
    return landscape, agent


def _evaluations(landscape, batch):
    losses = landscape.expected_rates(batch.actions_native)
    return tuple(
        CandidateEvaluation(candidate_id, losses[index])
        for index, candidate_id in enumerate(batch.candidate_ids)
    )



def test_public_high_shot_sampling_structure_is_exact_and_versioned():
    config = named_config("high_shot_reference")
    assert config.sampling.candidates_per_epoch == 40
    assert config.sampling.shots_per_candidate == 4000
    assert config.sampling.qec_cycles_per_shot == 25
    assert config.sampling.effective_cycles_per_candidate == 100_000
    assert config.sampling.candidate_design == "independent_gaussian"


def test_reward_and_loss_sign_converge_from_both_sides():
    result = run_analytic_certification(named_config("high_shot_reference"), seed=31)
    assert result["gates"]["converges_from_both_sides"]
    assert result["gates"]["positive_gradient_alignment"]


def test_candidate_reward_association_uses_ids_not_sequence_order():
    landscape_a, agent_a = _agent(seed=41)
    landscape_b, agent_b = _agent(seed=41)
    batch_a = agent_a.sample_candidates()
    batch_b = agent_b.sample_candidates()
    assert batch_a.candidate_ids == batch_b.candidate_ids
    evaluations = _evaluations(landscape_a, batch_a)
    agent_a.update(batch_a, evaluations)
    agent_b.update(batch_b, tuple(reversed(evaluations)))
    np.testing.assert_array_equal(agent_a.mean, agent_b.mean)
    np.testing.assert_array_equal(agent_a.log_stddev, agent_b.log_stddev)


def test_unknown_duplicate_and_stale_candidate_rewards_are_rejected():
    landscape, agent = _agent(seed=43)
    batch = agent.sample_candidates()
    evaluations = list(_evaluations(landscape, batch))
    with pytest.raises(ValueError, match="duplicate"):
        agent.update(batch, evaluations[:-1]+[evaluations[0], evaluations[0]])
    agent.update(batch, evaluations)
    with pytest.raises(ValueError, match="stale"):
        agent.update(batch, evaluations)


def test_transposed_mask_and_detector_region_index_errors_fail_closed():
    landscape = make_static_landscape()
    with pytest.raises(ValueError, match="mask has shape"):
        GaussianPolicyGradientAgent(
            landscape.control_ids, landscape.detector_ids, landscape.mask.T,
            landscape.sensitivity_scales, np.zeros(len(landscape.control_ids)),
            named_config("high_shot_reference"))
    duplicate_detectors = tuple([landscape.detector_ids[0]]*len(landscape.detector_ids))
    with pytest.raises(ValueError, match="identifiers must be unique"):
        GaussianPolicyGradientAgent(
            landscape.control_ids, duplicate_detectors, landscape.mask,
            landscape.sensitivity_scales, np.zeros(len(landscape.control_ids)),
            named_config("high_shot_reference"))


def test_sensitivity_units_are_explicit_positive_and_shape_checked():
    landscape = make_static_landscape()
    with pytest.raises(ValueError, match="means/scales"):
        GaussianPolicyGradientAgent(
            landscape.control_ids, landscape.detector_ids, landscape.mask,
            np.ones(2), np.zeros(len(landscape.control_ids)),
            named_config("high_shot_reference"))
    bad = landscape.sensitivity_scales.copy()
    bad[0] = -1
    with pytest.raises(ValueError, match="finite and positive"):
        GaussianPolicyGradientAgent(
            landscape.control_ids, landscape.detector_ids, landscape.mask,
            bad, np.zeros(len(landscape.control_ids)),
            named_config("high_shot_reference"))


def test_high_shot_is_independent_and_reduced_pairs_are_exactly_centred():
    _, high = _agent("high_shot_reference", seed=47)
    _, reduced = _agent("reduced_budget_candidate", seed=47)
    high_batch = high.sample_candidates()
    reduced_batch = reduced.sample_candidates()
    assert not np.allclose(
        high_batch.standardized_perturbations[0::2]
        + high_batch.standardized_perturbations[1::2], 0.0)
    np.testing.assert_allclose(
        reduced_batch.actions_normalized[0::2]
        + reduced_batch.actions_normalized[1::2],
        np.broadcast_to(2*reduced_batch.collection_mean,
                        reduced_batch.actions_normalized[0::2].shape), atol=1e-14)


def test_covariance_respects_entropy_floor_and_does_not_explode():
    landscape, agent = _agent(seed=53)
    for _ in range(20):
        batch = agent.sample_candidates()
        agent.update(batch, _evaluations(landscape, batch))
    config = named_config("high_shot_reference")
    assert np.all(agent.stddev >= config.policy.minimum_stddev_normalized-1e-12)
    assert np.all(agent.stddev <= config.policy.maximum_stddev_normalized+1e-12)


def test_agents_do_not_share_policy_state_between_arms():
    landscape, first = _agent(seed=59)
    _, second = _agent(seed=59)
    second_before = second.mean.copy()
    batch = first.sample_candidates()
    first.update(batch, _evaluations(landscape, batch))
    np.testing.assert_array_equal(second.mean, second_before)
    assert not np.shares_memory(first.mean, second.mean)

