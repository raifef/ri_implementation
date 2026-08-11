from dataclasses import replace

import numpy as np
import pytest

from hdfa_rl_suite.baselines.controllers import FullControlRLArm
from hdfa_rl_suite.google_rl_certification.agent import (
    CandidateEvaluation,
    GaussianPolicyGradientAgent,
)
from hdfa_rl_suite.google_rl_certification.analytic_landscape import (
    run_analytic_certification,
)
from hdfa_rl_suite.google_rl_certification.config import named_config
from hdfa_rl_suite.google_rl_certification.static_detector_landscape import (
    make_static_landscape,
)
from hdfa_rl_suite.simulator import DriftKind, LatentProcessSpec, ScalableQECDevice, SimulatorConfig
from hdfa_rl_suite.stage0.schema import PolicySnapshot
from hdfa_rl_suite.stage6 import ExplorationBudget, FullControlDetectorRL
from hdfa_rl_suite.stage6.schema import CandidateObservation


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


def _device(seed: int = 91):
    return ScalableQECDevice(SimulatorConfig(
        qubit_count=3,
        seed=seed,
        controller_latency_s=0.0,
        stationary_vectorized_acquisition=True,
        processes=(LatentProcessSpec(
            "stationary", DriftKind.CONSTANT, {}, amplitude=0.0),),
    ))


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


def test_legacy_full_control_is_not_labelled_faithful():
    assert "faithful" not in (FullControlDetectorRL.__doc__ or "").lower()
    assert "legacy" in (FullControlDetectorRL.__doc__ or "").lower()


def test_legacy_candidate_count_is_not_silently_changed():
    device = _device()
    snapshot = PolicySnapshot(
        dict(device.confirmed_policy.controls), device.confirmed_policy.policy_hash,
        device.now_s)
    with pytest.raises(ValueError, match="even and at least four"):
        FullControlDetectorRL(
            device.limits, device.detector_control_graph, snapshot,
            ExplorationBudget(1.0, 100.0), candidate_count=5)


def test_stale_candidate_rewards_do_not_update_legacy_policy():
    device = _device(seed=93)
    snapshot = PolicySnapshot(
        dict(device.confirmed_policy.controls), device.confirmed_policy.policy_hash,
        device.now_s)
    controller = FullControlDetectorRL(
        device.limits, device.detector_control_graph, snapshot,
        ExplorationBudget(10.0, 1000.0), candidate_count=4, seed=3)
    candidates = controller.propose()
    observations = tuple(CandidateObservation(
        candidate.candidate_id,
        {detector: (.01 if candidate.sign > 0 else .20)
         for detector in device.detector_control_graph},
        {detector: 100 for detector in device.detector_control_graph},
        observed_at_s=controller.proposed_package.activation_time_s+10.0,
    ) for candidate in candidates)
    before = dict(controller.current_policy.values)
    result = controller.update(observations)
    assert result.policy_version == 0
    assert result.replay_size == 0
    assert all(value == 0.0 for value in result.gradient.values())
    assert dict(controller.current_policy.values) == before
    assert result.exploration_damage > 0  # stale data are excluded, physical damage is not hidden


def test_reduced_budget_label_requires_track_a():
    result = FullControlRLArm(
        seed=97, candidate_count=4, candidate_cycles=2048,
        damage_budget=100.0, per_candidate_damage_budget=10.0,
    ).run_interval(_device(seed=97), cycles=16, interval=0)
    assert result.candidate_budget_class == "reduced-budget-candidate"
    assert result.candidate_budget_class != "validated-reduced-budget"
