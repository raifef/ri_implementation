from dataclasses import replace
import inspect
import json

import numpy as np
import pytest

from hdfa_rl_suite.google_reproduction.audit import forensic_audit
from hdfa_rl_suite.google_reproduction.config import (
    load_reference_config,
    repository_root,
)
from hdfa_rl_suite.google_reproduction.experiments import _make_agent, run_scaling
from hdfa_rl_suite.google_reproduction.reference_agent import (
    DetectorEvidence,
    ReferenceAgent,
    local_policy_ratios,
)
from hdfa_rl_suite.google_reproduction.reporting import public_anchor_registry
from hdfa_rl_suite.google_reproduction.surrogate import (
    PaperAnchoredSurrogate,
    surface_code_parameter_count,
)
from hdfa_rl_suite.google_reproduction.validation import validate_surrogate


def _agent(seed=7901):
    return _make_agent(seed)


def _evidence(plant, batch, *, regime=None, action_hashes=None):
    rates = plant.evaluate_native(batch.actions_native).detector_rates
    counts = np.rint(rates * 100_000).astype(int)
    hashes = batch.action_hashes if action_hashes is None else action_hashes
    return tuple(
        DetectorEvidence(
            batch.candidate_ids[i], hashes[i], counts[i], 100_000,
            batch.regime_id if regime is None else regime,
        )
        for i in range(40)
    )


def test_v2_reference_is_separate_from_legacy():
    import hdfa_rl_suite.google_reproduction.reference_agent as reference
    import hdfa_rl_suite.google_rl_certification.agent as legacy

    assert reference.__file__ != legacy.__file__
    assert "google_rl_certification" not in inspect.getsource(reference)


def test_v2_public_sampling_budget_is_exact():
    config = load_reference_config()
    sampling = config.sampling
    assert sampling.candidates_per_epoch == 40
    assert sampling.shots_per_candidate == 4000
    assert sampling.qec_cycles_per_shot == 25
    assert sampling.effective_cycles_per_candidate == 100_000
    assert config.cost(1)["candidate_native_qec_cycles"] == 4_000_000
    assert config.cost(1)["ideal_candidate_acquisition_seconds"] == 4.0


def test_v2_reward_sign_moves_mean_toward_optimum_from_both_sides():
    config = load_reference_config()
    for start in (-0.25, 0.25):
        agent = ReferenceAgent(["c"], ["d"], np.ones((1, 1)), np.ones(1), np.array([start]), config, seed=7901)
        initial = abs(agent.mean[0])
        for _ in range(90):
            batch = agent.sample_candidates(regime_id="static")
            probabilities = np.clip(0.01 + 0.08 * batch.actions_normalized[:, 0] ** 2, 0, 1)
            counts = np.rint(probabilities * 100_000).astype(int)
            evidence = tuple(
                DetectorEvidence(batch.candidate_ids[i], batch.action_hashes[i], np.array([counts[i]]), 100_000, "static")
                for i in range(40)
            )
            agent.update(batch, evidence)
        assert abs(agent.mean[0]) < initial


def test_v2_stale_rewards_fail_closed():
    _, plant, agent = _agent()
    batch = agent.sample_candidates(regime_id="static")
    evidence = _evidence(plant, batch)
    agent.update(batch, evidence)
    with pytest.raises(ValueError, match="stale|consumed"):
        agent.update(batch, evidence)


def test_v2_shuffled_order_is_safe_but_shuffled_candidate_action_labels_fail_closed():
    _, plant_a, agent_a = _agent(7901)
    _, plant_b, agent_b = _agent(7901)
    batch_a = agent_a.sample_candidates(regime_id="static")
    batch_b = agent_b.sample_candidates(regime_id="static")
    evidence = _evidence(plant_a, batch_a)
    agent_a.update(batch_a, evidence)
    agent_b.update(batch_b, tuple(reversed(evidence)))
    np.testing.assert_array_equal(agent_a.mean, agent_b.mean)
    _, plant_c, agent_c = _agent(7901)
    batch_c = agent_c.sample_candidates(regime_id="static")
    bad_hashes = batch_c.action_hashes[1:] + batch_c.action_hashes[:1]
    with pytest.raises(ValueError, match="provenance"):
        agent_c.update(batch_c, _evidence(plant_c, batch_c, action_hashes=bad_hashes))


def test_v2_mask_transpose_and_wrong_sensitivity_fail_closed():
    config, plant, _ = _agent()
    non_square_mask = np.array([[1, 0, 1], [0, 1, 0]], dtype=bool)
    with pytest.raises(ValueError, match="mask has shape"):
        ReferenceAgent(
            ["c0", "c1", "c2"], ["d0", "d1"], non_square_mask.T,
            np.ones(3), np.zeros(3), config, seed=7901,
        )
    with pytest.raises(ValueError, match="sensitivity scale"):
        plant.validate_sensitivity_calibration(plant.sensitivity * 10)


def test_v2_local_policy_ratio_matches_manual_factor_product():
    actions = np.array([[0.3, -0.1]])
    mean = np.array([0.1, 0.2])
    old_mean = np.array([0.0, 0.0])
    log_std = np.log(np.array([0.4, 0.3]))
    old_log_std = np.log(np.array([0.5, 0.6]))
    mask = np.array([[1, 0], [1, 1]], dtype=bool)
    ratio = local_policy_ratios(actions, mean, log_std, old_mean, old_log_std, mask)
    def density(x, mu, sigma):
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
    per_control = density(actions[0], mean, np.exp(log_std)) / density(actions[0], old_mean, np.exp(old_log_std))
    np.testing.assert_allclose(ratio[0], [per_control[0], per_control.prod()])


def test_v2_replay_of_incompatible_drift_regime_fails_closed():
    _, plant, agent = _agent()
    batch = agent.sample_candidates(regime_id="slow-drift")
    with pytest.raises(ValueError, match="incompatible"):
        agent.update(batch, _evidence(plant, batch, regime="step-drift"))


def test_v2_multiple_ppo_passes_exercise_ratio_and_compatible_replay():
    config, plant, agent = _agent()
    first = agent.sample_candidates(regime_id="static")
    first_result = agent.update(first, _evidence(plant, first))
    ratios_after = local_policy_ratios(
        first.actions_normalized,
        agent.mean,
        agent.log_stddev,
        first.collection_mean,
        first.collection_log_stddev,
        plant.dense_mask(),
    )
    assert config.agent.optimizer_steps > 1
    assert np.any(np.abs(ratios_after - 1.0) > 1e-8)
    second = agent.sample_candidates(regime_id="static")
    second_result = agent.update(second, _evidence(plant, second))
    assert first_result["compatible_replay_epochs_used"] == 0
    assert second_result["compatible_replay_epochs_used"] >= 1


def test_v2_covariance_collapse_and_explosion_are_clamped():
    config, plant, agent = _agent()
    for injected in (1e-30, 1e30):
        _, plant, agent = _agent()
        agent.log_stddev[:] = np.log(injected)
        batch = agent.sample_candidates(regime_id="scale-fault")
        agent.update(batch, _evidence(plant, batch))
        assert np.all(agent.stddev >= config.agent.minimum_stddev_normalized - 1e-12)
        assert np.all(agent.stddev <= config.agent.maximum_stddev_normalized + 1e-12)


def test_v2_candidates_are_centred_not_cumulative():
    config, _, agent = _agent()
    first = agent.sample_candidates(regime_id="static")
    np.testing.assert_allclose(
        first.actions_normalized,
        first.collection_mean + np.exp(first.collection_log_stddev) * first.standardized_perturbations,
    )
    assert not np.allclose(first.standardized_perturbations[0::2] + first.standardized_perturbations[1::2], 0)


def test_v2_agent_has_no_truth_api_and_mean_is_not_candidate_average():
    assert "optimum" not in inspect.signature(ReferenceAgent.update).parameters
    _, _, agent = _agent()
    batch = agent.sample_candidates(regime_id="static")
    assert not np.array_equal(agent.mean_native, batch.actions_native.mean(axis=0))


def test_v2_inactive_control_gradient_is_zero():
    result = validate_surrogate()
    assert result["checks"]["inactive_control_unchanged"]


def test_v2_distance_15_parameter_count():
    assert surface_code_parameter_count(15, 30) == 38_670
    scaling = run_scaling(7902)
    assert scaling["status"] == "PASS"


def test_v2_surrogate_sanity_ordering():
    result = validate_surrogate()
    assert all(result["checks"].values())


def test_v2_public_registry_keeps_control_and_decoder_results_separate():
    anchors = {item["id"]: item for item in public_anchor_registry()["anchors"]}
    assert anchors["control_only_stability"]["published_value"].startswith("2.4")
    assert anchors["control_plus_decoder_stability"]["published_value"].startswith("3.5")
    assert anchors["control_plus_decoder_stability"]["comparability"] == "not commensurable"


def test_v2_source_unspecified_choices_are_complete_and_explicit():
    path = repository_root() / "configs/google_rl/source_unspecified_choices.yaml"
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"candidate_values", "rationale", "selection_method", "frozen_final_value", "sensitivity_result"}
    assert payload["choices"]
    assert all(required <= set(choice) for choice in payload["choices"])
    assert all("pending" not in choice["sensitivity_result"] for choice in payload["choices"])
    assert forensic_audit()["certification_implication"].startswith("not evaluable")


def test_v2_certification_seeds_rejected_during_development():
    with pytest.raises(ValueError, match="certification seed"):
        _make_agent(8101)
