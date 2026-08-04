from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from google_rl_reimplementation.google_pure_v5.accounting import acquisition_accounting
from google_rl_reimplementation.google_pure_v5.baseline import DetectorBaseline
from google_rl_reimplementation.google_pure_v5.config import CERTIFICATION_SEEDS, guard_seed, paper_scale, repository_root, source_choices
from google_rl_reimplementation.google_pure_v5.experiments import validate_policy_schema
from google_rl_reimplementation.google_pure_v5.factor_graph import compose_detector_local_ratios, global_policy_ratio
from google_rl_reimplementation.google_pure_v5.injected_drift_test import generate_injected_tape
from google_rl_reimplementation.google_pure_v5.natural_drift_spectral_test import generate_natural_drift, welch_psd
from google_rl_reimplementation.google_pure_v5.policy import FactorizedGaussianPolicy, component_log_probability
from google_rl_reimplementation.google_pure_v5.protocol import dependency_audit
from google_rl_reimplementation.google_pure_v5.reference_agent import PureGoogleReferenceAgent, evidence_from_counts
from google_rl_reimplementation.google_pure_v5.reward import detector_advantages, detector_rewards
from google_rl_reimplementation.google_pure_v5.studies import run_certification, surface_code_control_count, surface_code_gate_count
from google_rl_reimplementation.google_pure_v5.update import clipped_objective_and_gradient
from google_rl_reimplementation.google_pure_v5.validation import audit_baseline, validate_algorithm


def test_pure_google_runtime_has_no_outside_workflow_dependencies():
    audit = dependency_audit()
    assert audit["status"] == "PASS", audit["violations"]


def test_paper_scale_accounting_exact():
    result = acquisition_accounting(1, paper_scale(), mean_evaluations=1, fixed_evaluations=1, logical_evaluations=1)
    assert result["complete_policy_candidates"] == 40
    assert result["effective_cycles_per_candidate"] == 100_000
    assert result["candidate_acquisition_cycles"] == 4_000_000
    assert result["mean_policy_diagnostic_cycles"] == 100_000
    bad = dict(paper_scale())
    bad["shots_per_candidate"] = 2048
    with pytest.raises(ValueError, match="40 x 4,000 x 25"):
        acquisition_accounting(1, bad)


def test_gaussian_log_probability_matches_scalar_reference():
    actions = np.array([[0.1, -0.2], [0.3, 0.4]])
    mean = np.array([0.02, -0.03])
    log_scale = np.log(np.array([0.3, 0.4]))
    actual = component_log_probability(actions, mean, log_scale)
    expected = np.empty_like(actual)
    for n in range(2):
        for c in range(2):
            sigma = np.exp(log_scale[c])
            expected[n, c] = -np.log(np.sqrt(2 * np.pi) * sigma) - 0.5 * ((actions[n, c] - mean[c]) / sigma) ** 2
    assert np.allclose(actual, expected)


def test_mean_and_log_scale_are_independent_state():
    policy = FactorizedGaussianPolicy(np.array([0.1, -0.1]), initial_scale=0.2, normalized_bounds=(-1, 1), native_sensitivity=np.array([2.0, 3.0]), seed=1)
    old_scale = policy.log_scale.copy()
    policy.mean[0] += 0.1
    assert np.array_equal(policy.log_scale, old_scale)
    policy.log_scale[1] += 0.2
    assert policy.mean[1] == -0.1


def test_candidate_hash_and_native_roundtrip():
    policy = FactorizedGaussianPolicy(np.zeros(3), initial_scale=0.1, normalized_bounds=(-1, 1), native_sensitivity=np.array([1.0, 2.0, 4.0]), seed=2)
    batch = policy.sample(5, policy_version=0, epoch=0)
    assert len(set(batch.action_hashes)) == 5
    assert np.allclose(policy.to_normalized(batch.native_actions), batch.normalized_actions)


def test_detector_reward_is_negative_event_rate():
    counts = np.array([[100, 250], [0, 1000]])
    assert np.array_equal(detector_rewards(counts, 1000), np.array([[-0.1, -0.25], [0.0, -1.0]]))


def test_local_ratio_differs_from_global_ratio():
    ratios = np.array([[1.1, 0.9, 1.05]])
    mask = np.array([[1, 0, 0], [0, 1, 1]], dtype=bool)
    local = compose_detector_local_ratios(ratios, mask)
    assert np.allclose(local, [[1.1, 0.945]])
    assert not np.isclose(local[0, 0], global_policy_ratio(ratios)[0])


def test_v5_componentwise_clip_matches_enumeration():
    actions = np.array([[0.4, -0.3]])
    advantages = np.array([[1.2]])
    mask = np.array([[1, 1]], dtype=bool)
    old_mean = np.zeros(2)
    old_scale = np.log(np.array([0.2, 0.2]))
    mean = np.array([0.04, -0.03])
    objective, _, _, _ = clipped_objective_and_gradient(actions, advantages, mask, mean, old_scale, old_mean, old_scale, clip=0.2, entropy_coefficient=0.0)
    ratios = np.exp(component_log_probability(actions, mean, old_scale) - component_log_probability(actions, old_mean, old_scale))
    expected = 1.2 * np.prod(np.clip(ratios, 0.8, 1.2))
    assert np.isclose(objective, expected)


def test_v5_gradient_matches_finite_difference():
    report = validate_algorithm()
    assert report["status"] == "PASS", report["checks"]
    assert report["maximum_absolute_errors"]["finite_difference_mean"] < 2e-7


def test_sparse_unrelated_gradient_is_exact_zero():
    actions = np.array([[0.1, 0.2, 0.3], [-0.2, 0.1, -0.1]])
    _, gm, gs, _ = clipped_objective_and_gradient(actions, np.array([[1.0], [-0.5]]), np.array([[1, 0, 0]], dtype=bool), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), clip=0.2, entropy_coefficient=0.0)
    assert np.array_equal(gm[1:], np.zeros(2))
    assert np.array_equal(gs[1:], np.zeros(2))


def test_entropy_and_log_scale_derivatives():
    actions = np.array([[0.2, -0.1], [-0.2, 0.1]])
    mask = np.eye(2, dtype=bool)
    _, _, with_entropy, _ = clipped_objective_and_gradient(actions, np.zeros((2, 2)), mask, np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2), clip=0.2, entropy_coefficient=0.007)
    assert np.allclose(with_entropy, 0.007)


def test_baseline_exact_sequences_and_reset():
    report = audit_baseline()
    assert report["status"] == "PASS"
    assert all(report["recurrence_checks"].values())


def test_baseline_subtraction_precedes_update():
    baseline = DetectorBaseline(2, learning_rate=0.1)
    rewards = np.array([[-0.2, -0.4]])
    frozen = baseline.snapshot()
    advantage = baseline.advantages(rewards, frozen=frozen)
    baseline.update(rewards)
    assert np.array_equal(advantage, rewards)
    assert not np.array_equal(advantage, detector_advantages(rewards, baseline.value))


def test_replay_keeps_original_advantage_and_collection_policy():
    agent = PureGoogleReferenceAgent(np.eye(2, dtype=bool), np.zeros(2), np.ones(2), source_choices(), seed=4)
    first = agent.sample(8)
    first_advantage_baseline = agent.baseline.snapshot()
    agent.update(first, evidence_from_counts(first, np.full((8, 2), 6000), 100_000))
    stored = agent.replay.items()[0]
    assert np.array_equal(stored.batch.collection_mean, np.zeros(2))
    assert np.array_equal(stored.advantages, np.full((8, 2), -0.06) - first_advantage_baseline)


def test_candidate_provenance_and_single_use():
    agent = PureGoogleReferenceAgent(np.eye(2, dtype=bool), np.zeros(2), np.ones(2), source_choices(), seed=5)
    batch = agent.sample(4)
    evidence = list(evidence_from_counts(batch, np.full((4, 2), 5000), 100_000))
    with pytest.raises(ValueError, match="provenance"):
        agent.update(batch, tuple([replace(evidence[0], action_hash="wrong"), *evidence[1:]]))
    agent = PureGoogleReferenceAgent(np.eye(2, dtype=bool), np.zeros(2), np.ones(2), source_choices(), seed=5)
    batch = agent.sample(4)
    evidence = evidence_from_counts(batch, np.full((4, 2), 5000), 100_000)
    agent.update(batch, evidence)
    with pytest.raises(ValueError, match="stale|consumed"):
        agent.update(batch, evidence)


def test_four_policy_evaluations_cannot_alias():
    traces = {name: np.zeros(3) for name in ("fixed_policy", "learned_mean", "stochastic_candidates", "oracle_optimum")}
    validate_policy_schema(traces)
    shared = np.zeros(3)
    with pytest.raises(ValueError, match="alias"):
        validate_policy_schema({"fixed_policy": shared, "learned_mean": shared, "stochastic_candidates": np.zeros(3), "oracle_optimum": np.zeros(3)})


def test_injected_and_natural_generators_are_structurally_separate():
    injected, _ = generate_injected_tape({"profile": "step", "category": "XY pulse amplitude", "location": 1, "amplitude": 0.2, "frequency": 0.0, "phase": 0.0}, 100, 20, 24)
    natural = generate_natural_drift({"family": "bounded_multi_sine", "seed": 2, "amplitude": 0.2, "affected_stride": 2}, 100, 24)
    assert np.all(injected[:20] == 0)
    assert not np.any(np.all(np.diff(natural, axis=0) == 0, axis=1))
    assert not np.array_equal(injected, natural)


def test_welch_uses_power_db_convention():
    t = np.arange(512)
    fixed = np.sin(2 * np.pi * t / 128)
    learned = fixed / np.sqrt(10)
    frequency, p_fixed = welch_psd(fixed, segment_length=128, overlap_fraction=0.5, taper="hann", detrend="constant")
    _, p_learned = welch_psd(learned, segment_length=128, overlap_fraction=0.5, taper="hann", detrend="constant")
    selected = (frequency > 0) & (frequency < 0.02)
    gain = 10 * np.log10(p_fixed[selected].sum() / p_learned[selected].sum())
    assert np.isclose(gain, 10.0, atol=1e-10)


def test_surface_code_scaling_includes_exact_distance_15():
    assert surface_code_gate_count(15) == 1289
    assert surface_code_control_count(15) == 38_670
    assert [surface_code_control_count(d) for d in (3, 5, 7, 9, 11, 13, 15)] == sorted(surface_code_control_count(d) for d in (3, 5, 7, 9, 11, 13, 15))


def test_certification_seeds_locked_and_command_fail_closed():
    assert CERTIFICATION_SEEDS == tuple(range(9101, 9113))
    with pytest.raises(ValueError, match="forbidden"):
        guard_seed(9101)
    with pytest.raises(RuntimeError, match="confirm"):
        run_certification(confirm=False, epochs=1000)


def test_required_v5_commands_registered():
    text = (repository_root() / "pyproject.toml").read_text(encoding="utf-8")
    commands = [
        "audit-source-compliance", "audit-baseline", "validate-algorithm", "run-static-tests",
        "run-injected-drift", "run-natural-drift-spectral", "audit-test-separation", "run-step-response",
        "run-steering-phase", "run-randomized-recovery", "run-convergence-scaling",
        "run-development-scorecard", "freeze-certification", "run-certification",
    ]
    assert all(f"google-rl-v5-{command}" in text for command in commands)
