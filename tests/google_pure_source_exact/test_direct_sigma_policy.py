from __future__ import annotations

import json

import numpy as np
import pytest

from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.comparison import (
    compare_positivity_guards,
    run_matched_seed,
)
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.contracts import (
    DIRECT_SIGMA_PARAMETERIZATION,
    NON_PAPER_LOG_SIGMA_ABLATION,
    PositivityGuard,
    require_parameterization,
)
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.gaussian import (
    DirectSigmaGaussianPolicy,
    component_log_probability,
    entropy,
    gaussian_scores,
)
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.losses import total_loss_and_gradients
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.optimizer import (
    DirectSigmaOptimizer,
    OptimizerConfig,
)
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.validation import (
    finite_difference,
    mathematical_audit,
)
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.cli import merge_shards, run_shard


def test_direct_gaussian_log_probability_and_scores_match_finite_difference() -> None:
    actions = np.asarray([[0.2, -0.1], [-0.4, 0.8]])
    mean = np.asarray([0.05, 0.2])
    sigma = np.asarray([0.4, 0.7])
    score_mean, score_sigma = gaussian_scores(actions, mean, sigma)
    for index, action in enumerate(actions):
        numeric_mean = finite_difference(
            lambda value: float(component_log_probability(action[None], value, sigma).sum()), mean)
        numeric_sigma = finite_difference(
            lambda value: float(component_log_probability(action[None], mean, value).sum()), sigma)
        np.testing.assert_allclose(score_mean[index], numeric_mean, atol=2e-7)
        np.testing.assert_allclose(score_sigma[index], numeric_sigma, atol=2e-7)


def test_entropy_gradient_and_negative_entropy_sign() -> None:
    sigma = np.asarray([0.3, 0.8, 1.2])
    numeric = finite_difference(entropy, sigma)
    np.testing.assert_allclose(numeric, 1.0 / sigma, atol=2e-7)
    entropy_gradient_of_loss = -0.4 / sigma
    assert entropy(sigma - 1e-3 * entropy_gradient_of_loss) > entropy(sigma)


def test_total_policy_and_entropy_gradients_match_finite_difference() -> None:
    rng = np.random.default_rng(12)
    mean, sigma = np.asarray([0.1, -0.2]), np.asarray([0.5, 0.7])
    policy = DirectSigmaGaussianPolicy(mean, sigma, seed=12)
    batch = policy.sample(20, standardized_noise=rng.normal(size=(20, 2)))
    rewards = rng.normal(size=(20, 2))
    mask = np.eye(2, dtype=bool)
    baseline = np.asarray([0.02, -0.01])
    common = dict(actions=batch.actions, rewards=rewards, mask=mask, baseline=baseline,
                  behavior=batch.behavior, clip=0.2, baseline_weight=0.0, entropy_weight=0.03)
    result = total_loss_and_gradients(mean=mean, sigma=sigma, **common)
    numeric_mean = finite_difference(
        lambda value: total_loss_and_gradients(mean=value, sigma=sigma, **common).total, mean)
    numeric_sigma = finite_difference(
        lambda value: total_loss_and_gradients(mean=mean, sigma=value, **common).total, sigma)
    np.testing.assert_allclose(result.grad_mean, numeric_mean, atol=2e-6)
    np.testing.assert_allclose(result.grad_sigma, numeric_sigma, atol=2e-6)


def test_baseline_gradient_matches_eq19_and_not_policy_loss() -> None:
    rng = np.random.default_rng(13)
    mean, sigma = np.zeros(2), np.full(2, 0.4)
    batch = DirectSigmaGaussianPolicy(mean, sigma, seed=13).sample(12)
    rewards, baseline = rng.normal(size=(12, 2)), np.asarray([0.1, -0.2])
    kwargs = dict(actions=batch.actions, rewards=rewards, mask=np.eye(2, dtype=bool),
                  mean=mean, sigma=sigma, behavior=batch.behavior, clip=0.2,
                  policy_weight=0.0, baseline_weight=0.7, entropy_weight=0.0)
    result = total_loss_and_gradients(baseline=baseline, **kwargs)
    numeric = finite_difference(lambda value: total_loss_and_gradients(baseline=value, **kwargs).total, baseline)
    np.testing.assert_allclose(result.grad_baseline, numeric, atol=2e-7)


def test_sample_and_log_probability_are_exact_and_behavior_is_immutable() -> None:
    mean, sigma = np.asarray([0.2, -0.3]), np.asarray([0.4, 0.9])
    noise = np.asarray([[1.0, -2.0], [0.0, 0.5]])
    batch = DirectSigmaGaussianPolicy(mean, sigma).sample(2, standardized_noise=noise)
    np.testing.assert_array_equal(batch.actions, mean + sigma * noise)
    manual = -0.5 * noise**2 - np.log(sigma) - 0.5 * np.log(2 * np.pi)
    np.testing.assert_allclose(batch.behavior.component_log_probability, manual)
    assert not batch.behavior.mean.flags.writeable
    with pytest.raises(ValueError):
        batch.behavior.sigma[0] = 9.0


def test_direct_checkpoint_has_sigma_and_no_hidden_log_conversion(tmp_path) -> None:
    policy = DirectSigmaGaussianPolicy(np.asarray([0.1]), np.asarray([0.3]), seed=14)
    optimizer = DirectSigmaOptimizer(1, 1, OptimizerConfig(0.1, 0.1, 0.1))
    state = policy.state_dict(optimizer_state=optimizer.state_dict(), baseline=np.asarray([0.2]))
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["parameterization"] == DIRECT_SIGMA_PARAMETERIZATION
    assert loaded["sigma"] == [0.3]
    assert loaded["optimizer_state"]["optimized_scale_variable"] == "sigma"
    assert loaded["baseline"] == [0.2]
    assert not {"log_sigma", "log_scale", "eta"}.intersection(loaded)
    restored = DirectSigmaGaussianPolicy.from_state_dict(loaded)
    np.testing.assert_array_equal(restored.sigma, policy.sigma)


@pytest.mark.parametrize("guard", list(PositivityGuard))
def test_every_direct_positivity_guard_survives_adversarial_gradient(guard: PositivityGuard) -> None:
    mean, sigma, baseline = np.zeros(2), np.full(2, 0.2), np.zeros(1)
    optimizer = DirectSigmaOptimizer(2, 1, OptimizerConfig(
        0.1, 1.0, 0.1, minimum_sigma=0.01, positivity_guard=guard))
    optimizer.step(mean, sigma, baseline, np.zeros(2), np.full(2, 1e9), np.zeros(1))
    assert np.all(sigma > 0)
    assert optimizer.state_dict()["optimized_scale_variable"] == "sigma"


def test_paper_mode_cannot_select_log_sigma() -> None:
    assert require_parameterization(DIRECT_SIGMA_PARAMETERIZATION, paper_mode=True) == DIRECT_SIGMA_PARAMETERIZATION
    with pytest.raises(ValueError, match="direct sigma"):
        require_parameterization(NON_PAPER_LOG_SIGMA_ABLATION, paper_mode=True)


def test_mathematical_audit_passes_and_guards_are_direct() -> None:
    audit = mathematical_audit()
    assert audit["finite_difference_pass"]
    assert audit["negative_entropy_descent_increases_entropy"]
    assert audit["behavior_snapshot_immutable"]
    assert compare_positivity_guards()["all_direct_and_positive"]


def test_stationary_sigma_shrinks_and_nonstationary_sigma_is_finite() -> None:
    profile = {
        "dimension": 3, "epochs": 60, "candidates_per_epoch": 1024, "curvature": 1.0,
        "initial_mean": 0.6, "initial_sigma": 0.5, "stationary_optimum": 0.0,
        "stationary_entropy_weight": 0.0005, "nonstationary_entropy_weight": 0.01,
        "drift_amplitude": 0.45, "drift_period_epochs": 24,
        "mean_learning_rate": 0.08, "sigma_learning_rate": 0.04,
        "baseline_learning_rate": 0.1, "baseline_weight": 0.2, "ppo_clip": 0.2,
        "minimum_sigma": 0.001, "maximum_sigma": 2.0, "mean_bounds": [-2.0, 2.0],
        "positivity_guard": "projected_gradient",
    }
    result = run_matched_seed(15, profile)
    assert result["gates"]["stationary_sigma_shrank"]
    assert result["gates"]["nonstationary_sigma_finite"]
    assert all(row["parameterization"] != DIRECT_SIGMA_PARAMETERIZATION
               or row["optimized_scale_variable"] == "sigma" for row in result["rows"])


def test_shard_checkpoint_resume_and_duplicate_rejection(tmp_path) -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    source = json.loads((root / "configs/google_pure_source_exact/policy_parameterization.json").read_text())
    source["profiles"]["smoke"].update({"dimension": 2, "epochs": 2,
                                          "candidates_per_epoch": 16, "seeds": [71, 72]})
    config = tmp_path / "config.json"
    config.write_text(json.dumps(source), encoding="utf-8")
    output = tmp_path / "artifacts"
    first = run_shard(config, "smoke", 0, 2, output, resume=False, allow_full=False)
    assert first["complete"] and first["executed_qec_cycles"] == 0
    with pytest.raises(RuntimeError, match="pass --resume"):
        run_shard(config, "smoke", 0, 2, output, resume=False, allow_full=False)
    resumed = run_shard(config, "smoke", 0, 2, output, resume=True, allow_full=False)
    assert resumed["results"] == first["results"]
    run_shard(config, "smoke", 1, 2, output, resume=False, allow_full=False)
    shard0 = output / "shards/smoke/shard-000-of-002.json"
    shard1 = output / "shards/smoke/shard-001-of-002.json"
    left, right = json.loads(shard0.read_text()), json.loads(shard1.read_text())
    right["results"] = left["results"]
    right["seeds"] = left["seeds"]
    shard1.write_text(json.dumps(right), encoding="utf-8")
    with pytest.raises(RuntimeError, match="duplicate"):
        merge_shards(config, "smoke", 2, output, "duplicate-must-fail")


def test_iteration_records_cannot_be_overwritten(tmp_path) -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    source = json.loads((root / "configs/google_pure_source_exact/policy_parameterization.json").read_text())
    source["profiles"]["smoke"].update({"dimension": 2, "epochs": 2,
                                          "candidates_per_epoch": 16, "seeds": [73]})
    config = tmp_path / "config.json"
    config.write_text(json.dumps(source), encoding="utf-8")
    output = tmp_path / "artifacts"
    run_shard(config, "smoke", 0, 1, output, resume=False, allow_full=False)
    merge_shards(config, "smoke", 1, output, "immutable-iteration")
    with pytest.raises(FileExistsError, match="cannot be overwritten"):
        merge_shards(config, "smoke", 1, output, "immutable-iteration")
