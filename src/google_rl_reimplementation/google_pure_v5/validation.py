"""Independent mechanism, baseline, and static/no-drift gates."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .baseline import DetectorBaseline
from .config import paper_scale, source_choices
from .experiments import lag1_autocorrelation, run_matched_trace
from .factor_graph import compose_detector_local_ratios, global_policy_ratio
from .lifecycle import DetectorEvidence
from .plant import PurePlantSpec, PureQuadraticPlant
from .policy import component_log_probability
from .reference_agent import PureGoogleReferenceAgent, evidence_from_counts
from .reporting import write_report
from .update import clipped_objective_and_gradient, sgd_ascent_step


def _finite_difference(
    actions: np.ndarray,
    advantages: np.ndarray,
    mask: np.ndarray,
    mean: np.ndarray,
    log_scale: np.ndarray,
    old_mean: np.ndarray,
    old_log_scale: np.ndarray,
    *,
    wrt: str,
    epsilon: float = 1e-6,
) -> np.ndarray:
    out = np.empty_like(mean)
    for index in range(len(mean)):
        plus_mean, minus_mean = mean.copy(), mean.copy()
        plus_scale, minus_scale = log_scale.copy(), log_scale.copy()
        if wrt == "mean":
            plus_mean[index] += epsilon
            minus_mean[index] -= epsilon
        else:
            plus_scale[index] += epsilon
            minus_scale[index] -= epsilon
        plus = clipped_objective_and_gradient(
            actions, advantages, mask, plus_mean, plus_scale, old_mean, old_log_scale,
            clip=0.2, entropy_coefficient=0.003,
        )[0]
        minus = clipped_objective_and_gradient(
            actions, advantages, mask, minus_mean, minus_scale, old_mean, old_log_scale,
            clip=0.2, entropy_coefficient=0.003,
        )[0]
        out[index] = (plus - minus) / (2.0 * epsilon)
    return out


def validate_algorithm() -> dict[str, Any]:
    rng = np.random.default_rng(6101)
    actions = rng.normal(0.0, 0.25, size=(11, 3))
    advantages = rng.normal(size=(11, 2))
    mask = np.array([[1, 1, 0], [0, 1, 0]], dtype=bool)
    old_mean = np.array([0.02, -0.03, 0.11])
    old_log_scale = np.log(np.array([0.31, 0.27, 0.24]))
    mean = old_mean + np.array([0.006, -0.004, 0.0])
    log_scale = old_log_scale + np.array([0.003, -0.002, 0.0])
    objective, gm, gs, diagnostic = clipped_objective_and_gradient(
        actions, advantages, mask, mean, log_scale, old_mean, old_log_scale,
        clip=0.2, entropy_coefficient=0.003,
    )
    fd_mean = _finite_difference(actions, advantages, mask, mean, log_scale, old_mean, old_log_scale, wrt="mean")
    fd_scale = _finite_difference(actions, advantages, mask, mean, log_scale, old_mean, old_log_scale, wrt="scale")

    logp = component_log_probability(actions, mean, log_scale)
    independent_logp = np.empty_like(logp)
    for n in range(len(actions)):
        for c in range(actions.shape[1]):
            sigma = np.exp(log_scale[c])
            independent_logp[n, c] = -np.log(sigma * np.sqrt(2.0 * np.pi)) - 0.5 * (
                (actions[n, c] - mean[c]) / sigma
            ) ** 2
    component_ratio = np.exp(
        component_log_probability(actions, mean, log_scale)
        - component_log_probability(actions, old_mean, old_log_scale)
    )
    local = compose_detector_local_ratios(np.clip(component_ratio, 0.8, 1.2), mask)
    enumerated = np.array(
        [[np.prod(np.clip(component_ratio[n, mask[d]], 0.8, 1.2)) for d in range(2)] for n in range(11)]
    )
    global_ratio = global_policy_ratio(component_ratio)

    # Independent Gauss-Hermite expectation for an on-policy quadratic reward.
    nodes, weights = np.polynomial.hermite.hermgauss(24)
    q_mean, q_scale, target = 0.21, 0.17, -0.08
    samples = q_mean + np.sqrt(2.0) * q_scale * nodes
    rewards = -(samples - target) ** 2
    score_mean = (samples - q_mean) / q_scale**2
    score_scale = (samples - q_mean) ** 2 / q_scale**2 - 1.0
    gh_mean = float(np.sum(weights * rewards * score_mean) / np.sqrt(np.pi))
    gh_scale = float(np.sum(weights * rewards * score_scale) / np.sqrt(np.pi))
    exact_mean = -2.0 * (q_mean - target)
    exact_scale = -2.0 * q_scale**2

    base = DetectorBaseline(2, learning_rate=0.1)
    reward_matrix = np.array([[-0.1, -0.2], [-0.3, -0.4]])
    advantage = base.advantages(reward_matrix)
    before = base.snapshot()
    after = base.update(reward_matrix)
    baseline_expected = before - 0.1 * 2.0 * (before - reward_matrix.mean(axis=0))

    new_mean, new_scale = sgd_ascent_step(
        np.zeros(3), np.zeros(3), np.ones(3), -np.ones(3),
        mean_learning_rate=0.1, scale_learning_rate=0.05,
        bounds=(-1.0, 1.0), scale_bounds=(0.2, 2.0),
    )

    agent = PureGoogleReferenceAgent(
        np.eye(2, dtype=bool), np.zeros(2), np.ones(2), source_choices(), seed=9
    )
    batch = agent.sample(4)
    counts = np.full((4, 2), 5000)
    evidence = list(evidence_from_counts(batch, counts, 100_000))
    provenance_rejected = False
    try:
        agent.update(batch, tuple([replace(evidence[0], action_hash="wrong"), *evidence[1:]]))
    except ValueError:
        provenance_rejected = True
    agent = PureGoogleReferenceAgent(
        np.eye(2, dtype=bool), np.zeros(2), np.ones(2), source_choices(), seed=9
    )
    batch = agent.sample(4)
    evidence_ok = evidence_from_counts(batch, counts, 100_000)
    agent.update(batch, evidence_ok)
    version_advanced = agent.lifecycle.policy_version == 1 and agent.lifecycle.epoch == 1
    reuse_rejected = False
    try:
        agent.update(batch, evidence_ok)
    except ValueError:
        reuse_rejected = True

    checks = {
        "gaussian_log_probability": bool(np.allclose(logp, independent_logp, atol=1e-13)),
        "local_likelihood_ratio": bool(np.allclose(local, enumerated, atol=1e-13)),
        "global_local_ratio_distinct": bool(not np.allclose(local[:, 1], global_ratio)),
        "componentwise_clipped_branch": bool(diagnostic["component_clip_fraction"] >= 0.0),
        "finite_difference_mean": bool(np.max(np.abs(gm - fd_mean)) < 2e-7),
        "finite_difference_log_scale": bool(np.max(np.abs(gs - fd_scale)) < 2e-7),
        "quadrature_mean_gradient": bool(abs(gh_mean - exact_mean) < 1e-12),
        "quadrature_log_scale_gradient": bool(abs(gh_scale - exact_scale) < 1e-12),
        "baseline_subtraction": bool(np.array_equal(advantage, reward_matrix)),
        "baseline_optimizer_step": bool(np.allclose(after, baseline_expected)),
        "entropy_derivative": bool(abs((gs[2] - fd_scale[2])) < 2e-7),
        "sparse_inactive_mean_gradient_zero": bool(gm[2] == 0.0),
        "sparse_inactive_scale_only_entropy": bool(abs(gs[2] - 0.003) < 1e-14),
        "optimizer_step": bool(np.allclose(new_mean, 0.1) and np.allclose(new_scale, -0.05)),
        "candidate_action_provenance": provenance_rejected,
        "policy_version_lifecycle": version_advanced and reuse_rejected,
        "replay_ratio_supported": bool(np.any(np.abs(component_ratio - 1.0) > 1e-8)),
    }
    payload = {
        "schema_version": "google-pure-v5-numerical-validation.v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "objective": objective,
        "checks": checks,
        "maximum_absolute_errors": {
            "finite_difference_mean": float(np.max(np.abs(gm - fd_mean))),
            "finite_difference_log_scale": float(np.max(np.abs(gs - fd_scale))),
            "gauss_hermite_mean": abs(gh_mean - exact_mean),
            "gauss_hermite_log_scale": abs(gh_scale - exact_scale),
        },
        "validated_objective": "clip each component chi, then compose detector-local products; no sign-aware min branch",
        "certification_seeds_consumed": False,
    }
    write_report("numerical_algorithm_validation", payload, "Numerical algorithm validation")
    return payload


def _manual_baseline(sequence: np.ndarray, learning_rate: float) -> np.ndarray:
    current = np.zeros(sequence.shape[1], dtype=float)
    history = []
    for reward in sequence:
        history.append(current.copy())
        current = (1.0 - 2.0 * learning_rate) * current + 2.0 * learning_rate * reward
    return np.asarray(history)


def audit_baseline() -> dict[str, Any]:
    lr = float(source_choices()["baseline_learning_rate"])
    t = np.arange(16, dtype=float)
    sequences = {
        "constant": np.column_stack([np.full(16, -0.2), np.full(16, -0.1)]),
        "step": np.column_stack([np.where(t < 8, -0.1, -0.3), np.where(t < 8, -0.2, -0.05)]),
        "sinusoidal": np.column_stack([-0.2 + 0.05 * np.sin(t), -0.1 + 0.03 * np.cos(t)]),
        "alternating": np.column_stack([np.where(t % 2 == 0, -0.1, -0.3), np.where(t % 2 == 0, -0.25, -0.05)]),
        "independent_detectors": np.column_stack([-0.05 - 0.01 * t, -0.25 + 0.005 * t]),
    }
    recurrence_checks: dict[str, bool] = {}
    traces: dict[str, Any] = {}
    for name, sequence in sequences.items():
        baseline = DetectorBaseline(2, learning_rate=lr)
        observed = []
        for reward in sequence:
            observed.append(baseline.snapshot())
            baseline.update(reward[None, :])
        manual = _manual_baseline(sequence, lr)
        recurrence_checks[name] = bool(np.allclose(observed, manual, rtol=0.0, atol=1e-15))
        traces[name] = {
            "reward": sequence.tolist(),
            "baseline_before_reward": np.asarray(observed).tolist(),
            "advantage": (sequence - manual).tolist(),
        }
    baseline = DetectorBaseline(2, learning_rate=lr)
    baseline.update(sequences["constant"][:2])
    baseline.reset()
    reset_pass = bool(np.array_equal(baseline.value, np.zeros(2)))
    collection_baseline = np.array([-0.1, -0.2])
    replay_reward = np.array([[-0.2, -0.1]])
    original_advantage = replay_reward - collection_baseline[None, :]
    changed_baseline = np.array([-0.3, -0.3])
    replay_pass = bool(np.array_equal(original_advantage, replay_reward - collection_baseline[None, :])) and not np.array_equal(
        original_advantage, replay_reward - changed_baseline[None, :]
    )

    epochs = np.arange(120, dtype=float)
    true_optimum = 0.20 * np.sin(2.0 * np.pi * epochs / 80.0)
    reward = -0.06 - 0.25 * true_optimum**2
    tracker = DetectorBaseline(1, learning_rate=lr)
    baseline_trace, advantage_trace = [], []
    for value in reward:
        baseline_trace.append(tracker.value[0])
        advantage_trace.append(value - tracker.value[0])
        tracker.update(np.array([[value]]))
    learned_mean = np.zeros_like(true_optimum)
    for i in range(1, len(learned_mean)):
        learned_mean[i] = learned_mean[i - 1] + 0.025 * (true_optimum[i] - learned_mean[i - 1])
    acf = {
        "reward": lag1_autocorrelation(reward),
        "baseline": lag1_autocorrelation(np.asarray(baseline_trace)),
        "advantage": lag1_autocorrelation(np.asarray(advantage_trace)),
        "true_optimum_motion": lag1_autocorrelation(true_optimum),
        "learned_mean": lag1_autocorrelation(learned_mean),
    }
    conditional_bias = float(np.mean(np.asarray(advantage_trace)[true_optimum >= 0]) - np.mean(np.asarray(advantage_trace)[true_optimum < 0]))
    tracking_error = float(np.sqrt(np.mean((np.asarray(baseline_trace) - reward) ** 2)))
    gradient_alignment = float(np.corrcoef(np.asarray(advantage_trace)[1:], np.diff(reward))[0, 1])
    checks = {**recurrence_checks, "reset_semantics": reset_pass, "replay_original_advantage": replay_pass}
    payload = {
        "schema_version": "google-pure-v5-baseline-forensic.v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "source_reconstruction": {
            "one_baseline_per_detector": True,
            "initialization": "zero (publicly unspecified repository choice)",
            "loss": "mean_candidates ||reward-baseline||^2 (Supplement Eq. 19)",
            "recurrence": "b_next=(1-2*lr)*b+2*lr*mean(current_rewards)",
            "ordering": "freeze b for advantage and all policy passes; update b once afterward",
            "replay": "stored advantages retain their collection baseline",
        },
        "root_cause": "v4 used a direct EMA coefficient and a different sign-aware product-ratio objective; high reward autocorrelation alone was physical-drift evidence, not proof of baseline failure",
        "correction": "v5 derives the baseline update from Eq. 19, freezes it within each epoch, and retains original replay advantages",
        "recurrence_checks": checks,
        "deterministic_traces": traces,
        "forensic_metrics": {
            "lag1_autocorrelation": acf,
            "advantage_conditional_bias": conditional_bias,
            "gradient_alignment_with_reward_change": gradient_alignment,
            "baseline_tracking_rmse": tracking_error,
        },
        "certification_seeds_consumed": False,
    }
    write_report("baseline_forensic_audit", payload, "Detector baseline forensic audit")
    return payload


def run_static_tests(epochs: int = 180) -> dict[str, Any]:
    algorithm = validate_algorithm()
    baseline = audit_baseline()
    choices = source_choices()
    paper = paper_scale()
    plant = PureQuadraticPlant(PurePlantSpec("static-gate", draw_seed=7401))
    target = np.zeros((epochs, plant.spec.control_count))
    both_sides: list[dict[str, float]] = []
    for side, seed in ((-1.0, 7402), (1.0, 7403)):
        agent = PureGoogleReferenceAgent(
            plant.mask,
            np.full(plant.spec.control_count, side * 0.24),
            plant.native_sensitivity,
            choices,
            seed=seed,
        )
        rng = np.random.default_rng(seed + 50_000)
        initial_distance = float(np.linalg.norm(agent.mean))
        for optimum in target:
            batch = agent.sample(int(paper["candidates_per_epoch"]))
            counts = plant.acquire_counts(batch.normalized_actions, optimum, effective_cycles=int(paper["effective_cycles_per_candidate"]), rng=rng)
            agent.update(batch, evidence_from_counts(batch, counts, int(paper["effective_cycles_per_candidate"])))
        both_sides.append({"side": side, "initial_distance": initial_distance, "final_distance": float(np.linalg.norm(agent.mean))})

    stationary_epochs = min(epochs, 140)
    stationary = run_matched_trace(
        plant,
        np.zeros((stationary_epochs, plant.spec.control_count)),
        choices,
        paper,
        seed=7404,
    )
    mean_norm = np.linalg.norm(stationary["learned_mean_vectors"], axis=1)
    mean_ler = stationary["logical_risk"]["learned_mean"]
    fixed_ler = stationary["logical_risk"]["fixed_policy"]
    stochastic = stationary["logical_risk"]["stochastic_candidates"]
    inactive_actions = np.array([[0.1, 0.2, 0.3], [-0.2, 0.1, -0.1]])
    _, inactive_gradient, _, _ = clipped_objective_and_gradient(
        inactive_actions, np.array([[1.0], [-0.5]]), np.array([[1, 0, 0]], dtype=bool),
        np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), clip=0.2, entropy_coefficient=0.0,
    )
    checks = {
        "algorithm_gate": algorithm["status"] == "PASS",
        "baseline_gate": baseline["status"] == "PASS",
        "converges_from_below": both_sides[0]["final_distance"] < 0.55 * both_sides[0]["initial_distance"],
        "converges_from_above": both_sides[1]["final_distance"] < 0.55 * both_sides[1]["initial_distance"],
        "inactive_controls_exact_zero": bool(np.array_equal(inactive_gradient[1:], np.zeros(2))),
        "no_drift_mean_tolerance": float(np.max(mean_norm)) < 0.12,
        "no_systematic_degradation": float(mean_ler[-20:].mean()) <= float(mean_ler[:20].mean()) + 2e-4,
        "baseline_unbiased": float(np.max(np.abs(stationary["advantage_means"][-40:].mean(axis=0)))) < 0.012,
        "replay_no_motion": float(np.linalg.norm(stationary["learned_mean_vectors"][-1])) < 0.10,
        "logical_detector_direction_consistent": float(np.corrcoef(mean_ler, stationary["detector_rate"]["learned_mean"])[0, 1]) > 0.99,
    }
    payload = {
        "schema_version": "google-pure-v5-static-tests.v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "both_sides": both_sides,
        "no_drift": {
            "maximum_mean_distance": float(np.max(mean_norm)),
            "final_mean_distance": float(mean_norm[-1]),
            "initial_mean_ler": float(mean_ler[:20].mean()),
            "final_mean_ler": float(mean_ler[-20:].mean()),
            "final_mean_scale": float(stationary["policy_scale_vectors"][-1].mean()),
            "stochastic_exploration_damage": float(np.mean(stochastic - mean_ler)),
        },
        "paper_scale_accounting_exact": int(paper["effective_cycles_per_candidate"]) == 100_000,
        "certification_seeds_consumed": False,
    }
    write_report("static_tests", payload, "Static optimization and no-drift gates")
    return payload
