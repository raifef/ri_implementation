from __future__ import annotations

import math

import numpy as np

from hdfa_rl_suite.google_pure_v20.core import (
    batch_snr,
    candidate_snr,
    cosine_alignment,
    decompose_mean_trace,
    fixed_budget_equal,
    fundamental_improvement,
    perturbation_rank,
    project_shared_subspace,
    quadratic_component_accounting,
    update_efficiency,
    wrong_sign_probability,
)
from hdfa_rl_suite.google_pure_v20.data import (
    FORBIDDEN_CAMPAIGNS,
    verify_import_manifest,
)
from hdfa_rl_suite.google_pure_v20.io import ARTIFACT_ROOT, NONFINAL, ROOT, read_json
from hdfa_rl_suite.google_pure_v20.repair import REPAIR_NAME


def test_synthetic_dc_fundamental_harmonic_orthogonal_decomposition_conserves_trace():
    epochs = np.arange(300)
    phase = 2*np.pi*epochs/150
    trace = (.2 + .4*np.sin(phase) + .1*np.cos(2*phase))[:, None] * np.ones((1, 3))
    trace[:, 0] += .07*np.sin(3*phase)
    trace[:, 1] -= .07*np.sin(3*phase)
    parts = decompose_mean_trace(trace, epochs, 1/150, harmonics=4)
    reconstructed = sum(parts[name] for name in (
        "dc", "fundamental", "harmonic", "transient", "orthogonal"))
    np.testing.assert_allclose(reconstructed, trace, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.mean(parts["orthogonal"], axis=1), 0, atol=1e-12)


def test_quadratic_cost_decomposition_including_cross_terms_conserves():
    components = {
        "target": np.asarray([[1.0, -2.0], [.5, .25]]),
        "dc": np.asarray([[.3, .4], [-.2, .1]]),
        "orthogonal": np.asarray([[-.1, .1], [.2, -.2]]),
    }
    result = quadratic_component_accounting(components, np.asarray([2.0, 5.0]))
    assert abs(result["conservation_error"]) < 1e-12


def test_fundamental_quadratic_improvement_identity():
    gain, phase = .12990240183443405, 1.2972248010666532
    assert math.isclose(fundamental_improvement(gain, phase),
                        2*gain*math.cos(phase)-gain**2, abs_tol=1e-15)
    assert fundamental_improvement(gain, phase) > 0


def test_candidate_and_batch_snr_use_candidate_standard_deviation():
    values = np.asarray([1.0, 2.0, 3.0, 4.0])
    expected = abs(float(np.mean(values))) / float(np.std(values, ddof=1))
    assert math.isclose(candidate_snr(values), expected)
    assert math.isclose(batch_snr(values), math.sqrt(4)*expected)


def test_wrong_sign_probability_and_update_efficiency():
    assert math.isclose(wrong_sign_probability(0), .5)
    beneficial = np.asarray([1.0, 0.0])
    supplied = np.asarray([2.0, 3.0])
    delta = .32 * supplied
    assert math.isclose(update_efficiency(delta, supplied, beneficial, .32), 1.0)


def test_reference_alignment_and_perturbation_rank_reporting():
    assert math.isclose(cosine_alignment(np.asarray([1, 0]), np.asarray([2, 0])), 1.0)
    noise = np.eye(4, 7)
    rank = perturbation_rank(noise)
    assert rank["raw_rank"] == 4
    assert rank["centered_rank"] == 3
    assert rank["candidate_count"] == 4 and rank["parameter_count"] == 7


def test_projection_is_an_orthogonal_shared_subspace_projection():
    gradient = np.asarray([1.0, 2.0, 6.0])
    projected = project_shared_subspace(gradient)
    np.testing.assert_allclose(projected, np.mean(gradient))
    np.testing.assert_allclose(np.sum(gradient-projected), 0, atol=1e-15)


def test_fixed_budget_accounting_requires_equal_K_times_M():
    matched = [
        {"candidates": 32, "cycles_per_candidate": 12000},
        {"candidates": 16, "cycles_per_candidate": 24000},
        {"candidates": 8, "cycles_per_candidate": 48000},
    ]
    assert fixed_budget_equal(matched)
    assert not fixed_budget_equal(matched + [
        {"candidates": 8, "cycles_per_candidate": 12000}])


def test_v20_artifacts_answer_required_causal_questions_and_are_nonfinal():
    required = (
        "import_manifest", "fast_mean_cost_decomposition",
        "transfer_evaluation_geometry_audit", "fast_gradient_statistics",
        "fast_update_efficiency", "fast_reference_gradients",
        "candidate_vs_shots_factorial", "fixed_budget_information_comparison",
        "frozen_scale_information_damage_frontier", "dynamic_sigma_signed_gradients",
        "acquisition_bias_audit", "population_gradient_fast_rollout",
        "root_cause_classification", "minimal_repair", "postrepair_fast_validation", "status",
    )
    assert all((ARTIFACT_ROOT / f"{name}.json").is_file() for name in required)
    assert (ARTIFACT_ROOT / "FINAL_REPORT.md").is_file()
    for name in required:
        value = read_json(ARTIFACT_ROOT / f"{name}.json")
        for key, expected in NONFINAL.items():
            assert value[key] == expected
        assert value.get("forbidden_auto_runs_launched", []) == []


def test_frozen_scale_frontier_does_not_update_policy_state():
    frontier = read_json(ARTIFACT_ROOT / "frozen_scale_information_damage_frontier.json")
    assert frontier["policy_state_unchanged"] is True
    assert frontier["stored_policy_state_hash_before"] == frontier[
        "stored_policy_state_hash_after"]


def test_population_rollout_and_acquisition_order_have_required_isolation():
    population = read_json(ARTIFACT_ROOT / "population_gradient_fast_rollout.json")
    acquisition = read_json(ARTIFACT_ROOT / "acquisition_bias_audit.json")
    assert population["population_gradient_removes_finite_candidate_mean_noise_only"] is True
    assert population["same_mean_learning_rate"] is True
    assert population["same_normalization"] is True
    assert acquisition["all_variants_use_matched_K8_information_budget"] is True


def test_root_cause_exists_before_exactly_one_repair_and_lineage_is_unchanged():
    root = read_json(ARTIFACT_ROOT / "root_cause_classification.json")
    repair = read_json(ARTIFACT_ROOT / "minimal_repair.json")
    validation = read_json(ARTIFACT_ROOT / "postrepair_fast_validation.json")
    assert root["root_cause_emitted_before_repair"] is True
    assert root["repair_permitted"] is True
    assert repair["repair"] == REPAIR_NAME
    assert repair["exactly_one_causal_repair"] is True
    assert validation["exactly_one_causal_repair"] is True
    assert validation["gates"]["source_style_branch_unchanged"] is True
    assert validation["gates"]["slow_intermediate_not_rerun"] is True
    assert validation["gates"]["paper_equivalence_claim_permitted"] is False
    assert verify_import_manifest()["pass"] is True


def test_v20_protocol_forbids_expensive_or_nonfast_automatic_campaigns():
    protocol = read_json(ROOT / "configs/google_pure_v20/protocol.json")
    assert protocol["automatic_acquisition_frequencies"] == [1/150]
    assert protocol["automatic_campaigns_permitted"] == []
    assert set(protocol["forbidden_auto_runs"]) == set(FORBIDDEN_CAMPAIGNS)
    assert protocol["maximum_causal_repairs"] == 1


def test_required_v20_cli_commands_are_registered_once():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    commands = (
        "hdfa-google-v20-decompose-fast-mean-cost",
        "hdfa-google-v20-audit-transfer-geometry",
        "hdfa-google-v20-audit-fast-gradient-statistics",
        "hdfa-google-v20-compute-reference-gradients",
        "hdfa-google-v20-run-candidate-shot-factorial",
        "hdfa-google-v20-run-fixed-budget-comparison",
        "hdfa-google-v20-run-scale-information-frontier",
        "hdfa-google-v20-audit-dynamic-sigma",
        "hdfa-google-v20-audit-acquisition-bias",
        "hdfa-google-v20-run-population-gradient-fast",
        "hdfa-google-v20-classify-root-cause",
        "hdfa-google-v20-run-minimal-repair-validation",
        "hdfa-google-v20-status",
        "hdfa-google-v20-report",
    )
    assert all(pyproject.count(command) == 1 for command in commands)
