from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from google_rl_reimplementation.google_pure_v10.contracts import (
    ExperimentFamily,
    corrected_fault_contract,
    enforce_family_separation,
    validate_evidence_class,
)
from google_rl_reimplementation.google_pure_v10.decoder.interface import CodeConfig, DeterministicParityDecoder
from google_rl_reimplementation.google_pure_v10.decoder.neural_stub import NeuralDecoderStub
from google_rl_reimplementation.google_pure_v10.spectral import (
    InsufficientSpectralDuration,
    SpectralPlan,
    positive_suppression_db,
)
from google_rl_reimplementation.google_pure_v10.preflight import preflight_gate
from google_rl_reimplementation.google_pure_v10.step_response import (
    fit_step_response,
    normalized_response,
    piecewise_constant_optimum,
)


def test_corrected_classifications_do_not_overclaim_causes():
    value = corrected_fault_contract()
    assert "ENTROPY_IMPLEMENTATION_PASS" in value["entropy"]
    assert "MINIMUM_SCALE_FLOOR_EFFECT_NOT_ESTABLISHED" in value["exploration"]
    assert "PPO_CLIPPING_CAUSAL_ROLE_UNESTABLISHED" in value["step_response"]
    assert "TEMPORAL_EVALUATION_PROTOCOL_FAILURE" in value["temporal"]


def test_spectral_plan_fails_before_underresolved_acquisition():
    with pytest.raises(InsufficientSpectralDuration):
        SpectralPlan(128, 128, 0.5, (0.001, 0.012), 4, 2)
    valid = SpectralPlan(512, 256, 0.5, (0.01, 0.08), 4, 3)
    assert valid.diagnostics()["number_of_low_frequency_bins"] >= 4


def test_positive_db_sign_convention_is_fixed():
    assert positive_suppression_db(10.0, 1.0) == pytest.approx(10.0)
    assert positive_suppression_db(1.0, 10.0) == pytest.approx(-10.0)


def test_control_only_and_decoder_values_cannot_be_merged():
    with pytest.raises(ValueError):
        enforce_family_separation(
            [
                {"experiment_family": ExperimentFamily.CONTROL_ONLY.value},
                {"experiment_family": ExperimentFamily.CONTROL_PLUS_FIXED_DECODER.value},
            ]
        )


def test_analytic_values_cannot_masquerade_as_decoder_coupled():
    with pytest.raises(ValueError):
        validate_evidence_class("decoder_coupled_simulation", decoder_executed=False)
    validate_evidence_class("analytic_scaling_model", decoder_executed=False)


def test_deterministic_decoder_is_explicitly_fixture_only():
    decoder = DeterministicParityDecoder()
    code = CodeConfig("surface_code:rotated_memory_x", 3, 3, 0.001)
    decoder.reset(code, 1)
    events = np.asarray([[1, 0, 0], [1, 1, 0]], dtype=np.uint8)
    truth = np.asarray([[1], [0]], dtype=np.uint8)
    result = decoder.decode(events, truth)
    assert result.logical_failures == 0
    assert not decoder.metrics()["reference_backend"]


def test_neural_stub_never_claims_a_trained_decoder():
    decoder = NeuralDecoderStub()
    decoder.reset(CodeConfig("surface_code:rotated_memory_x", 3, 3, 0.001), 1)
    assert not decoder.metrics()["trained"]
    with pytest.raises(RuntimeError):
        decoder.decode(np.zeros((1, 3), dtype=np.uint8))


def test_step_optimum_is_genuinely_piecewise_constant():
    tape = piecewise_constant_optimum(20, 5, np.zeros(2), np.ones(2))
    assert np.allclose(tape[:5], 0) and np.allclose(tape[5:], 1)


def test_step_response_uses_target_relative_weighted_normalization():
    tape = piecewise_constant_optimum(30, 10, np.zeros(2), np.asarray([2.0, 0.0]))
    means = np.zeros_like(tape)
    means[10:] = np.linspace(0, 2, 20)[:, None] * np.asarray([[1.0, 0.0]])
    value = normalized_response(means, tape, onset_epoch=10, weighting_matrix=np.eye(2))
    assert value["response"][9] == pytest.approx(0.0)
    assert value["optimum_response"][10] == pytest.approx(1.0)


def test_step_fit_reports_target_crossings_and_unsettled_horizon():
    response = np.zeros(80)
    response[20:] = 0.8 * (1 - np.exp(-np.arange(60) / 10))
    fit = fit_step_response(response, onset_epoch=20)
    assert fit["response_time_50_epochs"] is not None
    assert fit["response_time_90_epochs"] is None
    assert fit["response_classification"] == "NO_SETTLING_WITHIN_HORIZON"


def test_preflight_rejects_a_deliberately_broken_variant():
    checks = {
        "plant_no_disturbance_sanity": True,
        "fixed_oracle_disturbance_sanity": False,
        "periodic_intermediate_sanity": True,
        "toy_reference_convergence": True,
        "positive_gradient_direction": True,
        "sample_budget_adequacy": True,
        "mean_exploration_separation": True,
        "policy_lifecycle_consistency": True,
        "matched_disturbance_realization": True,
        "complete_hashing": True,
    }
    value = preflight_gate(checks)
    assert not value["pass"] and value["failed_checks"] == ["fixed_oracle_disturbance_sanity"]


def test_v10_cli_entries_are_registered_without_legacy_prefixes():
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    names = (
        "import-audits", "correct-classifications", "preflight", "plan-scale-entropy", "run-scale-entropy",
        "plan-temporal-validation", "run-temporal-validation", "plan-natural-drift", "run-natural-drift",
        "analyse-natural-drift", "validate-decoder", "run-control-only", "run-control-plus-decoder",
        "run-decoder-steering", "plan-step-response", "run-step-response", "run-step-ablation",
        "analyse-step-response", "freeze-held-out", "run-held-out", "report", "status",
    )
    for name in names:
        assert f"google-rl-v10-{name} =" in text
