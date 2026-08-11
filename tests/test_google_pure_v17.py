from __future__ import annotations

import inspect
import math

import numpy as np

from hdfa_rl_suite.google_pure_source_exact.figure5a.contracts import ratio_from_raw_counts
from hdfa_rl_suite.google_pure_source_exact.figure5a.plant import Figure5aStimPlant
from hdfa_rl_suite.google_pure_v17 import experiments
from hdfa_rl_suite.google_pure_v17.estimators import (
    complete_period_window, estimate_pure_delay, estimate_sinusoidal_transfer,
    measured_period_from_zero_crossings, paired_acceptance_v2, sinusoid,
)
from hdfa_rl_suite.google_pure_v17.imports import build_import_manifest, verify_import_manifest
from hdfa_rl_suite.google_pure_v17.io import ROOT, config, file_hash, read_json


def test_hessian_and_unit_variance_damage_are_distinct() -> None:
    coefficient = .01
    epsilon = 1e-4
    hessian = (coefficient * epsilon**2 + coefficient * (-epsilon)**2) / epsilon**2
    expected_damage = coefficient * np.mean(np.square(np.asarray([-1.0, 1.0])))
    assert hessian == .02
    assert expected_damage == .01


def test_exact_quadratic_half_identity() -> None:
    x = np.linspace(-2, 2, 101)
    coefficient = .01
    numerical_hessian = (coefficient * 1e-8 + coefficient * 1e-8) / 1e-8
    assert np.isclose(np.mean(coefficient * x**2) / np.mean(x**2), .5 * numerical_hessian)


def test_production_target_period_matches_one_over_f() -> None:
    frequency = 1 / 150
    epochs = np.arange(0, 500)
    values = np.asarray([Figure5aStimPlant.optimum(int(t), frequency)[0] for t in epochs])
    measured = measured_period_from_zero_crossings(values, epochs)
    assert abs(measured["measured_period_epochs"] - 150) < .05


def test_cycles_per_epoch_and_radians_per_epoch_are_not_interchangeable() -> None:
    frequency = 1 / 150
    epochs = np.arange(0, 1200)
    wrong = np.sin(frequency * epochs)
    measured = measured_period_from_zero_crossings(wrong, epochs)
    assert measured["measured_period_epochs"] is None or abs(measured["measured_period_epochs"] - 150) > 100


def test_deterministic_fixture_slow_outperforms_fast() -> None:
    slow = experiments._deterministic_trace(.001, gain=.85, delay=1, tau=133, phase=0)
    fast = experiments._deterministic_trace(1 / 150, gain=.85, delay=1, tau=133, phase=0)
    assert slow["normalized_performance"] > fast["normalized_performance"]


def test_metric_endpoints_are_exact() -> None:
    assert ratio_from_raw_counts(100, 100, 20)["source_ratio"] == 0
    assert ratio_from_raw_counts(20, 100, 20)["source_ratio"] == 1


def test_complete_period_window_is_exact() -> None:
    window = complete_period_window(1 / 150, burn_in_periods=1, evaluation_periods=5)
    assert window["burn_in_epochs"] == 150
    assert window["evaluation_epochs"] == 750
    assert window["fractional_periods"] == 0


def test_transfer_estimator_recovers_known_gain_and_phase() -> None:
    frequency = 1 / 100
    epochs = np.arange(0, 1000)
    gain, phase = .63, -.47
    output = gain * np.sin(2 * np.pi * frequency * epochs + phase) + .08
    fitted = estimate_sinusoidal_transfer(epochs, output, frequency)
    assert fitted["identifiable"]
    assert abs(fitted["gain"] - gain) < 1e-10
    assert abs(fitted["phase_radians"] - phase) < 1e-10


def test_pure_delay_estimator_recovers_known_delay() -> None:
    rng = np.random.default_rng(17)
    source = rng.normal(size=200)
    delayed = np.concatenate([np.zeros(7), source[:-7]])
    fitted = estimate_pure_delay(source, delayed, maximum_delay=15)
    assert fitted["estimated_delay_epochs"] == 7


def test_mean_stochastic_decomposition_algebra() -> None:
    metric = experiments._metric_from_totals(
        {"fixed": 1000, "optimal": 200, "learned_mean": 500, "stochastic": 580})
    assert math.isclose(metric["I_mean"] - metric["I_stochastic"],
                        metric["exploration_damage"] / metric["denominator"])


def test_paired_acceptance_v2_uses_complete_matched_units() -> None:
    units = []
    for seed, slow, fast in ((1, .7, .4), (2, .8, .45), (3, .75, .42)):
        common = {"seed": seed, "phase_radians": 0.0, "budget_hash": "b",
                  "crn_hash": f"c{seed}", "cycles_per_candidate": 300,
                  "burn_in_epochs": 1000, "evaluation_periods": 3, "complete_periods": 3}
        units.extend([{**common, "condition": "slow", "I_stochastic": slow},
                      {**common, "condition": "fast", "I_stochastic": fast}])
    result = paired_acceptance_v2(units, delta_min=.05, bootstrap_draws=1000)
    assert result["valid"] and result["pass"]
    assert result["lower_confidence_bound"] > .05


def test_paired_acceptance_rejects_incomplete_windows() -> None:
    unit = {"seed": 1, "phase_radians": 0.0, "budget_hash": "b", "crn_hash": "c",
            "cycles_per_candidate": 300, "burn_in_epochs": 0,
            "evaluation_periods": 3, "complete_periods": 0, "I_stochastic": .5}
    result = paired_acceptance_v2([{**unit, "condition": "slow"},
                                   {**unit, "condition": "fast"}], delta_min=.05)
    assert not result["valid"] and not result["pass"]


def test_scale_logging_is_keyed_by_epoch_frequency_and_phase() -> None:
    source = inspect.getsource(experiments.audit_scale_dynamics)
    for token in ("epoch", "frequency_per_epoch", "phase_radians", "sigma_x_median", "sigma_u_median"):
        assert token in source


def test_step_and_figure5a_mode_curvatures_are_separate_records() -> None:
    source = inspect.getsource(experiments.compare_step_figure5a_modes)
    assert "MATCHED_STEP" in source
    assert "FIGURE5A_SHARED_DRIFT" in source
    assert "MODE_SUPPORT_AND_REWARD_AGGREGATION_MISMATCH" in source


def test_v16_optimizer_is_immutable_without_source_fidelity_defect() -> None:
    frozen_path = ROOT / "artifacts/google_pure_v16/frozen_source_normalized_optimizer.json"
    before = file_hash(frozen_path)
    frozen = read_json(frozen_path)
    assert (frozen["mean_learning_rate"], frozen["sigma_learning_rate"],
            frozen["initial_sigma"], frozen["entropy_coefficient"]) == (.32, .08, .15, .01)
    assert file_hash(frozen_path) == before
    assert config()["optimizer_changes_permitted"] is False


def test_figure5c_is_untouched_by_v17() -> None:
    package_source = "\n".join(path.read_text(encoding="utf-8")
                               for path in (ROOT / "src/hdfa_rl_suite/google_pure_v17").glob("*.py"))
    assert "run_figure5c" not in package_source
    assert '"figure5c_executed": False' in package_source


def test_no_forbidden_campaign_is_auto_run() -> None:
    source = inspect.getsource(experiments.run_reduced_postrepair)
    assert "run_cell" not in source  # delegated only to the explicitly reduced helper
    reduced_source = inspect.getsource(experiments._run_reduced_cells)
    assert "AcquisitionMode.VALIDATION" in reduced_source
    assert "AcquisitionMode.REFERENCE" not in reduced_source
    assert config()["reduced_acquisition"]["epochs"] < 1000


def test_v17_import_manifest_is_fail_closed() -> None:
    built = build_import_manifest()
    checked = verify_import_manifest()
    assert built["import_manifest_hash"] == checked["import_manifest_hash"]
    assert built["all_imports_valid"]


def test_sinusoid_helper_uses_source_formula() -> None:
    epochs = np.arange(10)
    assert np.allclose(sinusoid(.1, epochs), np.sin(2 * np.pi * .1 * epochs))
