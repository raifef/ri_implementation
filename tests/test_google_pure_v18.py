from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from hdfa_rl_suite.google_pure_v18 import experiments as v18
from hdfa_rl_suite.google_pure_v18.contracts import NONFINAL
from hdfa_rl_suite.google_pure_v18.io import ROOT, config, file_hash, read_json


def _exact_keys(value):
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            found.append(key)
            found.extend(_exact_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_exact_keys(child))
    return found


def test_v18_protocol_is_bounded_and_staged():
    value = config()
    stages = value["transfer_acquisition"]
    assert stages["intermediate"]["frequency_per_epoch"] == 1 / 300
    assert stages["fast"]["frequency_per_epoch"] == 1 / 150
    assert stages["intermediate"]["epochs"] == 3 * 300
    assert stages["fast"]["epochs"] == 4 * 150
    assert stages["fast"]["analysis_periods"] >= 3
    assert value["optimizer_changes_permitted"] is False
    assert value["heldout_seeds"] == []
    assert set(value["forbidden_auto_runs"]) >= {
        "figure5c", "natural_drift", "heldout", "reference", "source_budget",
    }


def test_explicit_sensitivity_identity_and_no_ambiguous_active_keys():
    result = v18.build_sensitivity_field_cleanup()
    assert result["pass"] is True
    for row in result["explicit_semantics"]:
        assert row["unit_variance_damage_kappa_V"] == .5 * row["hessian_curvature_kappa_H"]
        assert row["quadratic_coefficient_a"] == row["unit_variance_damage_kappa_V"]
    active = {key for key in _exact_keys(result) if key in {"curvature", "normalized_curvature"}}
    assert active == set()
    assert result["frozen_artifacts_rewritten"] is False


def test_continuous_transfer_fixture_is_quantitative():
    result = v18.validate_deterministic_transfer()
    assert result["pass"] is True
    assert result["production_evaluator_changed"] is False
    assert result["old_discrete_recursion_values_used_for_quantitative_claim"] is False
    assert all(row["pass"] for row in result["rows"])
    improvements = {row["label"]: row["measured_normalized_improvement"]
                    for row in result["rows"]}
    assert improvements["slow"] > improvements["intermediate"] > improvements["fast"]


def test_period_bootstrap_and_mechanistic_ordering_are_direct():
    def rows(gain, lag):
        return [{"sine_coefficient": gain * math.cos(lag),
                 "cosine_coefficient": -gain * math.sin(lag)} for _ in range(3)]
    intermediate_bootstrap = v18._bootstrap_period_transfer(rows(.7, .2), draws=100, seed=1)
    fast_bootstrap = v18._bootstrap_period_transfer(rows(.3, .8), draws=100, seed=2)
    intermediate = {
        "bootstrap_uncertainty": intermediate_bootstrap,
        "mean_transfer_regression": {"gain": .7, "phase_lag_radians": .2},
    }
    fast = {
        "bootstrap_uncertainty": fast_bootstrap,
        "mean_transfer_regression": {"gain": .3, "phase_lag_radians": .8},
    }
    result = v18._ordering_gate(intermediate, fast)
    assert result["pass"] is True
    assert result["bootstrap_joint_probability"] == 1.0


def test_three_frequency_ordering_checks_gain_and_phase_together():
    def transfer(gain, lag):
        bootstrap = v18._bootstrap_period_transfer([
            {"sine_coefficient": gain * math.cos(lag),
             "cosine_coefficient": -gain * math.sin(lag)} for _ in range(3)
        ], draws=100, seed=1)
        return {
            "bootstrap_uncertainty": bootstrap,
            "mean_transfer_regression": {"gain": gain, "phase_lag_radians": lag},
        }

    result = v18._three_frequency_ordering_gate(
        transfer(.8, .1), transfer(.5, .4), transfer(.2, .9))
    assert result["point_estimate_pass"] is True
    assert result["bootstrap_joint_probability"] == 1.0
    assert result["pass"] is True

    reversed_phase = v18._three_frequency_ordering_gate(
        transfer(.8, .5), transfer(.5, .4), transfer(.2, .9))
    assert reversed_phase["gain_point_estimate_pass"] is True
    assert reversed_phase["phase_point_estimate_pass"] is False
    assert reversed_phase["pass"] is False


def test_steady_rule_requires_consecutive_period_stability():
    base = {
        "gain": .5, "phase_lag_radians": .4, "sigma_x_median": .15,
        "mean_offset": 0.0, "scale_guard_occupancy": 0.0,
    }
    rows = [{"period_index": index, **base} for index in range(3)]
    assert v18._steady_state_diagnostic("intermediate", rows)["pass"] is True
    rows[-1] = {**rows[-1], "gain": 1.0}
    assert v18._steady_state_diagnostic("intermediate", rows)["pass"] is False


def test_raw_count_decomposition_preserves_all_four_streams():
    result = v18._metric_from_totals({
        "fixed": 1000, "optimal": 200, "learned_mean": 400, "stochastic": 500,
    })
    assert result["I_mean"] == .75
    assert result["I_stochastic"] == .625
    assert result["exploration_damage"] == 100
    assert math.isclose(result["I_mean"] - result["I_stochastic"],
                        result["exploration_damage"] / result["normalization_denominator"])
    assert result["stream_separation_retained"] == list(v18.STREAMS)


def test_figure5b_note_is_per_epoch_not_aggregate():
    result = v18.build_figure5b_learning_rate_note()
    assert result["pass"] is True
    assert result["per_epoch"] is True
    assert result["per_update"] is True
    assert result["aggregate_run_total"] is False
    assert math.isclose(result["alpha_times_kappa_H"], .0064)
    assert result["figure5b_executed"] is False


def test_all_v18_outputs_remain_nonfinal():
    for name in ("sensitivity_field_cleanup", "deterministic_fixture_quantitative_validation",
                 "figure5b_learning_rate_note"):
        value = read_json(ROOT / f"artifacts/google_pure_v18/{name}.json")
        for key, expected in NONFINAL.items():
            assert value[key] is expected


def test_fresh_staged_transfer_lineage_and_truthful_gate():
    intermediate = read_json(ROOT / "artifacts/google_pure_v18/transfer_intermediate.json")
    fast = read_json(ROOT / "artifacts/google_pure_v18/transfer_fast.json")
    frozen = read_json(ROOT / "artifacts/google_pure_v16/frozen_source_normalized_optimizer.json")
    for value in (intermediate, fast):
        assert value["controller_mode"] == "PAPER_DIRECT_SIGMA"
        assert value["controller_hash"] == frozen["optimizer_bundle_hash"]
        assert value["parameterization"] == "DIRECT_SIGMA_SOURCE_EXACT"
        assert value["direct_mean_transfer_identifiable"] is True
        assert value["fresh_acquisition"] is True
        assert value["fresh_v18_acquisition_campaign"] is True
        assert value["reused_shard_ids"] == []
        assert value["four_stream_qec_cycles"] > 0
        assert set(value["stream_decomposition"]["stream_separation_retained"]) == set(v18.STREAMS)
        for key, expected in NONFINAL.items():
            assert value[key] is expected
    assert intermediate["steady_periodic_identification_accepted"] is True
    assert fast["steady_periodic_identification_accepted"] is False
    assert fast["stage_ab_ordering"]["pass"] is True
    assert fast["steady_state_diagnostic"]["stable_tail_transitions"] == 1
    assert fast["steady_state_diagnostic"]["required_stable_transitions"] == 2


def test_delta_and_acceptance_provenance_fail_closed():
    delta = read_json(ROOT / "artifacts/google_pure_v18/delta_min_provenance.json")
    readiness = read_json(ROOT / "artifacts/google_pure_v18/paired_acceptance_readiness.json")
    slow = read_json(ROOT / "artifacts/google_pure_v18/transfer_slow.json")
    slow_checkpoint = ROOT / "artifacts/google_pure_v18/acquisition/slow/checkpoint.json"
    assert delta["selection_order"] < delta["new_transfer_acquisition_order"]
    assert delta["any_v18_slow_fast_scientific_result_visible_at_selection"] is False
    assert delta["selection_independent_of_new_v18_transfer_outcomes"] is True
    assert readiness["pass"] is False
    assert readiness["available_complete_matched_pairs"] == 0
    assert readiness["v17_24_epoch_units_rejected"] is True
    assert readiness["paired_acceptance_executed"] is False
    assert slow_checkpoint.is_file()
    assert slow["fresh_v18_acquisition_campaign"] is True
    assert slow["checkpoint_sha256"] == file_hash(slow_checkpoint)
    assert slow["stage_slow_intermediate_fast_ordering"]["pass"] is True
    assert slow["fast_transfer_approval"]["pass"] is True


def test_new_v18_root_artifacts_forbid_ambiguous_exact_sensitivity_keys():
    forbidden = {"curvature", "normalized_curvature"}
    for path in (ROOT / "artifacts/google_pure_v18").glob("*.json"):
        assert forbidden.isdisjoint(_exact_keys(read_json(path))), path


def test_v18_cli_surface_has_explicit_stages_and_no_combined_long_run():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    required = {
        "hdfa-google-v18-run-transfer-intermediate",
        "hdfa-google-v18-run-transfer-fast",
        "hdfa-google-v18-run-transfer-slow",
        "hdfa-google-v18-audit-delta-min-provenance",
        "hdfa-google-v18-check-acceptance-readiness",
        "hdfa-google-v18-report",
    }
    assert all(name in pyproject for name in required)
    assert "hdfa-google-v18-run-full-acceptance" not in pyproject
