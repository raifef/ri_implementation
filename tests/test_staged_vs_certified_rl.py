from dataclasses import replace

import numpy as np

from hdfa_rl_suite.google_rl_certification.drift_tracking import one_control_landscape
from hdfa_rl_suite.staged_comparison.config import (
    FUTURE_TRACK_B_CONFIRMATORY_SEEDS,
    PROTECTED_CONFIRMATORY_V3_SEEDS,
    TrackBConfig,
)
from hdfa_rl_suite.staged_comparison.controllers import run_arm
from hdfa_rl_suite.staged_comparison.substrate import (
    build_common_substrate,
    expected_detector_rates,
    expected_logical_rate,
    plant_a_contract,
    plant_a_scenarios,
    plant_b_contract,
    plant_b_scenarios,
    track_a_freeze,
    validate_scenario_clones,
)


def test_track_a_reference_is_certified_and_frozen_before_track_b():
    frozen = track_a_freeze()
    assert frozen["high_shot_status"] == "HIGH_SHOT_REFERENCE_CERTIFIED"
    assert frozen["frozen_before_comparative_development"]
    assert len(frozen["aggregate_sha256"]) == 64


def test_development_and_confirmatory_seed_firewalls_are_disjoint():
    config = TrackBConfig()
    assert not set(config.development_seeds).intersection(PROTECTED_CONFIRMATORY_V3_SEEDS)
    assert not set(config.development_seeds).intersection(FUTURE_TRACK_B_CONFIRMATORY_SEEDS)


def test_plant_a_uses_exact_track_a_detector_formula():
    contract = plant_a_contract()
    landscape = one_control_landscape(lambda _epoch: .17, curvature=.08, floor=.012)
    actions = np.asarray([[-.2], [0.0], [.3]])
    expected = expected_detector_rates(contract, actions, np.asarray([.17]), np.zeros(1))
    reference = landscape.expected_rates(actions, 0.0)
    np.testing.assert_allclose(expected, reference, atol=1e-14)


def test_rich_plant_is_controllable_monotone_and_logically_aligned():
    contract = plant_b_contract()
    optimum = np.zeros(len(contract.control_ids))
    radii = np.asarray((0.0, .05, .10, .15))
    actions = np.zeros((len(radii), len(contract.control_ids)))
    actions[:, 0] = radii
    rates = expected_detector_rates(contract, actions, optimum, np.zeros_like(optimum))
    logical = expected_logical_rate(contract, rates)
    assert np.all(np.diff(rates[:, 0]) > 0)
    assert np.all(np.diff(logical) > 0)
    assert float(rates.max()) < contract.maximum_detector_probability


def test_common_scenario_clones_are_equal_but_not_aliased():
    config = replace(TrackBConfig(), development_seeds=(6201,),
                     plant_a_intervals=12, plant_b_intervals=14, onset_interval=2)
    scenario = plant_b_scenarios(config, 6201)[0]
    checks = validate_scenario_clones(scenario, ("fixed", "oracle", "staged"))
    assert all(checks.values())


def test_no_residual_stratum_abstains_without_residual_candidates():
    config = replace(
        TrackBConfig(), development_seeds=(6203,), plant_a_intervals=12,
        plant_b_intervals=16, onset_interval=2, final_window_intervals=4,
        stage2_probe_cycles=4096, residual_candidate_cycles=4096)
    contract = plant_b_contract(config)
    scenario = plant_b_scenarios(config, 6203)[0]
    run = run_arm(
        contract, scenario, "predictive_hdfa_conditional_residual_rl", config)
    residual_candidates = sum(
        ((row["stage_evidence"] or {}).get("stage6") or {}).get("candidate_count", 0)
        for row in run["trajectory"])
    assert residual_candidates == 0
    assert run["controller_truth_access_count"] == 0
    assert all(row["policy_transaction"]["lifecycle_valid"] for row in run["trajectory"])


def test_common_substrate_manifest_passes_without_consuming_confirmation(tmp_path):
    config = replace(TrackBConfig(), development_seeds=(6205,),
                     plant_a_intervals=12, plant_b_intervals=14, onset_interval=2)
    manifest = build_common_substrate(config, tmp_path)
    assert manifest["all_common_substrate_checks_pass"]
    assert not manifest["confirmatory_seeds_consumed"]
    assert (tmp_path/"common_substrate_manifest.json").exists()

