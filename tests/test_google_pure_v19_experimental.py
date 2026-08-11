from __future__ import annotations

import inspect

from hdfa_rl_suite.google_pure_v19_experimental import acquisition
from hdfa_rl_suite.google_pure_v19_experimental.controller import (
    CONTROLLER_MODE,
    PARAMETERIZATION,
    SCALE_OBJECTIVE,
)
from hdfa_rl_suite.google_pure_v19_experimental.dynamic_validation import _controller_spec
from hdfa_rl_suite.google_pure_v19_experimental.io import NONFINAL, ROOT, read_json
from hdfa_rl_suite.google_pure_v19_experimental.matched_validation import (
    MATCHED_ARTIFACT_ROOT,
    MATCHED_CONFIG_PATH,
    PILOT_ROOT,
)


def test_equilibrium_terminology_forbids_an_identified_source_scale_claim():
    derivation = read_json(ROOT / "artifacts/google_pure_v19/sigma_equilibrium_derivation.json")
    comparison = read_json(ROOT / "artifacts/google_pure_v19/sigma_equilibrium_comparison.json")
    assert derivation["source_scale_hyperparameters_identifiable"] is False
    assert derivation["source_exact_equilibrium_claim_permitted"] is False
    assert derivation["implemented_objective"] == (
        "IMPLEMENTED_SOURCE_STYLE_OBJECTIVE_WITH_INHERITED_HYPERPARAMETERS")
    assert comparison["source_scale_hyperparameters_identifiable"] is False
    encoded = (ROOT / "artifacts/google_pure_v19/FINAL_REPORT.md").read_text(encoding="utf-8")
    assert "implemented source-style/inherited-beta equilibrium" in encoded
    assert "source-objective equilibrium" not in encoded


def test_public_analogue_controller_is_a_distinct_beta_over_p_branch():
    controller = _controller_spec()
    assert controller.active_dimensions == 41
    assert controller.effective_entropy_coefficient == (
        controller.inherited_entropy_coefficient / 41)
    assert controller.controller_hash != controller.frozen_parent_controller_hash
    assert controller.identity_payload["controller_mode"] == CONTROLLER_MODE
    assert controller.identity_payload["parameterization"] == PARAMETERIZATION
    assert controller.identity_payload["scale_objective"] == SCALE_OBJECTIVE
    assert controller.identity_payload["source_exact"] is False
    assert controller.identity_payload["source_scale_hyperparameters_identifiable"] is False


def test_experimental_acquisition_has_bounded_windows_retry_and_explicit_identity():
    source = inspect.getsource(acquisition)
    assert "WINDOWS" not in source  # mechanism is platform-neutral despite its provenance label
    assert "except PermissionError" in source
    assert "_source_atomic_json" in source
    assert '"controller_mode": CONTROLLER_MODE' in source
    assert '"source_exact": False' in source
    assert "run_cell(" not in source


def test_failed_low_resolution_pilot_is_preserved_without_promotion():
    pilot = read_json(PILOT_ROOT / "status.json")
    revision = read_json(PILOT_ROOT / "preflight_manifest_revision_2.json")
    assert pilot["execution_complete"] is True
    assert pilot["pass"] is False
    assert pilot["ordering"]["gain_point_ordering_pass"] is True
    assert pilot["ordering"]["phase_point_ordering_pass"] is False
    assert pilot["sampled_policy_I_positive_all_frequencies"] is False
    assert pilot["frozen_source_branch_unchanged"] is True
    assert revision["execution_only_repair"] == "WINDOWS_ATOMIC_REPLACE_BOUNDED_RETRY"
    assert revision["scientific_protocol_changed"] is False


def test_matched_protocol_is_small_nonheldout_and_uses_prior_v18_horizons():
    protocol = read_json(MATCHED_CONFIG_PATH)
    assert protocol["heldout_seeds"] == []
    assert protocol["automatic_campaigns_permitted"] == []
    assert protocol["mean_hyperparameters_changed"] is False
    assert [protocol["frequencies"][label]["epochs"]
            for label in ("slow", "intermediate", "fast")] == [3000, 900, 750]
    assert all(row["candidates_per_epoch"] == 8
               for row in protocol["frequencies"].values())
    assert all(row["qec_cycles_per_candidate"] == 12000
               for row in protocol["frequencies"].values())


def test_matched_dynamic_result_passes_ordering_but_fails_fast_positive_I():
    status = read_json(MATCHED_ARTIFACT_ROOT / "status.json")
    assert status["execution_complete"] is True
    assert status["ordering"]["gain_point_ordering_pass"] is True
    assert status["ordering"]["phase_point_ordering_pass"] is True
    assert status["ordering"]["bootstrap_joint_ordering_probability"] >= 0.8
    assert status["ordering"]["pass"] is True
    by_label = {row["label"]: row for row in status["rows"]}
    assert by_label["slow"]["stream_decomposition"]["I_stochastic"] > 0
    assert by_label["intermediate"]["stream_decomposition"]["I_stochastic"] > 0
    assert by_label["fast"]["stream_decomposition"]["I_stochastic"] < 0
    assert by_label["fast"]["stream_decomposition"]["I_mean"] < 0
    assert status["sampled_policy_I_positive_all_frequencies"] is False
    assert status["pass"] is False
    assert status["frozen_source_branch_unchanged"] is True
    assert status["prior_failed_pilot_preserved"] is True


def test_experimental_outputs_remain_nonfinal_and_branch_provenance_is_internal():
    for root in (PILOT_ROOT, MATCHED_ARTIFACT_ROOT):
        status = read_json(root / "status.json")
        for key, expected in NONFINAL.items():
            assert status[key] == expected
        assert status["forbidden_auto_runs_launched"] == []
        assert (root / "REPORT.md").is_file()
        for label in ("slow", "intermediate", "fast"):
            transfer = read_json(root / f"transfer_{label}.json")
            assert transfer["controller_mode"] == CONTROLLER_MODE
            assert transfer["parameterization"] == PARAMETERIZATION
            assert transfer["scale_objective"] == SCALE_OBJECTIVE
            assert transfer["controller_hash"] != transfer["frozen_parent_controller_hash"]
            assert transfer["mean_hyperparameters_changed"] is False
            assert transfer["forbidden_auto_runs_launched"] == []


def test_experimental_cli_commands_are_registered_once():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    commands = (
        "hdfa-google-v19-run-public-analogue-three-frequency",
        "hdfa-google-v19-run-public-analogue-matched-three-frequency",
    )
    assert all(pyproject.count(command) == 1 for command in commands)
