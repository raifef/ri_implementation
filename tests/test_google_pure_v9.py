from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from google_rl_reimplementation.google_pure_v9.contracts import (
    ControllerConfig,
    TemporalProtocol,
    controller_selection_gates,
    entropy_operationality,
    five_policy_decomposition,
    scale_floor_classification,
    validate_source_choices,
    window_stability,
)
from google_rl_reimplementation.google_pure_v9.studies import corrected_fault_classification


def test_initial_and_minimum_scales_are_independent():
    value = ControllerConfig(initial_scale=0.02, minimum_scale=0.001, maximum_scale=0.25)
    assert value.initial_scale == 0.02 and value.minimum_scale == 0.001
    with pytest.raises(ValueError):
        ControllerConfig(initial_scale=0.02, minimum_scale=0.02, maximum_scale=0.25)


def test_floor_cannot_be_blamed_when_never_reached():
    assert scale_floor_classification(0.0) == "MINIMUM_SCALE_FLOOR_EFFECT_NOT_ESTABLISHED"
    assert "FAILURE" not in scale_floor_classification(0.25)


def test_entropy_implementation_and_operationality_are_separate():
    rows = [
        {"mean_scale": 0.1, "native_candidate_displacement": 0.2, "D_exploration": 0.3, "I_candidate": -0.1},
        {"mean_scale": 0.1, "native_candidate_displacement": 0.2, "D_exploration": 0.3, "I_candidate": -0.1},
    ]
    result = entropy_operationality(rows)
    corrected = corrected_fault_classification()
    assert not result["operational"]
    assert corrected["entropy"][0] == "ENTROPY_IMPLEMENTATION_PASS"


def test_temporal_aliasing_is_a_protocol_failure():
    value = corrected_fault_classification()
    assert value["temporal"] == ["TEMPORAL_IMPLEMENTATION_PASS", "TEMPORAL_EVALUATION_PROTOCOL_FAILURE"]


def test_validation_requires_phases_frequencies_and_complete_periods():
    with pytest.raises(ValueError):
        TemporalProtocol((1 / 60,), (0.0,), 1, 4, 1, (19001,), "validation")
    value = TemporalProtocol(
        (1 / 300, 1 / 150, 1 / 60),
        (0.0, 2 * np.pi / 3, 4 * np.pi / 3),
        1,
        5,
        1,
        (19001,),
        "validation",
    )
    assert value.primary_periods == 5 and len(value.phases) == 3


def test_window_sensitivity_is_explicit():
    primary = {"I_mean": 0.2, "I_candidate": 0.1, "D_tracking": 0.3, "D_exploration": 0.1}
    close = {"I_mean": 0.21, "I_candidate": 0.09, "D_tracking": 0.31, "D_exploration": 0.09}
    assert window_stability(primary, close, close, tolerance=0.05)["stable"]
    far = {**close, "I_candidate": -0.5}
    assert not window_stability(primary, far, close, tolerance=0.05)["stable"]


def test_every_cell_requires_the_five_policy_decomposition():
    costs = {"fixed": 5.0, "oracle": 1.0, "oracle_with_scale": 1.5, "learned_mean": 3.0, "sampled_candidates": 3.5}
    value = five_policy_decomposition(costs)
    assert value["I_fixed"] == 0 and value["I_oracle"] == 1 and value["decomposition_identity_pass"]
    with pytest.raises(ValueError):
        five_policy_decomposition({"fixed": 1.0})


def _selection(**overrides):
    value = {
        "I_mean_ci_lower": 0.1,
        "I_candidate_phase_average": 0.1,
        "D_fixed": 1.0,
        "D_exploration": 0.2,
        "tracking_gain_ci_lower": 0.1,
        "phase_identifiable": True,
        "window_stable": True,
        "clipping_fraction": 0.001,
        "entropy_operational": True,
        "held_out_protocol_frozen": True,
        "plant_hash_unchanged": True,
        "phase_count": 3,
        "mode": "validation",
    }
    value.update(overrides)
    return controller_selection_gates(value)


def test_mean_and_candidate_improvement_are_both_required():
    assert _selection()["eligible"]
    assert not _selection(I_mean_ci_lower=-0.1)["eligible"]
    assert not _selection(I_candidate_phase_average=-0.1)["eligible"]


def test_clipping_guard_and_evidence_mode_fail_closed():
    assert not _selection(clipping_fraction=0.011)["eligible"]
    assert not _selection(mode="smoke")["eligible"]


def test_source_classification_is_complete_and_pure():
    choices = {
        "initial_scale": "SOURCE_ANCHORED",
        "minimum_scale": "SOURCE_UNSPECIFIED_BUT_PREREGISTERED",
        "maximum_scale": "SOURCE_LITERAL",
        "mean_learning_rate": "SOURCE_ANCHORED",
        "scale_learning_rate": "SOURCE_UNSPECIFIED_BUT_PREREGISTERED",
        "entropy_coefficient": "SOURCE_ANCHORED",
        "replay_capacity_epochs": "SOURCE_LITERAL",
        "update_passes": "SOURCE_LITERAL",
        "optimizer": "SOURCE_LITERAL",
    }
    validate_source_choices(choices)
    choices["optimizer"] = "NON_SOURCE_EXTENSION"
    with pytest.raises(ValueError):
        validate_source_choices(choices)


def test_protected_seed_registry_remains_unused():
    with pytest.raises(RuntimeError):
        TemporalProtocol((1 / 12,), (0.0,), 1, 5, 1, (12101,), "smoke")
    with pytest.raises(RuntimeError):
        TemporalProtocol((1 / 12,), (0.0,), 1, 5, 1, (10101,), "smoke")


def test_v9_commands_use_only_the_standalone_command_namespace():
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    for name in (
        "import-v8-audits",
        "correct-root-cause-classification",
        "plan-stage-a",
        "run-stage-a",
        "report-stage-a",
        "plan-stage-b",
        "run-stage-b",
        "report-stage-b",
        "plan-stage-c",
        "run-stage-c",
        "report-stage-c",
        "freeze-held-out-protocol",
        "run-held-out-validation",
        "select-controller",
        "report-root-cause-update",
        "status",
    ):
        assert f"google-rl-v9-{name} =" in text
