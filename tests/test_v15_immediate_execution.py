from __future__ import annotations

import json

import numpy as np
import pytest

from hdfa_rl_suite.google_pure_paper_reproduction.experiment_families import ExperimentFamily
from hdfa_rl_suite.google_pure_paper_reproduction.hourly_workflow import FAMILY_PROFILES, WorkflowMode
from hdfa_rl_suite.google_pure_paper_reproduction.paper_figures import build_protocol
from hdfa_rl_suite.google_pure_paper_reproduction.storage import _validated_v15
from hdfa_rl_suite.google_pure_source_exact.source_normalization import (
    SourceNormalizationBoundary,
    require_v15_boundary_provenance,
)
from hdfa_rl_suite.google_pure_v15.immediate_execution import audit_calibration_objective
from hdfa_rl_suite.google_pure_v15.io import CONFIG_ROOT


def test_v15_boundary_applies_exactly_once_and_rejects_wrong_lineage():
    boundary = SourceNormalizationBoundary.from_training_objective(
        "TEST", np.asarray([.001, .002]), control_ids=("a", "b"))
    application = boundary.apply(np.asarray([.25, -.5]))
    assert np.allclose(application.native, boundary.native_scale * [.25, -.5])
    assert application.token["boundary_apply_count"] == 1
    with pytest.raises(RuntimeError, match="exactly once"):
        boundary.apply(np.zeros(2), application_count=1)
    with pytest.raises(RuntimeError, match="control order"):
        boundary.apply(np.zeros(2), control_order_hash="wrong")
    with pytest.raises(RuntimeError, match="sensitivity map"):
        boundary.apply(np.zeros(2), sensitivity_map_hash="wrong")


def test_required_paper_drivers_freeze_v15_hashes_before_acquisition():
    families = (
        ExperimentFamily.STEP_RESPONSE_INJECTED_DRIFT.value,
        ExperimentFamily.RANDOMIZED_RECOVERY_AFTER_SPOIL.value,
        ExperimentFamily.FIGURE5A_REAL_TIME_STEERING.value,
        ExperimentFamily.FIGURE5B_SPARSE_SCALING.value,
    )
    for family in families:
        protocol = build_protocol(family, mode="smoke")
        assert protocol["implementation_version"] == "google_pure_v15"
        assert protocol["sensitivity_map_hash"]
        assert protocol["sensitivity_definition_hash"]
        assert protocol["calibration_bundle_hash"]
        assert protocol["detector_degree_audit_hash"]
        assert protocol["boundary_transform_hash"]
        assert protocol["experiment_driver_hash"]


def test_storage_fails_closed_without_executed_boundary_provenance():
    protocol = build_protocol(ExperimentFamily.STEP_RESPONSE_INJECTED_DRIFT.value,
                              mode="smoke")
    with pytest.raises(RuntimeError, match="missing mandatory V15"):
        _validated_v15(protocol, {})


def test_objective_consumed_by_training_is_the_calibrated_objective():
    result = audit_calibration_objective()
    assert result["all_drivers_calibrated_to_the_objective_consumed_by_training"]
    assert all(row["minimum_conditioned_curvature"] == pytest.approx(.01)
               for row in result["rows"])
    assert all(row["maximum_conditioned_curvature"] == pytest.approx(.01)
               for row in result["rows"])


def test_workflow_modes_are_explicit():
    assert {item.value for item in WorkflowMode} == {
        "ANALYSIS_ONLY", "SMOKE_ACQUISITION", "ONE_HOUR_FRESH_ACQUISITION",
        "REFERENCE_ACQUISITION"}


def test_lineage_is_v12_v13_v15_with_no_v14_requirement():
    protocol = json.loads((CONFIG_ROOT / "protocol.json").read_text(encoding="utf-8"))
    assert "require_v14_lineage" not in protocol.get("gates", {})
