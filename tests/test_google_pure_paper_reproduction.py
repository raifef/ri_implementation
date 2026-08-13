from __future__ import annotations

import json
from pathlib import Path

import pytest

from hdfa_rl_suite.google_pure_paper_reproduction.claim_registry import claims
from hdfa_rl_suite.google_pure_paper_reproduction.experiment_families import (
    EvidenceClass,
    ExperimentFamily,
    assert_claim_compatible,
    assert_merge_compatible,
    final_evidence_allowed,
    guard_seed,
)
from hdfa_rl_suite.google_pure_paper_reproduction.paper_figures import (
    acquire,
    assigned_conditions,
    build_protocol,
    default_config,
)
from hdfa_rl_suite.google_pure_paper_reproduction.reporting import audit_pure_namespace
from hdfa_rl_suite.google_pure_paper_reproduction.source_registry import source_contract
from hdfa_rl_suite.google_pure_paper_reproduction.sparse_scaling import total_controls
from hdfa_rl_suite.google_pure_paper_reproduction.storage import REQUIRED_DIRS


def test_required_package_surface_exists() -> None:
    package = Path("src/hdfa_rl_suite/google_pure_paper_reproduction")
    required = {
        "__init__.py", "source_registry.py", "claim_registry.py", "experiment_families.py", "provenance.py",
        "public_data.py", "paper_figures.py", "paper_tables.py", "panel_a.py", "panel_b.py", "panel_c.py",
        "step_response.py", "natural_drift.py", "randomized_recovery.py", "sparse_scaling.py", "side_by_side.py",
        "comparison_metrics.py", "reporting.py", "storage.py", "validation.py", "cli.py",
    }
    assert required <= {path.name for path in package.glob("*.py")}
    assert set(REQUIRED_DIRS) >= {"source_contract", "claim_registry", "side_by_side", "validation", "reports"}


def test_source_contract_separates_public_and_proprietary_simulation() -> None:
    contract = source_contract()
    assert contract["sources"]["public_data_release"]["sha256"] == "39563ad104bcbec2e36907373b25d176cf7f2a2e3852d8390623223dadf96e76"
    assert contract["anti_conflation"]["public_data_is_hardware_output_not_simulator_source"]
    assert contract["figures"]["5a"]["public"]["candidate_cycles"] == 1_800_000_000
    assert "proprietary" in contract["code_availability"].lower()


def test_claim_registry_has_all_statuses_and_anti_conflation() -> None:
    rows = claims(); statuses = {row["status"] for row in rows}
    assert EvidenceClass.PUBLIC_EXACT.value in statuses
    assert EvidenceClass.SYNTHETIC.value in statuses
    assert EvidenceClass.VISUAL.value in statuses
    assert EvidenceClass.UNIDENTIFIABLE.value in statuses
    control = next(row for row in rows if row["claim_id"] == "drift.control_only_stability")
    decoder = next(row for row in rows if row["claim_id"] == "drift.control_plus_decoder_stability")
    assert control["must_not_be_combined_with"] and decoder["must_not_be_combined_with"]


def test_claim_family_mismatch_fails() -> None:
    endpoint = next(row for row in claims() if row["claim_id"] == "headline.surface_d7_alphaqubit2")
    assert_claim_compatible(endpoint, ExperimentFamily.PUBLIC_ENDPOINT_DATA_REPRODUCTION)
    with pytest.raises(RuntimeError, match="cannot be combined"):
        assert_claim_compatible(endpoint, ExperimentFamily.FIGURE5A_REAL_TIME_STEERING)


def test_mode_mixing_and_wrong_family_merge_fail() -> None:
    common = {"protocol_hash":"p","controller_hash":"c","plant_hash":"q","graph_hash":"g"}
    first = {"provenance":{**common,"experiment_family":"FIGURE5A_REAL_TIME_STEERING","mode":"smoke"}}
    second = {"provenance":{**common,"experiment_family":"FIGURE5A_REAL_TIME_STEERING","mode":"reference"}}
    with pytest.raises(RuntimeError, match="mixed mode"):
        assert_merge_compatible([first, second])
    second["provenance"]["mode"] = "smoke"; second["provenance"]["experiment_family"] = "FIGURE5B_SPARSE_SCALING"
    with pytest.raises(RuntimeError, match="mixed experiment_family"):
        assert_merge_compatible([first, second])


def test_merge_compatibility_allows_mixed_per_shard_freshness() -> None:
    common = {
        "experiment_family": "FIGURE5A_REAL_TIME_STEERING",
        "protocol_hash": "p",
        "controller_hash": "c",
        "plant_hash": "q",
        "graph_hash": "g",
        "mode": "validation",
        "reused_shard_ids": [],
    }
    fresh = {"provenance": {**common, "fresh_acquisition": True}}
    resumed = {"provenance": {**common, "fresh_acquisition": False}}

    # Freshness is per-shard acquisition provenance, not an experiment identity.
    # The merge layer combines it conservatively with all(...) and still rejects
    # any non-fresh evidence when the frozen protocol explicitly requires fresh data.
    assert_merge_compatible([fresh, resumed])


def test_smoke_and_validation_can_never_be_final() -> None:
    assert not final_evidence_allowed(mode="smoke", complete=True, scientifically_valid=True)
    assert not final_evidence_allowed(mode="validation", complete=True, scientifically_valid=True)
    assert final_evidence_allowed(mode="reference", complete=True, scientifically_valid=True)
    assert not final_evidence_allowed(mode="reference", complete=False, scientifically_valid=True)


def test_reserved_and_retired_seeds_fail() -> None:
    for seed in (10101, 12101, 12112):
        with pytest.raises(RuntimeError): guard_seed(seed)
    guard_seed(13101)


def test_protocols_preserve_paper_geometry_and_controller_hash() -> None:
    panel_a = build_protocol(ExperimentFamily.FIGURE5A_REAL_TIME_STEERING.value, mode="reference")
    assert panel_a["config"]["epochs"] == 1000
    assert panel_a["config"]["candidates"] == 50
    assert panel_a["config"]["cycles_per_candidate"] == 36000
    assert panel_a["controller_hash"] and panel_a["controller_code_hash"]
    panel_b = default_config(ExperimentFamily.FIGURE5B_SPARSE_SCALING.value, "smoke")
    assert panel_b["parameters_per_gate"] == [1, 10, 30]
    assert panel_b["distances"] == [3, 15]


def test_worker_condition_partitions_are_disjoint_complete_and_stable() -> None:
    protocol = build_protocol(ExperimentFamily.FIGURE5A_REAL_TIME_STEERING.value, mode="smoke")
    partitions = [assigned_conditions(protocol, worker_index=index, worker_count=4)
                  for index in range(4)]
    indices = [index for partition in partitions for index, _ in partition]
    assert sorted(indices) == list(range(protocol["condition_count"]))
    assert len(indices) == len(set(indices))
    assert partitions == [assigned_conditions(protocol, worker_index=index, worker_count=4)
                          for index in range(4)]
    with pytest.raises(ValueError, match="at least one"):
        assigned_conditions(protocol, worker_count=0)
    with pytest.raises(ValueError, match="worker_index"):
        assigned_conditions(protocol, worker_index=4, worker_count=4)


def test_worker_acquisition_skips_existing_assigned_shards(monkeypatch) -> None:
    import hdfa_rl_suite.google_pure_paper_reproduction.paper_figures as figures
    from hdfa_rl_suite.google_pure_v7.config import canonical_hash

    conditions = [{"seed": seed} for seed in range(4)]
    protocol = {"mode": "smoke", "experiment_family": "TEST_FAMILY",
                "protocol_hash": "protocol", "conditions": conditions,
                "condition_count": len(conditions)}
    existing_condition = conditions[1]
    existing_identity = {"family": "TEST_FAMILY", "protocol_hash": "protocol",
                         "condition": existing_condition}
    existing_id = canonical_hash(existing_identity)
    executed = []

    class Module:
        @staticmethod
        def acquire_condition(_protocol, condition):
            executed.append(condition)
            return {"seed": condition["seed"]}

    monkeypatch.setattr(figures, "discover", lambda _protocol: [{"shard_id": existing_id}])
    monkeypatch.setattr(figures, "_module", lambda _family: Module)
    monkeypatch.setattr(figures, "write_shard", lambda _protocol, condition, _data: {
        "shard_id": canonical_hash({"family": "TEST_FAMILY", "protocol_hash": "protocol",
                                    "condition": condition})})
    monkeypatch.setattr(figures, "atomic_json", lambda _path, _value: None)
    result = acquire(protocol, worker_index=1, worker_count=2)
    assert executed == [conditions[3]]
    assert result["assigned_shards"] == 2
    assert result["preexisting_assigned_shards"] == 1
    assert result["completed_this_call"] == 1


def test_one_hour_profile_is_bounded_complete_geometry_and_nonfinal() -> None:
    from hdfa_rl_suite.google_pure_paper_reproduction.hourly_workflow import (
        build_one_hour_protocols,
        profile_plan,
    )

    protocols = build_one_hour_protocols()
    plan = profile_plan(protocols, max_workers=6)
    assert plan["mode"] == "validation"
    assert plan["final_evidence"] is False
    assert plan["paper_equivalence_claim_permitted"] is False
    assert plan["literal_paper_budget_preserved"] is False
    assert plan["estimated_wall_minutes"] == [35, 55]
    assert plan["calibration_observed_seconds"]["figure5a_representative_cell"] == 305.92
    assert len(plan["families"]) == 6
    by_family = {row["experiment_family"]: row for row in plan["families"]}
    assert by_family["FIGURE5A_REAL_TIME_STEERING"]["conditions"] == 30
    assert by_family["FIGURE5B_SPARSE_SCALING"]["conditions"] == 42
    assert by_family["FIGURE5C_CONVERGENCE_LAW"]["conditions"] == 42
    assert by_family["NATURAL_DRIFT_SPECTRAL_SUPPRESSION"]["conditions"] == 6
    assert by_family["FIGURE5A_REAL_TIME_STEERING"]["checkpoint_every_candidates"] == 20
    assert by_family["STEP_RESPONSE_INJECTED_DRIFT"]["record_storage"] == "compact_directional"
    assert all(row["cycles_per_candidate"] == 36_000 for row in plan["families"])
    assert all(protocol["mode"] == "validation" for protocol in protocols)
    assert all(protocol["config"]["paper_equivalence_claim_permitted"] is False
               for protocol in protocols)


def test_one_hour_profile_refuses_oversubscription() -> None:
    from hdfa_rl_suite.google_pure_paper_reproduction.hourly_workflow import profile_plan

    with pytest.raises(ValueError, match="max_workers"):
        profile_plan([], max_workers=7)


def test_one_hour_worker_failure_is_attributed_and_persisted(monkeypatch, tmp_path) -> None:
    import concurrent.futures
    from hdfa_rl_suite.google_pure_paper_reproduction import hourly_workflow

    protocol = {
        "experiment_family": "TEST_FAMILY",
        "protocol_hash": "test-protocol",
        "condition_count": 1,
        "conditions": [{"seed": 17, "frequency": 0.01}],
    }

    def fail_acquisition(*_args, **_kwargs):
        raise ValueError("candidate left physical action domain")

    monkeypatch.setattr(hourly_workflow, "acquire", fail_acquisition)
    monkeypatch.setattr(hourly_workflow, "ProcessPoolExecutor",
                        concurrent.futures.ThreadPoolExecutor)
    monkeypatch.setattr(hourly_workflow, "initialise_layout", lambda: tmp_path)

    with pytest.raises(RuntimeError, match="TEST_FAMILY condition 0"):
        hourly_workflow._run_acquisitions([protocol], max_workers=1)

    manifest = json.loads(
        (tmp_path / "reports/one_hour_validation_worker_failure.json").read_text(
            encoding="utf-8"))
    assert manifest["failure"]["experiment_family"] == "TEST_FAMILY"
    assert manifest["failure"]["condition_index"] == 0
    assert manifest["failure"]["exception_type"] == "ValueError"
    assert manifest["failure"]["condition"] == protocol["conditions"][0]
    assert manifest["remaining_work_cancelled"] is True
    assert manifest["final_evidence"] is False


def test_sparse_scaling_anchor_and_no_dense_allocation_contract() -> None:
    assert total_controls(15, 30) == 38670
    source = Path("src/hdfa_rl_suite/google_pure_paper_reproduction/panel_b.py").read_text(encoding="utf-8")
    assert '"dense_parameter_matrix_allocated": False' in source


def test_pure_namespace_import_firewall() -> None:
    audit = audit_pure_namespace()
    assert audit["pure_import_firewall_pass"], audit["forbidden_imports"]


def test_public_reproduction_is_direct_and_synthetic_free() -> None:
    path = Path("artifacts/google_pure_paper_reproduction/public_data_reproduction/public_data_reproduction.json")
    assert path.exists()
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["public_data_direct"] is True
    assert value["synthetic_data_present"] is False
    exact = [row for row in value["rows"] if row["verdict"] == "EXACT_PUBLIC_REPRODUCTION"]
    assert len(exact) >= 4


def test_all_cli_commands_registered() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    for suffix in ("fig5a", "fig5b", "fig5c", "natural-drift", "randomized-recovery", "step-response"):
        for action in ("plan", "acquire", "merge", "validate", "plot", "compare"):
            assert f"hdfa-google-paper-{suffix}-{action} =" in text
    assert 'version = "0.12.0"' in text


def test_prior_figure5_smoke_is_preserved_and_reclassified() -> None:
    from hdfa_rl_suite.google_pure_paper_reproduction.reporting import reclassify_prior_figure5_smoke
    result = reclassify_prior_figure5_smoke()
    assert result["classification"] == "SMOKE_RENDER_ONLY"
    assert result["source_deleted"] is False
    assert result["eligible_as_paper_reproduction"] is False
