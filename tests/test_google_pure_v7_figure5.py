from __future__ import annotations
import ast,json
from pathlib import Path
import numpy as np
import pytest

from hdfa_rl_suite.google_pure_v7.controller import resolve_production_controller
from hdfa_rl_suite.google_pure_v7.figure5.accounting import acquisition_accounting,total_controls
from hdfa_rl_suite.google_pure_v7.figure5.common import SOURCE_STATUSES, require_mode
from hdfa_rl_suite.google_pure_v7.figure5.panel_a import _condition
from hdfa_rl_suite.google_pure_v7.figure5.panel_a import acquire as acquire_a
from hdfa_rl_suite.google_pure_v7.figure5.panel_b import scaling_trace
from hdfa_rl_suite.google_pure_v7.figure5.protocol import panel_plan
from hdfa_rl_suite.google_pure_v7.figure5.seed_registry import BLACKLIST,SEEDS,validate_registry
from hdfa_rl_suite.google_pure_v7.figure5.source_contract import build_source_contract
from hdfa_rl_suite.google_pure_v7.figure5.storage import discover_shards,shard_id,write_shard
from hdfa_rl_suite.google_pure_v7.figure5.validation import logical_floor,logical_metric,normalized_progress,validate_rows

@pytest.fixture(autouse=True,scope="module")
def _controller():resolve_production_controller()

def test_source_contract_uses_only_required_status_vocabulary():
    contract=build_source_contract();assert {r["status"] for r in contract["fields"]}<=set(SOURCE_STATUSES)
    assert contract["literal_reproduction_possible"] is False

def test_seed_blocks_are_disjoint_and_protected_seeds_unused():
    result=validate_registry();blocks=[set(v) for modes in SEEDS.values() for v in modes.values()]
    assert all(not (a&b) for i,a in enumerate(blocks) for b in blocks[i+1:])
    assert not set().union(*blocks)&BLACKLIST;assert result["certification_seeds_consumed"] is False

def test_exact_d15_p30_mapping_and_no_dense_count_object():
    assert total_controls(15,30)==38670

def test_cycle_accounting_matches_literal_panel_a_budget():
    result=acquisition_accounting(epochs=1000,candidates=50,cycles_per_candidate=36000)
    assert result["candidate_qec_cycles"]==1_800_000_000

def test_paper_scale_requires_explicit_execute():
    with pytest.raises(RuntimeError):require_mode("paper-scale")
    require_mode("paper-scale",execute_paper_scale=True)

def test_logical_metric_orientation_floor_and_endpoints():
    assert logical_metric([2,3],{"trials":10})==.5
    floor=logical_floor(None,{"irreducible_logical_floor":.1})
    assert normalized_progress(.5,floor,.5)==0
    assert normalized_progress(floor,floor,.5)==1
    assert normalized_progress(.3,floor,.5)>normalized_progress(.4,floor,.5)

def test_panel_a_dry_plan_records_all_cost_axes_and_hashes():
    cfg={"mode":"smoke","frequencies":[1/150],"entropy_coefficients":[.0004],"epochs":4,"candidates":4,"cycles_per_candidate":64,"controls":6,"drift_amplitude":.2,"seeds":[54001]}
    plan=panel_plan("5a",cfg);arrays,meta=_condition(plan,plan["conditions"][0])
    assert arrays["candidate_cost"].shape==(4,4);assert arrays["candidate_actions"].shape==(4,4,6)
    assert all(key in meta for key in ("improvement_candidate","improvement_mean","fixed_cost","oracle_cost"))

def test_paper_scale_plan_is_dry_runnable_but_acquisition_is_guarded():
    cfg={"mode":"paper-scale","frequencies":[1/150],"entropy_coefficients":[.0004],"epochs":1000,"candidates":50,"cycles_per_candidate":36000,"controls":6,"drift_amplitude":.2,"seeds":[54011]}
    assert acquire_a(cfg,dry_run=True)["condition_count"]==1
    with pytest.raises(RuntimeError):acquire_a(cfg)

def test_interruption_resume_preserves_finalized_shard_bytes(tmp_path,monkeypatch):
    import hdfa_rl_suite.google_pure_v7.figure5.storage as storage
    monkeypatch.setattr(storage,"figure5_root",lambda:tmp_path)
    cfg={"mode":"smoke","frequencies":[1/150],"entropy_coefficients":[.0004],"epochs":4,"candidates":4,"cycles_per_candidate":64,"controls":6,"drift_amplitude":.2,"seeds":[54021,54022]}
    first=acquire_a(cfg,max_shards=1);npz=next((tmp_path/"shards"/"5a").glob("*.npz"));before=npz.read_bytes()
    final=acquire_a(cfg,resume=True)
    assert len(first["completed_shards"])==1 and len(final["completed_shards"])==2
    assert npz.read_bytes()==before
    checkpoint=next((tmp_path/"manifests").glob("*checkpoint.json"));payload=json.loads(checkpoint.read_text())
    payload["plant_hash"]="changed";checkpoint.write_text(json.dumps(payload),encoding="utf-8")
    with pytest.raises(RuntimeError,match="resume rejected"):acquire_a(cfg,resume=True)

def test_scaling_trace_is_deterministic_monotone_and_sparse_sized():
    left=scaling_trace(15,30,55001,20);right=scaling_trace(15,30,55001,20)
    for key in left:assert np.array_equal(left[key],right[key])
    assert np.all(np.diff(left["lambda_ratio"])>=0);assert all(value.ndim==1 for value in left.values())

def test_scaling_floor_is_independent_not_trace_minimum():
    trace=scaling_trace(7,10,55002,20)
    assert trace["logical_floor"][0]<trace["logical_learned"].min()

def test_shard_identifier_changes_with_protocol_fields():
    base={"panel":"5b","protocol_hash":"a","controller_hash":"c","grid_cell":{},"distance":3,"parameters_per_gate":1,"seed":1,"replicate":0,"chunk":0}
    assert shard_id(base)!=shard_id({**base,"seed":2})

def test_atomic_shards_round_trip_and_corruption_is_rejected(tmp_path,monkeypatch):
    import hdfa_rl_suite.google_pure_v7.figure5.storage as module
    monkeypatch.setattr(module,"panel_shard_dir",lambda panel:tmp_path/panel)
    identity={"panel":"5b","protocol_hash":"a","controller_hash":"c","grid_cell":{},"distance":3,"parameters_per_gate":1,"seed":1,"replicate":0,"chunk":0}
    record=write_shard("5b",identity,{"x":[1.,2.]},{"mode":"smoke"});loaded=discover_shards("5b")
    assert len(loaded)==1 and np.array_equal(loaded[0][1]["x"],[1.,2.])
    (tmp_path/"5b"/record["npz"]).write_bytes(b"corrupt")
    with pytest.raises(RuntimeError):discover_shards("5b")

def test_panel_validators_reject_missing_or_wrong_semantics(tmp_path,monkeypatch):
    import hdfa_rl_suite.google_pure_v7.figure5.validation as validation
    monkeypatch.setattr(validation,"figure5_root",lambda:tmp_path)
    assert not validate_rows("5a",[],mode="smoke")["valid"]
    bad=[{"logical_floor":.5,"logical_initial":.4,"distance":3,"parameters_per_gate":1}]
    assert not validate_rows("5b",bad,mode="smoke")["valid"]

def test_all_27_commands_are_registered():
    text=Path("pyproject.toml").read_text(encoding="utf-8")
    expected=["fig5-source-contract","fig5-freeze-protocols","fig5-seed-registry","fig5-plan-all"]
    expected += [f"fig5{p}-{action}" for p in "abc" for action in ("plan","acquire","merge","validate","plot","report")]
    expected += ["fig5-merge-all","fig5-validate-all","fig5-plot-all","fig5-report-all","fig5-status"]
    assert len(expected)==27;assert all(f"hdfa-google-v7-{name}" in text for name in expected)

def test_figure5_package_does_not_import_v5_or_staged_controller_runtime():
    root=Path("src/hdfa_rl_suite/google_pure_v7/figure5")
    for path in root.glob("*.py"):
        tree=ast.parse(path.read_text(encoding="utf-8"));imports="\n".join(ast.unparse(n) for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom)))
        assert "google_pure_v5" not in imports;assert "stage6" not in imports;assert "hdfa_rl_suite.pipeline" not in imports

def test_plotting_module_contains_no_acquisition_call():
    text=Path("src/hdfa_rl_suite/google_pure_v7/figure5/plotting.py").read_text(encoding="utf-8")
    assert "acquire(" not in text and "scaling_trace" not in text and "seaborn" not in text
