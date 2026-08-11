from __future__ import annotations
import argparse,json
from .claim_registry import build_claim_registry
from .evidence_contracts import build_gate_contract
from .experiment_families import build_contract
from .figure5b import run_figure5b
from .figure5c import run_figure5c
from .hdfa_readiness import report_hdfa_readiness
from .manifest_validation import build_protocol_preflight,validate_manifests
from .natural_drift import run_natural_drift
from .paper_comparison import build_paper_comparison
from .recovery import run_recovery
from .step_response import run_step_response
from .common import root

def _args(name):
 p=argparse.ArgumentParser(prog=name); p.add_argument("--mode",choices=("smoke","reference"),default="smoke"); p.add_argument("--execute",action="store_true"); return p.parse_args()
def _show(x): print(json.dumps({k:x[k] for k in ("schema_version","mode","classification","outcome","status","artifact_hash") if k in x},indent=2))
def build_contracts_main(): build_contract();build_gate_contract();_show(build_protocol_preflight())
def validate_manifests_main(): _show(validate_manifests())
def _run(fn,name): a=_args(name); print(json.dumps({"mode":a.mode,"reference_execution_authorized":a.execute,"notice":"reference mode may be expensive"})); _show(fn(mode=a.mode,execute=a.execute))
def natural_main(): _run(run_natural_drift,"hdfa-google-evidence-v8-run-natural-drift")
def step_main(): _run(run_step_response,"hdfa-google-evidence-v8-run-step-response")
def recovery_main(): _run(run_recovery,"hdfa-google-evidence-v8-run-recovery")
def figure5b_main(): _run(run_figure5b,"hdfa-google-evidence-v8-run-figure5b")
def figure5c_main(): _run(run_figure5c,"hdfa-google-evidence-v8-run-figure5c")
def claims_main(): _show(build_claim_registry())
def comparison_main(): _show(build_paper_comparison())
def readiness_main(): _show(report_hdfa_readiness())
def status_main():
 preflight=build_protocol_preflight();files=sorted(str(p.relative_to(root())) for p in root().rglob("*.json")); print(json.dumps({"artifact_root":str(root()),"json_artifacts":files,"superseded_legacy_artifacts":["claim_registry.json","hdfa_readiness.json","step_response/summary.json","recovery/summary.json"],"protocol_gate_pass":preflight["protocol_gate_pass"],"reference_acquisition_permitted":preflight["reference_acquisition_permitted"],"missing_files":preflight["missing_files"]},indent=2))
