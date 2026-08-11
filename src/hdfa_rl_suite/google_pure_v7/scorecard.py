"""Fail-closed development scorecard and certification lifecycle."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import ACTIVE_CERTIFICATION_SEEDS, ALLOWED_OUTCOMES, RETIRED_SEEDS
from .config import artifact_dir, canonical_hash, guard_seed
from .controller import CONTROLLER_MODE, require_resolved_controller
from .gates import ScientificGate, gate_from_result
from .reporting import read_artifact, write_report


PRIMARY_ARTIFACTS = (
    "v6_immutable_snapshot", "certification_supersession", "scientific_gate_contract",
    "sine_estimator_validation", "long_step_response", "timescale_matched_sine_protocol",
    "timescale_matched_sine", "timescale_matched_strobe", "repaired_drift_production_controller",
    "replay_age_alignment", "natural_drift_regression_ablation", "natural_drift_full_ensemble",
    "hyperparameter_gate_contract", "hyperparameter_study", "exploration_on_valid_slow_drift",
    "recovery_final_controller", "scaling_final_controller",
)

HASH_BOUND_ARTIFACTS = (
    "long_step_response", "timescale_matched_sine_protocol", "timescale_matched_sine",
    "timescale_matched_strobe", "repaired_drift_production_controller", "replay_age_alignment",
    "natural_drift_regression_ablation", "natural_drift_full_ensemble", "hyperparameter_study",
    "exploration_on_valid_slow_drift", "recovery_final_controller", "scaling_final_controller",
)


def validate_scientific_gates() -> dict[str, Any]:
    fake = {"status":"PASS","metric":1.2}
    gate = gate_from_result(fake, required_fields=("status","metric"), mechanism_checks=(True,),
                            performance_checks=(fake["metric"]<1.0,), performance_reasons=("quantitative threshold failed",))
    payload={"schema_version":"google-pure-v7-scientific-gate-validation.v1",
             "fake_artifact":fake,"evaluated_gate":gate.to_dict(),
             "status_pass_cannot_override_performance_failure":not gate.passes,
             "artifact_complete":True,"mechanism_valid":True,"performance_pass":not gate.passes,
             "blocking_reasons":[],"certification_seeds_consumed":False,"status":"PASS" if not gate.passes else "FAIL"}
    return write_report("scientific_gate_validation",payload,"Scientific Gate Validation")


def _artifact_gate(name:str)->ScientificGate:
    try: result=read_artifact(name)
    except RuntimeError: return ScientificGate(False,False,False,("artifact missing",))
    required=("artifact_complete","mechanism_valid","performance_pass","blocking_reasons")
    if not all(field in result for field in required):
        # Integrity artifacts use a structural PASS but still cannot waive later scientific gates.
        if name in {"v6_immutable_snapshot","certification_supersession","scientific_gate_contract","hyperparameter_gate_contract"}:
            passed=result.get("status")=="PASS"
            return ScientificGate(passed,passed,passed,() if passed else ("integrity contract failed",))
        return ScientificGate(False,False,False,("three-layer gate fields missing",))
    reasons=tuple(str(value) for value in result.get("blocking_reasons",[]))
    return ScientificGate(bool(result["artifact_complete"]),bool(result["mechanism_valid"]),bool(result["performance_pass"]),reasons)


def run_development_scorecard()->dict[str,Any]:
    controller=require_resolved_controller(); gates={name:_artifact_gate(name).to_dict() for name in PRIMARY_ARTIFACTS}
    complete=all(item["artifact_complete"] for item in gates.values())
    mechanisms=all(item["mechanism_valid"] for item in gates.values())
    performance=all(item["performance_pass"] for item in gates.values())
    hashes=[]; legacy=False; hash_bindings={}
    for name in PRIMARY_ARTIFACTS:
        try:
            artifact=read_artifact(name)
            if name in HASH_BOUND_ARTIFACTS:
                hash_bindings[name]=artifact.get("resolved_config_hash")
                if artifact.get("resolved_config_hash") is not None: hashes.append(artifact["resolved_config_hash"])
            legacy|=artifact.get("objective_mode")=="legacy_v5_component_clipping_diagnostic_only"
        except RuntimeError: pass
    one_hash=set(hash_bindings)==set(HASH_BOUND_ARTIFACTS) and all(value==controller["resolved_config_hash"] for value in hash_bindings.values())
    integrity=one_hash and not legacy
    ready=complete and mechanisms and performance and integrity
    if not complete: outcome="PARTIAL_PURE_REPRODUCTION"
    elif not gates["timescale_matched_sine"]["mechanism_valid"] or not gates["timescale_matched_sine"]["performance_pass"]: outcome="BANDWIDTH_MISMATCH"
    elif not gates["replay_age_alignment"]["performance_pass"]: outcome="REPLAY_STALENESS"
    elif not gates["hyperparameter_study"]["performance_pass"]: outcome="EXPLORATION_CALIBRATION_FAILURE"
    elif not gates["natural_drift_full_ensemble"]["performance_pass"]: outcome="NATURAL_DRIFT_RETENTION_FAILURE"
    elif not integrity: outcome="OBJECTIVE_TRANSCRIPTION_FAILURE"
    elif not ready: outcome="GENUINE_CONTROLLER_FAILURE"
    else: outcome="PARTIAL_PURE_REPRODUCTION"
    blocking=[f"{name}: {reason}" for name,item in gates.items() if not item["gate_pass"] for reason in (item["blocking_reasons"] or ["gate failed"])]
    if not one_hash: blocking.append("final development artifacts do not share one resolved controller hash")
    if legacy: blocking.append("legacy diagnostic objective appears in final evidence")
    payload={"schema_version":"google-pure-v7-development-scorecard.v1","gates":gates,
             "all_required_artifacts_complete":complete,"all_primary_mechanisms_valid":mechanisms,
             "all_primary_performance_gates_pass":performance,"one_resolved_controller_hash":one_hash,
             "legacy_objective_absent":not legacy,"resolved_config_hash":controller["resolved_config_hash"],
             "certification_ready":ready,"certification_blocked":not ready,"blocking_reasons":blocking,
             "outcome_class":outcome,"outcome_allowed":outcome in ALLOWED_OUTCOMES,
             "certification_seeds_consumed":False,"status":"PASS" if ready else "BLOCKED"}
    return write_report("development_scorecard",payload,"v7 Development Scorecard")


def freeze_certification()->dict[str,Any]:
    score=read_artifact("development_scorecard")
    if not score.get("certification_ready"):
        payload={"schema_version":"google-pure-v7-certification-freeze-blocked.v1","scorecard_hash":canonical_hash(score),
                 "certification_ready":False,"blocking_reasons":score.get("blocking_reasons",[]),
                 "active_certification_seeds":list(ACTIVE_CERTIFICATION_SEEDS),"retired_seeds":list(RETIRED_SEEDS),
                 "certification_seeds_consumed":False,"status":"NOT_FROZEN_SCIENTIFIC_GATES_FAILED"}
        return write_report("certification_freeze_blocked",payload,"Certification Freeze Blocked")
    controller=require_resolved_controller()
    evidence={name:canonical_hash(read_artifact(name)) for name in PRIMARY_ARTIFACTS}
    protocol=read_artifact("timescale_matched_sine_protocol")
    payload={"schema_version":"google-pure-v7-certification-preregistration.v1","status":"FROZEN_READY_UNOPENED",
             "source_and_evidence_hashes":evidence,"resolved_controller":controller,
             "response_tau_epochs":protocol["response_tau_epochs"],"sine_protocol_hash":protocol["protocol_hash"],
             "strobe_protocol_hash":canonical_hash(read_artifact("timescale_matched_strobe")),
             "natural_ensemble_hash":canonical_hash(read_artifact("natural_drift_full_ensemble")),
             "metrics_uncertainty_thresholds_frozen":True,"active_certification_seeds":list(ACTIVE_CERTIFICATION_SEEDS),
             "retired_seeds":list(RETIRED_SEEDS),"one_run_permitted":True,"allowed_outcomes":list(ALLOWED_OUTCOMES),
             "certification_seeds_consumed":False}
    return write_report("certification_preregistration",payload,"v7 Certification Preregistration")


def run_certification(*,seed:int,confirm:bool=False,authorization_phrase:str|None=None)->dict[str,Any]:
    guard_seed(seed,certification=True)
    prereg=read_artifact("certification_preregistration")
    if prereg.get("status")!="FROZEN_READY_UNOPENED" or not prereg.get("one_run_permitted"):
        raise RuntimeError("v7 certification is not frozen and ready")
    if not confirm or authorization_phrase!="RUN-V7-HELD-OUT-ONCE":
        raise RuntimeError("certification requires explicit confirmation and exact one-run authorization phrase")
    raise RuntimeError("certification acquisition is intentionally not implemented as an implicit aggregate; run the frozen component protocol explicitly")
