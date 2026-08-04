from __future__ import annotations
from enum import StrEnum
from typing import Any,Iterable,Mapping
from .common import write

class ExperimentFamily(StrEnum):
    PUBLIC_ENDPOINT_DATA_REPRODUCTION="PUBLIC_ENDPOINT_DATA_REPRODUCTION"
    FIGURE5A_REAL_TIME_STEERING="FIGURE5A_REAL_TIME_STEERING"
    FIGURE5B_SPARSE_SCALING="FIGURE5B_SPARSE_SCALING"
    FIGURE5C_CONVERGENCE_LAW="FIGURE5C_CONVERGENCE_LAW"
    NATURAL_DRIFT_SPECTRAL_SUPPRESSION="NATURAL_DRIFT_SPECTRAL_SUPPRESSION"
    RANDOMIZED_RECOVERY_AFTER_SPOIL="RANDOMIZED_RECOVERY_AFTER_SPOIL"
    STEP_RESPONSE_INJECTED_DRIFT="STEP_RESPONSE_INJECTED_DRIFT"
    PUBLIC_TABLE_REPRODUCTION="PUBLIC_TABLE_REPRODUCTION"

RUN_FAMILIES={
 ExperimentFamily.PUBLIC_ENDPOINT_DATA_REPRODUCTION.value:"released_static_memory_endpoints",
 ExperimentFamily.FIGURE5A_REAL_TIME_STEERING.value:"synthetic_sinusoidal_candidate_stream",
 ExperimentFamily.FIGURE5B_SPARSE_SCALING.value:"synthetic_sparse_scaling",
 ExperimentFamily.FIGURE5C_CONVERGENCE_LAW.value:"synthetic_local_convergence",
 ExperimentFamily.NATURAL_DRIFT_SPECTRAL_SUPPRESSION.value:"synthetic_frozen_natural_ensemble",
 ExperimentFamily.RANDOMIZED_RECOVERY_AFTER_SPOIL.value:"synthetic_policy_spoil",
 ExperimentFamily.STEP_RESPONSE_INJECTED_DRIFT.value:"synthetic_persistent_optimum_step",
 ExperimentFamily.PUBLIC_TABLE_REPRODUCTION.value:"released_static_memory_tables"}

FORBIDDEN=frozenset(frozenset((a.value,b.value)) for i,a in enumerate(ExperimentFamily) for b in tuple(ExperimentFamily)[i+1:])

def require_single_primary(artifact:Mapping[str,Any])->str:
    value=artifact.get("experiment_family")
    if isinstance(value,(list,tuple,set)) or value not in {x.value for x in ExperimentFamily}:raise RuntimeError("artifact requires exactly one primary experiment family")
    return str(value)

def forbid_joint_score(families:Iterable[str],*,paper_explicitly_simultaneous:bool=False)->None:
    values=set(families)
    unknown=values-{x.value for x in ExperimentFamily}
    if unknown:raise RuntimeError(f"unknown experiment families: {sorted(unknown)}")
    if len(values)>1 and not paper_explicitly_simultaneous:raise RuntimeError(f"separate experiment families cannot be jointly scored: {sorted(values)}")

def require_control_only(records:Iterable[Mapping[str,Any]])->None:
    modes={str(row.get("decoder_assistance","CONTROL_ONLY")) for row in records}
    if "DECODER_ASSISTED" in modes and "CONTROL_ONLY" in modes:raise RuntimeError("decoder-assisted and control-only claims cannot be mixed")

def family_metadata(family:ExperimentFamily)->dict[str,Any]:
    return {"experiment_family":family.value,"run_family":RUN_FAMILIES[family.value],"same_run_claims":[],
      "forbidden_cross_run_claims":[x.value for x in ExperimentFamily if x is not family],"decoder_assistance":"CONTROL_ONLY"}

def build_contract()->dict[str,Any]:
    rows=[]
    for family in ExperimentFamily:
        rows.append({"experiment_family":family.value,"run_family":RUN_FAMILIES[family.value],"same_run_claims":[],
          "forbidden_cross_run_claims":[x.value for x in ExperimentFamily if x is not family],
          "required_identity_fields":["controller_hash","protocol_hash","plant_hash","graph_hash","seed_registry_hash","observable_definition","evaluation_budget"]})
    return write("experiment_family_contract",{"schema_version":"google-pure-evidence-v8-family-contract.v1","rows":rows,
      "one_primary_family_per_artifact":True,"separate_paper_runs_never_form_simultaneous_scorecard":True},"Experiment Family Contract")
