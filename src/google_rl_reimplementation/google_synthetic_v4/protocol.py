"""Evidence, split, metric, and certification protocol gates."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import DISCLAIMER
from .config import CERTIFICATION_SEEDS, artifact_dir, canonical_hash, config_dir, load_controller_choices, load_ensemble, load_splits, repository_root, sha256_file
from .plant import ensemble_contract, frozen_specs
from .reporting import read_artifact, write_report
from .validation import validate_ppo_reference


def write_evidence_boundary() -> dict[str, Any]:
    payload = {
        "schema_version":"google-synthetic-v4-evidence-boundary.v1",
        "layers":{
            "A_zenodo_constrained_observation":{
                "supports":["detector-rate marginals","detector covariance","overdispersion","effective sample size","endpoint logical estimators","decoder definitions","supported code and experiment metadata"],
                "does_not_support":["counterfactual actions","policy trajectories","control sensitivities","drift dynamics","training data"],
                "sources":["artifacts/google_reproduction_v3/zenodo_inventory.json","artifacts/google_reproduction_v3/empirical_statistics_fit.json","artifacts/google_reproduction_v3/public_data_reproduction.json"]},
            "B_paper_specified_synthetic_control":{
                "supports":["sparse detector-control factors","locally quadratic miscalibration response","multiple controls per gate","declared drift families","steps","sinusoidal steering","graph/control scaling","candidate and epoch budgets","mean and stochastic policy comparisons"],
                "claim":"synthetic task definition, not empirical Google dynamics"},
            "C_open_implementation_choices":{
                "includes":["optimizer","learning rates","replay age","baseline timescale","entropy schedule","scale clamps","PPO update count","drift amplitudes","plant coefficients"],
                "control":"preregistered priors, physically disjoint splits, one-variable amendments, Pareto selection"}},
        "permissible_claim":"An open masked-PPO implementation reproduces the public qualitative and quantitative controller behavior on a paper-specified synthetic control model whose observation statistics are constrained by the official Zenodo endpoint data.",
        "prohibited_claims":["Willow hardware reproduction","empirical reconstruction of hidden Google control dynamics","decoder improvement credited to the control agent"],
        "certification_seeds_consumed":False,
        "summary":{"status":"FROZEN","evidence_layers":3,"hardware_reproduction":False},
    }
    write_report("evidence_boundary",payload,"V4 evidence boundary")
    return payload


def _immutability_snapshot() -> dict[str, Any]:
    root = repository_root()
    paths: list[Path] = []
    for relative in ("src/google_rl_reimplementation/google_reproduction","src/google_rl_reimplementation/google_reproduction_v3",
                     "configs/google_rl","configs/google_rl_v3","artifacts/google_reproduction_v2",
                     "artifacts/google_reproduction_v3"):
        folder = root / relative
        if folder.exists():
            paths.extend(p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    for relative in ("tests/test_google_reproduction_v2.py","tests/test_google_reproduction_v3.py"):
        path = root / relative
        if path.exists():
            paths.append(path)
    files = {str(path.relative_to(root)).replace("\\","/"):sha256_file(path) for path in sorted(set(paths))}
    return {"schema_version":"google-synthetic-v4-prior-evidence-snapshot.v1","files":files,
            "file_count":len(files),"snapshot_hash":canonical_hash(files)}


def build_plant_ensemble() -> dict[str, Any]:
    write_evidence_boundary()
    payload = ensemble_contract()
    payload["prior_evidence_immutability_snapshot"] = _immutability_snapshot()
    write_report("plant_ensemble_contract",payload,"Frozen synthetic plant ensemble")
    return payload


def freeze_synthetic_splits() -> dict[str, Any]:
    splits = load_splits()
    ensemble = load_ensemble()
    groups = [set(splits[name]["plant_ids"]) for name in ("plant_construction","controller_development","development_validation","certification")]
    if any(groups[i] & groups[j] for i in range(len(groups)) for j in range(i+1,len(groups))):
        raise ValueError("plant splits overlap")
    seed_groups = [set(splits[name]["evaluation_seeds"]) for name in ("plant_construction","controller_development","development_validation","certification")]
    if any(seed_groups[i] & seed_groups[j] for i in range(len(seed_groups)) for j in range(i+1,len(seed_groups))):
        raise ValueError("evaluation seed splits overlap")
    if tuple(splits["certification"]["evaluation_seeds"]) != CERTIFICATION_SEEDS:
        raise ValueError("certification seeds are not the locked 8101-8112 sequence")
    rows = {s.plant_id:s for s in frozen_specs()}
    signatures = {}
    fields = tuple(splits["physical_disjointness_fields"])
    for plant_id in splits["certification"]["plant_ids"]:
        spec = rows[plant_id]
        aliases = {"graph_realization":"graph_offset","local_curvature":"curvature_mean","detector_covariance":"detector_covariance"}
        signatures[plant_id] = {field:getattr(spec,aliases.get(field,field)) for field in fields}
    payload = {
        "schema_version":"google-synthetic-v4-split-manifest.v1","frozen":True,
        "splits":splits,"configuration_hashes":{"synthetic_splits":sha256_file(config_dir()/"synthetic_splits.yaml"),
        "plant_ensemble":sha256_file(config_dir()/"plant_ensemble.yaml")},
        "certification_physical_signatures":signatures,
        "physical_parameters_distinct":len({canonical_hash(v) for v in signatures.values()})==len(signatures),
        "certification_seeds_consumed":False,
        "summary":{"status":"FROZEN","certification_status":"LOCKED_UNCONSUMED","split_count":4},
    }
    write_report("synthetic_split_manifest",payload,"Synthetic split manifest")
    return payload


def validate_ppo() -> dict[str, Any]:
    payload = validate_ppo_reference()
    write_report("ppo_reference_validation",payload,"Masked PPO independent reference validation")
    if payload["status"] != "PASS":
        raise RuntimeError("PPO numerical validation failed; downstream study is blocked")
    return payload


def stability_metric_contract() -> dict[str, Any]:
    ppo = read_artifact("ppo_reference_validation")
    if ppo["status"] != "PASS":
        raise RuntimeError("metric validation is downstream of a passing PPO gate")
    payload = {
        "schema_version":"google-synthetic-v4-stability-metric-contract.v1","frozen_before_amendments":True,
        "public_anchor_reconstructibility":"NOT_EXACTLY_RECONSTRUCTIBLE: released endpoints do not disclose the policy-evaluation interleave or hidden control traces",
        "mean_policy_control_stability":{"policy":"deterministic learned Gaussian mean","ratio":"sample standard deviation of fixed-policy logical risk / sample standard deviation of learned-mean logical risk","normalization":"dimensionless logical risk","detrending":"none; PSD is mean-centered","window":"exclude first 25% as declared transient","measurement_noise_correction":"reported separately; primary ratio uses common-tape latent synthetic risk","candidate_intervals_included":False},
        "operational_stochastic_policy_stability":{"policy":"mean logical risk over all sampled candidates in each epoch","ratio":"sample standard deviation of fixed-policy logical risk / sample standard deviation of stochastic aggregate logical risk","normalization":"same as mean metric","detrending":"none; PSD is mean-centered","window":"same frozen window","measurement_noise_correction":"reported separately","candidate_intervals_included":True},
        "common_disturbance_contract":"fixed, learned-mean, stochastic, and oracle evaluations use the same epoch optimum and plant realization",
        "transient_included":False,"ratio_kind":"standard deviation, not variance",
        "synthetic_analogue":{"mean_policy_target_interval":[1.8,3.1],"operational_target_interval":[1.45,2.8],"low_frequency_suppression_db":[2.0,6.5],"rationale":"frozen broad ranges around public approximate behavior; not an exact endpoint reconstruction"},
        "metric_separation_test":"mean and stochastic arrays and ratios have separate keys and cannot alias",
        "certification_seeds_consumed":False,
        "summary":{"status":"VALIDATED","public_anchor_exact":False,"metric_count":2},
    }
    write_report("stability_metric_contract",payload,"Stability metric contract")
    return payload


def freeze_certification() -> dict[str, Any]:
    score = read_artifact("development_scorecard")
    allowed = {"PASS_SYNTHETIC_ANALOGUE","PASS_STRONGER_WITHOUT_OTHER_REGRESSION"}
    all_gates = score.get("overall_status") in allowed and bool(score.get("all_development_gates_pass",False))
    if not all_gates:
        payload = {"schema_version":"google-synthetic-v4-certification-preregistration.v1","status":"NOT_FROZEN_GATES_FAILED",
                   "development_status":score.get("overall_status"),"certification_seeds":list(CERTIFICATION_SEEDS),
                   "certification_seeds_consumed":False,"one_run_permitted":False,
                   "summary":{"status":"NOT_FROZEN_GATES_FAILED","ready":False}}
        write_report("certification_preregistration",payload,"Synthetic certification preregistration")
        return payload
    root = repository_root()
    protected = list((root/"src/google_rl_reimplementation/google_synthetic_v4").glob("*.py")) + list(config_dir().glob("*.yaml"))
    protected += [artifact_dir()/f"{name}.json" for name in (
        "evidence_boundary","plant_ensemble_contract","synthetic_split_manifest","ppo_reference_validation",
        "drift_stability_decomposition","stability_metric_contract","amendment_log","randomized_recovery",
        "steering_phase","convergence_scaling","development_scorecard")]
    hashes = {str(path.relative_to(root)).replace("\\","/"):sha256_file(path) for path in protected if path.exists()}
    payload = {
        "schema_version":"google-synthetic-v4-certification-preregistration.v1","status":"FROZEN_READY_UNOPENED",
        "controller":"retained amendment from amendment_log.json","plant_ensemble_hash":read_artifact("plant_ensemble_contract")["ensemble_hash"],
        "metrics_and_tolerances":read_artifact("stability_metric_contract")["synthetic_analogue"],
        "certification_plants":load_splits()["certification"]["plant_ids"],"certification_seeds":list(CERTIFICATION_SEEDS),
        "protected_file_hashes":hashes,"protected_manifest_hash":canonical_hash(hashes),
        "one_run_permitted":True,"post_opening_amendments_prohibited":True,"certification_seeds_consumed":False,
        "allowed_outcomes":["SYNTHETIC_GOOGLE_STYLE_REPRODUCTION_CERTIFIED","PARTIAL_SYNTHETIC_REPRODUCTION","SYNTHETIC_REPRODUCTION_FAILED_CONTROLLER","SYNTHETIC_REPRODUCTION_FAILED_PLANT_ENSEMBLE"],
        "blocked_until_certified":["reduced-budget equivalence"],
        "summary":{"status":"FROZEN_READY_UNOPENED","ready":True,"seeds_opened":False},
    }
    write_report("certification_preregistration",payload,"Synthetic certification preregistration")
    return payload


def estimate_cost(name: str, *, epochs: int, plants: int, cells: int = 1, certification: bool = False) -> dict[str, Any]:
    cfg = load_controller_choices()["sampling"]
    candidates = epochs * plants * cells * int(cfg["paper_candidates_per_epoch"])
    cycles = candidates * int(cfg["paper_effective_cycles_per_candidate"])
    return {"command":name,"estimated_runtime":"host-dependent; short mode is usually under minutes",
            "candidate_count":candidates,"native_qec_cycle_cost":cycles,"estimated_peak_memory":"< 1 GiB short mode; scaling uses sparse batches",
            "estimated_disk":"< 50 MiB reports","certification_seeds_touched":certification}
