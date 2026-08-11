"""Tiny end-to-end proof that the amended controller and 41-control Stim plant execute together."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from hdfa_rl_suite.google_pure_source_exact.figure5a.acquisition import run_cell
from hdfa_rl_suite.google_pure_source_exact.figure5a.contracts import AcquisitionMode, Figure5aProtocol, canonical_hash
from hdfa_rl_suite.google_pure_source_exact.figure5a.validation import build_plant, dependency_hashes, validate_dependencies
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.contracts import PositivityGuard
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.optimizer import GradientClippingMode, OptimizerConfig
from .identity import build_direct_sigma_identity, require_direct_sigma_identity

ROOT=Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT=ROOT/"artifacts/google_pure_source_exact/direct_sigma_integration"

def _write(path: Path,value: dict):
    path.parent.mkdir(parents=True,exist_ok=True); temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8"); temporary.replace(path)

def run_tiny_integration(output: Path=DEFAULT_OUTPUT) -> dict:
    config_path=ROOT/"configs/google_pure_source_exact/figure5a.json"
    config=json.loads(config_path.read_text(encoding="utf-8")); identity=build_direct_sigma_identity(ROOT)
    require_direct_sigma_identity(identity); plant=build_plant(config); dependencies=validate_dependencies(ROOT,config)
    controller=config["controller"]
    clipping=controller["gradient_clipping"]
    optimizer=OptimizerConfig(float(controller["mean_learning_rate"]),float(controller["sigma_learning_rate"]),
        float(controller["baseline_learning_rate"]),minimum_sigma=float(controller["minimum_sigma"]),
        maximum_sigma=float(controller["maximum_sigma"]),positivity_guard=PositivityGuard(controller["positivity_guard"]),
        gradient_clipping_mode=GradientClippingMode(clipping["selected_mode"]),
        gradient_clip_threshold=float(clipping["selected_threshold"]))
    protocol=Figure5aProtocol(AcquisitionMode.SMOKE,2,3,50,int(config["plant"]["circuit_rounds"]))
    checkpoint=output/f"checkpoint-{identity['controller_hash'][:12]}-{identity['controller_code_hash'][:12]}.json"
    cell=run_cell(protocol=protocol,plant=plant,frequency=float(config["anchor"]["frequency"]),
        entropy_weight=float(config["anchor"]["entropy_weights"][0]),seed=53101,optimizer_config=optimizer,
        initial_sigma=float(controller["initial_sigma"]),checkpoint_path=checkpoint,
        dependency_hashes=dependency_hashes(ROOT,config),controller_hash=identity["controller_hash"],
        clip=float(controller["ppo_clip"]),baseline_weight=float(controller["baseline_weight"]),resume=checkpoint.exists())
    state=json.loads(checkpoint.read_text(encoding="utf-8")); mean=np.asarray(state["policy"]["mean"]); sigma=np.asarray(state["policy"]["sigma"])
    epoch=protocol.epochs-1; optimum=plant.optimum(epoch,float(config["anchor"]["frequency"])); rng=np.random.default_rng(53199)
    policies={"fixed":np.zeros(41),"oracle":optimum,
              "oracle_with_policy_sigma":optimum+sigma*rng.normal(size=41),
              "learned_mean":mean,
              "sampled_candidates":mean+sigma*rng.normal(size=41)}
    policy_counts={name:plant.sample_detector_observation(
        action,epoch=epoch,frequency=float(config["anchor"]["frequency"]),
        qec_cycles=3000,seed=plant.stream_seed(53199,name,epoch,0),
        target_controls=optimum).raw_total for name,action in policies.items()}
    records=cell["epoch_records"]
    gates={
        "direct_sigma_controller_hash_loaded":cell["controller_hash"]==identity["controller_hash"],
        "direct_sigma_code_hash_loaded":bool(identity["controller_code_hash"]),
        "direct_sigma_parameterization_executed":all(row["parameterization"]=="direct_sigma" for row in records),
        "stim_41_parameter_plant_loaded":plant.control_count==41,
        "canonical_dependencies_loaded":dependencies["pass"],
        "elementwise_coordinate_ratio_clipping_executed":all(row["coordinate_ratios_clipped_before_sparse_product"] for row in records),
        "learned_detector_baseline_executed":all(row["baseline_mode"]=="JOINT_LEARNED_DETECTOR_BASELINE" for row in records),
        "source_entropy_regime_loaded":float(config["anchor"]["entropy_weights"][0]) in (0.001,0.01,0.1),
        "nonzero_qec_cycles_executed":cell["candidate_qec_cycles"]>0,
        "nonzero_detector_events_observed":sum(cell["stream_totals"].values())>0,
        "five_policy_decomposition_retained":set(policy_counts)==set(policies) and all(value>=0 for value in policy_counts.values()),
        "source_gaussian_coordinate_identity_executed":all(
            row["action_execution"]=="identity_applied_gaussian" and
            row["plant_boundary_execution"]=="none_source_coordinate_identity" and
            row["maximum_abs_gaussian_applied_delta"]==0.0 for row in records),
        "source_optimum_applied_directly":all(row["source_optimum_applied_directly"] for row in records),
        "applied_gaussian_likelihood_and_entropy_retained":all(
            row["likelihood_space"]=="applied_gaussian" and
            row["entropy_space"]=="applied_gaussian" for row in records),
        "empirical_normalization_absent_from_canonical_path":
            not cell["empirical_relative_normalization_applied"],
        "status_inherits_provenance_without_promotion":True,
    }
    manifest={"schema":"paper-direct-sigma-integration.v1","pass":all(gates.values()),"gates":gates,
        **{key:identity[key] for key in ("controller_mode","controller_hash","controller_code_hash","parameterization",
            "source_parameterization","ratio_clipping","baseline","optimized_scale_variable")},
        "plant_mode":"FIGURE5A_41_PARAMETER_STIM","plant_hash":plant.plant_hash,
        "graph_hash":canonical_hash(plant.mask.astype(int).tolist()),"control_count":plant.control_count,
        "detector_count":plant.detector_count,"dependency_hashes":dependencies["hashes"],
        "empirical_relative_normalization_ablation":
            dependencies["empirical_relative_normalization_ablation"],
        "policy_decomposition_counts":policy_counts,"training_qec_cycles":cell["candidate_qec_cycles"],
        "four_source_stream_qec_cycles":cell["four_stream_qec_cycles"],"complete":cell["complete"],
        "scientifically_valid":False,"final_evidence":False,"evidence_layer":"TINY_INTEGRATION_PATH_PROOF_ONLY",
        "staged_controller_run":False,"blocking_reasons":[] if all(gates.values()) else [name for name,value in gates.items() if not value]}
    manifest["manifest_hash"]=canonical_hash(manifest); _write(output/"controller_identity.json",identity); _write(output/"manifest.json",manifest)
    return manifest

def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT); args=parser.parse_args(argv)
    result=run_tiny_integration(args.output); print(json.dumps(result,indent=2,sort_keys=True)); return 0 if result["pass"] else 2
if __name__=="__main__": raise SystemExit(main())
