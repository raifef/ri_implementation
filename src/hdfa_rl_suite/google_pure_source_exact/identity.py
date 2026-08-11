"""Frozen identity for the amended paper controller path."""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from .figure5a.acquisition import COORDINATE_CONTRACT, FIGURE5A_IMPLEMENTATION_VERSION

PAPER_DIRECT_SIGMA = "PAPER_DIRECT_SIGMA"
DIRECT_SIGMA = "direct_sigma"

def _file_hash(path: Path) -> str:
    digest=sha256(); digest.update(path.read_bytes()); return digest.hexdigest()

def _canonical_hash(value: Any) -> str:
    return sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def build_direct_sigma_identity(root: Path) -> dict[str, Any]:
    config_path=root/"configs/google_pure_source_exact/figure5a.json"
    config=json.loads(config_path.read_text(encoding="utf-8"))
    code_paths=[
        root/"src/hdfa_rl_suite/google_pure_source_exact/policy_parameterization/gaussian.py",
        root/"src/hdfa_rl_suite/google_pure_source_exact/policy_parameterization/losses.py",
        root/"src/hdfa_rl_suite/google_pure_source_exact/policy_parameterization/optimizer.py",
        root/"src/hdfa_rl_suite/google_pure_source_exact/source_normalization.py",
        root/"src/hdfa_rl_suite/google_pure_source_exact/figure5a/acquisition.py",
        root/"src/hdfa_rl_suite/google_pure_source_exact/figure5a/plant.py",
        root/"src/hdfa_rl_suite/google_pure_source_exact/paper_families/common.py",
        root/"src/hdfa_rl_suite/google_pure_source_exact/paper_families/scaling.py",
        root/"src/hdfa_rl_suite/google_pure_source_exact/paper_families/natural.py",
        root/"src/hdfa_rl_suite/google_pure_source_exact/paper_families/recovery.py",
        root/"src/hdfa_rl_suite/google_pure_source_exact/paper_families/step.py",
        root/"src/hdfa_rl_suite/google_pure_source_exact/step_response_130/acquisition.py",
        root/"src/hdfa_rl_suite/google_pure_source_exact/step_response_130/estimator.py",
        root/"src/hdfa_rl_suite/google_pure_source_exact/step_response_130/plant.py",
        root/"src/hdfa_rl_suite/google_pure_source_exact/natural_drift_dft/estimator.py",
    ]
    code_hashes={path.relative_to(root).as_posix():_file_hash(path) for path in code_paths}
    dependencies={name:_file_hash(root/relative) for name,relative in config["dependencies"].items()}
    controller_contract={
        "controller_mode":PAPER_DIRECT_SIGMA,
        "parameterization":DIRECT_SIGMA,
        "source_parameterization":"DIRECT_SIGMA_SOURCE_EXACT",
        "ratio_clipping":"SOURCE_ELEMENTWISE_COORDINATE_CLIPPING",
        "baseline":"JOINT_LEARNED_DETECTOR_BASELINE",
        "figure5a_coordinate_contract":COORDINATE_CONTRACT,
        "figure5a_action_execution":"identity_applied_gaussian",
        "figure5a_empirical_relative_normalization":"NONCANONICAL_ABLATION_ONLY",
        "non_figure5a_normalization_boundary":"google-pure-v15-production-boundary.v1",
        "non_figure5a_boundary_transform":"u = u0 + s*x",
        "optimized_scale_variable":"sigma",
        "controller_config":config["controller"],
        "dependency_hashes":dependencies,
    }
    return {
        **controller_contract,
        "controller_hash":_canonical_hash(controller_contract),
        "controller_code_hash":_canonical_hash(code_hashes),
        "controller_code_files":code_hashes,
        "figure5a_normalization_loaded":False,
        "non_figure5a_normalization_loaded":True,
        "figure5a_implementation_version":FIGURE5A_IMPLEMENTATION_VERSION,
        "implementation_version":"family-conditional-source-execution.v1",
    }

def require_direct_sigma_identity(identity: dict[str, Any], expected: dict[str, Any]|None=None) -> None:
    reference=expected or identity
    requirements={"controller_mode":PAPER_DIRECT_SIGMA,"parameterization":DIRECT_SIGMA,
        "source_parameterization":"DIRECT_SIGMA_SOURCE_EXACT","ratio_clipping":"SOURCE_ELEMENTWISE_COORDINATE_CLIPPING",
        "baseline":"JOINT_LEARNED_DETECTOR_BASELINE","optimized_scale_variable":"sigma"}
    mismatches=[key for key,value in requirements.items() if identity.get(key)!=value]
    mismatches += [key for key in ("controller_hash","controller_code_hash") if not identity.get(key) or identity.get(key)!=reference.get(key)]
    if mismatches: raise RuntimeError("direct-sigma identity mismatch: "+", ".join(sorted(set(mismatches))))
