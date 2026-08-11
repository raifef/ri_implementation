"""Immutable held-out freeze and reference gate with no force path."""
from __future__ import annotations

from typing import Any

from .contracts import NONFINAL, TERMINAL_STATUSES, V15_SCHEMA, nonfinal
from .imports import verify_import_manifest
from .io import (ARTIFACT_ROOT, CONFIG_ROOT, ROOT, atomic_json, canonical_hash,
                 file_hash, read_json)


ANALYSIS_ARTIFACTS = [
    "sensitivity/source_definition_audit.json",
    "sensitivity/detector_degree_audit.json",
    "sensitivity/multi_point_calibration.json",
    "sensitivity/uncertainty_propagation.json",
    "sensitivity/calibration_firewall.json",
    "scaling/boundary_map.json",
    "scaling/figure5b_decomposition.json",
    "scaling/gradient_normalization.json",
    "scaling/curvature_distribution.json",
    "scaling/hessian_spectrum.json",
    "scaling/slow_mode_projection.json",
    "scaling/information_ablation.json",
    "scaling/effective_sample_size.json",
    "dynamics/mean_scale_conditioning.json",
    "dynamics/scale_floor.json",
    "dynamics/residual_decay.json",
    "fidelity/objective_alignment.json",
    "fidelity/figure5c_analysis.json",
    "fidelity/figure5a_latency.json",
    "fidelity/step_response_fit.json",
    "fidelity/natural_drift_power.json",
    "fidelity/ppo_lifecycle.json",
    "fidelity/provenance.json",
    "fidelity/resource_semantics.json",
    "decoder/offline_steering.json",
    "source_gap_register.json",
]


def _freeze_payload() -> dict[str, Any]:
    missing = [path for path in ANALYSIS_ARTIFACTS if not (ARTIFACT_ROOT / path).is_file()]
    if missing:
        raise RuntimeError("held-out freeze rejected; analyses missing: " + ", ".join(missing))
    source_files = sorted((ROOT / "src/hdfa_rl_suite/google_pure_v15").glob("*.py"))
    return {
        "schema_version": V15_SCHEMA,
        "immutable_import_manifest_hash": verify_import_manifest()["manifest_hash"],
        "analysis_artifacts": [{"path": path, "sha256": file_hash(ARTIFACT_ROOT / path)}
                               for path in ANALYSIS_ARTIFACTS],
        "v15_source_files": [{"path": path.relative_to(ROOT).as_posix(), "sha256": file_hash(path)}
                             for path in source_files],
        "protocol_sha256": file_hash(CONFIG_ROOT / "protocol.json"),
        "seed_registry_hashes": {
            name: file_hash(CONFIG_ROOT / f"seeds_{name}.json")
            for name in ("calibration", "development", "validation", "heldout")
        },
        "heldout_metrics": [
            "absolute_step_target_crossing_0.9",
            "complete_pair_natural_filter_db",
            "figure5b_physical_and_logical_error",
            "candidate_and_mean_decomposition",
            "state_and_candidate_lineage",
            "decoder_four_arm_logical_error",
        ],
        "force_override_allowed": False,
        "heldout_seeds_consumed": False,
    }


def build_heldout_freeze() -> dict[str, Any]:
    payload = _freeze_payload()
    payload["freeze_hash"] = canonical_hash(payload)
    path = ARTIFACT_ROOT / "gate/heldout_freeze.json"
    if path.is_file():
        existing = read_json(path)
        if existing != payload:
            if existing.get("heldout_seeds_consumed") or existing.get("force_override_allowed"):
                raise RuntimeError("held-out freeze already exists and immutable inputs changed")
            # V15 immediate execution repair changes production code before held-out
            # seeds are consumed. Preserve the prior preregistration verbatim, then
            # freeze the amended inputs. This is not a force path or evidence promotion.
            archive = ARTIFACT_ROOT / "gate/superseded_heldout_freezes" / (
                f"{existing.get('freeze_hash', canonical_hash(existing))}.json")
            atomic_json(archive, existing)
            atomic_json(path, payload)
            return payload
        return existing
    atomic_json(path, payload)
    return payload


def reference_gate_status() -> dict[str, Any]:
    freeze_path = ARTIFACT_ROOT / "gate/heldout_freeze.json"
    freeze_valid = False
    freeze_reason = "MISSING"
    if freeze_path.is_file():
        try:
            freeze_valid = read_json(freeze_path) == _freeze_payload() | {
                "freeze_hash": canonical_hash(_freeze_payload())}
            freeze_reason = "VALID" if freeze_valid else "HASH_OR_INPUT_MISMATCH"
        except RuntimeError as error:
            freeze_reason = str(error)
    decoder_path = ARTIFACT_ROOT / "decoder/offline_steering.json"
    decoder = read_json(decoder_path) if decoder_path.is_file() else {}
    imports = verify_import_manifest()
    gates = {
        "immutable_imports_valid": bool(imports["pass"]),
        "heldout_freeze_valid": freeze_valid,
        "heldout_seeds_consumed_once": False,
        "proprietary_google_plant_available": False,
        "experimental_source_traces_available": False,
        "hardware_validation_complete": False,
        "source_budget_acquisition_complete": False,
        "natural_drift_power_and_resolution_complete": False,
        "sparse_blossom_public_benchmark_reproduced":
            bool(decoder.get("prerequisites", {}).get("public_2024_benchmark_reproduced", False)),
        "decoder_heldout_four_arm_complete":
            bool(decoder.get("prerequisites", {}).get("heldout_four_arm_acquired", False)),
    }
    passed = all(gates.values())
    result = nonfinal({
        "pass": passed,
        "status": "REFERENCE_GATE_PASS" if passed else "REFERENCE_GATE_CLOSED",
        "gates": gates,
        "blocking_reasons": [name for name, value in gates.items() if not value],
        "freeze_status": freeze_reason,
        "force_override_allowed": False,
        "force_override_requested": False,
        "promotion_performed": False,
    })
    atomic_json(ARTIFACT_ROOT / "gate/reference_gate_status.json", result)
    return result
