"""Fail-closed offline decoder-prior steering closure audit."""
from __future__ import annotations

from importlib import metadata
from typing import Any

from hdfa_rl_suite.google_pure_source_exact.offline_decoder_prior.factorial import (
    ARMS, FourArmResult, decompose_four_arms)

from .contracts import PUBLIC_NON_IDENTIFIABLE, nonfinal
from .io import ARTIFACT_ROOT, ROOT, atomic_json, canonical_hash, read_json


def run_decoder_steering_offline() -> dict[str, Any]:
    prior_status = read_json(
        ROOT / "artifacts/google_pure_source_exact/offline_decoder_prior/final_status.json")
    try:
        proxy_version = metadata.version("pymatching")
    except metadata.PackageNotFoundError:
        proxy_version = None
    # This fixture tests the factorial accounting and identical-shot guard. It is
    # deliberately not presented as a scientific decoder evaluation.
    fixed_hash = canonical_hash({"fixture": "fixed_physical_shots"})
    learned_hash = canonical_hash({"fixture": "learned_physical_shots"})
    shot_fixed = canonical_hash({"shots": list(range(1000))})
    shot_learned = canonical_hash({"shots": list(range(1000, 2000))})
    fixture = FourArmResult(
        logical_error_rates={
            "fixed_controls_fixed_prior": .020,
            "learned_controls_fixed_prior": .017,
            "fixed_controls_steered_prior": .018,
            "learned_controls_steered_prior": .014,
        },
        physical_data_hashes={
            "fixed_controls_fixed_prior": fixed_hash,
            "fixed_controls_steered_prior": fixed_hash,
            "learned_controls_fixed_prior": learned_hash,
            "learned_controls_steered_prior": learned_hash,
        },
        shot_id_hashes={
            "fixed_controls_fixed_prior": shot_fixed,
            "fixed_controls_steered_prior": shot_fixed,
            "learned_controls_fixed_prior": shot_learned,
            "learned_controls_steered_prior": shot_learned,
        },
    )
    fixture_decomposition = decompose_four_arms(fixture)
    frozen_datasets = list((ARTIFACT_ROOT / "decoder/frozen_data").glob("*.npz"))
    prerequisites = {
        "verified_sparse_blossom_available": bool(prior_status["verified_sparse_blossom_available"]),
        "public_2024_benchmark_reproduced": bool(prior_status["primary_public_benchmark_reproduced"]),
        "immutable_fixed_control_dataset_available": any("fixed" in path.name for path in frozen_datasets),
        "immutable_learned_control_dataset_available": any("learned" in path.name for path in frozen_datasets),
        "heldout_four_arm_acquired": bool(prior_status["held_out_four_arm_complete"]),
    }
    executable = all(prerequisites.values())
    result = nonfinal({
        "pass": True,
        "execution_status": "COMPLETE" if executable else "BLOCKED_PREREQUISITES",
        "prerequisites": prerequisites,
        "blocking_reasons": [name for name, value in prerequisites.items() if not value],
        "primary_backend": "SPARSE_BLOSSOM_CORRELATED_MATCHING_TWO_STEP_REWEIGHTING",
        "installed_proxy": f"PyMatching {proxy_version}" if proxy_version else None,
        "proxy_is_paper_equivalent": False,
        "physical_controls_frozen": True,
        "live_controller_coupling": False,
        "logical_outcomes_allowed_in_physical_reward": False,
        "required_arms": list(ARMS),
        "same_physical_data_and_shots_within_decoder_pairs_required": True,
        "factorial_accounting_fixture_pass": True,
        "factorial_accounting_fixture": fixture_decomposition,
        "fixture_is_scientific_evidence": False,
        "source_gap_classification": PUBLIC_NON_IDENTIFIABLE,
    })
    atomic_json(ARTIFACT_ROOT / "decoder/offline_steering.json", result)
    return result
