"""Direct-sigma V13 validation runtime with state and candidate provenance."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hdfa_rl_suite.google_pure_source_exact.paper_families.common import (
    SparseControlPlant,
    _sparse_source_loss,
    controller_config,
    optimizer_config,
)
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.gaussian import DirectSigmaGaussianPolicy
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.optimizer import DirectSigmaOptimizer
from hdfa_rl_suite.google_pure_source_exact.step_response_130.plant import SourceStepPlant

from .contracts import NONFINAL, V13_SCHEMA
from .io import ARTIFACT_ROOT, ROOT, atomic_json, atomic_text, canonical_hash, config, read_json
from .sensitivity import CoordinateBatch, SensitivityBoundary, require_native_boundary


STEP = "STEP_RESPONSE_INJECTED_DRIFT"
RECOVERY = "RANDOMIZED_RECOVERY_AFTER_SPOIL"


def _array_hash(value: np.ndarray) -> str:
    return canonical_hash(np.asarray(value, dtype=float).tolist())


def _state_payload(policy: DirectSigmaGaussianPolicy, optimizer: DirectSigmaOptimizer,
                   baseline: np.ndarray) -> dict[str, Any]:
    return policy.state_dict(optimizer_state=optimizer.state_dict(), baseline=baseline)


def _case(family: str, seed: int) -> dict[str, Any]:
    if family == STEP:
        plant = SourceStepPlant(onset_epoch=int(config()["branch_comparison"]["step_onset_epoch"]))
        boundary = SensitivityBoundary.from_artifact(family)
        normalized_target = np.zeros(924)
        normalized_target[0] = plant.target_delta
        native_target = boundary.scales * normalized_target
        return {"family": family, "plant": plant, "boundary": boundary,
                "owners": np.arange(924, dtype=np.int64) % 24,
                "initial_normalized": np.zeros(924), "normalized_target": normalized_target,
                "native_target": native_target, "detectors": 24,
                "plant_hash": plant.plant_hash,
                "graph_hash": canonical_hash(plant.mask.astype(int).tolist()),
                "onset": plant.onset_epoch}
    if family == RECOVERY:
        plant = SparseControlPlant(5, 924, 24, seed=10_100, curvature=.004)
        boundary = SensitivityBoundary.from_artifact(family)
        rng = np.random.default_rng(int(seed))
        spoiled = np.zeros(924)
        selected = rng.choice(924, 462, replace=False)
        spoiled[selected] = rng.choice((-1.0, 1.0), len(selected)) * .75
        return {"family": family, "plant": plant, "boundary": boundary,
                "owners": plant.control_detector, "initial_normalized": spoiled,
                "normalized_target": np.zeros(924),
                "native_target": np.zeros(924), "detectors": 24,
                "plant_hash": plant.plant_hash, "graph_hash": plant.graph_hash, "onset": 0}
    raise ValueError(f"unsupported V13 validation family: {family}")


def _rates(case: dict[str, Any], native_actions: np.ndarray, epoch: int) -> np.ndarray:
    actions = np.atleast_2d(native_actions)
    if case["family"] == STEP:
        plant: SourceStepPlant = case["plant"]
        target = case["native_target"] if epoch >= case["onset"] else np.zeros(924)
        cost = plant.sensitivity[None, :] * np.square(actions - target[None, :])
        return np.clip(plant.base_edr[None, :] + cost @ plant.mask.T, 1e-9, .49)
    return case["plant"].expected_detector_rates(actions, case["native_target"])


def _projection(case: dict[str, Any], mean_normalized: np.ndarray) -> float:
    target = case["normalized_target"]
    if case["family"] == STEP:
        return float(mean_normalized[0] / target[0])
    initial_distance = float(np.linalg.norm(case["initial_normalized"] - target))
    return 1.0 - float(np.linalg.norm(mean_normalized - target)) / initial_distance


def run_v13_arm(family: str, *, seed: int, epochs: int, candidates: int,
                cycles_per_candidate: int, entropy_coefficient: float,
                persist: bool = True) -> dict[str, Any]:
    """Run one source-normalized arm; targets are used by the plant/evaluator only."""
    if min(epochs, candidates, cycles_per_candidate) <= 0:
        raise ValueError("runtime budgets must be positive")
    case = _case(family, seed)
    boundary: SensitivityBoundary = case["boundary"]
    policy_cfg = controller_config()
    policy = DirectSigmaGaussianPolicy(case["initial_normalized"],
                                       np.full(924, float(policy_cfg["initial_sigma"])), seed=int(seed))
    optimizer = DirectSigmaOptimizer(924, case["detectors"], optimizer_config())
    baseline = np.zeros(case["detectors"])
    state_chain: list[dict[str, Any]] = []
    candidate_lineage: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    stored_x: list[np.ndarray] = []
    stored_sx: list[np.ndarray] = []
    stored_u: list[np.ndarray] = []
    previous_output_hash: str | None = None
    initial_distance = float(np.linalg.norm(case["initial_normalized"] - case["normalized_target"]))
    for epoch in range(int(epochs)):
        input_state = _state_payload(policy, optimizer, baseline)
        input_hash = canonical_hash(input_state)
        if previous_output_hash is not None and input_hash != previous_output_hash:
            raise RuntimeError("policy state continuity failed before sampling")
        behavior_version = policy.policy_version
        batch = policy.sample(int(candidates))
        boundary_result = boundary.apply(CoordinateBatch(
            batch.actions, boundary.control_order_hash, sensitivity_map_hash=boundary.sensitivity_map_hash))
        require_native_boundary(boundary_result.native,
                                control_order_hash=boundary.control_order_hash,
                                sensitivity_map_hash=boundary.sensitivity_map_hash)
        native_actions = np.atleast_2d(boundary_result.native.values)
        stored_x.append(np.asarray(batch.actions, dtype=float).copy())
        stored_sx.append(np.asarray(batch.actions, dtype=float) * boundary.scales[None, :])
        stored_u.append(native_actions.copy())
        probabilities = _rates(case, native_actions, epoch)
        stream_seed = int(canonical_hash(["v13-source-normalized", family, int(seed), epoch])[:16], 16)
        counts = np.random.default_rng(stream_seed).binomial(int(cycles_per_candidate), probabilities)
        rewards = -counts / float(cycles_per_candidate)
        advantage = rewards - baseline[None, :]
        centred_reward = rewards - np.mean(rewards, axis=0, keepdims=True)
        detector_covariance = centred_reward.T @ centred_reward / max(1, candidates - 1)
        detector_eigenvalues = np.maximum(np.linalg.eigvalsh(detector_covariance), 0.0)
        detector_effective_rank = (float(np.sum(detector_eigenvalues)) ** 2 /
                                   max(float(np.sum(np.square(detector_eigenvalues))),
                                       np.finfo(float).tiny))
        loss = _sparse_source_loss(
            batch.actions, rewards, case["owners"], policy.mean, policy.sigma, baseline,
            batch.behavior, clip=float(policy_cfg["ppo_clip"]),
            entropy_weight=float(entropy_coefficient), baseline_weight=float(policy_cfg["baseline_weight"]))
        gradient_hash = canonical_hash({
            "mean": _array_hash(loss["grad_mean"]), "sigma": _array_hash(loss["grad_sigma"]),
            "baseline": _array_hash(loss["grad_baseline"]),
        })
        before_mean = policy.mean.copy()
        update = optimizer.step(policy.mean, policy.sigma, baseline,
                                loss["grad_mean"], loss["grad_sigma"], loss["grad_baseline"],
                                mean_bounds=(-2.0, 2.0))
        policy.policy_version += 1
        output_state = _state_payload(policy, optimizer, baseline)
        output_hash = canonical_hash(output_state)
        state_chain.append({
            "epoch": epoch, "input_state_hash": input_hash, "behavior_policy_version": behavior_version,
            "sampled_policy_version": batch.behavior.policy_version,
            "optimizer_step_before": epoch, "gradient_hash": gradient_hash,
            "output_state_hash": output_hash, "next_epoch_input_state_hash": output_hash,
            "policy_version_after": policy.policy_version, "optimizer_step_after": optimizer.steps,
            "baseline_hash_after": _array_hash(baseline),
            "rng_state_hash_after": canonical_hash(policy.rng.bit_generator.state),
        })
        previous_output_hash = output_hash
        for candidate in range(candidates):
            candidate_id = canonical_hash({"family": family, "seed": int(seed), "epoch": epoch,
                                           "candidate": candidate, "input_state_hash": input_hash})
            candidate_lineage.append({
                "candidate_id": candidate_id, "epoch": epoch, "candidate_index": candidate,
                "sampling_noise_hash": _array_hash(batch.standardized_noise[candidate]),
                "normalized_action_hash": boundary_result.normalized_hashes[candidate],
                "scale_hash": boundary_result.scale_hash,
                "reference_native_hash": boundary_result.reference_hash,
                "scaled_action_hash": boundary_result.scaled_action_hashes[candidate],
                "native_applied_action_hash": boundary_result.native_action_hashes[candidate],
                "sensitivity_map_hash": boundary.sensitivity_map_hash,
                "control_order_hash": boundary.control_order_hash,
                "sensitivity_application_count": 1,
                "detector_count_hash": canonical_hash(counts[candidate].tolist()),
                "reward_hash": _array_hash(rewards[candidate]),
                "advantage_hash": _array_hash(advantage[candidate]),
                "batch_gradient_hash": gradient_hash,
                "behavior_policy_version": behavior_version,
                "boundary_value_row": epoch * candidates + candidate,
            })
        projection = _projection(case, policy.mean)
        trace.append({
            "epoch": epoch, "target_active": epoch >= case["onset"],
            "target_relative_progress": projection,
            "normalized_mean_coordinate_0": float(policy.mean[0]),
            "native_mean_coordinate_0": float(boundary.scales[0] * policy.mean[0]),
            "mean_sigma": float(np.mean(policy.sigma)),
            "candidate_mean_edr": float(np.mean(probabilities)),
            "mean_motion": float(np.linalg.norm(policy.mean - before_mean)),
            "gradient_mean_norm": float(np.linalg.norm(loss["grad_mean"])),
            "gradient_sigma_norm": float(np.linalg.norm(loss["grad_sigma"])),
            "gradient_snr_batch_proxy": float(np.linalg.norm(np.mean(advantage, axis=0)) /
                                               max(np.mean(np.std(advantage, axis=0, ddof=1)) /
                                                   np.sqrt(candidates), np.finfo(float).tiny)) if candidates > 1 else 0.0,
            "detector_reward_effective_rank": detector_effective_rank,
            "policy_importance_weight_ess": float(candidates),
            "fraction_at_sigma_guard": update["fraction_at_positivity_guard"],
        })
    onset = case["onset"]
    eligible = np.asarray([row["target_relative_progress"] for row in trace[onset:]], dtype=float)
    crossing = np.flatnonzero(eligible >= .5)
    identity = read_json(ROOT / "artifacts/google_pure_source_exact/direct_sigma_integration/controller_identity.json")
    boundary_value_store: dict[str, Any] | None = None
    if persist:
        directory = ARTIFACT_ROOT / "runs" / family.lower() / str(seed)
        directory.mkdir(parents=True, exist_ok=True)
        value_path = directory / "boundary_values.npz"
        temporary = value_path.with_suffix(".npz.tmp")
        x_values = np.concatenate(stored_x, axis=0)
        sx_values = np.concatenate(stored_sx, axis=0)
        u_values = np.concatenate(stored_u, axis=0)
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, x=x_values, s=boundary.scales[None, :],
                                u0=boundary.reference[None, :], sx=sx_values, u=u_values)
        temporary.replace(value_path)
        boundary_value_store = {
            "path": str(value_path.resolve()), "format": "NPZ_FLOAT64",
            "candidate_rows": int(len(x_values)), "coordinates": int(x_values.shape[1]),
            "datasets": {"x": list(x_values.shape), "s": [1, len(boundary.scales)],
                         "u0": [1, len(boundary.reference)], "sx": list(sx_values.shape),
                         "u": list(u_values.shape)},
            "mapping": "u=u0+s*x", "broadcast_s_and_u0_per_candidate": True,
        }
    result = {
        "schema_version": V13_SCHEMA, "family": family,
        "arm": "V13_SOURCE_LITERAL_ONE_PERCENT_EDR_NORMALIZATION",
        "seed": int(seed), "epochs": int(epochs), "onset_epoch": onset,
        "candidate_count_per_epoch": int(candidates), "cycles_per_candidate": int(cycles_per_candidate),
        "qec_cycles": int(epochs * candidates * cycles_per_candidate),
        "detector_event_trials": int(epochs * candidates * cycles_per_candidate * case["detectors"]),
        "controller_mode": identity["controller_mode"], "controller_hash": identity["controller_hash"],
        "controller_code_hash": identity["controller_code_hash"], "parameterization": "direct_sigma",
        "plant_hash": case["plant_hash"], "graph_hash": case["graph_hash"],
        "sensitivity_map_hash": boundary.sensitivity_map_hash, "mapping": "u=u0+s*x",
        "sensitivity_application_count": 1, "controller_target_access": False,
        "controller_direction_access": False, "five_policy_decomposition_retained": True,
        "initial_target_distance_normalized": initial_distance,
        "final_target_fraction": float(trace[-1]["target_relative_progress"]),
        "response_time_50_epochs_after_onset": int(crossing[0]) if crossing.size else None,
        "trace": trace, "state_chain": state_chain, "candidate_lineage": candidate_lineage,
        "boundary_value_store": boundary_value_store,
        "boundary_values_verified_in_memory": bool(
            all(np.array_equal(sx, x * boundary.scales[None, :]) for x, sx in zip(stored_x, stored_sx)) and
            all(np.array_equal(u, boundary.reference[None, :] + sx) for sx, u in zip(stored_sx, stored_u))),
        **NONFINAL,
    }
    result["state_chain_pass"] = verify_state_chain(result=result)["pass"]
    result["candidate_lineage_pass"] = verify_candidate_lineage(result=result)["pass"]
    if persist:
        atomic_json(directory / "run.json", result)
    return result


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    finals = np.asarray([row["final_target_fraction"] for row in rows], dtype=float)
    responses = [row.get("response_time_50_epochs_after_onset") for row in rows
                 if row.get("response_time_50_epochs_after_onset") is not None]
    return {"run_count": len(rows), "median_final_target_fraction": float(np.median(finals)),
            "minimum_final_target_fraction": float(np.min(finals)),
            "maximum_final_target_fraction": float(np.max(finals)),
            "response_identified_count": len(responses),
            "median_response_time_50_epochs_after_onset": float(np.median(responses)) if responses else None}


def compare_normalization_branches(*, epochs_override: int | None = None) -> dict[str, Any]:
    """Compare missing, V12 outcome-derived, and V13 source-literal maps on paired seeds."""
    settings = config()["branch_comparison"]
    imported = read_json(ROOT / "artifacts/google_pure_v12/directional_comparison/comparison.json")
    rows: list[dict[str, Any]] = []
    for row in imported["rows"]:
        copy = dict(row)
        copy["normalization_branch"] = ("A_MISSING_BOUNDARY" if row["arm"] == "UNCHANGED_IMPORTED_PROTOCOL"
                                        else "B_V12_OUTCOME_DERIVED_BOUNDARY")
        rows.append(copy)
    for family, seeds, epochs in (
        (STEP, settings["step_seeds"], int(settings["step_epochs"])),
        (RECOVERY, settings["recovery_seeds"], int(settings["recovery_epochs"])),
    ):
        case_name = "FAILED_STEP_RESPONSE" if family == STEP else "FAILED_RANDOMIZED_RECOVERY"
        for seed in seeds:
            run = run_v13_arm(family, seed=int(seed), epochs=int(epochs_override or epochs),
                              candidates=int(settings["candidates_per_epoch"]),
                              cycles_per_candidate=int(settings["cycles_per_candidate"]),
                              entropy_coefficient=float(settings["entropy_coefficient"]), persist=True)
            run["case"] = case_name
            run["normalization_branch"] = "C_V13_SOURCE_LITERAL_BOUNDARY"
            rows.append(run)
    summaries: dict[str, Any] = {}
    for case_name in ("FAILED_STEP_RESPONSE", "FAILED_RANDOMIZED_RECOVERY"):
        summaries[case_name] = {}
        for branch in ("A_MISSING_BOUNDARY", "B_V12_OUTCOME_DERIVED_BOUNDARY",
                       "C_V13_SOURCE_LITERAL_BOUNDARY"):
            selected = [row for row in rows if row.get("case") == case_name and row["normalization_branch"] == branch]
            summaries[case_name][branch] = _summarize(selected)
    full_horizon = epochs_override is None
    result = {"schema_version": V13_SCHEMA, "rows": rows, "summaries": summaries,
              "paired_seed_sets_identical": True, "one_boundary_difference_per_branch": True,
              "v12_boundary_source_equivalence": "NOT_ESTABLISHED",
              "v13_kappa_ref_edr_fraction": .01,
              "full_preregistered_horizon": full_horizon, "long_paper_scale_run_launched": False,
              **NONFINAL}
    atomic_json(ARTIFACT_ROOT / "sensitivity_calibration/comparison.json", result)
    return result


def run_step_validation(*, epochs_override: int | None = None) -> dict[str, Any]:
    settings = config()["branch_comparison"]
    rows = [run_v13_arm(STEP, seed=int(seed),
                        epochs=int(epochs_override or settings["step_epochs"]),
                        candidates=int(settings["candidates_per_epoch"]),
                        cycles_per_candidate=int(settings["cycles_per_candidate"]),
                        entropy_coefficient=float(settings["entropy_coefficient"]), persist=True)
            for seed in settings["step_seeds"]]
    result = {"schema_version": V13_SCHEMA, "family": STEP, "rows": rows,
              "summary": _summarize(rows), "full_preregistered_horizon": epochs_override is None,
              **NONFINAL}
    atomic_json(ARTIFACT_ROOT / "step_validation/runs.json", result)
    return result


def verify_state_chain(*, result: dict[str, Any] | None = None) -> dict[str, Any]:
    if result is None:
        paths = list((ARTIFACT_ROOT / "runs").glob("*/*/run.json"))
        runs = [read_json(path) for path in paths]
    else:
        runs = [result]
    failures = []
    transitions = 0
    for run in runs:
        chain = run["state_chain"]
        for index, row in enumerate(chain):
            transitions += 1
            if row["behavior_policy_version"] != row["sampled_policy_version"]:
                failures.append(f"behavior_version:{run['family']}:{run['seed']}:{index}")
            if row["policy_version_after"] != row["behavior_policy_version"] + 1:
                failures.append(f"policy_version:{run['family']}:{run['seed']}:{index}")
            if row["optimizer_step_after"] != row["optimizer_step_before"] + 1:
                failures.append(f"optimizer_version:{run['family']}:{run['seed']}:{index}")
            if index + 1 < len(chain) and row["output_state_hash"] != chain[index + 1]["input_state_hash"]:
                failures.append(f"continuity:{run['family']}:{run['seed']}:{index}")
    value = {"schema_version": V13_SCHEMA, "pass": bool(runs) and not failures,
             "runs_checked": len(runs), "transitions_checked": transitions, "failures": failures, **NONFINAL}
    if result is None:
        atomic_json(ARTIFACT_ROOT / "provenance/state_chain_validation.json", value)
    return value


def verify_candidate_lineage(*, result: dict[str, Any] | None = None) -> dict[str, Any]:
    if result is None:
        paths = list((ARTIFACT_ROOT / "runs").glob("*/*/run.json"))
        runs = [read_json(path) for path in paths]
    else:
        runs = [result]
    failures = []
    candidates_checked = 0
    for run in runs:
        rows = run["candidate_lineage"]
        candidates_checked += len(rows)
        ids = [row["candidate_id"] for row in rows]
        if len(ids) != len(set(ids)):
            failures.append(f"duplicate_candidate_id:{run['family']}:{run['seed']}")
        required = ("sampling_noise_hash", "normalized_action_hash", "scale_hash",
                    "reference_native_hash", "scaled_action_hash", "native_applied_action_hash",
                    "detector_count_hash", "reward_hash", "advantage_hash", "batch_gradient_hash")
        for row in rows:
            if row["sensitivity_application_count"] != 1 or any(not row.get(key) for key in required):
                failures.append(f"incomplete_lineage:{row['candidate_id']}")
        store = run.get("boundary_value_store")
        if store:
            path = Path(store["path"])
            if not path.is_file():
                failures.append(f"missing_boundary_values:{run['family']}:{run['seed']}")
            else:
                with np.load(path) as values:
                    if values["x"].shape[0] != len(rows):
                        failures.append(f"boundary_value_row_count:{run['family']}:{run['seed']}")
                    if not np.array_equal(values["sx"], values["x"] * values["s"]):
                        failures.append(f"scaled_boundary_values:{run['family']}:{run['seed']}")
                    if not np.array_equal(values["u"], values["u0"] + values["sx"]):
                        failures.append(f"native_boundary_values:{run['family']}:{run['seed']}")
        elif not run.get("boundary_values_verified_in_memory"):
            failures.append(f"missing_boundary_value_evidence:{run['family']}:{run['seed']}")
    value = {"schema_version": V13_SCHEMA, "pass": bool(runs) and not failures,
             "runs_checked": len(runs), "candidates_checked": candidates_checked,
             "failures": failures, **NONFINAL}
    if result is None:
        atomic_json(ARTIFACT_ROOT / "provenance/candidate_lineage_validation.json", value)
    return value
