"""Bounded three-frequency validation of the isolated public-analogue controller."""
from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from hdfa_rl_suite.google_pure_source_exact.figure5a.contracts import (
    AcquisitionMode,
    Figure5aProtocol,
)
from hdfa_rl_suite.google_pure_source_exact.figure5a.validation import (
    build_plant,
    dependency_hashes,
)
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.contracts import (
    PositivityGuard,
)
from hdfa_rl_suite.google_pure_source_exact.policy_parameterization.optimizer import (
    OptimizerConfig,
)
from hdfa_rl_suite.google_pure_source_exact.source_normalization import (
    SourceNormalizationBoundary,
)
from hdfa_rl_suite.google_pure_v17.estimators import estimate_sinusoidal_transfer

from .acquisition import STREAMS, run_experimental_cell
from .controller import (
    CONTROLLER_MODE,
    PARAMETERIZATION,
    SCALE_OBJECTIVE,
    PublicAnalogueControllerSpec,
)
from .io import (
    ARTIFACT_ROOT,
    CONFIG_PATH,
    ROOT,
    atomic_json,
    atomic_text,
    canonical_hash,
    config,
    file_hash,
    nonfinal,
    read_json,
)


FREQUENCY_ORDER = ("slow", "intermediate", "fast")
ACQUISITION_ORDER = ("fast", "intermediate", "slow")


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _source_config() -> dict[str, Any]:
    return read_json(ROOT / "configs/google_pure_source_exact/figure5a.json")


def _frozen_optimizer() -> dict[str, Any]:
    return read_json(ROOT / "artifacts/google_pure_v16/frozen_source_normalized_optimizer.json")


def _controller_spec() -> PublicAnalogueControllerSpec:
    frozen = _frozen_optimizer()
    return PublicAnalogueControllerSpec(
        inherited_entropy_coefficient=float(frozen["entropy_coefficient"]),
        active_dimensions=41,
        frozen_parent_controller_hash=str(frozen["optimizer_bundle_hash"]),
        mean_learning_rate=float(frozen["mean_learning_rate"]),
        sigma_learning_rate=float(frozen["sigma_learning_rate"]),
        baseline_learning_rate=float(frozen["baseline_learning_rate"]),
        ppo_clip=float(frozen["ppo_clip"]),
        baseline_loss_weight=float(frozen["baseline_loss_weight"]),
        minimum_sigma=float(frozen["minimum_sigma"]),
        maximum_sigma=float(frozen["maximum_sigma"]),
        initial_sigma=float(frozen["initial_sigma"]),
    )


def _optimizer_config(controller: PublicAnalogueControllerSpec) -> OptimizerConfig:
    frozen = _frozen_optimizer()
    return OptimizerConfig(
        controller.mean_learning_rate, controller.sigma_learning_rate,
        controller.baseline_learning_rate, momentum=float(frozen["momentum"]),
        minimum_sigma=controller.minimum_sigma, maximum_sigma=controller.maximum_sigma,
        positivity_guard=PositivityGuard(frozen["positivity_guard"]),
    )


def _boundary(plant: Any) -> SourceNormalizationBoundary:
    degree = np.sum(plant.mask, axis=0).astype(float)
    coefficient = np.asarray([row.omega_sensitivity for row in plant.inventory]) * degree
    return SourceNormalizationBoundary.from_training_objective(
        "FIGURE5A_REAL_TIME_STEERING", coefficient, control_ids=plant.parameter_ids)


def _frozen_source_paths() -> dict[str, Path]:
    return {
        "source_style_acquisition_code": ROOT /
            "src/hdfa_rl_suite/google_pure_source_exact/figure5a/acquisition.py",
        "source_style_loss_code": ROOT /
            "src/hdfa_rl_suite/google_pure_source_exact/policy_parameterization/losses.py",
        "source_style_optimizer_code": ROOT /
            "src/hdfa_rl_suite/google_pure_source_exact/policy_parameterization/optimizer.py",
        "source_style_gaussian_code": ROOT /
            "src/hdfa_rl_suite/google_pure_source_exact/policy_parameterization/gaussian.py",
        "source_style_optimizer_bundle": ROOT /
            "artifacts/google_pure_v16/frozen_source_normalized_optimizer.json",
        "v18_slow_checkpoint": ROOT /
            "artifacts/google_pure_v18/acquisition/slow/checkpoint.json",
        "v18_intermediate_checkpoint": ROOT /
            "artifacts/google_pure_v18/acquisition/intermediate/checkpoint.json",
        "v18_fast_checkpoint": ROOT /
            "artifacts/google_pure_v18/extended_fast/checkpoint.json",
        "v18_slow_transfer": ROOT / "artifacts/google_pure_v18/transfer_slow.json",
        "v18_intermediate_transfer": ROOT /
            "artifacts/google_pure_v18/transfer_intermediate.json",
        "v18_fast_transfer": ROOT /
            "artifacts/google_pure_v18/extended_fast/transfer_fast_extended.json",
        "v19_root_cause": ROOT /
            "artifacts/google_pure_v19/root_cause_classification.json",
        "v19_minimal_repair": ROOT / "artifacts/google_pure_v19/minimal_repair.json",
        "v19_postrepair_validation": ROOT /
            "artifacts/google_pure_v19/postrepair_validation.json",
    }


def _hashes(paths: Mapping[str, Path]) -> dict[str, dict[str, str]]:
    missing = [_relative(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing frozen parent inputs: {missing}")
    return {name: {"path": _relative(path), "sha256": file_hash(path)}
            for name, path in paths.items()}


def build_preflight_manifest() -> dict[str, Any]:
    settings = config()
    controller = _controller_spec()
    frozen = _frozen_optimizer()
    root_cause = read_json(ROOT / "artifacts/google_pure_v19/root_cause_classification.json")
    repair = read_json(ROOT / "artifacts/google_pure_v19/minimal_repair.json")
    validation = read_json(ROOT / "artifacts/google_pure_v19/postrepair_validation.json")
    source_hashes = _hashes(_frozen_source_paths())
    gates = {
        "v19_cause_authorizes_public_analogue": root_cause.get("classification") ==
            "SCALE_OBJECTIVE_EQUILIBRIUM_TOO_EXPLORATORY",
        "v19_repair_is_public_analogue": repair.get("repair") ==
            "PUBLIC_ANALOGUE_SCALE_OBJECTIVE" and repair.get("source_exact") is False,
        "v19_static_validation_passed": validation.get("pass") is True,
        "controller_hash_is_distinct": controller.controller_hash !=
            frozen["optimizer_bundle_hash"],
        "parent_hash_is_exact": controller.frozen_parent_controller_hash ==
            frozen["optimizer_bundle_hash"],
        "mean_learning_rate_unchanged": controller.mean_learning_rate ==
            float(frozen["mean_learning_rate"]),
        "sigma_learning_rate_unchanged": controller.sigma_learning_rate ==
            float(frozen["sigma_learning_rate"]),
        "baseline_learning_rate_unchanged": controller.baseline_learning_rate ==
            float(frozen["baseline_learning_rate"]),
        "only_entropy_reduction_changed": math.isclose(
            controller.effective_entropy_coefficient,
            controller.inherited_entropy_coefficient / controller.active_dimensions,
            rel_tol=0, abs_tol=0),
        "no_automatic_campaigns": settings["automatic_campaigns_permitted"] == [],
        "no_heldout_seeds": settings["heldout_seeds"] == [],
    }
    if not all(gates.values()):
        raise RuntimeError(f"public-analogue experimental preflight failed: {gates}")
    value = nonfinal({
        "pass": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "controller": controller.identity_payload,
        "controller_hash": controller.controller_hash,
        "frozen_parent_controller_hash": controller.frozen_parent_controller_hash,
        "frozen_source_branch_hashes": source_hashes,
        "experimental_code_hashes": {
            "controller": file_hash(Path(__file__).with_name("controller.py")),
            "acquisition": file_hash(Path(__file__).with_name("acquisition.py")),
            "dynamic_validation": file_hash(Path(__file__)),
            "protocol": file_hash(CONFIG_PATH),
        },
        "gates": gates,
        "acquisition_order": list(ACQUISITION_ORDER),
        "forbidden_auto_runs": settings["forbidden_auto_runs"],
        "forbidden_auto_runs_launched": [],
    })
    path = ARTIFACT_ROOT / "preflight_manifest.json"
    if path.is_file():
        previous = read_json(path)
        immutable = (
            "controller_hash", "frozen_parent_controller_hash", "frozen_source_branch_hashes",
            "experimental_code_hashes", "acquisition_order")
        changed = [key for key in immutable if previous.get(key) != value.get(key)]
        if changed:
            allowed = set(changed) <= {"experimental_code_hashes"}
            previous_code = previous.get("experimental_code_hashes", {})
            current_code = value.get("experimental_code_hashes", {})
            scientific_code_unchanged = all(
                previous_code.get(key) == current_code.get(key)
                for key in ("controller", "protocol"))
            status_absent = not (ARTIFACT_ROOT / "status.json").is_file()
            if not (allowed and scientific_code_unchanged and status_absent):
                raise RuntimeError(f"experimental preflight changed after creation: {changed}")
            value["preflight_revision"] = 2
            value["supersedes_preflight_sha256"] = file_hash(path)
            value["execution_only_repair"] = "WINDOWS_ATOMIC_REPLACE_BOUNDED_RETRY"
            value["scientific_protocol_changed"] = False
            revision_path = ARTIFACT_ROOT / "preflight_manifest_revision_2.json"
            if revision_path.is_file():
                existing_revision = read_json(revision_path)
                immutable_revision = {
                    key: value[key] for key in (
                        "controller_hash", "frozen_parent_controller_hash",
                        "frozen_source_branch_hashes", "experimental_code_hashes",
                        "acquisition_order", "preflight_revision",
                        "supersedes_preflight_sha256", "execution_only_repair",
                        "scientific_protocol_changed")
                }
                if any(existing_revision.get(key) != item
                       for key, item in immutable_revision.items()):
                    raise RuntimeError("experimental preflight revision changed after creation")
                return existing_revision
            atomic_json(revision_path, value)
            return value
        return previous
    atomic_json(path, value)
    return value


def _stream_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {stream: int(sum(row["stream_totals"][stream] for row in records))
              for stream in STREAMS}
    fixed, optimal = totals["fixed"], totals["optimal"]
    denominator = fixed - optimal
    standard_error = math.sqrt(max(fixed + optimal, 0))
    resolved = denominator > 3.0 * standard_error
    i_mean = ((fixed - totals["learned_mean"]) / denominator if denominator else None)
    i_stochastic = ((fixed - totals["stochastic"]) / denominator if denominator else None)
    return {
        "C_fixed": fixed,
        "C_optimal": optimal,
        "C_mean": totals["learned_mean"],
        "C_stochastic": totals["stochastic"],
        "normalization_denominator": denominator,
        "denominator_standard_error": standard_error,
        "denominator_snr": denominator / standard_error if standard_error else None,
        "denominator_resolved": resolved,
        "I_mean": i_mean,
        "I_stochastic": i_stochastic,
        "sampled_policy_I_positive": bool(i_stochastic is not None and i_stochastic > 0),
        "exploration_damage": totals["stochastic"] - totals["learned_mean"],
        "stream_separation_retained": list(STREAMS),
    }


def _fit_period(records: list[dict[str, Any]], frequency: float) -> dict[str, float]:
    epochs = np.asarray([row["epoch"] for row in records], dtype=float)
    direction = np.asarray([
        np.mean(np.asarray(row["normalized_behavior_mean"], dtype=float)) for row in records])
    return estimate_sinusoidal_transfer(
        epochs, direction, frequency, minimum_cycles=1.0,
        maximum_condition_number=1000.0)


def _bootstrap_periods(period_fits: list[dict[str, float]], *, draws: int,
                       seed: int) -> dict[str, Any]:
    coefficients = np.asarray([
        [row["sine_coefficient"], row["cosine_coefficient"]] for row in period_fits])
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(coefficients), size=(draws, len(coefficients)))
    sampled = np.mean(coefficients[indices], axis=1)
    gains = np.hypot(sampled[:, 0], sampled[:, 1])
    point = np.mean(coefficients, axis=0)
    point_phase = float(-math.atan2(point[1], point[0]))
    phases = np.asarray([
        point_phase + np.angle(np.exp(1j * (-math.atan2(row[1], row[0]) - point_phase)))
        for row in sampled])
    return {
        "method": "COMPLETE_PERIOD_NONPARAMETRIC_BOOTSTRAP",
        "complete_period_units": len(period_fits),
        "draws": draws,
        "gain_samples": gains.tolist(),
        "phase_lag_samples_radians": phases.tolist(),
        "gain_confidence_interval_95": np.quantile(gains, [.025, .975]).tolist(),
        "phase_lag_confidence_interval_95": np.quantile(phases, [.025, .975]).tolist(),
        "uncertainty_scope": "WITHIN_SINGLE_DEVELOPMENT_RUN_NOT_BETWEEN_SEEDS",
    }


def _analyse_cell(label: str, cell: dict[str, Any], settings: Mapping[str, Any],
                  controller: PublicAnalogueControllerSpec, *,
                  protocol_settings: Mapping[str, Any] | None = None,
                  artifact_root: Path = ARTIFACT_ROOT) -> dict[str, Any]:
    all_settings = dict(protocol_settings) if protocol_settings is not None else config()
    records = list(cell["epoch_records"])
    frequency = float(settings["frequency_per_epoch"])
    period = int(round(1.0 / frequency))
    transient = int(settings["transient_periods"])
    analysis_periods = int(settings["analysis_periods"])
    start = transient * period
    selected = records[start:]
    if len(selected) != analysis_periods * period:
        raise RuntimeError(f"{label} analysis window is not exact complete periods")
    regression = _fit_period(selected, frequency)
    period_fits = [
        _fit_period(records[(transient + index) * period:(transient + index + 1) * period],
                    frequency)
        for index in range(analysis_periods)
    ]
    bootstrap = _bootstrap_periods(
        period_fits, draws=int(all_settings["bootstrap_draws"]),
        seed=int(all_settings["bootstrap_seed"]) + FREQUENCY_ORDER.index(label))
    sigma = np.asarray([row["behavior_sigma"] for row in selected], dtype=float)
    post_sigma = np.asarray([row["post_update_sigma"] for row in selected], dtype=float)
    scale = {
        "initial_analysis_sigma_median": float(np.median(sigma[0])),
        "terminal_sigma_median": float(np.median(post_sigma[-1])),
        "analysis_sigma_median": float(np.median(sigma)),
        "analysis_sigma_minimum": float(np.min(sigma)),
        "analysis_sigma_maximum": float(np.max(sigma)),
        "floor_occupancy": float(np.mean(np.isclose(
            post_sigma, controller.minimum_sigma, rtol=0, atol=1e-12))),
        "ceiling_occupancy": float(np.mean(np.isclose(
            post_sigma, controller.maximum_sigma, rtol=0, atol=1e-12))),
        "reward_sigma_gradient_norm_median": float(np.median([
            row["reward_sigma_gradient_norm"] for row in selected])),
        "entropy_sigma_gradient_norm_median": float(np.median([
            row["entropy_sigma_gradient_norm"] for row in selected])),
        "sigma_update_norm_median": float(np.median([
            np.linalg.norm(np.asarray(row["post_update_sigma"]) -
                           np.asarray(row["behavior_sigma"])) for row in selected])),
        "mean_stochastic_decomposition_retained": True,
    }
    result = nonfinal({
        "pass": True,
        "label": label,
        "frequency_per_epoch": frequency,
        "period_epochs": period,
        "total_epochs": len(records),
        "analysis_epoch_window": [start, len(records)],
        "complete_analysis_periods": analysis_periods,
        "mean_transfer_regression": regression,
        "bootstrap_uncertainty": bootstrap,
        "stream_decomposition": _stream_metrics(selected),
        "sigma_diagnostics": scale,
        "controller_mode": CONTROLLER_MODE,
        "controller_hash": controller.controller_hash,
        "frozen_parent_controller_hash": controller.frozen_parent_controller_hash,
        "parameterization": PARAMETERIZATION,
        "scale_objective": SCALE_OBJECTIVE,
        "mean_hyperparameters_changed": False,
        "checkpoint": _relative(
            artifact_root / "acquisition" / label / "checkpoint.json"),
        "candidate_qec_cycles": cell["candidate_qec_cycles"],
        "four_stream_qec_cycles": cell["four_stream_qec_cycles"],
        "fresh_acquisition": cell["fresh_acquisition"],
        "forbidden_auto_runs_launched": [],
    })
    result["pass"] = bool(
        result["stream_decomposition"]["denominator_resolved"] and
        np.isfinite(result["mean_transfer_regression"]["gain"]) and
        np.isfinite(result["mean_transfer_regression"]["phase_lag_radians"]))
    atomic_json(artifact_root / f"transfer_{label}.json", result)
    return result


def _ordering(rows: Mapping[str, dict[str, Any]], *,
              protocol_settings: Mapping[str, Any] | None = None) -> dict[str, Any]:
    all_settings = dict(protocol_settings) if protocol_settings is not None else config()
    ordered = [rows[label] for label in FREQUENCY_ORDER]
    gains = [float(row["mean_transfer_regression"]["gain"]) for row in ordered]
    phases = [float(row["mean_transfer_regression"]["phase_lag_radians"]) for row in ordered]
    gain_samples = [np.asarray(row["bootstrap_uncertainty"]["gain_samples"])
                    for row in ordered]
    phase_samples = [np.asarray(row["bootstrap_uncertainty"]["phase_lag_samples_radians"])
                     for row in ordered]
    size = min(*(len(row) for row in gain_samples + phase_samples))
    gain_draw = ((gain_samples[0][:size] > gain_samples[1][:size]) &
                 (gain_samples[1][:size] > gain_samples[2][:size]))
    phase_draw = ((phase_samples[0][:size] < phase_samples[1][:size]) &
                  (phase_samples[1][:size] < phase_samples[2][:size]))
    threshold = float(all_settings["minimum_joint_ordering_probability"])
    point = gains[0] > gains[1] > gains[2] and phases[0] < phases[1] < phases[2]
    joint_probability = float(np.mean(gain_draw & phase_draw))
    return {
        "claim": "G_slow>G_intermediate>G_fast and phi_slow<phi_intermediate<phi_fast",
        "gain_point_estimates": gains,
        "phase_lag_point_estimates_radians": phases,
        "gain_point_ordering_pass": gains[0] > gains[1] > gains[2],
        "phase_point_ordering_pass": phases[0] < phases[1] < phases[2],
        "point_ordering_pass": point,
        "bootstrap_gain_ordering_probability": float(np.mean(gain_draw)),
        "bootstrap_phase_ordering_probability": float(np.mean(phase_draw)),
        "bootstrap_joint_ordering_probability": joint_probability,
        "minimum_joint_ordering_probability": threshold,
        "pass": bool(point and joint_probability >= threshold),
    }


def _verify_frozen_source_branch(preflight: Mapping[str, Any]) -> dict[str, Any]:
    after = _hashes(_frozen_source_paths())
    before = preflight["frozen_source_branch_hashes"]
    changed = {name: {"before": before.get(name), "after": value}
               for name, value in after.items() if before.get(name) != value}
    return {"pass": not changed, "hashes_before": before,
            "hashes_after": after, "changed": changed}


def _write_report(result: Mapping[str, Any], *, artifact_root: Path = ARTIFACT_ROOT) -> None:
    lines = [
        "# V19 public-analogue dynamic three-frequency validation", "",
        f"Controller: `{result['controller_mode']}`.",
        "This branch divides the inherited entropy coefficient by the 41 active coordinates. "
        "It is not an identified source controller or source-equilibrium claim.", "",
        "## Dynamic results", "",
    ]
    for row in result["rows"]:
        metric = row["stream_decomposition"]
        sigma = row["sigma_diagnostics"]
        lines.append(
            f"- {row['label']}: G={row['mean_transfer_regression']['gain']:.5f}, "
            f"phi={row['mean_transfer_regression']['phase_lag_radians']:.5f} rad, "
            f"I_mean={metric['I_mean']:.5f}, "
            f"I_stochastic={metric['I_stochastic']:.5f}, "
            f"sigma_median={sigma['analysis_sigma_median']:.5f}.")
    lines += [
        "", "## Gates", "",
        f"- Gain and phase ordering: {result['ordering']['pass']}.",
        f"- Sampled-policy I positive at all frequencies: "
        f"{result['sampled_policy_I_positive_all_frequencies']}.",
        f"- Frozen source-style branch unchanged: {result['frozen_source_branch_unchanged']}.",
        "- Mean/stochastic decomposition and sigma diagnostics were retained.",
        "- No held-out, source-budget, reference, natural-drift, Figure 5c, or paired-acceptance campaign ran.",
        "", "## Evidence boundary", "",
        "This is a small single-seed-per-frequency development validation. It cannot establish "
        "paper equivalence, final evidence, or between-seed uncertainty.",
    ]
    atomic_text(artifact_root / "REPORT.md", "\n".join(lines))


def run_three_frequency_validation() -> dict[str, Any]:
    completed_path = ARTIFACT_ROOT / "status.json"
    if completed_path.is_file():
        return read_json(completed_path)
    settings = config()
    preflight = build_preflight_manifest()
    controller = _controller_spec()
    source = _source_config()
    plant = build_plant(source)
    boundary = _boundary(plant)
    dependencies = {
        **dependency_hashes(ROOT, source),
        "experimental_controller_code": file_hash(Path(__file__).with_name("controller.py")),
        "experimental_acquisition_code": file_hash(Path(__file__).with_name("acquisition.py")),
        "experimental_protocol": file_hash(CONFIG_PATH),
    }
    analysed: dict[str, dict[str, Any]] = {}
    for label in ACQUISITION_ORDER:
        row = settings["frequencies"][label]
        protocol = Figure5aProtocol(
            AcquisitionMode.VALIDATION, int(row["epochs"]),
            int(row["candidates_per_epoch"]), int(row["qec_cycles_per_candidate"]),
            int(source["plant"]["circuit_rounds"]))
        checkpoint = ARTIFACT_ROOT / "acquisition" / label / "checkpoint.json"
        cell = run_experimental_cell(
            protocol=protocol, plant=plant,
            frequency=float(row["frequency_per_epoch"]), seed=int(row["seed"]),
            optimizer_config=_optimizer_config(controller), controller=controller,
            checkpoint_path=checkpoint, dependency_hashes=dependencies, boundary=boundary,
            resume=checkpoint.is_file())
        analysed[label] = _analyse_cell(label, cell, row, controller)
    ordering = _ordering(analysed)
    frozen_check = _verify_frozen_source_branch(preflight)
    rows = [analysed[label] for label in FREQUENCY_ORDER]
    sampled_positive = all(
        row["stream_decomposition"]["sampled_policy_I_positive"] for row in rows)
    result = nonfinal({
        "pass": bool(
            all(row["pass"] for row in rows) and ordering["pass"] and sampled_positive and
            frozen_check["pass"]),
        "execution_complete": True,
        "controller_mode": CONTROLLER_MODE,
        "controller_hash": controller.controller_hash,
        "frozen_parent_controller_hash": controller.frozen_parent_controller_hash,
        "parameterization": PARAMETERIZATION,
        "scale_objective": SCALE_OBJECTIVE,
        "rows": rows,
        "ordering": ordering,
        "sampled_policy_I_positive_all_frequencies": sampled_positive,
        "frozen_source_branch_unchanged": frozen_check["pass"],
        "frozen_source_branch_hash_audit": frozen_check,
        "mean_stochastic_decomposition_retained": True,
        "sigma_diagnostics_retained": True,
        "mean_hyperparameters_changed": False,
        "only_scale_objective_changed": True,
        "between_seed_uncertainty_available": False,
        "campaign_scope": "SMALL_SINGLE_SEED_THREE_FREQUENCY_DEVELOPMENT_VALIDATION",
        "forbidden_auto_runs": settings["forbidden_auto_runs"],
        "forbidden_auto_runs_launched": [],
    })
    atomic_json(ARTIFACT_ROOT / "three_frequency_validation.json", result)
    atomic_json(ARTIFACT_ROOT / "status.json", result)
    _write_report(result)
    return result
