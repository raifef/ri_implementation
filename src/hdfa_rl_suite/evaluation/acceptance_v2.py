"""Lossless, streaming reconstruction and audit of the retained v2 comparison."""
from __future__ import annotations

from dataclasses import asdict, replace
import codecs
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterable, Mapping

from .benchmark import (
    ArmMetrics, BenchmarkConfig, BenchmarkRunner, ExponentialRecoveryFit,
    RecoveryEndpoint,
)


REQUIRED_ARMS = (
    "fixed", "periodic_recalibration", "oracle",
    "full_control_detector_rl", "predictive_hdfa_no_residual",
    "predictive_hdfa_residual_rl",
)
CENTRAL_ARMS = ("full_control_detector_rl", "predictive_hdfa_residual_rl")
DEFAULT_SOURCE_SHA256 = "60f4c0d4908921b74a177e9794a38ea976e47397f1139d899ddcc9b1a5af9c5a"
DEFAULT_PART_SHA256 = (
    "13e5659af1975c768deb115043a581a8b885a5196683d80633cee8d5bb44adc8",
    "92c9c2501fa21668a47af9d83362d4de834d5ecb7d74f11feae6f545e62037ef",
)


class _ConcatenatedGzipText:
    def __init__(self, paths: Iterable[Path]) -> None:
        self.paths = iter(paths)
        self.handle = None
        self.decoder = codecs.getincrementaldecoder("utf-8")()
        self.digest = hashlib.sha256()
        self.eof = False

    def read(self, size: int = 1 << 20) -> str:
        if self.eof:
            return ""
        pieces: list[str] = []
        while not pieces:
            if self.handle is None:
                try:
                    self.handle = gzip.open(next(self.paths), "rb")
                except StopIteration:
                    self.eof = True
                    tail = self.decoder.decode(b"", final=True)
                    return tail
            raw = self.handle.read(size)
            if raw:
                self.digest.update(raw)
                pieces.append(self.decoder.decode(raw, final=False))
            else:
                self.handle.close()
                self.handle = None
        return "".join(pieces)


class _StreamingJSON:
    def __init__(self, source: _ConcatenatedGzipText) -> None:
        self.source, self.buffer, self.pos = source, "", 0
        self.decoder = json.JSONDecoder()

    def _compact(self) -> None:
        if self.pos > (1 << 20):
            self.buffer, self.pos = self.buffer[self.pos:], 0

    def _fill(self) -> bool:
        self._compact()
        chunk = self.source.read()
        self.buffer += chunk
        return bool(chunk)

    def _space(self) -> None:
        while True:
            while self.pos < len(self.buffer) and self.buffer[self.pos].isspace():
                self.pos += 1
            if self.pos < len(self.buffer) or not self._fill():
                return

    def peek(self) -> str:
        self._space()
        if self.pos >= len(self.buffer):
            raise EOFError("unexpected end of reconstructed JSON")
        return self.buffer[self.pos]

    def expect(self, token: str) -> None:
        if self.peek() != token:
            raise ValueError(f"expected {token!r} at reconstructed offset")
        self.pos += 1

    def value(self):
        self._space()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.pos)
                self.pos = end
                return value
            except json.JSONDecodeError:
                if not self._fill():
                    raise ValueError("truncated or invalid reconstructed JSON")

    def array(self, consume: Callable[[object], None]) -> None:
        self.expect("[")
        if self.peek() == "]":
            self.pos += 1
            return
        while True:
            consume(self.value())
            delimiter = self.peek()
            self.pos += 1
            if delimiter == "]":
                return
            if delimiter != ",":
                raise ValueError("invalid reconstructed JSON array delimiter")

    def top_level(self, array_consumers: Mapping[str, Callable[[object], None]]) -> dict:
        retained: dict[str, object] = {}
        self.expect("{")
        while self.peek() != "}":
            key = self.value()
            self.expect(":")
            if self.peek() == "[":
                values: list[object] = []
                self.array(array_consumers.get(key, values.append))
                if key not in array_consumers:
                    retained[key] = values
            else:
                retained[key] = self.value()
            delimiter = self.peek()
            self.pos += 1
            if delimiter == "}":
                break
            if delimiter != ",":
                raise ValueError("invalid reconstructed top-level delimiter")
        return retained


def _metric(row: Mapping[str, object]) -> ArmMetrics:
    values = dict(row)
    values["recovery_endpoints"] = tuple(
        RecoveryEndpoint(**item) for item in values.get("recovery_endpoints", ()))
    fit = values.get("exponential_fit")
    values["exponential_fit"] = ExponentialRecoveryFit(**fit) if fit else None
    for name in ("missing_data_reasons", "timing_invalidity_reasons"):
        values[name] = tuple(values.get(name, ()))
    return ArmMetrics(**values)


def _canonical(value: object) -> object:
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def reconstruct_acceptance_v2(parts: tuple[Path, Path], output_dir: Path,
                              *, expected_source_sha256: str | None = DEFAULT_SOURCE_SHA256,
                              expected_part_sha256: tuple[str, str] | None = DEFAULT_PART_SHA256
                              ) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    part_hashes = tuple(hashlib.sha256(path.read_bytes()).hexdigest() for path in parts)
    metrics: list[Mapping[str, object]] = []
    baselines: list[Mapping[str, object]] = []
    trajectories = 0
    truth_accesses = 0
    trajectory_keys: set[tuple[str, int, str, int]] = set()
    trajectory_run_counts: dict[tuple[str, int, str], int] = {}
    trajectory_disturbances: dict[tuple[str, int], set[str]] = {}
    timing_missing = 0
    rollback_rows = lifecycle_rows = 0

    def trajectory(row: object) -> None:
        nonlocal trajectories, truth_accesses, timing_missing, rollback_rows, lifecycle_rows
        item = row
        trajectories += 1
        key = (str(item["scenario_id"]), int(item["seed"]),
               str(item["arm"]), int(item["interval"]))
        trajectory_keys.add(key)
        if key[2] != "oracle":
            truth_accesses += len(item.get("controller_truth_accesses", ()))
        trajectory_run_counts[key[:3]] = trajectory_run_counts.get(key[:3], 0)+1
        trajectory_disturbances.setdefault(key[:2], set()).add(
            str(item.get("disturbance_realization_id", "")))
        timing_missing += int(key[2] in CENTRAL_ARMS and not item.get("timing"))
        rollback_rows += int(bool(item.get("rollback_outcomes")))
        lifecycle_rows += int(bool(item.get("lifecycle_violations")))

    source = _ConcatenatedGzipText(parts)
    retained = _StreamingJSON(source).top_level({
        "metrics": metrics.append,
        "pre_disturbance_baselines": baselines.append,
        "trajectories": trajectory,
    })
    source_hash = source.digest.hexdigest()
    config = retained["config"]
    scenarios = retained["scenarios"]
    expected_runs = len(scenarios)*len(config["seeds"])*len(REQUIRED_ARMS)
    arm_set = sorted({str(item["arm"]) for item in metrics})
    central = [item for item in metrics if item["arm"] in CENTRAL_ARMS]
    baseline_groups: dict[tuple[str, int], set[str]] = {}
    initial_groups: dict[tuple[str, int], set[tuple[str, str, str]]] = {}
    evaluator_groups: dict[tuple[str, int], set[tuple[str, str]]] = {}
    for item in baselines:
        key = (str(item["scenario_id"]), int(item["seed"]))
        baseline_groups.setdefault(key, set()).add(str(item.get("observation_hash", "")))
        initial_groups.setdefault(key, set()).add((
            str(item.get("initial_physical_state_id", "")),
            str(item.get("initial_disturbance_state_id", "")),
            str(item.get("initial_controller_state_hash", ""))))
        evaluator_groups.setdefault(key, set()).add((
            str(item.get("detector_evaluator_config_hash", "")),
            str(item.get("logical_evaluator_config_hash", ""))))

    typed_metrics = tuple(_metric(item) for item in metrics)
    trajectory_completion_exact = True
    for item in metrics:
        key = (str(item["scenario_id"]), int(item["seed"]), str(item["arm"]))
        count = trajectory_run_counts.get(key, 0)
        status = item.get("completion_status")
        trajectory_completion_exact = trajectory_completion_exact and (
            count == int(config["intervals"]) if status == "completed"
            else 0 < count <= int(config["intervals"]) if status == "censored"
            else False)
    runner = object.__new__(BenchmarkRunner)
    # V2 is an immutable historical analysis contract.  V3 separates the
    # worst-pair decision statistic from cluster-aggregate uncertainty, but
    # reconstruction must preserve V2's original serialized gate schema.
    runner.config = BenchmarkConfig(**{
        **config, "estimator_schema_version": "legacy.v1"})
    runner.scenarios = tuple(SimpleNamespace(
        scenario_id=item["scenario_id"], structured=item["structured"])
        for item in scenarios)
    recalculated_gates = BenchmarkRunner._gates(runner, typed_metrics)
    stored_gates = retained.get("gates", [])
    recalculated_gate_rows = [asdict(item) for item in recalculated_gates]
    for row in recalculated_gate_rows:
        if row.get("estimators") is None:
            row.pop("estimators", None)
    gate_exact = _canonical(recalculated_gate_rows) == _canonical(stored_gates)
    design = retained.get("design_audit", {})
    checks = {
        "compressed_part_hashes_exact": (expected_part_sha256 is None
                                         or part_hashes == expected_part_sha256),
        "reconstructed_source_hash_exact": (expected_source_sha256 is None
                                             or source_hash == expected_source_sha256),
        "all_480_runs_present": len(metrics) == 480 == expected_runs,
        "exact_six_arms": arm_set == sorted(REQUIRED_ARMS),
        "central_160_complete": (len(central) == 160 and all(
            item.get("completion_status") == "completed" for item in central)),
        "central_lifecycle_clean": all(
            int(item.get("lifecycle_violation_count", 0)) == 0 for item in central),
        "central_physical_rollback_failures_retained": sum(
            int(item.get("physical_rollback_failure_count", 0)) for item in central) == 3,
        "all_intermediate_trajectories_present": (
            len(trajectory_keys) == trajectories
            and len(trajectory_run_counts) == expected_runs
            and trajectory_completion_exact),
        "matched_baseline_observations": bool(baseline_groups) and all(
            len(value) == 1 and "" not in value for value in baseline_groups.values()),
        "matched_initial_states": bool(initial_groups) and all(
            len(value) == 1 for value in initial_groups.values()),
        "matched_disturbances": bool(trajectory_disturbances) and all(
            len(value) == 1 and "" not in value for value in trajectory_disturbances.values()),
        "matched_evaluators": bool(evaluator_groups) and all(
            len(value) == 1 for value in evaluator_groups.values()),
        "controller_truth_isolation": truth_accesses == 0,
        "timing_retained": timing_missing == 0,
        "stored_design_audit_claims_pass": all(bool(design.get(name)) for name in (
            "stationary_stage0", "held_out_native_qec_baseline",
            "matched_baseline_observations", "matched_disturbance_realizations",
            "synchronized_disturbance_onsets", "controller_truth_isolation",
            "matched_initial_physical_state", "matched_controller_state",
            "clone_mutability_isolated", "matched_evaluator_configuration")),
        "acceptance_gates_recalculate_exactly": gate_exact,
    }
    report = {
        "schema_version": "v2-reconstruction.v1",
        "passed": all(checks.values()),
        "source_parts": [str(path) for path in parts],
        "part_sha256": list(part_hashes),
        "reconstructed_source_sha256": source_hash,
        "checks": checks,
        "counts": {
            "metrics": len(metrics), "baselines": len(baselines),
            "trajectories": trajectories, "arms": arm_set,
            "central_runs": len(central), "truth_accesses": truth_accesses,
            "trajectory_rows_with_rollbacks": rollback_rows,
            "trajectory_rows_with_lifecycle_violations": lifecycle_rows,
        },
        "authoritative": retained.get("authoritative"),
        "accepted": retained.get("accepted"),
        "invalidity_reasons": retained.get("invalidity_reasons", ()),
        "acceptance_failure_reasons": retained.get("acceptance_failure_reasons", ()),
        "design_audit": design,
        "stored_gates": stored_gates,
        "recalculated_gates": recalculated_gate_rows,
        "original_report_hash": retained.get("report_hash"),
        "scientific_interpretation": (
            "This reconstructs and audits the immutable v2 evidence. It does not "
            "retroactively alter v2 outcomes or authorize a confirmatory rerun."),
    }
    json_path = output_dir/"v2_reconstruction.json"
    md_path = output_dir/"v2_reconstruction.md"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    lines = ["# Acceptance v2 lossless reconstruction", "",
             f"Overall audit: **{'PASS' if report['passed'] else 'FAIL'}**", "",
             f"Reconstructed SHA-256: `{source_hash}`", "", "## Checks", ""]
    lines.extend(f"- [{'x' if passed else ' '}] {name}" for name, passed in checks.items())
    lines.extend(["", "## Retained outcomes", "",
                  f"- Runs: {len(metrics)} across {len(arm_set)} arms",
                  f"- Interval trajectories: {trajectories}",
                  f"- Original authoritative: {retained.get('authoritative')}",
                  f"- Original accepted: {retained.get('accepted')}", "",
                  "The reconstruction is an integrity and design audit, not a new acquisition."])
    md_path.write_text("\n".join(lines)+"\n", encoding="utf-8")
    return report
