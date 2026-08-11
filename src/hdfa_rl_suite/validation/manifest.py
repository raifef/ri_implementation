"""Signed-by-content preflight manifest required by authoritative benchmarks."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping

from hdfa_rl_suite.common import deterministic_hash
from hdfa_rl_suite.common import TimingEnvironment
from hdfa_rl_suite import __version__
from hdfa_rl_suite.simulator import SIMULATOR_VERSION

from .common import ValidationReport
from .controller_sanity import CONTROLLER_VERSION
from .preflight import source_tree_hash


PREFLIGHT_MANIFEST_SCHEMA = "benchmark-preflight-manifest.v2"


@dataclass(frozen=True)
class PreflightManifest:
    schema_version: str
    passed: bool
    source_tree_hash: str
    benchmark_configuration_hash: str
    simulator_version: str
    controller_version: str
    validation_timestamp_utc: str
    maximum_age_hours: float
    minimum_candidate_cycles: int
    passed_check_ids: tuple[str, ...]
    thresholds: Mapping[str, object]
    result_hashes: Mapping[str, str]
    preflight_report_hash: str
    timing_environment_hash: str
    timing_environment: Mapping[str, object]
    manifest_hash: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


REQUIRED_PREFLIGHT_CHECKS = frozenset({
    "no_disturbance_plant_sanity",
    "fixed_oracle_step_sanity",
    "periodic_calibration_ordering",
    "disturbance_persistence_and_matched_cloning",
    "logical_detector_shared_state",
    "full_rl_analytic_convergence",
    "full_rl_static_detector_convergence",
    "positive_gradient_alignment",
    "calibrated_start_no_regression",
    "sample_budget_adequacy",
    "mean_exploration_separation",
    "policy_lifecycle_transactions",
    "report_schema_and_evidence_layers",
    "development_baseline_cohort",
    "stage2_6_numerical_equivalence",
    "failure_injection_coverage",
    "post_comparison_recovery_regressions",
    "rollback_fault_separation",
    "ou_nested_development_tail_latency",
    "compute_accounting_and_rmst",
    "compute_evidence_fault_matrix",
})


def build_preflight_manifest(report: ValidationReport,
                             benchmark_configuration_hash: str, *,
                             maximum_age_hours: float = 24.0) -> PreflightManifest:
    if maximum_age_hours <= 0:
        raise ValueError("preflight maximum age must be positive")
    passed_ids = tuple(sorted(item.check_id for item in report.checks if item.passed))
    missing = REQUIRED_PREFLIGHT_CHECKS - set(passed_ids)
    passed = bool(report.passed and not missing and benchmark_configuration_hash)
    manifest = PreflightManifest(
        PREFLIGHT_MANIFEST_SCHEMA, passed,
        str(report.metadata.get("source_tree_hash", "")),
        benchmark_configuration_hash,
        str(report.metadata.get("simulator_version", "")),
        str(report.metadata.get("controller_version", "")),
        str(report.metadata.get("validation_timestamp_utc", "")),
        float(maximum_age_hours),
        int(report.metadata.get("selected_validated_budget") or
            report.metadata.get("selected_validated_reduced_budget") or 0),
        passed_ids,
        dict(report.metadata.get("thresholds", {})),
        dict(report.metadata.get("result_hashes", {})),
        report.report_hash,
        str(report.metadata.get("timing_environment_hash", "")),
        dict(report.metadata.get("timing_environment", {})), "",
    )
    payload = manifest.to_dict()
    payload.pop("manifest_hash")
    return PreflightManifest(**{**payload, "manifest_hash": deterministic_hash(payload)})


def write_preflight_manifest(manifest: PreflightManifest,
                             path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True),
                      encoding="utf-8")
    return target


def load_preflight_manifest(path: str | Path) -> PreflightManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return PreflightManifest(**payload)


def validate_preflight_manifest(manifest: PreflightManifest, *,
                                expected_configuration_hash: str,
                                now: datetime | None = None) -> tuple[str, ...]:
    reasons: list[str] = []
    if manifest.schema_version != PREFLIGHT_MANIFEST_SCHEMA:
        reasons.append("unsupported preflight manifest schema")
    payload = manifest.to_dict()
    observed_hash = payload.pop("manifest_hash")
    if deterministic_hash(payload) != observed_hash:
        reasons.append("preflight manifest content hash mismatch")
    if not manifest.passed:
        reasons.append("preflight manifest is not passing")
    if manifest.source_tree_hash != source_tree_hash():
        reasons.append("preflight source-tree hash is stale")
    if manifest.benchmark_configuration_hash != expected_configuration_hash:
        reasons.append("preflight benchmark configuration hash mismatch")
    if manifest.simulator_version != SIMULATOR_VERSION:
        reasons.append("preflight simulator version mismatch")
    if manifest.controller_version != CONTROLLER_VERSION:
        reasons.append("preflight controller version mismatch")
    current_timing = TimingEnvironment.capture(__version__)
    if (manifest.timing_environment_hash != current_timing.environment_hash
            or manifest.timing_environment != current_timing.__dict__):
        reasons.append("preflight timing environment hash/configuration mismatch")
    if REQUIRED_PREFLIGHT_CHECKS - set(manifest.passed_check_ids):
        reasons.append("preflight manifest lacks one or more required scientific gates")
    if manifest.minimum_candidate_cycles <= 0:
        reasons.append("preflight manifest lacks a validated candidate-cycle floor")
    if not manifest.preflight_report_hash or not manifest.result_hashes:
        reasons.append("preflight result hashes are incomplete")
    try:
        timestamp = datetime.fromisoformat(manifest.validation_timestamp_utc)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        current = now or datetime.now(timezone.utc)
        age_hours = (current.astimezone(timezone.utc)
                     - timestamp.astimezone(timezone.utc)).total_seconds()/3600
        if age_hours < -1e-6 or age_hours > manifest.maximum_age_hours:
            reasons.append("preflight validation timestamp is stale or in the future")
    except ValueError:
        reasons.append("preflight validation timestamp is invalid")
    return tuple(reasons)
