"""Shared records and artifact writers for scientific validation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Mapping, Sequence

from hdfa_rl_suite.common import deterministic_hash


@dataclass(frozen=True)
class ValidationCheck:
    check_id: str
    passed: bool
    observed: object
    criterion: str
    details: str


@dataclass(frozen=True)
class ValidationReport:
    schema_version: str
    validation_type: str
    passed: bool
    checks: tuple[ValidationCheck, ...]
    trajectories: tuple[Mapping[str, object], ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)
    report_hash: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def finalize_report(report: ValidationReport) -> ValidationReport:
    payload = report.to_dict()
    payload.pop("report_hash", None)

    def stable_scientific_payload(value):
        if isinstance(value, dict):
            return {key: stable_scientific_payload(item) for key, item in value.items()
                    if key not in {"runtime_s", "validation_timestamp_utc"}}
        if isinstance(value, (list, tuple)):
            return [stable_scientific_payload(item) for item in value]
        return value

    return ValidationReport(
        report.schema_version, report.validation_type, report.passed,
        report.checks, report.trajectories, report.metadata,
        deterministic_hash(stable_scientific_payload(payload)),
    )


def _markdown(report: ValidationReport) -> str:
    status = "PASS" if report.passed else "FAIL"
    lines = [
        f"# {report.validation_type.replace('_', ' ').title()}", "",
        f"Overall status: **{status}**", "",
        "| Check | Status | Observed | Criterion |",
        "| --- | --- | --- | --- |",
    ]
    for check in report.checks:
        observed = json.dumps(check.observed, sort_keys=True, default=str).replace("|", "\\|")
        criterion = check.criterion.replace("|", "\\|")
        lines.append(f"| `{check.check_id}` | {'PASS' if check.passed else 'FAIL'} | {observed} | {criterion} |")
    lines.extend(["", "## Interpretation", ""])
    for check in report.checks:
        lines.append(f"- **{check.check_id}:** {check.details}")
    lines.extend(["", "## Provenance", ""])
    for key, value in sorted(report.metadata.items()):
        lines.append(f"- `{key}`: `{json.dumps(value, sort_keys=True, default=str)}`")
    lines.extend(["", f"Report hash: `{report.report_hash}`", ""])
    return "\n".join(lines)


def write_report(report: ValidationReport, output_dir: str | Path,
                 stem: str) -> tuple[Path, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{stem}.json"
    markdown_path = directory / f"{stem}.md"
    json_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True, default=str), encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def all_passed(checks: Sequence[ValidationCheck]) -> bool:
    return bool(checks) and all(check.passed for check in checks)
