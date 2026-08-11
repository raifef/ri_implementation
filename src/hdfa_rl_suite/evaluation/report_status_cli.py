"""Validate the terminal status of an authoritative effectiveness report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


def report_exit_code(report: Any) -> int:
    """Map a serialized report to the benchmark's public exit-code contract."""
    if not isinstance(report, dict):
        return 3
    authoritative = report.get("authoritative")
    accepted = report.get("accepted")
    if not isinstance(authoritative, bool) or not isinstance(accepted, bool):
        return 3
    if not authoritative:
        return 3
    return 0 if accepted else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate an effectiveness report and reproduce its 0/2/3 exit status."
    )
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--require-current-runtime", action="store_true",
        help="also require the report package and simulator versions to match this installation",
    )
    args = parser.parse_args(argv)
    try:
        with args.report.open(encoding="utf-8") as stream:
            report = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"INVALID: cannot read effectiveness report: {exc}", file=sys.stderr)
        return 3

    status = report_exit_code(report)
    if args.require_current_runtime:
        from hdfa_rl_suite import __version__
        from hdfa_rl_suite.simulator import SIMULATOR_VERSION

        provenance = report.get("provenance") if isinstance(report, dict) else None
        provenance = provenance if isinstance(provenance, dict) else {}
        observed = (provenance.get("package_version"),
                    provenance.get("simulator_version"))
        required = (__version__, SIMULATOR_VERSION)
        if observed != required:
            print(
                "INVALID: report runtime mismatch: "
                f"observed package/simulator={observed}, required={required}",
                file=sys.stderr,
            )
            status = 3
    print(
        f"authoritative={report.get('authoritative')} "
        f"accepted={report.get('accepted')}"
    )
    if status == 3 and (not isinstance(report, dict)
                        or report.get("authoritative") is not True):
        print("INVALID: report lacks a valid authoritative/accepted status", file=sys.stderr)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
