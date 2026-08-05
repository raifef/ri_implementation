"""Command-line entry points for the staged v9 amendment."""
from __future__ import annotations

import argparse
import json
from typing import Any, Callable

from .common import import_v8_contracts, read_json, artifact_root
from .reporting import report_root_cause_update, status, write_next_commands
from .studies import (
    corrected_fault_classification,
    freeze_held_out_protocol,
    plan_stage_a,
    plan_stage_b,
    plan_stage_c,
    run_held_out_validation,
    run_stage_a,
    run_stage_b,
    run_stage_c,
    select_controller,
)


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _mode_execute(description: str) -> tuple[str, bool]:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--mode", choices=("smoke", "development", "validation", "reference"), default="smoke")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    return args.mode, bool(args.execute)


def _report(path: str) -> None:
    target = artifact_root() / path
    if not target.is_file():
        raise RuntimeError(f"missing artifact: {path}")
    _print(read_json(target))


def import_v8_audits_main() -> None:
    _print(import_v8_contracts())


def correct_root_cause_classification_main() -> None:
    _print(corrected_fault_classification())


def plan_stage_a_main() -> None:
    mode, _ = _mode_execute("Plan the initial-scale feasibility stage")
    _print(plan_stage_a(mode))


def run_stage_a_main() -> None:
    mode, execute = _mode_execute("Run the initial-scale feasibility stage")
    _print(run_stage_a(mode=mode, execute=execute))


def report_stage_a_main() -> None:
    _report("stage_a_initial_scale/results.json")


def plan_stage_b_main() -> None:
    mode, _ = _mode_execute("Plan the entropy operationality stage")
    _print(plan_stage_b(mode))


def run_stage_b_main() -> None:
    mode, execute = _mode_execute("Run the entropy operationality stage")
    _print(run_stage_b(mode=mode, execute=execute))


def report_stage_b_main() -> None:
    _report("stage_b_entropy/results.json")


def plan_stage_c_main() -> None:
    mode, _ = _mode_execute("Plan the scale adaptation stage")
    _print(plan_stage_c(mode))


def run_stage_c_main() -> None:
    mode, execute = _mode_execute("Run the scale adaptation stage")
    _print(run_stage_c(mode=mode, execute=execute))


def report_stage_c_main() -> None:
    _report("stage_c_scale_learning_rate/results.json")


def freeze_held_out_protocol_main() -> None:
    _print(freeze_held_out_protocol())


def run_held_out_validation_main() -> None:
    mode, execute = _mode_execute("Run frozen held-out dynamic validation")
    if mode == "development":
        raise ValueError("held-out validation mode must be smoke, validation, or reference")
    _print(run_held_out_validation(mode=mode, execute=execute))


def select_controller_main() -> None:
    _print(select_controller())


def report_root_cause_update_main() -> None:
    _print(report_root_cause_update())
    _print(write_next_commands())


def status_main() -> None:
    _print(status())
