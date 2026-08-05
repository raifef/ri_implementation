"""Command-line entry points for the staged v10 amendment."""
from __future__ import annotations

import argparse
import json
from typing import Any

from .common import artifact_root, import_audits, read_json
from .contracts import corrected_fault_contract
from .controller import (
    freeze_held_out,
    plan_scale_entropy,
    plan_temporal_validation,
    run_scale_entropy,
    run_temporal_validation,
)
from .decoder.closed_loop import (
    run_control_only,
    run_control_plus_decoder,
    run_decoder_steering,
    validate_decoder,
)
from .reporting import next_commands, report_decoder, root_cause_update, status
from .preflight import run_preflight
from .spectral import plan_natural_drift, run_natural_drift
from .step_response import analyse_step_response, plan_step_response, run_step_ablation, run_step_response


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _mode_execute(description: str, *, modes: tuple[str, ...]) -> tuple[str, bool]:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--mode", choices=modes, default=modes[0])
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    return str(args.mode), bool(args.execute)


def import_audits_main() -> None: _print(import_audits())
def correct_classifications_main() -> None: _print(corrected_fault_contract())
def preflight_main() -> None: _print(run_preflight())


def plan_scale_entropy_main() -> None:
    mode, _ = _mode_execute("Plan scale and entropy studies", modes=("smoke", "development"))
    _print(plan_scale_entropy(mode))


def run_scale_entropy_main() -> None:
    mode, execute = _mode_execute("Run scale and entropy studies", modes=("smoke", "development"))
    _print(run_scale_entropy(mode=mode, execute=execute))


def plan_temporal_validation_main() -> None:
    mode, _ = _mode_execute("Plan temporal validation", modes=("smoke", "validation", "reference"))
    _print(plan_temporal_validation(mode))


def run_temporal_validation_main() -> None:
    mode, execute = _mode_execute("Run temporal validation", modes=("smoke", "validation", "reference"))
    _print(run_temporal_validation(mode=mode, execute=execute))


def plan_natural_drift_main() -> None:
    mode, _ = _mode_execute("Plan natural-drift spectral acquisition", modes=("smoke", "reference"))
    _print(plan_natural_drift(mode))


def run_natural_drift_main() -> None:
    mode, execute = _mode_execute("Run natural-drift spectral acquisition", modes=("smoke", "reference"))
    _print(run_natural_drift(mode=mode, execute=execute))


def analyse_natural_drift_main() -> None:
    _print(read_json(artifact_root() / "natural_drift" / "report.json"))


def validate_decoder_main() -> None: _print(validate_decoder())
def run_control_only_main() -> None: _print(run_control_only())
def run_control_plus_decoder_main() -> None: _print(run_control_plus_decoder())
def run_decoder_steering_main() -> None: _print(run_decoder_steering())


def plan_step_response_main() -> None:
    mode, _ = _mode_execute("Plan injected-step acquisition", modes=("smoke", "reference"))
    _print(plan_step_response(mode))


def run_step_response_main() -> None:
    mode, execute = _mode_execute("Run injected-step acquisition", modes=("smoke", "reference"))
    _print(run_step_response(mode=mode, execute=execute))


def run_step_ablation_main() -> None:
    mode, execute = _mode_execute("Run causal step-response ablations", modes=("smoke", "reference"))
    _print(run_step_ablation(mode=mode, execute=execute))


def analyse_step_response_main() -> None: _print(analyse_step_response())
def freeze_held_out_main() -> None: _print(freeze_held_out())


def run_held_out_main() -> None:
    mode, execute = _mode_execute("Run frozen held-out controller validation", modes=("smoke", "validation", "reference"))
    _print(run_temporal_validation(mode=mode, execute=execute))


def report_main() -> None:
    _print(report_decoder())
    _print(root_cause_update())
    _print(next_commands())


def status_main() -> None: _print(status())
