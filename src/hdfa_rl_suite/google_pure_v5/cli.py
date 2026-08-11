"""Fail-closed console commands for the pure v5 reproduction."""
from __future__ import annotations

import argparse
import json
from typing import Any, Callable

from .accounting import command_estimate
from .config import paper_scale
from .injected_drift_test import run_injected_drift, run_step_response
from .natural_drift_spectral_test import run_natural_drift_spectral
from .protocol import audit_test_separation, source_compliance_map
from .studies import (freeze_certification, run_certification, run_convergence_scaling,
                      run_development_scorecard, run_randomized_recovery, run_steering_phase)
from .validation import audit_baseline, run_static_tests, validate_algorithm


def _show(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _run(command: str, function: Callable[..., Any], *, epochs: int = 0, experiments: int = 0,
         certification: bool = False, scaling: bool = False, kwargs: dict[str, Any] | None = None) -> None:
    _show(command_estimate(command, epochs=epochs, experiments=experiments, paper_config=paper_scale(), certification=certification, scaling=scaling))
    _show(function(**(kwargs or {})))


def _epochs(description: str, default: int, minimum: int) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--epochs", type=int, default=default)
    args = parser.parse_args()
    if args.epochs < minimum:
        parser.error(f"--epochs must be at least {minimum}")
    return args.epochs


def source_main() -> None:
    _run("hdfa-google-v5-audit-source-compliance", source_compliance_map)


def baseline_main() -> None:
    _run("hdfa-google-v5-audit-baseline", audit_baseline)


def algorithm_main() -> None:
    _run("hdfa-google-v5-validate-algorithm", validate_algorithm)


def static_main() -> None:
    epochs = _epochs("Static optimization and no-drift gates", 180, 80)
    _run("hdfa-google-v5-run-static-tests", run_static_tests, epochs=epochs, experiments=3, kwargs={"epochs": epochs})


def injected_main() -> None:
    epochs = _epochs("Injected-drift stability", 360, 120)
    _run("hdfa-google-v5-run-injected-drift", run_injected_drift, epochs=epochs, experiments=6, kwargs={"epochs": epochs})


def natural_main() -> None:
    epochs = _epochs("Natural-drift spectral suppression", 768, 256)
    _run("hdfa-google-v5-run-natural-drift-spectral", run_natural_drift_spectral, epochs=epochs, experiments=7, kwargs={"epochs": epochs})


def separation_main() -> None:
    _run("hdfa-google-v5-audit-test-separation", audit_test_separation)


def step_main() -> None:
    _run("hdfa-google-v5-run-step-response", run_step_response)


def steering_main() -> None:
    epochs = _epochs("Fixed-entropy steering phase diagram", 300, 180)
    _run("hdfa-google-v5-run-steering-phase", run_steering_phase, epochs=epochs, experiments=30, kwargs={"epochs": epochs})


def recovery_main() -> None:
    epochs = _epochs("Frozen-severity randomized-policy recovery", 1000, 200)
    _run("hdfa-google-v5-run-randomized-recovery", run_randomized_recovery, epochs=epochs, experiments=3, kwargs={"epochs": epochs})


def scaling_main() -> None:
    epochs = _epochs("Actual sparse convergence scaling", 16, 8)
    _run("hdfa-google-v5-run-convergence-scaling", run_convergence_scaling, epochs=epochs, experiments=14, scaling=True, kwargs={"epochs": epochs})


def scorecard_main() -> None:
    _run("hdfa-google-v5-run-development-scorecard", run_development_scorecard)


def freeze_main() -> None:
    _run("hdfa-google-v5-freeze-certification", freeze_certification)


def certification_main() -> None:
    parser = argparse.ArgumentParser(description="One-shot locked pure-v5 certification opening")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--confirm-open-locked-seeds", action="store_true")
    args = parser.parse_args()
    _run("hdfa-google-v5-run-certification", run_certification, epochs=args.epochs, experiments=12, certification=True,
         kwargs={"confirm": args.confirm_open_locked_seeds, "epochs": args.epochs})
