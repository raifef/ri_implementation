"""Console entry points for the pure Google-style v6 repair program."""
from __future__ import annotations

import argparse
import json
from typing import Any, Callable

from .audits import AUDIT_FUNCTIONS
from .snapshot import migrate_v5_metric_schema, snapshot_v5
from .studies import (
    freeze_certification, freeze_repaired_drift_protocol, run_certification, run_development_scorecard,
    run_exploration_calibration, run_hyperparameter_study, run_natural_drift_retention,
    run_recovery_retention, run_repaired_drift_unchanged, run_scaling_retention,
    run_sine_bandwidth, run_static_validation,
)


def _print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def _simple(function: Callable[[], dict[str, Any]]) -> Callable[[], None]:
    def main() -> None:
        _print(function())
    return main


snapshot_v5_main = _simple(snapshot_v5)
migrate_v5_metric_schema_main = _simple(migrate_v5_metric_schema)
source_main = _simple(AUDIT_FUNCTIONS["audit-source-compliance"])
gaussian_main = _simple(AUDIT_FUNCTIONS["validate-gaussian-scores"])
ratios_main = _simple(AUDIT_FUNCTIONS["audit-local-ratios"])
ppo_main = _simple(AUDIT_FUNCTIONS["audit-ppo-clipping"])
entropy_main = _simple(AUDIT_FUNCTIONS["audit-entropy-normalization"])
aggregation_main = _simple(AUDIT_FUNCTIONS["audit-objective-aggregation"])
baseline_main = _simple(AUDIT_FUNCTIONS["audit-baseline"])
replay_main = _simple(AUDIT_FUNCTIONS["audit-replay"])
units_main = _simple(AUDIT_FUNCTIONS["audit-units"])
quadratic_main = _simple(AUDIT_FUNCTIONS["validate-quadratic-gradients"])
damage_main = _simple(AUDIT_FUNCTIONS["audit-candidate-damage"])
freeze_protocol_main = _simple(freeze_repaired_drift_protocol)
scorecard_main = _simple(run_development_scorecard)
freeze_certification_main = _simple(freeze_certification)


def _epochs(function: Callable[..., dict[str, Any]], default: int) -> Callable[[], None]:
    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--epochs", type=int, default=default)
        parser.add_argument("--seed", type=int, default=None)
        args = parser.parse_args()
        kwargs: dict[str, Any] = {"epochs": args.epochs}
        if args.seed is not None:
            kwargs["seed"] = args.seed
        _print(function(**kwargs))
    return main


repaired_unchanged_main = _epochs(run_repaired_drift_unchanged, 48)
sine_bandwidth_main = _epochs(run_sine_bandwidth, 72)
natural_retention_main = _epochs(run_natural_drift_retention, 96)
exploration_main = _epochs(run_exploration_calibration, 40)
hyperparameter_main = _epochs(run_hyperparameter_study, 36)
static_main = _epochs(run_static_validation, 64)
scaling_main = _epochs(run_scaling_retention, 32)
recovery_main = _epochs(run_recovery_retention, 4000)


def certification_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--authorization-phrase")
    args = parser.parse_args()
    _print(run_certification(seed=args.seed, confirm=args.confirm, authorization_phrase=args.authorization_phrase))
