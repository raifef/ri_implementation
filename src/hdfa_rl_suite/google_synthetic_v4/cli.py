"""Console commands for the gated v4 synthetic study."""
from __future__ import annotations

import argparse
import json
from typing import Any, Callable

from .protocol import (build_plant_ensemble, estimate_cost, freeze_certification, freeze_synthetic_splits,
                       stability_metric_contract, validate_ppo)
from .studies import (decompose_stability, run_amendment_study, run_certification, run_convergence_scaling,
                      run_development_scorecard, run_randomized_recovery, run_steering_phase)


def _show(value: Any) -> None:
    print(json.dumps(value,indent=2,sort_keys=True,allow_nan=False))


def _epochs(description: str, default: int) -> int:
    parser=argparse.ArgumentParser(description=description)
    parser.add_argument("--epochs",type=int,default=default)
    args=parser.parse_args()
    if args.epochs<8:
        parser.error("--epochs must be at least 8")
    return args.epochs


def build_main() -> None:
    _show(estimate_cost("hdfa-google-v4-build-plant-ensemble",epochs=0,plants=24))
    _show(build_plant_ensemble())


def splits_main() -> None:
    _show(estimate_cost("hdfa-google-v4-freeze-synthetic-splits",epochs=0,plants=24))
    _show(freeze_synthetic_splits())


def ppo_main() -> None:
    _show(estimate_cost("hdfa-google-v4-validate-ppo",epochs=100,plants=1))
    _show(validate_ppo())


def decompose_main() -> None:
    epochs=_epochs("Pre-amendment stability decomposition",96)
    _show(estimate_cost("hdfa-google-v4-decompose-stability",epochs=epochs,plants=12))
    _show(decompose_stability(epochs=epochs))


def metric_main() -> None:
    _show(estimate_cost("hdfa-google-v4-validate-stability-metric",epochs=0,plants=0))
    _show(stability_metric_contract())


def amendment_main() -> None:
    epochs=_epochs("One-variable amendment study",96)
    _show(estimate_cost("hdfa-google-v4-run-amendment-study",epochs=epochs,plants=5,cells=8))
    _show(run_amendment_study(epochs=epochs))


def recovery_main() -> None:
    epochs=_epochs("Recovery by frozen spoil severity",180)
    _show(estimate_cost("hdfa-google-v4-run-randomized-recovery",epochs=epochs,plants=2,cells=3))
    _show(run_randomized_recovery(epochs=epochs))


def steering_main() -> None:
    epochs=_epochs("Dense steering phase",96)
    _show(estimate_cost("hdfa-google-v4-run-steering-phase",epochs=epochs,plants=2,cells=21))
    _show(run_steering_phase(epochs=epochs))


def scaling_main() -> None:
    epochs=_epochs("Sparse convergence scaling",28)
    _show(estimate_cost("hdfa-google-v4-run-convergence-scaling",epochs=epochs,plants=42))
    _show(run_convergence_scaling(epochs=epochs))


def scorecard_main() -> None:
    epochs=_epochs("Development validation scorecard",120)
    _show(estimate_cost("hdfa-google-v4-run-development-scorecard",epochs=epochs,plants=12))
    _show(run_development_scorecard(epochs=epochs))


def freeze_main() -> None:
    _show(estimate_cost("hdfa-google-v4-freeze-certification",epochs=0,plants=12))
    _show(freeze_certification())


def certification_main() -> None:
    parser=argparse.ArgumentParser(description="One-shot locked synthetic certification")
    parser.add_argument("--epochs",type=int,default=480)
    parser.add_argument("--confirm-open-locked-seeds",action="store_true")
    args=parser.parse_args()
    _show(estimate_cost("hdfa-google-v4-run-certification",epochs=args.epochs,plants=12,certification=True))
    _show(run_certification(epochs=args.epochs,confirm=args.confirm_open_locked_seeds))
