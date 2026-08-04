"""Console entry points for the pure Google-style v7 amendment."""
from __future__ import annotations

import argparse
import json
from typing import Any, Callable

from .controller import require_resolved_controller, resolve_production_controller
from .gates import write_scientific_gate_contract
from .hyperparameters import (run_exploration_study, run_hyperparameter_study,
                              write_hyperparameter_gate_contract)
from .natural import run_full_natural_ensemble, run_natural_ablation
from .replay_audit import run_replay_age_alignment
from .retention import run_final_recovery, run_final_scaling
from .scorecard import (freeze_certification, run_certification, run_development_scorecard,
                        validate_scientific_gates)
from .sine import validate_sine_estimator
from .snapshot import snapshot_v6, supersede_certification
from .timescale_studies import (command_cost, freeze_timescale_sine_protocol, run_long_step,
                                run_production_repaired_drift, run_timescale_sine,
                                run_timescale_strobe)


def _print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def _simple(function: Callable[[], dict[str, Any]]) -> Callable[[], None]:
    def main() -> None: _print(function())
    return main


snapshot_main=_simple(snapshot_v6)
supersede_main=_simple(supersede_certification)
resolve_main=_simple(resolve_production_controller)
sine_validation_main=_simple(validate_sine_estimator)
freeze_sine_main=_simple(freeze_timescale_sine_protocol)
production_drift_main=_simple(run_production_repaired_drift)
replay_main=_simple(run_replay_age_alignment)
hyperparameter_main=_simple(run_hyperparameter_study)
scorecard_main=_simple(run_development_scorecard)
freeze_certification_main=_simple(freeze_certification)


def scientific_gates_main()->None:
    _print({"contract":write_scientific_gate_contract(),"validation":validate_scientific_gates()})


def long_step_smoke_main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument("--epochs",type=int,default=96); args=parser.parse_args()
    _print(run_long_step(smoke=True,epochs=args.epochs))


def long_step_main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument("--epochs",type=int,default=5000); parser.add_argument("--execute",action="store_true"); args=parser.parse_args()
    from .config import canonical_hash, load_config
    _print({"preflight":{**command_cost(epochs=args.epochs*2,candidates=40,cycles=100000),"protocol_hash":canonical_hash(load_config("long_step_response.yaml"))}})
    _print(run_long_step(smoke=False,execute=args.execute,epochs=args.epochs))


def timescale_sine_smoke_main()->None: _print(run_timescale_sine(smoke=True))


def timescale_sine_main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument("--execute",action="store_true"); args=parser.parse_args()
    from .reporting import read_artifact
    protocol=read_artifact("timescale_matched_sine_protocol"); epochs=sum(int(row["horizon_epochs"]) for row in protocol["rows"])
    _print({"preflight":{**command_cost(epochs=epochs,candidates=40,cycles=100000),"protocol_hash":protocol["protocol_hash"]}})
    _print(run_timescale_sine(smoke=False,execute=args.execute))


def timescale_strobe_main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument("--execute",action="store_true"); args=parser.parse_args()
    from .config import canonical_hash, load_config
    from .reporting import read_artifact
    tau=float(read_artifact("timescale_matched_sine_protocol")["response_tau_epochs"]); config=load_config("timescale_matched_strobe.yaml")
    epochs=sum(max(1,round(float(r)*tau))*int(config["transitions"]) for r in config["dwell_tau_ratios"])
    _print({"preflight":{**command_cost(epochs=int(epochs),candidates=40,cycles=100000),"protocol_hash":canonical_hash(config)}})
    _print(run_timescale_strobe(execute=args.execute))


def natural_ablation_main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument("--execute",action="store_true"); parser.add_argument("--epochs",type=int,default=768); args=parser.parse_args()
    from .config import canonical_hash
    from .natural import FAMILIES
    _print({"preflight":{**command_cost(epochs=args.epochs*6*5,candidates=40,cycles=100000,controls=24),"protocol_hash":canonical_hash(FAMILIES)}})
    _print(run_natural_ablation(execute=args.execute,horizon=args.epochs))


def natural_ensemble_main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument("--execute",action="store_true"); parser.add_argument("--epochs",type=int,default=768); args=parser.parse_args()
    from .config import canonical_hash
    from .natural import FAMILIES
    _print({"preflight":{**command_cost(epochs=args.epochs*6,candidates=40,cycles=100000,controls=24),"protocol_hash":canonical_hash(FAMILIES)}})
    _print(run_full_natural_ensemble(execute=args.execute,horizon=args.epochs))


def exploration_main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument("--execute",action="store_true"); args=parser.parse_args()
    from .reporting import read_artifact
    protocol=read_artifact("timescale_matched_sine_protocol")
    _print({"preflight":{**command_cost(epochs=1,candidates=32,cycles=100000),"epoch_count":"derived from frozen slow-sine protocol per candidate","protocol_hash":protocol["protocol_hash"]}})
    _print(run_exploration_study(execute=args.execute))


def recovery_main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument("--execute",action="store_true"); parser.add_argument("--epochs",type=int,default=4000); args=parser.parse_args()
    from .config import canonical_hash
    _print({"preflight":{**command_cost(epochs=args.epochs*9,candidates=40,cycles=100000),"protocol_hash":canonical_hash({"severities":[.25,.45,.65],"realizations":3})}})
    _print(run_final_recovery(execute=args.execute,epochs=args.epochs))


def scaling_main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument("--execute",action="store_true"); parser.add_argument("--epochs",type=int,default=64); args=parser.parse_args()
    from .config import canonical_hash
    _print({"preflight":{**command_cost(epochs=args.epochs*7*3,candidates=40,cycles=100000,controls=38670),"protocol_hash":canonical_hash({"distances":[3,5,7,9,11,13,15],"distance15_controls":38670})}})
    _print(run_final_scaling(execute=args.execute,epochs=args.epochs))


def certification_main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument("--seed",type=int,required=True); parser.add_argument("--confirm",action="store_true"); parser.add_argument("--authorization-phrase"); args=parser.parse_args()
    _print(run_certification(seed=args.seed,confirm=args.confirm,authorization_phrase=args.authorization_phrase))
