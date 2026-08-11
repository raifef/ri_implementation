"""Fail-closed contracts for the distance-5, 924-coordinate step analogue."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any


SOURCE_UNSPECIFIED = "SOURCE_UNSPECIFIED_PREREGISTERED"


@dataclass(frozen=True)
class StepProtocol:
    profile: str
    distance: int = 5
    controls: int = 924
    candidates_per_epoch: int = 40
    cycles_per_candidate: int = 100_000
    epochs: int = 720
    onset_epoch: int = 60
    direction_coordinate: int = 0
    target_delta_normalized: float = 0.5
    seed: int = 91_301
    certification: bool = False

    def validate(self) -> None:
        if self.distance != 5 or self.controls != 924:
            raise ValueError("source analogue requires distance 5 and exactly 924 coordinates")
        if self.onset_epoch <= 0 or self.onset_epoch >= self.epochs:
            raise ValueError("step onset must be strictly inside the trace")
        if not 0 <= self.direction_coordinate < self.controls:
            raise ValueError("direction coordinate is outside the control inventory")
        if self.certification and (self.candidates_per_epoch != 40 or self.cycles_per_candidate != 100_000):
            raise ValueError("certification requires 40 candidates and 100,000 effective cycles per candidate")


def _hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def build_control_inventory(controls: int = 924, direction_coordinate: int = 0) -> dict[str, Any]:
    """Build a declared public analogue inventory without inventing a hardware map."""
    if controls != 924:
        raise ValueError("the declared source analogue inventory has 924 controls")
    rows = []
    families = ("xy_amplitude", "virtual_phase", "cz_detuning", "reset_amplitude")
    for index in range(controls):
        rows.append({
            "index": index,
            "name": f"normalized_control_{index:03d}",
            "family": "xy_amplitude" if index == direction_coordinate else families[index % len(families)],
            "unit": "empirically_normalized_control_coordinate",
            "is_injected_direction": index == direction_coordinate,
            "mapping_status": SOURCE_UNSPECIFIED,
        })
    payload = {
        "distance": 5,
        "control_count": controls,
        "injected_direction_count": 1,
        "hardware_mapping_available": False,
        "limitation": "The proprietary 924-parameter pulse inventory and detector mask are not public.",
        "coordinates": rows,
    }
    payload["inventory_hash"] = _hash(payload)
    return payload


def controller_observation(policy_mean: list[float], detector_rewards: list[float]) -> dict[str, Any]:
    """The only permitted controller input; target and direction are deliberately absent."""
    return {"policy_mean": list(policy_mean), "detector_rewards": list(detector_rewards)}


def build_run_plan(protocol: StepProtocol) -> dict[str, Any]:
    protocol.validate()
    effective_cycles = protocol.epochs * protocol.candidates_per_epoch * protocol.cycles_per_candidate
    payload = {
        "schema": "google-pure-step-response-130.v1",
        "protocol": asdict(protocol),
        "source_budget": {
            "candidates_per_epoch": protocol.candidates_per_epoch,
            "effective_cycles_per_candidate": protocol.cycles_per_candidate,
            "total_training_effective_cycles": effective_cycles,
        },
        "experiment_family": "step_response_injected_xy_amplitude_drift",
        "explicit_exclusions": ["figure5a_metric", "randomized_recovery", "policy_spoil"],
        "controller_target_access": False,
        "controller_direction_access": False,
        "conditions": ["fixed_no_drift", "fixed_step", "learned_step"],
        "common_random_numbers": True,
        "checkpoint_boundary": "candidate_batch",
        "paper_comparable": False,
        "launch_automatically": False,
        "blocking_reasons": [
            "public analogue has no proprietary hardware control-to-detector map",
            "held-out reference-budget response-time evidence has not been acquired",
        ],
    }
    payload["plan_hash"] = _hash(payload)
    return payload
