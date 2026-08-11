"""Primary-method and decoder identity contracts."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any

SPARSE_BLOSSOM = "SPARSE_BLOSSOM_CORRELATED_MATCHING_TWO_STEP_REWEIGHTING"
PYMATCHING_PROXY = "NON_EQUIVALENT_PYMATCHING_PROXY"

@dataclass(frozen=True)
class DecoderIdentity:
    backend: str
    version: str
    graph_hash: str
    dem_hash: str
    boundary_hash: str
    implementation_hash: str

@dataclass(frozen=True)
class PriorSteeringProtocol:
    decoder: DecoderIdentity
    parameter_count: int
    candidates_per_epoch: int = 70
    training_epochs: int = 220
    policy_steps_per_epoch: int = 20
    shots_per_epoch: int = 3_500
    initial_log_sigma: float = .3
    learning_rate: float = .001
    importance_clip: float = .4
    gradient_clip: float = .1
    entropy_coefficient: float = .01
    physical_controls_frozen: bool = True
    paper_mode: bool = True

    def validate(self) -> None:
        if not self.physical_controls_frozen:
            raise ValueError("decoder-prior steering is offline; physical controls must be frozen")
        if self.parameter_count <= 0:
            raise ValueError("the DEM prior needs at least one log-hyperedge parameter")
        if self.paper_mode:
            require_sparse_blossom(self.decoder)
            expected = (70, 220, 20, 3_500, .3, .001, .4, .1, .01)
            actual = (self.candidates_per_epoch, self.training_epochs, self.policy_steps_per_epoch,
                      self.shots_per_epoch, self.initial_log_sigma, self.learning_rate,
                      self.importance_clip, self.gradient_clip, self.entropy_coefficient)
            if actual != expected:
                raise ValueError("paper mode requires the published surface-code optimization profile")

def require_sparse_blossom(identity: DecoderIdentity) -> None:
    if identity.backend != SPARSE_BLOSSOM:
        raise RuntimeError("paper mode requires verified Sparse Blossom correlated matching; PyMatching is a non-equivalent proxy")
    fields = ("version", "graph_hash", "dem_hash", "boundary_hash", "implementation_hash")
    missing = [name for name in fields if not getattr(identity, name)]
    if missing:
        raise RuntimeError(f"unverified decoder identity fields: {', '.join(missing)}")

def public_benchmark_contract(protocol: PriorSteeringProtocol) -> dict[str, Any]:
    protocol.validate()
    payload = {"schema": "offline-dem-prior-steering.v1",
        "primary_method": {"doi": "10.1103/PhysRevLett.133.150603", "arxiv": "2406.02700"},
        "protocol": asdict(protocol),
        "required_sequence": ["reproduce_2024_public_benchmark", "freeze_prior", "run_held_out_four_arm_evaluation"],
        "arms": ["fixed_controls_fixed_prior", "learned_controls_fixed_prior", "fixed_controls_steered_prior", "learned_controls_steered_prior"],
        "same_shots_within_each_physical_control_pair": True,
        "logical_outcomes_allowed_in_physical_reward": False,
        "live_controller_coupling": False, "paper_comparable": False, "launch_automatically": False}
    payload["contract_hash"] = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return payload

def non_equivalent_proxy_identity(version: str, *, graph_hash: str, dem_hash: str, boundary_hash: str) -> DecoderIdentity:
    return DecoderIdentity(PYMATCHING_PROXY, version, graph_hash, dem_hash, boundary_hash,
                           sha256(f"pymatching:{version}".encode()).hexdigest())
