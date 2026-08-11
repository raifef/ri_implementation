"""Source, budget, mode, and evidence contracts for Figure 5a."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "google-pure-source-exact-figure5a.v1"
SOURCE_EPOCHS = 1000
SOURCE_CANDIDATES_PER_EPOCH = 50
SOURCE_QEC_CYCLES_PER_CANDIDATE = 36_000
SOURCE_CANDIDATE_QEC_CYCLES = 1_800_000_000
SOURCE_ENTROPY_ANCHORS = (0.001, 0.01, 0.1)
DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"
DIAGNOSTIC_STREAM_ACQUISITION_CONTRACT = "figure5a-finite-shot-epoch-aggregate.v1"


class SourceIdentifiability(StrEnum):
    SOURCE_LITERAL = "SOURCE_LITERAL"
    SOURCE_DERIVED = "SOURCE_DERIVED"
    SOURCE_REFERENCED_PRIMARY_METHOD = "SOURCE_REFERENCED_PRIMARY_METHOD"
    SOURCE_UNSPECIFIED_PREREGISTERED = "SOURCE_UNSPECIFIED_PREREGISTERED"
    NOT_PUBLICLY_IDENTIFIABLE = "NOT_PUBLICLY_IDENTIFIABLE"


class AcquisitionMode(StrEnum):
    SMOKE = "smoke"
    VALIDATION = "validation"
    DYNAMIC_VALIDATION = "dynamic_validation"
    REFERENCE = "reference"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_hash(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class Figure5aProtocol:
    mode: AcquisitionMode
    epochs: int
    candidates_per_epoch: int
    qec_cycles_per_candidate: int
    circuit_rounds: int

    def __post_init__(self) -> None:
        if min(self.epochs, self.candidates_per_epoch, self.qec_cycles_per_candidate,
               self.circuit_rounds) <= 0:
            raise ValueError("all acquisition budgets must be positive")
        if self.qec_cycles_per_candidate % self.circuit_rounds:
            raise ValueError("QEC cycles must divide exactly into whole Stim shots")
        exact = (self.epochs, self.candidates_per_epoch, self.qec_cycles_per_candidate) == (
            SOURCE_EPOCHS, SOURCE_CANDIDATES_PER_EPOCH, SOURCE_QEC_CYCLES_PER_CANDIDATE)
        if self.mode == AcquisitionMode.REFERENCE and not exact:
            raise ValueError("reference mode requires exactly 1000 x 50 x 36000")
        if self.mode != AcquisitionMode.REFERENCE and exact:
            raise ValueError("an exact source budget cannot be watermarked as smoke or validation")

    @property
    def candidate_qec_cycles(self) -> int:
        return self.epochs * self.candidates_per_epoch * self.qec_cycles_per_candidate

    @property
    def four_stream_qec_cycles(self) -> int:
        return 4 * self.candidate_qec_cycles

    @property
    def shots_per_policy(self) -> int:
        return self.qec_cycles_per_candidate // self.circuit_rounds

    @property
    def protocol_hash(self) -> str:
        return canonical_hash(asdict(self))

    def assert_reference(self) -> None:
        if self.mode != AcquisitionMode.REFERENCE or self.candidate_qec_cycles != SOURCE_CANDIDATE_QEC_CYCLES:
            raise RuntimeError("artifact is permanently non-reference: exact source budget not executed")


def ratio_from_raw_counts(stochastic: int, fixed: int, optimal: int) -> dict[str, float]:
    denominator = int(optimal) - int(fixed)
    if denominator == 0:
        raise ValueError("fixed and optimal counts must differ")
    source = (int(stochastic) - int(fixed)) / denominator
    positive_cost = (int(fixed) - int(stochastic)) / (int(fixed) - int(optimal))
    if source != positive_cost:
        raise AssertionError("source and positive-cost ratios are not algebraically identical")
    return {"source_ratio": float(source), "positive_cost_ratio": float(positive_cost)}


def build_source_contract() -> dict[str, Any]:
    literal = SourceIdentifiability.SOURCE_LITERAL
    derived = SourceIdentifiability.SOURCE_DERIVED
    unspecified = SourceIdentifiability.SOURCE_UNSPECIFIED_PREREGISTERED
    fields = [
        {"field": "plant", "value": "distance-3 surface-code memory circuit in Stim", "status": literal,
         "source": "Supplement VI.A"},
        {"field": "inventory", "value": "17 one-qubit plus 24 two-qubit gate parameters = 41", "status": literal,
         "source": "Supplement VI.A and Eq. (7) at d=3, P=1"},
        {"field": "shared_optimum", "value": "sin(2*pi*f*t) in every coordinate", "status": literal,
         "source": "Supplement VI.A"},
        {"field": "gate_error", "value": "epsilon_tilde_i + Omega_i*(p_i-p_opt_i(t))^2", "status": literal,
         "source": "Supplement VI.A"},
        {"field": "policy_coordinate", "value": "p sampled directly from diagonal Gaussian and applied to plant",
         "status": literal, "source": "Supplement Eqs. (10)-(12), VI.A, and Algorithm 1"},
        {"field": "budget", "value": "1000 epochs x 50 candidates x 36000 QEC cycles = 1.8e9 candidate cycles",
         "status": literal, "source": "Supplement VI.A"},
        {"field": "policy_streams", "value": "fixed, instantaneous optimum, stochastic candidates, learned mean",
         "status": literal, "source": "Supplement VI.A"},
        {"field": "diagnostic_stream_batching",
         "value": "fixed, optimum, and learned-mean controls are each sampled once per epoch with K times the per-candidate shots; stochastic rewards remain K separate acquisitions",
         "status": derived,
         "source": "iid finite-shot aggregation equivalence for controls constant within an epoch",
         "contract": DIAGNOSTIC_STREAM_ACQUISITION_CONTRACT},
        {"field": "performance_ratio", "value": "(N_stochastic-N_fixed)/(N_optimal-N_fixed)",
         "status": literal, "source": "Supplement VI.A"},
        {"field": "entropy_anchors", "value": list(SOURCE_ENTROPY_ANCHORS), "status": literal,
         "source": "Supplement VI.A and Fig. S8"},
        {"field": "slow_anchor_frequency", "value": 1 / 1000, "status": literal,
         "source": "Fig. S8 caption"},
        {"field": "steerability_order", "value": "about 1/(150 epochs)", "status": literal,
         "source": "Supplement VI.A"},
        {"field": "epsilon_and_omega_distributions", "value": "frozen configuration ensemble",
         "status": unspecified, "source": "absent from Supplement and Zenodo release inventory"},
        {"field": "Stim_rounds_to_QEC_cycles_mapping", "value": "primary clean-room choice T=25 with preregistered T=5/10/25/50 sensitivity analysis and exact integer shots",
         "status": unspecified, "source": "not stated publicly for Figure 5a"},
        {"field": "gradient_clipping", "value": "per-component and joint global-L2 nuisance variants",
         "status": unspecified,
         "source": "the supplement names clipping magnitude as a hyperparameter but does not publish geometry or magnitude"},
        {"field": "optimizer_and_loss_weights", "value": "frozen configuration", "status": unspecified,
         "source": "not numerically published"},
        {"field": "original_simulation_code", "value": None,
         "status": SourceIdentifiability.NOT_PUBLICLY_IDENTIFIABLE,
         "source": "Nature code-availability statement"},
    ]
    payload = {"schema_version": SCHEMA_VERSION, "paper_doi": "10.1038/s41586-026-10759-2",
               "clean_room": True, "old_six_dimensional_plant": DIAGNOSTIC_ONLY, "fields": fields}
    payload["source_contract_hash"] = canonical_hash(payload)
    return payload
