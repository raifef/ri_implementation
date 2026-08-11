"""Typed source and evidence contracts for Supplement Section III / Figure S5."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SOURCE_ESTIMATOR_STATUS = "SOURCE_ESTIMATOR_EXACT; HARDWARE_NUMERICAL_RESULT_NOT_PUBLICLY_REPRODUCIBLE"
ALLOWED_STREAM = "DECODED_LER_EVALUATION"
FORBIDDEN_STREAM = "SAMPLED_CANDIDATE"


def canonical_hash(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


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
class SourceDFTConfig:
    cadence_epochs: int = 5
    warmup_epoch: int = 150
    shared_grid_points: int = 256
    gaussian_smoothing_sigma_bins: float = 5.0
    power_normalization: str = "ABS_DFT_SQUARED_OVER_N_SQUARED"
    filter_ratio: str = "LEARNED_OVER_FIXED"

    def __post_init__(self) -> None:
        if self.cadence_epochs != 5 or self.warmup_epoch != 150:
            raise ValueError("source panel requires five-epoch cadence and epoch-150 normalization")
        if self.shared_grid_points < 8 or self.gaussian_smoothing_sigma_bins <= 0:
            raise ValueError("invalid grid or smoothing configuration")
        if self.power_normalization != "ABS_DFT_SQUARED_OVER_N_SQUARED":
            raise ValueError("unknown preregistered DFT power normalization")
        if self.filter_ratio != "LEARNED_OVER_FIXED":
            raise ValueError("Figure S5 sign convention is 10log10(learned/fixed)")


@dataclass(frozen=True)
class EvaluationTrace:
    run_id: str
    epochs: tuple[int, ...]
    learned_mean_ler: tuple[float, ...]
    fixed_initial_ler: tuple[float, ...]
    stream_kind: str = ALLOWED_STREAM
    time_coordinate: str = "EPOCH"

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", str(self.run_id))
        object.__setattr__(self, "epochs", tuple(int(value) for value in self.epochs))
        object.__setattr__(self, "learned_mean_ler", tuple(float(value) for value in self.learned_mean_ler))
        object.__setattr__(self, "fixed_initial_ler", tuple(float(value) for value in self.fixed_initial_ler))
        object.__setattr__(self, "stream_kind", str(self.stream_kind))
        object.__setattr__(self, "time_coordinate", str(self.time_coordinate))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvaluationTrace":
        return cls(str(value["run_id"]), tuple(map(int, value["epochs"])),
                   tuple(map(float, value["learned_mean_ler"])),
                   tuple(map(float, value["fixed_initial_ler"])),
                   str(value.get("stream_kind", ALLOWED_STREAM)),
                   str(value.get("time_coordinate", "EPOCH")))

    def validate(self, config: SourceDFTConfig) -> None:
        epochs = np.asarray(self.epochs, dtype=int)
        learned = np.asarray(self.learned_mean_ler, dtype=float)
        fixed = np.asarray(self.fixed_initial_ler, dtype=float)
        if not self.run_id or self.stream_kind != ALLOWED_STREAM:
            raise ValueError("paper claim requires decoded learned-mean/fixed LER evaluation traces")
        if self.time_coordinate != "EPOCH":
            raise ValueError("source DFT is defined in epoch coordinates, not physical time")
        if len(epochs) < 4 or learned.shape != epochs.shape or fixed.shape != epochs.shape:
            raise ValueError("trace arrays must be aligned and contain at least four evaluations")
        if not np.array_equal(np.diff(epochs), np.full(len(epochs) - 1, config.cadence_epochs)):
            raise ValueError("LER evaluation cadence must be exactly every five epochs")
        if config.warmup_epoch not in epochs:
            raise ValueError("trace must include the epoch-150 normalization point")
        if np.any(~np.isfinite(learned)) or np.any(~np.isfinite(fixed)) or \
                np.any(learned <= 0) or np.any(fixed <= 0):
            raise ValueError("LER values must be finite and strictly positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_source_contract() -> dict[str, Any]:
    literal = "SOURCE_LITERAL"
    unspecified = "SOURCE_UNSPECIFIED_PREREGISTERED"
    fields = [
        {"field": "evaluation_stream", "value": "decoded LER for learned mean and fixed initial policy", "status": literal},
        {"field": "cadence", "value": "every 5 training epochs", "status": literal},
        {"field": "time_coordinate", "value": "epoch, not physical time", "status": literal},
        {"field": "warmup", "value": "exclude epochs before 150 and normalize each trace to one at epoch 150", "status": literal},
        {"field": "spectrum", "value": "per-run discrete Fourier power with zero frequency excluded", "status": literal},
        {"field": "run_aggregation", "value": "interpolate unequal lengths to a shared grid then geometric average", "status": literal},
        {"field": "filter_function", "value": "10log10(learned PSD / fixed PSD)", "status": "SOURCE_DERIVED"},
        {"field": "raw_and_smooth", "value": "raw ratio and separately labelled Gaussian guide-to-eye", "status": literal},
        {"field": "shared_grid_and_dft_normalization", "value": "log grid; |DFT|^2/N^2", "status": unspecified},
        {"field": "smoothing_bandwidth", "value": "5 shared-grid bins", "status": unspecified},
        {"field": "hardware_dynamic_evaluation_traces", "value": None, "status": "NOT_PUBLICLY_IDENTIFIABLE"},
    ]
    payload = {"schema_version": "natural-drift-dft-source-contract.v1",
               "paper_doi": "10.1038/s41586-026-10759-2", "section": "Supplement III / Figure S5",
               "fields": fields, "forbidden_source_panel_estimators": ["WELCH", "BAND_INTEGRAL", "ZERO_PADDING", "PHYSICAL_TIME_RESAMPLING"],
               "forbidden_claim_streams": [FORBIDDEN_STREAM],
               "source_panel_structural_anchors": {
                   "frequency_unit": "epochs^-1", "zero_frequency_absent": True,
                   "low_frequency_psd_order": "learned_below_fixed",
                   "low_frequency_filter_db": "approximately -4 dB",
                   "high_frequency_behavior": "converges toward weak filtering"}}
    payload["source_contract_hash"] = canonical_hash(payload)
    return payload
