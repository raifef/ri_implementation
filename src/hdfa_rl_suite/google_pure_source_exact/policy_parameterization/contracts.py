"""Machine-readable source and evidence contracts for direct sigma."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "google-pure-source-exact-direct-sigma.v1"
DIRECT_SIGMA_PARAMETERIZATION = "DIRECT_SIGMA_SOURCE_EXACT"
NON_PAPER_LOG_SIGMA_ABLATION = "NON_PAPER_LOG_SIGMA_ABLATION"
SOURCE_ELEMENTWISE_COORDINATE_CLIPPING = "SOURCE_ELEMENTWISE_COORDINATE_CLIPPING"
NON_SOURCE_PPO_ABLATION = "NON_SOURCE_PPO_ABLATION"
JOINT_LEARNED_DETECTOR_BASELINE = "JOINT_LEARNED_DETECTOR_BASELINE"
NON_SOURCE_EMA_BASELINE_ABLATION = "NON_SOURCE_EMA_BASELINE_ABLATION"


class SourceIdentifiability(StrEnum):
    SOURCE_LITERAL = "SOURCE_LITERAL"
    SOURCE_DERIVED = "SOURCE_DERIVED"
    SOURCE_REFERENCED_PRIMARY_METHOD = "SOURCE_REFERENCED_PRIMARY_METHOD"
    SOURCE_UNSPECIFIED_PREREGISTERED = "SOURCE_UNSPECIFIED_PREREGISTERED"
    NOT_PUBLICLY_IDENTIFIABLE = "NOT_PUBLICLY_IDENTIFIABLE"


class PositivityGuard(StrEnum):
    PROJECTED_GRADIENT = "projected_gradient"
    BOUNDED_OPTIMIZER = "bounded_optimizer"
    BACKTRACKING_STEP = "backtracking_step"


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
class EvidenceStatus:
    artifact_complete: bool
    mathematical_contract_pass: bool
    protocol_contract_pass: bool
    source_structure_match: bool
    quantitative_match: bool
    paper_comparable: bool
    blocking_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IterationRecord:
    iteration_id: str
    source_commit: str
    code_hash: str
    config_hash: str
    plant_hash: str
    protocol_hash: str
    analysis_hash: str
    seed_registry_hash: str
    changes_from_previous_iteration: tuple[str, ...]
    failed_gates: tuple[str, ...]
    numerical_results: Mapping[str, Any]
    next_diagnosis: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def require_parameterization(value: str, *, paper_mode: bool) -> str:
    selected = str(value)
    allowed = {DIRECT_SIGMA_PARAMETERIZATION, NON_PAPER_LOG_SIGMA_ABLATION}
    if selected not in allowed:
        raise ValueError(f"unknown policy parameterization: {selected}")
    if paper_mode and selected != DIRECT_SIGMA_PARAMETERIZATION:
        raise ValueError("paper mode requires direct sigma; log-sigma is an ablation only")
    return selected


def require_loss_semantics(ratio_mode: str, baseline_mode: str, *, paper_mode: bool) -> tuple[str, str]:
    ratio = str(ratio_mode)
    baseline = str(baseline_mode)
    if ratio not in {SOURCE_ELEMENTWISE_COORDINATE_CLIPPING, NON_SOURCE_PPO_ABLATION}:
        raise ValueError(f"unknown ratio clipping mode: {ratio}")
    if baseline not in {JOINT_LEARNED_DETECTOR_BASELINE, NON_SOURCE_EMA_BASELINE_ABLATION}:
        raise ValueError(f"unknown detector baseline mode: {baseline}")
    if paper_mode and ratio != SOURCE_ELEMENTWISE_COORDINATE_CLIPPING:
        raise ValueError("paper mode requires coordinate ratios clipped before the sparse product")
    if paper_mode and baseline != JOINT_LEARNED_DETECTOR_BASELINE:
        raise ValueError("paper mode requires the jointly learned detector baseline")
    return ratio, baseline


def build_source_contract() -> dict[str, Any]:
    fields = [
        {"field": "policy_distribution", "value": "factorized Gaussian with diagonal covariance diag(sigma^2)",
         "status": SourceIdentifiability.SOURCE_LITERAL, "source": "Supplement VIII.A Eq. (11)"},
        {"field": "learnable_policy_parameters", "value": "theta=(mu,sigma), P components each",
         "status": SourceIdentifiability.SOURCE_LITERAL, "source": "Supplement VIII.A and VIII.C after Eq. (22)"},
        {"field": "policy_loss", "value": "negative empirical detector-local clipped importance surrogate",
         "status": SourceIdentifiability.SOURCE_LITERAL, "source": "Supplement VIII.C Eq. (18)"},
        {"field": "ratio_clipping_order", "value": "clip each coordinate ratio, then exp(M @ log(chi_clipped))",
         "status": SourceIdentifiability.SOURCE_LITERAL, "source": "Supplement VIII.C Eqs. (17)-(18)"},
        {"field": "baseline_loss", "value": "empirical mean squared detector advantage",
         "status": SourceIdentifiability.SOURCE_LITERAL, "source": "Supplement VIII.C Eq. (19)"},
        {"field": "detector_baseline", "value": "one jointly optimized parameter per reduced detector component; pre-update baseline frozen within a batch",
         "status": SourceIdentifiability.SOURCE_LITERAL, "source": "Supplement VIII.C Eqs. (13), (19), and (22)"},
        {"field": "entropy_loss", "value": "negative Gaussian policy entropy",
         "status": SourceIdentifiability.SOURCE_LITERAL, "source": "Supplement VIII.C Eqs. (20)-(21)"},
        {"field": "total_loss", "value": "L_policy + L_baseline + L_entropy; gradient descent on mu, sigma, b",
         "status": SourceIdentifiability.SOURCE_LITERAL, "source": "Supplement VIII.C Eq. (22), Algorithm 1"},
        {"field": "direct_sigma_scores", "value": "dlogpi/dmu=(p-mu)/sigma^2; dlogpi/dsigma=(p-mu)^2/sigma^3-1/sigma",
         "status": SourceIdentifiability.SOURCE_DERIVED, "source": "analytic derivative of Eq. (11)"},
        {"field": "entropy_sigma_gradient", "value": "dH/dsigma=1/sigma",
         "status": SourceIdentifiability.SOURCE_DERIVED, "source": "analytic Gaussian entropy derivative"},
        {"field": "positivity_guard", "value": "projected gradient selected; bounded and backtracking compared on development fixtures",
         "status": SourceIdentifiability.SOURCE_UNSPECIFIED_PREREGISTERED, "source": "not reported publicly"},
        {"field": "baseline_loss_weight", "value": "development grid [0.05, 0.2, 1.0], selected 0.2 before certification",
         "status": SourceIdentifiability.SOURCE_UNSPECIFIED_PREREGISTERED, "source": "relative loss weights are not published"},
        {"field": "loss_weights_learning_rates_and_gradient_clip", "value": "profile configuration",
         "status": SourceIdentifiability.NOT_PUBLICLY_IDENTIFIABLE, "source": "Supplement VIII.C says hyperparameters were shallow-manually tuned but does not publish values"},
        {"field": "proprietary_controller_code", "value": None,
         "status": SourceIdentifiability.NOT_PUBLICLY_IDENTIFIABLE, "source": "Nature code-availability statement"},
    ]
    payload = {"schema_version": SCHEMA_VERSION, "paper_doi": "10.1038/s41586-026-10759-2",
               "clean_room": True, "old_version_artifacts_are_read_only": True,
               "paper_parameterization": DIRECT_SIGMA_PARAMETERIZATION, "fields": fields}
    payload["source_contract_hash"] = canonical_hash(payload)
    return payload
