"""Static empirical observation surrogate with a fail-closed action boundary."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


def _logit(value: float) -> float:
    clipped = float(np.clip(value, 1e-7, 1.0 - 1e-7))
    return math.log(clipped / (1.0 - clipped))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


@dataclass(frozen=True)
class EmpiricalStaticSurrogate:
    detector_logit_mean: float
    detector_logit_std: float
    shared_latent_std: float
    temporal_ar1: float
    logical_log_intercept: float | None
    logical_log_slope: float | None
    source_split_hash: str

    @classmethod
    def from_contract(cls, contract: dict[str, Any]) -> "EmpiricalStaticSurrogate":
        p = contract["fitted_parameters"]
        return cls(
            detector_logit_mean=float(p["detector_logit_mean"]["estimate"]),
            detector_logit_std=float(p["detector_logit_std"]["estimate"]),
            shared_latent_std=float(p["shared_latent_std"]["estimate"]),
            temporal_ar1=float(p["temporal_ar1"]["estimate"]),
            logical_log_intercept=None if p["logical_mapping"]["intercept"] is None else float(p["logical_mapping"]["intercept"]),
            logical_log_slope=None if p["logical_mapping"]["slope"] is None else float(p["logical_mapping"]["slope"]),
            source_split_hash=str(contract["fit_split_hash"]),
        )

    def assert_action_supported(self, action: np.ndarray | None) -> None:
        if action is not None:
            raise ValueError(
                "the released dataset contains no control-action support; counterfactual actions are rejected with infinite extrapolation penalty"
            )

    def sample_detection_events(
        self,
        *,
        shots: int,
        detectors: int,
        seed: int,
        action: np.ndarray | None = None,
    ) -> np.ndarray:
        self.assert_action_supported(action)
        if shots <= 0 or detectors <= 0:
            raise ValueError("shots and detectors must be positive")
        rng = np.random.default_rng(seed)
        detector_logits = rng.normal(self.detector_logit_mean, self.detector_logit_std, detectors)
        latent = np.empty(shots)
        innovation_scale = self.shared_latent_std * math.sqrt(max(0.0, 1.0 - self.temporal_ar1**2))
        latent[0] = rng.normal(0.0, self.shared_latent_std)
        for index in range(1, shots):
            latent[index] = self.temporal_ar1 * latent[index - 1] + rng.normal(0.0, innovation_scale)
        probabilities = _sigmoid(detector_logits[None, :] + latent[:, None])
        return rng.binomial(1, probabilities).astype(np.uint8)

    def logical_risk(self, mean_detector_rate: float) -> float | None:
        if self.logical_log_intercept is None or self.logical_log_slope is None:
            return None
        return float(math.exp(self.logical_log_intercept + self.logical_log_slope * math.log(max(mean_detector_rate, 1e-12))))


def fit_contract_parameters(empirical: dict[str, Any], *, split_hash: str) -> dict[str, Any]:
    summaries = empirical["per_experiment"]
    rates = np.asarray([x["detector_rate_mean"] for x in summaries], dtype=float)
    logits = np.asarray([_logit(x) for x in rates])
    covariances = np.asarray([max(0.0, x["mean_off_diagonal_covariance"]) for x in summaries])
    lag1 = np.asarray([x["autocorrelation"][1] for x in summaries])
    mapping_rows = [x for x in summaries if x.get("logical_error_per_cycle") is not None]
    if len(mapping_rows) >= 3:
        design = np.column_stack(
            [np.ones(len(mapping_rows)), np.log([max(1e-12, x["detector_rate_mean"]) for x in mapping_rows])]
        )
        target = np.log([x["logical_error_per_cycle"] for x in mapping_rows])
        beta = np.linalg.lstsq(design, target, rcond=None)[0]
        mapping = {"intercept": float(beta[0]), "slope": float(beta[1]), "rows": len(mapping_rows)}
    else:
        mapping = {"intercept": None, "slope": None, "rows": len(mapping_rows)}
    logit_mean = float(np.mean(logits))
    logit_std = float(max(0.05, np.std(logits, ddof=1) if len(logits) > 1 else 0.05))
    shared = float(math.sqrt(max(0.0, np.median(covariances))))
    rho = float(np.clip(np.median(lag1), -0.95, 0.95))
    return {
        "detector_logit_mean": {"estimate": logit_mean, "uncertainty": float(np.std(logits, ddof=1) / math.sqrt(len(logits))) if len(logits) > 1 else None},
        "detector_logit_std": {"estimate": logit_std, "uncertainty": "experiment-to-experiment bootstrap documented in fit artifact"},
        "shared_latent_std": {"estimate": shared, "uncertainty": "identified from median positive off-diagonal covariance"},
        "temporal_ar1": {"estimate": rho, "uncertainty": "shot-order proxy; acquisition semantics not documented"},
        "logical_mapping": mapping,
        "fit_split_hash": split_hash,
    }
