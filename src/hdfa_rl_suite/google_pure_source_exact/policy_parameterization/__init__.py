"""Paper-literal direct-standard-deviation Gaussian policy."""

from .contracts import (
    DIRECT_SIGMA_PARAMETERIZATION,
    NON_PAPER_LOG_SIGMA_ABLATION,
    PositivityGuard,
    SourceIdentifiability,
    build_source_contract,
)
from .gaussian import (
    BehaviorSnapshot,
    CandidateBatch,
    DirectSigmaGaussianPolicy,
    component_log_probability,
    entropy,
    gaussian_scores,
)
from .losses import LossResult, total_loss_and_gradients
from .optimizer import DirectSigmaOptimizer, OptimizerConfig

__all__ = [
    "BehaviorSnapshot",
    "CandidateBatch",
    "DIRECT_SIGMA_PARAMETERIZATION",
    "DirectSigmaGaussianPolicy",
    "DirectSigmaOptimizer",
    "LossResult",
    "NON_PAPER_LOG_SIGMA_ABLATION",
    "OptimizerConfig",
    "PositivityGuard",
    "SourceIdentifiability",
    "build_source_contract",
    "component_log_probability",
    "entropy",
    "gaussian_scores",
    "total_loss_and_gradients",
]
