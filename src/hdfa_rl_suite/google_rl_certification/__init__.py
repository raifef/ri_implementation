"""Independent certification of the public Google-style detector-RL baseline.

This package intentionally does not import HDFA, forecasting, MPC, residual RL, or the
staged supervisor.  It is a scientific comparator and certification harness, not a new
stage in the product controller.
"""

from .agent import CandidateBatch, CandidateEvaluation, GaussianPolicyGradientAgent
from .config import GoogleRLConfig, load_config

__all__ = [
    "CandidateBatch",
    "CandidateEvaluation",
    "GaussianPolicyGradientAgent",
    "GoogleRLConfig",
    "load_config",
]
