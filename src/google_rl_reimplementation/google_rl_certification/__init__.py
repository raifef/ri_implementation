"""Independent certification of the public Google-style detector-RL baseline.

This package contains the public-structure policy, declared surrogates, and certification harness. It has no dependency on another controller workflow.
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
