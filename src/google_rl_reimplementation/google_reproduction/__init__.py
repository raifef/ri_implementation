"""Public-paper anchored Google-style QEC RL reproduction.

This package reproduces the open algorithm on a declared surrogate and remains distinct from the earlier certification harness. It does not reproduce Willow hardware.
"""

from .config import ReferenceConfig, load_reference_config
from .reference_agent import CandidateBatch, DetectorEvidence, ReferenceAgent
from .surrogate import PaperAnchoredSurrogate, surface_code_gate_count

__all__ = [
    "CandidateBatch",
    "DetectorEvidence",
    "PaperAnchoredSurrogate",
    "ReferenceAgent",
    "ReferenceConfig",
    "load_reference_config",
    "surface_code_gate_count",
]
