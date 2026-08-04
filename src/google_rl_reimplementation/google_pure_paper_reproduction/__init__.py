"""Auditable public-data and pure Google-style RL paper reproduction workflows.

This namespace deliberately contains no outside-workflow controller, forecaster, MPC,
supervisor, or residual-policy implementation.  It freezes the comparator that
later studies may consume.
"""

from .experiment_families import EvidenceClass, ExperimentFamily, RunMode

__all__ = ["EvidenceClass", "ExperimentFamily", "RunMode"]

