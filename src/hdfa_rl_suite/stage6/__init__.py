"""Stage-6 residual detector-driven Gaussian policy optimisation."""

from .residual_rl import ExplorationBudget, FullControlDetectorRL, GaussianResidualPolicy, ResidualRLConfig, ResidualRLController
from .gating import ResidualActivationGate, ResidualGateConfig
from .schema import (CandidateObservation, ResidualCandidate, ResidualRLResult,
                     ResidualGateDecision, ResidualGateEvidence,
                     ResidualRLDisposition, ShadowValidation,
                     bind_candidate_lifecycle)

__all__ = ["ExplorationBudget", "FullControlDetectorRL", "GaussianResidualPolicy",
           "ResidualRLConfig", "ResidualRLController", "CandidateObservation",
           "ResidualCandidate", "ResidualRLResult", "bind_candidate_lifecycle",
           "ResidualActivationGate", "ResidualGateConfig", "ResidualGateDecision",
           "ResidualGateEvidence", "ResidualRLDisposition", "ShadowValidation"]
