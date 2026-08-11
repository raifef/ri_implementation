"""Read-only Zenodo analysis and empirical-surrogate identification (v3).

This namespace is intentionally independent of the v2 controller and of the
staged HDFA implementation.  It consumes released observations, never hidden
plant truth or certification seeds.
"""

from .schemas import ReproductionStatus, SurrogateValidationOutcome

__all__ = ["ReproductionStatus", "SurrogateValidationOutcome"]

