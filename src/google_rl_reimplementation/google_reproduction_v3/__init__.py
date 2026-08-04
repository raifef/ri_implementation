"""Read-only Zenodo analysis and empirical-surrogate identification (v3).

This namespace consumes released observations only and remains independent of controller execution. It never uses hidden plant truth or certification seeds.
"""

from .schemas import ReproductionStatus, SurrogateValidationOutcome

__all__ = ["ReproductionStatus", "SurrogateValidationOutcome"]

