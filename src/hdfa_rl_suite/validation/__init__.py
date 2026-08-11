"""Physical and algorithmic validation ladder for comparative QEC control."""

from .common import ValidationCheck, ValidationReport
from .controller_sanity import ControllerSanityConfig, run_controller_validation
from .development_cohort import DevelopmentCohortConfig, run_development_cohort
from .fault_matrix import run_fault_matrix_validation
from .lifecycle_sanity import run_lifecycle_validation
from .manifest import PreflightManifest, build_preflight_manifest, validate_preflight_manifest
from .performance import PerformanceConfig, run_performance_validation
from .plant_sanity import PlantSanityConfig, canonical_scenarios, run_plant_validation
from .preflight import PreflightConfig, run_preflight
from .sample_budget import SampleBudgetConfig, run_sample_budget_validation
from .report_sanity import run_report_validation
from .post_comparison import run_post_comparison_validation
from .compute_sanity import run_compute_accounting_validation

__all__ = [
    "ControllerSanityConfig",
    "DevelopmentCohortConfig",
    "PerformanceConfig",
    "PlantSanityConfig",
    "PreflightConfig",
    "SampleBudgetConfig",
    "ValidationCheck",
    "ValidationReport",
    "PreflightManifest",
    "build_preflight_manifest",
    "canonical_scenarios",
    "run_controller_validation",
    "run_development_cohort",
    "run_fault_matrix_validation",
    "run_lifecycle_validation",
    "run_performance_validation",
    "run_plant_validation",
    "run_preflight",
    "run_sample_budget_validation",
    "run_report_validation",
    "run_post_comparison_validation",
    "run_compute_accounting_validation",
    "validate_preflight_manifest",
]
