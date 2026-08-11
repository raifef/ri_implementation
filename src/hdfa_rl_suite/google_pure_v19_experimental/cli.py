"""CLI for the isolated bounded public-analogue dynamic validation."""
from __future__ import annotations

import json

from .dynamic_validation import run_three_frequency_validation
from .io import ARTIFACT_ROOT
from .matched_validation import MATCHED_ARTIFACT_ROOT, run_matched_three_frequency_validation


def three_frequency_main() -> int:
    result = run_three_frequency_validation()
    compact = {
        "pass": result["pass"],
        "execution_complete": result["execution_complete"],
        "controller_mode": result["controller_mode"],
        "controller_hash": result["controller_hash"],
        "ordering": result["ordering"],
        "sampled_policy_I_positive_all_frequencies":
            result["sampled_policy_I_positive_all_frequencies"],
        "frozen_source_branch_unchanged": result["frozen_source_branch_unchanged"],
        "rows": [{
            "label": row["label"],
            "gain": row["mean_transfer_regression"]["gain"],
            "phase_lag_radians": row["mean_transfer_regression"]["phase_lag_radians"],
            "I_stochastic": row["stream_decomposition"]["I_stochastic"],
            "sigma_median": row["sigma_diagnostics"]["analysis_sigma_median"],
        } for row in result["rows"]],
        "output_root": str(ARTIFACT_ROOT.resolve()),
        "source_budget_auto_launched": False,
        "heldout_auto_launched": False,
        "reference_auto_launched": False,
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0 if result["pass"] else 2


def matched_three_frequency_main() -> int:
    result = run_matched_three_frequency_validation()
    compact = {
        "pass": result["pass"],
        "execution_complete": result["execution_complete"],
        "controller_mode": result["controller_mode"],
        "controller_hash": result["controller_hash"],
        "ordering": result["ordering"],
        "sampled_policy_I_positive_all_frequencies":
            result["sampled_policy_I_positive_all_frequencies"],
        "frozen_source_branch_unchanged": result["frozen_source_branch_unchanged"],
        "rows": [{
            "label": row["label"],
            "gain": row["mean_transfer_regression"]["gain"],
            "phase_lag_radians": row["mean_transfer_regression"]["phase_lag_radians"],
            "I_stochastic": row["stream_decomposition"]["I_stochastic"],
            "sigma_median": row["sigma_diagnostics"]["analysis_sigma_median"],
        } for row in result["rows"]],
        "output_root": str(MATCHED_ARTIFACT_ROOT.resolve()),
        "prior_failed_pilot_preserved": True,
        "source_budget_auto_launched": False,
        "heldout_auto_launched": False,
        "reference_auto_launched": False,
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0 if result["pass"] else 2
