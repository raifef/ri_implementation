"""Independent repair of the public Google-style detector-driven RL reproduction."""

DISCLAIMER = (
    "This is an open synthetic reproduction of the published Google-style RL algorithm. "
    "Google’s proprietary controller code and hardware control dynamics were unavailable."
)

OUTCOME_CLASSES = (
    "BENCHMARK_FAILURE", "REPORTING_CONVENTION_FAILURE", "UNIT_OR_NORMALIZATION_FAILURE",
    "EXPLORATION_CALIBRATION_FAILURE", "BANDWIDTH_MISMATCH", "REPLAY_STALENESS",
    "BASELINE_FAILURE", "OBJECTIVE_TRANSCRIPTION_FAILURE",
    "SYNTHETIC_TASK_NON_COMMENSURABILITY", "GENUINE_CONTROLLER_FAILURE",
    "PURE_GOOGLE_STYLE_SYNTHETIC_REPRODUCTION_CERTIFIED", "PARTIAL_PURE_REPRODUCTION",
)

__all__ = ["DISCLAIMER", "OUTCOME_CLASSES"]
