"""Timescale-matched scientific-gate amendment for pure Google-style RL."""

DISCLAIMER = (
    "This is an open synthetic reproduction of the published Google-style RL algorithm. "
    "Google’s proprietary controller code and hardware control dynamics were unavailable."
)

ACTIVE_CERTIFICATION_SEEDS = tuple(range(12101, 12113))
RETIRED_SEEDS = (10101,)

ALLOWED_OUTCOMES = (
    "PURE_GOOGLE_STYLE_SYNTHETIC_REPRODUCTION_CERTIFIED",
    "PARTIAL_PURE_REPRODUCTION",
    "BANDWIDTH_MISMATCH",
    "REPLAY_STALENESS",
    "EXPLORATION_CALIBRATION_FAILURE",
    "NATURAL_DRIFT_RETENTION_FAILURE",
    "OBJECTIVE_TRANSCRIPTION_FAILURE",
    "SYNTHETIC_TASK_NON_COMMENSURABILITY",
    "GENUINE_CONTROLLER_FAILURE",
)
