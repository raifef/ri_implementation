# Frozen confirmatory-v3 protocol

Status: prospectively frozen and unexecuted. Seeds 5001--5024 must be used once only.

Primary question: does conditionally activated residual RL improve on predictive-only
control without degrading detector rate, circuit-level logical performance, observed
recovery latency, lifecycle safety, or compute efficiency?

The primary matched pair is `predictive_hdfa_no_residual` (reference) versus
`predictive_hdfa_residual_rl` (treatment). Secondary comparisons retain full-control RL,
periodic recalibration, fixed calibration, and oracle. Seven frozen scenarios include a
learnable persistent residual and a no-residual negative control. All arms share the same
pre-disturbance baseline, simulator clone, disturbance tape, detector evaluator, logical
evaluator, and seed.

The primary RMST origin is synchronized disturbance onset. Its horizon is 8.0 seconds;
every run is observed through at least 9.0 seconds. Controller completion does not end
endpoint follow-up. Safety censoring is a failure and missing data is not imputed.
Seed-cluster bootstrap uncertainty uses 10,000 replicates and seed is the independent
unit. The one-interval 50% recovery requirement is 90%; the interval remains 512 cycles.
Observed 90% recovery is never extrapolated. Estimators.v2 stores worst, median, cluster
aggregate, RMST, and tail quantities separately. Any non-evaluable primary metric makes
the comparison invalid.

Residual authority remains conditional on significant structured residual, independent
evidence, forecast validity, uncertainty and scope checks, detector validation, and
Stage-7 authorization. Abstention is required for pure noise or predictive adequacy.
Hard bounds, slew limits, rollback, re-entry, and QEC-operability gates are unchanged.
