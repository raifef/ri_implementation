# Preregistered compute-aware HDFA-RL confirmation v2

Protocol ID: `hdfa-rl-compute-aware-confirmation.v2`

Status: frozen prospectively before any seed 3001--3016 confirmatory disturbance
tape or arm is acquired. The protocol SHA-256 is bound in the sibling machine-readable
launch definition and then in the fresh preflight manifest. The campaign is executed
once after source/preflight freeze; failure is reported and is not followed by tuning on
these tapes.

## Scientific hypothesis and evidence boundary

In matched simulated QEC control, the staged Stage 0--7 controller will reduce restricted
mean end-to-end time to an observed 90% recovery relative to faithful full-control
detector RL after charging all online compute, QEC acquisition, diagnostics and
actuation/acknowledgement. It must simultaneously complete without lifecycle/
transaction corruption or uncontained physical rollback failure and pass the unchanged
EDR, exploration-damage, one-interval and final-rate rules.

This is internal simulator and Stim/PyMatching circuit-logical evidence. It is not a
Willow reproduction, hardware measurement, or proof of real-QPU superiority.

## Arms, matching and acquisition

All six mandatory arms run: faithful full-control detector RL, predictive HDFA without
residual RL, predictive HDFA with residual RL, fixed calibration, periodic recalibration
and oracle-informed control. Within every scenario/seed, arms receive an identical
stationary Stage-0 state, native-QEC pre-disturbance baseline, controller/physical state,
RNG state, evaluator configuration, disturbance tape and synchronized onset. Non-oracle
controllers cannot access latent simulator truth. Every arm runs 32 intervals or its
declared censoring boundary, and every intermediate trajectory is retained.

The 16 independent seeds are 3001--3016. None is used by a repository development
artifact. A prospective normal approximation for a paired one-sided standardized RMST
effect of 0.75 at alpha 0.05 and 80% power gives
`((1.6449 + 0.8416) / 0.75)^2 = 10.99` seeds; 25% allowance for heavy-tail precision and
safety censoring gives 14.65, rounded up to 16. Seed is the cluster unit; scenarios are
repeated conditions within seed, not independent replicates.

Five new definitions are frozen in `confirmatory_benchmark_scenarios`:

- periodic mixture: q0/q2 loadings 1.0/0.38, amplitude 0.31, period 1.83 s;
- semi-Markov: q1/q3 loadings 1.0/0.41, amplitude 0.29, rate 0.95 Hz, mean dwell 0.63 s;
- OU-step: q0/q2 OU diffusion 0.10 and kappa 0.42 plus q1 step amplitude 0.36 at 0.47 s;
- nested common: all-control loading 0.32, amplitude 0.27, period 2.77 s, with q1/q3
  semi-Markov child amplitude 0.20, rate 1.45 Hz and mean dwell 0.44 s;
- unstructured heavy tail: all-control loading 0.22 and diffusion 0.07.

These identifiers and parameterizations differ from all five v1 development scenarios.
The first four are the structured set for matched recovery gates; heavy-tail OOD remains
a declared safety/generalization condition.

## Primary compute-aware estimand

For arm `a` and observed target `q=0.90`:

`T_e2e(a,q) = T_qec + T_diagnostic + T_actuation_ack + T_online_compute_critical`.

QEC time is native cycles times 10 microseconds. Diagnostic downtime and explicit
actuation acknowledgements use simulated-device time. Online controller code uses
monotonic `perf_counter_ns`; every stage is recorded in a timestamped serial hybrid-clock
critical-path schedule. No modeled concurrency is used, so no overlap credit is taken.
Simulator/kernel host overhead and actual host control wall time are robustness fields,
not additions to the physical/compute critical-path formula. Offline Stim/PyMatching
logical evaluation, report serialization and analysis are reported and excluded.

Recovery is observed only. Fits cannot create an endpoint. A common administrative RMST
horizon of 8.0 s applies to every structured matched pair. Non-recovery and safety
censoring remain in the Kaplan-Meier risk set through that horizon. No complete-case
deletion is allowed.

The primary effect is
`RMST_e2e(full_control_detector_rl) - RMST_e2e(predictive_hdfa_residual_rl)`.
Uncertainty is a nonparametric cluster bootstrap over the 16 independent seeds with
10,000 replicates and RNG seed 20260802; all scenarios belonging to a sampled seed are
resampled together. The gate passes only if the one-sided 95% lower bound is greater
than zero and the staged arm has no safety censor.

The prospectively frozen tail safeguard is the 95th percentile of winsorized
`min(T_e2e, 8.0 s)` over the same complete risk set. Its seed-cluster bootstrap one-sided
95% upper bound for staged-minus-full-RL must be no more than 0.25 s. It is a co-required
primary safeguard.

## Other gates and multiplicity

The existing scientific gates are unchanged:

- all central runs complete, with zero lifecycle/transaction violations and zero
  uncontained physical rollback-validation failures;
- worst matched full-RL/staged integrated excess-EDR ratio is at least 5;
- worst matched full-RL/staged exploration-damage ratio is at least 2;
- at least 90% of structured matched staged runs observe 50% recovery within one
  control interval;
- the paired two-sided 95% CI upper bound for staged-minus-full-RL final EDR is at most
  0.005.

The former rigid 10x candidate-evaluation ratio remains a secondary diagnostic and does
not determine acceptance. Candidate evaluations and QEC cycles remain reported.

Acceptance is an intersection-union claim: every primary scientific and hard-safety gate
must pass. Each component is tested at its frozen level; there is no post-hoc selection
among favorable components, so no multiplicity credit or gate substitution is permitted.
Any missing pair, missing timing, source/config/environment mismatch, insufficient seed
cluster, endpoint extrapolation or non-evaluable primary gate makes the acquisition
invalid rather than accepted.

## Logical evidence and interpretation

Each interval uses a distance-3 rotated surface-code memory-Z circuit, three rounds,
4,096 shots, Stim circuit samples/error mechanisms and PyMatching MWPM. Detector and
logical evaluation carry identical physical, disturbance and policy state identifiers.
The logical result is evidence for the declared simulator-to-circuit mapping, not a
hardware-calibrated logical-error claim.

Exit/report interpretation is fixed:

- `authoritative: true, accepted: true`: every hard and primary gate passed;
- `authoritative: true, accepted: false`: valid scientific rejection;
- `authoritative: false`: invalid or non-evaluable acquisition.

The immutable v1 report and its rejection are never overwritten.
