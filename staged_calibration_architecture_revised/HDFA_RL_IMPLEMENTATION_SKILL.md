---
name: hdfa-rl-architecture
description: Implementation compass for the staged HDFA and detector-driven QEC reinforcement-learning architecture. Use alongside this folder's revision-2 stage specifications for Python implementation, tests, reviews, and benchmarks.
---

# HDFA–RL QEC Architecture

This is a light implementation skill. Treat the revision-2 Word documents in this folder as authoritative; read the relevant stage document before changing its contract.

## System invariant

```text
0 bootstrap -> 1 telemetry -> 2 detector likelihood/state belief
 -> 3 joint HDFA + dynamics -> 4 forecast -> 5 safe MPC -> 6 residual RL
                                  ^                            |
                                  +---- evidence / health ------+ -> 7 supervisor
```

Use `u = u_MPC + delta_u_RL`: physical prediction controls explainable motion, while RL corrects the remaining residual. Preserve control hashes, activation timing, circuit context, uncertainty, versions, and rollback provenance end to end.

Never deploy the easier sequential route (estimate a physical trajectory, then segment it) merely because it is simpler. It is an initialization, diagnostic, and ablation baseline unless it demonstrates equivalence to the joint/offline reference on posterior calibration, forecast score, closed-loop regret, and safety.

## Source routing

| Need | Source |
| --- | --- |
| Global claims, targets, comparison rules | `00_Architecture_Overview_Revised.docx` |
| QEC-operable policy, graph, sensitivity, safety | `01_Stage_0_Bootstrap_Calibration_Revised.docx` |
| Exact detector events, causal timing, count windows | `02_Stage_1_Native_QEC_Telemetry_Revised.docx` |
| Detector likelihood and operational-state observability | `03_Stage_2_Latent_Physical_State_Inference_Revised.docx` |
| Joint HDFA/dynamics, model bank, changepoints | `04_Stage_3_Joint_HDFA_and_Dynamical_Model_Selection_Revised.docx` |
| Latency-aware predictive distributions | `05_Stage_4_Probabilistic_Forecasting_Revised.docx` |
| Feedforward, scenario MPC, residual subspace | `06_Stage_5_Predictive_Feedforward_and_MPC_Revised.docx` |
| Detector-driven residual RL and exploration budgets | `07_Stage_6_Residual_Detector_Driven_RL_Revised.docx` |
| Modes, invariants, rollback, diagnostics, lifecycle | `08_Stage_7_Supervisory_Control_Revised.docx` |

## Build order

1. Build a deterministic non-stationary QEC calibration simulator; keep latent truth inaccessible to non-oracle controllers.
2. Reproduce and freeze a masked Gaussian full-policy detector-RL baseline before measuring any staged gain.
3. Implement Stage 0 then Stage 1 with deterministic replay and raw event retention.
4. Implement local Stage 2 sparse control-conditioned likelihoods, discrepancy, and observability/null-space reports.
5. Implement the finite-bank Stage-3 joint model—oscillator, RTN, OU/random walk, step/changepoint, and unknown heavy-tailed fallback—evaluating the Stage-2 likelihood inside the temporal loop.
6. Add activation-latency-aware forecast mixtures, constrained scenario MPC, then residual-coordinate RL.
7. Add the Stage-7 state machine, atomic rollback, diagnostics, and audit after all upstream invalidity flags exist.
8. Scale local sparse regions before shared/common-mode factors. Neural models are bounded proposals/residuals, never the unverified safety path.

## Engineering requirements

- **Causality:** map every event to controls physically active in its detecting region. Flag ambiguity; missing exposure is not a zero event; online statistics cannot use future data.
- **Uncertainty:** retain Bernoulli/binomial or sparse-factor likelihoods and propagate samples/mixtures. Do not collapse modes that imply different controls.
- **Observability:** report rank and unresolved directions. Use safe antithetic/context excitation only when ambiguity changes a decision.
- **Safety:** independently enforce hardware, slew, thermal, shared-resource, compiler, and exploration-damage constraints. Version every action and preserve rollback.
- **Validity:** attach trust regions, OOD/predictive checks, latency, forecast horizon, and solver certificates. Reject stale or extrapolative actions.
- **Accounting:** separately report detector cycles, wall time, diagnostic downtime/shots, exploration damage, and logical checks. Never hide dedicated characterization in native-QEC comparisons.

## MVP choices

- Sparse regional quadratic-logit response likelihood with uncertainty and discrepancy.
- Exact event/count factors; fast Stage-2 filters only as proposals.
- Rao–Blackwellized particle or interacting-multiple-model Stage-3 inference, continually compared to fixed-lag/offline smoothing.
- Forecast samples/analytic mixtures at confirmed activation and MPC horizons.
- Local scenario-constrained quadratic MPC with nonlinear pre-execution check and certified fallback.
- Normalized residual Gaussian policies, 4–16 close antithetic candidates, graph masks, adaptive shots, entropy floors, and strict damage budgets.
- Deterministic supervisory modes: `BOOTSTRAP`, `NOMINAL_PREDICTIVE`, `RESIDUAL_LEARNING`, `LOCAL_RECOVERY`, `UNKNOWN_EVENT`, `DIAGNOSTIC`, `DEGRADED`, `FAIL_SAFE`.

## Required gates

- Verify detector parity and streaming/offline causal replay bit-for-bit.
- Fault-inject timing skew, dropout, context switch, sign ambiguity, state collinearity, correlations, OOD response, fast switching, phase reset, absent model, solver failure, and failed rollback.
- Measure posterior coverage, model calibration, forecast score/coverage by horizon, constraint violations, recovery time, cumulative detector/logical damage, and online-vs-offline divergence.
- Use the frozen baseline with matched detector budget and safety. System targets are hypotheses: 10x fewer recovery candidates/cycles, 5x lower integrated excess detector rate, 2x lower exploration-induced logical damage, and no unstructured-drift regression through validated RL fallback.

## Change discipline

Before a change, identify the relevant stage's declared inputs, outputs, validity conditions, and handover. Implement immutable typed contracts and replay before optimization. Regard independence, diagonal covariance, linear response, Gaussian approximations, and sequential HDFA as approximations requiring offline-reference equivalence. Prefer abstention, rollback, or a logged diagnostic to confident extrapolation.
