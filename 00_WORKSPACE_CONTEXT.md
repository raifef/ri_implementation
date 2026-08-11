# Workspace Context — Staged Physics-Informed Calibration During QEC

## Purpose

This workspace implements the architecture specified in `00_Architecture_Overview_Revised.docx` and the eight stage documents beside it. The project is a theoretical and computational research platform for autonomous calibration during quantum error correction (QEC). It extends detector-driven reinforcement learning by adding explicit physical-state inference, hierarchical fluctuation modelling, probabilistic forecasting, predictive control, residual reinforcement learning, and a supervisory safety layer.

The central hypothesis is:

> Native QEC detector events, together with the exact applied controls, circuit context, and a sparse physical response model, can identify and forecast operational calibration errors sufficiently well that model-predictive control handles structured motion while reinforcement learning only optimizes the unexplained residual.

This is not a claim that syndrome data uniquely identify microscopic noise mechanisms. The target is an operational latent state whose value and dynamics determine a useful control correction.

## Authoritative documentation

Treat these documents as the specification, in this order:

1. `00_Architecture_Overview_Revised.docx`
2. The corresponding stage document for the code being changed.
3. This context file and the stage implementation prompts.
4. Existing repository tests, interfaces, and design decisions, unless they conflict with the documents above.

Do not silently simplify an ideal requirement merely because it is difficult. A lower-fidelity approximation may be implemented as a named baseline, initialization path, or ablation, but the code structure must permit the full implementation without a rewrite.

## End-to-end data flow

```text
Stage 0: bootstrap calibration and QEC-operable baseline
    ↓ baseline policy, parameter registry, detector–control graph
Stage 1: native QEC telemetry and causal conditioning
    ↓ event tensor, aligned interventions, multiscale statistics
Stage 2: latent physical-state inference
    ↓ posterior over physical calibration variables and observability
Stage 3: joint HDFA and dynamical-model selection
    ↓ posterior over regimes, hierarchy, model identity and parameters
Stage 4: probabilistic forecasting
    ↓ calibrated future-state and detector-risk distributions
Stage 5: predictive feedforward / constrained MPC
    ↓ safe predictive baseline control and residual search subspace
Stage 6: residual detector-driven RL
    ↓ residual correction, exploration evidence and model-discrepancy signal
Stage 7: supervision, authorization, rollback and escalation
    ↺ governs every stage and can return to earlier stages
```

## Global engineering principles

- **No hidden oracle access.** Simulator latent truth is available only to explicitly named oracle baselines and evaluation code.
- **Causal operation.** Online methods must use only data available by the decision time. Preserve acquisition order and activation latency.
- **Exact provenance.** Every detector event must be traceable to circuit version, context, policy hash, candidate, timestamp, region, and measurement record.
- **Uncertainty is part of every interface.** Point estimates without covariance, posterior samples, validity horizon, or equivalent uncertainty are incomplete outputs.
- **Abstention is valid.** If the physical model is unsupported, the supervisor must fall back to a reproduced Google-style RL baseline rather than inventing confidence.
- **Safety is not a tunable reward term.** Hard bounds, interlocks, rollback availability, policy atomicity, and data validity are invariants.
- **No performance sacrifice for convenience.** Sequential infer-then-segment HDFA, independent-detector likelihoods, diagonal covariances, linear response maps, Gaussian filters, and similar approximations are baselines. Retain them in the final path only where closed-loop equivalence to richer models is demonstrated.
- **Reproducibility.** Every experiment must be seedable, configuration-driven, replayable, and produce machine-readable results plus human-readable summaries.
- **Fair comparison.** Report both physical time and detector/QEC-cycle budget. Count dedicated characterization as downtime and extra measurement cost.

## Required architecture-wide baselines

Implement and preserve the following comparison arms:

1. Fixed calibration.
2. Periodic recalibration.
3. Google-style detector-driven RL reproduced as faithfully as practical.
4. Bootstrap/greedy calibration without predictive modelling.
5. Physical-state inference without HDFA.
6. Sequential infer-then-segment HDFA.
7. Joint inference and segmentation.
8. Predictive control without residual RL.
9. Full predictive control plus residual RL.
10. Oracle latent-state controller for an upper bound only.

## Architecture-wide MVP acceptance targets

The architecture MVP is not accepted merely because all stages run. On the declared structured-drift benchmark suite it must, relative to the faithful RL baseline:

- use at least **10× fewer detector cycles or candidate evaluations** to recover 90% of lost performance after familiar structured disturbances;
- reduce integrated excess detector-event rate during recovery by at least **5×**;
- reduce exploration-induced logical-risk proxy by at least **2×**;
- correct a previously identified periodic or discrete regime within one inference-and-control interval;
- preserve final steady-state performance;
- revert automatically to statistically indistinguishable baseline-RL performance on unstructured drift;
- avoid future information, hidden latent truth, or uncounted dedicated characterization.

These are falsifiable research targets, not assumed results. The reporting code must make it impossible to hide a failure behind aggregate averages.

## Shared packages and interfaces

Prefer a package layout similar to:

```text
src/
  common/          # IDs, units, time, schemas, numerical utilities
  simulator/       # latent device, QEC observation and benchmark scenarios
  stage0_bootstrap/
  stage1_telemetry/
  stage2_inference/
  stage3_hdfa/
  stage4_forecast/
  stage5_mpc/
  stage6_residual_rl/
  stage7_supervisor/
  baselines/
  evaluation/
  cli/
tests/
configs/
artifacts/
docs/
```

Use typed, versioned records. At minimum define stable identifiers for device, qubit, coupler, control parameter, detector, measurement channel, circuit, context, policy, candidate, acquisition batch, model version, region, and scenario.

All public stage APIs must support:

- serialization with schema version;
- validation and explicit invalidity reasons;
- deterministic replay;
- structured logging;
- unit-aware numerical values or rigorously documented canonical units;
- batch/offline and streaming/online operation where relevant.

## Testing standard

Each stage must include:

- unit tests for mathematics and schema validation;
- property-based or randomized tests for invariants;
- synthetic truth-recovery tests;
- failure-injection tests;
- deterministic replay tests;
- integration tests with preceding and following stage contracts;
- benchmark tests that report accuracy, coverage, latency, sample cost, and closed-loop regret.

Do not test only mean error. Check posterior coverage, calibration, false-alarm rates, missed events, worst-region performance, logical-risk divergence, rollback correctness, and out-of-distribution behaviour.

## Implementation order

Implement in dependency order. First reproduce the baseline RL and deterministic simulator sufficiently to measure improvement. Stage 1 provenance and replay must be reliable before any inference claim is trusted. Build one-region versions first, but use graph-indexed interfaces from the beginning so scaling does not require redesign.
