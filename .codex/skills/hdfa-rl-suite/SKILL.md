---
name: hdfa-rl-suite
description: Scientific implementation and review compass for the staged HDFA plus detector-driven QEC reinforcement-learning suite. Use for code, tests, simulation, benchmarks, or architecture reviews in this repository.
---

# HDFA-RL Suite

Read `00_WORKSPACE_CONTEXT.md`, `FINISHED_PRODUCT_AIMS.md`, the architecture overview, and the relevant revision-2 stage Word document before changing a public contract. The Word documents are authoritative.

## Invariant

```text
0 bootstrap -> 1 causal telemetry -> 2 detector likelihood/state
 -> 3 joint HDFA+dynamics -> 4 probabilistic forecast -> 5 safe MPC
 -> 6 residual RL; Stage 7 supervises every action and rollback
```

Use `u = u_MPC + R delta_u_RL`. Keep predictable structured motion in the physical model and restrict RL to the issued residual subspace. Sequential infer-then-segment is an ablation, never the deployed endpoint without equivalence evidence.

The product entry point is `hdfa_rl_suite.product.HDFAProductController`. Stage 0 runs at cold start and only re-enters after loss of QEC operability, major hardware reconfiguration, failed rollback, or OOD evidence requiring recalibration. Ordinary intervals reuse the validated Stage-0 state. Stage 6 is part of that persistent loop; Stage-5 baselines, residual candidates, and residual commits all require Stage-7 authorization before device execution.

## Implementation rules

- Preserve event acquisition order, exposure, context, exact active controls, activation uncertainty, hashes, versions, validity and rollback provenance.
- Keep simulator truth behind explicitly named oracle/evaluation capabilities.
- Evaluate Bernoulli/binomial or sparse correlated detector likelihoods inside joint temporal inference; preserve modes that imply different controls.
- Report actual Fisher null directions and request only safe interventions whose information can change a decision.
- Propagate state, model, parameter, process, discrepancy and latency uncertainty into detector/logical risk and MPC scenarios.
- Enforce hardware, slew, duty, thermal, leakage, shared-resource, atomicity, expiry, rollback and exploration budgets independently of objectives.
- Treat unknown/OOD evidence as a reason to abstain, diagnose or fall back.
- Keep fixed, periodic, greedy, state-only, sequential HDFA, joint HDFA, full-control detector RL, predictive-only, full staged and oracle arms independently executable.
- Count QEC cycles, candidates, diagnostic shots/downtime, exploration damage and logical checks separately.
- Match benchmark arms with a fixed-time disturbance realization, preserve every interval trajectory, and distinguish observed threshold non-recovery (censoring) from absent data.
- Use the named Stim rotated-surface-code plus PyMatching MWPM adapter for circuit-level logical evidence. Keep its evaluation-only control-error mapping separate from controller inputs and from real-hardware claims.

## Gates

Fault-inject timing skew, dropout, context changes, sign ambiguity, collinearity, correlations, fast switching, absent models, biased forecasts, solver failure, policy-hash mismatch, exhausted budgets and failed rollback. Report coverage, calibration, log/energy scores, boundary error, delay, regret, worst-region behavior, constraint violations, online/offline divergence, latency and scaling.

The 10x cycle/candidate, 5x excess-EDR, 2x exploration-damage, one-interval recurring correction and no-regression targets are falsifiable benchmark gates. Never encode them as assumed results.

Do not infer 90% recovery from an exponential fit unless the predeclared fit-credibility gate passes. Prefer observed 50/75/90% times, excess-EDR area, median and reached fraction, worst-region recovery, and censoring-aware survival summaries. A missing metric makes an acceptance report non-authoritative; an observed but failed threshold remains valid negative evidence.

For exact routing and implementation status, use `staged_calibration_architecture_revised/HDFA_RL_IMPLEMENTATION_SKILL.md` and `ARCHITECTURE_COMPLIANCE.md`.
