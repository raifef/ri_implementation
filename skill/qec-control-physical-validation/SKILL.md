---
name: qec-control-physical-validation
description: Validate and harden simulated quantum-calibration plants, adaptive controllers, baselines, and scientific benchmarks before accepting comparative performance claims.
version: 1.0
---

# QEC Control Physical Validation

## Purpose

Use this skill whenever implementing, modifying, reviewing, or benchmarking a simulated quantum calibration or QEC-control system. Its purpose is to prevent a benchmark from being procedurally reproducible yet physically meaningless.

The skill applies especially when:

- a learned controller performs worse than a fixed policy;
- fixed calibration does not degrade under applied drift;
- an oracle is only marginally better than fixed control;
- exploratory candidates are confused with the learned mean policy;
- simulator truth, detector likelihoods, and logical evaluation may use inconsistent states;
- a controller is called “faithful” despite a radically different sample budget;
- long acceptance runs are being attempted before short physical sanity tests pass;
- missing, censored, or selectively completed trajectories could bias conclusions.

The governing rule is:

> Do not compare sophisticated controllers until the uncontrolled plant, oracle, fixed baseline, periodic calibration, and reference adaptive baseline each pass independent physical and algorithmic validation.

A test can be deterministic, fully logged, statistically careful, and still test the wrong physics. Reproducibility is necessary but not sufficient.

## Authoritative project context

Before changing code, read:

1. `FINISHED_PRODUCT_AIMS.md`
2. `00_WORKSPACE_CONTEXT.md`
3. `ARCHITECTURE_COMPLIANCE.md`
4. the revised architecture specifications
5. the latest comparison report and its retained artifacts
6. the implementation and tests for the affected simulator, controller, benchmark, and reporting paths

Treat claims from different evidence layers separately:

- published hardware evidence;
- declared analytical or factor-graph surrogates;
- actually executed repository simulations;
- circuit-level logical adapters;
- hardware or controller deployment measurements.

Never relabel one layer as another.

## Non-negotiable physical invariants

Every benchmark plant must expose and test these invariants before controller comparison.

### No-disturbance invariance

With a stationary latent optimum and fixed policy:

- the latent optimum must remain constant;
- detector and logical metrics must remain stationary within shot noise;
- no arm may improve or degrade because of hidden reinitialization, policy leakage, or simulator cloning differences.

### Persistent-drift degradation

For a sustained step in a controllable parameter:

- fixed calibration must worsen after onset;
- it must remain degraded while the step persists;
- the degradation must increase with mismatch magnitude over the validated local range.

For sinusoidal drift:

- the control mismatch must oscillate with the specified period and phase;
- the detector response must follow the magnitude and sign sensitivity implied by the observation model;
- time averaging must not conceal the interval-wise degradation.

For random-telegraph drift:

- the same latent state must produce a reproducible detector distribution;
- state switching must be visible in interval-wise regional statistics;
- dwell-time and transition statistics must match the configured process.

For OU or random-walk drift:

- the latent state must not reset at interval boundaries;
- cloned controller arms must receive the same disturbance realization;
- error must depend on displacement from the active optimum, not merely on elapsed time.

### Oracle ordering

An exact oracle must remove most controllable degradation while retaining the irreducible floor. In a controllable scenario, the expected qualitative ordering is:

`oracle <= validated adaptive controller <= periodic recalibration <= fixed calibration`

The exact ordering can temporarily differ during exploration or immediately after onset, but any sustained inversion must be explained by logged physical quantities.

### Shared-state consistency

At every interval, the following must refer to the same active simulated state and policy version:

- latent optimum;
- applied control;
- physical gate-error parameters;
- detector-event generator;
- logical-evaluation adapter;
- supervisor decision;
- policy hash and lifecycle state.

Add immutable identifiers and assertions rather than relying on implicit sequencing.

## Controller validation ladder

Never begin with the full architecture or a long benchmark.

### Level 1 — Analytic toy landscape

Use a one-parameter convex objective with a known optimum and exact gradient.

Validate:

- reward sign;
- loss sign;
- optimizer direction;
- perturbation centring;
- candidate-to-reward alignment;
- mean update;
- covariance update;
- bounds and slew behaviour.

The learned mean must converge reliably from both sides of the optimum.

### Level 2 — Static detector model

Use a fixed multi-parameter detector model with a known sparse Jacobian.

Validate:

- factor-graph masks;
- gradient masking;
- sensitivity normalization;
- block covariance;
- parameter indexing;
- inactive-region invariance.

Compute the true simulator gradient and the estimated gradient. Require positive cosine similarity with a predeclared margin.

### Level 3 — Calibrated no-drift start

Starting from the optimum:

- the learned mean policy must not systematically regress;
- exploration damage must be reported separately from mean-policy performance;
- covariance must not grow without a justified non-stationary signal;
- the safety layer must suppress candidates outside the damage budget.

### Level 4 — Single controlled disturbance

Use one local step, then one sinusoid, then one RTN source.

Validate:

- direction of recovery;
- response latency;
- stable final tracking;
- no unrelated parameter motion;
- correct attribution of detector changes to the applied control and disturbance.

### Level 5 — Mixed and structured disturbance

Only after Levels 1–4 pass, combine structured components and test HDFA, forecasting, MPC, and residual RL.

### Level 6 — OOD and lifecycle stress

Test unknown heavy-tailed or model-violating drift, re-entry, rollback, regional isolation, and safe fallback.

### Level 7 — Paired scientific comparison

Only now run full matched acceptance studies.

## Required distinction: learned policy versus exploration

Always log and report separately:

- `mean_policy_metric`: evaluation of the current learned mean policy;
- `candidate_metric`: outcome for each exploratory candidate;
- `aggregate_exploration_metric`: mean or integrated performance while candidates are physically applied;
- `exploration_damage`: excess cost relative to the learned mean or a declared safe reference;
- `evaluation_policy_metric`: independent held-out evaluation, if used.

Never use the average candidate EDR as proof that the learned mean policy is poor, and never use the mean-policy EDR to hide damage incurred during exploration.

## Sample-budget validation

A controller is not “faithful” merely because it uses the same equations.

For every chosen number of cycles or shots per candidate, measure:

- reward-ranking accuracy;
- gradient cosine similarity;
- gradient norm bias;
- harmful-update probability;
- convergence probability;
- final mean-policy performance;
- exploration cost.

If the implementation uses a much smaller budget than the published method, it must either:

1. reproduce the published-scale budget; or
2. demonstrate through a predeclared budget sweep that the reduced budget preserves gradient quality and convergence.

Do not silently trade scientific validity for runtime.

## Plant calibration requirements

The control-mismatch-to-error map must be explicit, inspected, and tested. At minimum, document:

- the latent physical parameter;
- the active optimum;
- the applied control;
- the mismatch;
- local curvature or response coefficient;
- irreducible noise floor;
- cross-coupling terms;
- clipping or saturation;
- detector emission model;
- logical-error mapping.

Check monotonicity over the intended operating range. If a non-monotonic or asymmetric response is intentional, add explicit tests and plots.

Calibrate disturbance amplitudes so that:

- fixed control degrades materially;
- the oracle retains a clear advantage;
- the problem remains recoverable;
- detector statistics do not saturate;
- logical metrics remain measurable at the chosen shot count.

## Policy lifecycle invariants

Use an explicit transactional lifecycle:

`confirmed -> proposed -> pending validation -> authorized -> atomically active -> acknowledged -> confirmed`

Every policy action must carry:

- `policy_id`;
- `reference_policy_id`;
- `created_from_state_id`;
- `expected_activation_state_id`;
- bounds and slew certificate;
- supervisor authorization;
- activation acknowledgement.

If the active or confirmed reference changes, reject and reproject the action. Never project a probe from one policy and validate it after silently activating another.

## Benchmark preflight gate

A full benchmark must refuse to start unless all of these pass:

- plant no-disturbance sanity;
- fixed-versus-oracle disturbance sanity;
- periodic-calibration intermediate sanity;
- reference RL convergence on toy and static detector landscapes;
- positive gradient-direction diagnostic;
- sample-budget adequacy;
- mean-versus-exploration separation;
- policy lifecycle consistency;
- identical disturbance realization across matched arms;
- complete artifact and configuration hashing.

The preflight should be executable independently and should produce a machine-readable pass/fail manifest.

## Testing policy: protect credits and time

Use three test tiers.

### Tier A — CI smoke tests

- seconds, not minutes;
- tiny devices;
- deterministic seeds;
- analytic or short synthetic scenarios;
- test invariants, signs, indexing, lifecycle, and schema contracts.

### Tier B — development validation

- short paired runs;
- a few seeds;
- enough cycles to validate physical trends and gradient quality;
- no final scientific claims.

### Tier C — authoritative acquisition

- long, matched, predeclared;
- run only after Tier A and B pass;
- never launch automatically during ordinary implementation;
- print estimated cost and request explicit user execution when appropriate.

Do not spend substantial compute proving that a sign, state, or baseline bug still exists.

## Reporting requirements

Every controller comparison report must show:

- interval-wise fixed, periodic, oracle, and adaptive trajectories;
- latent optimum and control mismatch;
- learned mean-policy performance;
- exploratory candidate performance;
- number of active trajectories at each time;
- completion and censoring reasons;
- candidate and cycle budgets;
- detector and logical metrics;
- physical sanity checks;
- fit-quality diagnostics;
- evidence-layer labels.

Any acceptance result must fail closed when required physical-validation or lifecycle prerequisites are not met.

## Definition of done

A change is complete only when:

- the failing mechanism is reproduced in a minimal test;
- the physical or algorithmic root cause is identified;
- the implementation is corrected;
- short regression tests prevent recurrence;
- the benchmark preflight rejects deliberately broken variants;
- fixed, oracle, periodic, and reference-RL behaviour has the expected physical ordering;
- long acceptance runs are not required to detect the same class of error again.
