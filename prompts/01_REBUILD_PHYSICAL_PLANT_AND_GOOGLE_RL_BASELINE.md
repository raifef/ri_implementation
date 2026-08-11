# Codex Prompt 1 — Rebuild the Physical Plant and Google-Style RL Baseline

## Role and objective

Work in the `hdfa_rl_suite` repository as a senior scientific software engineer with expertise in stochastic control, Monte Carlo estimation, QEC detector models, and reproducible simulation.

The current comparison is not physically trustworthy. The reported Google-style RL arm is substantially worse than fixed calibration, fixed calibration does not appear to degrade convincingly under drift, and the oracle is only modestly better than fixed. Before modifying HDFA, forecasting, MPC, or residual RL, rebuild the validation foundation so that the simulated plant and reference full-control RL baseline are independently credible.

Do not run a long authoritative acquisition during this task. Use short deterministic tests and development-scale diagnostics. Leave a command for the user to run the long acquisition after all preflight gates pass.

## Authoritative sources

Read and follow:

- `FINISHED_PRODUCT_AIMS.md`
- `00_WORKSPACE_CONTEXT.md`
- `ARCHITECTURE_COMPLIANCE.md`
- the revised architecture documents
- the latest comparison report and `artifacts/comparison/nature-2026-v5`
- the current simulator, baseline controllers, pipeline, logical adapter, benchmark, and tests
- the workspace skill `qec-control-physical-validation/SKILL.md`

Preserve evidence-layer distinctions. Do not claim a Willow reproduction.

## Required outcome

Deliver a physically validated simulation plant and a functioning Google-style full-control detector-RL baseline that pass a staged validation ladder before they are permitted into the full comparative benchmark.

The work is not complete if only plots look more plausible. Add machine-checkable physical and algorithmic invariants.

---

## Part A — Audit the current implementation

Trace one complete interval for each arm and document, in code comments or a concise audit note:

1. how the latent optimum evolves;
2. how the active policy is chosen;
3. how mismatch becomes physical gate error;
4. how gate error becomes detector events;
5. how the same state becomes logical-evaluation noise;
6. how exploratory candidates are generated and activated;
7. how rewards are associated with candidates;
8. how the learned mean and covariance are updated;
9. which policy is evaluated in reported trajectories;
10. whether simulator cloning preserves or resets disturbance state.

Explicitly audit:

- reward and loss signs;
- rank direction;
- candidate-to-reward alignment;
- mask orientation and indices;
- sensitivity units;
- covariance initialization and update;
- perturbation centring;
- accidental cumulative perturbations;
- policy version and lifecycle state;
- fixed-arm immutability;
- disturbance persistence;
- logical and detector state consistency.

Create a short `artifacts/validation/current_baseline_audit.md` summarizing findings and fixes.

---

## Part B — Add canonical physical validation scenarios

Implement a reusable validation module, not report-specific ad hoc code. Prefer a package such as:

```text
src/hdfa_rl_suite/validation/
    __init__.py
    plant_sanity.py
    controller_sanity.py
    gradient_diagnostics.py
    sample_budget.py
    preflight.py
```

Add canonical scenarios:

1. no disturbance;
2. persistent single-parameter step;
3. sinusoidal local optimum with known period and phase;
4. two-state RTN with configured dwell statistics;
5. OU drift with persistent state across intervals.

For each scenario, retain interval-wise:

- latent optimum;
- applied control;
- mismatch;
- physical error components;
- detector rate by region;
- global EDR;
- logical metric;
- disturbance state identifier.

### Required plant assertions

No disturbance:

- fixed policy remains stationary within a tolerance justified by shot noise;
- oracle and fixed are statistically indistinguishable when both use the optimum.

Persistent step:

- fixed mismatch increases at onset and remains nonzero;
- fixed EDR and logical risk worsen after onset;
- oracle removes most controllable excess while retaining the irreducible floor;
- larger validated step magnitudes produce larger excess error.

Sinusoid:

- latent optimum has the configured period and phase;
- mismatch and EDR follow the expected envelope;
- interval-wise output is retained so averaging cannot hide the effect.

RTN:

- configured state labels map to reproducibly distinct detector distributions;
- empirical dwell statistics agree with the configured process within finite-sample tolerance.

OU:

- latent state persists rather than resetting;
- cloned arms receive identical paths;
- divergence between arms is caused only by control actions.

Add unit tests that deliberately break each property and prove that validation fails.

---

## Part C — Make the plant response explicit and calibrated

Refactor the control-error map if necessary so that the path is inspectable:

```text
active optimum
+ applied control
-> mismatch
-> gate/channel error parameters
-> detector emissions
-> logical-evaluation noise
```

Expose a diagnostic record for each interval.

The mapping must include:

- irreducible floor;
- controllable mismatch contribution;
- cross-coupling if enabled;
- clipping/saturation;
- the valid local operating range.

Add monotonicity tests across the validated range. If a response is intentionally asymmetric or non-monotonic, encode and test that explicitly.

Calibrate default development disturbances so that:

- fixed control degrades clearly but does not saturate;
- periodic calibration is intermediate;
- oracle is clearly best;
- the problem remains recoverable.

Do not tune the plant to make HDFA win. Tune it to produce a scientifically meaningful calibration problem.

---

## Part D — Separate learned-mean and exploration performance

Refactor metrics and artifacts so that full RL reports at least:

- held-out or independent evaluation of `mu_t`;
- per-candidate EDR and reward;
- aggregate EDR while exploratory candidates are applied;
- exploration damage relative to `mu_t` or another declared safe reference;
- covariance or exploration scale by block;
- active candidate and policy identifiers.

Do not use candidate-average EDR as the learned policy trajectory. Do not hide candidate damage behind mean-policy evaluation.

Update reports and schemas accordingly.

---

## Part E — Validate the full-control RL algorithm progressively

Add a deterministic controller validation ladder.

### E1. One-dimensional analytic objective

Use a convex objective with known optimum and exact gradient. Require convergence from both sides.

Test:

- update sign;
- reward sign;
- rank ordering;
- covariance contraction;
- bounds and slew;
- no cumulative perturbation error.

### E2. Static sparse detector objective

Create a small multi-parameter objective with a known detector-control Jacobian and sparse factor graph.

For each update, compute:

```text
estimated_gradient
true_simulator_gradient
cosine_similarity
```

Require positive mean cosine similarity with a configurable threshold. Test masks and parameter indexing.

### E3. Calibrated no-drift start

Starting from the optimum:

- the learned mean must not systematically worsen;
- covariance must remain bounded;
- exploration damage must respect the configured budget.

### E4. Single step and sinusoid

Require the learned mean to move in the correct direction and outperform fixed control after an appropriate adaptation window.

### E5. Randomized-policy recovery

Only after E1–E4 pass, test recovery from a deliberately spoiled but valid policy.

Add regression tests for every discovered bug.

---

## Part F — Validate the candidate sample budget

The current effectiveness run appears to use far fewer detector cycles per candidate than the published experiment. Do not call a reduced-budget implementation faithful unless its gradient quality is demonstrated.

Implement a short budget sweep over cycles/shots per candidate. For each budget report:

- reward-ranking accuracy against a high-shot reference;
- gradient cosine similarity;
- harmful-update probability;
- short-horizon convergence;
- final mean-policy EDR;
- runtime.

Choose the development and authoritative defaults based on explicit thresholds. Support a paper-scale configuration separately from a validated reduced-budget configuration.

The benchmark metadata must state the exact budget and whether it is:

- paper-scale;
- validated reduced budget;
- smoke-test only.

---

## Part G — Commands and tests

Add public commands resembling:

```bash
hdfa-validate-plant
hdfa-validate-full-rl
hdfa-validate-sample-budget
hdfa-benchmark-preflight
```

Each command must write machine-readable JSON and concise human-readable Markdown.

Add fast CI tests that complete in seconds. Add optional development tests marked separately. Do not make long scientific acquisitions part of ordinary test execution.

---

## Acceptance criteria

This prompt is complete only when all of the following hold:

1. fixed control demonstrably degrades under a persistent controllable disturbance;
2. oracle control removes most controllable degradation;
3. periodic recalibration gives an intermediate result at a declared cadence;
4. no-disturbance fixed and oracle policies remain stationary;
5. full RL converges on analytic and static detector objectives;
6. estimated gradients have reliably positive alignment with simulator truth;
7. the learned mean policy is evaluated separately from exploratory candidates;
8. the chosen candidate budget passes predeclared gradient-quality thresholds;
9. full RL does not regress from a calibrated no-drift start;
10. full RL recovers a simple step and tracks a slow sinusoid better than fixed control;
11. all behaviours have deterministic regression tests;
12. `hdfa-benchmark-preflight` fails closed if any prerequisite is broken.

Conclude with:

- files changed;
- root causes found;
- tests added;
- commands run;
- short results;
- remaining uncertainties;
- the exact long-run command the user may execute later.
