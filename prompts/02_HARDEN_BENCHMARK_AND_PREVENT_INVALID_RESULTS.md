# Codex Prompt 2 — Harden the Scientific Benchmark and Prevent Physically Invalid Comparisons

## Role and objective

Work in the `hdfa_rl_suite` repository after the physical plant and Google-style full-control RL baseline have passed the validation ladder.

Your task is to make it difficult for future code changes to produce a polished but physically invalid comparative report. Add benchmark preflight gates, lifecycle invariants, matched-state assertions, evidence-layer enforcement, regression fixtures, and reporting requirements.

Do not weaken safety, uncertainty, sample budgets, or acceptance targets merely to improve completion or runtime. Optimized implementations must be numerically equivalent to the reference within declared tolerances.

Do not run the final long acceptance acquisition. Build and run short validation cohorts, then provide the long command for the user.

## Authoritative sources

Read:

- `FINISHED_PRODUCT_AIMS.md`
- `00_WORKSPACE_CONTEXT.md`
- `ARCHITECTURE_COMPLIANCE.md`
- revised stage documents
- latest comparison report and retained artifacts
- `qec-control-physical-validation/SKILL.md`
- the new plant and full-RL validation artifacts from Prompt 1

---

## Part A — Implement an enforceable benchmark preflight

Create a preflight manifest that the scientific benchmark must require before launch.

The benchmark must refuse to run unless these are current and passing for the same source/configuration hashes:

- no-disturbance plant sanity;
- fixed/oracle step sanity;
- periodic-calibration ordering;
- disturbance persistence and matched cloning;
- logical/detector shared-state consistency;
- full-RL analytic convergence;
- full-RL static detector convergence;
- positive gradient alignment;
- calibrated-start no-regression;
- sample-budget adequacy;
- mean-policy versus exploration separation;
- policy lifecycle transaction tests;
- report schema and evidence-layer checks.

Include:

- source-tree hash;
- configuration hash;
- simulator version;
- controller version;
- validation timestamp;
- test thresholds;
- result hashes.

Any stale or missing preflight result must fail closed.

---

## Part B — Repair and formalize policy lifecycle semantics

Implement or harden the transactional lifecycle:

```text
confirmed
-> proposed
-> pending validation
-> authorized
-> atomically active
-> acknowledged
-> confirmed
```

Every proposal, probe, MPC action, residual-RL action, rollback, and Stage-0 re-entry action must carry:

- `policy_id`;
- `reference_policy_id`;
- `created_from_state_id`;
- `expected_activation_state_id`;
- projection certificate;
- bounds certificate;
- slew certificate;
- supervisor authorization;
- activation acknowledgement.

Required behaviour:

- projection and validation use the policy that will actually be active;
- if the reference changes, reject and reproject;
- activation is atomic;
- rollback restores a named confirmed version;
- fixed arms never inherit pending or active adaptive policies;
- cloned arms cannot share mutable policy objects;
- simulator and controller state hashes are retained each interval.

Add the previously failing pending-MPC/intervention-probe case as a permanent regression test. Add randomized state-machine tests covering delayed acknowledgements, re-entry, rollback, and concurrent proposals.

---

## Part C — Add matched-state and counterfactual assertions

For each scenario/seed pair, assert before disturbance onset:

- identical latent device state;
- identical baseline observations;
- identical active policy;
- identical noise RNG state or matched disturbance path;
- identical detector/logical evaluator configuration.

During the run, retain a disturbance-path identifier and confirm that arm differences arise only from allowed controller actions.

Add tests that deliberately:

- reset drift in one clone;
- alter one arm's disturbance seed;
- share mutable policy state;
- evaluate logical performance from stale state;
- apply an intervention to the wrong active policy.

The benchmark must detect and reject all of them.

---

## Part D — Strengthen baseline sanity gates

Before the full six-arm comparison, run a short baseline cohort for every disturbance family.

Required checks:

- fixed worsens after sustained controllable drift;
- oracle removes most excess;
- periodic calibration is intermediate for the chosen cadence;
- validated full RL eventually outperforms fixed for simple structured drift;
- no-disturbance controls do not regress;
- detector improvements and logical improvements are directionally consistent in the validated regime.

If any family fails, exclude it from the authoritative comparison only by explicitly declaring the plant family invalid and failing the release. Do not silently continue.

Show interval-wise trajectories for fixed, periodic, oracle, mean-policy RL, and aggregate exploratory RL.

---

## Part E — Prevent invalid sample-efficiency claims

Make recovery and efficiency metrics explicit about:

- candidate count;
- cycles per candidate;
- total native-QEC cycles;
- wall-clock time;
- mean-policy evaluations;
- exploratory candidate execution;
- censoring;
- threshold attainability.

Do not compare “epochs” when arms use materially different data per epoch without also reporting native-QEC cost.

Require observed target attainment. Do not infer 90% recovery from an exponential fit unless:

- the fit passes the declared `R^2` threshold;
- residual autocorrelation is acceptable;
- parameter uncertainty is bounded;
- extrapolation distance is limited.

Prefer observed quantile times and censoring-aware survival summaries. Keep fit-based results secondary.

---

## Part F — Harden evidence and reporting

Every report must label each result as one of:

- published hardware evidence;
- declared surrogate;
- executed repository simulation;
- circuit-level logical adapter;
- measured deployment result.

Add schema validation preventing:

- suite values being described as Willow measurements;
- surrogate parameter counts being presented as executed controls;
- short pipeline probes being interpreted as convergence studies;
- censored complete-case summaries being promoted as treatment effects;
- candidate-average performance being labelled learned-policy performance.

Required figures:

1. latent optimum and fixed mismatch;
2. fixed, periodic, oracle trajectories;
3. RL mean-policy trajectory;
4. exploratory aggregate trajectory and damage;
5. active risk set and censoring;
6. logical versus detector relation;
7. cycle/candidate budget;
8. convergence fit diagnostics when used;
9. lifecycle mode and re-entry burden.

---

## Part G — Add failure-injection regression suite

Extend the existing fault runner with scientifically targeted failures:

- reversed reward sign;
- shuffled candidate rewards;
- transposed factor-graph mask;
- wrong sensitivity units;
- oversized covariance;
- non-contracting covariance;
- cumulative perturbations;
- hidden fixed-policy update;
- disturbance reset on clone;
- stale logical-evaluation state;
- wrong policy activation reference;
- underpowered candidate budget;
- mean/candidate metric conflation;
- invalid convergence extrapolation;
- informative censoring misreported as complete-case superiority.

Each injected failure must be caught by a specific preflight or report-validation gate. Produce a matrix mapping failure to detector.

---

## Part H — Short development cohort

After all gates pass, run only a small predeclared development cohort:

- one or two seeds not used for final acceptance;
- no-disturbance;
- one step;
- one sinusoid;
- one RTN;
- short enough to finish quickly;
- enough samples to validate qualitative ordering and controller direction.

Do not tune on final held-out seeds.

Produce:

- `artifacts/validation/development_cohort.json`;
- `artifacts/validation/development_cohort.md`;
- plots required above;
- a clear pass/fail recommendation for launching the long run.

---

## Part I — Performance engineering without scientific compromise

Profile Stage 2–6 kernels, but do not reduce particle counts, scenario counts, safety checks, candidate budgets, or uncertainty propagation merely to meet latency.

Allowed optimizations include:

- vectorization;
- compiled kernels;
- cached factorizations;
- sparse operations;
- regional parallelism;
- accelerator residency;
- asynchronous forecasting with bounded staleness;
- deterministic fast-path safety checks.

For each optimized kernel, add reference-versus-optimized equivalence tests. Report numerical tolerance, latency distribution, and memory.

---

## Acceptance criteria

The work is complete only when:

1. the benchmark cannot launch without a fresh passing preflight manifest;
2. all policy actions have explicit transactional reference semantics;
3. matched arms are proven to share the same initial and disturbance states;
4. fixed/oracle/periodic/full-RL sanity ordering is checked per disturbance family;
5. learned mean and exploration performance are never conflated;
6. candidate budgets are labelled and validated;
7. invalid convergence extrapolation is rejected;
8. every targeted failure injection is caught;
9. the short development cohort is physically plausible;
10. long acceptance execution remains a separate explicit user action;
11. optimized kernels preserve reference behaviour within declared tolerance;
12. documentation explains which failures are scientific invalidity, controller failure, lifecycle censoring, or ordinary non-superiority.

Conclude with:

- files changed;
- new preflight gates;
- lifecycle fixes;
- failure-injection coverage;
- development-cohort results;
- remaining blockers;
- exact command and estimated runtime for the user-run authoritative acquisition.
