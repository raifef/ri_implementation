# Scientific benchmark hardening

## Compute-aware v2 amendment

The v2 acceptance path replaces the rigid 10x candidate ratio as a primary rule with a
seed-clustered, censoring-aware RMST difference in observed 90% end-to-end recovery time.
Both central arms use the same monotonic timing contract. QEC acquisition, explicit
diagnostics, actuation/acknowledgement and all online stage compute are charged;
simulator host overhead and offline Stim/PyMatching/report work are separate. A frozen
95th-percentile noninferiority safeguard protects the tail. Missing, negative,
nonmonotonic, changed-domain, excluded, overlapping or extrapolated timing evidence
fails closed. The 10x ratio remains a secondary continuity diagnostic.

Online OOD recovery is a structured evidence phase. Regional authority requires causal
locality, boundary detectors and frozen unaffected policies. Broad OOD uses all-control/
all-detector disturbance-aware recovery and is never described as stationary Stage 0.
Rollback transaction integrity and physical restoration are distinct hard failures.

The authoritative comparison is a fail-closed matched simulation experiment. It is not
Willow hardware evidence or a measured deployment result. The exact launch definition is
`experiments/physical_validation/authoritative-comparison-v1.json`; its five scenario
families, five final held-out seeds, six primary arms, 2,048 validated cycles per
candidate, and 10-microsecond simulated detector-cycle duration are content-bound into
the preflight manifest.

## Launch contract

The preflight manifest records the current source-tree and launch hashes, package,
simulator and controller versions, timestamp and maximum age, numerical thresholds,
the validated candidate-cycle floor, passing gate IDs, and all child result hashes. An
authoritative runner validates that manifest before constructing an arm or acquiring a
single sample. Missing, stale, edited, version-mismatched, underpowered, or differently
configured manifests cause exit code 3.

The preflight includes plant invariants, controller convergence and gradient direction,
sample adequacy, matched cloning, circuit-logical shared state, transactional lifecycle,
report/evidence contracts, the short held-out development cohort, Stage 2--6 numerical
equivalence, and detection of all fifteen deliberate scientific faults.

## Transactional policy lifecycle

Every Stage-0 action, probe, MPC policy, residual intervention, mean commit, rollback,
and re-entry action follows:

```text
confirmed -> proposed -> pending validation -> authorized
          -> atomically active -> acknowledged -> confirmed
```

The transaction carries the proposal and reference IDs/hashes, causal and expected
activation state IDs, projection/bounds/slew certificates, supervisor authorization,
and activation acknowledgement. A changed reference rejects the action and requires
reprojection. Rollback targets a named confirmed version. Clones have independent mutable
state, and every interval retains simulator and controller state hashes.

## Experimental accounting

Reports keep candidate count, cycles per candidate, total candidate cycles, native-QEC
cycles, mean-policy evaluation cycles, diagnostic shots and downtime, simulated elapsed
time, exploratory aggregate performance, learned-mean performance, exploration damage,
censoring status/reason, and observed threshold attainability separate. The 90% endpoint
is observed or censored; a fit cannot manufacture it. Exponential fits are secondary and
must pass the declared R-squared, residual-autocorrelation, and parameter-uncertainty
gates.

## Evidence layers

Every result is labelled as published hardware evidence, declared surrogate, executed
repository simulation, circuit-level logical adapter, or measured deployment result.
Schema validation rejects simulation described as Willow measurement, surrogate counts
described as executed controls, short probes described as convergence evidence,
candidate averages described as learned-policy performance, and complete-case superiority
claims under censoring.

## Failure taxonomy

- **Scientific invalidity (exit 3):** broken matching, stale/missing preflight, missing
  evidence, shared or mismatched state, absent logical stack, non-evaluable metrics, or a
  report-schema violation. No comparative conclusion is permitted.
- **Controller non-superiority (exit 2):** the experiment is valid and complete but one or
  more predeclared effectiveness/cost gates fail. This is retained negative evidence.
- **Lifecycle censoring or failure (exit 2):** a controller safely rolls back, reaches a
  declared censoring limit, or violates its lifecycle. Trajectories remain in the risk set
  with their reason; they are never silently dropped.
- **Ordinary completion:** all arms finish with matched state and evaluable metrics. Exit 0
  additionally requires every acceptance gate to pass; completion alone is not proof of
  superiority.

The development cohort uses seed 701, which is excluded from final seeds 101--105. It is
qualitative release validation, not an acceptance-effect estimate. Its JSON, Markdown,
and nine figures are stored under `artifacts/validation/`.
