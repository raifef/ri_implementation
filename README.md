# HDFA-RL Suite

A NumPy-accelerated Python research implementation of the eight-stage hierarchical discrete fluctuation autosegmentation (HDFA) and residual reinforcement-learning architecture for calibration during quantum error correction. Safety-critical numerical paths retain pure-Python reference implementations for equivalence testing and minimal environments.

The revision-2 Word documents in [`staged_calibration_architecture_revised/`](staged_calibration_architecture_revised/) are authoritative. [`00_WORKSPACE_CONTEXT.md`](00_WORKSPACE_CONTEXT.md) defines global engineering rules, and [`ARCHITECTURE_COMPLIANCE.md`](ARCHITECTURE_COMPLIANCE.md) traces the implemented rich path and the boundaries that require real hardware evidence.

## Implemented system

```text
scalable non-stationary QEC simulator
  -> Stage 0 graph/joint-block bootstrap
  -> Stage 1 causal streaming telemetry
  -> Stage 2 event/count/correlated state likelihoods
  -> Stage 3 joint composite HDFA/dynamics + fixed-lag reference
  -> Stage 4 calibrated latency-aware scenario forecasts
  -> Stage 5 multi-step robust scenario MPC
  -> Stage 6 residual block-covariance RL
  -> Stage 7 invariant-enforcing supervisor
```

`hdfa_rl_suite.product.HDFAProductController` is the primary device-owning path. It runs stationary Stage 0 at cold start (and only explicit conditions that genuinely require bootstrap), carries Stage 6 inside the ordinary Stage 1--7 interval, requires Stage-7 authorization for the MPC baseline, every residual candidate, and the residual commit, then reuses post-action telemetry as the next causal batch. Causal local OOD uses affected-plus-boundary online recovery with unaffected confirmed controls frozen; broad OOD uses a separately labelled disturbance-aware global recovery rather than pretending the active device is stationary Stage 0. Transactional rollback restoration and independent uncertainty-aware physical restoration remain separate fail-closed records. `hdfa_rl_suite.pipeline.build_default_loop()` remains the backend-neutral Stage 1--5 inference/control core. Simulator truth is absent from normal observations and is available only through an explicitly named oracle/evaluation capability.

The evaluation package runs the required fixed, periodic, greedy, state-only, sequential-HDFA, joint-HDFA, published-style 40-candidate full-control RL, predictive-only, full staged, and oracle arms. Every full-RL result records its exact cycles per candidate and labels the acquisition as `paper-scale`, `validated-reduced-budget`, or `smoke-test-only`; a reduced-budget run is not called faithful without the finite-shot budget gate. Each arm receives the same dedicated Stage-0 calibration while the latent disturbance is disarmed, followed by a held-out native-QEC baseline and a synchronized disturbance-onset boundary. A relative-time exogenous tape and counter-based circuit samples then give every arm the same disturbance realization for a scenario/seed without allowing calibration duration or adaptive acquisition to perturb the drift RNG stream.

The authoritative benchmark retains the common pre-disturbance baselines, every interval trajectory, explicit completion/censoring/missing-data status, matched-pair statistics and confidence intervals, observed 50/75/90% recovery times, excess-EDR area, worst-region recovery, censoring-aware Kaplan--Meier summaries, and exponential-fit credibility diagnostics. Stage-0 held-out block checks use a declared experiment-wide family-wise false-rejection rate, while the physical QEC-rate gate remains independent and unchanged. Candidate sample counts stop at the observed recovery endpoint, and confidence intervals use paired seeds—not scenario rows—as independent experimental units. A 90% recovery time is never extrapolated from a failed fit. Reports include an experimental-design audit, exact configuration, source revision or unversioned source-tree hash, simulator version, and logical-stack versions. Missing or non-evaluable metrics make the report non-authoritative; lifecycle violations and declared controller censoring are retained as valid negative evidence and fail the corresponding composite gates without mislabelling the experiment itself as invalid.

Logical scoring is no longer the simulator proxy alone. The evaluation-only adapter samples a named `surface_code:rotated_memory_z` circuit with Stim and decodes it using a fixed nominal detector-error model and PyMatching MWPM. The declared mapping from normalized control mismatch to circuit-level gate, data, measurement, and reset noise is recorded with every trajectory. This establishes circuit-simulation evidence, not real-QPU certification.

The Nature-2026 scalability experiment in [`experiments/nature_2026_scalability/`](experiments/nature_2026_scalability/) maps its outputs directly to Figure 5 and Supplementary Figure S8 of Sivak *et al.* It sweeps odd surface-code distances through 15 (449 physical qubits), reproduces the published 38,670-parameter structural count, fits convergence-rate invariance across distance, scans real-time steerability, and can execute bounded or full throughput probes of this suite. Published anchors, declared surrogates, and actually executed pipeline measurements are kept as separate evidence layers.

The executed controller remains graph-local: regional state, forecast, and MPC objects contain only owned one-hop controls. Stage-5 scenario evaluation is vectorized without reducing the 192/256 inference particles, 256 forecast scenarios, three control horizons, or any hard safety validation. Stationary simulator acquisition is bit-exact NumPy-vectorized, Stage-0 antithetic sensitivity is graph-coloured with neighbour-interference validation, and matched scalability arms share one validated pre-randomization bootstrap/baseline per condition. Independent distance/seed probes use atomic resumable checkpoints; the full checked-in scalability profile uses eight workers.

## Run

From the repository root with `src` on `PYTHONPATH`:

```powershell
py -m pip install -e .
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
python -m hdfa_rl_suite.cli --qubits 5 --intervals 8 --cycles 128
python -m hdfa_rl_suite.validation.preflight_cli `
  --benchmark-config experiments/physical_validation/authoritative-comparison-v1.json `
  --output artifacts/validation
python -m hdfa_rl_suite.evaluation.cli `
  --config experiments/physical_validation/authoritative-comparison-v1.json `
  --preflight-manifest artifacts/validation/benchmark-preflight-manifest.json `
  --output artifacts/acceptance/authoritative-comparison-v1.json
python -m hdfa_rl_suite.evaluation.scalability_cli --profile smoke
```

The post-v2 rollback/conditional-Stage-6 development workflow is deliberately separate
from confirmatory acquisition:

```powershell
hdfa-analyse-acceptance-v2 --output artifacts/development
hdfa-reproduce-rollbacks-v2 --output artifacts/development/rollback_reproductions
hdfa-validate-rollback-semantics --output artifacts/development/rollback_reproductions
hdfa-test-residual-rl-gating --output artifacts/development
hdfa-run-residual-rl-ablation --output artifacts/development
```

The reconstruction command verifies the lossless split SHA-256, all 480 runs, matched
initial/baseline/disturbance/evaluator evidence, truth isolation, retained lifecycle and
timing records, and exact acceptance-gate recomputation. Conditional residual RL is the
product default; `residual_activation_mode="always_on"` exists only as an explicit
development ablation.

The post-amendment recovery and confirmatory-v3 workflow is development-only until the
last command is run manually. Preregistration never acquires a v3 disturbance tape:

```powershell
hdfa-analyse-recovery-latency
hdfa-run-one-interval-development
hdfa-compare-periodic-end-to-end
hdfa-validate-rmst-support --config configs/acceptance/confirmatory-v3.yaml
hdfa-analyse-candidate-tail
hdfa-validate-report-estimators --report artifacts/development/estimator_consistency.json
hdfa-run-post-amendment-development
hdfa-preregister-confirmatory-v3
```

V3 freezes fresh seeds 5001–5024, a common 8-second RMST horizon, at least 9 seconds of
endpoint support, estimator-specific confidence intervals, conditional residual authority,
and the predictive-only versus conditional-residual primary comparison. Controller
completion and hold-policy endpoint follow-up are serialized separately. The checked-in
configuration is JSON-compatible YAML so the authoritative launcher requires no YAML
parser. The long run remains an explicit user action.

Before a development or authoritative controller comparison, run the physical-validation
ladder. Each command writes machine-readable JSON and a concise Markdown report:

```powershell
hdfa-validate-plant --output artifacts/validation
hdfa-validate-full-rl --output artifacts/validation
hdfa-validate-sample-budget --output artifacts/validation
hdfa-validate-lifecycle --output artifacts/validation
hdfa-validate-report --output artifacts/validation
hdfa-validate-performance --output artifacts/validation
hdfa-validate-fault-matrix --output artifacts/validation
hdfa-validate-development-cohort --output artifacts/validation
hdfa-benchmark-preflight `
  --benchmark-config experiments/physical_validation/authoritative-comparison-v1.json `
  --output artifacts/validation
```

The canonical plant suite verifies no-disturbance stationarity, persistent step damage,
oracle/periodic/fixed ordering, explicit monotone response, sinusoidal phase/envelope,
RTN dwell/state statistics, and persistent matched OU clones. The controller suite checks
the repaired full-control implementation on analytic, sparse detector, calibrated-start,
step, and sinusoidal objectives while reporting learned-mean and exploration metrics
separately. The default finite-shot sweep selects 2,048 cycles per candidate as its
validated reduced development budget and retains 100,000 cycles per candidate as the
paper-scale reference. These are simulator-specific preflight results, not Willow evidence.

The benchmark refuses acquisition without a fresh content-hashed manifest for the exact
source tree, launch configuration, simulator/controller versions, candidate-cycle floor,
and sixteen required scientific gates. The command returns exit code `0` only when the
report is authoritative and all five acceptance gates pass. Exit code `2` is a
scientifically valid rejection; exit code `3` means the experiment is invalid or
non-evaluable. See [`BENCHMARK_HARDENING.md`](BENCHMARK_HARDENING.md) for the failure
taxonomy, lifecycle, evidence layers, and sample-accounting contract.

For the checked-in full Nature profile:

```powershell
python -m hdfa_rl_suite.evaluation.scalability_cli `
  --config experiments/nature_2026_scalability/full-profile.json `
  --checkpoint-directory artifacts/scalability/nature-2026-full/checkpoints `
  --resume `
  --output artifacts/scalability/nature-2026-full
```

Use `--pipeline-workers 1` for uncontended latency measurements, or a bounded larger value
for faster experimental turnaround. Worker concurrency is retained on every pipeline row;
valid condition checkpoints can be resumed after interruption even when the pool is resized.
Wall time is measured without allocation tracing; absolute and baseline-subtracted peak
process memory are sampled separately. Parallel probes run every distance/seed condition
in a fresh worker process so retained allocator state cannot bias the next condition.

Installed console commands are `hdfa-stage0` through `hdfa-stage7`, `hdfa-loop`,
`hdfa-benchmark`, `hdfa-fault-injection`, `hdfa-scalability`, `hdfa-validate-plant`,
`hdfa-validate-full-rl`, `hdfa-validate-sample-budget`, `hdfa-validate-lifecycle`,
`hdfa-validate-report`, `hdfa-validate-performance`, `hdfa-validate-fault-matrix`,
`hdfa-validate-development-cohort`, and `hdfa-benchmark-preflight`.

Benchmark target failures are reported, never converted into assumed successes. Hardware pulse compilation, FPGA deployment, atomic QPU activation, and claims about real-device logical performance require a concrete backend and experimental data; the Stim/PyMatching evidence is a named reproducible circuit-level simulation, not hardware certification.

## Compute-aware confirmation v2

Both central arms carry symmetric monotonic critical-path schedules. The primary
convergence estimand is a seed-clustered Kaplan--Meier RMST difference in end-to-end
time, charging QEC acquisition, diagnostics, acknowledgements and online controller
compute; a frozen 95th-percentile safeguard protects the tail. The former rigid 10x
candidate ratio is a secondary diagnostic. Provenance includes the exact protocol-bound
launch, source, preflight and timing-environment hashes. Transaction/lifecycle failures
and uncontained physical rollback failures are separate hard gates.

The one-shot protocol is
`experiments/authoritative_acceptance/compute-aware-confirmation-v2.md`; its launch is
`compute-aware-confirmation-v2.json`. Use `--validate-only` before any held-out arm is
acquired. Development-tail profiling is available through `hdfa-development-tail`, and
deterministic repair timelines through `hdfa-post-comparison-diagnostics`.

## Public-paper Google RL reproduction v2

`hdfa_rl_suite.google_reproduction` is a clean-room, source-traceable implementation of
the local masked policy objective in Sivak *et al.* and is separate from both the legacy
Track A comparator and staged HDFA. It reproduces an open algorithm on a frozen sparse
quadratic surrogate; it does not claim Willow hardware equivalence.

Start with `hdfa-google-v2-extract-public-anchors`, `hdfa-google-v2-audit`, and
`hdfa-google-v2-validate-surrogate`. Development commands print their exact candidate and
native-QEC-cycle costs and require `--execute`. Certification additionally requires a
passing development scorecard, a frozen source hash, `--execute`, and
`--acknowledge-single-use`; a lock prevents a second held-out run. Reduced-budget and
staged-recheck commands fail closed until the paper-scale reference is certified. The
evidence ledger is in `artifacts/google_reproduction_v2/`.

## Synthetic Google-style RL reproduction v4

`hdfa_rl_suite.google_synthetic_v4` is the separately labelled follow-up authorized after
the v3 Zenodo analysis found no released action support. Zenodo constrains only the
observation layer; all action-response, drift, and recovery dynamics are frozen synthetic
plants. This is a synthetic-only algorithmic reproduction. Google hardware control
dynamics and proprietary training details were unavailable.

The gated short-development sequence is:

```powershell
hdfa-google-v4-build-plant-ensemble
hdfa-google-v4-freeze-synthetic-splits
hdfa-google-v4-validate-ppo
hdfa-google-v4-decompose-stability --epochs 96
hdfa-google-v4-validate-stability-metric
hdfa-google-v4-run-amendment-study --epochs 96
hdfa-google-v4-run-randomized-recovery --epochs 180
hdfa-google-v4-run-steering-phase --epochs 96
hdfa-google-v4-run-convergence-scaling --epochs 28
hdfa-google-v4-run-development-scorecard --epochs 120
hdfa-google-v4-freeze-certification
```

Every potentially expensive command prints candidate count, native-QEC-cycle cost,
memory/disk estimates, and whether locked seeds are touched. Certification is never
opened implicitly: `hdfa-google-v4-run-certification --confirm-open-locked-seeds` also
requires a `FROZEN_READY_UNOPENED` preregistration. The checked-in short scorecard is a
negative controller result, so certification, reduced-budget equivalence, and staged
comparison remain blocked. Broad development reruns may use the same commands with larger
`--epochs`; they do not use seeds 8101-8112.

## Pure Google-style RL repair v6

`hdfa_rl_suite.google_pure_v6` is an immutable successor to v5. It preserves v5 by hash,
migrates the stability/spectral reporting convention, and repairs the diagnosed benchmark,
PPO-sign, baseline, replay, and unit/provenance problems without importing HDFA, inference,
forecasting, MPC, or any other staged controller. Policy likelihoods are evaluated on the
unclipped latent normalized sample; the plant receives the separately retained bounded
native action.

After changing the source or console-entry configuration, reinstall from this directory so
the new commands are placed on `PATH`:

```powershell
Set-Location D:\Users\Raife\hdfa_rl_suite
python -m pip install -e .
```

The ordered development workflow is:

```powershell
hdfa-google-v6-snapshot-v5
hdfa-google-v6-migrate-v5-metric-schema
hdfa-google-v6-audit-source-compliance
hdfa-google-v6-validate-gaussian-scores
hdfa-google-v6-audit-local-ratios
hdfa-google-v6-audit-ppo-clipping
hdfa-google-v6-audit-entropy-normalization
hdfa-google-v6-audit-objective-aggregation
hdfa-google-v6-audit-baseline
hdfa-google-v6-audit-replay
hdfa-google-v6-audit-units
hdfa-google-v6-validate-quadratic-gradients
hdfa-google-v6-audit-candidate-damage
hdfa-google-v6-freeze-repaired-drift-protocol
hdfa-google-v6-run-repaired-drift-unchanged --epochs 48
hdfa-google-v6-run-sine-bandwidth --epochs 72
hdfa-google-v6-run-natural-drift-retention --epochs 96
hdfa-google-v6-run-exploration-calibration --epochs 40
hdfa-google-v6-run-hyperparameter-study --epochs 36
hdfa-google-v6-run-static-validation --epochs 64
hdfa-google-v6-run-scaling-retention --epochs 32
hdfa-google-v6-run-recovery-retention --epochs 4000
hdfa-google-v6-run-development-scorecard
hdfa-google-v6-freeze-certification
```

The unchanged v5-equivalent repaired-benchmark run is a mandatory prerequisite to tuning.
Step, sine, and one-sided strobe results are never pooled into a median, and a
non-identifiable denominator is a failure rather than zero. The held-out command is not
part of development: `hdfa-google-v6-run-certification --seed <preregistered-seed>
--confirm --authorization-phrase RUN-HELD-OUT-V6-ONCE` works only after a passing frozen preregistration. No active certification seed is used
by the checked-in development evidence.

## Pure Google-style RL timescale and scientific-gate repair v7

`hdfa_rl_suite.google_pure_v7` preserves v6 immutably and supersedes its certification
freeze without opening seeds 12101–12112. V7 separates artifact completeness, mechanism
validity, and quantitative performance. A file whose own status says `PASS` cannot satisfy
the scorecard unless its scientific thresholds also pass.

The corrected sine estimator fits
`c + a sin(omega t) + b cos(omega t)` and divides learned-mean amplitude by the moving
optimum amplitude. Forty-eight-epoch runs are smoke tests only. Full sine frequencies and
one-sided strobe dwell times are derived from a measured long-step response in dimensionless
`omega*tau` and `dwell/tau` units.

After reinstalling v0.8.0, run the deterministic integrity and smoke layer first:

```powershell
Set-Location D:\Users\Raife\hdfa_rl_suite
python -m pip install -e .
hdfa-google-v7-snapshot-v6
hdfa-google-v7-supersede-certification
hdfa-google-v7-validate-scientific-gates
hdfa-google-v7-resolve-production-controller
hdfa-google-v7-validate-sine-estimator
hdfa-google-v7-run-long-step-smoke --epochs 96
hdfa-google-v7-run-timescale-sine-smoke
hdfa-google-v7-run-development-scorecard
hdfa-google-v7-freeze-certification
```

The expected result at this stage is a blocked certification freeze. The following are
long development acquisitions and are never launched automatically; review the printed
controller hash, protocol hash, wall-time estimate, candidates, epochs, QEC cycles, memory,
disk, and seed status before adding `--execute`:

```powershell
hdfa-google-v7-run-long-step --epochs 5000 --execute
hdfa-google-v7-freeze-timescale-sine
hdfa-google-v7-run-timescale-sine --execute
hdfa-google-v7-run-timescale-strobe --execute
hdfa-google-v7-run-production-repaired-drift
hdfa-google-v7-run-replay-age-audit
hdfa-google-v7-run-natural-ablation --epochs 768 --execute
hdfa-google-v7-run-full-natural-ensemble --epochs 768 --execute
hdfa-google-v7-run-final-recovery --epochs 4000 --execute
hdfa-google-v7-run-final-scaling --epochs 64 --execute
hdfa-google-v7-run-exploration-study --execute
hdfa-google-v7-run-hyperparameter-study
hdfa-google-v7-run-development-scorecard
hdfa-google-v7-freeze-certification
```

Certification remains unavailable unless all three-layer gates pass with one controller
hash and no legacy objective. The held-out command additionally requires the exact explicit
authorization phrase; it is not part of development and is not run by tests.
# Google pure-RL v7 Figure 5 pipelines

The isolated `google_pure_v7.figure5` package reproduces the **public protocol shape** of
Sivak et al. Figure 5 with the resolved v7 pure controller and declared synthetic plants.
It does not claim access to Google's proprietary simulator or training code. Reduced runs
are labelled smoke/validation runs and cannot be presented as the 1.8-billion-cycle panel-a
protocol.

Build the contracts and inspect costs without acquiring data:

```powershell
hdfa-google-v7-resolve-production-controller
hdfa-google-v7-fig5-source-contract
hdfa-google-v7-fig5-freeze-protocols
hdfa-google-v7-fig5-seed-registry
hdfa-google-v7-fig5-plan-all
hdfa-google-v7-fig5a-acquire --dry-run
hdfa-google-v7-fig5b-acquire --dry-run
hdfa-google-v7-fig5c-acquire --dry-run
```

Run the deterministic smoke pipeline (acquisition commands resume finalized shards):

```powershell
hdfa-google-v7-fig5a-acquire
hdfa-google-v7-fig5b-acquire
hdfa-google-v7-fig5c-acquire
hdfa-google-v7-fig5-merge-all
hdfa-google-v7-fig5-validate-all
hdfa-google-v7-fig5-plot-all
hdfa-google-v7-fig5-report-all
hdfa-google-v7-fig5-status
```

The paper-scale panel-a contract is intentionally guarded:

```powershell
hdfa-google-v7-fig5a-acquire --config panel_a_reference.yaml --mode paper-scale --execute-paper-scale
hdfa-google-v7-fig5b-acquire --config panel_b_reference.yaml --mode reference
hdfa-google-v7-fig5c-acquire --config panel_c_reference.yaml --mode reference
hdfa-google-v7-fig5-merge-all --mode reference
```

Use `--max-shards N` to stop cleanly between atomic shards and `--resume` (the default) to
continue bit-for-bit. Merge rejects corrupt or duplicate shards; final plotting requires a
successful validation artifact and reads merged data only.

## Full-paper pure Google-style RL reproduction

Version 0.10 adds `google_pure_paper_reproduction`, an independent baseline workflow for
the complete set of public-data, Figure 5, natural-drift, randomized-spoil, and injected-step
claims. It uses the frozen v7 full-policy controller only. It never imports or runs the staged
inference, forecasting, MPC, supervisor, or residual-policy stack.

Build the paper contracts and replay the already-audited public Zenodo endpoints:

```powershell
Set-Location D:\Users\Raife\hdfa_rl_suite
python -m pip install -e .
hdfa-google-paper-build-source-contract
hdfa-google-paper-build-claim-registry
hdfa-google-paper-reproduce-public-data
hdfa-google-paper-values-table
hdfa-google-paper-audit-all
hdfa-google-paper-status
```

Each synthetic family has `plan`, `acquire`, `merge`, `validate`, `plot`, and `compare`
commands. For example:

```powershell
hdfa-google-paper-fig5a-plan --mode reference
hdfa-google-paper-fig5a-acquire --mode reference
hdfa-google-paper-fig5a-merge --mode reference
hdfa-google-paper-fig5a-validate --mode reference
hdfa-google-paper-fig5a-plot --mode reference
hdfa-google-paper-fig5a-compare --mode reference --paper-image D:\path\to\figure5a_crop.png
```

Replace `fig5a` with `fig5b`, `fig5c`, `natural-drift`, `randomized-recovery`, or
`step-response`. Smoke and validation figures are permanently watermarked and cannot satisfy
readiness. Paper-scale acquisition is never automatic and requires `--execute-paper-scale`.
The readiness report is multi-family: no master scalar can conceal a missing experiment.

### Bounded one-hour scientific validation

Before spending paper-scale compute, run the six-family validation profile. It uses a
six-process dynamic pool, the amended direct-sigma controller and plant identities, the
source 36,000-cycle per-candidate count, all seven scaling distances, six paired
natural-drift plants, target-relative step crossings, and censoring-aware recovery. It
reduces scan density, candidate batch size, and development replication, so its artifacts
are permanently `validation` mode with `final_evidence=false` and cannot establish paper
equivalence. Measured workstation target is 35–55 minutes when no other acquisition is
competing for the six physical cores.

```powershell
Set-Location D:\Users\Raife\hdfa_rl_suite
& ".\.codex-ri-implementation-work\.venv\Scripts\google-paper-one-hour-validation.exe" --plan-only --max-workers 6
& ".\.codex-ri-implementation-work\.venv\Scripts\google-paper-one-hour-validation.exe" --max-workers 6
```

The immutable plan and final manifest are written to
`artifacts/google_pure_paper_reproduction/reports/one_hour_validation_plan.json` and
`one_hour_validation_manifest.json`. Stop any existing paper-scale acquisition before
launching this profile; concurrent runs invalidate the workstation ETA and can exhaust
memory.

## Google pure RL v8 root-cause and evidence repair

V8 preserves all v5-v7 code and artifacts and adds two isolated layers. `google_pure_v8`
audits the mathematical and controller mechanisms; `google_pure_evidence_v8` keeps every
experiment family and claim status separate. Full/reference runs require both `--mode reference`
and `--execute`; smoke output is permanently non-claim-bearing.

```powershell
Set-Location D:\Users\Raife\hdfa_rl_suite
python -m pip install -e .
hdfa-google-v8-status
hdfa-google-v8-report-root-cause
hdfa-google-evidence-v8-build-contracts
hdfa-google-evidence-v8-status
```

Artifacts are saved under `artifacts/google_pure_v8` and
`artifacts/google_pure_evidence_v8`. The current root-cause gate blocks another Figure 5a
reference surface until exploration-floor, entropy-axis, and temporal-window failures are
resolved without inventing unpublished controller parameters.

Version 0.12 hardens this layer after scientific review. Figure 5a diagnostics now use
matched independent finite-shot accounting for fixed, oracle, oracle-with-scale, learned-mean,
and sampled-candidate policies. Evidence statuses are schema-constrained: an invalid diagnostic
cannot become final evidence, and artifact completeness is checked by the protocol preflight.
Step response is target-relative; recovery begins from an explicitly spoiled bounded policy;
Figure 5b uses distances 3, 5, 7, 9, 11, 13, and 15; and Figure 5c reports independent
phase-space and time-domain fits.

```powershell
hdfa-google-evidence-v8-validate-manifests
hdfa-google-evidence-v8-build-claim-registry
hdfa-google-evidence-v8-build-paper-comparison
hdfa-google-evidence-v8-report-hdfa-readiness
hdfa-google-evidence-v8-status
```

The exact scientific outputs include `step_response/results.json`,
`recovery/results.json`, `figure5b/report.md`, `figure5c/slopes.json`,
`paper_claim_registry.json`, and `hdfa_baseline_readiness.json`. A structurally passing
Prompt-2 protocol preflight is necessary but not sufficient for reference acquisition;
the independent Prompt-1 root-cause gate must also pass.
