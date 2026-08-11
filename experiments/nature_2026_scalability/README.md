# Nature 2026 scalability contrast

This experiment contrasts the HDFA–RL implementation with the scaling protocol in
Sivak *et al.*, “Reinforcement learning control of quantum error correction”, Nature 655,
879–884 (2026), DOI [10.1038/s41586-026-10759-2](https://doi.org/10.1038/s41586-026-10759-2).
It covers main-text Figure 5 and Supplementary Figure S8/sections VI and VIII.

It is a contrast experiment, not a claim to reproduce Google’s proprietary simulator or
learner. The report keeps three evidence layers separate:

1. `published_protocol_anchor`: equations, dimensions and numerical claims stated in the
   paper or supplement.
2. `declared_*_surrogate`: paired-seed sweeps of the published quadratic gate-error model
   and logical-error scaling law, with every additional assumption stored in the report.
3. `executed_suite_pipeline`: optional timings and outcomes from this repository’s actual
   full-control RL and predictive HDFA–residual-RL paths.

The official source-data record is
[10.5281/zenodo.17566521](https://doi.org/10.5281/zenodo.17566521), current version
[10.5281/zenodo.18896801](https://doi.org/10.5281/zenodo.18896801). Its single archive is
7,786,791,716 bytes (MD5 `ca54323082fcd0e3671d5b90ce45d85c`) and contains experimental
surface-/colour-code records. It is intentionally referenced rather than silently
downloading 7.8 GB, and it does not provide the proprietary Figure 5 simulation code.

## Paper-to-output mapping

| Paper quantity | This experiment | Definition |
|---|---|---|
| Fig. 5a; Fig. S8c,d | `fig5a-steerability.csv/.svg` | `(N_candidate-N_fixed)/(N_optimal-N_fixed)`; 1 is optimal, 0 fixed, negative harmful |
| Fig. 5b | `fig5b-scaling.csv/.svg` | odd distances 3–15; 10-cycle memory protocol; physical vs logical error during 500 epochs |
| Fig. 5c | `fig5c-convergence.csv/.svg`, `convergence-fits.csv` | `∂t(Λ/Λ*) = γ(1-Λ/Λ*)`, fitted through the origin for P=1,10,30 |
| Supplement Eq. 7 | `resource-scaling.csv/.svg` | `Ptot=[(2d²−1)+(4d²−4d)]P`; d=15, P=30 gives 38,670 |
| Native-QEC sample efficiency | `sample-efficiency.csv` | first epoch/cumulative cycles reaching 50%, 75% and 90% progress from the common initial `Λ/Λ*` to 1 |
| Actual implementation | `pipeline-probe.csv/.svg` | uninstrumented wall time, independently sampled process memory, EDR, logical sentinel and native-QEC/candidate counts |

The HDFA surrogate makes one explicit architectural hypothesis: Stage 2–5 explains 75%
of the initial structured miscalibration, leaving 25% of the parameter subspace to Stage 6.
This is a configurable sensitivity parameter, not an empirical fact. Both methods are
charged their actual declared candidate/cycle budgets, so convergence can be inspected by
epoch and by cumulative native-QEC cycles.

## Run

Fast validation:

```powershell
$env:PYTHONPATH = "src"
python -m hdfa_rl_suite.evaluation.scalability_cli --profile smoke
```

Paper-axis surrogate (d=3…15, P=1/10/30, 500 epochs, three paired seeds):

```powershell
python -m hdfa_rl_suite.evaluation.scalability_cli --profile paper --no-pipeline-probe `
  --output artifacts/scalability/nature-2026-paper
```

Full experiment, including five seeds and actual pipeline probes through 449 physical
qubits. The checked-in configuration uses eight independent worker processes; use
`--pipeline-workers 1` when uncontended per-interval latency, rather than total experiment
turnaround, is the primary measurement:

```powershell
python -m hdfa_rl_suite.evaluation.scalability_cli `
  --config experiments/nature_2026_scalability/full-profile.json `
  --checkpoint-directory artifacts/scalability/nature-2026-full/checkpoints `
  --resume `
  --output artifacts/scalability/nature-2026-full
```

For the complete effectiveness-plus-scalability comparison, use the checked-in launcher:

```powershell
& experiments/nature_2026_scalability/run_full_comparison.ps1
```

It first runs the six-arm, five-scenario, five-seed Stage-0--7 effectiveness experiment
with a stationary bootstrap, an experiment-wide `1e-4` Stage-0 block-validation
false-rejection rate, a 512-cycle matched baseline, and synchronized disturbance onset.
It then runs the full Figure-5 scalability/cost profile even when the effectiveness
hypothesis is validly rejected with exit code `2`. Scalability workers return structured
censoring or missing-data records instead of propagating worker exceptions. Both exit
codes are retained in `artifacts/comparison/nature-2026-v5/comparison-status.json`; code
`3` denotes an invalid or non-evaluable experiment.

If the v5 effectiveness report has already completed, resume only the scalability phase:

```powershell
& experiments/nature_2026_scalability/run_full_comparison.ps1 `
  -SkipEffectiveness -PipelineWorkers 8
```

The launcher reads and validates the existing effectiveness status before continuing; it
does not silently assume that the skipped phase passed. Each completed distance/seed
condition is written atomically under `scalability-and-cost/checkpoints`. Re-running the
same command resumes valid completed conditions. The worker pool may be resized on resume;
every result row records its own concurrency context so heterogeneous timing evidence
cannot be mistaken for an uncontended latency comparison. Use `-NoResume` for an expressly
fresh scalability execution.

The end-to-end probe is intentionally expensive. A bounded development run can retain the
complete Figure-5 sweep while limiting only the executed implementation probe:

```powershell
python -m hdfa_rl_suite.evaluation.scalability_cli --profile full --pipeline-probe `
  --max-pipeline-distance 7 --pipeline-epochs 1 `
  --output artifacts/scalability/nature-2026-bounded
```

The workers parallelize only independent distance/seed conditions. Each condition retains
the full 192-particle Stage-2 filter, 256-particle joint Stage-3 filter, 256 Stage-4
scenarios, three MPC horizons, native-QEC accounting, and all Stage-7 checks. Result rows
are restored to canonical order before hashing and artifact generation. Stationary
counter-based detector acquisition is NumPy-vectorized against a bit-exact scalar
reference. Stage-0 sensitivity uses conflict-free graph-coloured antithetic batches with
the same per-control exposure, local Jacobian columns, neighbour sentinels and an explicit
family-wise interference gate. One common Stage 0 and held-out baseline is executed per
distance/seed and its validated device state is cloned for the two counterfactual arms;
full resource cost and actual shared execution fraction are both retained.

Every output directory contains tidy CSV files, a complete JSON report, SVG plots, and a
SHA-256 manifest covering final outputs and in-tree condition checkpoints. Executed pipeline arms receive a common stationary Stage 0, held-out
baseline and synchronized disturbance onset. `pipeline-failures.csv` distinguishes a
scientifically observed censor (`2`) from missing worker data (`3`). A failed empirical
gate makes the CLI return status `2`; inconclusive and contrast-only results remain
visible rather than being coerced into passes.

Wall time is collected with Python allocation tracing disabled. A low-overhead sampler
records absolute and baseline-subtracted peak process resident memory independently. Each
parallel distance/seed condition runs in a fresh process, so allocator state retained by a
previous condition cannot contaminate its absolute RSS. The baseline-subtracted field is
retained as a supplemental transient-allocation diagnostic.

## Interpretation limits

- The surrogate is useful for testing sample/compute scaling and convergence-law
  falsifiability; it is not a Willow hardware performance prediction.
- The suite’s current device backend has a sparse line graph with `2Q−1` implemented
  controls. The exact paper-equivalent P=30 count is shown beside it and never relabelled as
  an executed 38,670-control hardware test.
- Logical error in the surrogate follows the paper’s stated scaling model. Only the
  optional pipeline probe uses this repository’s simulated logical-failure sentinel.
- Confidence comes from paired seeds and reported dispersion. It does not replace device
  repetitions, decoder uncertainty, or a Stim circuit-level reproduction.
