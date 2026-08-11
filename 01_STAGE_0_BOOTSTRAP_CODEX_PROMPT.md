# Codex Prompt — Implement Stage 0: Bootstrap Calibration and Entry into the QEC-Operable Regime

Read `01_Stage_0_Bootstrap_Calibration_Revised.docx` in full before changing code. Also read `00_Architecture_Overview_Revised.docx` and `00_WORKSPACE_CONTEXT.md`. Implement Stage 0 as a production-quality, deterministic module rather than a demonstration script.

## Objective

Build an automated, dependency-aware bootstrap calibration system that starts from broad hardware/simulator priors and produces a validated QEC-operable baseline. It must establish enough observability and control quality for native detector data to be meaningful. Do not attempt to replace this stage with high-dimensional RL: before readout, spectroscopy, basic pulses, reset, and entangling operations are viable, the QEC objective is often saturated or non-identifying.

## Required inputs

Implement validated schemas for:

- device topology: qubits, couplers, resonators, control lines, connectivity, channel IDs, supported pulse primitives, timing constraints;
- hard limits: amplitude, frequency, phase, flux, slew, duty cycle, leakage, thermal and interlock limits;
- coarse priors: frequency bands, resonance bands, expected coupling signs/ranges, coherence ranges, previous values with provenance;
- experiment backend: simulator or hardware abstraction able to compile, execute and return timestamped observations;
- target QEC circuit and detector definitions;
- calibration configuration and reproducibility seed.

## Required outputs

Return a versioned `BootstrapResult` containing:

- complete baseline policy `u0`, units, covariance/credible intervals, timestamp and policy hash;
- compiled QEC-operable circuit and validation evidence;
- parameter registry with physical meaning, owner node, units, bounds, trust radius, local region and dependencies;
- executable calibration DAG with node states and validity intervals;
- detector–gate–control factor graph;
- detector-sensitivity normalization and uncertainty;
- rollback snapshot;
- stage-health packet with all ambiguities and unresolved nodes;
- full event/decision log sufficient for deterministic replay.

## Core implementation

### 1. Calibration DAG

Represent each calibration operation as a typed node with:

- owned parameters;
- prerequisites and invalidation edges;
- hardware resources and conflict set;
- experiment generator;
- probabilistic model or bounded optimizer;
- acquisition policy and stopping rule;
- acceptance test using independent/held-out data;
- rollback action;
- validity duration and drift sensitivity;
- failure classifications and escalation actions.

Traverse ready nodes topologically. Parallelize only nodes whose hardware resources and causal dependencies do not conflict. Any upstream change must invalidate and rerun all downstream nodes whose assumptions are affected.

### 2. Required calibration sequence

Implement at least the following node families, with simulator-compatible defaults and extension points for hardware-specific versions:

1. controller timing, channel mapping, trigger and data-integrity self-test;
2. readout-resonator discovery using multi-peak models rather than single argmax peak finding;
3. readout discrimination, assignment matrix estimation, reset verification and uncertainty;
4. qubit spectroscopy with bidirectional or interleaved sweeps and wrong-peak checks;
5. coarse single-qubit frequency, amplitude, phase and DRAG-like control;
6. coarse two-qubit/coupler operation and conditional phase;
7. transfer-function and context correction where supported;
8. assembly and validation of a detector-generating QEC circuit;
9. detector/control sensitivity normalization using bounded perturbations;
10. independent final validation not reused for fitting.

The MVP may use simplified effective parameters in the simulator, but interfaces and records must preserve physical meaning and allow richer pulse-level backends.

### 3. Statistical treatment

Do not return unqualified least-squares point fits. Each node must estimate uncertainty and compare competing hypotheses when ambiguity is plausible. Support:

- binomial/multinomial likelihoods for counts;
- robust curve fitting with outlier models;
- multi-peak spectral posteriors;
- drift terms or interleaved references when a node takes long enough for non-stationarity to matter;
- model checking on held-out observations;
- acquisition stopping based on decision confidence, not a fixed sample count alone.

The bootstrap result is invalid when competing solutions imply materially different downstream controls and the data do not resolve them.

### 4. Detector–control graph and normalization

From the compiled circuit, map each detector to measurements, detecting-region gates, and the parameters affecting those gates. Preserve broad-impact controls such as transfer-function parameters. Estimate sensitivity scales by applying safe perturbations and fitting the detector response, including uncertainty and nonlinearity diagnostics. This graph is consumed by Stages 1, 2, 5 and 6.

### 5. Determinism and systems separation

Separate pure numerical inference from backend execution. Every command, parameter patch, activation acknowledgement, observation, fit, acceptance decision and rollback must be logged. Replaying the log against recorded observations must reproduce the same result bit-for-bit, modulo explicitly documented floating-point tolerance.

## Minimum viable implementation

The Stage-0 MVP must automatically calibrate a simulated device with readout, one effective single-qubit detuning/amplitude pair per qubit, one effective entangling parameter per edge, and a QEC memory circuit. It must build the DAG, detect wrong peaks, validate with held-out data, construct the factor graph, and reach QEC viability in less than one tenth of the matched RL-only detector-cycle budget.

The MVP is not permitted to expose latent simulator truth to the controller.

## Ideal full implementation

Support joint Bayesian calibration of strongly coupled blocks, active experiment design by expected reduction in downstream logical-risk uncertainty, non-stationary fitting during calibration, distributed resource scheduling, pulse-transfer models, leakage-sensitive objectives, and hardware execution with atomic policy activation and rollback.

## Plausible extension that may fail

Add a differentiable pulse-level digital twin that proposes the entire bootstrap policy. Treat it only as a proposal mechanism. It may fail through model bias, unmodelled electronics, non-identifiability or sim-to-real shift; every proposal must still pass physical bounds and independent experimental validation.

## Failure mechanisms and mandatory amendments

Implement explicit tests and recovery for:

- wrong resonance/transition selected → multi-hypothesis posterior, bidirectional sweeps, consistency tests, independent validation;
- drift during acquisition → timestamps, interleaved references, temporal model, shorter acquisition or joint fit;
- missing DAG dependency → structured residual detection, graph amendment, downstream invalidation and replay;
- apparently good local metric but poor QEC → multi-context validation and QEC-level gate;
- coupled parameters fitted independently → joint block inference or experiment redesign; do not accept biased independent fits;
- unsafe optimizer proposal → hard pre-execution shield and trust region;
- stale previous calibration → validity windows and revalidation probes;
- failed rollback → enter Stage-7 fail-safe/bootstrap mode; never continue under assumed restoration.

Where a compromise is unavoidable, emit it as a structured `DeclaredCompromise` with affected claim, expected performance cost and planned removal. Never hide it in configuration defaults.

## Tests and acceptance

Create unit, integration and benchmark tests covering wrong peaks, drifting nodes, graph invalidation, hardware conflicts, missing data, rollback and held-out validation. Report posterior coverage, wrong-peak rate, time/sample cost to QEC viability, final detector/logical performance and replay determinism. Provide a CLI that runs a complete bootstrap scenario and writes configuration, logs, metrics and plots to a timestamped artifact directory.
