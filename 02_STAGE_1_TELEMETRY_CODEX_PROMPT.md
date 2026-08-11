# Codex Prompt — Implement Stage 1: Native QEC Telemetry and Causal Data Conditioning

Read `02_Stage_1_Native_QEC_Telemetry_Revised.docx`, the architecture overview, and `00_WORKSPACE_CONTEXT.md`. Implement Stage 1 as the authoritative data/provenance layer. All later inference and control claims depend on this stage being exact.

## Objective

Convert raw repeated-QEC measurement records into detector events and causal, graph-indexed statistics while preserving exact acquisition order, context, policy attribution, missingness and uncertainty. The stage must never smooth away information needed to detect fast fluctuations and must never assign an event to a control policy without evidence that the policy was active.

## Inputs

- raw measurement records with shot/cycle/channel boundaries and hardware timestamps;
- detector definition table from the compiled circuit;
- exact policy timeline including candidate ID, mean policy, perturbation, requested and acknowledged activation times, policy hash and rollback events;
- circuit context: logical basis, code distance, schedule, reset/state-preparation mode, decoder/QEC metadata;
- detector–control graph and parameter registry;
- clock calibration and data-quality metadata.

## Outputs

Produce versioned records for:

- binary detector tensor `[shot, cycle, detector]` with exposure and missing masks;
- aligned event index mapping every event to policy, intervention, context, timestamp, region and source record;
- event-level append-only stream;
- causal multiscale counts/rates/credible intervals;
- selected graph-local joint counts and correlation statistics;
- `TelemetryRegionView` objects for downstream local inference;
- data-quality and synchronization diagnostics;
- deterministic replay manifest.

## Required implementation

### 1. Exact detector construction

Implement detector parity from the ordered measurement definitions, including boundary detectors, reference parities, reset conventions, missing measurements and undefined opportunities. Verify against an independent reference implementation on randomized circuits and records. Never infer missing values as zeros.

### 2. Time and policy attribution

Model the relationship between controller, digitizer and host clocks. Use activation acknowledgements or shared triggers where available. Represent uncertainty intervals for policy activation. Events whose integration window overlaps an ambiguous transition must be flagged or excluded from causal inference, not silently assigned.

Preserve actual acquisition order, including interleaved antithetic candidates and evaluation policies. Sorting by nominal timestamp must not change the sequence when timestamps collide or clock uncertainty overlaps.

### 3. Context boundaries

Do not aggregate across changes in circuit version, logical basis, code distance, schedule, detector definitions, reset strategy, policy hash or relevant decoder feedback unless a documented statistical model explicitly conditions on that difference.

### 4. Multiscale causal summaries

Maintain event-level data and a configurable bank of causal windows. For each detector and region output exact event count and exposure, not only a floating-point rate. Compute intervals from the correct binomial or beta-binomial model. Add graph-local pair or motif counts only for detector pairs connected by circuit locality, shared controls or a configured correlation hypothesis.

Use rolling/incremental algorithms so online cost is bounded. The shortest window should resolve the target fast process subject to shot noise; longer windows support slow drift and validation. Do not choose one window and discard the rest.

### 5. Known interventions

Attach the exact control perturbation to every statistic. Provide residualization hooks but do not erase the intervention at this stage. Downstream stages must be able to evaluate the full conditional likelihood `p(D | state, control, context)`.

### 6. Graph-indexed regional views

Construct overlapping regional views from the detector–control graph. Include local detectors, controls, broad shared controls, exposures, events, context, and cross-region links. Ensure overlap does not double count global metrics; retain stable IDs.

### 7. Data-quality diagnostics

Detect and report:

- dropped/reordered records;
- detector-definition/hash mismatch;
- clock skew and activation ambiguity;
- missing channels or exposures;
- saturation, frozen bits and impossible parity patterns;
- context contamination;
- window aliasing risk;
- insufficient event count;
- backend restart or policy discontinuity.

Hard-invalid data must block Stage-2/3/6 updates through the Stage-7 health contract.

## Minimum viable implementation

The MVP computes detector events bit-for-bit, provides exact policy attribution on a simulator/replay backend, preserves event-level data, exposes 3–5 causal window scales, computes per-detector counts/intervals and graph-local correlations, and emits explicit quality flags. It must replay deterministically.

## Ideal full implementation

Perform parity construction and first-level aggregation on controller-adjacent hardware; maintain a streaming event bus; use calibrated multi-clock synchronization; support lossless compressed event storage, distributed regions, online matched filters for known dwell-time signatures and direct low-latency consumption by inference and RL.

## Plausible extension that may fail

A learned graph/event encoder may compress detector streams. It may discard rare but control-relevant structure or become context-dependent. Keep raw events and exact statistics; use the learned representation only as an additional feature until sufficiency and out-of-distribution behaviour are demonstrated.

## Failure mechanisms and amendments

- policy/event skew → hardware acknowledgements, uncertainty windows, exclusion of ambiguous intervals, activation-latency tests;
- long-window aliasing → retain events and short windows; matched/dwell-aware filters; no irreversible aggregation;
- shot-noise false alarms → exact counts/exposures, Bayesian intervals, hierarchical pooling and posterior thresholds;
- context mixing → explicit context keys and hard partitioning;
- detector correlation explosion → sparse graph-based pair selection and configurable motifs;
- storage overload → lossless bit-packing/chunking and tiered retention, never destructive removal before validated feature extraction;
- missing exposure treated as no event → exposure masks and count likelihoods;
- learned preprocessing drift → versioned models, shadow operation and raw replay.

Declare any unavoidable data loss explicitly with its effect on identifiable timescales and claims.

## Tests and CLI

Add randomized parity tests, clock/activation fault injection, missing-data tests, context-boundary tests, online/offline equivalence, window-count checks and replay determinism. Provide a CLI to ingest raw/replayed batches, validate them, emit region views and generate telemetry-quality reports.
