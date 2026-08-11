# Codex Prompt — Implement Stage 6: Residual Detector-Driven Reinforcement Learning

Read `07_Stage_6_Residual_Detector_Driven_RL_Revised.docx`, the Google-style baseline requirements in the overview, Stage 5, and workspace context.

## Objective

Implement detector-driven reinforcement learning in residual coordinates around the Stage-5 predictive baseline. The RL layer should learn model discrepancy, weak couplings and unknown low-probability channels—not repeatedly relearn predictable drift already represented by Stages 2–5.

First implement a faithful full-control Google-style baseline for comparison. Then implement the residual version through the same tested core where possible.

## Inputs

- Stage-5 baseline control package and safe residual projection/bounds;
- candidate-labelled Stage-1 detector events/counts and contexts;
- detector–control graph, sensitivity normalization and local reward masks;
- current residual policy and replay/model versions;
- forecast confidence, regime identity and Stage-7 exploration budget;
- logical/correlation/leakage sentinels.

## Outputs

Return:

- residual policy distribution, mean/covariance/support and confidence;
- executed candidate schedule with exact antithetic/orthogonal design and exposure;
- accepted residual update and optimizer diagnostics;
- exploration damage estimate and budget state;
- candidate-conditioned response evidence for Stages 2/3;
- residual bias/model-discrepancy signal;
- replay provenance and invalidity flags.

## Baseline reproduction

Implement a configurable Gaussian parameter-exploring policy with detector-local multi-objective reward, sparse gradient masking, sensitivity normalization, entropy/exploration regulation, replay and non-stationary tracking. Verify qualitative fine-tuning and drift-tracking behaviour before claiming improvement.

## Residual policy

Use `δu ~ N(μ_R, Σ_R)` in the Stage-5 residual coordinates. The MVP may use diagonal/block-diagonal covariance, but retain an interface for graph-derived block covariance and natural-gradient/covariance adaptation. Keep the residual centre near zero; persistent large mean indicates physical-model failure and must be returned upstream.

## Candidate design and sample efficiency

Implement:

- antithetic pairs `+δ, -δ` to cancel common drift and estimate direction;
- orthogonal/Hadamard or low-discrepancy perturbations where dimension permits;
- graph colouring so non-overlapping parameter blocks can be perturbed simultaneously without detector conflict;
- adaptive candidate count based on gradient SNR/posterior uncertainty;
- sequential shot allocation and early stopping for clearly bad or resolved candidates;
- asynchronous/microbatch updates while preserving causal attribution.

The published 40-candidate fixed batch is a baseline, not a mandatory design.

## Local rewards and masking

Use per-detector/region objective vectors, not only global average EDR. Mask each residual parameter to detectors in its detecting/control region, with broad controls handled explicitly. Add correlation, leakage and calibrated logical-risk penalties. Verify that masking does not omit real crosstalk; Stage 2/3 evidence may add graph edges.

## Exploration safety

Before execution, Stage 5/7 must approve every candidate. Enforce per-candidate worst-case degradation, cumulative exploration-damage budget, slew and hardware limits. Evaluate the learned mean separately from exploratory candidates. Pause exploration on unknown/broad events.

## Replay under non-stationarity

Store regime/model/context/policy versions with every sample. Weight or exclude replay by current regime similarity and importance ratio. Never mix stale samples merely to increase batch size. Maintain a fresh online buffer plus validated recurring-regime buffers.

## Information feedback

Return intervention-response pairs and local Jacobian/curvature evidence to Stage 2. Return persistent structured residuals to Stage 3 as candidate missing dynamics. This feedback is essential: residual RL must improve the physical model over time rather than permanently compensating for it.

## Minimum viable implementation

Reproduced full-control baseline plus residual Gaussian RL using 4–16 antithetic candidates, detector masking, adaptive shots, strict exploration budgets and regime-aware replay. On structured familiar drift, the full staged system must target ≥10× sample reduction, ≥5× lower integrated excess EDR and ≥2× lower exploration damage with no final-performance loss.

## Ideal full implementation

Graph-derived block covariance, natural-gradient or evolution-strategy covariance adaptation, asynchronous controller-side updates, adaptive experiment design, local nonlinear response learning, distributed regions and low-latency FPGA/controller aggregation.

## Plausible extension that may fail

Meta-RL, transformer or learned world-model proposals from previous devices/regimes. They may transfer spurious actions or fail OOD. Use as candidate proposals under the same safe evaluation and uncertainty gates; never bypass residual bounds.

## Failure mechanisms and amendments

- residual not small → detect bias/large gradients, safely expand implicated subspace, update physical model and temporarily fall back to full-control baseline if authorized;
- exploration harms logical computation → pre-screen, antithetic/low-variance design, strict damage budget, pause/fallback;
- reward misalignment → local/correlation/leakage/logical sentinels and rollback;
- covariance collapse prevents tracking → entropy floor conditioned on drift uncertainty and periodic safe probes;
- too much covariance causes damage → posterior/curvature-scaled bounds and candidate rejection;
- stale replay bias → regime/version weighting and expiry;
- graph mask misses crosstalk → residual correlation tests and edge discovery;
- asynchronous update uses mixed device states → timestamp/regime conditioning and bounded microbatch age.

## Tests and benchmarking

Test reference fidelity, gradient estimates, antithetic variance reduction, graph colouring correctness, adaptive allocation, replay under regime changes, damage budgets, rollback, residual-to-physical feedback and compute latency. Report both physical time and detector-cycle/candidate budget. Provide CLIs for baseline reproduction, residual training, drift steering and benchmark aggregation.
