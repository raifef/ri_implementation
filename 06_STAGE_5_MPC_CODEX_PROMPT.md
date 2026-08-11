# Codex Prompt — Implement Stage 5: Predictive Feedforward and Constrained Model-Predictive Control

Read `06_Stage_5_Predictive_Feedforward_and_MPC_Revised.docx`, Stage 4, the overview and workspace context.

## Objective

Convert calibrated forecasts into a safe predictive baseline control. Correct familiar structured motion before detector performance degrades, while reserving a constrained residual subspace for Stage 6. The stage must optimize control under posterior/model uncertainty and never execute an infeasible or unjustified action.

## Inputs

- Stage-4 forecast scenarios and validity information;
- current controller-confirmed policy and activation state;
- response/optimal-control maps and detector–control graph;
- hard hardware limits, slew/duty/leakage limits and trust regions;
- QEC performance objective including local, correlation and logical-risk terms;
- Stage-6 residual statistics and proposed residual bounds;
- Stage-7 mode and authorization constraints.

## Outputs

Return a versioned `PredictiveControlPackage` containing:

- first control action and optional future trajectory;
- baseline/residual decomposition;
- predicted detector/logical-risk distribution before and after action;
- active constraints, solver status and robustness margin;
- residual projection/bounds for Stage 6;
- activation time, policy hash, rollback snapshot and expiry;
- fallback action and invalidity reasons.

## Control formulation

Implement feedforward for high-confidence invertible cases and stochastic/robust MPC for coupled or uncertain cases. A typical finite-horizon objective should combine:

- expected local detector cost;
- worst-region or tail-risk cost;
- correlation/leakage/logical-risk penalties;
- control movement and slew;
- deviation from validated policy;
- terminal uncertainty/performance;
- Stage-6 exploration risk.

Enforce hard constraints separately from the objective. Include chance or robust constraints over forecast scenarios where physical limits or detector degradation are uncertain.

Recede the horizon: execute only the first approved action, then re-infer and reforecast. Align actions to actual controller activation latency.

## Baseline and residual coordinates

Define `u = u_MPC + R δu_RL`, where `R` is a projection/basis for safe residual directions. Set residual bounds from posterior uncertainty, local curvature, forecast confidence and logical-damage budget. A confident physical forecast should narrow residual exploration; an unknown event should not silently widen it without Stage-7 authorization.

## Solver implementation

Start with a sparse quadratic/sequential quadratic program for locally quadratic response and scenario constraints. Provide nonlinear refinement or nonlinear solver hooks. Warm-start from the previous solution. Validate every proposed policy with an independent response/safety evaluation before execution.

If the solver fails or returns marginal feasibility, execute the explicit fallback—usually the last validated policy or bounded feedforward—not a partially solved vector.

## Minimum viable implementation

One- or few-region stochastic MPC over forecast scenarios for sinusoidal, telegraph, OU and step errors; hard bounds and slew; local/worst-region detector objective; residual projection; atomic package/rollback; and one-step correction of familiar structured drift. It must reduce integrated regret relative to RL-only and reactive control.

## Ideal full implementation

Sparse nonlinear or distributionally robust MPC over graph-coupled regions and common-mode processes, adaptive horizons, joint candidate/exploration scheduling, calibrated logical-risk objective, controller-adjacent solver and formal invariant verification.

## Plausible extension that may fail

Differentiable pulse-level MPC optimizing waveform shapes directly. It may be too slow or too model-biased and can exploit simulator errors. Restrict it to proposal generation until hardware-validated response and robust constraints are established.

## Failure mechanisms and amendments

- biased forecast causes proactive error → model/scenario mixture, robust penalties, trust region, predictive validation and immediate rollback;
- mean EDR decreases but logical/correlated error rises → local vector, worst-region, correlation/leakage and calibrated logical-risk objectives;
- quadratic response invalid → nonlinear evaluation/refinement or smaller verified trust region;
- solver misses deadline → precomputed regime policies, warm starts, reduced but certified scenario set; do not execute stale unverified result;
- interacting regions violate decomposition → shared constraints/common factors and joint block optimization;
- excessive conservatism erases advantage → empirically calibrate robust sets and report performance-risk frontier, but never relax hard safety;
- model uncertainty too high → hold/fallback and Stage-7 local recovery/diagnostics;
- action/policy partial update → atomic patching and confirmed hash before attributing data.

Any compromise between robustness and performance must be reported quantitatively and approved through configuration; do not hide it in penalty weights.

## Validation

Test feasibility, constraint satisfaction, action latency, predictive correction, regret, worst-region performance, solver failure, biased models, multimodal forecasts and rollback. Compare feedforward, deterministic MPC, stochastic MPC, robust MPC, RL-only and oracle. Provide a CLI that consumes recorded forecasts and produces inspectable control packages and scenario plots.
