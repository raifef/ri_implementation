# Codex Prompt — Implement Stage 4: Probabilistic Forecasting and Uncertainty Propagation

Read `05_Stage_4_Probabilistic_Forecasting_Revised.docx`, Stage 2 and Stage 3 documents, the overview and workspace context.

## Objective

Turn the joint physical/dynamical posterior into calibrated forecasts over future physical states, optimal controls, detector statistics and logical-risk proxies at the horizons required by Stage 5. Preserve full multimodality and model uncertainty where they affect a control decision.

## Inputs

- Stage-2/3 posterior particles, mixtures or sufficient statistics;
- model transition functions and parameter posteriors;
- proposed or nominal future control/context schedule;
- detector response models, factor graph and model discrepancy;
- control-loop timing and latency distribution;
- risk metrics and forecast horizons requested by Stage 5/7.

## Outputs

Create `ForecastBundle` with:

- timestamped horizons and activation-aligned future-state distributions;
- model-conditioned and model-averaged forecasts;
- future optimal-control distribution or correction distribution;
- detector-event/count distributions by detector and region;
- correlation/leakage/logical-risk surrogates where available;
- quantiles, covariance, samples, tail probabilities and scenario weights;
- validity horizon, calibration score, OOD/unknown-model mass;
- cached scenario tree for MPC;
- explicit reasons the forecast should not be trusted.

## Forecast engine

Support analytic propagation for linear-Gaussian oscillator/OU models and sample/mixture propagation for nonlinear, switching and hierarchical models. Forecast through the actual decision latency: estimate the state at the time a control patch will become active, not at request time.

For each posterior model/state sample:

1. propagate latent dynamics through each requested horizon;
2. include parameter uncertainty and process noise;
3. map state and future controls through the detector response model;
4. include model discrepancy and irreducible noise;
5. aggregate without collapsing distinct modes that imply different controls.

Use scenario reduction only with a measurable bound on lost control-relevant risk.

## Forecast calibration

Implement rolling-origin validation and proper scores: log score, Brier score for events/regimes, CRPS/energy score for continuous or multivariate forecasts, interval coverage and tail-event calibration. Calibrate uncertainty online by model/regime/context. A nominal 90% interval that covers 50% of outcomes is invalid even if its mean is accurate.

Maintain a validity horizon determined by predictive checks, unknown-model probability and divergence between online outcomes and forecast. Stage 7 must receive this status.

## Multiple objectives

Forecast more than global mean EDR. Include per-detector and worst-region rates, local correlations, leakage sentinels and a calibrated logical-risk surrogate. Preserve the mapping from scenario to implicated controls so Stage 5 can form robust constraints.

## Minimum viable implementation

Forecast sinusoidal, telegraph, OU and step dynamics over several control horizons using posterior mixtures/samples; align for latency; map to detector distributions; report calibrated intervals and tail risk; reject forecasts beyond their demonstrated horizon. Demonstrate accurate one-step prediction and closed-loop benefit.

## Ideal full implementation

Graph-coupled scenario forecasts with shared latent factors, adaptive scenario trees, distributional detector/logical forecasts, online calibration under regime recurrence and efficient controller-side incremental updates.

## Plausible extension that may fail

A neural generative world model could forecast high-dimensional detector streams. It may violate uncertainty calibration, fail under rare disturbances or reproduce correlations without causal control response. Keep it as an ensemble member/proposal until it beats physical models on held-out and OOD tests with calibrated tails.

## Failure mechanisms and amendments

- mean forecast hides opposite modes → preserve mixture/scenarios and optimize against both;
- latency ignored → activation-time propagation and measured latency distribution;
- uncertainty underestimated → posterior predictive calibration, discrepancy model and conformal/empirical correction where valid;
- long-horizon extrapolation unsupported → finite validity horizon and staged reforecasting;
- detector forecast good but logical risk wrong → periodic logical calibration and multi-objective sentinels;
- scenario tree too large → control-aware reduction with retained tail scenarios, never naive truncation;
- model probabilities collapse prematurely → probability floors/unknown model and evidence checks;
- context change invalidates forecast → context-conditioned models and immediate invalidation.

## Validation

Test forecast calibration, horizon-dependent error, activation-alignment, tail-event recall, model mixture preservation, scenario reduction error and compute latency. Provide plots of forecast fan/scenarios against truth and downstream regret. Add a CLI to replay a posterior trace and emit forecast artifacts independently of Stage 5.
