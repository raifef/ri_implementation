# Codex Prompt — Implement Stage 3: Joint HDFA and Dynamical-Model Selection

Read `04_Stage_3_Joint_HDFA_and_Dynamical_Model_Selection_Revised.docx`, `03_Stage_2_Latent_Physical_State_Inference_Revised.docx`, the overview, and workspace context.

## Objective

Build the temporal structure layer that jointly infers continuous physical state, discrete regimes, hierarchy, model identity and model parameters. Its purpose is not merely to declare a change; it must extract predictive physical structure so the controller can anticipate motion rather than rediscover it through RL.

The final target is joint inference and segmentation. Implement sequential “infer a trajectory, then segment it” only as an initialization, debugging tool and ablation. Do not make it the deployed endpoint solely because it is easier.

## Inputs

- Stage-1 causal event/count data and exact controls/context;
- Stage-2 response/observation models and current physical-state beliefs;
- candidate dynamics library and priors;
- regional graph and shared-factor structure;
- previous regime/model posterior;
- compute and latency budgets from Stage 7.

## Outputs

Return `DynamicsPosterior` containing:

- posterior over current discrete regime(s) and hierarchy levels;
- posterior change probability and segment-boundary distribution;
- dynamical model probabilities;
- model parameters such as period, phase, amplitude, OU rate, diffusion, transition matrix, dwell distribution and jump magnitude;
- continuous-state posterior jointly consistent with segmentation;
- recurring-regime identity and matching confidence;
- unknown-model probability;
- posterior predictive evidence and invalidity/OOD flags;
- messages required by Stage 4 forecasting.

## Model library

Implement a composable model bank with at least:

- constant/local stationary state;
- random walk;
- Ornstein–Uhlenbeck/mean-reverting drift;
- damped or undamped sinusoidal oscillator represented in state space;
- binary and multi-state random telegraph process;
- semi-Markov dwell-time models;
- abrupt step/change-point process;
- nested discrete processes inspired by HDFA;
- additive mixtures of periodic, switching and slow drift components;
- an explicit unknown/heavy-tailed model.

Each model must define transition density, parameter prior, forecasting operation, likelihood integration with Stage 2, and complexity/evidence accounting.

## Joint inference design

Implement a finite-model switching state-space framework first. Suitable algorithms include interacting multiple-model filtering, Rao–Blackwellized particle filtering, particle Gibbs/SMC for offline validation, or variational message passing. The code must support:

- uncertainty in state and segment boundaries influencing each other;
- deliberate control perturbations in the detector likelihood;
- uncertainty over model identity and parameters;
- hierarchical/nested regime variables;
- graph-local regimes plus optional common-mode regimes;
- causal online filtering and non-causal offline smoothing for evaluation.

For periodic dynamics, infer phase and derivative so “just past the peak” has a predictive meaning. For telegraph noise, infer state and switching/dwell statistics. For nested HDFA, allow fast discrete states whose centres/amplitudes follow slower segmented processes.

## Change and recurrence logic

Do not use a fixed residual threshold alone. Compute posterior odds/evidence for continuation versus change, include hazard/dwell priors, and calibrate false-alarm/miss costs. Match a new segment to stored regimes only after comparing physical state, response pattern, dynamics and expected control correction. A visually similar detector pattern is insufficient.

## Unknown dynamics

Maintain non-zero probability on an unknown model. If all named models fail posterior predictive checks, the output must widen uncertainty and request safe local system identification or diagnostics rather than forcing the nearest model. Unknown-model handling is a core requirement, not an optional extension.

## Minimum viable implementation

One region, a jointly inferred mixture of sinusoid, telegraph, OU and step models, using Stage-2 likelihoods; causal filtering; offline smoothing; recurring-regime memory; explicit unknown model; and forecasts that demonstrably reduce recovery cost relative to RL. Include sequential infer-then-segment as a tested ablation.

## Ideal full implementation

Sparse graph-coupled hierarchical switching state-space inference with shared latent processes, nonparametric or adaptive regime counts, semi-Markov dynamics, correlated disturbances, exact uncertainty propagation, GPU/parallel acceleration and continuous model lifecycle integration.

## Plausible extension that may fail

Use a transformer or neural state-space model to infer long-range regimes and dynamics. It may interpolate well but extrapolate poorly, violate physical phase evolution, or become overconfident. Use it as a proposal or additional model in the Bayesian bank, constrained by physical transitions and subject to OOD rejection.

## Failure mechanisms and amendments

- sequential segmentation commits too early → joint inference/smoothing and boundary uncertainty;
- sinusoid confused with drift over short data → model probabilities remain broad; active wait/intervention; no premature feedforward;
- rapid switching averaged away → shorter event windows and semi-Markov/event-level likelihood;
- hierarchy overfits noise → evidence/complexity penalty, held-out predictive checks and minimum supported timescale;
- model bank excludes truth → unknown model, residual process, safe identification and library expansion;
- regime label switching → canonicalization and control-effect-based identity;
- non-stationary period/rate → hierarchical parameter evolution rather than fixed constants;
- correlated events split into local changes → common-mode latent factors and cross-region residual monitoring;
- computational blow-up → pruning by posterior mass, Rao–Blackwellization and local graph decomposition, without removing the richer validation path.

## Validation and acceptance

Create synthetic suites containing isolated and mixed sinusoidal, telegraph, OU, steps, changing period, nested switching and unknown disturbances. Report boundary precision/recall, state/model posterior coverage, phase/period error, dwell-rate error, predictive log score, forecast calibration, false alarms, detection delay, runtime and closed-loop regret. The decisive test is whether forecasting plus Stage 5 uses materially fewer detector cycles and causes less logical damage than reactive RL.
