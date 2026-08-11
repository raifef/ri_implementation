# Codex Prompt — Implement Stage 2: Latent Physical-State Inference

Read `03_Stage_2_Latent_Physical_State_Inference_Revised.docx`, the architecture overview and `00_WORKSPACE_CONTEXT.md` before coding.

## Objective

Infer a posterior over operational physical calibration variables from native detector events, exact applied controls and circuit context. The state should contain only variables whose values affect a control decision: effective detuning, amplitude/phase error, coupling/conditional phase error, transfer distortion, readout displacement, leakage proxy, shared latent factors, or other explicitly modelled quantities. Do not claim microscopic source identity unless the observation model supports it.

## Inputs

- Stage-1 event tensor, counts, exposures, context and regional views;
- applied policy/intervention sequence with activation uncertainty;
- parameter registry and detector–control graph;
- response models and priors from Stage 0;
- previous posterior and model version;
- optional regime/model beliefs from Stage 3 for iterative/joint operation.

## Outputs

Create `PhysicalStatePosterior` with:

- posterior mean/covariance and/or weighted samples;
- named latent variables, units and physical interpretation;
- local and shared components;
- observability rank/singular spectrum and unresolved null directions;
- posterior predictive detector distribution and residuals;
- attribution confidence and model-discrepancy estimate;
- validity region/horizon, OOD score and invalidity reasons;
- sufficient state for Stage 3 and Stage 4.

## Observation-model hierarchy

Implement a common interface supporting progressively richer models:

1. **Exact event likelihood** for binary detector events, optionally conditional or factorized by sparse graph structure.
2. **Count likelihood** using binomial/beta-binomial or correlated count models with exact exposures.
3. **Local empirical response model**, e.g. generalized linear/quadratic response in state and control.
4. **Circuit-level physical surrogate** mapping coherent/incoherent physical parameters to detector probabilities.
5. **Model discrepancy layer** representing residual correlations and bias rather than forcing latent variables to absorb every mismatch.

The likelihood must condition on the applied control and circuit context. A naive model of detector rates alone is not acceptable.

## Identifiability through intervention

Near an optimum, detector cost may be even in a control error, causing sign ambiguity. Use existing antithetic or orthogonal control perturbations to estimate directional response. Provide an intervention-design API that can request a safe perturbation when the current posterior has a control-relevant null direction. Quantify expected observability gain before requesting it.

Do not invent a sign from temporal smoothness when both signs remain plausible. Maintain multimodality or report ambiguity.

## Filtering algorithms

Implement at least two inference paths behind one interface:

- a low-latency Gaussian path using extended/unscented Kalman or assumed-density filtering for locally smooth identifiable cases;
- a mixture/particle path for multimodality, nonlinearity, regime uncertainty or sign ambiguity.

Allow sparse regional filtering with shared latent factors. Use numerically stable covariance updates, square-root methods where practical, and deterministic resampling under fixed seeds.

The ideal target must permit joint iteration with Stage 3. A one-way point estimate passed to segmentation is only a baseline.

## Observability and model diagnostics

Compute local Fisher information/Jacobian rank, posterior contraction and control-relevant nullspaces. Distinguish:

- physically unobservable variables;
- variables observable only under intervention;
- confounded variables requiring a changed circuit/context;
- model mismatch causing apparent non-identifiability.

Posterior predictive checks must include detector marginals, local correlations, residual temporal structure and intervention-conditioned response. Inflate uncertainty only as a temporary conservative response; persistent discrepancy must trigger model extension or diagnostic escalation.

## Minimum viable implementation

One graph-local region with 1–4 latent calibration parameters; binomial detector-count likelihood; known sparse nonlinear response; antithetic interventions resolving sign; UKF/mixture filter; observability report; calibrated uncertainty; hidden simulator truth used only for evaluation. It must improve state estimation and closed-loop recovery over detector-rate-only RL.

## Ideal full implementation

Sparse hierarchical Bayesian inference over many regions with shared common-mode factors, exact or structured event likelihoods, particle/variational smoothing, learned but physically constrained response corrections, active intervention design and joint messages with Stage 3.

## Plausible extension that may fail

Use a graph neural network or transformer to amortize the posterior. It may become overconfident, fail under new circuit contexts or encode spurious correlations. Run it in parallel with a generative likelihood, calibrate uncertainty, require OOD rejection and never let it bypass posterior predictive checks.

## Failure mechanisms and mandatory amendments

- sign/mode ambiguity → preserve mixture posterior; use antithetic intervention or targeted context change;
- too many latent variables → remove/control-relevant reparameterization, sparsity/hierarchy, observability analysis;
- wrong response map → online discrepancy model, bounded system identification, revalidation and model version rollback;
- correlated detectors treated as independent → beta-binomial/local factor/correlation model where calibration tests show undercoverage;
- filter divergence → robust likelihood, square-root updates, particles/mixtures, reinitialization from broader prior;
- policy activation uncertainty → marginalize or exclude ambiguous events;
- state absorbs unrelated noise → explicit irreducible/background and discrepancy components;
- computational cost → sparse locality, incremental updates and multi-rate inference without dropping required rich path.

Any approximation retained in the deployed path must have an ablation demonstrating no material closed-loop loss on representative scenarios.

## Validation

Test posterior coverage, state RMSE, sign accuracy, observability classification, predictive log likelihood, calibration under model mismatch, intervention value, OOD detection and runtime. Include non-identifiable cases where the correct output is ambiguity rather than false certainty. Provide offline smoothing and online filtering CLIs plus artifact plots comparing posterior, hidden truth and controls.
