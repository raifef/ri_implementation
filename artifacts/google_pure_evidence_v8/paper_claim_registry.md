# Paper Claim Registry

- **schema_version**: `"google-pure-evidence-v8-paper-claim-registry.v2"`
- **allowed_statuses**: `["PUBLIC_DATA_EXACT_REPRODUCTION", "PAPER_ANCHORED_SYNTHETIC_MATCH", "QUALITATIVE_MATCH_ONLY", "MISMATCH", "INVALID_DIAGNOSTIC", "NOT_PUBLICLY_IDENTIFIABLE", "NOT_YET_RUN"]`
- **joint_cross_family_score**: `false`
- **joint_scorecard_validation**: `"rejects every multi-family scorecard unless paper_explicitly_simultaneous=True"`
- **decoder_and_control_claims_separate**: `true`
- **prompt1_hash**: `"68c3d8f21d675a98ac09407432166ad09fc403069e9574db29a0f95b5a5ac483"`
- **artifact_hash**: `"828c47bd463305704c6f1e87bd0b17da19ba45d0ea26cf3c686f17434f2cab09"`

## Records

| claim_id | experiment_family | run_family | paper_quantity | paper_value | paper_uncertainty | reproduction_quantity | reproduction_value | comparison_legitimacy | same_run_required |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| figure5a.real_time_sampled_candidate_steering | FIGURE5A_REAL_TIME_STEERING | synthetic_sinusoidal_candidate_stream | normalized EDR improvement surface | source panel values | source uncertainty | None | None | False | True |
| natural.low_frequency_4db | NATURAL_DRIFT_SPECTRAL_SUPPRESSION | synthetic_frozen_natural_ensemble | low-frequency suppression in dB | not locally identifiable | not locally identifiable | 10log10(integrated fixed LF PSD / integrated policy LF PSD), positive is suppression | 1.0975470454269263 | False | True |
| step.injected_persistent_optimum | STEP_RESPONSE_INJECTED_DRIFT | synthetic_persistent_optimum_step | target-relative step response | approximately 130 epochs | source uncertainty unavailable | target-relative W-weighted projection delta^T W(mu-mu_pre)/(delta^T W delta), W=I | {'exponential_fit': {'credibility_gate': 'R2>=0.8 and tau interior to preregistered grid', 'fit_r_squared': 0.9931151589226395, 'fit_sse': 0.000697542731404197, | False | True |
| recovery.spoiled_policy_90pct | RANDOMIZED_RECOVERY_AFTER_SPOIL | synthetic_policy_spoil | 90% recovery from spoiled policy | approximately 1000 epochs | source uncertainty unavailable | E(t)=L_mean(t)-L_oracle_floor; F(t)=1-E(t)/E(0); observed 90% crossing E<=0.1E0 | None | False | True |
| figure5b.sparse_scaling | FIGURE5B_SPARSE_SCALING | synthetic_sparse_scaling | physical error and LER scaling panel | paper panel supplied locally | paper graphical uncertainty | paper-axis physical error rate versus LER, epoch colour, explicit independent floor; normalized Lambda diagnostic separate | ANALYTIC_SCALING_MODEL | False | True |
| figure5c.convergence_law | FIGURE5C_CONVERGENCE_LAW | synthetic_local_convergence | distance-independent phase/time convergence rate | paper panel supplied locally | paper graphical uncertainty | x=1-Lambda/Lambda*, y=100*d_t Lambda/Lambda*; phase slope m=100 gamma plus independent log-x time fit | CONSTRUCTED_ANALYTIC_CONVERGENCE | False | True |
