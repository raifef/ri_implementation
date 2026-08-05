# v10 Root-cause Update

- **schema_version**: `"google-pure-v10-root-cause-update.v1"`
- **corrected_fault_contract_hash**: `"59f3a903d9caeb5b834fcf2947a1eea0dc24fe9df20876d9ee0f20bc7ea34abe"`
- **entropy_operational**: `true`
- **selected_controller**: `null`
- **controller_selected**: `false`
- **temporal_phase_window_stable**: `false`
- **natural_drift**: `{"frequency_resolution": 0.001953125, "median_candidate_suppression_db": -0.6136590729036713, "median_mean_suppression_db": 0.41975115867316337, "paper_comparable": false}`
- **decoder**: `{"control_only_and_decoder_assisted_separate": true, "validated": true}`
- **step_response**: `{"classification": "NO_SETTLING_WITHIN_HORIZON", "clipping_classification": "PPO_CLIPPING_CAUSAL_ROLE_TESTED", "learning_rate_classification": "LEARNING_RATE_CAUSAL_ROLE_TESTED_AT_DEVELOPMENT_SCALE", "tau_ci_95_epochs": [191.78808118971256, 270.0], "tau_epochs": 270.0}`
- **remaining_public_information_limits**: `["entropy reward balance is not source identifiable", "decoder steering is not source defined", "reference-scale held-out acquisition requires explicit execution"]`
- **pure_controller_ready_for_external_comparison**: `false`
- **comparative_benchmark_preflight_pass**: `true`
- **full_benchmark_permitted**: `false`
- **artifact_complete**: `true`
- **mechanism_valid**: `true`
- **claim_supported**: `false`
- **paper_comparable**: `false`
- **blocking_reasons**: `["FULL_SIX_PLANT_UNCERTAINTY_NOT_ACQUIRED", "HELD_OUT_ABLATION_REQUIRED", "SMOKE_NOT_HELD_OUT_REFERENCE_EVIDENCE", "SMOKE_NOT_REFERENCE_EVIDENCE", "held_out_evidence_mode", "learned_mean_positive_with_uncertainty", "phase_estimate_identifiable", "sampled_candidates_positive_phase_average", "tracking_gain_materially_positive", "window_stable"]`
- **certification_seeds_consumed**: `false`
- **artifact_hash**: `"27efd3bb6d1a0c8efd90eb3a704805aeffbac317121608b905d3291bbbcdf30a"`

## Blocking reasons

- FULL_SIX_PLANT_UNCERTAINTY_NOT_ACQUIRED
- HELD_OUT_ABLATION_REQUIRED
- SMOKE_NOT_HELD_OUT_REFERENCE_EVIDENCE
- SMOKE_NOT_REFERENCE_EVIDENCE
- held_out_evidence_mode
- learned_mean_positive_with_uncertainty
- phase_estimate_identifiable
- sampled_candidates_positive_phase_average
- tracking_gain_materially_positive
- window_stable
