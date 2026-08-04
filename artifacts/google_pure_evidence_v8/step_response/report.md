# Injected-drift Step-response Results

- **schema_version**: `"google-pure-evidence-v8-step-results.v2"`
- **experiment_family**: `"STEP_RESPONSE_INJECTED_DRIFT"`
- **run_family**: `"synthetic_persistent_optimum_step"`
- **same_run_claims**: `[]`
- **forbidden_cross_run_claims**: `["PUBLIC_ENDPOINT_DATA_REPRODUCTION", "FIGURE5A_REAL_TIME_STEERING", "FIGURE5B_SPARSE_SCALING", "FIGURE5C_CONVERGENCE_LAW", "NATURAL_DRIFT_SPECTRAL_SUPPRESSION", "RANDOMIZED_RECOVERY_AFTER_SPOIL", "PUBLIC_TABLE_REPRODUCTION"]`
- **decoder_assistance**: `"CONTROL_ONLY"`
- **mode**: `"smoke"`
- **controller_hash**: `"0b3f2d17251ee11aa09df730553b89767a8d45390cccadc62ed3248019d66d99"`
- **protocol_hash**: `"fa0391e8804136590c33dedae7f1fbc2bfd461776e7c033c3f6dfbf28ff45f9a"`
- **plant_hash**: `"706df884f931b80234df5d86ca8f007d47c462d35065879c4c5a85862e9f5e74"`
- **graph_hash**: `"405510bde5f033f7fb3f8bad6b4ff659f211147936add75b8433c213de66b2db"`
- **seed_registry_hash**: `"f9a400751e5b75ab8ff9028defa75389176584a3e449d46a4b0a464e3da2ae46"`
- **observable_definition**: `"target-relative W-weighted projection delta^T W(mu-mu_pre)/(delta^T W delta), W=I"`
- **evaluation_budget**: `{"epochs": 180, "candidates": 12, "cycles_per_candidate": 3000}`
- **optimum_trajectory_stored**: `true`
- **piecewise_constant_optimum_verified**: `true`
- **projection_definition**: `"projection = delta^T W(mu-mu_pre)/(delta^T W delta), W=I"`
- **response**: `{"pre_step_response": -0.0009983994625013819, "target_response": 1.0, "final_response": 0.08723785828570822, "final_residual": 0.9127621417142918, "response_time_50_epochs": null, "response_time_63_2_epochs": null, "response_time_90_epochs": null, "settling_time_95_epochs": null, "settling_tolerance_absolute": 0.05004991997312507, "overshoot": 0.0, "integrated_absolute_tracking_error": 128.81229838311987, "exponential_fit": {"valid": false, "tau_epochs": 270.0, "tau_profile_confidence_interval_95_epochs": [264.00905294915515, 270.0], "r_infinity": 0.23537684369754497, "r_zero_minus_r_infinity": -0.24041447037481756, "fit_r_squared": 0.9931151589226395, "fit_sse": 0.000697542731404197, "credibility_gate": "R2>=0.8 and tau interior to preregistered grid"}, "response_classification": "NO_SETTLING_WITHIN_HORIZON"}`
- **final_vector_residual**: `0.32126672496661374`
- **candidate_stream_response_stored**: `true`
- **fixed_baseline_stored**: `true`
- **prompt1_hash**: `"68c3d8f21d675a98ac09407432166ad09fc403069e9574db29a0f95b5a5ac483"`
- **evidence_gate**: `{"exact_claim_id": "step.injected_persistent_optimum", "artifact_complete": true, "mechanism_valid": true, "claim_supported": false, "paper_comparable": false, "evidence_status": "SYNTHETIC_MECHANISM_EVIDENCE", "blocking_reasons": ["PROMPT1_GATE_NOT_PASSED", "SMOKE_NOT_REFERENCE_EVIDENCE"], "final_evidence": false}`
- **blocking_reasons**: `["PROMPT1_GATE_NOT_PASSED", "SMOKE_NOT_REFERENCE_EVIDENCE"]`
- **artifact_hash**: `"d49e53690141f63b0ddd4f6ff99924a5fdb691b6d87bdd7f23b89262caa1f71a"`

## Blocking reasons

- PROMPT1_GATE_NOT_PASSED
- SMOKE_NOT_REFERENCE_EVIDENCE
