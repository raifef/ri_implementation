# STEP_RESPONSE_INJECTED_DRIFT Comparison

- **schema_version**: `"google-pure-evidence-v8-family-comparison.v2"`
- **claim_id**: `"step.injected_persistent_optimum"`
- **experiment_family**: `"STEP_RESPONSE_INJECTED_DRIFT"`
- **paper_reference_assets**: `[]`
- **reproduction_artifact**: `"artifacts/google_pure_evidence_v8/step_response/results.json"`
- **numerical_comparison**: `{"paper_value": "approximately 130 epochs", "paper_uncertainty": "source uncertainty unavailable", "reproduction_value": {"exponential_fit": {"credibility_gate": "R2>=0.8 and tau interior to preregistered grid", "fit_r_squared": 0.9931151589226395, "fit_sse": 0.000697542731404197, "r_infinity": 0.23537684369754497, "r_zero_minus_r_infinity": -0.24041447037481756, "tau_epochs": 270.0, "tau_profile_confidence_interval_95_epochs": [264.00905294915515, 270.0], "valid": false}, "final_residual": 0.9127621417142918, "final_response": 0.08723785828570822, "integrated_absolute_tracking_error": 128.81229838311987, "overshoot": 0.0, "pre_step_response": -0.0009983994625013819, "response_classification": "NO_SETTLING_WITHIN_HORIZON", "response_time_50_epochs": null, "response_time_63_2_epochs": null, "response_time_90_epochs": null, "settling_time_95_epochs": null, "settling_tolerance_absolute": 0.05004991997312507, "target_response": 1.0}}`
- **scientific_checklist**: `{"quantity_definition": true, "axes": "recorded in reproduction artifact", "units": "recorded in reproduction artifact", "normalization": "recorded in reproduction artifact", "run_family": "synthetic_persistent_optimum_step", "controller_mode": "hash required", "evaluation_budget": "required", "uncertainty": "required", "visual_grammar": "secondary", "scientific_conclusion": "QUALITATIVE_MATCH_ONLY"}`
- **pixel_similarity_metric_used**: `false`
- **side_by_side_composite_created**: `false`
- **composite_blocker**: `"LOCAL_PAPER_PANEL_NOT_SUPPLIED"`
- **verdict**: `"NOT_PUBLICLY_IDENTIFIABLE"`
- **prompt1_hash**: `"68c3d8f21d675a98ac09407432166ad09fc403069e9574db29a0f95b5a5ac483"`
- **artifact_hash**: `"fe08ecb3166dcd007b2fb91edb628e396fa3340a96eb9c5efcef0e0418024851"`
