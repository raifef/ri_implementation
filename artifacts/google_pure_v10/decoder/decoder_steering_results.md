# Decoder-steering Results

- **schema_version**: `"google-pure-v10-decoder-steering.v1"`
- **result**: `{"circuit_hash": "2cb771bf396dd9c335e26ba38a8b6111bee7250dfab1cd5ec83577a33a505d04", "control_and_decoder_contributions_reported_separately": true, "control_metrics": {"mean_detector_event_rate": 0.021484375000000003, "per_step_detector_event_rate": [0.0244140625, 0.024088541666666668, 0.016276041666666668, 0.021158854166666668]}, "controller_reward_input": "detector_events_only", "decoder_metrics": {"backend": "pymatching_mwpm", "configured": true, "decode_calls": 4, "decoder_hash": "e338da1b07bffe1b42e83679032fccc8e3d753545e48acc7dd3ea8c1e5b78427", "logical_error_rate": 0.001953125, "logical_failures": 1, "parameters": {"edge_weight_scale": 1.0}, "reference_backend": true, "shots": 512, "silent_fallback_used": false}, "experiment_family": "CONTROL_PLUS_DECODER_STEERING", "hidden_logical_outcome_used_by_physical_controller": false, "logical_failures_per_step": [0, 1, 0, 0], "sequence": ["physical_control_policy", "qec_circuit_and_detector_generation", "detector_events", "decoder", "logical_prediction_and_metrics", "optional_decoder_steering"], "shots_per_step": 128, "steering_actions": [{"action": {"edge_weight_scale": 1.0}, "step": 0}, {"action": {"edge_weight_scale": 1.0}, "step": 2}], "steering_policy_hash": "94ce4ccacfe9ee4f58265e75efd5bbc06bcb83e0f8ac4a815e69bd346d3e8ce9", "steps": 4}`
- **source_defined**: `false`
- **decoder_training_data**: `null`
- **control_metrics**: `{"mean_detector_event_rate": 0.021484375000000003, "per_step_detector_event_rate": [0.0244140625, 0.024088541666666668, 0.016276041666666668, 0.021158854166666668]}`
- **decoder_metrics**: `{"backend": "pymatching_mwpm", "configured": true, "decode_calls": 4, "decoder_hash": "e338da1b07bffe1b42e83679032fccc8e3d753545e48acc7dd3ea8c1e5b78427", "logical_error_rate": 0.001953125, "logical_failures": 1, "parameters": {"edge_weight_scale": 1.0}, "reference_backend": true, "shots": 512, "silent_fallback_used": false}`
- **artifact_complete**: `true`
- **mechanism_valid**: `true`
- **claim_supported**: `false`
- **paper_comparable**: `false`
- **blocking_reasons**: `["DECODER_STEERING_NOT_SOURCE_DEFINED"]`
- **certification_seeds_consumed**: `false`
- **artifact_hash**: `"7986ee19fa66751ce37c75a717c9a6c190b95faf57d57dbb795d3a6a21987025"`

## Blocking reasons

- DECODER_STEERING_NOT_SOURCE_DEFINED
