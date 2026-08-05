# Temporal Validation

- **schema_version**: `"google-pure-v10-temporal-validation.v1"`
- **mode**: `"smoke"`
- **plan_hash**: `"3dc805c7272525efd159d98ba69a86b7a7ccfe587a8f8d0a9ed1692b7e1237d5"`
- **held_out_artifact_hash**: `"662f406a694d58a2128f2f33c4c66e4badb7f210bfa6380887702486f826ebb1"`
- **summaries**: `[{"D_exploration": -0.0003996913580246823, "D_fixed": 0.0024861111111111104, "D_tracking": 0.0027507716049382605, "I_candidate_ci_95": [-0.2203420001709268, 0.30028771468077126], "I_candidate_phase_average": 0.03997285725492223, "I_mean_ci_95": [-0.25308614993903367, 0.04087807839079301], "I_mean_ci_lower": -0.25308614993903367, "I_mean_phase_average": -0.10610403577412032, "cell_count": 3, "clipping_fraction": 0.0, "config_id": "candidate-small-scale-a", "controller": {"baseline_coefficient": 0.08, "entropy_coefficient": 0.01, "initial_scale": 0.02, "maximum_scale": 0.25, "mean_learning_rate": 0.02, "minimum_scale": 0.001, "optimizer": "plain_sgd_ascent", "ppo_clip": 0.2, "replay_capacity_epochs": 1, "scale_learning_rate": 0.01, "scale_parameterization": "log_scale", "update_passes": 1}, "entropy_operational": true, "held_out_protocol_frozen": true, "mode": "smoke", "phase_I_candidate_span": 0.4042278520539111, "phase_I_mean_span": 0.25773252959181847, "phase_count": 3, "phase_identifiable": false, "plant_hash_unchanged": true, "selection": {"blocking_reasons": ["learned_mean_positive_with_uncertainty", "tracking_gain_materially_positive", "phase_estimate_identifiable", "window_stable", "held_out_evidence_mode"], "eligible": false, "gates": {"clipping_guard": true, "entropy_axis_operational": true, "exploration_below_fixed_degradation": true, "held_out_evidence_mode": false, "held_out_protocol_frozen": true, "learned_mean_positive_with_uncertainty": false, "phase_averaging_complete": true, "phase_estimate_identifiable": false, "plant_frozen": true, "sampled_candidates_positive_phase_average": true, "tracking_gain_materially_positive": false, "window_stable": false}}, "tracking_gain": 0.0010563384784458877, "tracking_gain_ci_lower": 0.0, "window_sensitivity_max": 0.24299114867978, "window_stable": false}, {"D_exploration": 0.00031481481481480483, "D_fixed": 0.0021550925925925926, "D_tracking": 0.0024691358024691488, "I_candidate_ci_95": [-1.1929818091282829, 0.2527015618661637], "I_candidate_phase_average": -0.47014012363105956, "I_mean_ci_95": [-1.4241296146517186, 0.6687516418465607], "I_mean_ci_lower": -1.4241296146517186, "I_mean_phase_average": -0.37768898640257903, "cell_count": 3, "clipping_fraction": 0.0, "config_id": "candidate-small-scale-b", "controller": {"baseline_coefficient": 0.08, "entropy_coefficient": 0.02, "initial_scale": 0.04, "maximum_scale": 0.25, "mean_learning_rate": 0.02, "minimum_scale": 0.001, "optimizer": "plain_sgd_ascent", "ppo_clip": 0.2, "replay_capacity_epochs": 1, "scale_learning_rate": 0.01, "scale_parameterization": "log_scale", "update_passes": 1}, "entropy_operational": true, "held_out_protocol_frozen": true, "mode": "smoke", "phase_I_candidate_span": 1.2407366684992, "phase_I_mean_span": 1.6784137078255041, "phase_count": 3, "phase_identifiable": false, "plant_hash_unchanged": true, "selection": {"blocking_reasons": ["learned_mean_positive_with_uncertainty", "sampled_candidates_positive_phase_average", "tracking_gain_materially_positive", "phase_estimate_identifiable", "window_stable", "held_out_evidence_mode"], "eligible": false, "gates": {"clipping_guard": true, "entropy_axis_operational": true, "exploration_below_fixed_degradation": true, "held_out_evidence_mode": false, "held_out_protocol_frozen": true, "learned_mean_positive_with_uncertainty": false, "phase_averaging_complete": true, "phase_estimate_identifiable": false, "plant_frozen": true, "sampled_candidates_positive_phase_average": false, "tracking_gain_materially_positive": false, "window_stable": false}}, "tracking_gain": 0.003587794085994028, "tracking_gain_ci_lower": 0.0, "window_sensitivity_max": 0.719995690583908, "window_stable": false}]`
- **phase_averaging_executed**: `true`
- **complete_period_requirement_enforced**: `true`
- **one_period_window_sensitivity_executed**: `true`
- **selected_controller**: `null`
- **selected_controller_status**: `"NO_SOURCE_COMPATIBLE_CONTROLLER_IDENTIFIED"`
- **artifact_complete**: `true`
- **mechanism_valid**: `true`
- **claim_supported**: `false`
- **paper_comparable**: `false`
- **blocking_reasons**: `["held_out_evidence_mode", "learned_mean_positive_with_uncertainty", "phase_estimate_identifiable", "sampled_candidates_positive_phase_average", "tracking_gain_materially_positive", "window_stable"]`
- **certification_seeds_consumed**: `false`
- **artifact_hash**: `"3dde39ed2dd8d135a8773fd407a4b6a78ac01446213288c86bff11891e905469"`

## Blocking reasons

- held_out_evidence_mode
- learned_mean_positive_with_uncertainty
- phase_estimate_identifiable
- sampled_candidates_positive_phase_average
- tracking_gain_materially_positive
- window_stable
