# Machine-readable Mathematical Contracts

- **schema_version**: `"google-pure-v8-mathematical-contracts.v1"`
- **figure5a**: `{"metric": "normalized_candidate_edr_improvement", "numerator": "fixed_edr - candidate_edr", "denominator": "fixed_edr - oracle_edr", "fixed_substitution_expected": 0.0, "oracle_substitution_expected": 1.0, "higher_is_better": true, "denominator_gate": "positive, finite, statistically resolved"}`
- **ppo**: `{"ratio": "product over i in S_j of pi_current_i / pi_behaviour_i", "behaviour_snapshot": "immutable value copy", "local_mask_only": true, "denominator_detached": true}`
- **coordinates**: `{"native": "u0 + s*x", "normalized": "(u-u0)/s", "native_sigma": "abs(s)*normalized_sigma"}`
- **temporal**: `{"optimum": "u0 + A*sin(2*pi*f*t + phase)", "frequency_unit": "cycles_per_epoch"}`
- **entropy**: `{"optimized_variable": "log_sigma", "gradient_per_coordinate": 1.0, "counted": "exactly once globally"}`
- **certification_seeds_consumed**: `false`
- **artifact_hash**: `"c55263700691f21f340621d6e2429b3c7167d4088696635a4c4d99578f424f01"`
