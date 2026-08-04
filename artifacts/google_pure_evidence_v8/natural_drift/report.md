# Natural-drift Spectral Suppression

- **schema_version**: `"google-pure-evidence-v8-natural.v2"`
- **experiment_family**: `"NATURAL_DRIFT_SPECTRAL_SUPPRESSION"`
- **run_family**: `"synthetic_frozen_natural_ensemble"`
- **same_run_claims**: `[]`
- **forbidden_cross_run_claims**: `["PUBLIC_ENDPOINT_DATA_REPRODUCTION", "FIGURE5A_REAL_TIME_STEERING", "FIGURE5B_SPARSE_SCALING", "FIGURE5C_CONVERGENCE_LAW", "RANDOMIZED_RECOVERY_AFTER_SPOIL", "STEP_RESPONSE_INJECTED_DRIFT", "PUBLIC_TABLE_REPRODUCTION"]`
- **decoder_assistance**: `"CONTROL_ONLY"`
- **mode**: `"smoke"`
- **controller_hash**: `"0b3f2d17251ee11aa09df730553b89767a8d45390cccadc62ed3248019d66d99"`
- **protocol_hash**: `"43bb0abd28eacd50f1699400e35c66ede6f2302d712a298d10c05d8e184116e0"`
- **plant_hash**: `"frozen-natural-v7"`
- **graph_hash**: `"paired-local-v7"`
- **seed_registry_hash**: `"4c9e49e99bd56a8e4fb682f92713c0b98bc1f22ec1f6454ef5c4fe53bdf5a6be"`
- **observable_definition**: `"10log10(integrated fixed LF PSD / integrated policy LF PSD), positive is suppression"`
- **evaluation_budget**: `{"epochs": 256, "candidates": 12, "cycles_per_candidate": 3000}`
- **sensitivity_records**: `[{"plant_id": "instrumental_a", "detrending": "constant", "window": "hann", "band": [0.001, 0.012], "mean_low_power": 8.972362509646606e-08}, {"plant_id": "instrumental_a", "detrending": "constant", "window": "hann", "band": [0.002, 0.01], "mean_low_power": 8.972362509646606e-08}, {"plant_id": "instrumental_a", "detrending": "constant", "window": "hann", "band": [0.001, 0.02], "mean_low_power": 1.0017526987273586e-07}, {"plant_id": "instrumental_a", "detrending": "constant", "window": "boxcar", "band": [0.001, 0.012], "mean_low_power": 1.9799550118174575e-07}, {"plant_id": "instrumental_a", "detrending": "constant", "window": "boxcar", "band": [0.002, 0.01], "mean_low_power": 1.9799550118174575e-07}, {"plant_id": "instrumental_a", "detrending": "constant", "window": "boxcar", "band": [0.001, 0.02], "mean_low_power": 2.2319479774489053e-07}, {"plant_id": "instrumental_a", "detrending": "linear", "window": "hann", "band": [0.001, 0.012], "mean_low_power": 9.589530051637287e-09}, {"plant_id": "instrumental_a", "detrending": "linear", "window": "hann", "band": [0.002, 0.01], "mean_low_power": 9.589530051637287e-09}, {"plant_id": "instrumental_a", "detrending": "linear", "window": "hann", "band": [0.001, 0.02], "mean_low_power": 1.2823713518435332e-08}, {"plant_id": "instrumental_a", "detrending": "linear", "window": "boxcar", "band": [0.001, 0.012], "mean_low_power": 1.2182047789461807e-08}, {"plant_id": "instrumental_a", "detrending": "linear", "window": "boxcar", "band": [0.002, 0.01], "mean_low_power": 1.2182047789461807e-08}, {"plant_id": "instrumental_a", "detrending": "linear", "window": "boxcar", "band": [0.001, 0.02], "mean_low_power": 1.4942942067914597e-08}, {"plant_id": "common_mode_a", "detrending": "constant", "window": "hann", "band": [0.001, 0.012], "mean_low_power": 1.3283843043932154e-06}, {"plant_id": "common_mode_a", "detrending": "constant", "window": "hann", "band": [0.002, 0.01], "mean_low_power": 1.3283843043932154e-06}, {"plant_id": "common_mode_a", "detrending": "constant", "window": "hann", "band": [0.001, 0.02], "mean_low_power": 1.4787646865972338e-06}, {"plant_id": "common_mode_a", "detrending": "constant", "window": "boxcar", "band": [0.001, 0.012], "mean_low_power": 3.0511137205288007e-06}, {"plant_id": "common_mode_a", "detrending": "constant", "window": "boxcar", "band": [0.002, 0.01], "mean_low_power": 3.0511137205288007e-06}, {"plant_id": "common_mode_a", "detrending": "constant", "window": "boxcar", "band": [0.001, 0.02], "mean_low_power": 3.605020924947211e-06}, {"plant_id": "common_mode_a", "detrending": "linear", "window": "hann", "band": [0.001, 0.012], "mean_low_power": 3.9803104567480415e-07}, {"plant_id": "common_mode_a", "detrending": "linear", "window": "hann", "band": [0.002, 0.01], "mean_low_power": 3.9803104567480415e-07}, {"plant_id": "common_mode_a", "detrending": "linear", "window": "hann", "band": [0.001, 0.02], "mean_low_power": 4.926133097254148e-07}, {"plant_id": "common_mode_a", "detrending": "linear", "window": "boxcar", "band": [0.001, 0.012], "mean_low_power": 6.206991046210506e-07}, {"plant_id": "common_mode_a", "detrending": "linear", "window": "boxcar", "band": [0.002, 0.01], "mean_low_power": 6.206991046210506e-07}, {"plant_id": "common_mode_a", "detrending": "linear", "window": "boxcar", "band": [0.001, 0.02], "mean_low_power": 6.443628484397063e-07}]`
- **median_mean_suppression_db**: `1.0975470454269263`
- **median_candidate_suppression_db**: `0.3701957749161093`
- **mean_suppression_ci_95**: `[0.7267484512904123, 1.46834563956344]`
- **candidate_suppression_ci_95**: `[-0.14297510604727878, 0.8833666558794974]`
- **complete_plant_seed_runs_are_resampling_units**: `true`
- **low_frequency_identifiable**: `false`
- **old_plot_reclassification**: `"DESCRIPTIVE_LEARNED_MEAN_TRACES_ONLY"`
- **prompt1_hash**: `"68c3d8f21d675a98ac09407432166ad09fc403069e9574db29a0f95b5a5ac483"`
- **evidence_gate**: `{"exact_claim_id": "natural.low_frequency_4db", "artifact_complete": true, "mechanism_valid": false, "claim_supported": false, "paper_comparable": false, "evidence_status": "INVALID_DIAGNOSTIC", "blocking_reasons": ["INSUFFICIENT_DURATION_FOR_LOW_FREQUENCY_CLAIM", "PROMPT1_GATE_NOT_PASSED"], "final_evidence": false}`
- **blocking_reasons**: `["INSUFFICIENT_DURATION_FOR_LOW_FREQUENCY_CLAIM", "PROMPT1_GATE_NOT_PASSED"]`
- **artifact_hash**: `"315ce85e78436479c1bb8f9dc2754903495047e5c14b21794b39827013a3e2f9"`

## Records

| plant_id | seed | drift_tape_hash | fixed_low_power | mean_low_power | candidate_low_power | mean_suppression_db | candidate_suppression_db | independent_low_frequency_modes | welch_segments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| instrumental_a | 15110 | ea2e9dbd15eabb0214de06b476e6be4613534b48e817260e620f2494bd1b82bd | 1.2581759401101998e-07 | 8.972362509646606e-08 | 1.300286000021896e-07 | 1.46834563956344 | -0.14297510604727878 | 1 | 3 |
| common_mode_a | 15111 | 89e0b23245e61037f55797e6ac7352f1e228d4bc2400ce5f363979941c86c088 | 1.5703576721316095e-06 | 1.3283843043932154e-06 | 1.2813327160777149e-06 | 0.7267484512904123 | 0.8833666558794974 | 1 | 3 |

## Blocking reasons

- INSUFFICIENT_DURATION_FOR_LOW_FREQUENCY_CLAIM
- PROMPT1_GATE_NOT_PASSED
