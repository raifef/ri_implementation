# Natural-drift Spectral Run Plan

- **schema_version**: `"google-pure-v10-natural-plan.v1"`
- **experiment_family**: `"NATURAL_DRIFT_SPECTRAL_SUPPRESSION_V10"`
- **mode**: `"smoke"`
- **target_low_frequency_band**: `[0.01, 0.08]`
- **duration_epochs**: `512`
- **frequency_resolution_from_duration**: `0.001953125`
- **welch_bin_width**: `0.00390625`
- **number_of_low_frequency_bins**: `18`
- **number_of_independent_low_frequency_modes**: `35`
- **welch_segment_length**: `256`
- **welch_overlap**: `0.5`
- **number_of_independent_segments**: `3`
- **estimated_uncertainty**: `"complete plant/seed runs are resampling units; finite Welch segments reported"`
- **plant_indices**: `[0, 1]`
- **runs**: `2`
- **candidates**: `6`
- **cycles_per_candidate**: `1200`
- **estimated_qec_cycles**: `7372800`
- **estimated_runtime**: `"under two minutes smoke; long explicit user-run reference acquisition"`
- **estimated_memory_storage**: `"raw four-policy traces plus paired PSD arrays; under 50 MiB smoke"`
- **analysis_contract**: `{"detrending": "constant", "integration": "sum of one-sided Welch density times bin width", "overlap_fraction": 0.5, "positive_suppression": "10*log10(fixed_band_power/policy_band_power)", "window": "hann"}`
- **protocol_hash**: `"ae588dd2ddb8a5f1b751682f63b4d80f6c0f788ebb4d3363dbec12b923c11797"`
- **artifact_complete**: `true`
- **mechanism_valid**: `true`
- **claim_supported**: `false`
- **paper_comparable**: `false`
- **blocking_reasons**: `["ACQUISITION_NOT_EXECUTED"]`
- **certification_seeds_consumed**: `false`
- **artifact_hash**: `"faf4cae64137eae86f088cf0b18ad4ff0868f507e1b0db2ba5e46048ccadb71f"`

## Blocking reasons

- ACQUISITION_NOT_EXECUTED
