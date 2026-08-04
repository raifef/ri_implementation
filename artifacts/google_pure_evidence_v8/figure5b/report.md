# Figure 5b Sparse Scaling Evidence

- **schema_version**: `"google-pure-evidence-v8-5b.v2"`
- **experiment_family**: `"FIGURE5B_SPARSE_SCALING"`
- **run_family**: `"synthetic_sparse_scaling"`
- **same_run_claims**: `[]`
- **forbidden_cross_run_claims**: `["PUBLIC_ENDPOINT_DATA_REPRODUCTION", "FIGURE5A_REAL_TIME_STEERING", "FIGURE5C_CONVERGENCE_LAW", "NATURAL_DRIFT_SPECTRAL_SUPPRESSION", "RANDOMIZED_RECOVERY_AFTER_SPOIL", "STEP_RESPONSE_INJECTED_DRIFT", "PUBLIC_TABLE_REPRODUCTION"]`
- **decoder_assistance**: `"CONTROL_ONLY"`
- **mode**: `"smoke"`
- **controller_hash**: `"analytic-scaling-model-v7"`
- **protocol_hash**: `"e408d91b6fa99361e4d8ba132fb35001f2388aa8579682377177e5ba48fa932a"`
- **plant_hash**: `"analytic-sparse-paper-anchored-surrogate"`
- **graph_hash**: `"surface-code-local-count-graph-v1"`
- **seed_registry_hash**: `"d9e4fd5c9b31cb5097d8f7cce435f18842d0ecdfd6a2f1122d40afb55f5be899"`
- **observable_definition**: `"paper-axis physical error rate versus LER, epoch colour, explicit independent floor; normalized Lambda diagnostic separate"`
- **evaluation_budget**: `{"epochs_per_distance": 128, "distances": 7, "replicates": 3}`
- **classification**: `"ANALYTIC_SCALING_MODEL"`
- **normalized_old_plot_role**: `"NORMALIZED_SPARSE_CONVERGENCE_DIAGNOSTIC"`
- **distances**: `[3, 5, 7, 9, 11, 13, 15]`
- **distance_15_control_count**: `38670`
- **paper_axis_transforms**: `{"x": "physical error rate", "y": "LER logarithmic", "colour": "epoch"}`
- **uncertainty**: `"2.5-97.5 percent replicate envelope"`
- **trajectories_share_analytic_recurrence**: `true`
- **distance_normalization_constructed**: `true`
- **prompt1_hash**: `"68c3d8f21d675a98ac09407432166ad09fc403069e9574db29a0f95b5a5ac483"`
- **evidence_gate**: `{"exact_claim_id": "figure5b.analytic_sparse_scaling", "artifact_complete": true, "mechanism_valid": true, "claim_supported": false, "paper_comparable": false, "evidence_status": "PAPER_ANCHORED_SYNTHETIC_EVIDENCE", "blocking_reasons": ["ANALYTIC_RECURRENCE_NOT_EMPIRICAL_PPO_SCALING", "PROMPT1_GATE_NOT_PASSED"], "final_evidence": false}`
- **blocking_reasons**: `["ANALYTIC_RECURRENCE_NOT_EMPIRICAL_PPO_SCALING", "PROMPT1_GATE_NOT_PASSED"]`
- **artifact_hash**: `"e2182fc64040a8bba06ccbfcca5ac833f50ad8d29658bcea2580f4095b3c1d0b"`

## Records

| distance | replicate | seed | parameters_per_gate | total_controls | epochs | initial_physical_error | final_physical_error | initial_ler | final_ler |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 0 | 15700 | 30 | 1230 | 128 | 0.00125 | 0.0005208806549078207 | 0.004876564401860117 | 0.000846779615671172 |
| 3 | 1 | 15701 | 30 | 1230 | 128 | 0.00125 | 0.0005234205780622588 | 0.004876564401860117 | 0.0008550578993758906 |
| 3 | 2 | 15702 | 30 | 1230 | 128 | 0.00125 | 0.000526249914535218 | 0.004876564401860117 | 0.0008643268704107995 |
| 5 | 0 | 15900 | 30 | 3870 | 128 | 0.00125 | 0.0005274942988619703 | 0.003405422068338071 | 0.00025591410166786867 |
| 5 | 1 | 15901 | 30 | 3870 | 128 | 0.00125 | 0.0005254256086802164 | 0.003405422068338071 | 0.000252915016103458 |
| 5 | 2 | 15902 | 30 | 3870 | 128 | 0.00125 | 0.0005255767976056668 | 0.003405422068338071 | 0.00025313340450749543 |
| 7 | 0 | 16100 | 30 | 7950 | 128 | 0.00125 | 0.0005247634668736673 | 0.0023780880365489323 | 7.386560281351437e-05 |
| 7 | 1 | 16101 | 30 | 7950 | 128 | 0.00125 | 0.0005252006086570399 | 0.0023780880365489323 | 7.411203849818524e-05 |
| 7 | 2 | 16102 | 30 | 7950 | 128 | 0.00125 | 0.0005249941889592033 | 0.0023780880365489323 | 7.399559409403922e-05 |
| 9 | 0 | 16300 | 30 | 13470 | 128 | 0.00125 | 0.0005268413307938639 | 0.001660676003176629 | 2.2086862911968885e-05 |
| 9 | 1 | 16301 | 30 | 13470 | 128 | 0.00125 | 0.0005254021540447018 | 0.001660676003176629 | 2.178683227269079e-05 |
| 9 | 2 | 16302 | 30 | 13470 | 128 | 0.00125 | 0.0005249905233093865 | 0.001660676003176629 | 2.170162051240983e-05 |
| 11 | 0 | 16500 | 30 | 20430 | 128 | 0.00125 | 0.0005253960270130054 | 0.0011596899463523944 | 6.394440054503331e-06 |
| 11 | 1 | 16501 | 30 | 20430 | 128 | 0.00125 | 0.0005247195139498977 | 0.0011596899463523944 | 6.345196963841804e-06 |
| 11 | 2 | 16502 | 30 | 20430 | 128 | 0.00125 | 0.0005261018596726694 | 0.0011596899463523944 | 6.4461563620112716e-06 |
| 13 | 0 | 16700 | 30 | 28830 | 128 | 0.00125 | 0.000526387517134025 | 0.0008098393480114487 | 1.9018131919969566e-06 |
| 13 | 1 | 16701 | 30 | 28830 | 128 | 0.00125 | 0.0005260359093425027 | 0.0008098393480114487 | 1.8929385953732017e-06 |
| 13 | 2 | 16702 | 30 | 28830 | 128 | 0.00125 | 0.0005249182046304294 | 0.0008098393480114487 | 1.8649630272697027e-06 |
| 15 | 0 | 16900 | 30 | 38670 | 128 | 0.00125 | 0.0005249001028975845 | 0.0005655302709577155 | 5.46750281604111e-07 |
| 15 | 1 | 16901 | 30 | 38670 | 128 | 0.00125 | 0.0005242109725910213 | 0.0005655302709577155 | 5.410340642470049e-07 |
| 15 | 2 | 16902 | 30 | 38670 | 128 | 0.00125 | 0.0005253331528603176 | 0.0005655302709577155 | 5.503693322020532e-07 |

## Blocking reasons

- ANALYTIC_RECURRENCE_NOT_EMPIRICAL_PPO_SCALING
- PROMPT1_GATE_NOT_PASSED
