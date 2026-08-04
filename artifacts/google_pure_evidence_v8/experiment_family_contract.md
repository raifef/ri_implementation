# Experiment Family Contract

- **schema_version**: `"google-pure-evidence-v8-family-contract.v1"`
- **one_primary_family_per_artifact**: `true`
- **separate_paper_runs_never_form_simultaneous_scorecard**: `true`
- **artifact_hash**: `"2fe7d9b0efd5569e156e50d525d1b6509d908e7c8b0722608a90486689673c9a"`

## Records

| experiment_family | run_family | same_run_claims | forbidden_cross_run_claims | required_identity_fields |
| --- | --- | --- | --- | --- |
| PUBLIC_ENDPOINT_DATA_REPRODUCTION | released_static_memory_endpoints | [] | ['FIGURE5A_REAL_TIME_STEERING', 'FIGURE5B_SPARSE_SCALING', 'FIGURE5C_CONVERGENCE_LAW', 'NATURAL_DRIFT_SPECTRAL_SUPPRESSION', 'RANDOMIZED_RECOVERY_AFTER_SPOIL',  | ['controller_hash', 'protocol_hash', 'plant_hash', 'graph_hash', 'seed_registry_hash', 'observable_definition', 'evaluation_budget'] |
| FIGURE5A_REAL_TIME_STEERING | synthetic_sinusoidal_candidate_stream | [] | ['PUBLIC_ENDPOINT_DATA_REPRODUCTION', 'FIGURE5B_SPARSE_SCALING', 'FIGURE5C_CONVERGENCE_LAW', 'NATURAL_DRIFT_SPECTRAL_SUPPRESSION', 'RANDOMIZED_RECOVERY_AFTER_SP | ['controller_hash', 'protocol_hash', 'plant_hash', 'graph_hash', 'seed_registry_hash', 'observable_definition', 'evaluation_budget'] |
| FIGURE5B_SPARSE_SCALING | synthetic_sparse_scaling | [] | ['PUBLIC_ENDPOINT_DATA_REPRODUCTION', 'FIGURE5A_REAL_TIME_STEERING', 'FIGURE5C_CONVERGENCE_LAW', 'NATURAL_DRIFT_SPECTRAL_SUPPRESSION', 'RANDOMIZED_RECOVERY_AFTE | ['controller_hash', 'protocol_hash', 'plant_hash', 'graph_hash', 'seed_registry_hash', 'observable_definition', 'evaluation_budget'] |
| FIGURE5C_CONVERGENCE_LAW | synthetic_local_convergence | [] | ['PUBLIC_ENDPOINT_DATA_REPRODUCTION', 'FIGURE5A_REAL_TIME_STEERING', 'FIGURE5B_SPARSE_SCALING', 'NATURAL_DRIFT_SPECTRAL_SUPPRESSION', 'RANDOMIZED_RECOVERY_AFTER | ['controller_hash', 'protocol_hash', 'plant_hash', 'graph_hash', 'seed_registry_hash', 'observable_definition', 'evaluation_budget'] |
| NATURAL_DRIFT_SPECTRAL_SUPPRESSION | synthetic_frozen_natural_ensemble | [] | ['PUBLIC_ENDPOINT_DATA_REPRODUCTION', 'FIGURE5A_REAL_TIME_STEERING', 'FIGURE5B_SPARSE_SCALING', 'FIGURE5C_CONVERGENCE_LAW', 'RANDOMIZED_RECOVERY_AFTER_SPOIL', ' | ['controller_hash', 'protocol_hash', 'plant_hash', 'graph_hash', 'seed_registry_hash', 'observable_definition', 'evaluation_budget'] |
| RANDOMIZED_RECOVERY_AFTER_SPOIL | synthetic_policy_spoil | [] | ['PUBLIC_ENDPOINT_DATA_REPRODUCTION', 'FIGURE5A_REAL_TIME_STEERING', 'FIGURE5B_SPARSE_SCALING', 'FIGURE5C_CONVERGENCE_LAW', 'NATURAL_DRIFT_SPECTRAL_SUPPRESSION' | ['controller_hash', 'protocol_hash', 'plant_hash', 'graph_hash', 'seed_registry_hash', 'observable_definition', 'evaluation_budget'] |
| STEP_RESPONSE_INJECTED_DRIFT | synthetic_persistent_optimum_step | [] | ['PUBLIC_ENDPOINT_DATA_REPRODUCTION', 'FIGURE5A_REAL_TIME_STEERING', 'FIGURE5B_SPARSE_SCALING', 'FIGURE5C_CONVERGENCE_LAW', 'NATURAL_DRIFT_SPECTRAL_SUPPRESSION' | ['controller_hash', 'protocol_hash', 'plant_hash', 'graph_hash', 'seed_registry_hash', 'observable_definition', 'evaluation_budget'] |
| PUBLIC_TABLE_REPRODUCTION | released_static_memory_tables | [] | ['PUBLIC_ENDPOINT_DATA_REPRODUCTION', 'FIGURE5A_REAL_TIME_STEERING', 'FIGURE5B_SPARSE_SCALING', 'FIGURE5C_CONVERGENCE_LAW', 'NATURAL_DRIFT_SPECTRAL_SUPPRESSION' | ['controller_hash', 'protocol_hash', 'plant_hash', 'graph_hash', 'seed_registry_hash', 'observable_definition', 'evaluation_budget'] |
