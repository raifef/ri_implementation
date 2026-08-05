# Scale and Entropy Plan

- **schema_version**: `"google-pure-v10-scale-entropy-plan.v1"`
- **mode**: `"smoke"`
- **plans**: `{"stage_a": {"controller_hash": "4bdb5cc282855cd6778234cf827c5d25734df5e55f60612ccb10b27532e3659d", "epochs": 84, "estimated_runtime": "under one minute smoke; explicit development run otherwise", "estimated_storage": "under 20 MiB", "initial_scale_grid": [0.01, 0.02, 0.04, 0.06, 0.09, 0.12, 0.14], "mode": "smoke", "periods": 5, "phases": [0.0], "protocol_hash": "bffb70c0ac256bce071f9c41a1603540755b92605685db3640e8faedacaa073c", "qec_cycles": "reported exactly by run artifact", "runs": 7, "seeds": [19101, 19102, 19103], "stage": "A_INITIAL_SCALE"}, "stage_b": {"controller_hash": "01c1a290b00b42eb8d07ea8b797947c6021522da9e6b174bbf29aa689694f7d2", "entropy_grid": [0.0, 0.0004, 0.001, 0.01, 0.02, 0.1], "epochs": "frequency-dependent", "estimated_runtime": "under two minutes smoke; explicit development run otherwise", "estimated_storage": "under 30 MiB", "initial_scales": [0.02, 0.06, 0.14], "mode": "smoke", "periods": 5, "phases": [0.0], "protocol_hash": "c0ed55c38d41d6e43beec546a8f98a56b4f023c166e055c64d6ef6c566a82822", "qec_cycles": "reported exactly by run artifact", "runs": 18, "seeds": [19101, 19102, 19103], "stage": "B_ENTROPY_OPERATIONALITY"}, "stage_c": {"controller_hash": "349de624c13ad501b82ab8f08695284cf6487f2a0f7638666c060f8abfe771e8", "epochs": "frequency-dependent", "estimated_runtime": "under one minute smoke; explicit development run otherwise", "estimated_storage": "under 20 MiB", "mean_learning_rate_frozen": 0.02, "mode": "smoke", "periods": 5, "phases": [0.0], "protocol_hash": "77276bdaff9ab3ad6db26502d53cd559709465bcf275bcc0cb8ad86070000c71", "qec_cycles": "reported exactly by run artifact", "runs": 5, "scale_learning_rate_grid": [0.001, 0.002, 0.005, 0.01, 0.02], "seeds": [19101, 19102, 19103], "stage": "C_SCALE_ADAPTATION"}}`
- **runs**: `30`
- **epochs**: `"frequency-dependent; listed by each stage"`
- **periods**: `[5, 5, 5]`
- **candidates**: `"listed by each stage"`
- **cycles**: `"reported exactly by stage artifacts"`
- **estimated_runtime**: `"smoke only by default; development requires explicit execution"`
- **estimated_memory_storage**: `"under 80 MiB smoke"`
- **seeds**: `[19101, 19102, 19103]`
- **controller_hash**: `"08f668ca5578d1519e72476ef2d5e8e671e06c41654b4cebca375917d6f70a8c"`
- **protocol_hash**: `"d2b064d6db66cd576310c2eda58adec0d25f7526f4b41ca7ce40abae859a6b84"`
- **artifact_complete**: `true`
- **mechanism_valid**: `true`
- **claim_supported**: `false`
- **paper_comparable**: `false`
- **blocking_reasons**: `["ACQUISITION_NOT_EXECUTED"]`
- **certification_seeds_consumed**: `false`
- **artifact_hash**: `"1b7686f8a73086ba6490881aacc356af41f4020241dd0dc7e07cade57805a7d2"`

## Blocking reasons

- ACQUISITION_NOT_EXECUTED
