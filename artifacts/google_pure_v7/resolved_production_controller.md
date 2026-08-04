# Resolved Production Controller

This is an open synthetic reproduction of the published Google-style RL algorithm. Google’s proprietary controller code and hardware control dynamics were unavailable.

- **schema_version:** `google-pure-v7-resolved-controller.v1`
- **controller_mode:** `source_mapped_v7_production_ppo`
- **controller_code_hash:** `3d697c1443c4e710b1d9617814679e027ee502f7754a67ac856987e9992f1130`
- **base_config_hash:** `3fc768d46d227400d95d1868b506dfddfe126befc65a68d8b88fa606ba191a58`
- **resolved_config_hash:** `0b3f2d17251ee11aa09df730553b89767a8d45390cccadc62ed3248019d66d99`
- **parameters**
  - **initial_scale:** `0.14`
  - **minimum_scale:** `0.04`
  - **maximum_scale:** `0.25`
  - **entropy_coefficient:** `0.0004`
  - **mean_learning_rate:** `0.02`
  - **scale_learning_rate:** `0.002`
  - **replay_capacity_epochs:** `1`
  - **baseline_coefficient:** `0.08`
  - **ppo_clip:** `0.2`
  - **update_passes:** `1`
  - **optimizer:** `plain_sgd_ascent`
- **selection_provenance**
  - **initial_scale:** `v6 exploration study selected 0.14 under its then-current gates`
  - **mean_learning_rate:** `v6 one-factor study selected 0.02 under its then-current gates`
  - **combined_configuration:** `preregistered v7 development candidate; not promoted as scientifically passing`
  - **hard_gate_status:** `PENDING_V7_SCIENTIFIC_TESTS`
- **all_parameters_explicit:** `True`
- **legacy_v5_defaults_used:** `False`
- **objective_mode:** `source_mapped_v7_production_ppo`
- **certification_seeds_consumed:** `False`
- **status:** `RESOLVED_FOR_DEVELOPMENT`
- **disclaimer:** `This is an open synthetic reproduction of the published Google-style RL algorithm. Google’s proprietary controller code and hardware control dynamics were unavailable.`
