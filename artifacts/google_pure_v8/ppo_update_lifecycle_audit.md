# PPO Update Lifecycle Audit

- **schema_version**: `"google-pure-v8-ppo-lifecycle.v1"`
- **on_policy_ratio_min**: `1.0`
- **on_policy_ratio_max**: `1.0`
- **shifted_ratio_min**: `0.3793107022473588`
- **shifted_ratio_max**: `2.6131144957855215`
- **behaviour_snapshot_writeable**: `false`
- **stale_replay_ratio_nontrivial**: `true`
- **multiple_pass_ratio_after_first_nontrivial**: `true`
- **outside_mask_invariance**: `true`
- **negative_advantage_clipping_exercised**: `true`
- **one_pass_on_policy_clipping_structurally_inactive**: `true`
- **v7_update_passes**: `1`
- **v7_replay_capacity_epochs**: `1`
- **classification**: `"PPO_CLIPPING_STRUCTURALLY_INACTIVE"`
- **implementation_bug**: `false`
- **scientific_description_limit**: `"PPO formula is implemented, but a fresh one-pass batch begins at ratio one; stale replay supplies nontrivial ratios."`
- **certification_seeds_consumed**: `false`
- **artifact_hash**: `"df9c2a1439c85b1538aae42ac06639b0817f4a47dd2e434c6887728a8c0e5e93"`
