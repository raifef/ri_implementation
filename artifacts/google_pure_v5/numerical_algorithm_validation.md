# Numerical algorithm validation

This is an open synthetic reproduction of the published Google-style RL algorithm. Google’s proprietary controller code and hardware control dynamics were unavailable.

- **schema_version:** `google-pure-v5-numerical-validation.v1`
- **status:** `PASS`
- **objective:** `-0.2877474660603268`
- **checks**
  - **gaussian_log_probability:** `True`
  - **local_likelihood_ratio:** `True`
  - **global_local_ratio_distinct:** `True`
  - **componentwise_clipped_branch:** `True`
  - **finite_difference_mean:** `True`
  - **finite_difference_log_scale:** `True`
  - **quadrature_mean_gradient:** `True`
  - **quadrature_log_scale_gradient:** `True`
  - **baseline_subtraction:** `True`
  - **baseline_optimizer_step:** `True`
  - **entropy_derivative:** `True`
  - **sparse_inactive_mean_gradient_zero:** `True`
  - **sparse_inactive_scale_only_entropy:** `True`
  - **optimizer_step:** `True`
  - **candidate_action_provenance:** `True`
  - **policy_version_lifecycle:** `True`
  - **replay_ratio_supported:** `True`
- **maximum_absolute_errors**
  - **finite_difference_mean:** `2.085798200823774e-11`
  - **finite_difference_log_scale:** `5.318964713119101e-11`
  - **gauss_hermite_mean:** `0.0`
  - **gauss_hermite_log_scale:** `1.3877787807814457e-17`
- **validated_objective:** `clip each component chi, then compose detector-local products; no sign-aware min branch`
- **certification_seeds_consumed:** `False`
- **disclaimer:** `This is an open synthetic reproduction of the published Google-style RL algorithm. Google’s proprietary controller code and hardware control dynamics were unavailable.`
