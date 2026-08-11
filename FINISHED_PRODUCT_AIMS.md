# Finished Product Aims

The completed staged calibration system should:

- **Automate calibration end to end**, from coarse device initialization to continuous QEC-time steering.
- **Use native QEC detector data as the primary online measurement stream**, without requiring routine interruption for dedicated characterization.
- **Infer latent physical calibration errors** from detector events, known control settings, and the detector–control factor graph.
- **Identify and separate structured fluctuation processes**, including periodic drift, random-telegraph switching, smooth drift, abrupt jumps, and nested timescales.
- **Forecast future calibration errors probabilistically**, with calibrated uncertainty and explicit model confidence.
- **Apply predictive feedforward or model-predictive corrections** before detector performance degrades substantially.
- **Use residual reinforcement learning only for unmodelled, coupled, or slowly varying errors**, rather than forcing RL to rediscover predictable dynamics.
- **Warm-start and recover calibration automatically** using physics-informed, greedy, or Bayesian local optimization before handing control to RL.
- **Reduce training and exploration cost substantially relative to a faithful Google-style detector-driven RL baseline.**
- **Reduce cumulative logical damage during adaptation**, including errors caused by delayed response and exploratory policies.
- **Maintain or improve final steady-state logical performance**, with no performance sacrifice accepted merely to simplify implementation.
- **Operate across multiple timescales**, separating fast decoding, rapid regime correction, local recovery, and slow global fine-tuning.
- **Scale through sparse locality**, so computational and experimental cost depends mainly on affected detector regions rather than total system size.
- **Handle correlated multi-qubit noise**, shared drift sources, crosstalk, and broad hardware disturbances.
- **Distinguish uncertainty from physical non-identifiability**, and avoid claiming a unique microscopic source when detector data do not support one.
- **Escalate intelligently to targeted characterization only when native QEC data are insufficient** to distinguish models requiring different corrections.
- **Enforce hard safety constraints**, including bounded updates, trust regions, rollback, exploration suppression, and safe fallback policies.
- **Detect model failure and out-of-distribution behaviour**, then revert automatically to a validated controller rather than applying unreliable predictions.
- **Provide auditable state estimates, forecasts, control decisions, confidence values, and failure diagnoses** for scientific analysis.
- **Support rigorous ablation and benchmarking** against fixed calibration, periodic recalibration, greedy calibration, HDFA-only methods, RL-only methods, and oracle-informed control.
- **Demonstrate robust generalization** to unseen drift profiles, switching rates, parameter couplings, code sizes, and noise combinations.
- **Meet real-time implementation targets**, including controller-local telemetry aggregation, low-latency inference, and substantially faster effective updates than the present 1–10 minute experimental epochs.
- **Expose modular interfaces for later decoder, code, and application adaptation**, while keeping the initial calibration controller independently testable.
- **Remain scientifically falsifiable**, with predefined acceptance criteria for recovery time, sample efficiency, logical regret, exploration damage, uncertainty calibration, and scaling.
