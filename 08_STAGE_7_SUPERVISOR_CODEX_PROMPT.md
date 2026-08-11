# Codex Prompt — Implement Stage 7: Supervisory Control, Safety, Escalation and Continual Learning

Read `08_Stage_7_Supervisory_Control_Revised.docx`, all stage contracts, the architecture overview and workspace context.

## Objective

Implement the explicit supervisory layer that authorizes, limits, rejects or rolls back all inference and control actions. It is not another optimizer. It decides which operating mode is justified by current evidence, enforces hard invariants, manages model lifecycle, and requests targeted characterization only when native QEC information cannot resolve a control-relevant ambiguity.

## Inputs

- health packets from Stages 0–6: validity, uncertainty, OOD, latency, convergence, solver and damage state;
- controller-confirmed current policy, baseline/residual decomposition, activation status and rollback snapshots;
- detector/local/correlation/leakage/logical performance streams;
- forecast validity and unknown-model probability;
- hardware limits/interlocks and governance configuration;
- diagnostic experiment catalogue with cost and expected decision value.

## Outputs

Produce:

- authorized operating mode;
- approved/clipped/delayed/rejected/rollback control decision with reason;
- transition record with evidence, thresholds, versions and expected exit condition;
- exploration/downtime/diagnostic budgets;
- targeted characterization request when necessary;
- model promotion/quarantine/archive decisions;
- decoder/workload advisory where supported;
- complete auditable research log.

## Operating modes

Implement at minimum:

- `BOOTSTRAP`;
- `NOMINAL_PREDICTIVE`;
- `RESIDUAL_LEARNING`;
- `LOCAL_RECOVERY`;
- `UNKNOWN_EVENT`;
- `DIAGNOSTIC`;
- `DEGRADED`;
- `FAIL_SAFE`.

Define exactly which stages/actions are permitted in each mode, entry/exit conditions, maximum dwell, fallback and required evidence.

## Hard invariants

Enforce before and after every action:

- parameter/hardware/interlock bounds;
- pulse compiler feasibility;
- slew, duty, thermal and leakage limits;
- policy patch atomicity and confirmed hash;
- valid Stage-1 attribution/data quality;
- available rollback snapshot;
- per-candidate and cumulative exploration damage;
- solver/forecast validity and non-expiry;
- no learned component may override an invariant.

Changes to invariants are governance/configuration changes, not optimizer actions.

## Transition logic

Use posterior probabilities and trends with separate entry/exit thresholds, minimum dwell and transition costs. Preserve immediate fail-fast paths for hard safety. Calibrate thresholds from labelled fault-injection operating curves and explicit asymmetric costs. Avoid chattering without delaying catastrophic alarms.

## Monitoring hierarchy

1. data integrity;
2. inference observability/OOD/model validity;
3. forecast calibration and validity horizon;
4. control feasibility and solver health;
5. RL covariance, residual bias and exploration damage;
6. detector/local/correlation/leakage and logical performance;
7. divergence between surrogate and logical metrics;
8. cross-region/common-mode event evidence.

## Targeted characterization

Request an extra experiment only if its expected information changes a control decision. Implement an acquisition-value interface based on expected reduction in control regret/logical-risk uncertainty divided by interruption and safety cost. Candidate diagnostics may include Ramsey sign resolution, leakage-sensitive checks, readout reassignment, context/basis change or local antithetic probe.

This is an explicit compromise: diagnostic measurement interrupts or modifies nominal QEC. Log it as downtime/extra measurement and state why detector-only operation was insufficient. Do not hide it inside the calibration budget.

## Rollback and recovery

Snapshot every accepted complete policy. Rollback atomically and verify by observed telemetry. If performance does not return, the old policy is no longer valid; transition to `UNKNOWN_EVENT` or `BOOTSTRAP`. Mark stale replay/model data after rollback or regime change.

## Model lifecycle

Version response, dynamics, forecast, MPC and RL models with training interval, validity region and history. Support candidate, shadow, validated, promoted, quarantined and archived states. A recurring regime may warm-start state/control only after a short current-device validation.

## Minimum viable implementation

A deterministic version-controlled state machine implementing all modes, hard invariants, hysteretic transitions, rollback verification, diagnostic escalation, budgets and full logs. It must correctly handle a labelled fault suite and guarantee safe fallback to the reproduced RL/fixed-policy baseline where prediction is unsupported.

## Ideal full implementation

A constrained partially observable supervisory decision process optimizing regret and diagnostic cost subject to inviolable safety, with formally checked invariants, learned transition-value estimates, cross-region event reasoning, decoder/workload coordination and multi-device model lifecycle.

## Plausible extension that may fail

Jointly adapt calibration, decoder, code layout and application mapping. It may create coupled objectives, validation complexity and unsafe feedback. Keep these actions behind independent logical validation and explicit overhead accounting.

## Failure mechanisms and amendments

- mode chattering → probability thresholds, hysteresis, dwell and transition costs;
- thresholds too conservative → optimize empirical risk/performance frontier while preserving hard invariants;
- common-mode event missed → cross-region residual coherence, environment/controller telemetry and global motifs;
- bad model confidently authorized → independent predictive checks, shadow models, unknown-model mass and rollback;
- rollback assumed rather than verified → post-rollback validation and escalation;
- diagnostic chosen but decision unchanged → expected decision-value criterion and counterfactual action table;
- manual tuning contaminates result → log all human intervention and exclude/score separately;
- supervisor becomes single point of failure → deterministic core, watchdog, redundant policy hash checks and fail-safe fixed policy.

## Tests and deliverables

Build exhaustive state-transition tests, property tests for invariants, corrupted telemetry, stale forecast, solver failure, excessive exploration, broad events, rollback failure, policy-hash mismatch, ambiguous diagnostics and human veto. Provide a simulator fault-injection CLI, transition diagrams, event logs and a report showing safety, transition correctness, alarm delay, chattering, downtime and closed-loop regret.
