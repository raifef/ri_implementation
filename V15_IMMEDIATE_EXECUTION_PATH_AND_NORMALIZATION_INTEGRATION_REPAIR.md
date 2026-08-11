# Codex Prompt — V15 Immediate Execution-Path Verification and Source-Normalization Integration Repair

Work in:

`D:/Users/Raife/hdfa_rl_suite`

Act as a senior scientific-software auditor and quantum-control/QEC replication engineer.

## Mission

The latest validation plots are effectively unchanged from the old failed results despite extensive V13/V15 amendments.

The immediate problem is **not another broad RL-theory audit**.

The immediate question is:

> **Did the final V13/V15 source-literal sensitivity-normalization implementation actually reach the paper-reproduction experiment drivers, and if it did, why does it fail to reproduce the dramatic V12 conditioning repair?**

The latest one-hour validation completed in ~33 seconds while merging many shards. Step and recovery again fail catastrophically, Figure 5b remains a total mismatch, and Figure 5c remains unidentifiable.

This task must determine whether the failure is:

```text
A. execution-path/provenance bug:
   paper-reproduction runners are not actually using the V15 source-literal map

B. source-normalization/plant incompatibility:
   V15 source-literal map is used correctly but does not condition the current surrogate correctly

C. protocol-specific integration bug:
   V15 map works in isolation but is lost, altered, or bypassed inside step/recovery/Figure5b drivers
```

Do not create another architectural version unless necessary.

Do not perform broad hyperparameter tuning.

Do not modify Figure 5c.

Do not launch long runs.

Do not weaken scientific gates.

The required sequence is:

```text
prove execution path
-> reproduce V12/V15 contrast on one tiny matched fixture
-> localize failure
-> minimally repair integration/mapping
-> rerun tiny matched fixtures
-> only then rerun reduced scientific validation
```

---

## 1. Freeze the exact code/artifact versions under test

Create:

```text
artifacts/google_pure_v15/immediate_execution_audit/
```

Record and SHA-256 hash:

```text
controller code
controller config
V15 source-definition audit
V15 fitted sensitivity map
V15 calibration bundle
V15 detector-degree audit
V15 boundary-map implementation
step driver
recovery driver
Figure 5a driver
Figure 5b driver
paper-reproduction workflow
one-hour validation workflow
plant
graph
```

Create:

```text
execution_inputs.json
execution_inputs.md
```

The audit must distinguish:

```text
algorithm package version
paper-family driver version
analysis/plotting version
source-normalization version
plant version
graph version
```

Do not accept a generic integration-manifest hash as sufficient evidence.

---

## 2. Add mandatory V15 provenance to every acquisition

Every new candidate-level shard and every experiment-family manifest must record:

```text
implementation_version = google_pure_v15
controller_hash
controller_code_hash
controller_mode
parameterization
plant_hash
graph_hash

sensitivity_map_hash
sensitivity_definition_hash
calibration_bundle_hash
detector_degree_audit_hash
boundary_transform_hash
boundary_transform_name = "u = u0 + s*x"
boundary_apply_count
control_order_hash

experiment_driver_hash
protocol_hash
source_budget_profile
fresh_acquisition = true/false
reused_shard_ids
```

Abort acquisition if any required V15 field is missing.

In final scientific mode require:

```text
boundary_apply_count == 1
fresh_acquisition == true
```

No plot may be labelled as a V15 validation result unless these fields are present and verified.

---

## 3. Prove the boundary transform numerically for one candidate

For one step-response candidate, one recovery candidate, and one Figure 5b candidate, print/store for the materially affected coordinates:

\[
x_i,\quad
s_i,\quad
u_{0,i},\quad
s_i x_i,\quad
u_i.
\]

Verify exactly:

\[
u_i-u_{0,i}=s_i x_i.
\]

Also verify:

```text
same control index ordering
same s_i used by mean and sampled candidates
same s_i used during evaluation
no second scaling inside plant
no inverse scaling
no stale V12 map
no unscaled legacy path
```

Create:

```text
boundary_trace_step.json
boundary_trace_recovery.json
boundary_trace_figure5b.json
```

Any failure is an immediate hard bug.

---

## 4. Run the decisive A/B/C matched step experiment

Use exactly one small deterministic/reduced step fixture.

Hold fixed:

```text
seed
initial policy
candidate standard-normal draws
detector/QEC noise tape where possible
plant
graph
reward definition
baseline initialization
learning rates
entropy
candidate count
cycles/candidate
epoch count
```

Run exactly three branches:

### A — broken legacy/no-map branch

\[
u=u_0+x.
\]

### B — successful V12 compensatory map

Use the exact V12 boundary map that previously produced ~98% final step progress.

### C — final V15 source-literal calibrated map

\[
u=u_0+s_{V15}x.
\]

For every epoch record:

```text
native curvature in stepped direction
normalized curvature
directional gradient mean
directional gradient SE
actual mean update
sigma
target-relative progress
EDR
```

Report:

```text
final progress
t50
t63.2
t90 if reached
fitted tau if identifiable
```

Create:

```text
abc_step_comparison.json
abc_step_comparison.md
abc_step_comparison.png
```

### Required interpretation

If:

```text
A fails
B succeeds
C succeeds
```

conclude:

```text
CURRENT PAPER-REPRODUCTION RUNNER IS NOT ACTUALLY USING V15 SOURCE MAP
```

If:

```text
A fails
B succeeds
C fails
```

conclude:

```text
V15 SOURCE-LITERAL NORMALIZATION IS INCOMPATIBLE WITH CURRENT STEP PLANT
```

Then locate whether:

```text
plant native curvature is inconsistent with calibration
EDR calibration and training objective use different aggregation
source normalization has been interpreted incorrectly
step plant was defined in units incompatible with source calibration
```

Do not revert to V12 merely because it performs better.

If:

```text
A fails
B succeeds
C succeeds in isolated fixture
full step runner fails
```

conclude:

```text
STEP DRIVER INTEGRATION BUG
```

---

## 5. Compare V12 and V15 scales directly

For every coordinate touched by the reduced step fixture report:

```text
s_V12
s_V15
ratio s_V15/s_V12

native curvature h_u
normalized curvature under V12 = s_V12^2 h_u
normalized curvature under V15 = s_V15^2 h_u

detector degree
EDR calibration coefficient
training-objective curvature
```

Do the same for a representative recovery direction.

Create:

```text
v12_v15_scale_comparison.json
v12_v15_scale_comparison.md
```

The implementation must verify the **actual resulting normalized EDR curvature**, not merely trust a `SOURCE_LITERAL` label.

---

## 6. Verify empirical calibration and experiment objective are the same quantity

For the same coordinate and physical perturbations calculate side by side:

```text
EDR used during calibration
EDR used as controller reward
connected-detector sum
connected-detector mean
global detector sum
global detector mean
detector degree
```

Numerically prove which quantity satisfies:

\[
\mathrm{EDR}
=
\mathrm{EDR}_0
+
(\sigma/\sigma_0)^2.
\]

Then prove that the controller uses the same objective convention.

If calibration uses one detector aggregation and training another, classify:

```text
CALIBRATION_TRAINING_OBJECTIVE_MISMATCH
```

and repair that mismatch rather than retuning scales.

---

## 7. Run the same A/B/C diagnostic on one tiny Figure 5b cell

Use only:

```text
d = 3
P = 1
25–50 preregistered epochs
reduced candidate/cycle budget sufficient for diagnosis
```

Run:

```text
A = legacy/no map
B = V12 compensatory map
C = V15 source-literal map
```

Record per epoch:

```text
mean EDR
mean physical error
mean LER
Lambda/Lambda*
gradient norm
mean update norm
sigma
exploration penalty
distance to local regime
```

Create:

```text
abc_figure5b_d3_p1.json
abc_figure5b_d3_p1.md
abc_figure5b_d3_p1.png
```

Do not launch the full scaling campaign unless C demonstrates genuine progress and its relation to B is physically/source justified.

---

## 8. Audit the paper-reproduction workflow for shard reuse

The latest “one-hour” validation completed in ~33 seconds while merging many shards.

Inspect the workflow and report:

```text
number of newly acquired shards
number of reused shards
number of re-analysed shards
creation timestamps
source artifact paths
controller hash per shard
sensitivity-map hash per shard
```

Create:

```text
shard_freshness_audit.json
shard_freshness_audit.md
```

A run must not be labelled “post-V15” if it is plotting pre-V15 shards.

If mixed-version shards are present, fail:

```text
MIXED_VERSION_EVIDENCE
```

Do not merge shards across differing:

```text
controller hashes
plant hashes
graph hashes
sensitivity-map hashes
boundary implementations
protocol hashes
```

---

## 9. Fix one-hour validation semantics

Implement explicit modes:

```text
ANALYSIS_ONLY
SMOKE_ACQUISITION
ONE_HOUR_FRESH_ACQUISITION
REFERENCE_ACQUISITION
```

For `ONE_HOUR_FRESH_ACQUISITION`:

- fresh acquisition is mandatory;
- reused scientific shards are forbidden;
- work should target the requested wall-time budget;
- early stop is allowed only for a documented scientific criterion;
- acquisition and analysis times must be separate.

Persist:

```text
acquisition_seconds
analysis_seconds
new_QEC_cycles
new_candidates
new_shards
reused_shards
```

A 33-second merge must never masquerade as a one-hour fresh scientific run.

---

## 10. Repair experiment-family driver integration

If branch C succeeds in isolated fixtures but not full paper-family drivers, trace:

```text
CLI
-> protocol loading
-> controller creation
-> calibration loading
-> boundary adapter
-> plant
-> acquisition
-> checkpoint/resume
-> evaluation
-> shard serialization
-> merge
-> analysis
```

At each stage log:

```text
controller hash
sensitivity-map hash
boundary-adapter hash
plant hash
```

The same values must persist end to end.

Add integration tests for:

```text
step
recovery
Figure 5a
Figure 5b
```

that intentionally inject a wrong map hash and require hard failure.

---

## 11. Do not touch Figure 5c yet

Figure 5c remains downstream of Figure 5b.

Do not modify:

```text
fit window
derivative estimator
R^2 gate
local-regime definition
```

until a genuine V15 Figure 5b run enters the preregistered local region.

---

## 12. Natural drift: classify current plot correctly

Do not rerun the full 48-run plan in this task.

Only verify that the current six-shard plot is labelled:

```text
UNDERPOWERED_DEVELOPMENT_VALIDATION
```

and cannot be interpreted as V15 natural-drift evidence.

Keep the 48-run power plan frozen.

---

## 13. Freeze the repaired execution path

Once the immediate fault is localized and repaired, freeze:

```text
controller hash
controller code hash
V15 sensitivity-map hash
calibration-bundle hash
boundary-adapter hash
plant hash
graph hash
step protocol hash
recovery protocol hash
Figure 5a protocol hash
Figure 5b protocol hash
```

Create:

```text
frozen_execution_contract.json
frozen_execution_contract.md
```

No later validation may silently substitute a different map or driver.

---

## 14. Reduced post-fix scientific validation

Only after isolated branch-C tests pass, run **fresh acquisition** for:

```text
step response
recovery
one slow Figure 5a cell
one fast Figure 5a cell
one Figure 5b d=3/P=1 cell
```

Do not run Figure 5c or full natural drift yet.

Required behavior:

### Step
Final V15 map must show substantial target-relative response.

### Recovery
Final V15 map must materially recover after spoil.

### Figure 5a
Slow cell must outperform fast cell in the expected direction.

### Figure 5b
Learned mean must make measurable floor-normalized progress beyond the legacy/no-map branch.

---

## 15. Permanent tests

Add tests that fail when:

- a paper-family driver omits the V15 map;
- boundary scaling is applied zero or two times;
- calibration and training objective aggregations differ;
- a shard lacks sensitivity-map provenance;
- shards with different map hashes are merged;
- a fresh-acquisition profile reuses scientific shards;
- a V15 result is generated from a non-V15 driver;
- isolated V15 fixture and full driver receive different calibration/config state.

---

## 16. Required CLIs

Register:

```bash
hdfa-google-v15-audit-execution-path
hdfa-google-v15-trace-boundary-step
hdfa-google-v15-trace-boundary-recovery
hdfa-google-v15-trace-boundary-figure5b
hdfa-google-v15-run-abc-step
hdfa-google-v15-compare-v12-v15-scales
hdfa-google-v15-audit-calibration-objective
hdfa-google-v15-run-abc-figure5b
hdfa-google-v15-audit-shard-freshness
hdfa-google-v15-verify-driver-integration
hdfa-google-v15-freeze-execution-contract
hdfa-google-v15-run-reduced-postfix-validation
```

Do not automatically launch long runs.

---

## 17. Completion criteria

This task is complete only when the repository answers unambiguously:

1. **Did the latest failed plots actually use the final V15 map?**
   Proven by hashes, not code presence.

2. **What happens under perfectly matched A/B/C conditions for step and minimal Figure 5b?**

3. **If V15 fails while V12 succeeds, what exact mathematical/physical mismatch causes it?**
   Identify the divergence in calibration objective, scale magnitude, native curvature, normalized curvature, plant units, or detector aggregation.

4. **If isolated V15 succeeds but the full driver fails, where exactly is the map/state lost or altered?**

5. **Are post-fix validation plots generated entirely from fresh, same-version V15 shards?**

---

## 18. Final report

Produce a concise report containing:

1. whether the latest failed plots used V15 or stale/mixed evidence;
2. exact V12-versus-V15 scale comparison;
3. A/B/C step outcomes;
4. A/B/C Figure 5b outcomes;
5. normalized curvature under each map;
6. calibration-objective consistency result;
7. execution-path bug, if found;
8. shard-freshness findings;
9. corrected one-hour workflow semantics;
10. reduced post-fix validation results;
11. exact remaining blocker before a genuine source-budget Figure 5a/Figure 5b campaign.

Do not begin another broad fault hunt until this execution-path question is definitively resolved.
