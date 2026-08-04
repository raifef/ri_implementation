# Google detector-driven RL reimplementation

This is a standalone reproduction and scientific-validation workflow for the detector-driven reinforcement-learning controller reported by Google Quantum AI and Google DeepMind. It contains only the reference-controller lineage: public-data analysis, empirical and analytical surrogates, the masked Gaussian policy, finite-shot PPO diagnostics, drift and recovery studies, Figure-5 pipelines, and fail-closed evidence contracts.

The project has its own Python package, command namespace, configurations, tests, and artifact directory. It does not import the source package it was extracted from. Historical generated results are not bundled; every result is regenerated here with local hashes and paths.

## Install

```powershell
Set-Location D:\Users\Raife\google_rl_reimplementation
& C:\Users\Raife\AppData\Local\Programs\Python\Python312\python.exe -m pip install -e .
```

If the Python scripts directory is not already on `PATH`:

```powershell
$env:Path = "$env:LOCALAPPDATA\Programs\Python\Python312\Scripts;$env:Path"
```

## Current reference workflow

Materialize the small retained provenance inputs and regenerate all compact smoke prerequisites:

```powershell
google-rl-bootstrap
```

This command does not launch reference-scale acquisition or consume certification seeds. To inspect or rerun individual root-cause diagnostics:

```powershell
google-rl-v8-snapshot
google-rl-v8-audit-mathematical-contracts
google-rl-v8-audit-figure5a-edr
google-rl-v8-run-figure5a-feasibility
google-rl-v8-audit-exploration-floor
google-rl-v8-audit-entropy-scale
google-rl-v8-audit-native-units
google-rl-v8-audit-clipping-likelihood
google-rl-v8-audit-ppo-lifecycle
google-rl-v8-audit-baselines
google-rl-v8-audit-temporal-protocol
google-rl-v8-run-compact-fault-matrix
google-rl-v8-report-root-cause
google-rl-v8-status
```

Then build and inspect the evidence protocol:

```powershell
google-rl-evidence-v8-build-contracts
google-rl-evidence-v8-validate-manifests
google-rl-evidence-v8-status
```

Reference-scale acquisition is intentionally fail-closed. It requires explicit execution and passing scientific preflight gates:

```powershell
google-rl-evidence-v8-run-natural-drift --mode reference --execute
google-rl-evidence-v8-run-step-response --mode reference --execute
google-rl-evidence-v8-run-recovery --mode reference --execute
google-rl-evidence-v8-run-figure5b --mode reference --execute
google-rl-evidence-v8-run-figure5c --mode reference --execute
google-rl-evidence-v8-build-claim-registry
google-rl-evidence-v8-build-paper-comparison
google-rl-evidence-v8-status
```

Outputs are written under `artifacts/` inside this project. Earlier `v2` through `v7` and paper-panel commands remain installed for reproducibility and historical protocol replay.

## Test

```powershell
& C:\Users\Raife\AppData\Local\Programs\Python\Python312\python.exe -m pytest -q
```

The test suite includes a repository-boundary audit that rejects dependencies on the source project and rejects prohibited legacy terminology in filenames and file contents.
