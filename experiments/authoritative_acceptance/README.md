# Authoritative compute-aware acceptance

The frozen v2 protocol and launch are `compute-aware-confirmation-v2.md` and
`compute-aware-confirmation-v2.json`.

Seeds 3001--3016 and all `confirmatory_*` scenarios are one-shot held-out evidence. Do
not run an arm until the code is frozen and a fresh exact-config preflight passes. First
run manifest-only validation; it acquires no disturbance tape:

```powershell
Set-Location D:\Users\Raife\hdfa_rl_suite
$env:PYTHONPATH = "src"
$python = "C:\Users\Raife\AppData\Local\Programs\Python\Python312\python.exe"

& $python -m hdfa_rl_suite.validation.preflight_cli `
  --benchmark-config experiments\authoritative_acceptance\compute-aware-confirmation-v2.json `
  --output artifacts\validation\compute-aware-v2

& $python -m hdfa_rl_suite.evaluation.cli `
  --config experiments\authoritative_acceptance\compute-aware-confirmation-v2.json `
  --preflight-manifest artifacts\validation\compute-aware-v2\benchmark-preflight-manifest.json `
  --validate-only
```

Only after both return exit code 0, execute the one-shot acquisition by removing
`--validate-only` and adding:

```powershell
  --output artifacts\acceptance\compute-aware-v2\authoritative-comparison-v2.json
```

Exit code 0 is acceptance, 2 is a valid scientific rejection, and 3 is invalid or
non-evaluable acquisition. Never overwrite the immutable v1 report.
