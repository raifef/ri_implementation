param(
    [string]$Python = "python",
    [string]$OutputRoot = "artifacts/comparison/nature-2026-v5",
    [int]$PipelineWorkers = 8,
    [switch]$NoResume,
    [switch]$SkipEffectiveness
)

$ErrorActionPreference = "Stop"
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$resolvedOutput = Join-Path $repositoryRoot $OutputRoot
$previousPythonPath = $env:PYTHONPATH
$startedAt = (Get-Date).ToUniversalTime().ToString("o")

if ($PipelineWorkers -lt 1) {
    throw "PipelineWorkers must be at least one"
}

Push-Location $repositoryRoot
try {
    $env:PYTHONPATH = Join-Path $repositoryRoot "src"
    $effectivenessPath = Join-Path $resolvedOutput "authoritative-effectiveness.json"

    if ($SkipEffectiveness) {
        if (-not (Test-Path -LiteralPath $effectivenessPath)) {
            throw "Cannot resume: effectiveness report is missing at $effectivenessPath"
        }
        & $Python -m hdfa_rl_suite.evaluation.report_status_cli `
            $effectivenessPath --require-current-runtime
        $effectivenessExitCode = $LASTEXITCODE
    }
    else {
        & $Python -m hdfa_rl_suite.evaluation.cli `
            --primary-only `
            --qubits 5 `
            --intervals 32 `
            --cycles 512 `
            --candidate-cycles 32 `
            --logical-shots 4096 `
            --bootstrap-shots 384 `
            --bootstrap-cycles 512 `
            --baseline-cycles 512 `
            --bootstrap-target-stddev 0.035 `
            --bootstrap-edr-limit 0.10 `
            --bootstrap-block-familywise-alpha 0.0001 `
            --seed 101 --seed 102 --seed 103 --seed 104 --seed 105 `
            --output $effectivenessPath
        $effectivenessExitCode = $LASTEXITCODE
    }

    # Exit 2 above is a valid rejected hypothesis.  It must not suppress collection of
    # the independently defined scalability and computational-cost evidence.
    $scalabilityOutput = Join-Path $resolvedOutput "scalability-and-cost"
    $checkpointDirectory = Join-Path $scalabilityOutput "checkpoints"
    $scalabilityArguments = @(
        "-m", "hdfa_rl_suite.evaluation.scalability_cli",
        "--config", "experiments/nature_2026_scalability/full-profile.json",
        "--pipeline-workers", $PipelineWorkers,
        "--checkpoint-directory", $checkpointDirectory,
        "--output", $scalabilityOutput
    )
    if (-not $NoResume) {
        $scalabilityArguments += "--resume"
    }
    & $Python @scalabilityArguments
    $scalabilityExitCode = $LASTEXITCODE

    New-Item -ItemType Directory -Force -Path $resolvedOutput | Out-Null
    $status = [ordered]@{
        protocol = "nature-2026-two-part-comparison.v5"
        started_at_utc = $startedAt
        completed_at_utc = (Get-Date).ToUniversalTime().ToString("o")
        effectiveness_exit_code = $effectivenessExitCode
        scalability_exit_code = $scalabilityExitCode
        pipeline_workers = $PipelineWorkers
        checkpoint_directory = $checkpointDirectory
        resumed_checkpoints = (-not $NoResume)
        exit_code_meaning = [ordered]@{
            zero = "all applicable gates passed"
            two = "scientifically evaluable rejection"
            three = "invalid or non-evaluable effectiveness experiment"
        }
    }
    $status | ConvertTo-Json -Depth 4 | Set-Content `
        -LiteralPath (Join-Path $resolvedOutput "comparison-status.json") -Encoding UTF8

    if ($effectivenessExitCode -eq 3 -or $scalabilityExitCode -notin @(0, 2)) {
        exit 3
    }
    if ($effectivenessExitCode -eq 2 -or $scalabilityExitCode -eq 2) {
        exit 2
    }
    exit 0
}
finally {
    $env:PYTHONPATH = $previousPythonPath
    Pop-Location
}
