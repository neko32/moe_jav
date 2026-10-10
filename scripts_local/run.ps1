# Local helper for moe_jav (Windows PowerShell).
#
#   .\scripts_local\run.ps1 check          fmt + clippy + tests + coverage report
#   .\scripts_local\run.ps1 live           tests against LM Studio + Jev (needs API_KEY_JEV)
#   .\scripts_local\run.ps1 apply FILE     underwrite one application
#   .\scripts_local\run.ps1 eval [MODE]    routing | e2e | all  (default: routing)
param(
    [Parameter(Position = 0)][string]$Command = "check",
    [Parameter(Position = 1, ValueFromRemainingArguments = $true)][string[]]$Rest = @()
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$App = Join-Path $Root "app"

function Invoke-Step([string]$Exe, [string[]]$Arguments) {
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Exe $($Arguments -join ' ') failed ($LASTEXITCODE)" }
}

Push-Location $App
try {
    switch ($Command) {
        "check" {
            Invoke-Step cargo @("fmt", "--check")
            Invoke-Step cargo @("clippy", "--all-targets", "--", "-W", "clippy::pedantic", "-D", "warnings")
            Invoke-Step cargo @("llvm-cov", "--fail-under-lines", "90", "--ignore-filename-regex", "main\.rs",
                "--html", "--output-dir", (Join-Path $Root "docs\coverage"))
            Write-Host "Coverage report: $(Join-Path $Root 'docs\coverage\html\index.html')"
        }
        "live" {
            Invoke-Step cargo @("test", "--test", "live_services", "--", "--ignored", "--test-threads=1")
        }
        "apply" {
            if ($Rest.Count -lt 1) { throw "usage: run.ps1 apply FILE [extra args]" }
            $file = (Resolve-Path (Join-Path $Root $Rest[0]) -ErrorAction SilentlyContinue)
            if (-not $file) { $file = Resolve-Path $Rest[0] }
            $extra = if ($Rest.Count -gt 1) { $Rest[1..($Rest.Count - 1)] } else { @() }
            Invoke-Step cargo (@("run", "--release", "--", "apply", "--file", $file.Path) + $extra)
        }
        "eval" {
            $mode = if ($Rest.Count -ge 1) { $Rest[0] } else { "routing" }
            $extra = if ($Rest.Count -gt 1) { $Rest[1..($Rest.Count - 1)] } else { @() }
            Invoke-Step cargo (@("run", "--release", "--", "eval", "--mode", $mode) + $extra)
        }
        default { throw "unknown command: $Command" }
    }
}
finally {
    Pop-Location
}
