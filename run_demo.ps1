$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (-not $env:MODEL_PATH) {
    $env:MODEL_PATH = Join-Path $root "visionguard_outputs\visionguard_comparison_champion.keras"
}
if (-not $env:THRESHOLD) {
    $env:THRESHOLD = "0.8569827079772949"
}

& "S:\MLOps\venv\python.exe" "demo_inference.py"
if ($LASTEXITCODE -ne 0) {
    throw "VisionGuard demo inference exited with status $LASTEXITCODE"
}
