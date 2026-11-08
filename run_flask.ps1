$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (-not $env:MODEL_PATH) {
    $env:MODEL_PATH = Join-Path $root "visionguard_outputs\visionguard_comparison_champion.keras"
}
if (-not $env:THRESHOLD) {
    $env:THRESHOLD = "0.8569827079772949"
}
if (-not $env:MODEL_NAME) {
    $env:MODEL_NAME = "VisionGuard MobileNetV2 champion"
}
if (-not $env:DATABASE_URL) {
    $databasePath = (Join-Path $root "visionguard_outputs\prediction_history.db").Replace("\", "/")
    $env:DATABASE_URL = "sqlite:///$databasePath"
}
if (-not $env:MAX_UPLOAD_BYTES) {
    $env:MAX_UPLOAD_BYTES = "8388608"
}
if (-not $env:HOST) {
    $env:HOST = "127.0.0.1"
}
if (-not $env:PORT) {
    $env:PORT = "5000"
}

& "S:\MLOps\venv\python.exe" "webapp.py"
if ($LASTEXITCODE -ne 0) {
    throw "VisionGuard web application exited with status $LASTEXITCODE"
}
