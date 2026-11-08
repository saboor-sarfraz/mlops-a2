param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$ProjectId,
    [ValidateNotNullOrEmpty()]
    [string]$Region = "us-central1"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    throw "Google Cloud CLI is not installed. Install it, reopen PowerShell, then run gcloud auth login."
}

$activeAccountCount = @(
    & gcloud auth list --filter="status:ACTIVE" --format="value(account)" 2>$null
).Count
if ($LASTEXITCODE -ne 0 -or $activeAccountCount -lt 1) {
    throw "No active gcloud account. Run: gcloud auth login"
}

& gcloud config set project $ProjectId
if ($LASTEXITCODE -ne 0) { throw "Could not select project $ProjectId." }

& gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com --project $ProjectId
if ($LASTEXITCODE -ne 0) { throw "Could not enable required Google Cloud APIs." }

$repositories = @(
    & gcloud artifacts repositories list --location $Region --project $ProjectId --filter="name:visionguard" --format="value(name)"
)
if ($LASTEXITCODE -ne 0) { throw "Could not list Artifact Registry repositories." }
if ($repositories.Count -eq 0) {
    & gcloud artifacts repositories create visionguard --repository-format=docker --location $Region --project $ProjectId --description "VisionGuard Cloud Run inference"
    if ($LASTEXITCODE -ne 0) { throw "Could not create the Artifact Registry repository." }
}

$image = "$Region-docker.pkg.dev/$ProjectId/visionguard/model-api:latest"
& gcloud builds submit --config cloudbuild-cloudrun.yaml --substitutions "_IMAGE=$image" --project $ProjectId --region $Region .
if ($LASTEXITCODE -ne 0) { throw "Cloud Build failed. Review the build logs before retrying." }

$secretName = "visionguard-model-api-key"
$secrets = @(
    & gcloud secrets list --project $ProjectId --filter="name:$secretName" --format="value(name)"
)
if ($LASTEXITCODE -ne 0) { throw "Could not list Secret Manager secrets." }
if ($secrets.Count -eq 0) {
    & gcloud secrets create $secretName --replication-policy=automatic --project $ProjectId
    if ($LASTEXITCODE -ne 0) { throw "Could not create the Secret Manager secret." }
}

$apiKey = (& "S:\MLOps\venv\python.exe" -c "import secrets; print(secrets.token_urlsafe(48))").Trim()
if ($LASTEXITCODE -ne 0 -or $apiKey.Length -lt 32) {
    throw "Could not generate an API key."
}
$secretFile = Join-Path ([System.IO.Path]::GetTempPath()) ([System.IO.Path]::GetRandomFileName())
try {
    [System.IO.File]::WriteAllText($secretFile, $apiKey, [System.Text.Encoding]::ASCII)
    & gcloud secrets versions add $secretName --data-file $secretFile --project $ProjectId
    if ($LASTEXITCODE -ne 0) { throw "Could not add the API key version to Secret Manager." }
}
finally {
    if (Test-Path -LiteralPath $secretFile) {
        Remove-Item -LiteralPath $secretFile -Force
    }
}

$serviceAccount = "visionguard-cloudrun@$ProjectId.iam.gserviceaccount.com"
$serviceAccounts = @(
    & gcloud iam service-accounts list --project $ProjectId --filter="email:$serviceAccount" --format="value(email)"
)
if ($LASTEXITCODE -ne 0) { throw "Could not list IAM service accounts." }
if ($serviceAccounts.Count -eq 0) {
    & gcloud iam service-accounts create visionguard-cloudrun --display-name "VisionGuard Cloud Run" --project $ProjectId
    if ($LASTEXITCODE -ne 0) { throw "Could not create the Cloud Run service account." }
}

& gcloud secrets add-iam-policy-binding $secretName --member "serviceAccount:$serviceAccount" --role roles/secretmanager.secretAccessor --project $ProjectId
if ($LASTEXITCODE -ne 0) { throw "Could not grant the service account access to the API key secret." }

& gcloud run deploy visionguard-model-api `
    --image $image `
    --region $Region `
    --project $ProjectId `
    --platform managed `
    --allow-unauthenticated `
    --port 8080 `
    --memory 4Gi `
    --cpu 2 `
    --concurrency 1 `
    --min 0 `
    --max 2 `
    --timeout 180 `
    --service-account $serviceAccount `
    --set-secrets "MODEL_API_KEY=$secretName:latest" `
    --set-env-vars "MODEL_NAME=VisionGuard MobileNetV2 champion,THRESHOLD=0.8569827079772949,MAX_UPLOAD_BYTES=8388608"
if ($LASTEXITCODE -ne 0) { throw "Cloud Run deployment failed." }

$serviceUrl = (& gcloud run services describe visionguard-model-api --region $Region --project $ProjectId --format "value(status.url)").Trim()
if ($LASTEXITCODE -ne 0 -or -not $serviceUrl.StartsWith("https://")) {
    throw "Could not retrieve the deployed Cloud Run URL."
}

$envPath = Join-Path $root ".env"
if (-not (Test-Path -LiteralPath $envPath)) {
    [System.IO.File]::WriteAllText($envPath, "")
}
$envLines = @(
    Get-Content -LiteralPath $envPath |
        Where-Object { $_ -notmatch '^\s*MODEL_API_(URL|KEY)\s*=' }
)
$envLines += "MODEL_API_URL=$serviceUrl"
$envLines += "MODEL_API_KEY=$apiKey"
[System.IO.File]::WriteAllLines(
    $envPath,
    $envLines,
    (New-Object System.Text.UTF8Encoding($false))
)

Write-Output "Cloud Run service deployed: $serviceUrl"
Write-Output "Prediction endpoint: $serviceUrl/api/v1/predict"
Write-Output "API key saved in Secret Manager and the ignored local .env; its value is not displayed."
Write-Output "Run .\run_flask.ps1 to make the local Flask UI forward inference to Cloud Run."
