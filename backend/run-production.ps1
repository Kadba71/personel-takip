# Runs Personel Takip API in production mode on Windows
# Usage: Right-click -> Run with PowerShell (or run in admin PowerShell)

$ErrorActionPreference = 'Stop'

# Ensure venv
$venv = Join-Path $PSScriptRoot '.venv'
if (!(Test-Path $venv)) {
  python -m venv $venv
}

# Activate venv
. "$venv\Scripts\Activate.ps1"

# Install deps
pip install --upgrade pip
pip install -r "$PSScriptRoot\requirements.txt"

# Copy production env if missing
$envFile = Join-Path $PSScriptRoot '.env'
if (!(Test-Path $envFile)) {
  if (Test-Path "$PSScriptRoot\.env.production.example") {
    Copy-Item "$PSScriptRoot\.env.production.example" $envFile
    Write-Host "Created .env from .env.production.example. Please edit secure values."
  } else {
    Write-Warning ".env not found. Please create it from .env.example or .env.production.example"
  }
}

# Start server
$serverHost = $env:PERSONEL_TAKIP_SERVER__HOST
if (-not $serverHost) { $serverHost = '0.0.0.0' }
$serverPort = $env:PERSONEL_TAKIP_SERVER__PORT
if (-not $serverPort) { $serverPort = '8000' }

Write-Host ("Starting API on {0}:{1} in production mode..." -f $serverHost, $serverPort)
uvicorn main:app --host $serverHost --port $serverPort --workers 2 --log-level info
