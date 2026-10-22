param([switch]$StartApproved,[ValidateRange(0,50)][int]$MaxInFlight=0)
# Starts the IFM collector (continuous.py) with the repository's uv environment.
# The API key must already be in the IFM_API_KEY environment variable of this PowerShell session.
# It is removed from the session after the run. It is never written to arguments, logs or reports.
# Optional: TABCREW_WORK_DIR (venv and uv cache; default <repo>/work).
$ErrorActionPreference='Stop'
if(-not $StartApproved){throw 'Continuous generation must be authorized.'}
if(Test-Path -LiteralPath (Join-Path $PSScriptRoot 'PAUSE')){throw 'PAUSE exists; inspect the saved state.'}
if(-not $env:IFM_API_KEY){throw 'Set IFM_API_KEY in this session before starting.'}
$repo=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$workDir=if($env:TABCREW_WORK_DIR){$env:TABCREW_WORK_DIR}else{Join-Path $repo 'work'}
$activeIfm=@(Get-CimInstance Win32_Process | Where-Object {
    ($_.Name -match '^(python|pythonw|uv)\.exe$') -and
    ($_.CommandLine -match 'ifm[\\/](collector|continuous|controls)\.py(?:["\s]|$)')
})
if($activeIfm.Count){throw 'An IFM worker is already running.'}
$launch=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'launch-config.json') -Raw | ConvertFrom-Json
if($MaxInFlight -eq 0){$MaxInFlight=[int]$launch.max_in_flight}
$oldProject=$env:UV_PROJECT_ENVIRONMENT
$oldCache=$env:UV_CACHE_DIR
try {
    $env:UV_PROJECT_ENVIRONMENT=Join-Path $workDir 'venv-rl'
    $env:UV_CACHE_DIR=Join-Path $workDir 'uv-cache'
    & uv run --directory $repo --offline --frozen --no-sync python (Join-Path $PSScriptRoot 'continuous.py') --token-budget $launch.token_budget --max-hours $launch.max_hours --target $launch.target --max-in-flight $MaxInFlight
    if($LASTEXITCODE -ne 0){throw 'Collector exited unsuccessfully; inspect saved status.'}
} finally {
    $env:IFM_API_KEY=$null
    $env:UV_PROJECT_ENVIRONMENT=$oldProject
    $env:UV_CACHE_DIR=$oldCache
}
