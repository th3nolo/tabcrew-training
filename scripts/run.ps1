param(
  [Parameter(Mandatory=$true)][ValidateSet('test')][string]$Action
)
$ErrorActionPreference='Stop'
$projectDir=Split-Path $PSScriptRoot -Parent
$workDir=if($env:TABCREW_WORK_DIR){$env:TABCREW_WORK_DIR}else{Join-Path $projectDir 'work'}
$env:TABCREW_WORK_DIR=$workDir
$env:UV_CACHE_DIR=Join-Path $workDir 'uv-cache'
$env:UV_PROJECT_ENVIRONMENT=Join-Path $workDir 'venv-rl'
$env:PYTHONPATH=$projectDir
New-Item -ItemType Directory -Force -Path (Join-Path $workDir 'tmp') | Out-Null
$env:TMP=Join-Path $workDir 'tmp'
$env:TEMP=$env:TMP
Set-Location -LiteralPath $projectDir
# Training and evaluation use reproduce.ps1 with a fresh named run directory.
uv run --frozen python -m unittest discover -s tests -v
exit $LASTEXITCODE
