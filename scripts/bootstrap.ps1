param([Parameter(Mandatory=$true)][string]$PythonPath)
$ErrorActionPreference='Stop'
$projectDir=Split-Path $PSScriptRoot -Parent
# Local caches, the virtual environment and model files live in <repo>/work unless TABCREW_WORK_DIR is set.
$workDir=if($env:TABCREW_WORK_DIR){$env:TABCREW_WORK_DIR}else{Join-Path $projectDir 'work'}
$env:TABCREW_WORK_DIR=$workDir
New-Item -ItemType Directory -Force -Path (Join-Path $workDir 'tmp') | Out-Null
$env:UV_CACHE_DIR=Join-Path $workDir 'uv-cache'
$env:UV_PROJECT_ENVIRONMENT=Join-Path $workDir 'venv-rl'
$env:UV_PYTHON_DOWNLOADS='never'
$env:TMP=Join-Path $workDir 'tmp';$env:TEMP=$env:TMP;$env:PYTHONPATH=$projectDir
Set-Location -LiteralPath $projectDir
uv run --no-project --python $PythonPath python scripts/audit_dependencies.py
if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}
uv sync --frozen --python $PythonPath
if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}
uv run --frozen python scripts/check_stack.py
if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}
uv run --frozen python scripts/download_model.py
exit $LASTEXITCODE
