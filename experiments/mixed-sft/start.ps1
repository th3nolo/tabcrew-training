# Starts or resumes the mixed-teacher SFT pipeline (training, then three matched validations).
# Uses the repository's existing uv environment. It installs nothing and makes no network calls.
# Optional: TABCREW_WORK_DIR (venv, uv cache, model snapshot; default <repo>/work),
#           TABCREW_RUN_DIR (outputs; default this folder), TABCREW_EARLY705_RUN_DIR (baseline adapters).
$ErrorActionPreference='Stop'
$experiment=$PSScriptRoot
$repo=Split-Path (Split-Path $experiment -Parent) -Parent
$workDir=if($env:TABCREW_WORK_DIR){$env:TABCREW_WORK_DIR}else{Join-Path $repo 'work'}
$runDir=if($env:TABCREW_RUN_DIR){$env:TABCREW_RUN_DIR}else{$experiment}
if(Test-Path -LiteralPath (Join-Path $runDir 'PAUSE')){throw 'Training PAUSE marker exists; inspect before resuming'}
$active=@(Get-CimInstance Win32_Process|Where-Object {
    ($_.Name -match '^(python|pythonw|uv).*') -and
    ($_.CommandLine -match 'mixed-sft[\/]run.py|tabcrew_rl.evaluate|early-705[\/]run.py')
})
if($active.Count){throw 'A training/evaluation process is already running'}
$os=Get-CimInstance Win32_OperatingSystem
if($os.FreePhysicalMemory -lt 4*1MB){throw 'Less than 4GiB physical memory available'}
$free=& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
if($LASTEXITCODE -ne 0){throw 'GPU memory query failed'}
$freeMiB=[int](@($free)[0])
if($freeMiB -lt 6500){throw 'Insufficient free GPU memory for the verified batch4 plan'}
$env:TABCREW_WORK_DIR=$workDir
$env:UV_PROJECT_ENVIRONMENT=Join-Path $workDir 'venv-rl'
$env:UV_CACHE_DIR=Join-Path $workDir 'uv-cache'
& uv run --directory $repo --offline --frozen --no-sync python (Join-Path $experiment 'run.py') pipeline
if($LASTEXITCODE -ne 0){throw 'Experiment process exited unsuccessfully; inspect saved logs'}
