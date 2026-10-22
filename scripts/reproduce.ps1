param([Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9_-]+$')][string]$Name)
# Reproduces the first bounded pilot on the generated curriculum (data/v1):
# baseline evaluation, GRPO, SFT, GRPO after SFT, fixed final-test suite, paired comparison.
$ErrorActionPreference='Stop'
$projectDir=Split-Path $PSScriptRoot -Parent
$workDir=if($env:TABCREW_WORK_DIR){$env:TABCREW_WORK_DIR}else{Join-Path $projectDir 'work'}
$env:TABCREW_WORK_DIR=$workDir
$env:UV_CACHE_DIR=Join-Path $workDir 'uv-cache';$env:UV_PROJECT_ENVIRONMENT=Join-Path $workDir 'venv-rl'
New-Item -ItemType Directory -Force -Path (Join-Path $workDir 'tmp') | Out-Null
$env:TMP=Join-Path $workDir 'tmp';$env:TEMP=$env:TMP;$env:PYTHONPATH=$projectDir
Set-Location -LiteralPath $projectDir
$resultDir='runs/'+$Name
if(Test-Path -LiteralPath $resultDir){throw 'Run directory already exists. Choose a fresh name; no evidence will be overwritten.'}
New-Item -ItemType Directory -Path $resultDir | Out-Null
function Invoke-Stage([string[]]$StageArgs){ & uv run --frozen python @StageArgs; if($LASTEXITCODE -ne 0){throw 'Stage failed. Inspect its preserved log.'} }
if(-not (Test-Path -LiteralPath 'data/v1/manifest.json')){ Invoke-Stage @('-m','tabcrew_rl.data') }
Invoke-Stage @('scripts/bounded_run.py','--seconds','360','--log',"$resultDir/baseline.log",'--','python','-m','tabcrew_rl.evaluate','--split','validation','--max-tabs','12','--out',"$resultDir/baseline")
Invoke-Stage @('scripts/bounded_run.py','--seconds','720','--log',"$resultDir/grpo.log",'--','python','-m','tabcrew_rl.train','--out',"$resultDir/grpo")
Invoke-Stage @('scripts/bounded_run.py','--seconds','360','--log',"$resultDir/sft.log",'--','python','-m','tabcrew_rl.sft','--out',"$resultDir/sft")
Invoke-Stage @('scripts/bounded_run.py','--seconds','720','--log',"$resultDir/grpo-sft.log",'--','python','-m','tabcrew_rl.train','--config','configs/grpo-after-sft-pilot.json','--init-adapter',"$resultDir/sft/adapter",'--out',"$resultDir/grpo-sft")
Invoke-Stage @('scripts/final_suite.py','--results',$resultDir)
Invoke-Stage @('-m','tabcrew_rl.compare','--before',"$resultDir/final-sft",'--after',"$resultDir/final-sft-grpo",'--out',"$resultDir/incremental-rl-comparison.json")
