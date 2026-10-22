param([switch]$Resume, [ValidateRange(0.01,8)][double]$BudgetUsd=8,
      [ValidateRange(1,4)][int]$Workers=4, [ValidateRange(0.01,24)][double]$MaxHours=24,
      [ValidateRange(1,5000)][int]$MaxNew=5000)
# Starts the DeepSeek collector in the background with the repository's uv environment.
# The API key must already be in the DEEPSEEK_API_KEY environment variable of this PowerShell session.
# The collector reads it once and removes it from its own environment.
# Optional: TABCREW_WORK_DIR (venv and uv cache; default <repo>/work).
$ErrorActionPreference='Stop'
$repo=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$workDir=if($env:TABCREW_WORK_DIR){$env:TABCREW_WORK_DIR}else{Join-Path $repo 'work'}
if(-not $env:DEEPSEEK_API_KEY){throw 'Set DEEPSEEK_API_KEY in this session before starting.'}
$statusPath=Join-Path $PSScriptRoot 'status.json'
if (Test-Path -LiteralPath $statusPath) {
    $saved=Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
    if ($saved.pid) {
        $recordedProcess = Get-CimInstance Win32_Process -Filter ('ProcessId = ' + [int]$saved.pid)
        if ($recordedProcess) {
            if (-not $recordedProcess.CommandLine) { throw 'Cannot verify the recorded process identity; inspect before starting.' }
            $collectorPath = Join-Path $PSScriptRoot 'collector.py'
            if ($recordedProcess.CommandLine.Contains($collectorPath)) { throw 'The recorded collector is still alive; wait for draining.' }
            Write-Output 'The historical PID was reused by another process; the collector lock will prevent duplicate collection.'
        }
    }
}
$pausePath=Join-Path $PSScriptRoot 'PAUSE'
if (Test-Path -LiteralPath $pausePath) {
    if (!$Resume) { throw 'This corpus is paused. Use resume.ps1 after the collector has drained.' }
    Remove-Item -LiteralPath $pausePath
}
$oldCache=$env:UV_CACHE_DIR
$oldEnvironment=$env:UV_PROJECT_ENVIRONMENT
try {
    $env:UV_CACHE_DIR=Join-Path $workDir 'uv-cache'
    $env:UV_PROJECT_ENVIRONMENT=Join-Path $workDir 'venv-rl'
    $logs=Join-Path $PSScriptRoot 'logs'
    New-Item -ItemType Directory -Path $logs -Force | Out-Null
    $stamp=Get-Date -Format 'yyyyMMdd-HHmmss'
    $arguments=@('run','--offline','--frozen','--no-sync','python',('"'+(Join-Path $PSScriptRoot 'collector.py')+'"'),
        '--budget-usd',$BudgetUsd.ToString([Globalization.CultureInfo]::InvariantCulture),
        '--workers',([string]$Workers),'--max-hours',$MaxHours.ToString([Globalization.CultureInfo]::InvariantCulture),
        '--max-new',([string]$MaxNew))
    $uv=(Get-Command uv -ErrorAction Stop).Source
    $child=Start-Process -FilePath $uv -ArgumentList $arguments -WorkingDirectory $repo -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logs ($stamp+'-stdout.log')) -RedirectStandardError (Join-Path $logs ($stamp+'-stderr.log')) -PassThru
    [ordered]@{launcher_pid=$child.Id;started_at=[DateTime]::UtcNow.ToString('o');workers=$Workers;budget_usd=$BudgetUsd;max_hours=$MaxHours;max_new=$MaxNew;stdout=(Join-Path $logs ($stamp+'-stdout.log'));stderr=(Join-Path $logs ($stamp+'-stderr.log'))} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'launch.json') -Encoding utf8
    Write-Output ('Started DeepSeek collector launcher PID '+$child.Id)
} finally {
    $env:DEEPSEEK_API_KEY=$null
    $env:UV_CACHE_DIR=$oldCache
    $env:UV_PROJECT_ENVIRONMENT=$oldEnvironment
}
