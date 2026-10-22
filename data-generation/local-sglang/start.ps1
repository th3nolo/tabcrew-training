param(
    [ValidateSet('pilot','overnight')][string]$Mode='overnight',
    [ValidateRange(0.01,8)][double]$MaxHours=1,
    [ValidateRange(1,5000)][int]$NewExamples=5000,
    [switch]$Resume,[switch]$StartWsl,[switch]$CheckOnly
)
# Starts the local SGLang collector (Windows host, SGLang runtime inside WSL).
# Required environment:
#   TABCREW_SGLANG_RUNTIME  Linux path of the SGLang runtime folder inside WSL (.venv, uv.lock, model folder)
#   TABCREW_SGLANG_LOCK     Windows path of a reviewed copy of that runtime's uv.lock
# Optional: TABCREW_WSL_DISTRO (default Ubuntu), TABCREW_WORK_DIR (venv and uv cache; default <repo>/work).
$ErrorActionPreference='Stop'
if ($Mode -eq 'pilot') { $MaxHours=0.5 }
$root=$PSScriptRoot
$repo=Split-Path (Split-Path $root -Parent) -Parent
$workDir=if($env:TABCREW_WORK_DIR){$env:TABCREW_WORK_DIR}else{Join-Path $repo 'work'}
$distro=if($env:TABCREW_WSL_DISTRO){$env:TABCREW_WSL_DISTRO}else{'Ubuntu'}
if (-not $env:TABCREW_SGLANG_RUNTIME -or -not $env:TABCREW_SGLANG_LOCK) { throw 'Set TABCREW_SGLANG_RUNTIME and TABCREW_SGLANG_LOCK first.' }
$limits=Get-Content -LiteralPath (Join-Path $root 'settings.json') -Raw | ConvertFrom-Json
$requiredCommit=if ($Mode -eq 'pilot') { $limits.pilot_preflight_free_commit_gib } else { $limits.preflight_free_commit_gib }
# A saved contract pins the code hashes of an existing corpus. A new corpus has no contract yet.
$contractPath=Join-Path $root 'contract.json'
if (Test-Path -LiteralPath $contractPath) {
    $frozen=Get-Content -LiteralPath $contractPath -Raw | ConvertFrom-Json
    foreach ($entry in $frozen.code_sha256.PSObject.Properties) {
        if ((Get-FileHash -LiteralPath (Join-Path $root $entry.Name) -Algorithm SHA256).Hash.ToLower() -ne $entry.Value) {
            throw ('Frozen source changed: '+$entry.Name)
        }
    }
}
$lockPath=Join-Path $root 'launch.lock'
$lease=[System.IO.File]::Open($lockPath,[System.IO.FileMode]::OpenOrCreate,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::None)
try {
    $runtimeState=Join-Path $root 'runtime.json'
    if (Test-Path -LiteralPath $runtimeState) {
        $runtime=Get-Content -LiteralPath $runtimeState -Raw | ConvertFrom-Json
        if ($runtime.state -ne 'stopped') { throw 'Runtime shutdown is unconfirmed; inspect ownership before resuming.' }
    }
    $statusPath=Join-Path $root 'status.json'
    if (Test-Path -LiteralPath $statusPath) {
        $status=Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
        if ($status.state -eq 'running') { throw 'Collector still reports running; inspect its command identity.' }
    }
    $running=Get-CimInstance Win32_Process | Where-Object {
        $_.Name -like 'python*' -and $_.CommandLine -match 'data-generation[\\/]+local-[a-z]+[\\/]+(?:collector|generate)\.py'
    }
    if ($running) { throw 'An existing local collector is running; inspect before starting a duplicate.' }
    $memory=Get-CimInstance Win32_OperatingSystem
    $freeCommit=$memory.FreeVirtualMemory/1MB
    $freeRam=$memory.FreePhysicalMemory/1MB
    $gpu=& nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits
    if ($LASTEXITCODE -ne 0 -or @($gpu).Count -ne 1) { throw 'Expected one readable NVIDIA GPU.' }
    $gpuValues=$gpu.Split(',')
    $freeGpu=[double]::Parse($gpuValues[0],[cultureinfo]::InvariantCulture)
    $utilization=[double]::Parse($gpuValues[1],[cultureinfo]::InvariantCulture)
    if ($freeCommit -lt $requiredCommit) { throw ('Need '+$requiredCommit+' GiB free Windows commit for '+$Mode+'; currently '+[math]::Round($freeCommit,2)+' GiB. No runtime was started.') }
    if ($freeRam -lt $limits.preflight_free_ram_gib) { throw ('Need '+$limits.preflight_free_ram_gib+' GiB free physical RAM; currently '+[math]::Round($freeRam,2)+' GiB.') }
    if ($freeGpu -lt $limits.preflight_free_gpu_mib -or $utilization -gt 70) { throw 'GPU is occupied or lacks the reviewed memory headroom.' }
    [ordered]@{at=[DateTime]::UtcNow.ToString('o');free_commit_gib=$freeCommit;free_ram_gib=$freeRam;required_commit_gib=$requiredCommit;free_gpu_mib=$freeGpu;gpu_utilization_percent=$utilization;mode=$Mode;hours=$MaxHours} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $root 'preflight.json') -Encoding utf8
    if ($CheckOnly) { Write-Output 'Resource and frozen-code preflight passed; receipt saved. No runtime started.'; return }
    $distros=(& wsl.exe --list --running 2>&1 | Out-String).Replace([string][char]0,'')
    if ($distros -notmatch [regex]::Escape($distro)) {
        if (-not $StartWsl) { throw ('WSL distribution '+$distro+' is stopped. -StartWsl explicitly authorizes starting it for this run.') }
        & wsl.exe -d $distro --exec /bin/true
        if ($LASTEXITCODE -ne 0) { throw 'WSL distribution did not start.' }
    }
    $linuxProcesses=& wsl.exe -d $distro --exec /bin/ps -eo args
    if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect existing WSL processes.' }
    if ($linuxProcesses -match 'sglang.launch_server|sglang serve|sglang::schedul|(?:sglang|local-sglang)[^ ]*/entrypoint[.]py') { throw 'An SGLang runtime is already running; no duplicate will be launched.' }
    foreach ($name in @('PAUSE','RUNTIME_STOP')) {
        $flag=Join-Path $root $name
        if (Test-Path -LiteralPath $flag) {
            if (-not $Resume) { throw 'Stop flag is set; use resume.ps1.' }
        }
    }
    New-Item -ItemType Directory -Force -Path (Join-Path $root 'logs') | Out-Null
    $uv=(Get-Command uv -ErrorAction Stop).Source
    $runId=[guid]::NewGuid().ToString()
    $stamp=Get-Date -Format 'yyyyMMdd-HHmmss'
    $stdout=Join-Path $root ('logs/'+$stamp+'-'+$runId+'.out')
    $stderr=Join-Path $root ('logs/'+$stamp+'-'+$runId+'.err')
    $arguments=@('run','--offline','--frozen','--no-sync','python',('"'+(Join-Path $root 'collector.py')+'"'),
        '--run-id',$runId,'--max-hours',$MaxHours.ToString([cultureinfo]::InvariantCulture),'--max-new',$NewExamples,'--max-attempts','4')
    if ($Mode -eq 'pilot') { $arguments+='--pilot' }
    $originalEnvironment=@{}
    Get-ChildItem Env: | ForEach-Object { $originalEnvironment[$_.Name]=$_.Value }
    $allowed=@('PATH','SystemRoot','WINDIR','COMSPEC','PATHEXT','TEMP','TMP','USERPROFILE','APPDATA','LOCALAPPDATA',
        'HOMEDRIVE','HOMEPATH','PROCESSOR_ARCHITECTURE','NUMBER_OF_PROCESSORS','PROGRAMFILES','PROGRAMFILES(X86)','PROGRAMDATA','SYSTEMDRIVE',
        'USERNAME','USERDOMAIN','USERDOMAIN_ROAMINGPROFILE','SESSIONNAME','LOGONSERVER','PUBLIC','ALLUSERSPROFILE','PROGRAMW6432',
        'COMMONPROGRAMFILES','COMMONPROGRAMFILES(X86)','COMMONPROGRAMW6432','OS','PSMODULEPATH','PROCESSOR_IDENTIFIER','PROCESSOR_LEVEL','PROCESSOR_REVISION',
        'TABCREW_SGLANG_RUNTIME','TABCREW_SGLANG_LOCK','TABCREW_WSL_DISTRO')
    try {
        foreach ($name in @($originalEnvironment.Keys)) { if ($name -notin $allowed) { [Environment]::SetEnvironmentVariable($name,$null,'Process') } }
        $env:UV_PROJECT_ENVIRONMENT=Join-Path $workDir 'venv-rl'
        $env:UV_CACHE_DIR=Join-Path $workDir 'uv-cache'
        $env:PYTHONUNBUFFERED='1'
        foreach ($name in @('PAUSE','RUNTIME_STOP')) {
            $flag=Join-Path $root $name
            if (Test-Path -LiteralPath $flag) { Remove-Item -LiteralPath $flag }
        }
        $launcher=Start-Process -FilePath $uv -ArgumentList $arguments -WorkingDirectory $repo -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    } catch {
        Set-Content -LiteralPath (Join-Path $root 'PAUSE') 'Launch failed; inspect logs before resume.'
        Set-Content -LiteralPath (Join-Path $root 'RUNTIME_STOP') 'Launch failed; stop owned runtime.'
        throw
    } finally {
        foreach ($name in @('UV_PROJECT_ENVIRONMENT','UV_CACHE_DIR','PYTHONUNBUFFERED')) { [Environment]::SetEnvironmentVariable($name,$null,'Process') }
        foreach ($name in $originalEnvironment.Keys) { [Environment]::SetEnvironmentVariable($name,$originalEnvironment[$name],'Process') }
    }
    [ordered]@{launcher_pid=$launcher.Id;run_id=$runId;started_at=[DateTime]::UtcNow.ToString('o');mode=$Mode;max_hours=$MaxHours;max_new=$NewExamples;workers=4;stdout=$stdout;stderr=$stderr} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $root 'launch.json') -Encoding utf8
    Write-Output ('Started bounded '+$Mode+' collector run '+$runId+'; launcher PID '+$launcher.Id)
} finally { $lease.Dispose() }
