param([ValidateSet('pilot','overnight')][string]$Mode='overnight',[ValidateRange(0.01,8)][double]$MaxHours=8,
      [ValidateRange(1,5000)][int]$NewExamples=5000,[switch]$StartWsl)
$ErrorActionPreference='Stop'
& (Join-Path $PSScriptRoot 'start.ps1') -Mode $Mode -MaxHours $MaxHours -NewExamples $NewExamples -Resume -StartWsl:$StartWsl
