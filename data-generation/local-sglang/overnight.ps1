param([ValidateRange(1,5000)][int]$NewExamples=5000,[switch]$StartWsl)
$ErrorActionPreference='Stop'
& (Join-Path $PSScriptRoot 'resume.ps1') -Mode overnight -MaxHours 8 -NewExamples $NewExamples -StartWsl:$StartWsl
