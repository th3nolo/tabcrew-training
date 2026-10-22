$ErrorActionPreference = 'Stop'
Set-Content -LiteralPath (Join-Path $PSScriptRoot 'PAUSE') -Value 'Pause requested by local operator.' -Encoding utf8
Write-Output 'Pause requested. Active calls will finish, candidate stages will be saved, and the owned SGLang runtime will release the GPU. Check status.json for confirmation.'
