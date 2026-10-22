$ErrorActionPreference='Stop'
[IO.File]::WriteAllText((Join-Path $PSScriptRoot 'PAUSE'),'User requested graceful pause. Finish current HTTP calls and preserve stage checkpoints.')
Write-Output 'Pause requested. Up to four in-flight HTTP calls may finish (timeout 180 seconds). Wait for status paused and pending_api_calls 0 before closing the computer or resuming.'
