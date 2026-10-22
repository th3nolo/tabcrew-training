param([ValidateRange(0.01,8)][double]$BudgetUsd=8, [ValidateRange(1,4)][int]$Workers=4,
      [ValidateRange(0.01,24)][double]$MaxHours=24, [ValidateRange(1,5000)][int]$MaxNew=5000)
& (Join-Path $PSScriptRoot 'start.ps1') -Resume -BudgetUsd $BudgetUsd -Workers $Workers -MaxHours $MaxHours -MaxNew $MaxNew
