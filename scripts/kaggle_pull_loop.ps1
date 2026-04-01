param(
  [Parameter(Mandatory = $true)]
  [string]$KernelRef, # e.g. dzqqdz/csi300-1

  [Parameter(Mandatory = $true)]
  [string]$OutputRoot, # e.g. C:\Users\qdz\Desktop\开题报告\alphagen\data\kaggle_pull

  [string]$MergeRoot = "", # e.g. C:\Users\qdz\Desktop\开题报告\alphagen\data
  [int]$InitialDelayMinutes = 0,
  [int]$IntervalMinutes = 24
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
if ($MergeRoot -ne "") {
  New-Item -ItemType Directory -Force -Path $MergeRoot | Out-Null
}

$latestDir = Join-Path $OutputRoot "latest"
New-Item -ItemType Directory -Force -Path $latestDir | Out-Null

Write-Host "[start] kernel=$KernelRef initial_delay=${InitialDelayMinutes}m interval=${IntervalMinutes}m output=$OutputRoot"
Write-Host "[start] pull policy: always try pulling latest available output"

if ($InitialDelayMinutes -gt 0) {
  Write-Host "[start] waiting initial delay ${InitialDelayMinutes}m ..."
  Start-Sleep -Seconds ($InitialDelayMinutes * 60)
}

while ($true) {
  $now = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  $statusText = (kaggle kernels status $KernelRef 2>&1 | Out-String).Trim()
  Write-Host "[$now] status check ..."
  Write-Host $statusText

  Write-Host "[$now] trying pull -> $latestDir"
  kaggle kernels output $KernelRef -p $latestDir --force
  $exitCode = $LASTEXITCODE

  if ($exitCode -eq 0) {
    if ($MergeRoot -ne "") {
      $srcRuns = Join-Path $latestDir "runs"
      $srcCkpt = Join-Path $latestDir "checkpoints"
      $srcTb = Join-Path $latestDir "tb_log"

      if (Test-Path $srcRuns) {
        robocopy $srcRuns (Join-Path $MergeRoot "runs") /E /NFL /NDL /NJH /NJS /NP | Out-Null
      }
      if (Test-Path $srcCkpt) {
        robocopy $srcCkpt (Join-Path $MergeRoot "checkpoints") /E /NFL /NDL /NJH /NJS /NP | Out-Null
      }
      if (Test-Path $srcTb) {
        robocopy $srcTb (Join-Path $MergeRoot "tb_log") /E /NFL /NDL /NJH /NJS /NP | Out-Null
      }
      Write-Host "[$now] merged into $MergeRoot"
    }
  } else {
    Write-Host "[$now] pull skipped/failed (exit=$exitCode). will retry next interval."
  }

  Start-Sleep -Seconds ([Math]::Max(60, $IntervalMinutes * 60))
}
