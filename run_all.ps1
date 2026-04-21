# run_all.ps1 — F1 Pipeline Runner
# Usage: .\run_all.ps1
# Logs to: logs/run_YYYYMMDD_HHMMSS.log

$env:PYTHONUTF8 = "1"

# ── Log setup ──────────────────────────────────────────────────────────
$timestamp  = Get-Date -Format "yyyyMMdd_HHmmss"
$LogDir     = "logs"
$LogFile    = "$LogDir\run_$timestamp.log"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Log($msg) {
    $line = "[$(Get-Date -Format 'HH:mm:ss')] $msg"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

function Run-Step($name, $script) {
    Log "── $name ──────────────────────────────────────"
    $start = Get-Date

    # Tee stdout+stderr to both console and log
    python $script 2>&1 | Tee-Object -FilePath $LogFile -Append

    $elapsed = [math]::Round(((Get-Date) - $start).TotalSeconds, 1)

    if ($LASTEXITCODE -ne 0) {
        Log "✗ $name failed after ${elapsed}s (exit code $LASTEXITCODE)"
        Log "  See full log: $LogFile"
        exit 1
    }

    Log "✓ $name completed in ${elapsed}s"
    Log ""
}

# ── Pipeline ───────────────────────────────────────────────────────────
$pipelineStart = Get-Date
Log "F1 Pipeline starting — log: $LogFile"
Log ""

Run-Step "ingest.py"     "src/ingest.py"
Run-Step "features.py"   "src/features.py"
Run-Step "train.py"      "src/train.py"
Run-Step "infer_2026.py" "src/infer_2026.py"
Run-Step "visualise.py"  "src/visualise.py"

$total = [math]::Round(((Get-Date) - $pipelineStart).TotalSeconds, 1)
Log "All steps completed in ${total}s"