# run_all.ps1 - F1 Pipeline Runner
# Usage: .\run_all.ps1
# Logs to: pipeline.log (in project root, overwritten each run)

$env:PYTHONUTF8 = "1"

$LogFile = "pipeline.log"

# Clear log from previous run
Clear-Content -Path $LogFile -ErrorAction SilentlyContinue

function Log($msg) {
    $line = "[$(Get-Date -Format 'HH:mm:ss')] $msg"
    Write-Host $line
    Add-Content -Encoding UTF8 -Path $LogFile -Value $line
}

function Run-Step($name, $script) {
    Log "--- $name ---"
    $start = Get-Date

    python $script 2>&1 | ForEach-Object {
        Write-Host $_
        Add-Content -Encoding UTF8 -Path $LogFile -Value $_
    }

    $elapsed = [math]::Round(((Get-Date) - $start).TotalSeconds, 1)

    if ($LASTEXITCODE -ne 0) {
        Log "FAILED: $name after ${elapsed}s (exit code $LASTEXITCODE)"
        exit 1
    }

    Log "OK: $name completed in ${elapsed}s"
    Log ""
}

$pipelineStart = Get-Date
Log "F1 Pipeline starting"
Log ""

Run-Step "ingest.py"     "src/ingest.py"
Run-Step "features.py"   "src/features.py"
Run-Step "train.py"      "src/train.py"
Run-Step "infer_2026.py" "src/infer_2026.py"
Run-Step "visualise.py"  "src/visualise.py"

$total = [math]::Round(((Get-Date) - $pipelineStart).TotalSeconds, 1)
Log "All steps completed in ${total}s"