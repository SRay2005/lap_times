$env:PYTHONUTF8=1
Write-Host "Starting ingest.py..."
python src/ingest.py
if ($LASTEXITCODE -ne 0) { Write-Host "ingest.py failed!"; exit $LASTEXITCODE }

Write-Host "Starting features.py..."
python src/features.py
if ($LASTEXITCODE -ne 0) { Write-Host "features.py failed!"; exit $LASTEXITCODE }

Write-Host "Starting train.py..."
python src/train.py
if ($LASTEXITCODE -ne 0) { Write-Host "train.py failed!"; exit $LASTEXITCODE }

Write-Host "Starting infer_2026.py..."
python src/infer_2026.py
if ($LASTEXITCODE -ne 0) { Write-Host "infer_2026.py failed!"; exit $LASTEXITCODE }

Write-Host "Starting visualise.py..."
python src/visualise.py
if ($LASTEXITCODE -ne 0) { Write-Host "visualise.py failed!"; exit $LASTEXITCODE }

Write-Host "All scripts finished successfully!"
