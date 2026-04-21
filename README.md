# F1 2026 Lap Time Predictor

This repository contains a machine learning pipeline designed to predict Formula 1 lap times and analyze the impact of the upcoming 2026 technical regulations.

## Pipeline Overview

The project is structured into five sequential steps:

1. **Ingestion (`src/ingest.py`)**: Connects to the FastF1 API to pull historical telemetry and lap data for the 2023–2026 seasons. Cleans the data, filtering out Safety Car, VSC, and invalid laps.
2. **Feature Engineering (`src/features.py`)**: Processes the ingested `.parquet` data. Engineers track temperatures, fuel loads (via lap number proxies), tyre degradation, and car telemetry aggregates.
3. **Training (`src/train.py`)**: Trains an XGBoost Machine Learning model to predict lap times based on historical driver and car performance features.
4. **2026 Inference (`src/infer_2026.py`)**: Runs predictions on the 2026 season data. Computes the "Regulation Impact Delta" by comparing predicted pre-2026 lap times with the actual 2026 lap times.
5. **Visualization (`src/visualise.py`)**: Generates publication-quality charts in the `outputs/` directory to help analyze the regulation impacts.

## Getting Started

### Prerequisites

Make sure you have all required dependencies installed:
```bash
pip install -r requirements.txt
```

### Running the Pipeline

You can run the entire pipeline from start to finish using the included PowerShell script. This will run all 5 steps sequentially in the background and log the output.

```powershell
.\run_all.ps1
```
*(You can monitor progress by checking the `pipeline.log` file generated in the root directory).*

## Data Limits & Caching

The `fastf1` library caches significant amounts of data locally. To avoid running into GitHub's 50MB/100MB file limits, the `data/fastf1_cache/` folder is intentionally ignored in the `.gitignore`. 

If you clone this repository to a new machine, the initial run will take longer as FastF1 downloads and rebuilds the telemetry cache.
