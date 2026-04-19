"""
Step 4 — 2026 Inference
Pull 2026 races, feature engineer using existing encoders, and apply XGBoost
to generate predictions and compute delta (Regulation Impact).
"""

import os
import sys
import pandas as pd
import joblib

# Ensure we can import from src.ingest and src.features
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.ingest import process_season
from src.features import engineer_features

# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_FILE = os.path.join(DATA_DIR, "2026_impact.parquet")
ENCODERS_FILE = os.path.join(MODELS_DIR, "label_encoders.pkl")
MODEL_FILE = os.path.join(MODELS_DIR, "xgb_lap_predictor.pkl")


def main():
    print("=" * 60)
    print("  F1 2026 REGULATION IMPACT INFERENCE")
    print("=" * 60)

    if not os.path.exists(MODEL_FILE):
        print(f"✗ Model not found at {MODEL_FILE}. Run train.py first.")
        return

    if not os.path.exists(ENCODERS_FILE):
        print(f"✗ Encoders not found at {ENCODERS_FILE}. Run features.py first.")
        return

    print("  Loading model and encoders...")
    model = joblib.load(MODEL_FILE)
    encoders = joblib.load(ENCODERS_FILE)

    # 1. Pull 2026 data
    print("  Pulling 2026 race data...")
    df_2026 = process_season(2026)
    
    if df_2026.empty:
        print("\n  ⚠ No 2026 race data available yet!")
        return

    # 2. Engineer features
    print("  Engineering features for 2026 data...")
    features_2026 = engineer_features(df_2026, is_train=False, encoders=encoders)
    
    if features_2026.empty:
        print("\n  ⚠ No valid features generated for 2026.")
        return
        
    features_cols = [
        'compound_encoded', 'circuit_encoded', 'TyreLife', 'fuel_lap_number',
        'AirTemp', 'TrackTemp', 'Year',
        'mean_speed', 'max_speed', 'mean_throttle', 'pct_full_throttle',
        'mean_brake', 'pct_braking', 'mean_rpm', 'drs_active'
    ]

    # 3. Predict & Compute Deltas
    print("  Generating predictions...")
    X_2026 = features_2026[features_cols]
    preds = model.predict(X_2026)
    
    # Store results
    results = df_2026[['Circuit', 'Driver', 'Team', 'LapNumber', 'Compound']].copy()
    
    # Check if actual elapsed time exists (FastF1 lap time)
    if 'LapTime_seconds' in features_2026.columns:
        results['actual_laptime'] = features_2026['LapTime_seconds'].values
    else:
        results['actual_laptime'] = df_2026['LapTime'].dt.total_seconds().values
        
    results['predicted_laptime'] = preds
    results['delta'] = results['actual_laptime'] - results['predicted_laptime']
    
    print(f"  Aggregated 2026 Delta (Actual - Predicted): {results['delta'].mean():.4f}s")
    
    print(f"  Saving results to {OUTPUT_FILE}...")
    results.to_parquet(OUTPUT_FILE, index=False)
    
    print("  ✓ DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
