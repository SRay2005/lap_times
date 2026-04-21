"""
Step 4 — 2026 Inference
Pull 2026 races, feature engineer using existing encoders, and apply XGBoost
to generate predictions and compute delta (Regulation Impact).

Delta convention:
    regulation_delta = predicted - actual
    positive → car is faster than the pre-2026 model expected  → regulation gain
    negative → car is slower than expected                      → regulation cost
"""

import os
import sys
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.ingest import process_season
from src.features import engineer_features, FEATURE_COLS   # single source of truth

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

    # ── 1. Load model and encoders ─────────────────────────────────────
    print("  Loading model and encoders...")
    model = joblib.load(MODEL_FILE)
    encoders = joblib.load(ENCODERS_FILE)

    # Validate model was trained with current FEATURE_COLS before doing anything else
    model_features = model.get_booster().feature_names
    if model_features != FEATURE_COLS:
        print("\n  ✗ Feature mismatch between model and FEATURE_COLS.")
        print(f"    Model expects : {model_features}")
        print(f"    FEATURE_COLS  : {FEATURE_COLS}")
        print("\n  Retrain the model (run train.py) to sync with current features.")
        return

    # ── 2. Pull 2026 data ──────────────────────────────────────────────
    print("  Pulling 2026 race data...")
    df_2026 = process_season(2026)

    if df_2026.empty:
        print("\n  ⚠ No 2026 race data available yet!")
        return

    print(f"  Raw 2026 laps loaded: {len(df_2026)}")

    # ── 3. Feature engineering ─────────────────────────────────────────
    print("  Engineering features for 2026 data...")
    features_2026 = engineer_features(df_2026, is_train=False, encoders=encoders)
    features_2026 = features_2026.reset_index(drop=True)

    if features_2026.empty:
        print("\n  ⚠ No valid features generated for 2026.")
        return

    # ── 4. Check all required feature columns are present ─────────────
    missing_cols = [col for col in FEATURE_COLS if col not in features_2026.columns]
    if missing_cols:
        print(f"  ✗ Missing feature columns after engineering: {missing_cols}")
        return

    # ── 5. Predict ─────────────────────────────────────────────────────
    print("  Generating predictions...")
    X_2026 = features_2026[FEATURE_COLS]   # column order matches model exactly

    if X_2026.empty:
        print("\n  ⚠ No valid rows for prediction.")
        return

    preds = model.predict(X_2026)

    # ── 6. Build results from features_2026 (index-safe) ──────────────
    # Do NOT pull identifiers from df_2026 — lengths may differ after
    # feature engineering drops rows. features_2026 retains Driver/Circuit.
    results = features_2026[['Driver', 'Circuit']].copy()

    # Merge Team and Compound back via a reset-index join on shared columns
    meta_cols = ['Driver', 'Circuit', 'LapNumber', 'Team', 'Compound']
    available_meta = [c for c in meta_cols if c in df_2026.columns]
    df_meta = df_2026[available_meta].reset_index(drop=True)

    # Only join columns we don't already have
    new_meta = [c for c in available_meta if c not in results.columns]
    if new_meta:
        results = results.join(df_meta[new_meta])

    results['actual_laptime'] = features_2026['LapTime_seconds'].values
    results['predicted_laptime'] = preds

    # positive = faster than pre-2026 model expected (regulation gain)
    results['regulation_delta'] = results['predicted_laptime'] - results['actual_laptime']

    # ── 7. Summary ─────────────────────────────────────────────────────
    valid = results['regulation_delta'].dropna()
    total = len(results)
    print(f"\n  Laps predicted : {total}")
    print(f"  Valid deltas   : {len(valid)} ({total - len(valid)} NaN actual times skipped)")
    print(f"  Mean regulation delta : {valid.mean():.4f}s")
    print(f"    (positive = 2026 cars faster than baseline model expected)")

    # ── 8. Save ────────────────────────────────────────────────────────
    print(f"\n  Saving results to {OUTPUT_FILE}...")
    results.to_parquet(OUTPUT_FILE, index=False)

    print("  ✓ DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()