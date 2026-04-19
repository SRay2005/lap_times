"""
Step 3 — Model Training
Chronological train/val/test split. Baseline LinearRegression and XGBoost tuning.
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
INPUT_FILE = os.path.join(DATA_DIR, "features.parquet")
MODEL_FILE = os.path.join(MODELS_DIR, "xgb_lap_predictor.pkl")


def main():
    print("=" * 60)
    print("  F1 LAP TIME MODEL TRAINING")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        print(f"✗ Input file not found: {INPUT_FILE}")
        return

    df = pd.read_parquet(INPUT_FILE)

    # ── 1. Chronological Split ──────────────────────────────────────────
    # Train: 2023, 2024
    # Val: 2025 first half
    # Test: 2025 second half
    
    rounds_2025 = df[df['Year'] == 2025]['RoundNumber'].unique()
    if len(rounds_2025) > 0:
        median_round_2025 = np.median(rounds_2025)
    else:
        median_round_2025 = 12 # Fallback if no 2025 data
    
    train_mask = (df['Year'] == 2023) | (df['Year'] == 2024)
    val_mask = (df['Year'] == 2025) & (df['RoundNumber'] <= median_round_2025)
    test_mask = (df['Year'] == 2025) & (df['RoundNumber'] > median_round_2025)
    
    # If there's no test set (e.g., incomplete 2025), fallback strategy
    if not test_mask.any():
        print("  ⚠ No data for 2025 second half. Adjusting split...")
        val_mask = (df['Year'] == 2025) & (df['RoundNumber'] <= np.percentile(rounds_2025, 50))
        test_mask = (df['Year'] == 2025) & (df['RoundNumber'] > np.percentile(rounds_2025, 50))
    if not test_mask.any():
        print("  ⚠ Still no test data. Reverting to random 80/10/10 split (fallback)")
        shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        n = len(shuffled)
        train_mask = shuffled.index < int(0.8 * n)
        val_mask = (shuffled.index >= int(0.8 * n)) & (shuffled.index < int(0.9 * n))
        test_mask = shuffled.index >= int(0.9 * n)
        df = shuffled
    
    train_df = df[train_mask]
    val_df = df[val_mask]
    test_df = df[test_mask]

    print(f"  Train: {len(train_df)} laps")
    print(f"  Val:   {len(val_df)} laps")
    print(f"  Test:  {len(test_df)} laps")

    features = [
        'compound_encoded', 'circuit_encoded', 'TyreLife', 'fuel_lap_number',
        'AirTemp', 'TrackTemp', 'Year',
        'mean_speed', 'max_speed', 'mean_throttle', 'pct_full_throttle',
        'mean_brake', 'pct_braking', 'mean_rpm', 'drs_active'
    ]
    target = 'LapTime_seconds'

    X_train, y_train = train_df[features], train_df[target]
    X_val, y_val = val_df[features], val_df[target]
    X_test, y_test = test_df[features], test_df[target]

    # ── 2. Baseline Linear Regression ───────────────────────────────────
    print("\n  [Baseline: Linear Regression]")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    
    lr_mae = mean_absolute_error(y_test, lr_preds)
    lr_r2 = r2_score(y_test, lr_preds)
    
    print(f"  LR Test MAE: {lr_mae:.4f}s")
    print(f"  LR Test R²:  {lr_r2:.4f}")

    # ── 3. XGBoost Grid Search w/ Early Stopping ────────────────────────
    print("\n  [XGBoost Tuning]")
    
    param_grid = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [200, 500]
    }
    
    best_mae = float('inf')
    best_model = None
    best_params = None

    import itertools
    keys = param_grid.keys()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*param_grid.values())]
    
    for i, params in enumerate(combinations):
        print(f"    Evaluating {i+1}/{len(combinations)}: {params}...")
        model = xgb.XGBRegressor(
            **params,
            random_state=42,
            eval_metric="mae",
            early_stopping_rounds=50
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        val_preds = model.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_preds)
        
        if val_mae < best_mae:
            best_mae = val_mae
            best_model = model
            best_params = params

    print(f"\n  Best Params: {best_params}")
    print(f"  Best Val MAE: {best_mae:.4f}s")

    # ── 4. Final Evaluation on Test Set ─────────────────────────────────
    print("\n  [Evaluating Best Model on Test Set]")
    xgb_preds = best_model.predict(X_test)
    
    final_mae = mean_absolute_error(y_test, xgb_preds)
    final_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
    final_r2 = r2_score(y_test, xgb_preds)
    
    print(f"  Final Test MAE:  {final_mae:.4f}s")
    print(f"  Final Test RMSE: {final_rmse:.4f}s")
    print(f"  Final Test R²:   {final_r2:.4f}")

    # ── 5. Save best model ──────────────────────────────────────────────
    print(f"\n  Saving best model to {MODEL_FILE}...")
    joblib.dump(best_model, MODEL_FILE)
    
    # Save the test set predictions for visualization
    test_df_out = test_df.copy()
    test_df_out['predicted_laptime'] = xgb_preds
    preds_file = os.path.join(DATA_DIR, "test_predictions.parquet")
    test_df_out.to_parquet(preds_file, index=False)
    
    print("  ✓ DONE")
    print("=" * 60)

if __name__ == "__main__":
    main()
