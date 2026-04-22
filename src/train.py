"""
Step 3 - Model Training
Chronological train/val/test split. Baseline LinearRegression and XGBoost tuning.
"""

import os
import sys
import itertools
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Path bootstrap - needed so 'from src.features import ...' resolves correctly
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.features import FEATURE_COLS   # single source of truth

# -- Paths -----------------------------------------------------------------
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
INPUT_FILE = os.path.join(DATA_DIR, "features.parquet")
MODEL_FILE = os.path.join(MODELS_DIR, "xgb_lap_predictor.pkl")

TARGET = 'LapTime_seconds'


def make_split(df):
    """
    Chronological split:
        Train : 2023-2025
        Val   : first half of available 2026 rounds
        Test  : second half of available 2026 rounds

    Falls back to 80/10/10 random split if 2026 data is insufficient.
    """
    rounds_2026 = df[df['Year'] == 2026]['RoundNumber'].unique()

    if len(rounds_2026) >= 2:
        median_round = np.median(rounds_2026)
        train_mask = (df['Year'] >= 2023) & (df['Year'] <= 2025)
        val_mask   = (df['Year'] == 2026) & (df['RoundNumber'] <= median_round)
        test_mask  = (df['Year'] == 2026) & (df['RoundNumber'] >  median_round)

        if test_mask.any():
            return train_mask, val_mask, test_mask, "chronological"

    print("  Warning: insufficient 2026 rounds for chronological split - using 80/10/10 fallback")
    shuffled   = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n          = len(shuffled)
    train_mask = shuffled.index < int(0.8 * n)
    val_mask   = (shuffled.index >= int(0.8 * n)) & (shuffled.index < int(0.9 * n))
    test_mask  = shuffled.index >= int(0.9 * n)
    return train_mask, val_mask, test_mask, "random"


def main():
    print("=" * 60)
    print("  F1 LAP TIME MODEL TRAINING")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        print(f"  Input file not found: {INPUT_FILE}")
        sys.exit(1)

    df = pd.read_parquet(INPUT_FILE)

    # -- 1. Split ----------------------------------------------------------
    train_mask, val_mask, test_mask, split_type = make_split(df)

    train_df = df[train_mask]
    val_df   = df[val_mask]
    test_df  = df[test_mask]

    print(f"  Split type : {split_type}")
    print(f"  Train      : {len(train_df)} laps")
    print(f"  Val        : {len(val_df)} laps")
    print(f"  Test       : {len(test_df)} laps")

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET]
    X_val,   y_val   = val_df[FEATURE_COLS],   val_df[TARGET]
    X_test,  y_test  = test_df[FEATURE_COLS],  test_df[TARGET]

    # -- 2. Baseline -------------------------------------------------------
    print("\n  [Baseline: Linear Regression]")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    print(f"  LR Test MAE : {mean_absolute_error(y_test, lr_preds):.4f}s")
    print(f"  LR Test R2  : {r2_score(y_test, lr_preds):.4f}")

    # -- 3. XGBoost grid search --------------------------------------------
    print("\n  [XGBoost Tuning]")

    param_grid = {
        'max_depth':     [4, 6, 8],
        'learning_rate': [0.05, 0.1],
        'n_estimators':  [200, 500],
    }

    best_mae   = float('inf')
    best_model = None
    best_params = None

    combos = [dict(zip(param_grid, v)) for v in itertools.product(*param_grid.values())]

    for i, params in enumerate(combos):
        print(f"    Evaluating {i+1}/{len(combos)}: {params}...")
        model = xgb.XGBRegressor(
            **params,
            random_state=42,
            eval_metric='mae',
            early_stopping_rounds=50,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        val_mae = mean_absolute_error(y_val, model.predict(X_val))
        if val_mae < best_mae:
            best_mae    = val_mae
            best_model  = model
            best_params = params

    print(f"\n  Best Params  : {best_params}")
    print(f"  Best Val MAE : {best_mae:.4f}s")

    # -- 4. Test set evaluation --------------------------------------------
    print("\n  [Evaluating Best Model on Test Set]")
    xgb_preds = best_model.predict(X_test)

    print(f"  Final Test MAE  : {mean_absolute_error(y_test, xgb_preds):.4f}s")
    print(f"  Final Test RMSE : {np.sqrt(mean_squared_error(y_test, xgb_preds)):.4f}s")
    print(f"  Final Test R2   : {r2_score(y_test, xgb_preds):.4f}")

    # -- 5. Save model -----------------------------------------------------
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"\n  Saving model to {MODEL_FILE}...")
    joblib.dump(best_model, MODEL_FILE)

    # Save test predictions for visualise.py Plot 2
    # Column names must match what visualise.py expects: actual_laptime, predicted_laptime
    test_preds = test_df[['Driver', 'Circuit', TARGET]].copy()
    test_preds = test_preds.rename(columns={TARGET: 'actual_laptime'})
    test_preds['predicted_laptime'] = xgb_preds
    test_preds.to_parquet(os.path.join(DATA_DIR, "test_predictions.parquet"), index=False)

    print("  Done")
    print("=" * 60)


if __name__ == "__main__":
    main()