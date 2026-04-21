"""
Step 2 — Feature Engineering
Load ingested parquet, engineer contextual and telemetry features.
Save the prepared feature set and label encoders.
"""

import os
import logging
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
INPUT_FILE = os.path.join(DATA_DIR, "laps_2023_2026.parquet")
OUTPUT_FILE = os.path.join(DATA_DIR, "features.parquet")

os.makedirs(MODELS_DIR, exist_ok=True)

# ── Shared feature column list — single source of truth ───────────────
# infer_2026.py and train.py must import from here, never redefine.
FEATURE_COLS = [
    'compound_encoded', 'circuit_encoded', 'TyreLife', 'fuel_lap_number',
    'AirTemp', 'TrackTemp', 'Year', 'RoundNumber',
    'mean_speed', 'max_speed', 'mean_throttle', 'pct_full_throttle',
    'mean_brake', 'pct_braking', 'mean_rpm', 'drs_active',
]

TELEMETRY_COLS = [
    'mean_speed', 'max_speed', 'mean_throttle', 'pct_full_throttle',
    'mean_brake', 'pct_braking', 'mean_rpm', 'drs_active',
]

# Columns filled with training medians — must match what we save in artifacts
MEDIAN_COLS = ['AirTemp', 'TrackTemp'] + TELEMETRY_COLS


def engineer_features(df, is_train=True, encoders=None):
    """
    Apply feature engineering to a dataframe of laps.

    Parameters
    ----------
    df        : raw laps dataframe from ingest
    is_train  : if True, fit encoders and compute medians from this data
    encoders  : dict with keys 'compound', 'circuit', 'medians'
                (required when is_train=False)

    Returns
    -------
    is_train=True  → (features_df, encoders)
    is_train=False → features_df
    """
    df = df.copy()
    print(f"  Input rows: {len(df)}")

    # ── 1. Target variable ─────────────────────────────────────────────
    if 'LapTime' in df.columns:
        if pd.api.types.is_timedelta64_dtype(df['LapTime']):
            df['LapTime_seconds'] = df['LapTime'].dt.total_seconds()
        elif pd.api.types.is_numeric_dtype(df['LapTime']):
            df['LapTime_seconds'] = df['LapTime'].astype(float)
        else:
            raise ValueError(f"Unexpected LapTime dtype: {df['LapTime'].dtype}")

    # ── 2. Compound & missing contextual fills ─────────────────────────
    df['Compound'] = df['Compound'].fillna('UNKNOWN')
    df['TyreLife'] = df['TyreLife'].fillna(1.0).replace([np.inf, -np.inf], 1.0)

    # ── 3. Median fills — use training medians at inference to avoid skew
    if is_train:
        train_medians = {col: float(df[col].median()) for col in MEDIAN_COLS if col in df.columns}
    else:
        if encoders is None or 'medians' not in encoders:
            raise ValueError("encoders dict with 'medians' key required for inference")
        train_medians = encoders['medians']

    for col in MEDIAN_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(train_medians.get(col, 0.0))
        else:
            log.warning("Column '%s' missing from input — filling with 0.0", col)
            df[col] = 0.0

    # ── 4. Encoding ────────────────────────────────────────────────────
    if is_train:
        encoders = {}

        # Ensure UNKNOWN is always a known class so inference can map it
        compounds = sorted(df['Compound'].astype(str).unique().tolist())
        if 'UNKNOWN' not in compounds:
            compounds = ['UNKNOWN'] + compounds

        le_compound = LabelEncoder()
        le_compound.fit(compounds)
        df['compound_encoded'] = le_compound.transform(df['Compound'].astype(str))
        encoders['compound'] = le_compound

        le_circuit = LabelEncoder()
        df['circuit_encoded'] = le_circuit.fit_transform(df['Circuit'].astype(str))
        encoders['circuit'] = le_circuit

        encoders['medians'] = train_medians

    else:
        le_compound = encoders['compound']
        le_circuit = encoders['circuit']

        compound_map = {cls: idx for idx, cls in enumerate(le_compound.classes_)}
        # Unseen compound → UNKNOWN (always present from training fit above)
        df['compound_encoded'] = (
            df['Compound'].astype(str)
            .map(lambda x: x if x in compound_map else 'UNKNOWN')
            .map(compound_map)
            .fillna(-1)
            .astype(int)
        )

        circuit_map = {cls: idx for idx, cls in enumerate(le_circuit.classes_)}
        unseen_circuits = set(df['Circuit'].astype(str).unique()) - set(circuit_map)
        if unseen_circuits:
            log.warning("Unseen circuits at inference — encoding as -1: %s", unseen_circuits)
        df['circuit_encoded'] = (
            df['Circuit'].astype(str)
            .map(circuit_map)
            .fillna(-1)
            .astype(int)
        )

    # ── 5. Derived features ────────────────────────────────────────────
    df['fuel_lap_number'] = df['LapNumber']

    # ── 6. Select output columns ───────────────────────────────────────
    keep_cols = FEATURE_COLS + ['LapTime_seconds', 'Driver', 'Circuit']

    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        log.warning("Columns missing from input, padding with NaN: %s", missing)
        for col in missing:
            df[col] = np.nan

    out_df = df[keep_cols].copy()

    if is_train:
        before = len(out_df)
        out_df = out_df.dropna(subset=['LapTime_seconds'])
        dropped = before - len(out_df)
        if dropped:
            print(f"  Dropped {dropped} rows with null LapTime_seconds")

    print(f"  Output rows: {len(out_df)}")

    if is_train:
        return out_df, encoders
    return out_df


def main():
    print("=" * 60)
    print("  F1 LAP TIME FEATURE ENGINEERING")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        print(f"✗ Input file not found: {INPUT_FILE}")
        return

    print("  Loading raw parquet...")
    df = pd.read_parquet(INPUT_FILE)

    print("  Engineering features...")
    features_df, encoders = engineer_features(df, is_train=True)

    print(f"  Saving features to {OUTPUT_FILE}...")
    features_df.to_parquet(OUTPUT_FILE, index=False)

    encoders_path = os.path.join(MODELS_DIR, "label_encoders.pkl")
    print(f"  Saving encoders to {encoders_path}...")
    joblib.dump(encoders, encoders_path)

    print("  ✓ DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()