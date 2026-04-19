"""
Step 2 — Feature Engineering
Load ingested parquet, engineer contextual and telemetry features.
Save the prepared feature set and label encoders.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
INPUT_FILE = os.path.join(DATA_DIR, "laps_2023_2025.parquet")
OUTPUT_FILE = os.path.join(DATA_DIR, "features.parquet")

os.makedirs(MODELS_DIR, exist_ok=True)


def engineer_features(df, is_train=True, encoders=None):
    """
    Apply feature engineering to a dataframe of laps.
    If is_train=True, fit LabelEncoders and return them.
    If is_train=False, use the provided encoders.
    """
    df = df.copy()

    print(f"  Input rows: {len(df)}")

    # 1. Target Variable
    if 'LapTime' in df.columns:
        if pd.api.types.is_timedelta64_dtype(df['LapTime']):
            df['LapTime_seconds'] = df['LapTime'].dt.total_seconds()
        else:
            # If loaded as float/int somehow
            df['LapTime_seconds'] = df['LapTime'].astype(float) / 1e9 if df['LapTime'].dtype.kind == 'm' else df['LapTime']
    
    # 2. Contextual Features
    # Fill NAs
    df['AirTemp'] = df['AirTemp'].fillna(df['AirTemp'].median())
    df['TrackTemp'] = df['TrackTemp'].fillna(df['TrackTemp'].median())
    df['Compound'] = df['Compound'].fillna('UNKNOWN')
    df['TyreLife'] = df['TyreLife'].fillna(1.0)
    
    # In some datasets TyreLife can be Na/inf
    df['TyreLife'] = df['TyreLife'].replace([np.inf, -np.inf], 1.0)

    # 3. Handle Telemetry NAs (if some laps failed telemetry aggregation)
    telem_cols = [
        'mean_speed', 'max_speed', 'mean_throttle', 'pct_full_throttle',
        'mean_brake', 'pct_braking', 'mean_rpm', 'drs_active'
    ]
    for col in telem_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = 0.0

    # 4. Encoding
    if is_train:
        encoders = {}
        
        le_compound = LabelEncoder()
        df['compound_encoded'] = le_compound.fit_transform(df['Compound'].astype(str))
        encoders['compound'] = le_compound
        
        le_circuit = LabelEncoder()
        df['circuit_encoded'] = le_circuit.fit_transform(df['Circuit'].astype(str))
        encoders['circuit'] = le_circuit
    else:
        if encoders is None:
            raise ValueError("Encoders must be provided for inference")
        
        le_compound = encoders['compound']
        le_circuit = encoders['circuit']

        # Handle unseen categories by assigning them to a special fallback or just -1
        # More safely: map known, default to -1 for unknown
        known_compounds = set(le_compound.classes_)
        df['Compound_mapped'] = df['Compound'].astype(str).apply(lambda x: x if x in known_compounds else 'UNKNOWN')
        # If UNKNOWN wasn't in train, we might still have a problem. Let's just add it if needed during inference.
        # Simple workaround for unseen labels:
        compound_dict = {cls: idx for idx, cls in enumerate(le_compound.classes_)}
        df['compound_encoded'] = df['Compound'].astype(str).map(compound_dict).fillna(-1).astype(int)
        
        circuit_dict = {cls: idx for idx, cls in enumerate(le_circuit.classes_)}
        df['circuit_encoded'] = df['Circuit'].astype(str).map(circuit_dict).fillna(-1).astype(int)

    # Proxy for fuel
    df['fuel_lap_number'] = df['LapNumber']

    # Keep necessary columns
    feature_cols = [
        'compound_encoded', 'circuit_encoded', 'TyreLife', 'fuel_lap_number',
        'AirTemp', 'TrackTemp', 'Year', 'RoundNumber'
    ] + telem_cols

    # Include Target and identifiers for tracking
    keep_cols = feature_cols + ['LapTime_seconds', 'Driver', 'Circuit']
    
    # We might miss target during completely blind inference, but we have it for 2026 actuals.
    missing = [c for c in keep_cols if c not in df.columns]
    for m in missing:
        df[m] = np.nan

    out_df = df[keep_cols].copy()
    
    # Drop rows without target LapTime_seconds during training dataset prep
    if is_train:
        out_df = out_df.dropna(subset=['LapTime_seconds'])
    
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
