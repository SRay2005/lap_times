"""
Step 1 — Data Ingestion
Pull all race laps from 2023, 2024, 2025, 2026 seasons via FastF1.
Filter to accurate, clean racing laps. Aggregate per-lap telemetry.
Save to data/laps_2023_2026.parquet.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import fastf1

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "fastf1_cache")
OUTPUT_FILE = os.path.join(DATA_DIR, "laps_2023_2026.parquet")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Enable FastF1 cache
fastf1.Cache.enable_cache(CACHE_DIR)

# ── Track status codes that indicate SC / VSC / red flag ───────────────
# FastF1 TrackStatus: '1'=Green, '2'=Yellow, '4'=SC, '5'=Red, '6'=VSC, '7'=VSC Ending
SC_VSC_CODES = {'4', '5', '6', '7'}

SEASONS = [2023, 2024, 2025, 2026]


def aggregate_telemetry(lap):
    """
    Compute per-lap telemetry aggregates from car data.
    Returns a dict of features or None if telemetry unavailable.
    """
    try:
        car = lap.get_car_data()
        if car is None or car.empty:
            return None

        speed = car['Speed']
        throttle = car['Throttle']
        rpm = car['RPM']
        brake = car['Brake']
        drs = car['DRS']

        result = {
            'mean_speed': speed.mean() if not speed.empty else np.nan,
            'max_speed': speed.max() if not speed.empty else np.nan,
            'mean_throttle': throttle.mean() if not throttle.empty else np.nan,
            'pct_full_throttle': (throttle > 98).mean() if not throttle.empty else np.nan,
            'mean_brake': brake.astype(float).mean() if not brake.empty else np.nan,
            'pct_braking': brake.astype(float).mean() if not brake.empty else np.nan,
            'mean_rpm': rpm.mean() if not rpm.empty else np.nan,
            'drs_active': int((drs >= 10).any()) if not drs.empty else 0,
        }
        return result
    except Exception:
        return None


def get_weather_for_lap(session, lap):
    """Get nearest weather data for a lap's start time."""
    air_temp = np.nan
    track_temp = np.nan
    try:
        weather = session.weather_data
        if weather is not None and not weather.empty and hasattr(lap, 'LapStartTime'):
            lap_time = lap['LapStartTime']
            if pd.notna(lap_time):
                # Find nearest weather reading
                idx = (weather['Time'] - lap_time).abs().idxmin()
                air_temp = weather.loc[idx, 'AirTemp'] if 'AirTemp' in weather.columns else np.nan
                track_temp = weather.loc[idx, 'TrackTemp'] if 'TrackTemp' in weather.columns else np.nan
    except Exception:
        pass
    return air_temp, track_temp


def has_sc_vsc(track_status_str):
    """Check if a lap's TrackStatus string contains SC/VSC/Red flag codes."""
    if pd.isna(track_status_str):
        return False
    status_str = str(track_status_str)
    return any(code in status_str for code in SC_VSC_CODES)


def process_season(year):
    """Process all race sessions for a given season year."""
    print(f"\n{'='*60}")
    print(f"  Processing {year} season")
    print(f"{'='*60}")

    schedule = fastf1.get_event_schedule(year)
    # Filter to conventional rounds (exclude pre-season testing)
    schedule = schedule[schedule['EventFormat'].notna()]
    # Only keep actual race rounds (RoundNumber > 0)
    schedule = schedule[schedule['RoundNumber'] > 0]

    season_laps = []

    for _, event in schedule.iterrows():
        event_name = event['EventName']
        round_num = event['RoundNumber']
        print(f"\n  [{year} R{round_num}] {event_name}...")

        try:
            session = fastf1.get_session(year, round_num, 'R')
            session.load(laps=True, telemetry=True, weather=True, messages=True)
        except Exception as e:
            print(f"    ⚠ Failed to load session: {e}")
            continue

        try:
            laps = session.laps
            if laps is None or laps.empty:
                print(f"    No lap data available")
                continue
        except Exception as e:
            print(f"    Data not loaded for this race (likely hasn't occurred yet). Skipping.")
            continue

        # ── Core Filters ────────────────────────────────────────
        # 1. Accurate laps only
        mask = laps['IsAccurate'] == True

        # 2. Valid LapTime
        valid_lt = laps['LapTime'].notna()
        mask = mask & valid_lt

        # 3. LapTime between 60s and 200s
        lap_secs = laps['LapTime'].dt.total_seconds()
        mask = mask & (lap_secs >= 60) & (lap_secs <= 200)

        # 4. Exclude in-laps and out-laps (PitInTime / PitOutTime present)
        mask = mask & laps['PitInTime'].isna()
        mask = mask & laps['PitOutTime'].isna()

        # 5. Filter out SC / VSC / incident laps via TrackStatus
        if 'TrackStatus' in laps.columns:
            sc_mask = laps['TrackStatus'].apply(has_sc_vsc)
            mask = mask & ~sc_mask

        filtered = laps[mask].copy()
        if filtered.empty:
            print(f"    ⚠ No valid laps after filtering")
            continue

        # ── Circuit & weather info ──────────────────────────────
        circuit_name = event_name  # Use event name as circuit identifier

        # Get weather data once for the session
        weather_air = []
        weather_track = []
        for _, lap in filtered.iterrows():
            at, tt = get_weather_for_lap(session, lap)
            weather_air.append(at)
            weather_track.append(tt)

        filtered['AirTemp'] = weather_air
        filtered['TrackTemp'] = weather_track
        filtered['Circuit'] = circuit_name
        filtered['Year'] = year
        filtered['RoundNumber'] = round_num

        # ── Telemetry aggregation ───────────────────────────────
        print(f"    Aggregating telemetry for {len(filtered)} laps...")
        telem_rows = []
        for idx in filtered.index:
            lap_obj = laps.loc[idx]
            telem = aggregate_telemetry(lap_obj)
            telem_rows.append(telem if telem else {
                'mean_speed': np.nan, 'max_speed': np.nan,
                'mean_throttle': np.nan, 'pct_full_throttle': np.nan,
                'mean_brake': np.nan, 'pct_braking': np.nan,
                'mean_rpm': np.nan, 'drs_active': 0,
            })

        telem_df = pd.DataFrame(telem_rows, index=filtered.index)
        filtered = pd.concat([filtered, telem_df], axis=1)

        # ── Select columns to keep ──────────────────────────────
        keep_cols = [
            'Driver', 'DriverNumber', 'Team', 'LapNumber', 'LapTime',
            'Stint', 'Compound', 'TyreLife', 'Position',
            'TrackStatus', 'IsAccurate',
            'Sector1Time', 'Sector2Time', 'Sector3Time',
            'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST',
            'Circuit', 'Year', 'RoundNumber',
            'AirTemp', 'TrackTemp',
            'mean_speed', 'max_speed', 'mean_throttle', 'pct_full_throttle',
            'mean_brake', 'pct_braking', 'mean_rpm', 'drs_active',
        ]
        # Keep only columns that exist
        keep_cols = [c for c in keep_cols if c in filtered.columns]
        filtered = filtered[keep_cols]

        season_laps.append(filtered)
        print(f"    ✓ {len(filtered)} clean laps collected")

    if season_laps:
        return pd.concat(season_laps, ignore_index=True)
    return pd.DataFrame()


def main():
    print("  F1 LAP TIME INGESTION — 2023, 2024, 2025, 2026")

    all_seasons = []
    for year in SEASONS:
        df = process_season(year)
        if not df.empty:
            all_seasons.append(df)
            print(f"\n  → {year}: {len(df)} total laps")

    if not all_seasons:
        print("\n✗ No data collected! Exiting.")
        sys.exit(1)

    combined = pd.concat(all_seasons, ignore_index=True)

    # Convert timedelta columns to serializable format for parquet
    for col in ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']:
        if col in combined.columns:
            # Keep as timedelta — pyarrow handles it natively
            pass

    combined.to_parquet(OUTPUT_FILE, engine='pyarrow', index=False)
    print(f"\n{'='*60}")
    print(f"  ✓ DONE — Saved {len(combined)} laps to {OUTPUT_FILE}")
    print(f"  Seasons: {combined['Year'].unique()}")
    print(f"  Circuits: {combined['Circuit'].nunique()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
