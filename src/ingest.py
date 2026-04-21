"""
Step 1 — Data Ingestion
Pull all race laps from 2023, 2024, 2025, 2026 seasons via FastF1.
Filter to accurate, clean racing laps. Aggregate per-lap telemetry.
Save to data/laps_2023_2026.parquet.
"""

import os
import sys
import numpy as np
import pandas as pd
import fastf1

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "fastf1_cache")
OUTPUT_FILE = os.path.join(DATA_DIR, "laps_2023_2026.parquet")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

fastf1.Cache.enable_cache(CACHE_DIR)

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

        speed    = car['Speed']
        throttle = car['Throttle']
        rpm      = car['RPM']
        brake    = car['Brake'].astype(float)
        drs      = car['DRS']

        return {
            'mean_speed':        speed.mean()              if not speed.empty    else np.nan,
            'max_speed':         speed.max()               if not speed.empty    else np.nan,
            'mean_throttle':     throttle.mean()           if not throttle.empty else np.nan,
            'pct_full_throttle': (throttle > 98).mean()   if not throttle.empty else np.nan,
            # mean_brake   : average brake pressure (0–100 scale)
            # pct_braking  : fraction of lap samples where brake is applied at all
            'mean_brake':        brake.mean()              if not brake.empty    else np.nan,
            'pct_braking':       (brake > 0).mean()        if not brake.empty    else np.nan,
            'mean_rpm':          rpm.mean()                if not rpm.empty      else np.nan,
            # Fraction of lap with DRS open (was binary int — lost information)
            'drs_active':        float((drs >= 10).mean()) if not drs.empty      else 0.0,
        }
    except Exception:
        return None


def attach_weather(filtered_df, session):
    """
    Vectorised nearest-match weather join using merge_asof.
    Replaces the O(n * m) per-lap loop.
    """
    weather = session.weather_data
    if weather is None or weather.empty or 'LapStartTime' not in filtered_df.columns:
        filtered_df['AirTemp']   = np.nan
        filtered_df['TrackTemp'] = np.nan
        return filtered_df

    w = (weather[['Time', 'AirTemp', 'TrackTemp']]
         .dropna(subset=['Time'])
         .sort_values('Time')
         .rename(columns={'Time': 'LapStartTime'}))

    laps_sorted = filtered_df.sort_values('LapStartTime').copy()
    merged = pd.merge_asof(laps_sorted, w, on='LapStartTime', direction='nearest')

    # merge_asof adds the columns in-place; restore original index order
    return merged.reindex(filtered_df.index)


def has_sc_vsc(track_status_str):
    """Check if a lap's TrackStatus string contains SC/VSC/Red flag codes."""
    if pd.isna(track_status_str):
        return False
    return any(code in str(track_status_str) for code in SC_VSC_CODES)


def process_season(year):
    """Process all race sessions for a given season year."""
    print(f"\n{'='*60}")
    print(f"  Processing {year} season")
    print(f"{'='*60}")

    schedule = fastf1.get_event_schedule(year)
    schedule = schedule[schedule['EventFormat'].notna()]
    schedule = schedule[schedule['RoundNumber'] > 0]

    season_laps = []

    for _, event in schedule.iterrows():
        event_name = event['EventName']
        round_num  = event['RoundNumber']
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
        except Exception:
            print(f"    Data not loaded for this race (likely hasn't occurred yet). Skipping.")
            continue

        # ── Core Filters ────────────────────────────────────────────────
        mask = laps['IsAccurate'] == True
        mask = mask & laps['LapTime'].notna()

        lap_secs = laps['LapTime'].dt.total_seconds()
        mask = mask & (lap_secs >= 60) & (lap_secs <= 200)

        mask = mask & laps['PitInTime'].isna()
        mask = mask & laps['PitOutTime'].isna()

        if 'TrackStatus' in laps.columns:
            mask = mask & ~laps['TrackStatus'].apply(has_sc_vsc)

        filtered = laps[mask].copy()
        if filtered.empty:
            print(f"    ⚠ No valid laps after filtering")
            continue

        # ── Weather (vectorised) ─────────────────────────────────────────
        filtered = attach_weather(filtered, session)

        filtered['Circuit']     = event_name
        filtered['Year']        = year
        filtered['RoundNumber'] = round_num

        # ── Telemetry aggregation ────────────────────────────────────────
        print(f"    Aggregating telemetry for {len(filtered)} laps...")
        nan_telem = {
            'mean_speed': np.nan, 'max_speed': np.nan,
            'mean_throttle': np.nan, 'pct_full_throttle': np.nan,
            'mean_brake': np.nan, 'pct_braking': np.nan,
            'mean_rpm': np.nan, 'drs_active': 0.0,
        }
        telem_rows = [aggregate_telemetry(laps.loc[idx]) or nan_telem for idx in filtered.index]
        telem_df   = pd.DataFrame(telem_rows, index=filtered.index)
        filtered   = pd.concat([filtered, telem_df], axis=1)

        # ── Select columns ───────────────────────────────────────────────
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
        keep_cols = [c for c in keep_cols if c in filtered.columns]
        filtered  = filtered[keep_cols]

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
    combined.to_parquet(OUTPUT_FILE, engine='pyarrow', index=False)

    print(f"\n{'='*60}")
    print(f"  ✓ DONE — Saved {len(combined)} laps to {OUTPUT_FILE}")
    print(f"  Seasons  : {sorted(combined['Year'].unique())}")
    print(f"  Circuits : {combined['Circuit'].nunique()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()