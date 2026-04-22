"""
Step 5 - Visualisation
Generate 4 publication-quality plots.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Path bootstrap - allows 'from src.features import ...' when run as python src/visualise.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.features import FEATURE_COLS   # single source of truth - never redefine locally

# -- Paths -----------------------------------------------------------------
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUTS_DIR, exist_ok=True)

# -- Global style ----------------------------------------------------------
plt.style.use('dark_background')
sns.set_theme(
    style="darkgrid",
    rc={
        "axes.facecolor":   "#111111",
        "figure.facecolor": "#111111",
        "grid.color":       "#333333",
        "text.color":       "white",
        "axes.labelcolor":  "white",
        "xtick.color":      "white",
        "ytick.color":      "white",
    }
)

F1_RED  = '#E10600'
F1_TEAL = '#00D2BE'
F1_GREY = '#A6A6A6'


def save(filename):
    path = os.path.join(OUTPUTS_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved -> {path}")


# -- Plot 1 ----------------------------------------------------------------
def plot_feature_importance(model):
    print("  Generating Plot 1: Feature Importance...")
    importances = model.feature_importances_

    df = pd.DataFrame({'Feature': FEATURE_COLS, 'Importance': importances})
    df = df.sort_values('Importance', ascending=True)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=df, color=F1_RED)
    plt.title('XGBoost Feature Importance', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Relative Importance', fontsize=12)
    plt.ylabel('')
    save('feature_importance.png')


# -- Plot 2 ----------------------------------------------------------------
def plot_predicted_vs_actual(test_preds_file):
    print("  Generating Plot 2: Predicted vs Actual...")
    if not os.path.exists(test_preds_file):
        print("    Warning: Test predictions file missing, skipping.")
        return

    df = pd.read_parquet(test_preds_file)

    if df['Circuit'].nunique() > 10:
        top = df['Circuit'].value_counts().nlargest(10).index
        df  = df[df['Circuit'].isin(top)]

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='actual_laptime', y='predicted_laptime',
        hue='Circuit', data=df,
        alpha=0.6, palette='tab10', s=40
    )

    lo = min(df['actual_laptime'].min(), df['predicted_laptime'].min())
    hi = max(df['actual_laptime'].max(), df['predicted_laptime'].max())
    plt.plot([lo, hi], [lo, hi], 'w--', alpha=0.8, linewidth=2, label='Perfect fit')

    plt.title('Predicted vs Actual Lap Times (2025 Test Set)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Actual Lap Time (s)',    fontsize=12)
    plt.ylabel('Predicted Lap Time (s)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    save('predicted_vs_actual.png')


# -- Plot 3 ----------------------------------------------------------------
def plot_regulation_impact(impact_file):
    """
    Bar chart of mean regulation_delta per circuit with 95% CI error bars.

    regulation_delta = predicted - actual
        positive -> car is FASTER than baseline expected  -> teal
        negative -> car is SLOWER than baseline expected  -> red
    """
    print("  Generating Plot 3: 2026 Regulation Impact...")
    if not os.path.exists(impact_file):
        print("    Warning: 2026 impact file missing, skipping.")
        return

    df = pd.read_parquet(impact_file)
    if df.empty:
        print("    Warning: 2026 impact data is empty, skipping.")
        return

    stats = (df.groupby('Circuit')['regulation_delta']
               .agg(['mean', 'std', 'count'])
               .reset_index())
    stats['ci95'] = 1.96 * stats['std'] / np.sqrt(stats['count'])
    stats = stats.sort_values('mean', ascending=False)

    # positive delta = faster than expected = teal
    colors = [F1_TEAL if x > 0 else F1_RED for x in stats['mean']]

    plt.figure(figsize=(12, 7))
    plt.bar(
        stats['Circuit'], stats['mean'],
        yerr=stats['ci95'], capsize=5,
        color=colors, alpha=0.85, ecolor='white'
    )
    plt.axhline(0, color='white', linewidth=1)
    plt.title(
        '2026 Regulation Impact per Circuit\n'
        '(Predicted - Actual: positive = faster than pre-2026 baseline)',
        fontsize=14, fontweight='bold', pad=20
    )
    plt.ylabel('Delta (seconds)', fontsize=12)
    plt.xlabel('Circuit',         fontsize=12)
    plt.xticks(rotation=45, ha='right')
    save('regulation_impact.png')


# -- Plot 4 ----------------------------------------------------------------
def plot_season_trend(impact_file):
    """
    Season-long line chart: pre-2026 baseline prediction vs 2026 actuals.
    Circuits ordered by RoundNumber (chronological).
    """
    print("  Generating Plot 4: Season Trend...")
    if not os.path.exists(impact_file):
        print("    Warning: 2026 impact file missing, skipping.")
        return

    df = pd.read_parquet(impact_file)
    if df.empty:
        print("    Warning: 2026 impact data is empty, skipping.")
        return

    if 'RoundNumber' in df.columns:
        round_order = (df.groupby('Circuit')['RoundNumber']
                         .min()
                         .sort_values()
                         .index
                         .tolist())
    else:
        round_order = df['Circuit'].unique().tolist()

    agg = (df.groupby('Circuit')[['actual_laptime', 'predicted_laptime']]
             .mean()
             .reindex(round_order)
             .reset_index())

    plt.figure(figsize=(12, 6))
    plt.plot(
        agg['Circuit'], agg['predicted_laptime'],
        marker='o', linestyle='dashed', linewidth=2,
        color=F1_GREY, label='Predicted (Pre-2026 Baseline)'
    )
    plt.plot(
        agg['Circuit'], agg['actual_laptime'],
        marker='D', linewidth=2.5,
        color=F1_RED, label='2026 Actual'
    )
    plt.title(
        'Average Lap Time: Pre-2026 Baseline vs 2026 Actuals',
        fontsize=16, fontweight='bold', pad=20
    )
    plt.ylabel('Average Lap Time (s)', fontsize=12)
    plt.xlabel('Circuit',              fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(fontsize=12)
    save('season_trend.png')


# -- Entry point -----------------------------------------------------------
def main():
    print("=" * 60)
    print("  F1 RESULTS VISUALISATION")
    print("=" * 60)

    model_path = os.path.join(MODELS_DIR, "xgb_lap_predictor.pkl")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        plot_feature_importance(model)
    else:
        print(f"  Warning: Model not found at {model_path}. Skipping Feature Importance plot.")

    test_preds_file = os.path.join(DATA_DIR, "test_predictions.parquet")
    plot_predicted_vs_actual(test_preds_file)

    impact_file = os.path.join(DATA_DIR, "2026_impact.parquet")
    plot_regulation_impact(impact_file)
    plot_season_trend(impact_file)

    print("  Done: Visualisations saved to outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
    