"""
Step 5 — Visualisation
Generate 4 publication-quality plots.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Set global seaborn styling for an 'F1-inspired' look
plt.style.use('dark_background')
sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#111111", "figure.facecolor": "#111111", 
                                    "grid.color": "#333333", "text.color": "white", 
                                    "axes.labelcolor": "white", "xtick.color": "white", 
                                    "ytick.color": "white"})


def plot_feature_importance(model, feature_names):
    """Plot 1 — XGBoost feature importance horizontal bar chart"""
    print("  Generating Plot 1: Feature Importance...")
    importances = model.feature_importances_
    
    df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    df = df.sort_values(by='Importance', ascending=True)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=df, color='#E10600') # F1 Red
    
    plt.title('XGBoost Feature Importance', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Relative Importance', fontsize=12)
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'feature_importance.png'), dpi=300)
    plt.close()


def plot_predicted_vs_actual(test_preds_file):
    """Plot 2 — Predicted vs actual scatter plot on 2025 test set"""
    print("  Generating Plot 2: Predicted vs Actual...")
    if not os.path.exists(test_preds_file):
        print("    ⚠ Test predictions file missing, skipping Plot 2.")
        return
        
    df = pd.read_parquet(test_preds_file)
    
    plt.figure(figsize=(10, 8))
    
    # Limit to top circuits if there are too many for the color palette
    if df['Circuit'].nunique() > 10:
        top_circuits = df['Circuit'].value_counts().nlargest(10).index
        df_plot = df[df['Circuit'].isin(top_circuits)]
    else:
        df_plot = df
        
    sns.scatterplot(x='actual_laptime', y='predicted_laptime', hue='Circuit', 
                    data=df_plot, alpha=0.6, palette='tab10', s=40)
    
    # Add y=x reference line
    min_val = min(df_plot['actual_laptime'].min(), df_plot['predicted_laptime'].min())
    max_val = max(df_plot['actual_laptime'].max(), df_plot['predicted_laptime'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'w--', alpha=0.8, linewidth=2)
    
    plt.title('Predicted vs Actual Lap Times (2025 Test Set)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Actual Lap Time (s)', fontsize=12)
    plt.ylabel('Predicted Lap Time (s)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'predicted_vs_actual.png'), dpi=300)
    plt.close()


def plot_regulation_impact(impact_2026_file):
    """Plot 3 — Regulation impact delta per circuit in 2026, bar chart with error bars"""
    print("  Generating Plot 3: 2026 Regulation Impact...")
    if not os.path.exists(impact_2026_file):
        print("    ⚠ 2026 impact file missing, skipping Plot 3.")
        return
        
    df = pd.read_parquet(impact_2026_file)
    if df.empty:
        print("    ⚠ 2026 impact data is empty, skipping Plot 3.")
        return
        
    plt.figure(figsize=(12, 7))
    
    # Group by circuit to plot mean + std error
    circuit_stats = df.groupby('Circuit')['delta'].agg(['mean', 'std', 'count']).reset_index()
    # Approx 95% CI: 1.96 * std / sqrt(n)
    circuit_stats['err'] = 1.96 * circuit_stats['std'] / np.sqrt(circuit_stats['count'])
    
    # Sort by mean impact
    circuit_stats = circuit_stats.sort_values(by='mean', ascending=False)
    
    # Define colors: positive delta (slower) = Red, negative (faster) = Green
    colors = ['#E10600' if x > 0 else '#00D2BE' for x in circuit_stats['mean']]
    
    bars = plt.bar(circuit_stats['Circuit'], circuit_stats['mean'], 
                   yerr=circuit_stats['err'], capsize=5, color=colors, alpha=0.8, ecolor='white')
    
    plt.axhline(0, color='white', linewidth=1)
    
    plt.title('2026 Regulation Impact (Actual - Predicted)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Delta (seconds) [Positive = Slower]', fontsize=12)
    plt.xlabel('Circuit', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'regulation_impact.png'), dpi=300)
    plt.close()


def plot_season_trend(impact_2026_file):
    """Plot 4 — Season-long lap time trend showing predicted vs 2026 actuals"""
    print("  Generating Plot 4: Season Trend...")
    if not os.path.exists(impact_2026_file):
        print("    ⚠ 2026 impact file missing, skipping Plot 4.")
        return
        
    df = pd.read_parquet(impact_2026_file)
    if df.empty:
        print("    ⚠ 2026 impact data is empty, skipping Plot 4.")
        return
    
    # Order circuits logically - assuming order of appearance in dataset matches temporal order
    circuits = df['Circuit'].unique()
    
    plt.figure(figsize=(12, 6))
    
    agg_df = df.groupby('Circuit')[['actual_laptime', 'predicted_laptime']].mean().reset_index()
    # Reorder according to the chronological sequence of appearance
    agg_df['circuit_idx'] = agg_df['Circuit'].map({circuit: i for i, circuit in enumerate(circuits)})
    agg_df = agg_df.sort_values('circuit_idx')
    
    plt.plot(agg_df['Circuit'], agg_df['predicted_laptime'], marker='o', 
             linestyle='dashed', linewidth=2, color='#A6A6A6', label='Predicted (Pre-2026 Baseline)')
             
    plt.plot(agg_df['Circuit'], agg_df['actual_laptime'], marker='D', 
             linewidth=2.5, color='#E10600', label='2026 Actual')
             
    plt.title('Average Lap Time Trend: Pre-2026 Baseline vs 2026 Actuals', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Average Lap Time (s)', fontsize=12)
    plt.xlabel('Circuit', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'season_trend.png'), dpi=300)
    plt.close()


def main():
    print("=" * 60)
    print("  F1 RESULTS VISUALISATION")
    print("=" * 60)
    
    model_path = os.path.join(MODELS_DIR, "xgb_lap_predictor.pkl")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        feature_cols = [
            'compound_encoded', 'circuit_encoded', 'TyreLife', 'fuel_lap_number',
            'AirTemp', 'TrackTemp', 'Year',
            'mean_speed', 'max_speed', 'mean_throttle', 'pct_full_throttle',
            'mean_brake', 'pct_braking', 'mean_rpm', 'drs_active'
        ]
        plot_feature_importance(model, feature_cols)
    else:
        print(f"  ⚠ Model not found at {model_path}. Skipping Feature Importance plot.")

    test_preds_file = os.path.join(DATA_DIR, "test_predictions.parquet")
    plot_predicted_vs_actual(test_preds_file)
    
    impact_file = os.path.join(DATA_DIR, "2026_impact.parquet")
    plot_regulation_impact(impact_file)
    plot_season_trend(impact_file)

    print("  ✓ Visualisations saved to outputs/")
    print("=" * 60)

if __name__ == "__main__":
    main()
