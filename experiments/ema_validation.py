"""
EMA Ground-Truth Label Validation Script
Evaluates discovered latent physiological states (Recovery, Baseline, Strain)
against authentic Ecological Momentary Assessment (EMA) self-report survey labels.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Define paths
REPO_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_DIR / "data" / "daily_fitbit_sema_df_unprocessed.csv"
OUTPUT_DIR = REPO_DIR / "experiments" / "results" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_ema_validation():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Rename columns to standard schema if needed
    rename_dict = {
        "id": "user_id",
        "resting_hr": "resting_hr_bpm",
        "bpm": "avg_hr_day_bpm",
        "sleep_duration": "sleep_duration_hours",
    }
    df = df.rename(columns=rename_dict)
    
    # Unit conversions if needed
    if "sleep_duration_hours" in df.columns and df["sleep_duration_hours"].median() > 100:
        df["sleep_duration_hours"] = df["sleep_duration_hours"] / 3600000.0
    if "distance" in df.columns and df["distance"].median() > 100:
        df["distance_km"] = df["distance"] / 1000.0
    elif "distance" in df.columns:
        df["distance_km"] = df["distance"]
        
    if "calories" in df.columns:
        df["calories_kcal"] = df["calories"]
        
    core_cols = ["resting_hr_bpm", "avg_hr_day_bpm", "steps", "distance_km", "calories_kcal", "sleep_duration_hours"]
    ema_cols = ["ALERT", "HAPPY", "NEUTRAL", "RESTED/RELAXED", "SAD", "TENSE/ANXIOUS", "TIRED"]
    
    # Keep rows with valid core columns
    clean_df = df.sort_values(["user_id", "date"]).copy()
    for col in core_cols:
        clean_df[col] = clean_df.groupby("user_id")[col].transform(lambda s: s.ffill(limit=2))
        
    clean_df = clean_df.dropna(subset=core_cols).reset_index(drop=True)
    
    # Compute 7-day causal rolling baselines and z-scores per user
    z_cols = []
    for col in core_cols:
        roll_mean = clean_df.groupby("user_id")[col].transform(lambda s: s.shift(1).rolling(7, min_periods=7).mean())
        roll_std = clean_df.groupby("user_id")[col].transform(lambda s: s.shift(1).rolling(7, min_periods=7).std())
        z_col = f"{col}_z"
        clean_df[z_col] = (clean_df[col] - roll_mean) / (roll_std + 1e-6)
        z_cols.append(z_col)
        
    # Valid post-warmup rows
    valid_df = clean_df.dropna(subset=z_cols).copy()
    valid_df["severity_score"] = valid_df[z_cols].abs().sum(axis=1)
    
    # Fit GMM K=3
    X = valid_df[z_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    gmm = GaussianMixture(n_components=3, covariance_type="diag", random_state=42)
    raw_clusters = gmm.fit_predict(X_scaled)
    valid_df["raw_cluster"] = raw_clusters
    
    # Map clusters by severity: lowest=Recovery (0), mid=Baseline (1), highest=Strain (2)
    sev_means = valid_df.groupby("raw_cluster")["severity_score"].median()
    cluster_order = sev_means.sort_values().index.tolist()
    label_map = {cluster_order[0]: "Recovery", cluster_order[1]: "Baseline", cluster_order[2]: "Strain"}
    valid_df["latent_state"] = valid_df["raw_cluster"].map(label_map)
    
    print("\n--- Latent State Occupancy ---")
    print(valid_df["latent_state"].value_counts(normalize=True))
    
    # Perform statistical validation across EMA columns
    results = []
    print("\n--- EMA Ground-Truth Label Validation Results ---")
    for ema in ema_cols:
        if ema not in valid_df.columns:
            continue
        sub = valid_df.dropna(subset=[ema])
        if len(sub) < 30:
            print(f"Skipping {ema} (insufficient data: N={len(sub)})")
            continue
            
        g_rec = sub[sub["latent_state"] == "Recovery"][ema].values
        g_base = sub[sub["latent_state"] == "Baseline"][ema].values
        g_str = sub[sub["latent_state"] == "Strain"][ema].values
        
        # Kruskal-Wallis non-parametric ANOVA
        h_stat, p_val = stats.kruskal(g_rec, g_base, g_str)
        
        rec_mean, rec_std = np.mean(g_rec), np.std(g_rec)
        base_mean, base_std = np.mean(g_base), np.std(g_base)
        str_mean, str_std = np.mean(g_str), np.std(g_str)
        
        results.append({
            "EMA Label": ema,
            "N": len(sub),
            "Recovery (Mean ± SD)": f"{rec_mean:.2f} ± {rec_std:.2f}",
            "Baseline (Mean ± SD)": f"{base_mean:.2f} ± {base_std:.2f}",
            "Strain (Mean ± SD)": f"{str_mean:.2f} ± {str_std:.2f}",
            "Kruskal-Wallis H": round(h_stat, 4),
            "p-value": f"{p_val:.4e}" if p_val < 0.001 else f"{p_val:.4f}",
            "Significant (p < 0.05)": "YES" if p_val < 0.05 else "NO",
        })
        
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    
    res_df.to_csv(OUTPUT_DIR / "ema_validation_summary.csv", index=False)
    with open(OUTPUT_DIR / "ema_validation_summary.md", "w", encoding="utf-8") as f:
        f.write("# EMA Ground-Truth Label Validation Summary\n\n")
        f.write(res_df.to_markdown(index=False))
        
    print(f"\nSaved validation results to {OUTPUT_DIR / 'ema_validation_summary.csv'}")
    return res_df

if __name__ == "__main__":
    run_ema_validation()
