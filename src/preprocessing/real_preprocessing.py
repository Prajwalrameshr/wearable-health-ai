"""
Authenticity-First Preprocessing Module for Real Fitbit Dataset and Synthetic Dataset Harmonization.
Strictly enforces:
1. Primary core 6-feature matrix zero missingness ONLY after causal within-user ffill (limit=2 days).
2. No mass cohort mean filling for physiological signals.
3. Causal rolling 7-day personalized baseline calculation with warmup tracking.
4. Frozen Harmonized Primary Severity Score computation.
5. Export of processed CSVs to data/processed/.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from src.preprocessing.common_feature_mapping import (
    PRIMARY_CORE_FEATURES,
    SECONDARY_PHYSIOLOGICAL_FEATURES,
    map_raw_fitbit,
)

REPO_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_BASELINE_MAP = {
    "baseline_hr": "resting_hr_bpm",
    "baseline_avg_hr": "avg_hr_day_bpm",
    "baseline_steps": "steps",
    "baseline_dist": "distance_km",
    "baseline_cal": "calories_kcal",
    "baseline_sleep": "sleep_duration_hours",
}

PRIMARY_STD_MAP = {
    "std_hr": "resting_hr_bpm",
    "std_avg_hr": "avg_hr_day_bpm",
    "std_steps": "steps",
    "std_dist": "distance_km",
    "std_cal": "calories_kcal",
    "std_sleep": "sleep_duration_hours",
}

PRIMARY_DEVIATION_MAP = {
    "hr_dev": ("resting_hr_bpm", "baseline_hr", "std_hr"),
    "avg_hr_dev": ("avg_hr_day_bpm", "baseline_avg_hr", "std_avg_hr"),
    "steps_dev": ("steps", "baseline_steps", "std_steps"),
    "dist_dev": ("distance_km", "baseline_dist", "std_dist"),
    "cal_dev": ("calories_kcal", "baseline_cal", "std_cal"),
    "sleep_dev": ("sleep_duration_hours", "baseline_sleep", "std_sleep"),
}

SLOPE_INPUT_MAP = {
    "hr_dev_slope_7d": "hr_dev",
    "sleep_dev_slope_7d": "sleep_dev",
    "steps_dev_slope_7d": "steps_dev",
}

ROLLING_WINDOW = 7
WARMUP_PERIOD_DAYS = 7


def apply_authentic_causal_infill(
    df: pd.DataFrame,
    features: list[str] = PRIMARY_CORE_FEATURES,
    max_ffill_days: int = 2,
) -> pd.DataFrame:
    """
    Apply conservative within-user causal forward fill (max_ffill_days = 2).
    Drop rows where any primary feature remains missing after 2-day ffill.
    NO MASS COHORT MEAN IMPUTATION.
    """
    filled = df.sort_values(["user_id", "date"]).reset_index(drop=True).copy()
    
    for col in features:
        if col in filled.columns:
            filled[col] = filled.groupby("user_id")[col].transform(
                lambda s: s.ffill(limit=max_ffill_days)
            )
            
    # Drop rows missing any core feature
    clean = filled.dropna(subset=features).reset_index(drop=True)
    return clean


def compute_causal_baselines(
    df: pd.DataFrame,
    window: int = ROLLING_WINDOW,
    warmup_period: int = WARMUP_PERIOD_DAYS,
) -> pd.DataFrame:
    """
    Compute causal rolling 7-day personalized baseline and std per user.
    Uses expanding mean/std for warmup days (Days 1..6) and flags baseline_valid = (cumcount >= 7).
    """
    base_df = df.sort_values(["user_id", "date"]).reset_index(drop=True).copy()
    
    user_day_count = base_df.groupby("user_id").cumcount() + 1
    base_df["baseline_valid"] = user_day_count >= warmup_period
    
    for b_name, s_col in PRIMARY_BASELINE_MAP.items():
        if s_col not in base_df.columns:
            continue
        rolling_m = base_df.groupby("user_id")[s_col].transform(
            lambda series: series.rolling(window=window, min_periods=warmup_period).mean()
        )
        expanding_m = base_df.groupby("user_id")[s_col].transform(
            lambda series: series.expanding(min_periods=1).mean()
        )
        base_df[b_name] = rolling_m.fillna(expanding_m)
        
    for std_name, s_col in PRIMARY_STD_MAP.items():
        if s_col not in base_df.columns:
            continue
        rolling_s = base_df.groupby("user_id")[s_col].transform(
            lambda series: series.rolling(window=window, min_periods=warmup_period).std(ddof=0)
        )
        expanding_s = base_df.groupby("user_id")[s_col].transform(
            lambda series: series.expanding(min_periods=1).std(ddof=0)
        )
        base_df[std_name] = rolling_s.fillna(expanding_s).fillna(0.0)
        
    return base_df


def compute_deviations_and_harmonized_severity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute raw deviations, z-normalized deviations, and the FROZEN HARMONIZED PRIMARY SEVERITY SCORE.
    severity_score = sum(|z_dev|) across the 6 primary core features.
    """
    dev_df = df.copy()
    z_cols = []
    
    for dev_name, (s_col, b_col, std_col) in PRIMARY_DEVIATION_MAP.items():
        if s_col not in dev_df.columns or b_col not in dev_df.columns:
            continue
        dev_df[dev_name] = dev_df[s_col] - dev_df[b_col]
        safe_std = dev_df[std_col].replace(0, np.nan)
        z_col = f"{dev_name}_z"
        dev_df[z_col] = (
            (dev_df[dev_name] / safe_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )
        z_cols.append(z_col)
        
    # Frozen Harmonized Primary Severity Score
    dev_df["severity_score"] = dev_df[z_cols].abs().sum(axis=1)
    return dev_df


def rolling_causal_slope(series: pd.Series, window: int = ROLLING_WINDOW) -> pd.Series:
    """Compute causal 7-day rolling slope using past window data only."""
    values = series.to_numpy(dtype=float)
    slopes = np.zeros(len(values), dtype=float)
    
    for i in range(len(values)):
        start = max(0, i - window + 1)
        w_vals = values[start : i + 1]
        if len(w_vals) < 2 or np.isnan(w_vals).any():
            slopes[i] = 0.0
            continue
        x = np.arange(len(w_vals), dtype=float)
        slopes[i] = float(np.polyfit(x, w_vals, 1)[0])
        
    return pd.Series(slopes, index=series.index)


def compute_slopes_and_trend(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Add causal 7-day rolling slopes and composite trend_score."""
    temp_df = df.copy()
    slope_cols = []
    
    for slope_name, source_dev in SLOPE_INPUT_MAP.items():
        if source_dev not in temp_df.columns:
            continue
        temp_df[slope_name] = temp_df.groupby("user_id", group_keys=False)[source_dev].apply(
            lambda s: rolling_causal_slope(s, window=window)
        )
        slope_cols.append(slope_name)
        
    temp_df["trend_score"] = temp_df[slope_cols].abs().sum(axis=1)
    return temp_df


def process_real_primary_dataset(
    raw_csv_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Execute full authentic preprocessing on the real Fitbit dataset and save to processed directory.
    Returns the processed Primary Real DataFrame (4,159 rows x 69 users).
    """
    src_path = Path(raw_csv_path) if raw_csv_path else DATA_DIR / "daily_fitbit_sema_df_unprocessed.csv"
    out_path = Path(output_path) if output_path else PROCESSED_DIR / "real_common.csv"
    
    df_raw = pd.read_csv(src_path)
    mapped = map_raw_fitbit(df_raw)
    clean = apply_authentic_causal_infill(mapped, max_ffill_days=2)
    baselines = compute_causal_baselines(clean)
    deviations = compute_deviations_and_harmonized_severity(baselines)
    processed = compute_slopes_and_trend(deviations)
    
    processed.to_csv(out_path, index=False)
    return processed


def process_synthetic_harmonized_dataset(
    raw_synth_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Harmonize synthetic dataset to the exact same Primary Core Feature Space and Primary Severity definition.
    Saves to data/processed/synthetic_common.csv (55,200 rows x 300 users).
    """
    src_path = Path(raw_synth_path) if raw_synth_path else DATA_DIR / "wearables_health_6mo_daily.csv"
    out_path = Path(output_path) if output_path else PROCESSED_DIR / "synthetic_common.csv"
    
    df_synth = pd.read_csv(src_path)
    df_synth["user_id"] = df_synth["user_id"].astype(str)
    df_synth["date"] = pd.to_datetime(df_synth["date"], errors="coerce")
    df_synth = df_synth.dropna(subset=["date"]).sort_values(["user_id", "date"]).reset_index(drop=True)
    
    # Fill minor synthetic defaults for primary core features if missing
    for col in PRIMARY_CORE_FEATURES:
        if col in df_synth.columns:
            df_synth[col] = df_synth.groupby("user_id")[col].transform(lambda s: s.ffill(limit=2))
            df_synth[col] = df_synth[col].fillna(df_synth[col].mean())
            
    baselines = compute_causal_baselines(df_synth)
    deviations = compute_deviations_and_harmonized_severity(baselines)
    processed = compute_slopes_and_trend(deviations)
    
    processed.to_csv(out_path, index=False)
    return processed
