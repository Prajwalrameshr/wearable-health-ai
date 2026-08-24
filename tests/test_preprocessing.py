from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    CLUSTER_FEATURE_COLUMNS,
    chronological_split,
    compute_baseline,
    compute_deviation,
    compute_severity,
    compute_temporal_features,
    fit_scaler,
    handle_missing,
    remove_outliers,
    subject_independent_split,
    transform_features,
)


@pytest.fixture
def sample_user_df() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=20, freq="D")
    df = pd.DataFrame({
        "user_id": ["user_1"] * 20,
        "date": dates,
        "resting_hr_bpm": [60.0, 62.0, 61.0, 63.0, 62.0, 64.0, 65.0, 66.0, 65.0, 67.0, 68.0, 70.0, 72.0, 71.0, 69.0, 68.0, 66.0, 64.0, 63.0, 62.0],
        "hrv_rmssd_ms": [50.0, 48.0, 49.0, 47.0, 46.0, 45.0, 44.0, 42.0, 41.0, 40.0, 38.0, 36.0, 35.0, 37.0, 39.0, 41.0, 43.0, 45.0, 47.0, 49.0],
        "spo2_avg_pct": [98.0] * 20,
        "steps": [8000.0] * 20,
        "sleep_duration_hours": [7.5] * 20,
        "sbp_mmHg": [120.0] * 20,
        "dbp_mmHg": [80.0] * 20,
        "caffeine_mg": [100.0] * 20,
        "screen_time_min": [120.0] * 20,
        "workout_type": ["none"] * 20,
        "mood": ["good"] * 20,
        "gender": ["male"] * 20,
        "region": "US",
        "device_model": "Watch",
    })
    return df


def test_handle_missing_causal_no_bfill(sample_user_df: pd.DataFrame) -> None:
    """Verify missing value imputation does NOT use future observations (no bfill)."""
    df_missing = sample_user_df.copy()
    # Introduce NaN at index 5
    df_missing.loc[5, "resting_hr_bpm"] = np.nan

    filled = handle_missing(df_missing)
    # Forward fill should copy index 4 value (62.0) into index 5
    assert filled.loc[5, "resting_hr_bpm"] == df_missing.loc[4, "resting_hr_bpm"]


def test_remove_outliers_causal(sample_user_df: pd.DataFrame) -> None:
    """Verify outlier capping uses expanding causal bounds up to time t."""
    df_outlier = sample_user_df.copy()
    # Inject outlier at time t=15
    df_outlier.loc[15, "resting_hr_bpm"] = 250.0

    capped = remove_outliers(df_outlier)
    assert capped.loc[15, "resting_hr_bpm"] < 250.0


def test_compute_baseline_warmup(sample_user_df: pd.DataFrame) -> None:
    """Verify 7-day rolling baseline marks Days 1-6 as warmup mode (baseline_valid=False)."""
    df_base = compute_baseline(sample_user_df, warmup_period=7)
    assert "baseline_valid" in df_base.columns
    assert not df_base.loc[0, "baseline_valid"]
    assert not df_base.loc[5, "baseline_valid"]
    assert df_base.loc[6, "baseline_valid"]  # Day 7 (index 6) is valid


def test_fit_transform_scaler_split(sample_user_df: pd.DataFrame) -> None:
    """Verify scaler is fitted on train set and transforms test set without refitting."""
    df_feat = compute_severity(compute_temporal_features(compute_deviation(compute_baseline(sample_user_df))))
    train_df = df_feat.iloc[:12]
    test_df = df_feat.iloc[12:]

    scaler, cols = fit_scaler(train_df, CLUSTER_FEATURE_COLUMNS)
    scaled_test = transform_features(test_df, scaler, cols)

    assert scaled_test.shape == (8, len(cols))
    assert not scaled_test.isna().any().any()


def test_chronological_split(sample_user_df: pd.DataFrame) -> None:
    """Verify chronological split splits time series cleanly without shuffling."""
    train_df, val_df, test_df = chronological_split(sample_user_df, train_pct=0.70, val_pct=0.15, test_pct=0.15)
    assert len(train_df) == 14
    assert len(val_df) == 3
    assert len(test_df) == 3
    # Check strict date ordering across splits
    assert train_df["date"].max() < val_df["date"].min()
    assert val_df["date"].max() < test_df["date"].min()


def test_subject_independent_split_user_disjointness(sample_user_df: pd.DataFrame) -> None:
    """Verify subject-independent 5-fold grouped split has zero user overlap between train and test splits."""
    rows = []
    for uid in [f"user_{i}" for i in range(10)]:
        for d in pd.date_range("2026-01-01", periods=10, freq="D"):
            rows.append({"user_id": uid, "date": d, "resting_hr_bpm": 60.0, "hrv_rmssd_ms": 50.0, "sleep_duration_hours": 7.0, "steps": 5000.0})
    multi_user_df = pd.DataFrame(rows)

    splits = subject_independent_split(multi_user_df, n_splits=5, seed=42)
    for fold_idx, (sub_train, sub_test) in enumerate(splits):
        train_users = set(sub_train["user_id"].unique())
        test_users = set(sub_test["user_id"].unique())
        assert len(train_users.intersection(test_users)) == 0, f"Fold {fold_idx+1} has overlapping users!"


def test_causal_initial_imputation_no_future_leakage() -> None:
    """Verify modifying future test data does not alter past imputed feature values."""
    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    df1 = pd.DataFrame({
        "user_id": ["u1"] * 10,
        "date": dates,
        "resting_hr_bpm": [np.nan, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0],
        "hrv_rmssd_ms": [50.0] * 10,
        "spo2_avg_pct": [98.0] * 10,
        "steps": [5000.0] * 10,
        "sleep_duration_hours": [7.0] * 10,
    })
    # Modify future value at t=9
    df2 = df1.copy()
    df2.loc[9, "resting_hr_bpm"] = 999.0

    filled1 = handle_missing(df1)
    filled2 = handle_missing(df2)

    # Initial imputed value at t=0 must be identical regardless of future t=9 value
    assert filled1.loc[0, "resting_hr_bpm"] == filled2.loc[0, "resting_hr_bpm"]


def test_environmental_context_absent_from_pipeline() -> None:
    """Verify active preprocessing schema contains zero environmental, weather, AQI, or PM2.5 columns."""
    from src.preprocessing import CLUSTER_FEATURE_COLUMNS, DAILY_REQUIRED_COLUMNS
    env_terms = ["aqi", "pm2_5", "pm10", "weather", "openweather", "city", "temperature", "humidity"]
    for col in CLUSTER_FEATURE_COLUMNS + DAILY_REQUIRED_COLUMNS:
        for term in env_terms:
            assert term not in col.lower(), f"Environmental term '{term}' found in column '{col}'!"

