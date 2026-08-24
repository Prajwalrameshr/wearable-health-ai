"""
Common Feature Mapping and Harmonization Schema definitions.
Harmonizes synthetic dataset and real-world Fitbit dataset onto a unified Primary Core Feature Space.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

# Primary Core Feature Space (6 Signals)
PRIMARY_CORE_FEATURES = [
    "resting_hr_bpm",
    "avg_hr_day_bpm",
    "steps",
    "distance_km",
    "calories_kcal",
    "sleep_duration_hours",
]

# Secondary / Extended Physiological Features (Authentic observed days only)
SECONDARY_PHYSIOLOGICAL_FEATURES = [
    "hrv_rmssd_ms",
    "spo2_avg_pct",
]

# Mapping dictionary from raw Fitbit dataset to Harmonized Primary Schema
FITBIT_COLUMN_MAP = {
    "id": "user_id",
    "date": "date",
    "resting_hr": "resting_hr_bpm",
    "bpm": "avg_hr_day_bpm",
    "steps": "steps",
    "distance": "distance_km",
    "calories": "calories_kcal",
    "sleep_duration": "sleep_duration_hours",
    "rmssd": "hrv_rmssd_ms",
    "spo2": "spo2_avg_pct",
    "minutesToFallAsleep": "sleep_latency_min",
    "minutesAwake": "minutes_awake",
    "minutesAfterWakeup": "minutes_after_wakeup",
    "sleep_efficiency": "sleep_efficiency",
    "age": "age",
    "gender": "gender",
    "bmi": "bmi",
    "stress_score": "stress_score",
}

def map_raw_fitbit(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Map raw Fitbit DataFrame columns to standard schema and apply explicit unit conversions.
    """
    df = df_raw.copy()
    
    # 1. Identifier and Timestamp
    user_col = "id" if "id" in df.columns else "user_id"
    df["user_id"] = df[user_col].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["user_id", "date"]).drop_duplicates(subset=["user_id", "date"], keep="last").reset_index(drop=True)
    
    mapped = pd.DataFrame()
    mapped["user_id"] = df["user_id"]
    mapped["date"] = df["date"]
    
    # 2. Core Signals & Unit Conversions
    mapped["resting_hr_bpm"] = pd.to_numeric(df.get("resting_hr", np.nan), errors="coerce")
    mapped["avg_hr_day_bpm"] = pd.to_numeric(df.get("bpm", np.nan), errors="coerce")
    mapped["steps"] = pd.to_numeric(df.get("steps", np.nan), errors="coerce")
    
    # Distance: meters -> kilometers
    raw_dist = pd.to_numeric(df.get("distance", np.nan), errors="coerce")
    mapped["distance_km"] = raw_dist / 1000.0
    
    mapped["calories_kcal"] = pd.to_numeric(df.get("calories", np.nan), errors="coerce")
    
    # Sleep duration: milliseconds -> hours
    raw_sleep = pd.to_numeric(df.get("sleep_duration", np.nan), errors="coerce")
    mapped["sleep_duration_hours"] = raw_sleep / (1000.0 * 3600.0)
    
    # Optional / Secondary signals
    mapped["hrv_rmssd_ms"] = pd.to_numeric(df.get("rmssd", np.nan), errors="coerce")
    mapped["spo2_avg_pct"] = pd.to_numeric(df.get("spo2", np.nan), errors="coerce")
    mapped["sleep_efficiency"] = pd.to_numeric(df.get("sleep_efficiency", np.nan), errors="coerce")
    mapped["sleep_latency_min"] = pd.to_numeric(df.get("minutesToFallAsleep", np.nan), errors="coerce")
    
    awake = pd.to_numeric(df.get("minutesAwake", np.nan), errors="coerce").fillna(0)
    after = pd.to_numeric(df.get("minutesAfterWakeup", np.nan), errors="coerce").fillna(0)
    mapped["wake_after_sleep_onset_min"] = awake + after
    
    # Metadata
    mapped["age"] = df.get("age", "unknown").fillna("unknown").astype(str)
    mapped["gender"] = df.get("gender", "unknown").fillna("unknown").astype(str).str.lower()
    mapped["bmi"] = df.get("bmi", "unknown").fillna("unknown").astype(str)
    mapped["stress_score"] = pd.to_numeric(df.get("stress_score", np.nan), errors="coerce")
    
    return mapped
