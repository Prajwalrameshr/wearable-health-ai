import pandas as pd


def summarize_baseline(df: pd.DataFrame) -> dict:
    """Generate a compact health baseline from wearable readings."""
    return {
        "avg_heart_rate": round(df["heart_rate"].mean(), 2),
        "avg_daily_steps": round(df["steps"].mean(), 2),
        "avg_sleep_hours": round(df["sleep_hours"].mean(), 2),
        "avg_spo2": round(df["spo2"].mean(), 2),
        "avg_stress_level": round(df["stress_level"].mean(), 2),
    }
