import pandas as pd


def create_feature_vector(df: pd.DataFrame) -> pd.DataFrame:
    """Create lightweight aggregate features for downstream models."""
    features = {
        "heart_rate_mean": df["heart_rate"].mean(),
        "heart_rate_std": df["heart_rate"].std(ddof=0),
        "steps_mean": df["steps"].mean(),
        "sleep_mean": df["sleep_hours"].mean(),
        "spo2_mean": df["spo2"].mean(),
        "stress_mean": df["stress_level"].mean(),
    }
    return pd.DataFrame([features]).fillna(0.0)
