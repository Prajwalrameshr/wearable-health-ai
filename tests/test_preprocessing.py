import unittest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from src.preprocessing import (
    chronological_split,
    save_scaler,
    load_scaler,
    scale_features,
    preprocess_for_modeling,
)

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2026-01-01", periods=20, freq="D")
        self.df = pd.DataFrame({
            "user_id": ["user_1"] * 20,
            "date": dates,
            "resting_hr_bpm": np.random.normal(65, 5, 20),
            "hrv_rmssd_ms": np.random.normal(45, 10, 20),
            "sleep_duration_hours": np.random.normal(7.5, 1.0, 20),
            "steps": np.random.normal(8000, 1500, 20),
            "spo2_avg_pct": np.random.normal(98, 1, 20),
            "caffeine_mg": [100] * 20,
            "screen_time_min": [120] * 20,
            "age": [30] * 20,
            "gender": ["male"] * 20,
        })

    def test_chronological_split(self):
        train_df, val_df, test_df = chronological_split(self.df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
        self.assertEqual(len(train_df), 14)
        self.assertEqual(len(val_df), 3)
        self.assertEqual(len(test_df), 3)
        self.assertTrue(train_df["date"].max() <= val_df["date"].min())
        self.assertTrue(val_df["date"].max() <= test_df["date"].min())

    def test_scaler_persistence(self):
        scaler = StandardScaler()
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scaler.fit(X)
        temp_path = Path("models") / "test_scaler.pkl"
        save_scaler(scaler, temp_path)
        self.assertTrue(temp_path.exists())

        loaded_scaler = load_scaler(temp_path)
        X_trans = loaded_scaler.transform([[1.0, 2.0]])
        self.assertEqual(X_trans.shape, (1, 2))
        temp_path.unlink(missing_ok=True)

    def test_scale_features(self):
        sample_df = pd.DataFrame({
            "hr_dev": [0.1, -0.2, 0.5],
            "hrv_dev": [-0.3, 0.1, 0.4],
            "sleep_dev": [0.0, -0.5, 0.2],
            "severity_score": [1.0, 2.0, 1.5],
        })
        scaled_df, scaler, cols = scale_features(sample_df, feature_columns=["hr_dev", "hrv_dev"])
        self.assertIn("hr_dev", scaled_df.columns)
        self.assertEqual(len(cols), 2)
        self.assertIsNotNone(scaler)

if __name__ == "__main__":
    unittest.main()
