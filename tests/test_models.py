import unittest
import numpy as np
import pandas as pd
from pathlib import Path

from src.gmm import run_gmm_pipeline, find_optimal_k, save_gmm_model, load_gmm_model
from src.kmeans import run_kmeans_pipeline, save_kmeans_model, load_kmeans_model
from src.hmm_model import run_hmm_pipeline, save_hmm_model, load_hmm_model

class TestModels(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 100
        self.feature_df = pd.DataFrame({
            "user_id": ["user_1"] * n,
            "date": pd.date_range("2026-01-01", periods=n, freq="D"),
            "hr_dev": np.random.normal(0, 1, n),
            "hrv_dev": np.random.normal(0, 1, n),
            "sleep_dev": np.random.normal(0, 1, n),
            "severity_score": np.random.uniform(0.5, 4.0, n),
            "resting_hr_bpm": np.random.normal(65, 5, n),
            "hrv_rmssd_ms": np.random.normal(45, 10, n),
            "sleep_duration_hours": np.random.normal(7.5, 1.0, n),
        })

    def test_gmm_pipeline(self):
        out = run_gmm_pipeline(self.feature_df, feature_columns=["hr_dev", "hrv_dev", "sleep_dev", "severity_score"])
        self.assertIn("labeled_df", out)
        self.assertIn("gmm_state_label", out["labeled_df"].columns)
        self.assertIn(out["selected_k"], [2, 3, 4, 5])

    def test_kmeans_pipeline(self):
        out = run_kmeans_pipeline(self.feature_df, feature_columns=["hr_dev", "hrv_dev", "sleep_dev", "severity_score"])
        self.assertIn("labeled_df", out)
        self.assertIn("kmeans_cluster", out["labeled_df"].columns)

    def test_gmm_model_persistence(self):
        out = run_gmm_pipeline(self.feature_df, feature_columns=["hr_dev", "hrv_dev", "sleep_dev", "severity_score"])
        model = out["model"]
        temp_path = Path("models") / "test_gmm.pkl"
        save_gmm_model(model, temp_path)
        self.assertTrue(temp_path.exists())

        loaded_model = load_gmm_model(temp_path)
        self.assertEqual(loaded_model.n_components, model.n_components)
        temp_path.unlink(missing_ok=True)

if __name__ == "__main__":
    unittest.main()
