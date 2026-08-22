import unittest
from pathlib import Path
import pandas as pd

from run_inference import load_model_outputs, compute_state
from src.preprocessing import preprocess_for_modeling

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.csv_path = Path("data") / "sample_user.csv"
        if not self.csv_path.exists():
            self.csv_path = Path("data") / "wearables_health_6mo_daily.csv"

    def test_end_to_end_inference(self):
        outputs = preprocess_for_modeling(source=self.csv_path)
        feature_df = outputs["feature_df"].copy()
        feature_df, cluster_out, hmm_bundle = load_model_outputs(feature_df, model_type="gmm")
        final_df, analysis = compute_state("gmm", hmm_bundle["labeled_df"], hmm_bundle, {}, cluster_out)

        self.assertIsNotNone(analysis)
        self.assertIn(analysis["state"], ["Recovery", "Baseline", "Strain", "Unknown"])
        self.assertGreater(analysis["risk"]["score"], 0)

if __name__ == "__main__":
    unittest.main()
