import unittest
import pandas as pd

from src.risk_scoring import (
    calculate_risk_score,
    detect_early_warning_signal,
    evaluate_clinical_persistence,
    compute_health_risk_intelligence,
)

class TestRiskScoring(unittest.TestCase):
    def setUp(self):
        self.sample_row = pd.Series({
            "user_id": "test_user",
            "date": "2026-01-01",
            "severity_score": 2.5,
            "hrv_dev": -1.2,
            "hr_dev": 1.5,
            "sleep_dev": -0.8,
            "sleep_duration_hours": 6.0,
            "steps": 4000.0,
            "screen_time_min": 250.0,
            "caffeine_mg": 200.0,
            "hrv_dev_slope_7d": -0.08,
            "sleep_dev_slope_7d": -0.06,
            "spo2_avg_pct": 95.0,
            "hr_dev_z": 2.2,
            "hrv_dev_z": -1.8,
        })

    def test_early_warning_trigger(self):
        result = detect_early_warning_signal(self.sample_row)
        self.assertTrue(result["triggered"])
        self.assertEqual(result["trend"], "Deteriorating")

    def test_health_risk_intelligence(self):
        intelligence = compute_health_risk_intelligence(self.sample_row)
        self.assertIn("cardiovascular_strain", intelligence)
        self.assertIn("sleep_deficit", intelligence)
        self.assertGreater(intelligence["cardiovascular_strain"]["score"], 0)

    def test_calculate_risk_score(self):
        risk = calculate_risk_score(self.sample_row, state="Strain", state_duration=3)
        self.assertGreater(risk["score"], 0)
        self.assertIn(risk["level"], ["Low", "Moderate", "High"])
        self.assertEqual(risk["environment_impact"], None)

if __name__ == "__main__":
    unittest.main()
