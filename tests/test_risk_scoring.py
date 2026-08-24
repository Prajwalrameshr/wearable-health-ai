from __future__ import annotations

import pandas as pd
import pytest

from src.risk_scoring import calculate_risk_score, compute_health_risk_intelligence, evaluate_clinical_persistence


@pytest.fixture
def sample_features() -> pd.Series:
    return pd.Series({
        "user_id": "test_u",
        "date": "2026-01-10",
        "resting_hr_bpm": 65.0,
        "hrv_rmssd_ms": 50.0,
        "spo2_avg_pct": 98.0,
        "steps": 7500.0,
        "sleep_duration_hours": 7.5,
        "severity_score": 1.2,
        "hr_dev": 2.0,
        "hrv_dev": -3.0,
        "sleep_dev": 0.5,
        "hr_dev_z": 0.4,
        "hrv_dev_z": -0.5,
    })


def test_calculate_risk_score_no_environment(sample_features: pd.Series) -> None:
    """Verify physiological risk index is calculated deterministically without environmental inputs."""
    res = calculate_risk_score(sample_features, state="Baseline", state_confidence=0.85, state_duration=2)

    assert "score" in res
    assert 0.0 <= res["score"] <= 100.0
    assert "level" in res
    assert res["level"] in ["Low", "Moderate", "High"]
    assert "environment_impact" not in res or res["environment_impact"] is None
    assert "multi_risk" in res
    assert len(res["multi_risk"]) == 10


def test_compute_health_risk_intelligence(sample_features: pd.Series) -> None:
    """Verify 10 sub-dimension physiological risk scores are computed purely from wearable signals."""
    intel = compute_health_risk_intelligence(sample_features)
    assert len(intel) == 10
    assert "cardiovascular_strain" in intel
    assert "sleep_deficit" in intel
    assert "autonomic_imbalance" in intel

    for k, v in intel.items():
        assert "score" in v
        assert "level" in v
        assert 0.0 <= v["score"] <= 100.0


def test_evaluate_clinical_persistence(sample_features: pd.Series) -> None:
    """Verify multi-day physiological strain persistence evaluation operates cleanly."""
    res = evaluate_clinical_persistence(sample_features)
    assert "advisory_level" in res
    assert res["advisory_level"] in ["Normal", "Attention Needed", "Strain Advisory"]
