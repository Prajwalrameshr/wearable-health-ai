from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from run_inference import run_pipeline

ROOT_DIR = Path(__file__).resolve().parents[1]
SAMPLE_CSV = ROOT_DIR / "data" / "wearables_health_6mo_daily.csv"


def test_full_pipeline_execution(tmp_path) -> None:
    """Verify end-to-end inference pipeline execution and UI compatibility contract."""
    output_file = tmp_path / "test_out.csv"
    res = run_pipeline(csv_path=str(SAMPLE_CSV), output_path=str(output_file), model_type="gmm")

    assert output_file.exists()
    assert "analysis_result" in res
    assert "final_results_df" in res
    assert "feature_df" in res

    analysis = res["analysis_result"]
    assert "state" in analysis
    assert "confidence" in analysis
    assert "trend" in analysis
    assert "risk" in analysis
    assert "recommendations" in analysis
    assert "state_probabilities" in analysis
    assert "hmm_state_probabilities" in analysis
    assert "stress_level" in analysis

    final_df = res["final_results_df"]
    assert "risk_score" in final_df.columns
    assert "risk_level" in final_df.columns
    assert "stress_level_standardized" in final_df.columns
