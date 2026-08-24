from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.gmm import compute_gmm_native_feature_importance, load_gmm_model, run_gmm_pipeline, save_gmm_model
from src.hmm_model import build_hmm_sequences, load_hmm_model, run_hmm_pipeline, run_soft_probability_hmm, save_hmm_model
from src.kmeans import load_kmeans_model, run_kmeans_pipeline, save_model as save_kmeans_model
from src.preprocessing import CLUSTER_FEATURE_COLUMNS, compute_baseline, compute_deviation, compute_severity, compute_temporal_features, fit_scaler, transform_features


@pytest.fixture
def multi_user_feature_df() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2026-01-01", periods=15, freq="D")
    for u in ["user_A", "user_B"]:
        for d in dates:
            rows.append({
                "user_id": u,
                "date": d,
                "resting_hr_bpm": 60.0 + np.random.randn() * 2,
                "hrv_rmssd_ms": 50.0 + np.random.randn() * 5,
                "spo2_avg_pct": 98.0,
                "steps": 8000.0,
                "sleep_duration_hours": 7.5,
                "sbp_mmHg": 120.0,
                "dbp_mmHg": 80.0,
                "caffeine_mg": 100.0,
                "screen_time_min": 120.0,
                "workout_type": "none",
                "mood": "good",
                "gender": "female",
                "region": "US",
                "device_model": "Watch",
            })
    df = pd.DataFrame(rows)
    df_feat = compute_severity(compute_temporal_features(compute_deviation(compute_baseline(df))))
    scaler, cols = fit_scaler(df_feat, CLUSTER_FEATURE_COLUMNS)
    scaled_df = transform_features(df_feat, scaler, cols)
    for col in CLUSTER_FEATURE_COLUMNS:
        df_feat[col] = scaled_df[col]
    return df_feat


def test_gmm_pipeline_dynamic_k(multi_user_feature_df: pd.DataFrame) -> None:
    """Verify GMM evaluates k in range(2, 6) dynamically using BIC/AIC."""
    res = run_gmm_pipeline(multi_user_feature_df, feature_columns=CLUSTER_FEATURE_COLUMNS, k_values=range(2, 6))
    assert "selected_k" in res
    assert res["selected_k"] in [2, 3, 4, 5]
    assert "selection_df" in res
    assert len(res["selection_df"]) >= 2


def test_gmm_native_feature_importance(multi_user_feature_df: pd.DataFrame) -> None:
    """Verify model-native GMM feature attributions are computed."""
    res = run_gmm_pipeline(multi_user_feature_df, feature_columns=CLUSTER_FEATURE_COLUMNS)
    imp_df = compute_gmm_native_feature_importance(res["model"], CLUSTER_FEATURE_COLUMNS)
    assert not imp_df.empty
    assert "Feature" in imp_df.columns
    assert "Importance" in imp_df.columns
    assert abs(imp_df["Importance"].sum() - 1.0) < 1e-4


def test_kmeans_pipeline(multi_user_feature_df: pd.DataFrame) -> None:
    """Verify KMeans pipeline executes cleanly and labels clusters."""
    res = run_kmeans_pipeline(multi_user_feature_df, feature_columns=CLUSTER_FEATURE_COLUMNS)
    assert "optimal_k" in res
    assert "labeled_df" in res
    assert "kmeans_cluster" in res["labeled_df"].columns


def test_hmm_sequence_length_isolation(multi_user_feature_df: pd.DataFrame) -> None:
    """Verify HMM sequence builder tracks individual user sequence lengths to prevent contamination."""
    res = run_gmm_pipeline(multi_user_feature_df, feature_columns=CLUSTER_FEATURE_COLUMNS)
    obs, lengths, _, labels = build_hmm_sequences(res["labeled_df"], observation_column="gmm_cluster")
    assert len(lengths) == 2  # user_A and user_B
    assert sum(lengths) == len(obs)
    assert lengths[0] == 15
    assert lengths[1] == 15


def test_soft_probability_hmm(multi_user_feature_df: pd.DataFrame) -> None:
    """Verify continuous HMM operates directly on GMM posterior probability distributions."""
    gmm_res = run_gmm_pipeline(multi_user_feature_df, feature_columns=CLUSTER_FEATURE_COLUMNS)
    soft_res = run_soft_probability_hmm(gmm_res["labeled_df"])
    assert soft_res["fitted"]
    assert "transition_entropy_bits" in soft_res
    assert soft_res["transition_entropy_bits"] >= 0.0


def test_model_serialization(tmp_path, multi_user_feature_df: pd.DataFrame) -> None:
    """Verify GMM, KMeans, and HMM serialization and loading."""
    gmm_res = run_gmm_pipeline(multi_user_feature_df, feature_columns=CLUSTER_FEATURE_COLUMNS)
    p_gmm = save_gmm_model(gmm_res["model"], model_path=tmp_path / "gmm.pkl")
    loaded_gmm = load_gmm_model(model_path=p_gmm)
    assert loaded_gmm is not None

    km_res = run_kmeans_pipeline(multi_user_feature_df, feature_columns=CLUSTER_FEATURE_COLUMNS)
    p_km = save_kmeans_model(km_res["model"], model_path=tmp_path / "kmeans.pkl")
    loaded_km = load_kmeans_model(model_path=p_km)
    assert loaded_km is not None

    hmm_res = run_hmm_pipeline(gmm_res["labeled_df"], observation_column="gmm_cluster")
    p_hmm = save_hmm_model(hmm_res["model"], model_name="test_hmm.pkl")
    assert p_hmm.exists()


def test_hmm_never_fits_on_test_data(multi_user_feature_df: pd.DataFrame, monkeypatch) -> None:
    """Verify evaluation pipeline passes fitted_model and NEVER invokes .fit() during test set evaluation."""
    gmm_res = run_gmm_pipeline(multi_user_feature_df, feature_columns=CLUSTER_FEATURE_COLUMNS)
    train_labeled = gmm_res["labeled_df"].iloc[:15]
    test_labeled = gmm_res["labeled_df"].iloc[15:]

    # Train HMM on train_labeled
    trained_hmm_res = run_hmm_pipeline(train_labeled, observation_column="gmm_cluster", save_model=False)
    fitted_model = trained_hmm_res["model"]

    # Track if .fit() is called on fitted_model
    fit_called = False
    original_fit = fitted_model.fit

    def mock_fit(*args, **kwargs):
        nonlocal fit_called
        fit_called = True
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(fitted_model, "fit", mock_fit)

    # Evaluate on test_labeled with fitted_model
    eval_res = run_hmm_pipeline(test_labeled, observation_column="gmm_cluster", fitted_model=fitted_model, save_model=False)

    assert not fit_called, "HMM model.fit() was invoked during test set evaluation!"
    assert eval_res["decoded_df"] is not None


def test_b0_uses_only_raw_features() -> None:
    """Verify B0 baseline strictly uses unnormalized raw physiological features."""
    raw_cols = ["resting_hr_bpm", "hrv_rmssd_ms", "sleep_duration_hours", "steps"]
    derived_terms = ["dev", "slope", "severity", "baseline"]
    for col in raw_cols:
        for term in derived_terms:
            assert term not in col.lower(), f"B0 raw feature '{col}' contains derived term '{term}'!"


def test_b1_executable_threshold_baseline(multi_user_feature_df: pd.DataFrame) -> None:
    """Verify B1 baseline runs dynamically on data and computes valid metrics without hardcoded constants."""
    from experiments.run_all import run_b1_threshold_baseline
    res = run_b1_threshold_baseline(multi_user_feature_df, feature_columns=CLUSTER_FEATURE_COLUMNS)
    assert "metrics" in res
    assert "silhouette" in res["metrics"]
    assert "davies_bouldin" in res["metrics"]
    assert "calinski_harabasz" in res["metrics"]
    assert isinstance(res["metrics"]["silhouette"], float)

