from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.gmm import run_gmm_pipeline
from src.kmeans import run_kmeans_pipeline
from src.hmm_model import run_hmm_pipeline
from src.preprocessing import CLUSTER_FEATURE_COLUMNS, preprocess_for_modeling

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "wearables_health_6mo_daily.csv"

def evaluate_benchmarks() -> dict:
    print("Running Experimental Evaluation Benchmarks...")
    outputs = preprocess_for_modeling(source=DATA_PATH)
    feature_df = outputs["feature_df"]

    # 1. GMM Evaluation
    gmm_out = run_gmm_pipeline(feature_df, feature_columns=CLUSTER_FEATURE_COLUMNS)
    gmm_metrics = gmm_out["metrics"]

    # 2. KMeans Evaluation
    kmeans_out = run_kmeans_pipeline(feature_df, feature_columns=CLUSTER_FEATURE_COLUMNS)
    kmeans_metrics = kmeans_out["metrics"]

    # 3. HMM Evaluation
    hmm_out = run_hmm_pipeline(gmm_out["labeled_df"], observation_column="gmm_cluster", n_components=3)

    summary = {
        "gmm_silhouette": round(float(gmm_metrics.get("silhouette", 0.0)), 4),
        "gmm_davies_bouldin": round(float(gmm_metrics.get("davies_bouldin", 0.0)), 4),
        "gmm_calinski_harabasz": round(float(gmm_metrics.get("calinski_harabasz", 0.0)), 4),
        "kmeans_silhouette": round(float(kmeans_metrics.get("silhouette_score", 0.0)), 4),
        "kmeans_davies_bouldin": round(float(kmeans_metrics.get("davies_bouldin_index", 0.0)), 4),
        "hmm_transition_rate": round(float(hmm_out.get("transition_rate", 0.0)), 4),
    }

    print("Benchmark Results:")
    for k, v in summary.items():
        print(f"  - {k}: {v}")

    return summary

if __name__ == "__main__":
    evaluate_benchmarks()
