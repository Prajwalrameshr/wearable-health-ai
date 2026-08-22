from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.gmm import run_gmm_pipeline
from src.preprocessing import CLUSTER_FEATURE_COLUMNS, preprocess_for_modeling

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "wearables_health_6mo_daily.csv"

def run_robustness_tests(noise_levels: list[float] = [0.0, 0.05, 0.10, 0.20]) -> dict:
    print("Running Pipeline Robustness & Noise Injection Suite...")
    outputs = preprocess_for_modeling(source=DATA_PATH)
    clean_feature_df = outputs["feature_df"].copy()

    baseline_gmm = run_gmm_pipeline(clean_feature_df, feature_columns=CLUSTER_FEATURE_COLUMNS)
    baseline_labels = baseline_gmm["labeled_df"]["gmm_cluster"].to_numpy()

    results = {}
    for sigma in noise_levels:
        if sigma == 0.0:
            results["0% Noise"] = 1.0
            continue

        noisy_df = clean_feature_df.copy()
        for col in CLUSTER_FEATURE_COLUMNS:
            noise = np.random.normal(0, sigma * noisy_df[col].std(), len(noisy_df))
            noisy_df[col] = noisy_df[col] + noise

        noisy_gmm = run_gmm_pipeline(noisy_df, feature_columns=CLUSTER_FEATURE_COLUMNS)
        noisy_labels = noisy_gmm["labeled_df"]["gmm_cluster"].to_numpy()

        agreement = float(np.mean(baseline_labels == noisy_labels))
        results[f"{int(sigma*100)}% Gaussian Noise"] = round(agreement, 4)
        print(f"  - Stability at {int(sigma*100)}% Gaussian Noise: {round(agreement*100, 2)}% label agreement")

    return results

if __name__ == "__main__":
    run_robustness_tests()
