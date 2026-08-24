from __future__ import annotations

import numpy as np
import pandas as pd

from src.gmm import run_gmm_pipeline
from src.hmm_model import run_soft_probability_hmm
from src.preprocessing import CLUSTER_FEATURE_COLUMNS


def inject_missingness(df: pd.DataFrame, missing_pct: float, seed: int = 42) -> pd.DataFrame:
    """Inject controlled MCAR missingness into physiological columns for robustness simulation."""
    corrupted = df.copy()
    rng = np.random.default_rng(seed)
    target_cols = [col for col in CLUSTER_FEATURE_COLUMNS if col in corrupted.columns]

    for col in target_cols:
        mask = rng.random(len(corrupted)) < missing_pct
        corrupted.loc[mask, col] = np.nan

    return corrupted


def inject_measurement_noise(df: pd.DataFrame, noise_std: float, seed: int = 42) -> pd.DataFrame:
    """Inject Gaussian measurement noise into physiological columns for robustness simulation."""
    corrupted = df.copy()
    rng = np.random.default_rng(seed)
    target_cols = [col for col in CLUSTER_FEATURE_COLUMNS if col in corrupted.columns]

    for col in target_cols:
        noise = rng.normal(0, noise_std, size=len(corrupted))
        corrupted[col] = corrupted[col] + noise

    return corrupted


def run_robustness_experiments(feature_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Execute controlled robustness simulations across missingness levels (10%, 20%, 30%)
    and measurement noise levels (sigma = 0.05, 0.10, 0.20).
    Label as synthetic robustness simulation.
    """
    rows = []

    # Baseline unperturbed
    gmm_base = run_gmm_pipeline(feature_df, random_state=seed, save_trained_model=False)
    soft_base = run_soft_probability_hmm(gmm_base["labeled_df"], random_state=seed)

    rows.append({
        "Experiment_Type": "Baseline (Clean)",
        "Perturbation_Level": "0%",
        "Silhouette": round(float(gmm_base["metrics"]["silhouette"]), 4),
        "Davies_Bouldin": round(float(gmm_base["metrics"]["davies_bouldin"]), 4),
        "Transition_Entropy_Bits": round(float(soft_base.get("transition_entropy_bits", 0.0)), 4),
        "Confidence_Pct": round(float(soft_base.get("mean_confidence", 0.0)), 1),
    })

    # Missingness simulations
    for missing_pct in [0.10, 0.20, 0.30]:
        corrupted = inject_missingness(feature_df, missing_pct=missing_pct, seed=seed)
        # Handle causally
        corrupted[CLUSTER_FEATURE_COLUMNS] = corrupted[CLUSTER_FEATURE_COLUMNS].ffill().fillna(0.0)

        gmm_out = run_gmm_pipeline(corrupted, random_state=seed, save_trained_model=False)
        soft_out = run_soft_probability_hmm(gmm_out["labeled_df"], random_state=seed)

        rows.append({
            "Experiment_Type": "Missingness Simulation (MCAR)",
            "Perturbation_Level": f"{int(missing_pct*100)}%",
            "Silhouette": round(float(gmm_out["metrics"]["silhouette"]), 4),
            "Davies_Bouldin": round(float(gmm_out["metrics"]["davies_bouldin"]), 4),
            "Transition_Entropy_Bits": round(float(soft_out.get("transition_entropy_bits", 0.0)), 4),
            "Confidence_Pct": round(float(soft_out.get("mean_confidence", 0.0)), 1),
        })

    # Noise simulations
    for noise_std in [0.05, 0.10, 0.20]:
        corrupted = inject_measurement_noise(feature_df, noise_std=noise_std, seed=seed)

        gmm_out = run_gmm_pipeline(corrupted, random_state=seed, save_trained_model=False)
        soft_out = run_soft_probability_hmm(gmm_out["labeled_df"], random_state=seed)

        rows.append({
            "Experiment_Type": "Gaussian Noise Simulation",
            "Perturbation_Level": f"sigma={noise_std}",
            "Silhouette": round(float(gmm_out["metrics"]["silhouette"]), 4),
            "Davies_Bouldin": round(float(gmm_out["metrics"]["davies_bouldin"]), 4),
            "Transition_Entropy_Bits": round(float(soft_out.get("transition_entropy_bits", 0.0)), 4),
            "Confidence_Pct": round(float(soft_out.get("mean_confidence", 0.0)), 1),
        })

    return pd.DataFrame(rows)
