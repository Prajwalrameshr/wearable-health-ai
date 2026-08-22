from __future__ import annotations

from pathlib import Path
import joblib

from src.gmm import run_gmm_pipeline, save_gmm_model
from src.hmm_model import run_hmm_pipeline, save_hmm_model
from src.kmeans import run_kmeans_pipeline, save_kmeans_model
from src.preprocessing import (
    CLUSTER_FEATURE_COLUMNS,
    chronological_split,
    preprocess_for_modeling,
    save_scaler,
)

ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "wearables_health_6mo_daily.csv"
MODEL_DIR = ROOT_DIR / "models"


def main() -> None:
    print("Training and serializing models with chronological data split (70% Train)...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    outputs = preprocess_for_modeling(source=DATA_PATH)
    feature_df = outputs["feature_df"]
    scaler = outputs["scaler"]

    # Save StandardScaler
    scaler_path = MODEL_DIR / "scaler.pkl"
    save_scaler(scaler, scaler_path)
    print(f"StandardScaler saved to {scaler_path}")

    # Chronological Split (70% train)
    train_df, val_df, test_df = chronological_split(feature_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    print(f"Dataset split: Train={len(train_df)} rows, Val={len(val_df)} rows, Test={len(test_df)} rows")

    # 1. KMeans
    print("Fitting KMeans on Train Split...")
    kmeans_out = run_kmeans_pipeline(
        train_df,
        feature_columns=CLUSTER_FEATURE_COLUMNS,
        model_path=MODEL_DIR / "kmeans.pkl",
        save_trained_model=True,
    )
    print(f"KMeans saved to {kmeans_out['model_path']}")

    # 2. GMM
    print("Fitting GMM on Train Split...")
    gmm_out = run_gmm_pipeline(
        train_df,
        feature_columns=CLUSTER_FEATURE_COLUMNS,
        model_path=MODEL_DIR / "gmm.pkl",
        save_trained_model=True,
    )
    print(f"GMM saved to {gmm_out['model_path']}")

    # 3. HMM
    print("Fitting HMM on Train Split...")
    hmm_input = gmm_out["labeled_df"].copy()
    hmm_out = run_hmm_pipeline(
        hmm_input,
        observation_column="gmm_cluster",
        n_components=3,
        model_name="hmm.pkl",
        save_model=True,
    )
    print(f"HMM saved to {hmm_out['model_path']}")

    print("\nModel training and serialization complete!")


if __name__ == "__main__":
    main()

