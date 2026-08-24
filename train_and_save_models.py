from __future__ import annotations

from pathlib import Path

from src.gmm import run_gmm_pipeline
from src.hmm_model import run_hmm_pipeline
from src.kmeans import run_kmeans_pipeline
from src.preprocessing import CLUSTER_FEATURE_COLUMNS, chronological_split, preprocess_for_modeling, save_scaler

ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "wearables_health_6mo_daily.csv"
MODEL_DIR = ROOT_DIR / "models"


def main() -> None:
    print("Training and serializing models on training split...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    outputs = preprocess_for_modeling(source=DATA_PATH)
    feature_df = outputs["feature_df"]
    scaler = outputs["scaler"]
    save_scaler(scaler, MODEL_DIR / "scaler.pkl")

    # Train on first 70% chronological split per user
    train_df, val_df, test_df = chronological_split(feature_df, train_pct=0.70, val_pct=0.15, test_pct=0.15)
    print(f"Dataset split: Train={len(train_df)} rows, Val={len(val_df)} rows, Test={len(test_df)} rows.")

    # 1. KMeans
    print("Fitting KMeans...")
    kmeans_out = run_kmeans_pipeline(
        train_df,
        feature_columns=CLUSTER_FEATURE_COLUMNS,
        model_path=MODEL_DIR / "kmeans.pkl",
        save_trained_model=True,
    )
    print(f"KMeans saved to {kmeans_out['model_path']}")

    # 2. GMM
    print("Fitting GMM...")
    gmm_out = run_gmm_pipeline(
        train_df,
        feature_columns=CLUSTER_FEATURE_COLUMNS,
        model_path=MODEL_DIR / "gmm.pkl",
        save_trained_model=True,
    )
    print(f"GMM saved to {gmm_out['model_path']}")

    # 3. HMM
    print("Fitting HMM...")
    hmm_input = gmm_out["labeled_df"].copy()
    hmm_out = run_hmm_pipeline(
        hmm_input,
        observation_column="gmm_cluster",
        n_components=gmm_out["selected_k"],
        model_name="hmm.pkl",
        save_model=True,
    )
    print(f"HMM saved to {hmm_out['model_path']}")

    print("\nModel training and serialization complete!")


if __name__ == "__main__":
    main()
