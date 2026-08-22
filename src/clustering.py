from pathlib import Path

import joblib
import pandas as pd


MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def load_cluster_model(model_name: str = "kmeans.pkl"):
    """Load a serialized clustering model if available."""
    model_path = MODEL_DIR / model_name
    if model_path.exists() and model_path.stat().st_size > 0:
        return joblib.load(model_path)
    return None


def assign_health_cluster(features: pd.DataFrame) -> dict:
    """Infer a health cluster or return a rule-based fallback."""
    model = load_cluster_model()
    if model is not None:
        cluster_id = int(model.predict(features)[0])
        return {"cluster_id": cluster_id, "method": "loaded_model"}

    heuristic_cluster = 1 if features.loc[0, "stress_mean"] > 45 else 0
    return {"cluster_id": heuristic_cluster, "method": "rule_based_fallback"}
