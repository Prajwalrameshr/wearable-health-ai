from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture

STANDARD_STATES = ["Recovery", "Baseline", "Strain"]
DEFAULT_K_RANGE = range(2, 5)
DEFAULT_GMM_FEATURES = ["hr_dev", "hrv_dev", "sleep_dev", "severity_score"]
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def prepare_gmm_features(df: pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
    selected_columns = DEFAULT_GMM_FEATURES if feature_columns is None else feature_columns
    missing_columns = [column for column in selected_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing GMM feature columns: {', '.join(missing_columns)}")
    feature_matrix = df[selected_columns].replace([np.inf, -np.inf], np.nan).dropna(axis=0).copy()
    if feature_matrix.empty:
        raise ValueError("No valid rows available for GMM after removing NaN/inf values.")
    return feature_matrix


def find_optimal_k(feature_matrix: pd.DataFrame, k_values: range = DEFAULT_K_RANGE, random_state: int = 42) -> tuple[pd.DataFrame, int]:
    rows: list[dict[str, float]] = []
    for k in k_values:
        if len(feature_matrix) < k:
            continue
        model = GaussianMixture(n_components=k, covariance_type="diag", random_state=random_state, reg_covar=1e-4)
        model.fit(feature_matrix)
        rows.append({"k": k, "bic": float(model.bic(feature_matrix)), "aic": float(model.aic(feature_matrix))})
    if not rows:
        raise ValueError("Not enough samples to evaluate GMM models.")
    selection_df = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    available_k = selection_df["k"].astype(int).tolist()
    selected_k = 3 if 3 in available_k else int(selection_df.loc[selection_df["bic"].idxmin(), "k"])
    return selection_df, selected_k


def train_gmm(feature_matrix: pd.DataFrame, n_components: int, random_state: int = 42) -> tuple[GaussianMixture, np.ndarray, np.ndarray]:
    model = GaussianMixture(n_components=n_components, covariance_type="diag", random_state=random_state, reg_covar=1e-4)
    model.fit(feature_matrix)
    labels = model.predict(feature_matrix)
    probabilities = model.predict_proba(feature_matrix)
    return model, labels, probabilities


def build_state_map(cluster_summary: pd.DataFrame, cluster_column: str = "gmm_cluster") -> tuple[dict[int, str], dict[int, str]]:
    state_map: dict[int, str] = {}
    probability_column_map: dict[int, str] = {}
    summary_sorted = cluster_summary.copy()
    if "severity_score" not in summary_sorted.columns:
        hr_dev = summary_sorted["hr_dev"].abs() if "hr_dev" in summary_sorted.columns else 0.0
        hrv_dev = summary_sorted["hrv_dev"].abs() if "hrv_dev" in summary_sorted.columns else 0.0
        sleep_dev = summary_sorted["sleep_dev"].abs() if "sleep_dev" in summary_sorted.columns else 0.0
        summary_sorted["severity_score"] = hr_dev + hrv_dev + sleep_dev

    summary_sorted = summary_sorted.sort_values("severity_score").reset_index(drop=True)
    n_clusters = len(summary_sorted)

    if n_clusters == 3:
        assigned_states = ["Recovery", "Baseline", "Strain"]
    elif n_clusters == 2:
        assigned_states = ["Recovery", "Strain"]
    elif n_clusters == 1:
        assigned_states = ["Baseline"]
    else:
        assigned_states = [STANDARD_STATES[min(i, len(STANDARD_STATES) - 1)] for i in range(n_clusters)]

    for idx, row in summary_sorted.iterrows():
        cluster_id = int(row[cluster_column])
        state_name = assigned_states[idx]
        state_map[cluster_id] = state_name
        probability_column_map[cluster_id] = f"prob_{state_name.lower().replace(' ', '_')}"
    return state_map, probability_column_map


def assign_state_labels(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    cluster_probabilities: np.ndarray,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str], dict[int, str]]:
    selected_columns = DEFAULT_GMM_FEATURES if feature_columns is None else feature_columns
    labeled = df.copy()
    labeled["gmm_cluster"] = cluster_labels.astype(int)
    summary_columns = list(dict.fromkeys(selected_columns + ["hrv_rmssd_ms", "resting_hr_bpm", "sleep_duration_hours", "severity_score"]))
    summary_columns = [column for column in summary_columns if column in labeled.columns]
    cluster_summary = labeled.groupby("gmm_cluster")[summary_columns].mean().reset_index()
    state_map, probability_column_map = build_state_map(cluster_summary)
    labeled["gmm_state_label"] = labeled["gmm_cluster"].map(state_map)
    for state_label in STANDARD_STATES:
        labeled[f"prob_{state_label.lower().replace(' ', '_')}"] = 0.0
    for cluster_id, probability_column in probability_column_map.items():
        labeled[probability_column] = np.maximum(labeled[probability_column], cluster_probabilities[:, cluster_id])
    probability_columns = [f"prob_{state.lower().replace(' ', '_')}" for state in STANDARD_STATES]
    labeled["state_confidence"] = labeled[probability_columns].max(axis=1)
    labeled["dominant_probability_state"] = labeled[probability_columns].idxmax(axis=1).str.replace("prob_", "", regex=False).str.replace("_", " ").str.title()
    cluster_summary["gmm_state_label"] = cluster_summary["gmm_cluster"].map(state_map)
    cluster_summary["severity_rank"] = cluster_summary["gmm_state_label"].map({"Recovery": 1, "Baseline": 2, "Strain": 3}).fillna(2)
    cluster_summary = cluster_summary.sort_values(["severity_rank", "gmm_cluster"]).reset_index(drop=True)
    return labeled, cluster_summary, state_map, probability_column_map


def compute_metrics(feature_matrix: pd.DataFrame, cluster_labels: np.ndarray, labeled_df: pd.DataFrame) -> dict[str, Any]:
    unique_clusters = np.unique(cluster_labels)
    if len(unique_clusters) < 2:
        silhouette = float("nan")
        davies_bouldin = float("nan")
        calinski_harabasz = float("nan")
    else:
        silhouette = float(silhouette_score(feature_matrix, cluster_labels))
        davies_bouldin = float(davies_bouldin_score(feature_matrix, cluster_labels))
        calinski_harabasz = float(calinski_harabasz_score(feature_matrix, cluster_labels))
    ordered = labeled_df.groupby("gmm_state_label")["severity_score"].mean().reindex(STANDARD_STATES).dropna()
    return {
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "calinski_harabasz": calinski_harabasz,
        "risk_monotonicity": bool(ordered.is_monotonic_increasing),
        "cluster_distribution": labeled_df["gmm_cluster"].value_counts().sort_index().astype(int).to_dict(),
        "severity_by_state": ordered.round(4).to_dict(),
    }


def plot_bic_aic_curve(selection_df: pd.DataFrame, output_path: str | Path | None = None) -> Path | None:
    plt.figure(figsize=(7, 4))
    plt.plot(selection_df["k"], selection_df["bic"], marker="o", linewidth=2, label="BIC")
    plt.plot(selection_df["k"], selection_df["aic"], marker="s", linewidth=2, label="AIC")
    plt.legend()
    plt.tight_layout()
    if output_path is None:
        plt.close()
        return None
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    return output


def save_model(model: GaussianMixture, model_path: str | Path | None = None) -> Path:
    target_path = Path(model_path) if model_path is not None else MODEL_DIR / "gmm.pkl"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target_path)
    return target_path


def save_gmm_model(model: GaussianMixture, model_path: str | Path | None = None) -> Path:
    """Save trained GaussianMixture model to disk."""
    return save_model(model, model_path=model_path)


def load_gmm_model(model_path: str | Path | None = None) -> GaussianMixture:
    """Load persisted GaussianMixture model from disk."""
    target_path = Path(model_path) if model_path is not None else MODEL_DIR / "gmm.pkl"
    return joblib.load(target_path)



def run_gmm_pipeline(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    k_values: range = DEFAULT_K_RANGE,
    random_state: int = 42,
    bic_aic_plot_path: str | Path | None = None,
    model_path: str | Path | None = None,
    save_trained_model: bool = False,
) -> dict[str, Any]:
    selected_columns = DEFAULT_GMM_FEATURES if feature_columns is None else feature_columns
    feature_matrix = prepare_gmm_features(df, feature_columns=selected_columns)
    aligned_df = df.loc[feature_matrix.index].copy()
    selection_df, selected_k = find_optimal_k(feature_matrix, k_values=k_values, random_state=random_state)
    bic_aic_plot = plot_bic_aic_curve(selection_df, output_path=bic_aic_plot_path)
    model, cluster_labels, cluster_probabilities = train_gmm(feature_matrix, n_components=selected_k, random_state=random_state)
    labeled_df, cluster_summary, state_map, probability_column_map = assign_state_labels(aligned_df, cluster_labels, cluster_probabilities, feature_columns=selected_columns)
    metrics = compute_metrics(feature_matrix, cluster_labels, labeled_df)
    saved_model_path = save_model(model, model_path=model_path) if save_trained_model else None
    return {
        "labeled_df": labeled_df,
        "metrics": metrics,
        "cluster_summary": cluster_summary,
        "feature_matrix": feature_matrix,
        "feature_columns": selected_columns,
        "selection_df": selection_df,
        "selected_k": selected_k,
        "cluster_probabilities": cluster_probabilities,
        "state_map": state_map,
        "probability_column_map": probability_column_map,
        "model": model,
        "bic_aic_plot_path": bic_aic_plot,
        "model_path": saved_model_path,
    }
