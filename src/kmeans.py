from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

STANDARD_STATES = ["Recovery", "Baseline", "Strain"]
DEFAULT_K_RANGE = range(2, 6)
DEFAULT_KMEANS_FEATURES = ["hr_dev", "hrv_dev", "sleep_dev", "severity_score"]
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def prepare_kmeans_features(df: pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
    selected_columns = DEFAULT_KMEANS_FEATURES if feature_columns is None else feature_columns
    missing_columns = [column for column in selected_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing clustering feature columns: {', '.join(missing_columns)}")
    feature_matrix = df[selected_columns].replace([np.inf, -np.inf], np.nan).dropna(axis=0).copy()
    if feature_matrix.empty:
        raise ValueError("No valid rows available for KMeans after removing NaN/inf values.")
    return feature_matrix


def compute_elbow_curve(feature_matrix: pd.DataFrame, k_values: range = DEFAULT_K_RANGE, random_state: int = 42) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for k in k_values:
        if len(feature_matrix) < k:
            continue
        model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = model.fit_predict(feature_matrix)
        sil = float(silhouette_score(feature_matrix, labels)) if len(np.unique(labels)) > 1 else -1.0
        db = float(davies_bouldin_score(feature_matrix, labels)) if len(np.unique(labels)) > 1 else 999.0
        rows.append({"k": k, "inertia": float(model.inertia_), "silhouette": sil, "davies_bouldin": db})
    if not rows:
        raise ValueError("Not enough samples to compute elbow curve.")
    return pd.DataFrame(rows)


def select_optimal_k(elbow_df: pd.DataFrame) -> int:
    """Select optimal k maximizing Silhouette score dynamically."""
    if "silhouette" in elbow_df.columns and not elbow_df["silhouette"].empty:
        best_k = int(elbow_df.loc[elbow_df["silhouette"].idxmax(), "k"])
        return best_k
    available_k = elbow_df["k"].astype(int).tolist()
    return int(available_k[0])


def fit_kmeans(feature_matrix: pd.DataFrame, n_clusters: int, random_state: int = 42) -> KMeans:
    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
    model.fit(feature_matrix)
    return model


def summarize_clusters(df: pd.DataFrame, cluster_column: str = "kmeans_cluster", feature_columns: list[str] | None = None) -> pd.DataFrame:
    summary_columns = list(DEFAULT_KMEANS_FEATURES if feature_columns is None else feature_columns)
    for column in ["hrv_rmssd_ms", "resting_hr_bpm", "sleep_duration_hours", "severity_score"]:
        if column not in summary_columns and column in df.columns:
            summary_columns.append(column)
    summary = df.groupby(cluster_column)[summary_columns].mean().reset_index()
    if "severity_score" not in summary.columns:
        hr_dev = summary["hr_dev"].abs() if "hr_dev" in summary.columns else 0.0
        hrv_dev = summary["hrv_dev"].abs() if "hrv_dev" in summary.columns else 0.0
        sleep_dev = summary["sleep_dev"].abs() if "sleep_dev" in summary.columns else 0.0
        summary["severity_score"] = hr_dev + hrv_dev + sleep_dev

    summary = summary.sort_values("severity_score").reset_index(drop=True)
    n_clusters = len(summary)

    if n_clusters == 3:
        assigned_states = ["Recovery", "Baseline", "Strain"]
    elif n_clusters == 2:
        assigned_states = ["Recovery", "Strain"]
    elif n_clusters == 1:
        assigned_states = ["Baseline"]
    else:
        assigned_states = [f"State_{i+1}" for i in range(n_clusters)]

    summary["cluster_label"] = assigned_states
    severity_rank = {"Recovery": 1, "Baseline": 2, "Strain": 3}
    summary["severity_rank"] = summary["cluster_label"].map(severity_rank).fillna(2)
    summary = summary.sort_values(["severity_rank", cluster_column]).reset_index(drop=True)
    return summary


def attach_cluster_labels(df: pd.DataFrame, cluster_labels: np.ndarray, cluster_summary: pd.DataFrame) -> pd.DataFrame:
    labeled = df.copy()
    labeled["kmeans_cluster"] = cluster_labels.astype(int)
    label_map = dict(zip(cluster_summary["kmeans_cluster"], cluster_summary["cluster_label"]))
    labeled["cluster_label"] = labeled["kmeans_cluster"].map(label_map)
    return labeled


def compute_metrics(feature_matrix: pd.DataFrame, cluster_labels: np.ndarray, labeled_df: pd.DataFrame | None = None) -> dict[str, float | bool]:
    unique_clusters = np.unique(cluster_labels)
    if len(unique_clusters) < 2:
        metrics: dict[str, float | bool] = {
            "silhouette_score": float("nan"),
            "davies_bouldin_index": float("nan"),
            "calinski_harabasz_index": float("nan"),
        }
    else:
        metrics = {
            "silhouette_score": float(silhouette_score(feature_matrix, cluster_labels)),
            "davies_bouldin_index": float(davies_bouldin_score(feature_matrix, cluster_labels)),
            "calinski_harabasz_index": float(calinski_harabasz_score(feature_matrix, cluster_labels)),
        }
    if labeled_df is not None and "cluster_label" in labeled_df.columns and "severity_score" in labeled_df.columns:
        order_map = {state: idx for idx, state in enumerate(STANDARD_STATES)}
        ordered_severity = labeled_df.groupby("cluster_label")["severity_score"].mean()
        ordered_severity = ordered_severity.reindex(sorted(ordered_severity.index, key=lambda value: order_map.get(value, 99))).dropna()
        metrics["risk_monotonicity"] = bool(ordered_severity.is_monotonic_increasing)
    return metrics


def plot_elbow_curve(elbow_df: pd.DataFrame, output_path: str | Path | None = None) -> Path | None:
    plt.figure(figsize=(7, 4))
    plt.plot(elbow_df["k"], elbow_df["inertia"], marker="o", linewidth=2)
    plt.title("KMeans Model Selection")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.tight_layout()
    if output_path is None:
        plt.close()
        return None
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    return output


def save_model(model: KMeans, model_path: str | Path | None = None) -> Path:
    target_path = Path(model_path) if model_path is not None else MODEL_DIR / "kmeans.pkl"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target_path)
    return target_path


def load_kmeans_model(model_path: str | Path | None = None) -> KMeans | None:
    target_path = Path(model_path) if model_path is not None else MODEL_DIR / "kmeans.pkl"
    if target_path.exists() and target_path.stat().st_size > 0:
        try:
            return joblib.load(target_path)
        except Exception:
            return None
    return None


def run_kmeans_pipeline(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    k_values: range = DEFAULT_K_RANGE,
    random_state: int = 42,
    elbow_plot_path: str | Path | None = None,
    model_path: str | Path | None = None,
    save_trained_model: bool = True,
    fitted_model: KMeans | None = None,
) -> dict[str, Any]:
    selected_columns = DEFAULT_KMEANS_FEATURES if feature_columns is None else feature_columns
    feature_matrix = prepare_kmeans_features(df, feature_columns=selected_columns)
    aligned_df = df.loc[feature_matrix.index].copy()

    if fitted_model is None:
        elbow_df = compute_elbow_curve(feature_matrix, k_values=k_values, random_state=random_state)
        optimal_k = select_optimal_k(elbow_df)
        plot_path = plot_elbow_curve(elbow_df, output_path=elbow_plot_path)
        model = fit_kmeans(feature_matrix, n_clusters=optimal_k, random_state=random_state)
        cluster_labels = model.labels_
    else:
        model = fitted_model
        optimal_k = model.n_clusters
        elbow_df = pd.DataFrame([{"k": optimal_k, "inertia": float(model.inertia_)}])
        plot_path = None
        cluster_labels = model.predict(feature_matrix)

    unlabeled_df = aligned_df.copy()
    unlabeled_df["kmeans_cluster"] = cluster_labels.astype(int)
    cluster_summary = summarize_clusters(unlabeled_df, cluster_column="kmeans_cluster", feature_columns=selected_columns)
    labeled_df = attach_cluster_labels(aligned_df, cluster_labels, cluster_summary)
    metrics = compute_metrics(feature_matrix, cluster_labels, labeled_df=labeled_df)
    saved_model_path = save_model(model, model_path=model_path) if save_trained_model and fitted_model is None else None

    return {
        "labeled_df": labeled_df,
        "metrics": metrics,
        "cluster_summary": cluster_summary,
        "feature_matrix": feature_matrix,
        "feature_columns": selected_columns,
        "elbow_curve": elbow_df,
        "optimal_k": optimal_k,
        "model": model,
        "plot_path": plot_path,
        "model_path": saved_model_path,
    }
