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
DEFAULT_K_RANGE = range(2, 6)  # k = 2, 3, 4, 5
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
    """Evaluate k in range(2, 6) using BIC/AIC without hardcoded k=3 bias."""
    rows: list[dict[str, float]] = []
    for k in k_values:
        if len(feature_matrix) < k:
            continue
        model = GaussianMixture(n_components=k, covariance_type="diag", random_state=random_state, reg_covar=1e-4)
        model.fit(feature_matrix)
        rows.append({
            "k": k,
            "bic": float(model.bic(feature_matrix)),
            "aic": float(model.aic(feature_matrix)),
            "log_likelihood": float(model.score(feature_matrix)),
        })
    if not rows:
        raise ValueError("Not enough samples to evaluate GMM models.")
    selection_df = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    # Select k that minimizes BIC strictly
    selected_k = int(selection_df.loc[selection_df["bic"].idxmin(), "k"])
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
        assigned_states = [f"State_{i+1}" for i in range(n_clusters)]

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

    # Initialize state probability columns
    for state_label in STANDARD_STATES:
        labeled[f"prob_{state_label.lower().replace(' ', '_')}"] = 0.0

    for cluster_id, probability_column in probability_column_map.items():
        labeled[probability_column] = cluster_probabilities[:, cluster_id]

    probability_columns = [f"prob_{state.lower().replace(' ', '_')}" for state in STANDARD_STATES if f"prob_{state.lower().replace(' ', '_')}" in labeled.columns]
    labeled["state_confidence"] = labeled[probability_columns].max(axis=1) if probability_columns else cluster_probabilities.max(axis=1)
    labeled["dominant_probability_state"] = labeled["gmm_state_label"]

    cluster_summary["gmm_state_label"] = cluster_summary["gmm_cluster"].map(state_map)
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

    if "severity_score" in labeled_df.columns:
        ordered = labeled_df.groupby("gmm_state_label")["severity_score"].mean().reindex(STANDARD_STATES).dropna()
        risk_mono = bool(ordered.is_monotonic_increasing)
        sev_by_state = ordered.round(4).to_dict()
    else:
        risk_mono = True
        sev_by_state = {}

    return {
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "calinski_harabasz": calinski_harabasz,
        "risk_monotonicity": risk_mono,
        "cluster_distribution": labeled_df["gmm_cluster"].value_counts().sort_index().astype(int).to_dict(),
        "severity_by_state": sev_by_state,
    }


def compute_gmm_native_feature_importance(
    model: GaussianMixture,
    feature_names: list[str],
    sample: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Model-native GMM feature attribution derived from component means and variances.
    Computes absolute distance of cluster component means from baseline center weighted by component variance.
    """
    means = model.means_  # shape: (n_components, n_features)
    covars = model.covariances_  # shape: (n_components, n_features) if diag

    if covars.ndim == 1:
        stds = np.sqrt(covars)
    elif covars.ndim == 2:  # (n_components, n_features)
        stds = np.sqrt(covars)
    else:
        stds = np.ones_like(means)

    # Feature importance per feature: variance across cluster means divided by mean feature variance
    feature_var_between = np.var(means, axis=0)
    feature_var_within = np.mean(stds ** 2, axis=0)
    importance_scores = feature_var_between / (feature_var_within + 1e-6)

    # Normalize to sum to 1.0
    if np.sum(importance_scores) > 0:
        importance_scores = importance_scores / np.sum(importance_scores)

    imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importance_scores}).sort_values("Importance", ascending=False)
    return imp_df.reset_index(drop=True)


def plot_bic_aic_curve(selection_df: pd.DataFrame, output_path: str | Path | None = None) -> Path | None:
    plt.figure(figsize=(7, 4))
    plt.plot(selection_df["k"], selection_df["bic"], marker="o", linewidth=2, label="BIC")
    plt.plot(selection_df["k"], selection_df["aic"], marker="s", linewidth=2, label="AIC")
    plt.xlabel("Number of Clusters k")
    plt.ylabel("Information Criterion")
    plt.title("GMM Model Selection (BIC / AIC)")
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


def save_gmm_model(model: GaussianMixture, model_path: str | Path | None = None) -> Path:
    target_path = Path(model_path) if model_path is not None else MODEL_DIR / "gmm.pkl"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target_path)
    return target_path


def load_gmm_model(model_path: str | Path | None = None) -> GaussianMixture | None:
    target_path = Path(model_path) if model_path is not None else MODEL_DIR / "gmm.pkl"
    if target_path.exists() and target_path.stat().st_size > 0:
        try:
            return joblib.load(target_path)
        except Exception:
            return None
    return None


def run_gmm_pipeline(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    k_values: range = DEFAULT_K_RANGE,
    random_state: int = 42,
    bic_aic_plot_path: str | Path | None = None,
    model_path: str | Path | None = None,
    save_trained_model: bool = True,
    fitted_model: GaussianMixture | None = None,
) -> dict[str, Any]:
    selected_columns = DEFAULT_GMM_FEATURES if feature_columns is None else feature_columns
    feature_matrix = prepare_gmm_features(df, feature_columns=selected_columns)
    aligned_df = df.loc[feature_matrix.index].copy()

    if fitted_model is None:
        selection_df, selected_k = find_optimal_k(feature_matrix, k_values=k_values, random_state=random_state)
        bic_aic_plot = plot_bic_aic_curve(selection_df, output_path=bic_aic_plot_path)
        model, cluster_labels, cluster_probabilities = train_gmm(feature_matrix, n_components=selected_k, random_state=random_state)
    else:
        model = fitted_model
        selected_k = model.n_components
        selection_df = pd.DataFrame([{"k": selected_k, "bic": float(model.bic(feature_matrix)), "aic": float(model.aic(feature_matrix))}])
        bic_aic_plot = None
        cluster_labels = model.predict(feature_matrix)
        cluster_probabilities = model.predict_proba(feature_matrix)

    labeled_df, cluster_summary, state_map, probability_column_map = assign_state_labels(
        aligned_df, cluster_labels, cluster_probabilities, feature_columns=selected_columns
    )
    metrics = compute_metrics(feature_matrix, cluster_labels, labeled_df)
    native_importance = compute_gmm_native_feature_importance(model, selected_columns)

    saved_model_path = save_gmm_model(model, model_path=model_path) if save_trained_model and fitted_model is None else None

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
        "native_importance": native_importance,
        "bic_aic_plot_path": bic_aic_plot,
        "model_path": saved_model_path,
    }
