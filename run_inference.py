from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.cohort_benchmarks import benchmark_against_cohort
from src.gmm import run_gmm_pipeline
from src.hmm_model import run_hmm_pipeline, run_soft_probability_hmm
from src.kmeans import run_kmeans_pipeline
from src.preprocessing import CLUSTER_FEATURE_COLUMNS, preprocess_for_modeling, load_scaler
from src.recommendation_engine import generate_recommendations
from src.risk_scoring import build_user_state_summary, calculate_risk_score, evaluate_clinical_persistence

DEFAULT_OUTPUT_PATH = Path("outputs") / "final_results.csv"
MODEL_OPTIONS = {"gmm": "GMM + HMM", "kmeans": "KMeans + HMM"}
STANDARD_STATES = ["Recovery", "Baseline", "Strain"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run wearable health analytics inference on a single-user CSV.")
    parser.add_argument("csv_path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--model", choices=sorted(MODEL_OPTIONS.keys()), default="gmm")
    return parser.parse_args()



def _get_cluster_state_column(model_type: str) -> str:
    return "gmm_state_label" if model_type == "gmm" else "cluster_label"


def _get_observation_column(model_type: str) -> str:
    return "gmm_cluster" if model_type == "gmm" else "kmeans_cluster"


def _normalize_state_label(state_label: str) -> str:
    normalized = str(state_label).strip()
    if normalized == "High Stress":
        return "Strain"
    if normalized in STANDARD_STATES:
        return normalized
    return "Baseline"


def _attach_kmeans_probabilities(labeled_df: pd.DataFrame, kmeans_out: dict[str, Any]) -> pd.DataFrame:
    enriched = labeled_df.copy()
    model = kmeans_out["model"]
    feature_matrix = kmeans_out["feature_matrix"]
    cluster_summary = kmeans_out["cluster_summary"].copy()
    if "cluster_label" in cluster_summary.columns:
        cluster_summary["cluster_label"] = cluster_summary["cluster_label"].map(_normalize_state_label)
    if "cluster_label" in enriched.columns:
        enriched["cluster_label"] = enriched["cluster_label"].map(_normalize_state_label)
    label_map = dict(zip(cluster_summary["kmeans_cluster"], cluster_summary["cluster_label"]))
    distances = model.transform(feature_matrix)
    similarity = np.exp(-distances)
    cluster_probabilities = similarity / similarity.sum(axis=1, keepdims=True)
    for state_label in STANDARD_STATES:
        column_name = f"prob_{state_label.lower().replace(' ', '_')}"
        if column_name not in enriched.columns:
            enriched[column_name] = 0.0
    for cluster_id, raw_state_label in label_map.items():
        state_label = _normalize_state_label(raw_state_label)
        probability_column = f"prob_{state_label.lower().replace(' ', '_')}"
        if probability_column not in enriched.columns:
            enriched[probability_column] = 0.0
        enriched[probability_column] = np.maximum(enriched[probability_column].to_numpy(dtype=float), cluster_probabilities[:, int(cluster_id)])
    probability_columns = [f"prob_{state.lower().replace(' ', '_')}" for state in STANDARD_STATES if f"prob_{state.lower().replace(' ', '_')}" in enriched.columns]
    enriched["state_confidence"] = enriched[probability_columns].max(axis=1)
    enriched["dominant_probability_state"] = enriched[probability_columns].idxmax(axis=1).str.replace("prob_", "", regex=False).str.replace("_", " ").str.title()
    return enriched


def load_model_outputs(feature_df: pd.DataFrame, model_type: str) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    missing_columns = [column for column in CLUSTER_FEATURE_COLUMNS if column not in feature_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required clustering columns after preprocessing: {', '.join(missing_columns)}")
    scaler = StandardScaler()
    scaled_cluster_df = feature_df.copy()
    scaled_cluster_df[CLUSTER_FEATURE_COLUMNS] = scaler.fit_transform(feature_df[CLUSTER_FEATURE_COLUMNS])
    if model_type == "gmm":
        cluster_out = run_gmm_pipeline(scaled_cluster_df, feature_columns=CLUSTER_FEATURE_COLUMNS)
        labeled_df = cluster_out["labeled_df"].copy()
    elif model_type == "kmeans":
        cluster_out = run_kmeans_pipeline(scaled_cluster_df, feature_columns=CLUSTER_FEATURE_COLUMNS)
        labeled_df = _attach_kmeans_probabilities(cluster_out["labeled_df"], cluster_out)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    hmm_out = run_hmm_pipeline(labeled_df, observation_column=_get_observation_column(model_type), n_components=3)
    soft_hmm = run_soft_probability_hmm(labeled_df, n_components=3)
    hmm_out["soft_hmm"] = soft_hmm
    return feature_df, cluster_out, {**hmm_out, "labeled_df": labeled_df}


def _build_state_probabilities(latest_row: pd.Series) -> dict[str, float]:
    return {
        state: round(float(latest_row.get(f"prob_{state.lower().replace(' ', '_')}", 0.0)) * 100.0, 1)
        for state in STANDARD_STATES
    }


def _build_hmm_state_probabilities(latest_row: pd.Series, hmm_out: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    state_column = hmm_out.get("state_column")
    for state in STANDARD_STATES:
        column = f"{state_column}_{state.lower().replace(' ', '_')}_prob"
        if column in latest_row.index:
            result[state] = round(float(latest_row.get(column, 0.0)) * 100.0, 1)
    return result


def _build_model_feature_influence(model_type: str, cluster_out: dict[str, Any], latest_row: pd.Series) -> dict[str, float]:
    feature_columns = list(cluster_out.get("feature_columns", []))
    if not feature_columns:
        return {}
    if model_type == "gmm" and hasattr(cluster_out.get("model"), "means_"):
        influence = cluster_out["model"].means_[int(latest_row.get("gmm_cluster", 0))]
    elif model_type == "kmeans" and hasattr(cluster_out.get("model"), "cluster_centers_"):
        influence = cluster_out["model"].cluster_centers_[int(latest_row.get("kmeans_cluster", 0))]
    else:
        return {}
    return {feature: round(abs(float(value)), 4) for feature, value in sorted(zip(feature_columns, influence), key=lambda item: abs(item[1]), reverse=True)}


def compute_metrics(model_type: str, cluster_out: dict[str, Any], hmm_out: dict[str, Any]) -> dict[str, Any]:
    soft_hmm = hmm_out.get("soft_hmm", {})
    if model_type == "gmm":
        selection_df = cluster_out.get("selection_df")
        selected_k = int(cluster_out.get("selected_k", 0))
        selected_row = selection_df.loc[selection_df["k"] == selected_k].iloc[0] if selection_df is not None and not selection_df.empty and selected_k in selection_df["k"].tolist() else None
        return {
            "silhouette_score": round(float(cluster_out["metrics"].get("silhouette", 0.0)), 4),
            "davies_bouldin_index": round(float(cluster_out["metrics"].get("davies_bouldin", 0.0)), 4),
            "transition_rate": round(float(hmm_out.get("transition_rate", 0.0)), 4),
            "risk_monotonicity": bool(cluster_out["metrics"].get("risk_monotonicity", False)),
            "bic": round(float(selected_row["bic"]), 4) if selected_row is not None else None,
            "aic": round(float(selected_row["aic"]), 4) if selected_row is not None else None,
            "soft_hmm_entropy_bits": soft_hmm.get("transition_entropy_bits"),
        }
    return {
        "silhouette_score": round(float(cluster_out["metrics"].get("silhouette_score", 0.0)), 4),
        "davies_bouldin_index": round(float(cluster_out["metrics"].get("davies_bouldin_index", 0.0)), 4),
        "transition_rate": round(float(hmm_out.get("transition_rate", 0.0)), 4),
        "risk_monotonicity": bool(cluster_out["metrics"].get("risk_monotonicity", False)),
        "bic": None,
        "aic": None,
        "soft_hmm_entropy_bits": soft_hmm.get("transition_entropy_bits"),
    }


def compute_state(model_type: str, merged_df: pd.DataFrame, hmm_out: dict[str, Any], environment: dict[str, Any], cluster_out: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    observation_column = _get_observation_column(model_type)
    state_column = _get_cluster_state_column(model_type)
    duration_column = hmm_out["duration_column"]
    hmm_label_column = f"{hmm_out['state_column']}_label"
    def apply_risk(row: pd.Series) -> pd.Series:
        risk = calculate_risk_score(row, state=str(row.get(state_column, "Unknown")), state_confidence=float(row.get("state_confidence", 0.0)), state_duration=int(row.get(duration_column, 1)))
        stress_level = "Low" if str(row.get(state_column, "Baseline")) == "Recovery" else "Moderate" if str(row.get(state_column, "Baseline")) == "Baseline" else "High"
        return pd.Series({
            "risk_score": risk["score"],
            "risk_level": risk["level"],
            "stress_level_standardized": stress_level,
        })
    risk_df = merged_df.apply(apply_risk, axis=1)
    final_results_df = pd.concat([merged_df.reset_index(drop=True), risk_df.reset_index(drop=True)], axis=1)
    latest_row = final_results_df.sort_values(["user_id", "date"]).reset_index(drop=True).iloc[-1]
    probabilities = _build_state_probabilities(latest_row)
    hmm_probabilities = _build_hmm_state_probabilities(latest_row, hmm_out)
    risk = calculate_risk_score(latest_row, state=str(latest_row.get(state_column, "Unknown")), state_confidence=float(latest_row.get("state_confidence", 0.0)), state_duration=int(latest_row.get(duration_column, 1)))
    clinical_persistence = evaluate_clinical_persistence(final_results_df, user_id=str(latest_row.get("user_id")))

    cohort_benchmarks = benchmark_against_cohort(
        latest_row.to_dict(),
        age=latest_row.get("age", 35),
        gender=str(latest_row.get("gender", "unknown")),
    )
    ordered = final_results_df.sort_values(["user_id", "date"]).reset_index(drop=True)
    previous_state = str(ordered.iloc[-2].get(state_column, latest_row.get(state_column, "Unknown"))) if len(ordered) > 1 else str(latest_row.get(state_column, "Unknown"))
    temporal_trend = "Stable" if previous_state == str(latest_row.get(state_column, "Unknown")) else "Improving" if previous_state == "Strain" and str(latest_row.get(state_column, "Unknown")) in {"Baseline", "Recovery"} else "Deteriorating" if previous_state in {"Recovery", "Baseline"} and str(latest_row.get(state_column, "Unknown")) == "Strain" else risk["early_warning"].get("trend", "Stable")
    analysis_result = {
        "user_id": latest_row.get("user_id"),
        "date": pd.to_datetime(latest_row.get("date")).strftime("%Y-%m-%d"),
        "model_type": model_type,
        "model_name": MODEL_OPTIONS[model_type],
        "state": str(latest_row.get(state_column, "Unknown")),
        "previous_state": previous_state,
        "confidence": round(float(latest_row.get("state_confidence", 0.0)) * 100.0, 1),
        "trend": temporal_trend,
        "risk": {"score": risk["score"], "level": risk["level"], "severity_score": risk["severity_score"]},
        "state_duration_days": int(latest_row.get(duration_column, 1)),
        "temporal_state": latest_row.get(hmm_label_column),
        "early_warning": risk["early_warning"],
        "clinical_escalation": clinical_persistence,
        "cohort_benchmarks": cohort_benchmarks,
        "soft_hmm_entropy_bits": (hmm_out.get("soft_hmm") or {}).get("transition_entropy_bits"),
        "recommendation_context": risk["recommendation_context"],
        "environment": environment,
        "environment_impact": risk["environment_impact"],
        "multi_risk": risk.get("multi_risk", {}),
        "risk_intelligence": risk.get("risk_intelligence", {}),
        "user_state_summary": build_user_state_summary(hmm_out.get("state_distribution")),
        "state_probabilities": probabilities,
        "gmm_probabilities": probabilities if model_type == "gmm" else {},
        "hmm_state_probabilities": hmm_probabilities,
        "hmm_confidence": round(float(latest_row.get(f"{hmm_out['state_column']}_confidence", 0.0)) * 100.0, 1),
        "model_feature_influence": _build_model_feature_influence(model_type, cluster_out, latest_row),
        "cluster_observation_column": observation_column,
        "stress_level": latest_row.get("stress_level_standardized", "Moderate"),
    }
    return final_results_df, analysis_result


def run_pipeline(csv_path: str, city: str, output_path: str, model_type: str = "gmm") -> dict[str, Any]:
    outputs = preprocess_for_modeling(source=csv_path)
    feature_df = outputs["feature_df"].copy()
    feature_df, cluster_out, hmm_bundle = load_model_outputs(feature_df, model_type=model_type)
    decoded_df = hmm_bundle["decoded_df"]
    merge_columns = ["user_id", "date", hmm_bundle["state_column"], f"{hmm_bundle['state_column']}_label", hmm_bundle["duration_column"]]
    merge_columns.extend(list(hmm_bundle.get("named_probability_columns", [])))
    confidence_column = f"{hmm_bundle['state_column']}_confidence"
    if confidence_column in decoded_df.columns:
        merge_columns.append(confidence_column)
    merge_columns = [column for column in merge_columns if column in decoded_df.columns]
    merged_df = hmm_bundle["labeled_df"].merge(
        decoded_df[merge_columns],
        on=["user_id", "date"],
        how="left",
    )
    environment = get_environment(city)
    final_results_df, analysis_result = compute_state(model_type, merged_df, hmm_bundle, environment, cluster_out)
    analysis_result["performance"] = compute_metrics(model_type, cluster_out, hmm_bundle)
    analysis_result["recommendations"] = generate_recommendations(analysis_result)
    analysis_result["transition_rate"] = analysis_result["performance"]["transition_rate"]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    final_results_df.to_csv(output, index=False)
    return {"raw_data": outputs["cleaned_df"], "feature_df": feature_df, "final_results_df": final_results_df, "analysis_result": analysis_result, "output_path": output, "cluster_out": cluster_out, "hmm_out": hmm_bundle, "state_distribution": hmm_bundle["state_distribution"], "transition_matrix": hmm_bundle["transition_matrix"], "cluster_summary": cluster_out["cluster_summary"]}


def main() -> None:
    args = parse_args()
    result = run_pipeline(args.csv_path, city=args.city, output_path=args.output, model_type=args.model)
    print(result["analysis_result"])


if __name__ == "__main__":
    main()






