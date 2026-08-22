from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from app.dashboard_utils import (
    apply_dashboard_theme,
    render_baseline_comparison,
    render_early_warning,
    render_gmm_probabilities,
    render_recovery_score_trend,
    render_shap_explainability,
    render_time_series,
    render_transition_heatmap,
)


st.set_page_config(page_title="Analysis", layout="wide")
apply_dashboard_theme()

analysis_result = st.session_state.get("analysis_result")
final_results_df = st.session_state.get("final_results_df")
transition_matrix = st.session_state.get("transition_matrix")
performance = st.session_state.get("performance") or {}
feature_df = st.session_state.get("feature_df")

st.title("Analysis")
st.caption("Deep technical insight into clustering, temporal modeling, and explainability.")

if analysis_result is None or st.session_state.get("data") is None:
    st.warning("Upload dataset first")
else:
    st.subheader("Model Performance")
    p1, p2, p3 = st.columns(3)
    p1.metric("Silhouette Score", f"{float(performance.get('silhouette_score', 0.0)):.4f}")
    p2.metric("Davies-Bouldin Index", f"{float(performance.get('davies_bouldin_index', 0.0)):.4f}")
    p3.metric("Transition Rate", f"{float(performance.get('transition_rate', 0.0)):.4f}")

    st.markdown("---")
    render_time_series(final_results_df)

    st.markdown("---")
    render_baseline_comparison(final_results_df)

    st.markdown("---")
    render_gmm_probabilities(analysis_result.get("gmm_probabilities"))

    st.markdown("---")
    st.subheader("HMM Transition Matrix")
    st.write("Shows probability of moving between physiological states.")
    render_transition_heatmap(transition_matrix)

    st.markdown("---")
    st.subheader("Explainability")
    render_shap_explainability(final_results_df, analysis_result.get("gmm_feature_influence"))

    st.markdown("---")
    left, right = st.columns(2)
    with left:
        st.subheader("Early Warning")
        render_early_warning(analysis_result)
    with right:
        if feature_df is not None and "hrv_dev_slope_7d" in feature_df.columns and "sleep_dev_slope_7d" in feature_df.columns:
            latest_feature_row = feature_df.sort_values(["user_id", "date"]).reset_index(drop=True).iloc[-1]
            st.metric("HRV Slope", f"{float(latest_feature_row.get('hrv_dev_slope_7d', 0.0)):.3f}")
            st.metric("Sleep Slope", f"{float(latest_feature_row.get('sleep_dev_slope_7d', 0.0)):.3f}")
            st.info("Declining HRV slope indicates early physiological stress.")
        else:
            st.warning("Slope features are not available.")

    st.markdown("---")
    render_recovery_score_trend(final_results_df)
