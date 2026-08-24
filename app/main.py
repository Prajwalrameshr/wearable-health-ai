from __future__ import annotations

import importlib
import inspect
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from app.dashboard_utils import (
    APP_BG,
    CARD_BG,
    DANGER,
    INFO,
    SUCCESS,
    WARNING,
    apply_dashboard_theme,
    build_daily_insight,
    build_future_risk_text,
    compute_explainability,
    compute_summary_stats,
    compute_warning_flags,
    plot_baseline_comparison,
    plot_cohort_benchmarks,
    plot_explainability_bars,
    plot_feature_distribution,
    plot_risk_gauge,
    plot_state_probabilities,
    plot_transition_heatmap,
    plot_trends,
    plot_waterfall_fallback,
    render_app_header,
    render_card,
    render_clinical_advisory_card,
    render_environment_cards,
    status_color,
)
import run_inference as run_inference_module
from src.preprocessing import validate_required_columns
import src.risk_scoring as risk_scoring_module

run_inference_module = importlib.reload(run_inference_module)
risk_scoring_module = importlib.reload(risk_scoring_module)
run_pipeline = run_inference_module.run_pipeline
calculate_risk_score = risk_scoring_module.calculate_risk_score

SAMPLE_FILE = ROOT_DIR / "data" / "wearables_health_6mo_daily.csv"
OUTPUT_DIR = ROOT_DIR / "outputs"
PAGES = ["Dashboard", "Analysis", "Recommendations"]
MODEL_LABELS = {"KMeans + HMM": "kmeans", "GMM + HMM": "gmm"}
CACHE_SCHEMA_VERSION = "v4"
SESSION_DEFAULTS = {"active_df": None, "model_cache": {}, "last_data_key": None}
RISK_CARD_ORDER = [
    ("cardiovascular_strain", "Cardiovascular Strain"),
    ("sleep_deficit", "Sleep Deficit Risk"),
    ("chronic_stress", "Chronic Stress Risk"),
    ("recovery_failure", "Recovery Failure Risk"),
    ("overtraining", "Overtraining Risk"),
    ("fatigue_accumulation", "Fatigue Accumulation"),
    ("burnout", "Burnout Risk"),
    ("circadian_disruption", "Circadian Disruption"),
    ("metabolic_stress", "Metabolic Stress"),
    ("autonomic_imbalance", "Autonomic Imbalance"),
]


def init_state() -> None:
    for key, value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def dataframe_key(df: pd.DataFrame | None) -> str | None:
    if df is None or df.empty:
        return None
    return f"{len(df)}-{len(df.columns)}-{int(pd.util.hash_pandas_object(df.fillna('__nan__'), index=True).sum())}"


def load_model(dataframe: pd.DataFrame, model_type: str) -> dict[str, Any]:
    data_key = dataframe_key(dataframe)
    cache_key = f"{CACHE_SCHEMA_VERSION}:{model_type}:{data_key}"
    cache = st.session_state["model_cache"]
    if cache_key in cache:
        return cache[cache_key]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
        dataframe.to_csv(temp_file.name, index=False)
        temp_path = temp_file.name

    result = run_pipeline(
        csv_path=temp_path,
        output_path=str(OUTPUT_DIR / f"streamlit_{model_type}_results.csv"),
        model_type=model_type,
    )
    cache[cache_key] = result
    st.session_state["model_cache"] = cache
    st.session_state["last_data_key"] = data_key
    return result


def get_latest_baseline(final_results_df: pd.DataFrame | None) -> tuple[pd.Series | None, pd.Series | None]:
    if final_results_df is None or final_results_df.empty:
        return None, None
    ordered = final_results_df.sort_values(["user_id", "date"]).reset_index(drop=True)
    latest = ordered.iloc[-1]
    baseline = ordered[[column for column in ["resting_hr_bpm", "hrv_rmssd_ms", "sleep_duration_hours", "steps", "spo2_avg_pct"] if column in ordered.columns]].mean(numeric_only=True)
    return latest, baseline


def render_upload_controls() -> tuple[pd.DataFrame | None, bool]:
    upload_col, action_col = st.columns([2.0, 1.0])
    with upload_col:
        uploaded_file = st.file_uploader("Upload wearable CSV", type=["csv"])
    with action_col:
        st.write("")
        use_sample = st.button("Use Sample Data", use_container_width=True)
        run_clicked = st.button("Run Selected Model", type="primary", use_container_width=True)
    source_df = None
    if uploaded_file is not None:
        source_df = pd.read_csv(uploaded_file)
    elif use_sample:
        source_df = pd.read_csv(SAMPLE_FILE)
    if source_df is not None:
        missing = validate_required_columns(source_df)
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
            st.session_state["active_df"] = None
        else:
            st.session_state["active_df"] = source_df.copy()
            st.success("Dataset loaded. Rolling baselines, causal normalization, and temporal features ready.")
    return st.session_state.get("active_df"), run_clicked


def build_personalized_actions(analysis: dict[str, Any], latest: pd.Series | None, baseline: pd.Series | None) -> list[str]:
    if latest is None:
        return ["Continue monitoring your wearable trends while keeping your routine balanced."]
    actions: list[str] = []
    if float(latest.get("sleep_duration_hours", 0.0)) < 7:
        actions.append("Increase sleep opportunity tonight to improve next-day recovery capacity.")
    if baseline is not None and float(latest.get("hrv_rmssd_ms", 0.0)) < float(baseline.get("hrv_rmssd_ms", 0.0)):
        actions.append("Reduce high-intensity effort because HRV is below your usual baseline.")
    if baseline is not None and float(latest.get("resting_hr_bpm", 0.0)) > float(baseline.get("resting_hr_bpm", 0.0)):
        actions.append("Keep workload moderate because resting heart rate is elevated versus baseline.")
    if float(latest.get("steps", 0.0)) < 4000:
        actions.append("Add gentle walking or mobility instead of full inactivity.")
    if analysis.get("trend") == "Deteriorating":
        actions.append("Your temporal trend is deteriorating, so prioritize stabilization in the next 24 hours.")
    return actions or ["Your signals look stable, so consistency matters more than large changes today."]


def render_metric_cards(analysis: dict[str, Any], final_results_df: pd.DataFrame | None) -> None:
    summary = compute_summary_stats(analysis, final_results_df)
    latest = summary["latest"]
    sleep_quality = "Good" if latest is not None and float(latest.get("sleep_duration_hours", 0.0)) >= 7 else "Warning" if latest is not None and float(latest.get("sleep_duration_hours", 0.0)) >= 6 else "Risk"
    cols = st.columns(4)
    with cols[0]:
        render_card("Health Score", f"{summary['health_score']:.0f}", "Composite recovery outlook", SUCCESS if summary["health_score"] >= 70 else WARNING if summary["health_score"] >= 45 else DANGER)
    with cols[1]:
        render_card("Sleep Quality", sleep_quality, "Latest sleep status", status_color(sleep_quality))
    with cols[2]:
        render_card("Recovery Level", str(analysis.get("state", "Unknown")), "Standardized physiological state", status_color(str(analysis.get("state", "Unknown"))))
    with cols[3]:
        render_card("Stress Level", str(analysis.get("stress_level", analysis.get("risk", {}).get("level", "Unknown"))), "State-aligned stress category", status_color(str(analysis.get("stress_level", "Moderate"))))


def _build_display_risk_map(analysis: dict[str, Any], latest: pd.Series | None = None) -> dict[str, dict[str, Any]]:
    risk_map = analysis.get("risk_intelligence") or {}
    multi_risk = analysis.get("multi_risk") or {}

    normalized: dict[str, dict[str, Any]] = {}
    for key, label in RISK_CARD_ORDER:
        payload = risk_map.get(key) or {}
        score = payload.get("score")
        if score is None:
            score = multi_risk.get(label, 0.0)
        score = float(score or 0.0)
        level = payload.get("level")
        if level is None:
            level = "High" if score >= 70 else "Moderate" if score >= 40 else "Low"
        normalized[key] = {"label": label, "score": score, "level": level}

    all_zero = all(float(item["score"]) == 0.0 for item in normalized.values())
    if all_zero and latest is not None:
        fallback_risk = calculate_risk_score(
            latest,
            state=str(analysis.get("state", "Baseline")),
            state_confidence=float(analysis.get("confidence", 0.0)) / 100.0,
            state_duration=int(analysis.get("state_duration_days", 1)),
        )
        fallback_map = fallback_risk.get("risk_intelligence") or {}
        for key, label in RISK_CARD_ORDER:
            payload = fallback_map.get(key) or {}
            score = float(payload.get("score", 0.0))
            level = payload.get("level") or ("High" if score >= 70 else "Moderate" if score >= 40 else "Low")
            normalized[key] = {"label": label, "score": score, "level": level}

    return normalized


def render_risk_intelligence(analysis: dict[str, Any], latest: pd.Series | None = None) -> None:
    risk_map = _build_display_risk_map(analysis, latest=latest)
    st.markdown("<div class='section-title'>Physiological Strain Intelligence</div>", unsafe_allow_html=True)
    for index in range(0, len(RISK_CARD_ORDER), 2):
        row_keys = RISK_CARD_ORDER[index : index + 2]
        cols = st.columns(2)
        for col, (risk_key, _) in zip(cols, row_keys):
            payload = risk_map[risk_key]
            with col:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(f"<div class='card-title'>{payload['label']}</div>", unsafe_allow_html=True)
                risk_progress = min(max(int(payload['score']), 0), 100)
                st.progress(risk_progress)
                st.caption(f"{payload['score']:.1f}/100 | {payload['level']}")
                st.markdown("</div>", unsafe_allow_html=True)


def render_what_if_simulation(analysis: dict[str, Any], latest: pd.Series | None) -> None:
    st.markdown("<div class='section-title'>What If Simulation</div>", unsafe_allow_html=True)
    if latest is None:
        st.warning("Simulation requires at least one valid latest record.")
        return

    sim_col, result_col = st.columns([1.0, 1.0])
    with sim_col:
        sleep_hours = st.slider("Sleep hours", 3.0, 10.0, float(latest.get("sleep_duration_hours", 7.0)), 0.5)
        hrv = st.slider("HRV", 10.0, 100.0, float(latest.get("hrv_rmssd_ms", 50.0)), 1.0)
        resting_hr = st.slider("Resting heart rate", 45.0, 95.0, float(latest.get("resting_hr_bpm", 65.0)), 1.0)
        activity = st.slider("Activity level", 1000, 18000, int(latest.get("steps", 6000)), 250)

    sim_row = latest.copy()
    sim_row["sleep_duration_hours"] = sleep_hours
    sim_row["hrv_rmssd_ms"] = hrv
    sim_row["resting_hr_bpm"] = resting_hr
    sim_row["steps"] = activity
    sim_row["sleep_dev"] = float(latest.get("sleep_dev", 0.0)) + (sleep_hours - float(latest.get("sleep_duration_hours", sleep_hours)))
    sim_row["hrv_dev"] = float(latest.get("hrv_dev", 0.0)) + (hrv - float(latest.get("hrv_rmssd_ms", hrv))) / 10.0
    sim_row["hr_dev"] = float(latest.get("hr_dev", 0.0)) + (resting_hr - float(latest.get("resting_hr_bpm", resting_hr))) / 5.0
    sim_row["severity_score"] = abs(float(sim_row.get("hr_dev", 0.0))) + abs(float(sim_row.get("hrv_dev", 0.0))) + abs(float(sim_row.get("sleep_dev", 0.0)))

    sim_sev = float(sim_row.get("severity_score", 0.0))
    sim_state = "Recovery" if sim_sev < 1.8 else "Strain" if sim_sev > 3.5 else "Baseline"
    sim_risk = calculate_risk_score(sim_row, state=sim_state, state_confidence=0.85, state_duration=1)

    with result_col:
        render_card("Simulated State", sim_state, "Updated from your slider inputs", status_color(sim_state))
        render_card("Simulated Strain", f"{float(sim_risk.get('score', 0.0)):.1f}", f"Level: {sim_risk.get('level', 'Unknown')}", status_color(str(sim_risk.get("level", "Moderate"))))

        original_sleep_risk = next(
            (item.get("score") for item in (analysis.get("risk_intelligence") or {}).values() if item.get("label") == "Sleep Deficit Risk"),
            None,
        )
        simulated_risk_map = sim_risk.get("risk_intelligence", {}) or {}
        new_sleep_risk = (simulated_risk_map.get("sleep_deficit") or {}).get("score")

        if original_sleep_risk is not None and new_sleep_risk is not None:
            delta = float(original_sleep_risk) - float(new_sleep_risk)
            st.info(f"If sleep changes to {sleep_hours:.1f} h, sleep risk changes by {delta:.1f} points and predicted state becomes {sim_state}.")
        else:
            st.info(f"If these inputs hold, predicted state becomes {sim_state} with estimated strain index of {float(sim_risk.get('score', 0.0)):.1f}.")

    st.markdown("<div class='section-title'>Simulated Strain Intelligence</div>", unsafe_allow_html=True)
    render_risk_intelligence({"risk_intelligence": sim_risk.get("risk_intelligence", {}), "multi_risk": sim_risk.get("multi_risk", {}), "state": sim_state, "confidence": 85.0, "state_duration_days": 1}, latest=sim_row)


def render_dashboard_page(analysis: dict[str, Any], result: dict[str, Any]) -> None:
    render_clinical_advisory_card(analysis.get("clinical_escalation"))
    final_results_df = result.get("final_results_df")
    latest, baseline = get_latest_baseline(final_results_df)
    top = st.columns(3)
    with top[0]:
        render_card("Current State", str(analysis.get("state", "Unknown")), f"Previous: {analysis.get('previous_state', 'Unknown')}", status_color(str(analysis.get("state", "Unknown"))))
    with top[1]:
        render_card("Confidence", f"{float(analysis.get('confidence', 0.0)):.1f}%", f"Active model: {analysis.get('model_name', 'Unknown')}", INFO)
    with top[2]:
        render_card("Transition Trend", str(analysis.get("trend", "Stable")), f"Temporal state: {analysis.get('temporal_state', 'Unknown')}", status_color(str(analysis.get("trend", "Stable"))))
    render_metric_cards(analysis, final_results_df)
    left, right = st.columns([1.15, 0.85])
    with left:
        st.markdown("<div class='card'><div class='section-title'>Daily Insight</div>", unsafe_allow_html=True)
        st.write(build_daily_insight(analysis, latest, baseline))
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='card'><div class='section-title'>Strain Index Gauge</div>", unsafe_allow_html=True)
        st.plotly_chart(plot_risk_gauge(float(analysis.get("risk", {}).get("score", 0.0))), use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)


def render_analysis_page(analysis: dict[str, Any], result: dict[str, Any], model_type: str) -> None:
    final_results_df = result.get("final_results_df")
    feature_df = result.get("feature_df")
    transition_matrix = result.get("transition_matrix")
    metrics = analysis.get("performance", {}) or {}

    metric_items = [
        ("Silhouette Score", f"{float(metrics.get('silhouette_score', 0.0)):.4f}", INFO),
        ("Davies-Bouldin Index", f"{float(metrics.get('davies_bouldin_index', 0.0)):.4f}", WARNING),
        ("Transition Rate", f"{float(metrics.get('transition_rate', 0.0)):.4f}", SUCCESS),
        ("Risk Monotonicity", "Yes" if metrics.get("risk_monotonicity") else "No", SUCCESS if metrics.get("risk_monotonicity") else DANGER),
    ]
    if model_type == "gmm":
        metric_items.extend([
            ("AIC", f"{float(metrics.get('aic', 0.0)):.2f}" if metrics.get("aic") is not None else "N/A", INFO),
            ("BIC", f"{float(metrics.get('bic', 0.0)):.2f}" if metrics.get("bic") is not None else "N/A", INFO),
        ])

    cols = st.columns(len(metric_items))
    for col, (label, value, accent) in zip(cols, metric_items):
        with col:
            render_card(label, value, "Model evaluation", accent)

    row1 = st.columns([1.2, 0.8])
    with row1[0]:
        st.markdown("<div class='card'><div class='section-title'>Time-Series Trends</div>", unsafe_allow_html=True)
        fig = plot_trends(final_results_df)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Time-series trends unavailable.")
        st.markdown("</div>", unsafe_allow_html=True)
    with row1[1]:
        st.markdown("<div class='card'><div class='section-title'>Baseline vs Current</div>", unsafe_allow_html=True)
        fig = plot_baseline_comparison(final_results_df)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Baseline comparison unavailable.")
        st.markdown("</div>", unsafe_allow_html=True)

    row2 = st.columns(2)
    with row2[0]:
        st.markdown("<div class='card'><div class='section-title'>Feature Distribution</div>", unsafe_allow_html=True)
        available = [column for column in ["hrv_rmssd_ms", "resting_hr_bpm", "sleep_duration_hours", "steps", "severity_score"] if final_results_df is not None and column in final_results_df.columns]
        selected = st.selectbox("Select feature", available, key=f"feature_distribution_{model_type}") if available else None
        if selected:
            fig = plot_feature_distribution(final_results_df, selected)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Distribution unavailable.")
        else:
            st.warning("No feature distribution data available.")
        st.markdown("</div>", unsafe_allow_html=True)
    with row2[1]:
        title = "GMM Probabilities" if model_type == "gmm" else "Cluster State Probabilities"
        st.markdown(f"<div class='card'><div class='section-title'>{title}</div>", unsafe_allow_html=True)
        fig = plot_state_probabilities(analysis.get("gmm_probabilities") or analysis.get("state_probabilities"))
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Probabilities unavailable.")
        st.markdown("</div>", unsafe_allow_html=True)

    row3 = st.columns([0.85, 1.15])
    with row3[0]:
        st.markdown("<div class='card'><div class='section-title'>HMM Probabilistic States</div>", unsafe_allow_html=True)
        fig = plot_state_probabilities(analysis.get("hmm_state_probabilities"))
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Temporal confidence: {float(analysis.get('hmm_confidence', 0.0)):.1f}%")
        else:
            st.warning("HMM posterior probabilities unavailable.")
        st.markdown("</div>", unsafe_allow_html=True)
    with row3[1]:
        st.markdown("<div class='card'><div class='section-title'>HMM Transition Matrix</div>", unsafe_allow_html=True)
        fig = plot_transition_heatmap(transition_matrix)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Transition matrix unavailable.")
        st.markdown("</div>", unsafe_allow_html=True)

    row4 = st.columns([0.8, 1.2])
    warning_flags = compute_warning_flags(feature_df)
    with row4[0]:
        st.markdown("<div class='card'><div class='section-title'>Early Warning System</div>", unsafe_allow_html=True)
        render_card("HRV Slope", f"{warning_flags['hrv_slope']:.3f}", "7-day slope", DANGER if warning_flags["hrv_alert"] else SUCCESS)
        render_card("Sleep Slope", f"{warning_flags['sleep_slope']:.3f}", "7-day slope", DANGER if warning_flags["sleep_alert"] else SUCCESS)
        st.markdown("</div>", unsafe_allow_html=True)
    explainability = compute_explainability(final_results_df, analysis, model_type)
    with row4[1]:
        st.markdown(f"<div class='card'><div class='section-title'>Surrogate-Model SHAP Explainability</div><p style='color:#9aa4b2'>Method: {explainability['method']}</p>", unsafe_allow_html=True)
        if explainability.get("fidelity"):
            fid = explainability["fidelity"]
            st.caption(f"Surrogate Fidelity: Acc = {fid['accuracy']}% | Bal Acc = {fid['balanced_accuracy']}% | Macro F1 = {fid['macro_f1']}")
        fig = plot_explainability_bars(explainability)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Explainability unavailable.")
        waterfall = plot_waterfall_fallback(explainability)
        if waterfall is not None:
            st.plotly_chart(waterfall, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    cohort_data = analysis.get("cohort_benchmarks")
    if cohort_data:
        st.markdown(f"<div class='card'><div class='section-title'>Demographic Cohort Benchmarking (Peer Group: Age {cohort_data.get('age_group')}, Gender {cohort_data.get('gender', 'unknown').title()})</div>", unsafe_allow_html=True)
        fig_cohort = plot_cohort_benchmarks(cohort_data)
        if fig_cohort is not None:
            st.plotly_chart(fig_cohort, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    latest, _ = get_latest_baseline(final_results_df)
    render_risk_intelligence(analysis, latest=latest)
    render_what_if_simulation(analysis, latest)


def render_recommendations_page(analysis: dict[str, Any], result: dict[str, Any]) -> None:
    final_results_df = result.get("final_results_df")
    latest, baseline = get_latest_baseline(final_results_df)
    top_col, future_col = st.columns([1.15, 0.85])
    with top_col:
        render_card("Top Recommendation", str((analysis.get("recommendations") or ["Maintain a balanced routine."])[0]), "Highest-priority action", status_color(str(analysis.get("state", "Baseline"))))
    with future_col:
        render_card("Future State Prediction", build_future_risk_text(analysis, result.get("transition_matrix")), "HMM transition outlook", WARNING)
    left, right = st.columns([1.0, 1.0])
    with left:
        st.markdown("<div class='card'><div class='section-title'>Personalized Actions</div>", unsafe_allow_html=True)
        for action in build_personalized_actions(analysis, latest, baseline):
            st.markdown(f"- {action}")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='card'><div class='section-title'>Physiological Metrics Summary</div>", unsafe_allow_html=True)
        render_environment_cards(latest if latest is not None else {})
        st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    apply_dashboard_theme()
    init_state()
    nav_col, model_col = st.columns([1.35, 1.0])
    with nav_col:
        page = st.radio("Navigation", PAGES, horizontal=True, label_visibility="collapsed")
    with model_col:
        selected_model_label = st.radio("Model", ["KMeans + HMM", "GMM + HMM"], horizontal=True)
    selected_model = MODEL_LABELS[selected_model_label]
    render_app_header(selected_model_label, page)
    active_df, run_clicked = render_upload_controls()
    if active_df is None:
        st.info("Upload a CSV or use the sample dataset to activate the dashboard.")
        return
    if run_clicked or dataframe_key(active_df) != st.session_state.get("last_data_key"):
        with st.spinner(f"Running {selected_model_label} pipeline..."):
            result = load_model(active_df, selected_model)
    else:
        cache_key = f"{CACHE_SCHEMA_VERSION}:{selected_model}:{dataframe_key(active_df)}"
        result = st.session_state["model_cache"].get(cache_key) or load_model(active_df, selected_model)
    analysis = result.get("analysis_result", {})
    output_csv = result.get("final_results_df")
    topbar_left, topbar_right = st.columns([1.1, 0.9])
    with topbar_left:
        st.caption(f"Active model: {analysis.get('model_name', selected_model_label)} | Processed date: {analysis.get('date', 'N/A')}")
    with topbar_right:
        if output_csv is not None and not output_csv.empty:
            st.download_button("Download Results CSV", data=output_csv.to_csv(index=False).encode("utf-8"), file_name=f"{selected_model}_results.csv", mime="text/csv", use_container_width=True)
    if page == "Dashboard":
        render_dashboard_page(analysis, result)
    elif page == "Analysis":
        render_analysis_page(analysis, result, selected_model)
    else:
        render_recommendations_page(analysis, result)


if __name__ == "__main__":
    main()
