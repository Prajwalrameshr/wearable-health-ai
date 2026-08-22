from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None

APP_BG = "#0e1117"
CARD_BG = "#151a23"
CARD_BORDER = "#242c3a"
TEXT = "#f3f4f6"
MUTED = "#9aa4b2"
SUCCESS = "#22c55e"
WARNING = "#facc15"
DANGER = "#ef4444"
INFO = "#38bdf8"
STATE_COLORS = {"Recovery": SUCCESS, "Baseline": WARNING, "Strain": DANGER}
PLOT_TEMPLATE = "plotly_dark"


def apply_dashboard_theme() -> None:
    st.set_page_config(page_title="Wearable Health Intelligence", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(f"""
    <style>
    .stApp {{ background: radial-gradient(circle at top right, #18202d 0%, {APP_BG} 45%, {APP_BG} 100%); color:{TEXT}; }}
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {{ display:none; }}
    .block-container {{ max-width: 1400px; padding-top: 1.2rem; padding-bottom: 2rem; }}
    .card {{ background:{CARD_BG}; border:1px solid {CARD_BORDER}; border-radius:22px; padding:1rem 1.1rem; height:100%; }}
    .top-shell {{ background:linear-gradient(135deg, rgba(56,189,248,0.09), rgba(34,197,94,0.05)); border:1px solid {CARD_BORDER}; border-radius:24px; padding:1.2rem 1.3rem; margin-bottom:1rem; }}
    .section-title {{ font-size:1.05rem; margin:0 0 .7rem; }}
    .card-title {{ color:{MUTED}; font-size:.86rem; text-transform:uppercase; letter-spacing:.06em; }}
    .card-value {{ color:{TEXT}; font-size:1.7rem; font-weight:700; margin:.2rem 0; }}
    .card-subtitle {{ color:{MUTED}; font-size:.92rem; }}
    .risk-row {{ padding:.55rem 0; border-bottom:1px solid {CARD_BORDER}; }}
    </style>
    """, unsafe_allow_html=True)


def status_color(value: str) -> str:
    if value in {"Recovery", "Good", "Low", "Improving", "Stable"}:
        return SUCCESS
    if value in {"Baseline", "Moderate", "Warning"}:
        return WARNING
    return DANGER


def render_app_header(model_name: str, page_name: str) -> None:
    st.markdown(f"<div class='top-shell'><h1>Wearable Health Intelligence</h1><p>Active model: {model_name} | {page_name} view</p></div>", unsafe_allow_html=True)


def render_card(title: str, value: str, subtitle: str = "", accent: str | None = None) -> None:
    border = accent or CARD_BORDER
    st.markdown(f"<div class='card' style='border-color:{border}; box-shadow:0 0 0 1px {border}22 inset;'><div class='card-title'>{title}</div><div class='card-value'>{value}</div><div class='card-subtitle'>{subtitle}</div></div>", unsafe_allow_html=True)


def compute_summary_stats(analysis: dict[str, Any], final_results_df: pd.DataFrame | None) -> dict[str, Any]:
    if final_results_df is None or final_results_df.empty:
        return {"health_score": max(0.0, 100.0 - float(analysis.get("risk", {}).get("score", 0.0))), "latest": None, "baseline": None}
    ordered = final_results_df.sort_values(["user_id", "date"]).reset_index(drop=True)
    latest = ordered.iloc[-1]
    baseline = ordered[[column for column in ["resting_hr_bpm", "hrv_rmssd_ms", "sleep_duration_hours", "steps", "spo2_avg_pct"] if column in ordered.columns]].mean(numeric_only=True)
    return {"health_score": round(max(0.0, min(100.0, 100.0 - float(analysis.get("risk", {}).get("score", 0.0)))), 1), "latest": latest, "baseline": baseline}


def build_daily_insight(analysis: dict[str, Any], latest: pd.Series | None, baseline: pd.Series | None) -> str:
    if latest is None or baseline is None:
        return f"Current state is {analysis.get('state', 'Unknown')} with a {analysis.get('trend', 'Stable').lower()} trend."
    if analysis.get("state") == "Recovery":
        return f"Recovery is leading today, with HRV at {float(latest.get('hrv_rmssd_ms', 0.0)):.1f} ms and sleep at {float(latest.get('sleep_duration_hours', 0.0)):.1f} h."
    if analysis.get("state") == "Strain":
        return "Strain is elevated and recovery inputs should be prioritized immediately."
    return "Baseline physiology is holding, but the trend still deserves monitoring." 


def plot_risk_gauge(score: float) -> go.Figure:
    fig = go.Figure(go.Indicator(mode="gauge+number", value=float(score), title={"text": "Risk Gauge"}, gauge={"axis": {"range": [0, 100]}, "bar": {"color": DANGER}, "steps": [{"range": [0, 35], "color": "rgba(34,197,94,.28)"}, {"range": [35, 65], "color": "rgba(250,204,21,.28)"}, {"range": [65, 100], "color": "rgba(239,68,68,.28)"}] }))
    fig.update_layout(height=280, template=PLOT_TEMPLATE, paper_bgcolor=APP_BG, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def plot_trends(final_results_df: pd.DataFrame | None) -> go.Figure | None:
    if final_results_df is None or final_results_df.empty:
        return None
    ordered = final_results_df.sort_values(["user_id", "date"]).copy()
    metrics = [("hrv_rmssd_ms", "HRV", SUCCESS), ("resting_hr_bpm", "Resting HR", DANGER), ("sleep_duration_hours", "Sleep", WARNING)]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07, subplot_titles=[label for _, label, _ in metrics])
    row = 1
    for column, label, color in metrics:
        if column in ordered.columns:
            fig.add_trace(go.Scatter(x=ordered["date"], y=ordered[column], mode="lines+markers", line=dict(color=color, width=2.5), marker=dict(size=6), hovertemplate=f"%{{x|%Y-%m-%d}}<br>{label}: %{{y:.2f}}<extra></extra>"), row=row, col=1)
        row += 1
    fig.update_layout(height=620, template=PLOT_TEMPLATE, paper_bgcolor=APP_BG, plot_bgcolor=CARD_BG, margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
    return fig


def plot_baseline_comparison(final_results_df: pd.DataFrame | None) -> go.Figure | None:
    if final_results_df is None or final_results_df.empty:
        return None
    ordered = final_results_df.sort_values(["user_id", "date"]).reset_index(drop=True)
    latest = ordered.iloc[-1]
    rows = []
    for column, label in [("hrv_rmssd_ms", "HRV"), ("resting_hr_bpm", "Resting HR"), ("sleep_duration_hours", "Sleep"), ("steps", "Steps")]:
        if column in ordered.columns:
            rows.append({"Metric": label, "Baseline": float(ordered[column].mean()), "Latest": float(latest.get(column, 0.0))})
    if not rows:
        return None
    fig = px.bar(pd.DataFrame(rows).melt(id_vars="Metric", value_vars=["Baseline", "Latest"], var_name="Window", value_name="Value"), x="Metric", y="Value", color="Window", barmode="group", color_discrete_map={"Baseline": INFO, "Latest": WARNING}, template=PLOT_TEMPLATE)
    fig.update_layout(height=360, paper_bgcolor=APP_BG, plot_bgcolor=CARD_BG, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def plot_feature_distribution(final_results_df: pd.DataFrame | None, feature_name: str) -> go.Figure | None:
    if final_results_df is None or final_results_df.empty or feature_name not in final_results_df.columns:
        return None
    fig = px.histogram(final_results_df[[feature_name]].dropna(), x=feature_name, nbins=24, template=PLOT_TEMPLATE, color_discrete_sequence=[INFO])
    fig.update_layout(height=320, paper_bgcolor=APP_BG, plot_bgcolor=CARD_BG, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def plot_state_probabilities(probabilities: dict[str, Any] | None) -> go.Figure | None:
    if not probabilities:
        return None
    prob_df = pd.DataFrame({"State": list(probabilities.keys()), "Probability": list(probabilities.values())})
    fig = px.bar(prob_df, x="Probability", y="State", orientation="h", color="State", color_discrete_map=STATE_COLORS, text="Probability", template=PLOT_TEMPLATE)
    fig.update_traces(texttemplate="%{text:.1f}%", hovertemplate="%{y}: %{x:.1f}%<extra></extra>")
    fig.update_layout(height=300, paper_bgcolor=APP_BG, plot_bgcolor=CARD_BG, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    return fig


def plot_transition_heatmap(transition_matrix: pd.DataFrame | None) -> go.Figure | None:
    if transition_matrix is None or transition_matrix.empty:
        return None
    fig = go.Figure(data=go.Heatmap(z=transition_matrix.astype(float).values, x=list(transition_matrix.columns), y=list(transition_matrix.index), colorscale=[[0.0, "#132117"], [0.5, "#423714"], [1.0, "#401419"]], text=transition_matrix.round(2).astype(str).values, texttemplate="%{text}", hovertemplate="From %{y} to %{x}: %{z:.3f}<extra></extra>"))
    fig.update_layout(height=360, template=PLOT_TEMPLATE, paper_bgcolor=APP_BG, plot_bgcolor=CARD_BG, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def compute_warning_flags(feature_df: pd.DataFrame | None) -> dict[str, Any]:
    if feature_df is None or feature_df.empty:
        return {"hrv_slope": 0.0, "sleep_slope": 0.0, "hrv_alert": False, "sleep_alert": False}
    latest = feature_df.sort_values(["user_id", "date"]).reset_index(drop=True).iloc[-1]
    hrv_slope = float(latest.get("hrv_dev_slope_7d", 0.0))
    sleep_slope = float(latest.get("sleep_dev_slope_7d", 0.0))
    return {"hrv_slope": round(hrv_slope, 3), "sleep_slope": round(sleep_slope, 3), "hrv_alert": hrv_slope < -0.15, "sleep_alert": sleep_slope < -0.1}


def compute_explainability(final_results_df: pd.DataFrame | None, analysis: dict[str, Any], model_type: str) -> dict[str, Any]:
    fallback_reason = "Permutation importance"
    if final_results_df is None or final_results_df.empty:
        return {"method": fallback_reason, "importance_df": pd.DataFrame(columns=["Feature", "Importance"]), "waterfall_df": pd.DataFrame(columns=["Feature", "Contribution"])}
    target_column = "gmm_state_label" if model_type == "gmm" else "cluster_label"
    feature_columns = [column for column in ["resting_hr_bpm", "hrv_rmssd_ms", "sleep_duration_hours", "steps", "spo2_avg_pct", "severity_score", "hr_dev", "hrv_dev", "sleep_dev"] if column in final_results_df.columns]
    if target_column in final_results_df.columns and len(feature_columns) >= 3:
        explain_df = final_results_df[feature_columns + [target_column]].dropna().copy()
        if not explain_df.empty and explain_df[target_column].nunique() >= 2:
            X = explain_df[feature_columns]
            y = explain_df[target_column]
            model = RandomForestClassifier(n_estimators=220, max_depth=6, random_state=42)
            model.fit(X, y)
            latest_features = X.iloc[[-1]]
            if shap is not None:
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X)
                    shap_array = np.array(shap_values[0] if isinstance(shap_values, list) else shap_values)
                    if shap_array.ndim == 3:
                        shap_array = shap_array[0]
                    importance_df = pd.DataFrame({"Feature": feature_columns, "Importance": np.abs(shap_array).mean(axis=0)}).sort_values("Importance", ascending=False)
                    waterfall_df = pd.DataFrame({"Feature": feature_columns, "Contribution": shap_array[-1]}).sort_values("Contribution", key=lambda s: s.abs(), ascending=False)
                    return {"method": "SHAP TreeExplainer", "importance_df": importance_df, "waterfall_df": waterfall_df}
                except Exception:
                    pass
            perm = permutation_importance(model, X, y, n_repeats=5, random_state=42)
            importance_df = pd.DataFrame({"Feature": feature_columns, "Importance": perm.importances_mean}).sort_values("Importance", ascending=False)
            waterfall_df = pd.DataFrame({"Feature": feature_columns, "Contribution": latest_features.iloc[0].values - X.mean().values}).sort_values("Contribution", key=lambda s: s.abs(), ascending=False)
            return {"method": fallback_reason, "importance_df": importance_df, "waterfall_df": waterfall_df}
    model_influence = analysis.get("model_feature_influence") or {}
    importance_df = pd.DataFrame({"Feature": list(model_influence.keys()), "Importance": list(model_influence.values())}).sort_values("Importance", ascending=False) if model_influence else pd.DataFrame(columns=["Feature", "Importance"])
    waterfall_df = pd.DataFrame({"Feature": importance_df.get("Feature", []), "Contribution": importance_df.get("Importance", [])})
    return {"method": "Model center influence", "importance_df": importance_df, "waterfall_df": waterfall_df}


def plot_explainability_bars(explainability: dict[str, Any]) -> go.Figure | None:
    importance_df = explainability.get("importance_df")
    if importance_df is None or importance_df.empty:
        return None
    plot_df = importance_df.head(5).sort_values("Importance", ascending=True)
    fig = px.bar(plot_df, x="Importance", y="Feature", orientation="h", template=PLOT_TEMPLATE, color_discrete_sequence=[INFO])
    fig.update_layout(height=320, paper_bgcolor=APP_BG, plot_bgcolor=CARD_BG, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def plot_waterfall_fallback(explainability: dict[str, Any]) -> go.Figure | None:
    waterfall_df = explainability.get("waterfall_df")
    if waterfall_df is None or waterfall_df.empty:
        return None
    plot_df = waterfall_df.head(5).iloc[::-1]
    colors = [SUCCESS if value >= 0 else DANGER for value in plot_df["Contribution"]]
    fig = go.Figure(go.Bar(x=plot_df["Contribution"], y=plot_df["Feature"], orientation="h", marker_color=colors))
    fig.update_layout(height=300, template=PLOT_TEMPLATE, paper_bgcolor=APP_BG, plot_bgcolor=CARD_BG, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def render_environment_cards(environment: dict[str, Any]) -> None:
    c1, c2, c3 = st.columns(3)
    with c1:
        render_card("AQI", str(environment.get("aqi", "-")), str(environment.get("weather", "Unknown")), status_color("Low" if float(environment.get("aqi", 1)) <= 2 else "Moderate" if float(environment.get("aqi", 1)) <= 3 else "High"))
    with c2:
        render_card("Temperature", f"{float(environment.get('temperature', 0.0)):.1f} C", f"Feels like {float(environment.get('feels_like', 0.0)):.1f} C", INFO)
    with c3:
        render_card("Humidity", f"{float(environment.get('humidity', 0.0)):.0f}%", "Outdoor readiness", WARNING if float(environment.get("humidity", 0.0)) > 75 else SUCCESS)


def build_future_risk_text(analysis: dict[str, Any], transition_matrix: pd.DataFrame | None) -> str:
    if transition_matrix is None or transition_matrix.empty:
        return "Future-state estimate is unavailable because transition history is limited."
    current_state = str(analysis.get("temporal_state") or analysis.get("state") or "")
    if current_state not in transition_matrix.index:
        return "Future-state estimate is unavailable for the current temporal state."
    next_state = str(transition_matrix.loc[current_state].astype(float).idxmax())
    if next_state == current_state:
        return f"You are most likely to remain in {current_state} if the current pattern continues."
    return f"You may move to {next_state} if the current pattern continues."


def render_clinical_advisory_card(clinical_escalation: dict[str, Any] | None) -> None:
    if not clinical_escalation:
        return
    level = clinical_escalation.get("advisory_level", "Normal")
    if level == "Normal":
        return
    bg_color = DANGER if level == "Clinical Advisory" else WARNING
    st.markdown(
        f"<div class='card' style='border-color:{bg_color}; background-color:{CARD_BG}; margin-bottom:1rem;'>"
        f"<div class='card-title' style='color:{bg_color}; font-weight:bold;'>⚠️ Clinical Advisory Level: {level.upper()}</div>"
        f"<div style='font-size:1.05rem; margin:.4rem 0;'>{clinical_escalation.get('clinical_summary_message', '')}</div>"
        f"<div class='card-subtitle'>Consecutive Anomaly Days: {clinical_escalation.get('consecutive_high_risk_days', 0)}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def plot_cohort_benchmarks(cohort_data: dict[str, Any] | None) -> go.Figure | None:
    if not cohort_data or "metrics" not in cohort_data:
        return None
    metrics = cohort_data["metrics"]
    rows = []
    for k, v in metrics.items():
        rows.append({"Metric": v["label"], "Percentile": v["percentile"], "Rating": v["rating"]})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    fig = px.bar(
        df,
        x="Percentile",
        y="Metric",
        orientation="h",
        text="Percentile",
        color="Rating",
        color_discrete_map={"Above Average": SUCCESS, "Average": INFO, "Below Average": DANGER},
        template=PLOT_TEMPLATE,
    )
    fig.update_traces(texttemplate="%{text:.1f}%", hovertemplate="%{y}: %{x:.1f}th percentile<extra></extra>")
    fig.update_layout(height=320, paper_bgcolor=APP_BG, plot_bgcolor=CARD_BG, xaxis=dict(range=[0, 100]), margin=dict(l=10, r=10, t=10, b=10))
    return fig

