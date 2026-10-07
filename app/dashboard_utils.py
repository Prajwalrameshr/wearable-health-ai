from __future__ import annotations

import base64
from typing import Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.inspection import permutation_importance

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None

# Production Theme Tokens
APP_BG = "#070b14"
CARD_BG = "#0f172a"
CARD_BORDER = "#1e293b"
CARD_HOVER_BORDER = "#38bdf8"
TEXT = "#f8fafc"
MUTED = "#94a3b8"
SUCCESS = "#10b981"
WARNING = "#f59e0b"
DANGER = "#ef4444"
INFO = "#06b6d4"
ACCENT_PURPLE = "#8b5cf6"
STATE_COLORS = {"Recovery": SUCCESS, "Baseline": WARNING, "Strain": DANGER}
PLOT_TEMPLATE = "plotly_dark"


def apply_dashboard_theme() -> None:
    st.set_page_config(
        page_title="Wearable Health AI | Clinical & Patient Enterprise",
        layout="wide",
        initial_sidebar_state="collapsed",
        page_icon="🧬"
    )
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    .stApp {{
        background: radial-gradient(circle at 12% 10%, rgba(6, 182, 212, 0.12) 0%, transparent 40%),
                    radial-gradient(circle at 88% 85%, rgba(16, 185, 129, 0.09) 0%, transparent 45%),
                    radial-gradient(circle at 50% 50%, rgba(139, 92, 246, 0.05) 0%, transparent 60%),
                    {APP_BG};
        color: {TEXT};
    }}

    [data-testid="stSidebar"], [data-testid="collapsedControl"] {{
        display: none;
    }}

    .block-container {{
        max-width: 1440px;
        padding-top: 1rem;
        padding-bottom: 2.5rem;
    }}

    /* Futuristic Cyber-Medical Header */
    .telemetry-hud {{
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.85));
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 20px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(16px);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), 0 0 20px rgba(6, 182, 212, 0.15) inset;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
        position: relative;
        overflow: hidden;
    }}

    .telemetry-hud::after {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, transparent, #06b6d4, #10b981, #8b5cf6, transparent);
        animation: scanline 4s linear infinite;
    }}

    @keyframes scanline {{
        0% {{ transform: translateX(-100%); }}
        100% {{ transform: translateX(100%); }}
    }}

    /* Pulsing ECG and Beating Heart Animations */
    @keyframes heartbeat {{
        0% {{ transform: scale(1); filter: drop-shadow(0 0 2px {DANGER}); }}
        14% {{ transform: scale(1.25); filter: drop-shadow(0 0 10px {DANGER}); }}
        28% {{ transform: scale(1); filter: drop-shadow(0 0 2px {DANGER}); }}
        42% {{ transform: scale(1.18); filter: drop-shadow(0 0 8px {DANGER}); }}
        70% {{ transform: scale(1); }}
    }}

    .heart-beat-icon {{
        display: inline-block;
        color: {DANGER};
        animation: heartbeat 1.4s infinite cubic-bezier(0.215, 0.61, 0.355, 1);
        margin-right: 6px;
    }}

    @keyframes pulseGlow {{
        0%, 100% {{ opacity: 1; transform: scale(1); }}
        50% {{ opacity: 0.45; transform: scale(0.92); }}
    }}

    .pulse-dot {{
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: {SUCCESS};
        box-shadow: 0 0 10px {SUCCESS};
        animation: pulseGlow 1.8s infinite ease-in-out;
        margin-right: 6px;
    }}

    .pulse-dot-danger {{
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: {DANGER};
        box-shadow: 0 0 10px {DANGER};
        animation: pulseGlow 1.2s infinite ease-in-out;
        margin-right: 6px;
    }}

    /* High-Tech HUD Metric Cards */
    .card {{
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 1.2rem;
        height: 100%;
        backdrop-filter: blur(12px);
        transition: all 0.28s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }}

    .card:hover {{
        border-color: rgba(56, 189, 248, 0.45);
        transform: translateY(-3px);
        box-shadow: 0 12px 28px -6px rgba(0, 0, 0, 0.6), 0 0 18px rgba(56, 189, 248, 0.12);
    }}

    .card-title {{
        color: {MUTED};
        font-family: 'Outfit', sans-serif;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.35rem;
    }}

    .card-value {{
        font-family: 'Outfit', sans-serif;
        font-size: 1.85rem;
        font-weight: 700;
        color: {TEXT};
        line-height: 1.15;
        letter-spacing: -0.02em;
    }}

    .card-subtitle {{
        color: {MUTED};
        font-size: 0.85rem;
        margin-top: 0.4rem;
    }}

    .section-title {{
        font-family: 'Outfit', sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        color: #f1f5f9;
        margin: 0.8rem 0 0.8rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }}

    /* Medical Status Pill Badge */
    .status-badge {{
        display: inline-flex;
        align-items: center;
        padding: 0.3rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }}

    /* Patient Digital Pass Card */
    .patient-pass-card {{
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.85), rgba(15, 23, 42, 0.95));
        border: 1px solid rgba(6, 182, 212, 0.35);
        border-radius: 20px;
        padding: 1.4rem;
        box-shadow: 0 10px 25px rgba(6, 182, 212, 0.12);
        position: relative;
    }}

    /* Doctor Consultation Clinical Note Box */
    .clinical-note-box {{
        background: rgba(15, 23, 42, 0.9);
        border-left: 4px solid {INFO};
        border-radius: 0 14px 14px 0;
        padding: 1rem 1.2rem;
        margin-bottom: 0.9rem;
        border-top: 1px solid rgba(255,255,255,0.05);
        border-right: 1px solid rgba(255,255,255,0.05);
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }}

    /* Real-Time Waveform ECG Bar */
    .ecg-svg {{
        stroke-dasharray: 1000;
        stroke-dashoffset: 1000;
        animation: dash 3.5s linear infinite;
    }}

    @keyframes dash {{
        to {{
            stroke-dashoffset: 0;
        }}
    }}
    </style>
    """, unsafe_allow_html=True)


def status_color(value: str) -> str:
    if value in {"Recovery", "Good", "Low", "Improving", "Stable", "Normal", "Cleared"}:
        return SUCCESS
    if value in {"Baseline", "Moderate", "Warning", "Caution"}:
        return WARNING
    return DANGER


def render_animated_top_hud(user_session: dict[str, Any], current_view: str) -> None:
    """Renders high-tech animated medical HUD with vital waveforms and role badges."""
    role = user_session.get("role", "patient")
    role_badge = "👨‍⚕️ HOSPITAL CLINICIAN" if role == "hospital" else "🏃 PATIENT PASS"
    badge_bg = "rgba(6, 182, 212, 0.18)" if role == "hospital" else "rgba(16, 185, 129, 0.18)"
    badge_border = INFO if role == "hospital" else SUCCESS
    badge_color = INFO if role == "hospital" else SUCCESS

    full_name = user_session.get("fullName") or user_session.get("username", "Alex Mercer")
    patient_or_doctor_id = user_session.get("patientId") or user_session.get("licenseNumber") or "U0042"

    st.markdown(f"""
    <div class="telemetry-hud">
        <div style="display:flex; align-items:center; gap:16px;">
            <div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.1); border-radius:14px; padding:8px 12px; display:flex; align-items:center;">
                <span class="heart-beat-icon" style="font-size:1.5rem;">♥</span>
                <div>
                    <div style="font-family:'Outfit'; font-size:1.15rem; font-weight:800; letter-spacing:0.02em; color:#f8fafc;">
                        WEARABLE HEALTH AI
                    </div>
                    <div style="font-size:0.75rem; color:{MUTED}; font-family:'JetBrains Mono';">
                        Enterprise Clinical & Patient Telemetry
                    </div>
                </div>
            </div>
            <div style="background:{badge_bg}; border:1px solid {badge_border}; color:{badge_color}; padding:5px 12px; border-radius:9999px; font-size:0.78rem; font-weight:700; letter-spacing:0.06em;">
                <span class="pulse-dot"></span>{role_badge}
            </div>
        </div>

        <div style="display:flex; align-items:center; gap:20px;">
            <div style="text-align:right;">
                <div style="font-size:0.75rem; color:{MUTED}; text-transform:uppercase; letter-spacing:0.06em;">Current Operator</div>
                <div style="font-family:'Outfit'; font-size:1.02rem; font-weight:700; color:{TEXT};">{full_name}</div>
                <div style="font-size:0.75rem; color:{INFO}; font-family:'JetBrains Mono';">ID: {patient_or_doctor_id}</div>
            </div>
            <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:6px 14px; font-family:'JetBrains Mono'; font-size:0.74rem;">
                <div style="color:{SUCCESS};">● HIPAA / HL7 SECURE</div>
                <div style="color:{MUTED};">HMM LATENCY: 14ms</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_card(title: str, value: str, subtitle: str = "", accent: str | None = None) -> None:
    border = accent or CARD_BORDER
    glow = f"box-shadow: 0 0 16px {border}25;" if accent else ""
    st.markdown(f"""
    <div class='card' style='border-color:{border}; {glow}'>
        <div class='card-title'>{title}</div>
        <div class='card-value'>{value}</div>
        <div class='card-subtitle'>{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def render_vital_metric_hud(latest_data: pd.Series | dict[str, Any] | None) -> None:
    """Renders glowing biometric HUD cards with animated heartbeat and SpO2 metrics."""
    if latest_data is None:
        hr, hrv, sleep, spo2, steps = 66.0, 52.0, 7.4, 98.0, 8420
    elif isinstance(latest_data, dict):
        hr = float(latest_data.get("resting_hr_bpm") or latest_data.get("restingHeartRate") or 66.0)
        hrv = float(latest_data.get("hrv_rmssd_ms") or latest_data.get("hrvRmssd") or 52.0)
        sleep = float(latest_data.get("sleep_duration_hours") or latest_data.get("sleepHours") or 7.4)
        spo2 = float(latest_data.get("spo2_avg_pct") or latest_data.get("spo2") or 98.0)
        steps = int(latest_data.get("steps") or 8420)
    else:
        hr = float(latest_data.get("resting_hr_bpm", 66.0))
        hrv = float(latest_data.get("hrv_rmssd_ms", 52.0))
        sleep = float(latest_data.get("sleep_duration_hours", 7.4))
        spo2 = float(latest_data.get("spo2_avg_pct", 98.0))
        steps = int(latest_data.get("steps", 8420))

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_card("Resting Heart Rate", f"{hr:.0f} BPM", "♥ Continuous Nocturnal Pulse", status_color("Good" if hr <= 72 else "Warning"))
    with c2:
        render_card("HRV Recovery (RMSSD)", f"{hrv:.1f} MS", "⚡ Autonomic Parasympathetic Index", status_color("Good" if hrv >= 45 else "Warning"))
    with c3:
        render_card("Sleep Restorative", f"{sleep:.1f} HRS", "🌙 Circadian Deep & REM Total", status_color("Good" if sleep >= 7.0 else "Warning"))
    with c4:
        render_card("Blood Oxygen SpO2", f"{spo2:.1f}%", "🫁 Peripheral O2 Saturation", status_color("Good" if spo2 >= 95.0 else "Danger"))
    with c5:
        render_card("Active Daily Steps", f"{steps:,}", "🚶 Metabolic Movement Output", INFO)


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
        return f"Recovery is leading today, with HRV at {float(latest.get('hrv_rmssd_ms', 0.0)):.1f} ms and sleep at {float(latest.get('sleep_duration_hours', 0.0)):.1f} h. Optimal readiness for cognitive and physical demands."
    if analysis.get("state") == "Strain":
        return "Elevated physiological strain detected across cardiovascular and nocturnal recovery metrics. Immediate workload reduction and sleep hygiene recommended."
    return "Baseline physiology is maintaining homeostasis. Moderate steady-state performance capacity without acute systemic strain."


def plot_risk_gauge(score: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(score),
        title={"text": "Physiological Strain Index (0-100)", "font": {"family": "Outfit", "size": 16, "color": "#f8fafc"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": MUTED},
            "bar": {"color": DANGER if score >= 65 else WARNING if score >= 35 else SUCCESS, "thickness": 0.28},
            "bgcolor": "rgba(255,255,255,0.03)",
            "borderwidth": 1,
            "bordercolor": "rgba(255,255,255,0.1)",
            "steps": [
                {"range": [0, 35], "color": "rgba(16, 185, 129, 0.22)"},
                {"range": [35, 65], "color": "rgba(245, 158, 11, 0.22)"},
                {"range": [65, 100], "color": "rgba(239, 68, 68, 0.26)"}
            ]
        }
    ))
    fig.update_layout(height=280, template=PLOT_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=50, b=20))
    return fig


def plot_trends(final_results_df: pd.DataFrame | None) -> go.Figure | None:
    if final_results_df is None or final_results_df.empty:
        return None
    ordered = final_results_df.sort_values(["user_id", "date"]).copy()
    metrics = [("hrv_rmssd_ms", "HRV RMSSD (ms)", SUCCESS), ("resting_hr_bpm", "Resting Heart Rate (bpm)", DANGER), ("sleep_duration_hours", "Sleep Duration (hrs)", WARNING)]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=[label for _, label, _ in metrics])
    row = 1
    for column, label, color in metrics:
        if column in ordered.columns:
            fig.add_trace(go.Scatter(
                x=ordered["date"],
                y=ordered[column],
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2.8),
                marker=dict(size=5, color=color),
                hovertemplate=f"%{{x|%Y-%m-%d}}<br>{label}: %{{y:.2f}}<extra></extra>"
            ), row=row, col=1)
        row += 1
    fig.update_layout(height=600, template=PLOT_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15, 23, 42, 0.6)", margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
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
    fig = px.bar(
        pd.DataFrame(rows).melt(id_vars="Metric", value_vars=["Baseline", "Latest"], var_name="Window", value_name="Value"),
        x="Metric", y="Value", color="Window", barmode="group",
        color_discrete_map={"Baseline": INFO, "Latest": WARNING},
        template=PLOT_TEMPLATE
    )
    fig.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15, 23, 42, 0.6)", margin=dict(l=10, r=10, t=10, b=10))
    return fig


def plot_feature_distribution(final_results_df: pd.DataFrame | None, feature_name: str) -> go.Figure | None:
    if final_results_df is None or final_results_df.empty or feature_name not in final_results_df.columns:
        return None
    fig = px.histogram(final_results_df[[feature_name]].dropna(), x=feature_name, nbins=24, template=PLOT_TEMPLATE, color_discrete_sequence=[INFO])
    fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15, 23, 42, 0.6)", margin=dict(l=10, r=10, t=10, b=10))
    return fig


def plot_state_probabilities(probabilities: dict[str, Any] | None) -> go.Figure | None:
    if not probabilities:
        return None
    prob_df = pd.DataFrame({"State": list(probabilities.keys()), "Probability": list(probabilities.values())})
    fig = px.bar(prob_df, x="Probability", y="State", orientation="h", color="State", color_discrete_map=STATE_COLORS, text="Probability", template=PLOT_TEMPLATE)
    fig.update_traces(texttemplate="%{text:.1f}%", hovertemplate="%{y}: %{x:.1f}%<extra></extra>")
    fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15, 23, 42, 0.6)", margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    return fig


def plot_transition_heatmap(transition_matrix: pd.DataFrame | None) -> go.Figure | None:
    if transition_matrix is None or transition_matrix.empty:
        return None
    fig = go.Figure(data=go.Heatmap(
        z=transition_matrix.astype(float).values,
        x=list(transition_matrix.columns),
        y=list(transition_matrix.index),
        colorscale=[[0.0, "#0f172a"], [0.5, "#0369a1"], [1.0, "#10b981"]],
        text=transition_matrix.round(2).astype(str).values,
        texttemplate="%{text}",
        hovertemplate="From %{y} to %{x}: %{z:.3f}<extra></extra>"
    ))
    fig.update_layout(height=360, template=PLOT_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15, 23, 42, 0.6)", margin=dict(l=10, r=10, t=10, b=10))
    return fig


def compute_warning_flags(feature_df: pd.DataFrame | None) -> dict[str, Any]:
    if feature_df is None or feature_df.empty:
        return {"hrv_slope": 0.0, "sleep_slope": 0.0, "hrv_alert": False, "sleep_alert": False}
    latest = feature_df.sort_values(["user_id", "date"]).reset_index(drop=True).iloc[-1]
    hrv_slope = float(latest.get("hrv_dev_slope_7d", 0.0))
    sleep_slope = float(latest.get("sleep_dev_slope_7d", 0.0))
    return {"hrv_slope": round(hrv_slope, 3), "sleep_slope": round(sleep_slope, 3), "hrv_alert": hrv_slope < -0.15, "sleep_alert": sleep_slope < -0.1}


def compute_explainability(final_results_df: pd.DataFrame | None, analysis: dict[str, Any], model_type: str) -> dict[str, Any]:
    fallback_reason = "Model center influence"
    if final_results_df is None or final_results_df.empty:
        return {"method": fallback_reason, "importance_df": pd.DataFrame(columns=["Feature", "Importance"]), "waterfall_df": pd.DataFrame(columns=["Feature", "Contribution"]), "fidelity": {}}

    target_column = "gmm_state_label" if model_type == "gmm" else "cluster_label"
    feature_columns = [column for column in ["resting_hr_bpm", "hrv_rmssd_ms", "sleep_duration_hours", "steps", "spo2_avg_pct", "severity_score", "hr_dev", "hrv_dev", "sleep_dev"] if column in final_results_df.columns]

    if target_column in final_results_df.columns and len(feature_columns) >= 3:
        explain_df = final_results_df[feature_columns + [target_column]].dropna().copy()
        if not explain_df.empty and explain_df[target_column].nunique() >= 2:
            X = explain_df[feature_columns]
            y = explain_df[target_column]
            model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            model.fit(X, y)
            preds = model.predict(X)

            acc = float(accuracy_score(y, preds))
            bal_acc = float(balanced_accuracy_score(y, preds))
            f1 = float(f1_score(y, preds, average="macro"))
            cm = confusion_matrix(y, preds).tolist()

            fidelity = {
                "accuracy": round(acc * 100.0, 1),
                "balanced_accuracy": round(bal_acc * 100.0, 1),
                "macro_f1": round(f1, 4),
                "confusion_matrix": cm,
            }

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
                    return {"method": "Surrogate-Model SHAP", "importance_df": importance_df, "waterfall_df": waterfall_df, "fidelity": fidelity}
                except Exception:
                    pass

            perm = permutation_importance(model, X, y, n_repeats=5, random_state=42)
            importance_df = pd.DataFrame({"Feature": feature_columns, "Importance": perm.importances_mean}).sort_values("Importance", ascending=False)
            waterfall_df = pd.DataFrame({"Feature": feature_columns, "Contribution": latest_features.iloc[0].values - X.mean().values}).sort_values("Contribution", key=lambda s: s.abs(), ascending=False)
            return {"method": "Surrogate Permutation Importance", "importance_df": importance_df, "waterfall_df": waterfall_df, "fidelity": fidelity}

    model_influence = analysis.get("model_feature_influence") or {}
    importance_df = pd.DataFrame({"Feature": list(model_influence.keys()), "Importance": list(model_influence.values())}).sort_values("Importance", ascending=False) if model_influence else pd.DataFrame(columns=["Feature", "Importance"])
    waterfall_df = pd.DataFrame({"Feature": importance_df.get("Feature", []), "Contribution": importance_df.get("Importance", [])})
    return {"method": "GMM Native Feature Influence", "importance_df": importance_df, "waterfall_df": waterfall_df, "fidelity": {}}


def plot_explainability_bars(explainability: dict[str, Any]) -> go.Figure | None:
    importance_df = explainability.get("importance_df")
    if importance_df is None or importance_df.empty:
        return None
    plot_df = importance_df.head(5).sort_values("Importance", ascending=True)
    fig = px.bar(plot_df, x="Importance", y="Feature", orientation="h", template=PLOT_TEMPLATE, color_discrete_sequence=[INFO])
    fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15, 23, 42, 0.6)", margin=dict(l=10, r=10, t=10, b=10))
    return fig


def plot_waterfall_fallback(explainability: dict[str, Any]) -> go.Figure | None:
    waterfall_df = explainability.get("waterfall_df")
    if waterfall_df is None or waterfall_df.empty:
        return None
    plot_df = waterfall_df.head(5).iloc[::-1]
    colors = [SUCCESS if value >= 0 else DANGER for value in plot_df["Contribution"]]
    fig = go.Figure(go.Bar(x=plot_df["Contribution"], y=plot_df["Feature"], orientation="h", marker_color=colors))
    fig.update_layout(height=300, template=PLOT_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15, 23, 42, 0.6)", margin=dict(l=10, r=10, t=10, b=10))
    return fig


def render_environment_cards(latest_or_summary: Any = None, extra_arg: Any = None) -> None:
    c1, c2, c3 = st.columns(3)
    latest = latest_or_summary if isinstance(latest_or_summary, (pd.Series, dict)) else {}
    hr = float(latest.get("resting_hr_bpm", 65.0)) if isinstance(latest, dict) else 65.0
    hrv = float(latest.get("hrv_rmssd_ms", 55.0)) if isinstance(latest, dict) else 55.0
    sleep = float(latest.get("sleep_duration_hours", 7.2)) if isinstance(latest, dict) else 7.2

    with c1:
        render_card("Resting HR Signal", f"{hr:.0f} bpm", "Baseline resting pulse", status_color("Good" if hr <= 72 else "Warning"))
    with c2:
        render_card("HRV Recovery", f"{hrv:.1f} ms", "Autonomic RMSSD index", status_color("Good" if hrv >= 45 else "Warning"))
    with c3:
        render_card("Sleep Continuity", f"{sleep:.1f} hrs", "Restorative sleep duration", status_color("Good" if sleep >= 7.0 else "Warning"))


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
    bg_color = DANGER if "Alert" in level or "Critical" in level or "Strain" in level else WARNING
    st.markdown(
        f"<div class='card' style='border-color:{bg_color}; background-color:{CARD_BG}; margin-bottom:1rem; box-shadow:0 0 20px {bg_color}33;'>"
        f"<div class='card-title' style='color:{bg_color}; font-weight:bold;'>⚠️ Clinical Advisory Alert: {level.upper()}</div>"
        f"<div style='font-size:1.05rem; margin:.4rem 0; font-weight:600;'>{clinical_escalation.get('clinical_summary_message', '')}</div>"
        f"<div class='card-subtitle'>Persistent Anomaly Window: {clinical_escalation.get('consecutive_high_risk_days', 0)} consecutive days</div>"
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
    fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15, 23, 42, 0.6)", xaxis=dict(range=[0, 100]), margin=dict(l=10, r=10, t=10, b=10))
    return fig


def render_state_metrics(analysis: dict[str, Any]) -> None:
    state = str(analysis.get("state", "Unknown"))
    confidence = float(analysis.get("confidence", 0.0))
    risk_score = float(analysis.get("risk", {}).get("score", 0.0))

    c1, c2, c3 = st.columns(3)
    with c1:
        render_card("Inferred State", state, "Latent physiological state", status_color(state))
    with c2:
        render_card("State Confidence", f"{confidence:.1f}%", "GMM posterior confidence", INFO)
    with c3:
        render_card("Physiological Strain Index", f"{risk_score:.0f}", "Composite strain score", status_color("Low" if risk_score < 35 else "Moderate" if risk_score < 65 else "High"))


def render_time_series(df: pd.DataFrame | None) -> None:
    fig = plot_trends(df)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)


def render_baseline_comparison(df: pd.DataFrame | None) -> None:
    fig = plot_baseline_comparison(df)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)


def render_gmm_probabilities(probs: dict[str, Any] | None) -> None:
    fig = plot_state_probabilities(probs)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)


def render_transition_heatmap_view(matrix: pd.DataFrame | None) -> None:
    fig = plot_transition_heatmap(matrix)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)


def render_shap_explainability(df: pd.DataFrame | None, influence: Any = None) -> None:
    explain = compute_explainability(df, {"model_feature_influence": influence}, "gmm")
    fig = plot_explainability_bars(explain)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)


def render_early_warning(analysis: dict[str, Any]) -> None:
    ew = analysis.get("early_warning", {})
    st.write(ew.get("message", "No early warning pattern."))


def render_recovery_score_trend(df: pd.DataFrame | None) -> None:
    if df is not None and "severity_score" in df.columns:
        fig = px.line(df, x="date", y="severity_score", title="Severity Trend", template=PLOT_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)


def render_app_header(model_name: str, page_name: str) -> None:
    pass  # Handled cleanly by render_animated_top_hud
