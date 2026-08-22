from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from app.dashboard_utils import apply_dashboard_theme, render_environment_cards, render_state_metrics


st.set_page_config(page_title="Recommendations", layout="wide")
apply_dashboard_theme()

analysis_result = st.session_state.get("analysis_result")
final_results_df = st.session_state.get("final_results_df")
transition_matrix = st.session_state.get("transition_matrix")

st.title("Recommendations")
st.caption("Action engine driven by physiological state, trend, and environment context.")


def build_personalized_actions(analysis: dict, latest_row: pd.Series | None) -> list[str]:
    actions: list[str] = []
    state = str(analysis.get("state", "Unknown"))
    environment = analysis.get("environment", {}) or {}

    sleep_hours = float(latest_row.get("sleep_duration_hours", 0.0)) if latest_row is not None else 0.0
    steps = float(latest_row.get("steps", 0.0)) if latest_row is not None else 0.0
    hrv = float(latest_row.get("hrv_rmssd_ms", 0.0)) if latest_row is not None else 0.0

    if state in {"Strain", "High Stress"}:
        actions.append("Reduce training intensity and prioritize recovery today.")
    if sleep_hours < 6.5:
        actions.append("Improve sleep duration tonight and protect your recovery window.")
    if hrv < 40:
        actions.append("Low HRV suggests reduced recovery capacity, so choose rest or light activity.")
    if steps < 4000:
        actions.append("Add light movement such as walking instead of remaining fully sedentary.")
    if float(environment.get("aqi", 0.0)) >= 4:
        actions.append("Avoid outdoor activity because current air quality is poor.")
    if float(environment.get("temperature", 0.0)) > 35:
        actions.append("Increase hydration and avoid heavy exertion in the heat.")

    if not actions:
        actions.append("Maintain your current routine and continue monitoring recovery.")
    return actions


def build_reason_lines(analysis: dict, latest_row: pd.Series | None, baseline_row: pd.Series | None) -> list[str]:
    reasons: list[str] = []
    if latest_row is None or baseline_row is None:
        return ["Recommendations are based on current state and trend signals."]

    latest_hrv = float(latest_row.get("hrv_rmssd_ms", 0.0))
    baseline_hrv = float(baseline_row.get("hrv_rmssd_ms", 0.0))
    latest_sleep = float(latest_row.get("sleep_duration_hours", 0.0))
    baseline_sleep = float(baseline_row.get("sleep_duration_hours", 0.0))
    latest_hr = float(latest_row.get("resting_hr_bpm", 0.0))
    baseline_hr = float(baseline_row.get("resting_hr_bpm", 0.0))

    if latest_hrv < baseline_hrv:
        reasons.append("Low HRV relative to your baseline triggered recovery-focused advice.")
    if latest_sleep < baseline_sleep:
        reasons.append("Sleep below baseline increased the urgency of recovery guidance.")
    if latest_hr > baseline_hr:
        reasons.append("Higher resting heart rate suggests added physiological strain.")
    if analysis.get("trend") == "Increasing Stress":
        reasons.append("The temporal trend indicates stress is building rather than resolving.")

    if not reasons:
        reasons.append("Your recommendations are driven by stable physiology and low current risk.")
    return reasons


def build_future_risk_text(analysis: dict, transition_frame: pd.DataFrame | None) -> str:
    if transition_frame is None or transition_frame.empty:
        return "Future-state estimate is unavailable because transition history is limited."

    current_state = str(analysis.get("temporal_state") or analysis.get("state") or "")
    if current_state not in transition_frame.index:
        return "Future-state estimate is unavailable for the current state."

    next_state = str(transition_frame.loc[current_state].astype(float).idxmax())
    if next_state == current_state:
        return f"You are most likely to remain in {current_state} if the current pattern continues."
    return f"You are likely to transition to {next_state} if the current trend continues."


if analysis_result is None or st.session_state.get("data") is None:
    st.warning("Upload dataset first")
else:
    ordered = None
    latest = None
    baseline = None
    if final_results_df is not None and not final_results_df.empty:
        ordered = final_results_df.sort_values(["user_id", "date"]).reset_index(drop=True)
        latest = ordered.iloc[-1]
        baseline = ordered[[column for column in ["resting_hr_bpm", "hrv_rmssd_ms", "sleep_duration_hours", "steps"] if column in ordered.columns]].mean()

    render_state_metrics(analysis_result)
    st.markdown("---")

    st.subheader("Top Recommendation")
    top_recommendation = str(analysis_result.get("recommendations", [""])[0] or "Maintain your current routine.")
    if analysis_result.get("state") in {"Strain", "High Stress"}:
        st.error(top_recommendation)
    elif analysis_result.get("state") == "Recovery":
        st.success(top_recommendation)
    else:
        st.warning(top_recommendation)

    st.markdown("---")
    left, right = st.columns([2, 1])
    with left:
        st.subheader("Personalized Actions")
        for action in build_personalized_actions(analysis_result, latest):
            st.write(f"- {action}")

        st.markdown("---")
        st.subheader("Why These Recommendations?")
        for reason in build_reason_lines(analysis_result, latest, baseline):
            st.write(f"- {reason}")
    with right:
        st.subheader("Future Risk")
        future_text = build_future_risk_text(analysis_result, transition_matrix)
        if "transition to" in future_text.lower() or "strain" in future_text.lower() or "stress" in future_text.lower():
            st.warning(future_text)
        else:
            st.info(future_text)

    st.markdown("---")
    st.subheader("Environment-Aware Advice")
    render_environment_cards(
        analysis_result.get("environment", {}),
        (analysis_result.get("environment_impact") or {}).get("environment_summary"),
    )
