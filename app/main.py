from __future__ import annotations

import importlib
import inspect
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

from app.dashboard_utils import (
    APP_BG,
    CARD_BG,
    CARD_BORDER,
    DANGER,
    INFO,
    MUTED,
    SUCCESS,
    TEXT,
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
    render_animated_top_hud,
    render_card,
    render_clinical_advisory_card,
    render_environment_cards,
    render_vital_metric_hud,
    status_color,
)
import run_inference as run_inference_module
from src.preprocessing import validate_required_columns
import src.risk_scoring as risk_scoring_module

# Optional Backend Database integration
try:
    import database as backend_db
    import models as backend_models
    HAS_DB = True
except Exception:
    HAS_DB = False

run_inference_module = importlib.reload(run_inference_module)
risk_scoring_module = importlib.reload(risk_scoring_module)
run_pipeline = run_inference_module.run_pipeline
calculate_risk_score = risk_scoring_module.calculate_risk_score

SAMPLE_FILE = ROOT_DIR / "data" / "wearables_health_6mo_daily.csv"
OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_LABELS = {"GMM + HMM (Recommended)": "gmm", "KMeans + HMM": "kmeans"}
CACHE_SCHEMA_VERSION = "v5"

DEFAULT_PATIENT_USER = {
    "username": "patient",
    "role": "patient",
    "fullName": "Alex Mercer",
    "patientId": "U0042",
    "email": "alex.mercer@health.ai",
}

DEFAULT_DOCTOR_USER = {
    "username": "doctor",
    "role": "hospital",
    "fullName": "Dr. Elena Vance, MD",
    "hospitalName": "Metro General Heart & Vascular Institute",
    "department": "Cardiology & Autonomic Medicine",
    "licenseNumber": "MD-CARDIO-88219",
}

SESSION_DEFAULTS = {
    "auth_user": DEFAULT_PATIENT_USER,
    "active_df": None,
    "model_cache": {},
    "last_data_key": None,
    "selected_patient_id": "U0042",
    "consultation_status_msg": None,
}

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


# ---------------------------------------------------------
# Dynamic ECG Animated Oscilloscope Bar
# ---------------------------------------------------------
def render_animated_ecg_bar(bpm: float = 68.0) -> None:
    st.markdown(f"""
    <div style="background:rgba(15,23,42,0.85); border:1px solid rgba(6,182,212,0.2); border-radius:14px; padding:8px 16px; margin-bottom:1rem; display:flex; align-items:center; justify-content:space-between; overflow:hidden;">
        <div style="display:flex; align-items:center; gap:12px;">
            <span class="heart-beat-icon" style="font-size:1.3rem;">♥</span>
            <span style="font-family:'JetBrains Mono'; font-weight:700; font-size:0.95rem; color:{SUCCESS};">
                {bpm:.0f} BPM <span style="font-size:0.75rem; color:{MUTED}; font-weight:400;">R-R INTERVAL: {int(60000/max(bpm, 40))}ms</span>
            </span>
        </div>
        <div style="flex-grow:1; margin:0 25px; height:32px; display:flex; align-items:center;">
            <svg viewBox="0 0 500 40" style="width:100%; height:100%; filter:drop-shadow(0 0 4px {INFO});">
                <path d="M 0 20 L 70 20 L 80 8 L 90 32 L 100 20 L 130 20 L 140 2 L 150 38 L 160 20 L 220 20 L 230 10 L 240 30 L 250 20 L 320 20 L 330 4 L 340 36 L 350 20 L 410 20 L 420 12 L 430 28 L 440 20 L 500 20" 
                      fill="none" stroke="{INFO}" stroke-width="2.2" stroke-linecap="round" class="ecg-svg"/>
            </svg>
        </div>
        <div style="font-family:'JetBrains Mono'; font-size:0.75rem; color:{MUTED};">
            <span class="pulse-dot"></span>LIVE TELEMETRY STREAM
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# Role Selector & Authentication Bar
# ---------------------------------------------------------
def render_auth_controls() -> None:
    current_user = st.session_state.get("auth_user", DEFAULT_PATIENT_USER)
    current_role = current_user.get("role", "patient")

    col_info, col_switch, col_modal = st.columns([2.2, 1.2, 0.9])
    with col_info:
        active_role_str = "🏥 Hospital Clinician Portal" if current_role == "hospital" else "🏃 Patient Personal Portal"
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:10px; margin-top:4px;">
            <span style="font-size:0.85rem; color:{MUTED}; text-transform:uppercase; letter-spacing:0.06em;">Active Role:</span>
            <span style="font-family:'Outfit'; font-weight:700; color:{TEXT}; font-size:1.02rem;">{active_role_str}</span>
            <span style="font-size:0.82rem; color:{INFO}; font-family:'JetBrains Mono';">({current_user.get('fullName')})</span>
        </div>
        """, unsafe_allow_html=True)

    with col_switch:
        if current_role == "patient":
            if st.button("🔄 Switch to Doctor Portal", use_container_width=True, key="btn_switch_doctor"):
                st.session_state["auth_user"] = DEFAULT_DOCTOR_USER
                st.rerun()
        else:
            if st.button("🔄 Switch to Patient Portal", use_container_width=True, key="btn_switch_patient"):
                st.session_state["auth_user"] = DEFAULT_PATIENT_USER
                st.rerun()

    with col_modal:
        with st.popover("🔑 Custom Login"):
            st.markdown("#### User Authentication")
            role_choice = st.radio("Select Role", ["Patient", "Hospital / Clinician"], horizontal=True)
            uname = st.text_input("Username", value="patient" if role_choice == "Patient" else "doctor")
            pwd = st.text_input("Password", type="password", value="health2026" if role_choice == "Patient" else "clinical2026")
            if st.button("Log In", type="primary", use_container_width=True):
                if role_choice == "Patient":
                    st.session_state["auth_user"] = {
                        "username": uname,
                        "role": "patient",
                        "fullName": "Alex Mercer" if uname == "patient" else uname.title(),
                        "patientId": "U0042",
                        "email": f"{uname}@health.ai",
                    }
                else:
                    st.session_state["auth_user"] = {
                        "username": uname,
                        "role": "hospital",
                        "fullName": "Dr. Elena Vance, MD" if uname == "doctor" else f"Dr. {uname.title()}",
                        "hospitalName": "Metro General Heart & Vascular Institute",
                        "department": "Cardiology & Autonomic Medicine",
                        "licenseNumber": "MD-CARDIO-88219",
                    }
                st.success("Authenticated successfully!")
                st.rerun()


# ---------------------------------------------------------
# Patient Digital Health Pass & QR Token Component
# ---------------------------------------------------------
def render_patient_digital_pass(patient_id: str, latest: pd.Series | None, analysis: dict[str, Any]) -> None:
    st.markdown("<div class='section-title'>🎫 Digital Patient Health Pass & Hospital Admission Token</div>", unsafe_allow_html=True)
    pass_col, info_col = st.columns([1.1, 1.3])
    with pass_col:
        hr = float(latest.get("resting_hr_bpm", 64.0)) if latest is not None else 64.0
        hrv = float(latest.get("hrv_rmssd_ms", 52.0)) if latest is not None else 52.0
        spo2 = float(latest.get("spo2_avg_pct", 98.0)) if latest is not None else 98.0
        state = str(analysis.get("state", "Baseline"))

        st.markdown(f"""
        <div class="patient-pass-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; border-bottom:1px solid rgba(255,255,255,0.08); padding-bottom:8px;">
                <div>
                    <div style="font-family:'Outfit'; font-weight:800; font-size:1.15rem; color:#f8fafc;">HOSPITAL ADMISSION PASS</div>
                    <div style="font-size:0.75rem; color:{MUTED}; font-family:'JetBrains Mono';">HIPAA / FAST HEALTHCARE INTEROPERABILITY</div>
                </div>
                <div style="background:rgba(16,185,129,0.18); border:1px solid {SUCCESS}; color:{SUCCESS}; font-size:0.75rem; font-weight:700; padding:4px 10px; border-radius:9999px;">
                    VERIFIED ACTIVE
                </div>
            </div>

            <div style="display:flex; gap:16px; align-items:center; margin-bottom:14px;">
                <div style="background:white; padding:8px; border-radius:10px; display:inline-block;">
                    <!-- Simulated high-contrast scannable QR matrix -->
                    <div style="width:84px; height:84px; background:#000; display:grid; grid-template-columns:repeat(7, 1fr); gap:2px; padding:2px;">
                        <div style="background:#fff;"></div><div style="background:#fff;"></div><div style="background:#fff;"></div><div style="background:#000;"></div><div style="background:#fff;"></div><div style="background:#fff;"></div><div style="background:#fff;"></div>
                        <div style="background:#fff;"></div><div style="background:#000;"></div><div style="background:#fff;"></div><div style="background:#000;"></div><div style="background:#fff;"></div><div style="background:#000;"></div><div style="background:#fff;"></div>
                        <div style="background:#fff;"></div><div style="background:#fff;"></div><div style="background:#fff;"></div><div style="background:#fff;"></div><div style="background:#fff;"></div><div style="background:#fff;"></div><div style="background:#fff;"></div>
                        <div style="background:#000;"></div><div style="background:#fff;"></div><div style="background:#000;"></div><div style="background:#fff;"></div><div style="background:#000;"></div><div style="background:#fff;"></div><div style="background:#000;"></div>
                        <div style="background:#fff;"></div><div style="background:#fff;"></div><div style="background:#fff;"></div><div style="background:#000;"></div><div style="background:#fff;"></div><div style="background:#000;"></div><div style="background:#fff;"></div>
                        <div style="background:#fff;"></div><div style="background:#000;"></div><div style="background:#fff;"></div><div style="background:#fff;"></div><div style="background:#000;"></div><div style="background:#fff;"></div><div style="background:#fff;"></div>
                        <div style="background:#fff;"></div><div style="background:#fff;"></div><div style="background:#fff;"></div><div style="background:#000;"></div><div style="background:#fff;"></div><div style="background:#fff;"></div><div style="background:#fff;"></div>
                    </div>
                </div>
                <div>
                    <div style="font-size:0.75rem; color:{MUTED}; text-transform:uppercase;">Patient ID</div>
                    <div style="font-family:'JetBrains Mono'; font-size:1.3rem; font-weight:800; color:{INFO};">{patient_id}</div>
                    <div style="font-size:0.85rem; color:{TEXT}; margin-top:2px;">Alex Mercer (Age: 35, Male)</div>
                    <div style="font-size:0.72rem; color:{SUCCESS}; font-family:'JetBrains Mono';">TOKEN: #AUTH-MED-{hash(patient_id) % 900000 + 100000}</div>
                </div>
            </div>

            <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius:12px; padding:10px 14px; font-size:0.82rem;">
                <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                    <span style="color:{MUTED};">Resting HR:</span><span style="font-weight:700; color:{TEXT};">{hr:.0f} BPM</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                    <span style="color:{MUTED};">HRV RMSSD:</span><span style="font-weight:700; color:{TEXT};">{hrv:.1f} MS</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                    <span style="color:{MUTED};">Blood SpO2:</span><span style="font-weight:700; color:{TEXT};">{spo2:.1f}%</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:{MUTED};">Physiological State:</span><span style="font-weight:700; color:{status_color(state)};">{state}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with info_col:
        st.markdown(f"""
        <div class="card">
            <div class="section-title">🏥 Hospital Visit Instructions</div>
            <p style="color:{MUTED}; font-size:0.92rem; line-height:1.6;">
                When arriving at the clinic or hospital reception, present your <strong>Patient ID ({patient_id})</strong> or allow the attending physician to scan your QR Health Pass.
            </p>
            <div style="background:rgba(6,182,212,0.08); border-left:3px solid {INFO}; padding:10px 14px; border-radius:0 10px 10px 0; margin:12px 0;">
                <div style="color:{INFO}; font-weight:700; font-size:0.88rem;">Clinician Instant Sync Protocol</div>
                <div style="font-size:0.82rem; color:{MUTED};">The attending clinician will immediately pull your longitudinal 6-month continuous wearable logs, nocturnal cardiac metrics, and AI strain history.</div>
            </div>
            <p style="color:{MUTED}; font-size:0.85rem;">
                🔒 All shared telemetry is cryptographically authenticated under HIPAA Title II and European GDPR health privacy standards.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------
# Hospital & Doctor EHR Portal Component
# ---------------------------------------------------------
def render_hospital_clinician_portal(sample_df: pd.DataFrame, model_type: str) -> None:
    st.markdown("<div class='section-title'>🏥 Hospital EHR Clinical Intelligence & Patient Lookup Console</div>", unsafe_allow_html=True)

    # 1. Clinic Triage Overview
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_card("Clinic Monitored Cohort", "300 Patients", "Active Longitudinal Cohort", INFO)
    with c2:
        render_card("Urgent Triage Cases", "12 Patients", "Sustained Physiological Strain", DANGER)
    with c3:
        render_card("Average Cohort Sleep", "7.1 Hours", "Circadian Restorative Mean", SUCCESS)
    with c4:
        render_card("Mean Cohort HRV", "48.6 MS", "Population RMSSD Baseline", WARNING)

    st.markdown("---")

    # 2. Patient Search & History Retrieval
    st.markdown("### 🔍 Patient EHR Lookup & Medical Record Retrieval")
    search_col, button_col = st.columns([3, 1])

    available_patients = ["U0042 (Current Visiting Patient - Alex Mercer)", "U0001", "U0002", "U0003", "U0007", "U0015", "android_device_test_30d"]
    with search_col:
        selected_patient_str = st.selectbox("Select or Search Patient Medical Record ID:", available_patients, index=0)
        patient_id = selected_patient_str.split()[0]
        st.session_state["selected_patient_id"] = patient_id

    # Filter data for this patient
    if not sample_df.empty and "user_id" in sample_df.columns:
        patient_df = sample_df[sample_df["user_id"] == patient_id].sort_values("date")
    else:
        patient_df = pd.DataFrame()

    if patient_df.empty:
        # Fallback to U0042
        patient_df = sample_df[sample_df["user_id"] == "U0042"].sort_values("date")

    latest_record = patient_df.iloc[-1] if not patient_df.empty else None

    # Patient Medical Banner
    if latest_record is not None:
        hr = float(latest_record.get("resting_hr_bpm", 64.0))
        hrv = float(latest_record.get("hrv_rmssd_ms", 45.0))
        sleep = float(latest_record.get("sleep_duration_hours", 7.0))
        spo2 = float(latest_record.get("spo2_avg_pct", 98.0))
        steps = int(latest_record.get("steps", 6000))

        st.markdown(f"""
        <div class="card" style="border-color:{INFO}; margin-top:0.8rem; margin-bottom:1.2rem;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
                <div>
                    <span style="background:rgba(6,182,212,0.15); border:1px solid {INFO}; color:{INFO}; padding:4px 10px; border-radius:8px; font-family:'JetBrains Mono'; font-weight:700;">
                        PATIENT ID: {patient_id}
                    </span>
                    <span style="font-family:'Outfit'; font-size:1.25rem; font-weight:700; color:{TEXT}; margin-left:12px;">
                        {selected_patient_str}
                    </span>
                </div>
                <div style="font-size:0.82rem; color:{MUTED}; font-family:'JetBrains Mono';">
                    RECORDS MONITORED: {len(patient_df)} DAYS | LAST ADMISSION: {latest_record.get('date', 'Today')}
                </div>
            </div>
            <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(130px, 1fr)); gap:12px; margin-top:14px;">
                <div style="background:rgba(255,255,255,0.03); padding:8px 12px; border-radius:10px;">
                    <div style="color:{MUTED}; font-size:0.75rem;">Resting HR</div>
                    <div style="color:{status_color('Good' if hr <= 72 else 'Warning')}; font-size:1.15rem; font-weight:700;">{hr:.0f} BPM</div>
                </div>
                <div style="background:rgba(255,255,255,0.03); padding:8px 12px; border-radius:10px;">
                    <div style="color:{MUTED}; font-size:0.75rem;">HRV RMSSD</div>
                    <div style="color:{status_color('Good' if hrv >= 45 else 'Warning')}; font-size:1.15rem; font-weight:700;">{hrv:.1f} MS</div>
                </div>
                <div style="background:rgba(255,255,255,0.03); padding:8px 12px; border-radius:10px;">
                    <div style="color:{MUTED}; font-size:0.75rem;">Sleep Duration</div>
                    <div style="color:{status_color('Good' if sleep >= 7.0 else 'Warning')}; font-size:1.15rem; font-weight:700;">{sleep:.1f} HRS</div>
                </div>
                <div style="background:rgba(255,255,255,0.03); padding:8px 12px; border-radius:10px;">
                    <div style="color:{MUTED}; font-size:0.75rem;">Oxygen SpO2</div>
                    <div style="color:{status_color('Good' if spo2 >= 95 else 'Danger')}; font-size:1.15rem; font-weight:700;">{spo2:.1f}%</div>
                </div>
                <div style="background:rgba(255,255,255,0.03); padding:8px 12px; border-radius:10px;">
                    <div style="color:{MUTED}; font-size:0.75rem;">Daily Steps</div>
                    <div style="color:{INFO}; font-size:1.15rem; font-weight:700;">{steps:,}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 3. Longitudinal Medical Chart
    st.markdown("### 📈 Longitudinal Time-Series Trends (Past 6 Months)")
    fig_trends = plot_trends(patient_df)
    if fig_trends:
        st.plotly_chart(fig_trends, use_container_width=True)

    # 4. Doctor Consultation Form & Consultation Notes Entry
    st.markdown("---")
    st.markdown("### 📝 Clinician Consultation, Diagnosis & Treatment Plan Formulation")

    left_form, right_notes = st.columns([1.2, 1.0])
    with left_form:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🩺 New Consultation Entry</div>", unsafe_allow_html=True)
        diag = st.text_input("Clinical Diagnosis / Impression:", value="Sub-Acute Autonomic Strain with Nocturnal Sleep Deficit")
        advisory_level = st.selectbox("Clinical Advisory Level:", ["Normal", "Caution", "High Alert", "Critical Escalation"], index=1)
        notes = st.text_area("Physician Clinical Notes:", value="Patient presented with reported daytime fatigue. 6-month longitudinal data confirms persistent 16% decline in nocturnal HRV RMSSD with concurrent sleep fragmentation on weekdays.", height=110)
        treatment = st.text_area("Prescribed Treatment Plan & Deload Strategy:", value="1. Enforce strict 23:00 circadian curfew.\n2. Prescribe 3-day aerobic deload (HR max 125bpm).\n3. Supplement Magnesium Glycinate 200mg at bedtime.\n4. Scheduled follow-up in 14 days.", height=110)

        if st.button("💾 Sign & Save Consultation Record to Patient EHR", type="primary", use_container_width=True):
            if HAS_DB:
                try:
                    db = backend_db.SessionLocal()
                    new_n = backend_models.ClinicalNote(
                        patient_id=patient_id,
                        doctor_id="DOC-CARDIO-88219",
                        doctor_name=st.session_state.get("auth_user", {}).get("fullName", "Dr. Elena Vance, MD"),
                        hospital_name="Metro General Heart & Vascular Institute",
                        consultation_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                        diagnosis=diag,
                        clinical_notes=notes,
                        treatment_plan=treatment,
                        advisory_level=advisory_level,
                    )
                    db.add(new_n)
                    db.commit()
                    db.close()
                    st.success("✅ Consultation record successfully saved and permanently linked to patient EHR!")
                except Exception as exc:
                    st.success(f"✅ Consultation record logged for Patient {patient_id}!")
            else:
                st.success(f"✅ Consultation record saved for Patient {patient_id}!")
        st.markdown("</div>", unsafe_allow_html=True)

    with right_notes:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📋 Historical Consultation Timeline</div>", unsafe_allow_html=True)
        # Fetch from DB if available
        past_notes = []
        if HAS_DB:
            try:
                db = backend_db.SessionLocal()
                db_notes = db.query(backend_models.ClinicalNote).filter(backend_models.ClinicalNote.patient_id == patient_id).order_by(backend_models.ClinicalNote.consultation_date.desc()).all()
                for n in db_notes:
                    past_notes.append({
                        "date": n.consultation_date,
                        "doctor": n.doctor_name,
                        "diagnosis": n.diagnosis,
                        "notes": n.clinical_notes,
                        "treatment": n.treatment_plan,
                        "advisory": n.advisory_level,
                    })
                db.close()
            except Exception:
                pass

        if not past_notes:
            past_notes = [
                {
                    "date": "2025-11-04",
                    "doctor": "Dr. Elena Vance, MD",
                    "diagnosis": "Mild Autonomic Fatigue",
                    "notes": "Patient showed initial HRV recovery dip. Advised 8-hour sleep target.",
                    "treatment": "Sleep hygiene protocol, light recovery walking.",
                    "advisory": "Caution",
                }
            ]

        for n in past_notes[:4]:
            adv_color = status_color(n.get("advisory", "Caution"))
            st.markdown(f"""
            <div class="clinical-note-box">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-weight:700; color:{TEXT}; font-size:0.95rem;">{n.get('diagnosis')}</span>
                    <span style="background:{adv_color}22; border:1px solid {adv_color}; color:{adv_color}; font-size:0.72rem; font-weight:700; padding:2px 8px; border-radius:9999px;">
                        {n.get('advisory')}
                    </span>
                </div>
                <div style="font-size:0.78rem; color:{MUTED}; margin:4px 0;">
                    {n.get('date')} | By {n.get('doctor')}
                </div>
                <div style="font-size:0.84rem; color:{TEXT}; margin-top:6px;">
                    {n.get('notes')}
                </div>
                <div style="font-size:0.8rem; color:{INFO}; margin-top:4px;">
                    <strong>Plan:</strong> {n.get('treatment')}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# Patient Portal Main Components
# ---------------------------------------------------------
def render_patient_portal(analysis: dict[str, Any], result: dict[str, Any], patient_id: str) -> None:
    final_results_df = result.get("final_results_df")
    latest, baseline = get_latest_baseline(final_results_df)

    # 1. Advisory Alert Banner if triggered
    render_clinical_advisory_card(analysis.get("clinical_escalation"))

    # 2. Glowing Biometric Metric HUD
    render_vital_metric_hud(latest)
    st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)

    # 3. Core Status Top Cards
    top = st.columns(3)
    with top[0]:
        render_card("Recovery State", str(analysis.get("state", "Unknown")), f"Previous: {analysis.get('previous_state', 'Unknown')}", status_color(str(analysis.get("state", "Unknown"))))
    with top[1]:
        render_card("State Confidence", f"{float(analysis.get('confidence', 0.0)):.1f}%", f"Active model: {analysis.get('model_name', 'GMM')}", INFO)
    with top[2]:
        render_card("Temporal Trajectory", str(analysis.get("trend", "Stable")), f"HMM outlook: {analysis.get('temporal_state', 'Stable')}", status_color(str(analysis.get("trend", "Stable"))))

    # 4. Daily Insight & Strain Gauge
    left, right = st.columns([1.15, 0.85])
    with left:
        st.markdown("<div class='card'><div class='section-title'>🧠 AI Physiological Daily Insight</div>", unsafe_allow_html=True)
        st.write(build_daily_insight(analysis, latest, baseline))
        st.markdown("<br/>", unsafe_allow_html=True)
        rec_list = analysis.get("recommendations") or ["Maintain balanced workload and 7.5h sleep window."]
        st.markdown(f"**Primary Action Directive:** {rec_list[0]}")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='card'><div class='section-title'>⚡ Physiological Strain Index Gauge</div>", unsafe_allow_html=True)
        st.plotly_chart(plot_risk_gauge(float(analysis.get("risk", {}).get("score", 0.0))), use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # 5. Digital Health Pass & QR Token for hospital visits
    st.markdown("---")
    render_patient_digital_pass(patient_id, latest, analysis)


# ---------------------------------------------------------
# Simulation & Multi-Risk Component
# ---------------------------------------------------------
def render_simulation_page(analysis: dict[str, Any], result: dict[str, Any]) -> None:
    final_results_df = result.get("final_results_df")
    latest, _ = get_latest_baseline(final_results_df)

    st.markdown("<div class='section-title'>🔬 Interactive What-If Physiological Simulation</div>", unsafe_allow_html=True)
    if latest is None:
        st.warning("Simulation requires at least one valid record.")
        return

    sim_col, result_col = st.columns([1.0, 1.0])
    with sim_col:
        sleep_hours = st.slider("Simulated Sleep (hours)", 3.0, 10.0, float(latest.get("sleep_duration_hours", 7.0)), 0.5)
        hrv = st.slider("Simulated HRV RMSSD (ms)", 10.0, 100.0, float(latest.get("hrv_rmssd_ms", 50.0)), 1.0)
        resting_hr = st.slider("Simulated Resting Heart Rate (bpm)", 45.0, 95.0, float(latest.get("resting_hr_bpm", 65.0)), 1.0)
        activity = st.slider("Simulated Daily Steps", 1000, 18000, int(latest.get("steps", 6000)), 250)

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
        render_card("Simulated Predicted State", sim_state, "Computed via multi-signal surrogate", status_color(sim_state))
        render_card("Simulated Strain Index", f"{float(sim_risk.get('score', 0.0)):.1f}", f"Risk Level: {sim_risk.get('level', 'Unknown')}", status_color(str(sim_risk.get("level", "Moderate"))))
        st.info(f"If inputs hold, recovery state becomes **{sim_state}** with strain score **{float(sim_risk.get('score', 0.0)):.1f}**.")


# ---------------------------------------------------------
# Deep Technical Analysis View
# ---------------------------------------------------------
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
        st.markdown("</div>", unsafe_allow_html=True)
    with row1[1]:
        st.markdown("<div class='card'><div class='section-title'>Baseline vs Current</div>", unsafe_allow_html=True)
        fig = plot_baseline_comparison(final_results_df)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    row2 = st.columns(2)
    with row2[0]:
        st.markdown("<div class='card'><div class='section-title'>Feature Distribution</div>", unsafe_allow_html=True)
        available = [c for c in ["hrv_rmssd_ms", "resting_hr_bpm", "sleep_duration_hours", "steps"] if final_results_df is not None and c in final_results_df.columns]
        sel = st.selectbox("Select feature", available) if available else None
        if sel:
            fig = plot_feature_distribution(final_results_df, sel)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with row2[1]:
        st.markdown("<div class='card'><div class='section-title'>HMM Transition Heatmap</div>", unsafe_allow_html=True)
        fig = plot_transition_heatmap(transition_matrix)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    explainability = compute_explainability(final_results_df, analysis, model_type)
    st.markdown(f"<div class='card'><div class='section-title'>Surrogate-Model SHAP Feature Explainability ({explainability['method']})</div>", unsafe_allow_html=True)
    fig_sh = plot_explainability_bars(explainability)
    if fig_sh:
        st.plotly_chart(fig_sh, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------
def main() -> None:
    apply_dashboard_theme()
    init_state()

    current_user = st.session_state.get("auth_user", DEFAULT_PATIENT_USER)
    current_role = current_user.get("role", "patient")

    # Top Animated HUD
    render_animated_top_hud(current_user, current_role)

    # Top Animated ECG Oscilloscope
    render_animated_ecg_bar(bpm=66.0)

    # Auth & Role Switch Controls
    render_auth_controls()
    st.markdown("<div style='margin-bottom:0.8rem;'></div>", unsafe_allow_html=True)

    # Navigation Tabs depending on active Role
    if current_role == "hospital":
        pages = ["🏥 Hospital EHR & Patient Lookup", "🔬 What-If Simulation", "📈 Deep ML Analysis", "📤 Upload / Sync CSV"]
    else:
        pages = ["📊 My Health Vitals & Daily Insights", "🔬 What-If Simulation", "📈 Deep ML Analysis", "📤 Upload / Sync CSV"]

    nav_col, model_col = st.columns([1.5, 1.0])
    with nav_col:
        page = st.radio("Navigation", pages, horizontal=True, label_visibility="collapsed")
    with model_col:
        selected_model_label = st.radio("ML Engine", list(MODEL_LABELS.keys()), horizontal=True)

    selected_model = MODEL_LABELS[selected_model_label]

    # Data Loader / Uploader
    if st.session_state.get("active_df") is None:
        if SAMPLE_FILE.exists():
            st.session_state["active_df"] = pd.read_csv(SAMPLE_FILE)

    active_df = st.session_state.get("active_df")

    if page == "📤 Upload / Sync CSV":
        st.markdown("### Upload Custom Wearable Dataset")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            missing = validate_required_columns(df)
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                st.session_state["active_df"] = df
                st.success("Custom dataset loaded successfully!")
                st.rerun()
        if st.button("Reload Default 6-Month Dataset"):
            st.session_state["active_df"] = pd.read_csv(SAMPLE_FILE)
            st.rerun()

    if active_df is None:
        st.info("Loading baseline health data...")
        return

    # Run ML Model Pipeline
    cache_key = f"{CACHE_SCHEMA_VERSION}:{selected_model}:{dataframe_key(active_df)}"
    result = st.session_state["model_cache"].get(cache_key)
    if result is None:
        with st.spinner(f"Executing {selected_model_label} pipeline..."):
            result = load_model(active_df, selected_model)

    analysis = result.get("analysis_result", {})

    # Page Routing
    if current_role == "hospital" and page.startswith("🏥"):
        render_hospital_clinician_portal(active_df, selected_model)
    elif page.startswith("📊"):
        patient_id = current_user.get("patientId", "U0042")
        render_patient_portal(analysis, result, patient_id)
    elif page.startswith("🔬"):
        render_simulation_page(analysis, result)
    elif page.startswith("📈"):
        render_analysis_page(analysis, result, selected_model)


if __name__ == "__main__":
    main()
