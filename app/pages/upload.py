from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from app.dashboard_utils import apply_dashboard_theme
from run_inference import run_pipeline
from src.preprocessing import validate_required_columns

DEFAULT_OUTPUT_PATH = ROOT_DIR / "outputs" / "final_results.csv"
SAMPLE_FILE = ROOT_DIR / "data" / "wearables_health_6mo_daily.csv"

st.set_page_config(page_title="Upload Data", layout="wide")
apply_dashboard_theme()

for key in ["analysis_result", "output_path", "final_results_df", "data", "feature_df", "transition_matrix", "state_distribution", "performance"]:
    if key not in st.session_state:
        st.session_state[key] = None

st.title("Upload Page")
st.caption("Upload wearable physiological data, validate required signals, and run the complete analytics pipeline.")

left, right = st.columns([1, 1.1])

with left:
    uploaded_file = st.file_uploader("Upload wearable CSV", type=["csv"])
    use_sample = st.button("Use bundled sample dataset")
    run_clicked = st.button("Run Full Analysis", type="primary", use_container_width=True)

with right:
    st.subheader("Accepted Input")
    st.write("The app accepts either the full project dataset or a minimal physiological schema.")
    st.write("Required minimal columns: `user_id`, `date`, `resting_hr_bpm`, `hrv_rmssd_ms`, `spo2_avg_pct`, `steps`, `sleep_duration_hours`.")

source_df = None
source_name = None
if uploaded_file is not None:
    source_df = pd.read_csv(uploaded_file)
    source_name = uploaded_file.name
elif use_sample:
    source_df = pd.read_csv(SAMPLE_FILE)
    source_name = SAMPLE_FILE.name

if source_df is not None:
    missing_columns = validate_required_columns(source_df)
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
    else:
        st.success(f"Dataset ready: {source_name}")
        st.session_state["data"] = source_df
        st.dataframe(source_df.head(10), use_container_width=True)

if run_clicked:
    if source_df is None and st.session_state.get("data") is None:
        st.warning("Upload a CSV or load the sample dataset first.")
    else:
        try:
            active_df = source_df if source_df is not None else st.session_state["data"]
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
                active_df.to_csv(temp_file.name, index=False)
                temp_path = temp_file.name

            with st.spinner("Running preprocessing, clustering, HMM, explainability, and recommendation pipeline..."):
                result = run_pipeline(temp_path, output_path=str(DEFAULT_OUTPUT_PATH))

            st.session_state["analysis_result"] = result["analysis_result"]
            st.session_state["output_path"] = str(result["output_path"])
            st.session_state["final_results_df"] = result["final_results_df"]
            st.session_state["feature_df"] = result["feature_df"]
            st.session_state["data"] = result["raw_data"]
            st.session_state["transition_matrix"] = result["transition_matrix"]
            st.session_state["state_distribution"] = result["state_distribution"]
            st.session_state["performance"] = result["analysis_result"].get("performance")
            st.success("Pipeline complete. Move to the Main Dashboard, Analysis, or Recommendations page.")
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
