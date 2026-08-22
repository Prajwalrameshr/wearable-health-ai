from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request
import pandas as pd

from run_inference import compute_state, load_model_outputs
from src.cohort_benchmarks import benchmark_against_cohort
from src.preprocessing import preprocess_for_modeling
from src.recommendation_engine import generate_recommendations
from src.risk_scoring import calculate_risk_score, evaluate_clinical_persistence

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "wearables_health_6mo_daily.csv"


def _process_android_payload(payload: dict[str, Any]) -> dict[str, Any]:
    device_user_id = str(payload.get("deviceUserId") or payload.get("user_id") or "android_user").strip()
    steps = float(payload.get("steps") or 5000.0)
    heart_rate = float(payload.get("heartRate") or payload.get("resting_hr_bpm") or 65.0)
    oxygen_sat = float(payload.get("oxygenSaturation") or payload.get("spo2_avg_pct") or 98.0)
    sleep_minutes = float(payload.get("sleepMinutes") or 420.0)
    sleep_hours = sleep_minutes / 60.0 if sleep_minutes > 24.0 else sleep_minutes

    raw_start = str(payload.get("recordStartTime") or "")
    if raw_start and "T" in raw_start:
        record_date = raw_start.split("T")[0]
    else:
        record_date = datetime.now().strftime("%Y-%m-%d")

    model_type = str(payload.get("model_type") or "gmm").lower()

    # Load baseline dataset
    if DATA_PATH.exists():
        raw_df = pd.read_csv(DATA_PATH)
    else:
        raw_df = pd.DataFrame()

    new_row = {
        "user_id": device_user_id,
        "date": record_date,
        "resting_hr_bpm": heart_rate,
        "hrv_rmssd_ms": float(payload.get("hrv_rmssd_ms") or 45.0),
        "sleep_duration_hours": sleep_hours,
        "steps": steps,
        "spo2_avg_pct": oxygen_sat,
        "caffeine_mg": float(payload.get("caffeine_mg") or 150.0),
        "screen_time_min": float(payload.get("screen_time_min") or 180.0),
        "age": int(payload.get("age") or 35),
        "gender": str(payload.get("gender") or "male"),
    }

    combined_df = pd.concat([raw_df, pd.DataFrame([new_row])], ignore_index=True)

    # Save to temp CSV path for preprocessing
    temp_dir = BASE_DIR / "outputs"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_csv_path = temp_dir / f"api_payload_{device_user_id}.csv"
    combined_df.to_csv(temp_csv_path, index=False)

    # Preprocess & run ML pipeline
    outputs = preprocess_for_modeling(source=str(temp_csv_path))
    feature_df = outputs["feature_df"].copy()
    feature_df, cluster_out, hmm_bundle = load_model_outputs(feature_df, model_type=model_type)

    final_results_df, analysis = compute_state(model_type, hmm_bundle["labeled_df"], hmm_bundle, {}, cluster_out)

    # Add recommendations
    recommendations = generate_recommendations(analysis)
    analysis["recommendations"] = recommendations

    clinical = analysis.get("clinical_escalation", {})
    cohort = analysis.get("cohort_benchmarks", {})

    return {
        "status": "success",
        "userId": device_user_id,
        "date": record_date,
        "modelType": model_type,
        "state": analysis.get("state", "Baseline"),
        "previousState": analysis.get("previous_state", "Baseline"),
        "confidence": analysis.get("confidence", 95.0),
        "trend": analysis.get("trend", "Stable"),
        "riskScore": analysis.get("risk", {}).get("score", 0.0),
        "riskLevel": analysis.get("risk", {}).get("level", "Low"),
        "severityScore": analysis.get("risk", {}).get("severity_score", 0.0),
        "clinicalAdvisoryLevel": clinical.get("advisory_level", "Normal"),
        "clinicalSummaryMessage": clinical.get("clinical_summary_message", "All signals normal."),
        "persistentTriggers": clinical.get("persistent_triggers", []),
        "recommendations": recommendations,
        "cohortBenchmarks": cohort,
        "multiRisk": analysis.get("multi_risk", {}),
    }



@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Wearable Health AI ML REST API",
        "status": "running",
        "endpoints": {
            "health_check": "/health",
            "send_health_records": "POST /api/health/records"
        }
    })


@app.route("/health", methods=["GET"])
@app.route("/api/health/status", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "dataset_loaded": DATA_PATH.exists(),
    })


@app.route("/api/health/records", methods=["POST"])
def receive_health_records():
    try:
        data = request.get_json(force=True, silent=True) or {}
        result = _process_android_payload(data)
        return jsonify(result), 200
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


if __name__ == "__main__":
    print("Starting Wearable Health AI REST API Server on http://0.0.0.0:5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=False)
