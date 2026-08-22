import sys
from pathlib import Path
from typing import Any, Dict
from sqlalchemy.orm import Session
import pandas as pd

# Dynamically resolve ML project root directory (one level up from backend/)
ML_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(ML_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(ML_PROJECT_DIR))

from run_inference import load_model_outputs, compute_state
from src.preprocessing import preprocess_for_modeling
from src.recommendation_engine import generate_recommendations
from src.risk_scoring import calculate_risk_score, evaluate_clinical_persistence
from src.cohort_benchmarks import benchmark_against_cohort

from models import HealthLog


def predict_health_risk(device_user_id: str, db: Session, city: str = "Bangalore", model_type: str = "gmm") -> Dict[str, Any]:
    """
    Evaluates stored historical logs for device_user_id in PostgreSQL:
    - < 7 records: Returns status 'insufficient_data'
    - 7 to 29 records: Calculates baseline z-scores using last 7 days (window_used='7_days')
    - >= 30 records: Calculates baseline z-scores using past 30 days (window_used='30_days')
    """
    logs = (
        db.query(HealthLog)
        .filter(HealthLog.device_user_id == device_user_id)
        .order_by(HealthLog.record_date.asc())
        .all()
    )

    n_records = len(logs)
    if n_records < 7:
        return {
            "status": "insufficient_data",
            "message": f"Please collect at least 7 days of health data before generating predictions. Currently available: {n_records} days.",
            "days_available": n_records,
            "window_used": "none",
            "state": "Insufficient Data",
            "confidence": 0.0,
            "riskScore": 0.0,
            "riskLevel": "Unknown",
            "clinicalAdvisoryLevel": "Normal",
            "clinicalSummaryMessage": f"Need {7 - n_records} more day(s) of health data.",
            "recommendations": ["Continue wearing your device to complete your initial 7-day baseline."],
        }

    if n_records >= 30:
        target_logs = logs[-30:]
        window_used = "30_days"
    else:
        target_logs = logs[-7:]
        window_used = "7_days"

    # Convert DB logs to DataFrame
    rows = []
    for log in target_logs:
        sleep_hrs = (log.sleep_minutes / 60.0) if (log.sleep_minutes and log.sleep_minutes > 24) else (log.sleep_minutes or 7.0)
        rows.append({
            "user_id": log.device_user_id,
            "date": log.record_date,
            "resting_hr_bpm": float(log.heart_rate or 65.0),
            "hrv_rmssd_ms": 45.0,  # Default fallback if HRV sensor unavailable
            "sleep_duration_hours": float(sleep_hrs),
            "steps": float(log.steps or 5000),
            "spo2_avg_pct": float(log.oxygen_saturation or 98.0),
            "caffeine_mg": 150.0,
            "screen_time_min": 180.0,
            "age": 35,
            "gender": "male",
        })

    user_df = pd.DataFrame(rows)

    # Save to temp CSV path for preprocess_for_modeling
    temp_dir = Path(__file__).resolve().parent / "temp_cache"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_csv_path = temp_dir / f"user_{device_user_id}.csv"
    user_df.to_csv(temp_csv_path, index=False)

    # Preprocess with ML pipeline
    outputs = preprocess_for_modeling(source=str(temp_csv_path))
    feature_df = outputs["feature_df"].copy()
    feature_df, cluster_out, hmm_bundle = load_model_outputs(feature_df, model_type=model_type)

    final_results_df, analysis = compute_state(model_type, hmm_bundle["labeled_df"], hmm_bundle, {}, cluster_out)

    recommendations = generate_recommendations(analysis)
    clinical = analysis.get("clinical_escalation", {})
    cohort = analysis.get("cohort_benchmarks", {})

    return {
        "status": "success",
        "message": f"Prediction generated using {window_used.replace('_', ' ')} baseline window.",
        "days_available": n_records,
        "window_used": window_used,
        "state": analysis.get("state", "Baseline"),
        "previousState": analysis.get("previous_state", "Baseline"),
        "confidence": analysis.get("confidence", 95.0),
        "trend": analysis.get("trend", "Stable"),
        "riskScore": analysis.get("risk", {}).get("score", 0.0),
        "riskLevel": analysis.get("risk", {}).get("level", "Low"),
        "severityScore": analysis.get("risk", {}).get("severity_score", 0.0),
        "clinicalAdvisoryLevel": clinical.get("advisory_level", "Normal"),
        "clinicalSummaryMessage": clinical.get("clinical_summary_message", "All physiological signals normal."),
        "persistentTriggers": clinical.get("persistent_triggers", []),
        "recommendations": recommendations,
        "cohortBenchmarks": cohort,
        "multiRisk": analysis.get("multi_risk", {}),
    }
