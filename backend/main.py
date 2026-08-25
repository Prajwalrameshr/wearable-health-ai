from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import Base, engine, get_db
from models import HealthLog
from ml_engine import predict_health_risk

# Automatically create database tables if they do not exist
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Wearable Health AI Backend",
    description="FastAPI Backend orchestrating Android Health Connect, PostgreSQL Database, and GMM/HMM ML Models.",
    version="2.2.0",
)

# Enable CORS for cross-origin / mobile app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Schemas (Matching IEEE Paper Metrics)
class HealthPayload(BaseModel):
    deviceUserId: str = Field(..., example="android_device_9a8b7c")
    steps: int = Field(0, example=4714)
    calories: Optional[float] = Field(None, example=2150.0)             # Calories burned (kcal)
    heartRate: Optional[float] = Field(None, example=69.8)               # All-day mean HR
    heartRateResting: Optional[float] = Field(None, example=61.2)        # Resting HR
    hrvRmssdAvg: Optional[float] = Field(None, example=42.7)             # Nocturnal HRV RMSSD (ms)
    oxygenSaturation: Optional[float] = Field(None, example=96.7)        # Mean SpO2 (%)
    oxygenSaturationNadir: Optional[float] = Field(None, example=94.1)  # Minimum SpO2 (%)
    sleepMinutes: int = Field(0, example=375)
    recordStartTime: Optional[str] = Field(None, example="2025-11-10T00:00:00Z")
    recordEndTime: Optional[str] = Field(None, example="2025-11-10T23:59:59Z")
    collectedAt: Optional[str] = Field(None, example="2025-11-10T23:55:00Z")
    city: Optional[str] = Field("Bangalore", example="Bangalore")
    modelType: Optional[str] = Field("gmm", example="gmm")


class HealthResponse(BaseModel):
    status: str
    message: Optional[str] = None
    days_available: int
    window_used: str
    userId: Optional[str] = None
    date: Optional[str] = None
    modelType: Optional[str] = None
    state: Optional[str] = None
    previousState: Optional[str] = None
    confidence: Optional[float] = None
    trend: Optional[str] = None
    riskScore: Optional[float] = None
    riskLevel: Optional[str] = None
    severityScore: Optional[float] = None
    clinicalAdvisoryLevel: Optional[str] = None
    clinicalSummaryMessage: Optional[str] = None
    persistentTriggers: Optional[List[str]] = None
    recommendations: Optional[List[str]] = None
    cohortBenchmarks: Optional[Dict[str, Any]] = None
    multiRisk: Optional[Dict[str, Any]] = None
    environment: Optional[Dict[str, Any]] = None


@app.get("/")
def index():
    return {
        "service": "Wearable Health AI FastAPI Backend",
        "version": "2.2.0",
        "status": "running",
        "docs_url": "/docs",
        "endpoints": {
            "health_check": "GET /health",
            "send_health_records": "POST /api/health/records"
        }
    }


@app.get("/health")
@app.get("/api/health/status")
def health_check(db: Session = Depends(get_db)):
    try:
        db_log_count = db.query(HealthLog).count()
        return {
            "status": "healthy",
            "database": "connected",
            "total_health_logs": db_log_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database connection issue: {str(exc)}"
        )


@app.post("/api/health/records", response_model=HealthResponse)
def receive_health_records(payload: HealthPayload, db: Session = Depends(get_db)):
    try:
        if payload.recordStartTime and "T" in payload.recordStartTime:
            record_date = payload.recordStartTime.split("T")[0]
        else:
            record_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        existing_log = (
            db.query(HealthLog)
            .filter(
                HealthLog.device_user_id == payload.deviceUserId,
                HealthLog.record_date == record_date
            )
            .first()
        )

        # Field-Aware Upsert
        if existing_log:
            existing_log.steps = payload.steps
            existing_log.heart_rate = payload.heartRate
            existing_log.oxygen_saturation = payload.oxygenSaturation
            existing_log.sleep_minutes = payload.sleepMinutes
            existing_log.record_start_time = payload.recordStartTime
            existing_log.record_end_time = payload.recordEndTime
            existing_log.collectedAt = payload.collectedAt

            if payload.calories is not None:
                existing_log.calories = payload.calories
            if payload.heartRateResting is not None:
                existing_log.heart_rate_resting = payload.heartRateResting
            if payload.hrvRmssdAvg is not None:
                existing_log.hrv_rmssd_avg = payload.hrvRmssdAvg
            if payload.oxygenSaturationNadir is not None:
                existing_log.oxygen_saturation_nadir = payload.oxygenSaturationNadir

            log_entry = existing_log
        else:
            log_entry = HealthLog(
                device_user_id=payload.deviceUserId,
                record_date=record_date,
                steps=payload.steps,
                calories=payload.calories,
                heart_rate=payload.heartRate,
                heart_rate_resting=payload.heartRateResting,
                hrv_rmssd_avg=payload.hrvRmssdAvg,
                oxygen_saturation=payload.oxygenSaturation,
                oxygen_saturation_nadir=payload.oxygenSaturationNadir,
                sleep_minutes=payload.sleepMinutes,
                record_start_time=payload.recordStartTime,
                record_end_time=payload.recordEndTime,
                collectedAt=payload.collectedAt,
            )
            db.add(log_entry)

        db.commit()
        db.refresh(log_entry)

        city_name = payload.city or "Bangalore"
        model_name = payload.modelType or "gmm"
        prediction = predict_health_risk(
            device_user_id=payload.deviceUserId,
            db=db,
            city=city_name,
            model_type=model_name,
        )

        log_entry.predicted_state = str(prediction.get("state"))
        log_entry.risk_score = float(prediction.get("riskScore", 0.0))
        log_entry.risk_level = str(prediction.get("riskLevel"))
        log_entry.clinical_advisory_level = str(prediction.get("clinicalAdvisoryLevel"))
        log_entry.clinical_summary_message = str(prediction.get("clinicalSummaryMessage"))
        log_entry.window_used = str(prediction.get("window_used"))

        db.commit()

        prediction["userId"] = payload.deviceUserId
        prediction["date"] = record_date
        prediction["modelType"] = model_name

        return prediction

    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process health payload: {str(exc)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
