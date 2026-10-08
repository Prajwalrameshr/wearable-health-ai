from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import Depends, FastAPI, HTTPException, status, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import pandas as pd

from database import Base, SessionLocal, engine, get_db
from models import HealthLog, User, ClinicalNote
from ml_engine import predict_health_risk

# Automatically create database tables if they do not exist
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Wearable Health AI Enterprise Backend",
    description="Enterprise FastAPI Backend supporting Role-Based Access Control (Patient & Hospital/Doctor Portals), Android Health Connect, and GMM/HMM ML Models.",
    version="3.0.0",
)

# Enable CORS for cross-origin / mobile app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def hash_password(password: str) -> str:
    """Computes SHA-256 hash for secure password storage."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


# ---------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------

class UserRegisterRequest(BaseModel):
    username: str = Field(..., example="patient_john")
    password: str = Field(..., example="secret123")
    role: str = Field("patient", example="patient")  # 'patient' or 'hospital'
    fullName: Optional[str] = Field(None, example="John Doe")
    email: Optional[str] = Field(None, example="john@example.com")
    patientId: Optional[str] = Field(None, example="U0042")
    hospitalName: Optional[str] = Field(None, example="Metro General Hospital")
    department: Optional[str] = Field(None, example="Cardiology")
    licenseNumber: Optional[str] = Field(None, example="MD-98412")


class UserLoginRequest(BaseModel):
    username: str = Field(..., example="patient")
    password: str = Field(..., example="health2026")


class AuthResponse(BaseModel):
    status: str
    token: str
    user: Dict[str, Any]


class ClinicalNoteCreate(BaseModel):
    patientId: str = Field(..., example="U0042")
    doctorName: Optional[str] = Field(None, example="Dr. Elena Vance, MD")
    hospitalName: Optional[str] = Field(None, example="Metro General Hospital")
    diagnosis: Optional[str] = Field(None, example="Nocturnal Autonomic Strain")
    clinicalNotes: str = Field(..., example="Patient shows continuous HRV drop over past 7 days with sleep fragmentation.")
    treatmentPlan: Optional[str] = Field(None, example="Prescribe 3-day recovery deload, circadian sleep hygiene protocol, follow up in 14 days.")
    advisoryLevel: Optional[str] = Field("Caution", example="Caution")


class HealthPayload(BaseModel):
    deviceUserId: str = Field(..., example="android_device_9a8b7c")
    steps: int = Field(0, example=4714)
    distanceKm: Optional[float] = Field(None, example=3.5)
    distanceMeters: Optional[float] = Field(None, example=3500.0)
    calories: Optional[float] = Field(None, example=2150.0)
    caloriesKcal: Optional[float] = Field(None, example=2150.0)
    heartRate: Optional[float] = Field(None, example=69.8)
    averageHeartRate: Optional[float] = Field(None, example=69.8)
    heartRateResting: Optional[float] = Field(None, example=61.2)
    hrvRmssdAvg: Optional[float] = Field(None, example=42.7)
    oxygenSaturation: Optional[float] = Field(None, example=96.7)
    oxygenSaturationNadir: Optional[float] = Field(None, example=94.1)
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


# ---------------------------------------------------------
# DB Seed Helper (Ensures Out-of-the-Box Demo Experience)
# ---------------------------------------------------------

def seed_demo_data():
    db = SessionLocal()
    try:
        # 1. Seed demo accounts if missing
        patient_user = db.query(User).filter(User.username == "patient").first()
        if not patient_user:
            demo_patient = User(
                username="patient",
                password_hash=hash_password("health2026"),
                role="patient",
                full_name="Alex Mercer",
                email="alex.mercer@health.ai",
                patient_id="U0042",
            )
            db.add(demo_patient)

        doctor_user = db.query(User).filter(User.username == "doctor").first()
        if not doctor_user:
            demo_doctor = User(
                username="doctor",
                password_hash=hash_password("clinical2026"),
                role="hospital",
                full_name="Dr. Elena Vance, MD",
                email="elena.vance@metrohealth.org",
                hospital_name="Metro General Heart & Vascular Institute",
                department="Cardiology & Autonomic Physiology",
                license_number="MD-CARDIO-88219",
            )
            db.add(demo_doctor)

        db.commit()

        # 2. Check if patient U0042 has health logs in DB. If not, seed from CSV
        u0042_count = db.query(HealthLog).filter(HealthLog.device_user_id == "U0042").count()
        if u0042_count == 0:
            csv_path = Path(__file__).resolve().parent.parent / "data" / "wearables_health_6mo_daily.csv"
            if csv_path.exists():
                import math
                def _sf(val, default=0.0):
                    try:
                        f = float(val)
                        return default if math.isnan(f) else f
                    except Exception:
                        return default

                def _si(val, default=0):
                    try:
                        f = float(val)
                        return default if math.isnan(f) else int(f)
                    except Exception:
                        return default

                df = pd.read_csv(csv_path)
                u42_df = df[df["user_id"] == "U0042"].sort_values("date")
                for _, row in u42_df.iterrows():
                    sleep_hrs = _sf(row.get("sleep_duration_hours"), 7.0)
                    sleep_min = int(sleep_hrs * 60)
                    hrv_val = _sf(row.get("hrv_rmssd_ms"), 45.0)
                    new_log = HealthLog(
                        device_user_id="U0042",
                        record_date=str(row["date"]),
                        steps=_si(row.get("steps"), 6000),
                        distance_km=_sf(row.get("distance_km"), 4.2),
                        calories=_sf(row.get("calories_kcal"), 2100.0),
                        heart_rate=_sf(row.get("avg_hr_day_bpm"), 72.0),
                        heart_rate_resting=_sf(row.get("resting_hr_bpm"), 64.0),
                        hrv_rmssd_avg=hrv_val,
                        oxygen_saturation=_sf(row.get("spo2_avg_pct"), 98.0),
                        oxygen_saturation_nadir=_sf(row.get("spo2_nadir_pct"), 94.0),
                        sleep_minutes=sleep_min,
                        predicted_state="Baseline" if hrv_val > 40 else "Strain",
                        risk_score=28.5 if hrv_val > 40 else 64.2,
                        risk_level="Low" if hrv_val > 40 else "Moderate",
                        clinical_advisory_level="Normal" if hrv_val > 40 else "Caution",
                        clinical_summary_message="Physiological signals within normative baseline." if hrv_val > 40 else "Sustained autonomic dip observed.",
                        window_used="30_days",
                    )
                    db.add(new_log)
                db.commit()

        # 3. Seed sample clinical notes for patient U0042 if none exist
        notes_count = db.query(ClinicalNote).filter(ClinicalNote.patient_id == "U0042").count()
        if notes_count == 0:
            sample_note = ClinicalNote(
                patient_id="U0042",
                doctor_id="DOC-88219",
                doctor_name="Dr. Elena Vance, MD",
                hospital_name="Metro General Heart & Vascular Institute",
                consultation_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                diagnosis="Mild Autonomic Fatigue & Sleep Fragmentation",
                clinical_notes="Patient visited with reports of afternoon lethargy. Wearable data confirms 18% decline in nocturnal HRV RMSSD over the past 14 days, coupled with sub-6h sleep on weekdays.",
                treatment_plan="1. Enforce 23:00 curfew with zero blue light 60min prior.\n2. Limit caffeine intake after 13:00.\n3. Prescribe magnesium glycinate 200mg at bedtime.\n4. Follow up review in 2 weeks.",
                advisory_level="Caution",
            )
            db.add(sample_note)
            db.commit()

    except Exception as exc:
        db.rollback()
        print(f"Notice: Initial data seeding skipped: {exc}")
    finally:
        db.close()

# Execute demo data seed
seed_demo_data()


# ---------------------------------------------------------
# Core API Routes
# ---------------------------------------------------------

@app.get("/")
def index():
    return {
        "service": "Wearable Health AI Enterprise Backend",
        "version": "3.0.0",
        "status": "running",
        "roles_supported": ["patient", "hospital"],
        "docs_url": "/docs",
        "endpoints": {
            "health_check": "GET /health",
            "auth_login": "POST /api/auth/login",
            "auth_register": "POST /api/auth/register",
            "hospital_patients": "GET /api/hospital/patients",
            "hospital_patient_history": "GET /api/hospital/patient/{patient_id}/history",
            "hospital_notes": "POST /api/hospital/patient/{patient_id}/notes",
            "send_health_records": "POST /api/health/records",
        }
    }


@app.get("/health")
@app.get("/api/health/status")
def health_check(db: Session = Depends(get_db)):
    try:
        db_log_count = db.query(HealthLog).count()
        users_count = db.query(User).count()
        notes_count = db.query(ClinicalNote).count()
        return {
            "status": "healthy",
            "database": "connected",
            "total_health_logs": db_log_count,
            "total_users": users_count,
            "total_clinical_notes": notes_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database connection issue: {str(exc)}"
        )


# ---------------------------------------------------------
# Authentication & Role-Based Access Endpoints
# ---------------------------------------------------------

@app.post("/api/auth/register", response_model=AuthResponse)
def register_user(payload: UserRegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.username == payload.username.strip().lower()).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already exists")

    role = payload.role.strip().lower()
    if role not in {"patient", "hospital"}:
        role = "patient"

    patient_id = payload.patientId or f"PAT-{datetime.now().strftime('%m%d%H%M')}"
    new_user = User(
        username=payload.username.strip().lower(),
        email=payload.email,
        password_hash=hash_password(payload.password),
        role=role,
        full_name=payload.fullName or payload.username.title(),
        patient_id=patient_id if role == "patient" else None,
        hospital_name=payload.hospitalName if role == "hospital" else None,
        department=payload.department if role == "hospital" else None,
        license_number=payload.licenseNumber if role == "hospital" else None,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    token = f"token_{new_user.role}_{new_user.id}_{int(datetime.now().timestamp())}"
    return {
        "status": "success",
        "token": token,
        "user": {
            "id": new_user.id,
            "username": new_user.username,
            "role": new_user.role,
            "fullName": new_user.full_name,
            "email": new_user.email,
            "patientId": new_user.patient_id,
            "hospitalName": new_user.hospital_name,
            "department": new_user.department,
            "licenseNumber": new_user.license_number,
        }
    }


@app.post("/api/auth/login", response_model=AuthResponse)
def login_user(payload: UserLoginRequest, db: Session = Depends(get_db)):
    pwd_hash = hash_password(payload.password)
    user = (
        db.query(User)
        .filter(User.username == payload.username.strip().lower())
        .filter(User.password_hash == pwd_hash)
        .first()
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials. Use 'patient'/'health2026' or 'doctor'/'clinical2026' for demo access."
        )

    token = f"token_{user.role}_{user.id}_{int(datetime.now().timestamp())}"
    return {
        "status": "success",
        "token": token,
        "user": {
            "id": user.id,
            "username": user.username,
            "role": user.role,
            "fullName": user.full_name,
            "email": user.email,
            "patientId": user.patient_id,
            "hospitalName": user.hospital_name,
            "department": user.department,
            "licenseNumber": user.license_number,
        }
    }


# ---------------------------------------------------------
# Hospital & Doctor Portal Endpoints (Clinician Access)
# ---------------------------------------------------------

@app.get("/api/hospital/overview")
def get_hospital_overview(db: Session = Depends(get_db)):
    """Returns clinic summary stats: patient counts, alerts, average vital metrics."""
    total_logs = db.query(HealthLog).count()
    distinct_patients = [p[0] for p in db.query(HealthLog.device_user_id).distinct().all()]

    high_risk_count = db.query(HealthLog).filter(HealthLog.risk_level == "High").count()
    caution_count = db.query(HealthLog).filter(HealthLog.clinical_advisory_level == "Caution").count()
    alert_count = db.query(HealthLog).filter(HealthLog.clinical_advisory_level.in_(["High Alert", "Critical Escalation"])).count()

    return {
        "totalMonitoredPatients": len(distinct_patients),
        "totalHistoricalLogs": total_logs,
        "activeHighRiskCases": high_risk_count,
        "cautionAdvisoryCases": caution_count,
        "criticalAlertCases": alert_count,
        "activePatientsList": distinct_patients,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/hospital/patients")
def list_hospital_patients(db: Session = Depends(get_db)):
    """
    Returns list of all registered and recorded patients with their latest known vitals and ML risk level.
    """
    distinct_ids = [p[0] for p in db.query(HealthLog.device_user_id).distinct().all()]

    patients = []
    for pid in distinct_ids:
        latest_log = (
            db.query(HealthLog)
            .filter(HealthLog.device_user_id == pid)
            .order_by(HealthLog.record_date.desc())
            .first()
        )
        total_records = db.query(HealthLog).filter(HealthLog.device_user_id == pid).count()
        notes_count = db.query(ClinicalNote).filter(ClinicalNote.patient_id == pid).count()

        patients.append({
            "patientId": pid,
            "lastRecordDate": latest_log.record_date if latest_log else "N/A",
            "totalRecords": total_records,
            "clinicalNotesCount": notes_count,
            "latestState": latest_log.predicted_state if latest_log else "Unknown",
            "latestRiskScore": latest_log.risk_score if latest_log else 0.0,
            "latestRiskLevel": latest_log.risk_level if latest_log else "Unknown",
            "clinicalAdvisoryLevel": latest_log.clinical_advisory_level if latest_log else "Normal",
            "latestRestingHr": latest_log.heart_rate_resting if latest_log else None,
            "latestHrv": latest_log.hrv_rmssd_avg if latest_log else None,
            "latestSleepHours": round(latest_log.sleep_minutes / 60.0, 1) if (latest_log and latest_log.sleep_minutes) else None,
            "latestSpo2": latest_log.oxygen_saturation if latest_log else None,
            "latestSteps": latest_log.steps if latest_log else 0,
        })

    # Sort so urgent/high-risk cases appear at top for doctor triage
    patients.sort(key=lambda x: (x.get("latestRiskScore") or 0.0), reverse=True)
    return {"status": "success", "count": len(patients), "patients": patients}


@app.get("/api/hospital/patient/{patient_id}/history")
def get_patient_history(patient_id: str, limit: int = Query(180, ge=1, le=500), db: Session = Depends(get_db)):
    """
    Clinician retrieves complete longitudinal history and past medical telemetry for a specific patient.
    """
    logs = (
        db.query(HealthLog)
        .filter(HealthLog.device_user_id == patient_id)
        .order_by(HealthLog.record_date.asc())
        .limit(limit)
        .all()
    )

    if not logs:
        # Check if patient exists in project CSV dataset
        csv_path = Path(__file__).resolve().parent.parent / "data" / "wearables_health_6mo_daily.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            matched = df[df["user_id"] == patient_id].sort_values("date")
            if not matched.empty:
                # Dynamically load into DB for permanent access
                for _, row in matched.iterrows():
                    sleep_min = int(float(row.get("sleep_duration_hours", 7.0)) * 60)
                    new_log = HealthLog(
                        device_user_id=patient_id,
                        record_date=str(row["date"]),
                        steps=int(float(row.get("steps", 6000))),
                        distance_km=float(row.get("distance_km", 4.0)),
                        calories=float(row.get("calories_kcal", 2000.0)),
                        heart_rate=float(row.get("avg_hr_day_bpm", 70.0)),
                        heart_rate_resting=float(row.get("resting_hr_bpm", 64.0)),
                        hrv_rmssd_avg=float(row.get("hrv_rmssd_ms", 45.0)),
                        oxygen_saturation=float(row.get("spo2_avg_pct", 98.0)),
                        sleep_minutes=sleep_min,
                        predicted_state="Baseline",
                        risk_score=25.0,
                        risk_level="Low",
                        clinical_advisory_level="Normal",
                    )
                    db.add(new_log)
                db.commit()
                logs = db.query(HealthLog).filter(HealthLog.device_user_id == patient_id).order_by(HealthLog.record_date.asc()).all()

    if not logs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient '{patient_id}' not found in clinic records."
        )

    records = []
    for log in logs:
        records.append({
            "id": log.id,
            "date": log.record_date,
            "steps": log.steps,
            "restingHeartRate": log.heart_rate_resting,
            "heartRate": log.heart_rate,
            "hrvRmssd": log.hrv_rmssd_avg,
            "spo2": log.oxygen_saturation,
            "spo2Nadir": log.oxygen_saturation_nadir,
            "sleepMinutes": log.sleep_minutes,
            "sleepHours": round(log.sleep_minutes / 60.0, 1) if log.sleep_minutes else 0.0,
            "state": log.predicted_state,
            "riskScore": log.risk_score,
            "riskLevel": log.risk_level,
            "advisoryLevel": log.clinical_advisory_level,
            "summaryMessage": log.clinical_summary_message,
        })

    # Fetch doctor consultation notes
    notes = (
        db.query(ClinicalNote)
        .filter(ClinicalNote.patient_id == patient_id)
        .order_by(ClinicalNote.consultation_date.desc())
        .all()
    )
    notes_list = [
        {
            "id": n.id,
            "doctorName": n.doctor_name,
            "hospitalName": n.hospital_name,
            "consultationDate": n.consultation_date,
            "diagnosis": n.diagnosis,
            "clinicalNotes": n.clinical_notes,
            "treatmentPlan": n.treatment_plan,
            "advisoryLevel": n.advisory_level,
        }
        for n in notes
    ]

    latest = records[-1]
    return {
        "status": "success",
        "patientId": patient_id,
        "totalRecords": len(records),
        "latestMetrics": latest,
        "longitudinalHistory": records,
        "clinicalNotes": notes_list,
    }


@app.post("/api/hospital/patient/{patient_id}/notes")
def add_patient_clinical_note(patient_id: str, note_data: ClinicalNoteCreate, db: Session = Depends(get_db)):
    """Clinician submits consultation diagnosis, notes, and treatment plan for patient."""
    new_note = ClinicalNote(
        patient_id=patient_id,
        doctor_id="DOC-STATION-1",
        doctor_name=note_data.doctorName or "Dr. Attending Physician",
        hospital_name=note_data.hospitalName or "Metro Health Clinic",
        consultation_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        diagnosis=note_data.diagnosis,
        clinical_notes=note_data.clinicalNotes,
        treatment_plan=note_data.treatmentPlan,
        advisory_level=note_data.advisoryLevel or "Caution",
    )
    db.add(new_note)
    db.commit()
    db.refresh(new_note)

    return {
        "status": "success",
        "message": f"Clinical consultation note recorded for Patient {patient_id}",
        "noteId": new_note.id,
        "consultationDate": new_note.consultation_date,
    }


@app.get("/api/hospital/patient/{patient_id}/notes")
def get_patient_clinical_notes(patient_id: str, db: Session = Depends(get_db)):
    notes = (
        db.query(ClinicalNote)
        .filter(ClinicalNote.patient_id == patient_id)
        .order_by(ClinicalNote.consultation_date.desc())
        .all()
    )
    return {
        "status": "success",
        "patientId": patient_id,
        "notes": [
            {
                "id": n.id,
                "doctorName": n.doctor_name,
                "hospitalName": n.hospital_name,
                "consultationDate": n.consultation_date,
                "diagnosis": n.diagnosis,
                "clinicalNotes": n.clinical_notes,
                "treatmentPlan": n.treatment_plan,
                "advisoryLevel": n.advisory_level,
            }
            for n in notes
        ]
    }


# ---------------------------------------------------------
# Wearable Device Sync Endpoint (Health Connect Upsert)
# ---------------------------------------------------------

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

        dist_km = payload.distanceKm if payload.distanceKm is not None else ((payload.distanceMeters / 1000.0) if payload.distanceMeters is not None else None)
        cals = payload.calories if payload.calories is not None else payload.caloriesKcal
        hr = payload.heartRate if payload.heartRate is not None else payload.averageHeartRate

        # Field-Aware Upsert
        if existing_log:
            existing_log.steps = payload.steps
            existing_log.heart_rate = hr
            existing_log.oxygen_saturation = payload.oxygenSaturation
            existing_log.sleep_minutes = payload.sleepMinutes
            existing_log.record_start_time = payload.recordStartTime
            existing_log.record_end_time = payload.recordEndTime
            existing_log.collected_at = payload.collectedAt

            if cals is not None:
                existing_log.calories = cals
            if dist_km is not None:
                existing_log.distance_km = dist_km
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
                distance_km=dist_km,
                calories=cals,
                heart_rate=hr,
                heart_rate_resting=payload.heartRateResting,
                hrv_rmssd_avg=payload.hrvRmssdAvg,
                oxygen_saturation=payload.oxygenSaturation,
                oxygen_saturation_nadir=payload.oxygenSaturationNadir,
                sleep_minutes=payload.sleepMinutes,
                record_start_time=payload.recordStartTime,
                record_end_time=payload.recordEndTime,
                collected_at=payload.collectedAt,
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


@app.get("/download/apk", summary="Download compiled Android APK directly to phone")
@app.get("/apk", summary="Download compiled Android APK directly to phone")
def download_apk():
    """Serves the compiled Android WearableHealthAI.apk directly to any device/phone."""
    possible_paths = [
        Path(__file__).resolve().parent.parent / "WearableHealthAI.apk",
        Path("D:/downloads/WearableHealthAI.apk"),
        Path(__file__).resolve().parent.parent / "android_app/app/build/outputs/apk/debug/app-debug.apk",
    ]
    for apk_path in possible_paths:
        if apk_path.exists():
            return FileResponse(
                path=str(apk_path),
                filename="WearableHealthAI.apk",
                media_type="application/vnd.android.package-archive"
            )
    raise HTTPException(status_code=404, detail="WearableHealthAI.apk not found on server")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
