import sys
from pathlib import Path
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import main
import database
import models

@pytest.fixture
def db_session():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_auth_login_patient(db_session):
    login_req = main.UserLoginRequest(username="patient", password="health2026")
    resp = main.login_user(login_req, db_session)
    assert resp["status"] == "success"
    assert resp["user"]["role"] == "patient"
    assert resp["user"]["patientId"] == "U0042"
    assert "token" in resp


def test_auth_login_doctor(db_session):
    login_req = main.UserLoginRequest(username="doctor", password="clinical2026")
    resp = main.login_user(login_req, db_session)
    assert resp["status"] == "success"
    assert resp["user"]["role"] == "hospital"
    assert "Elena Vance" in resp["user"]["fullName"]
    assert "token" in resp


def test_hospital_list_patients(db_session):
    resp = main.list_hospital_patients(db_session)
    assert resp["status"] == "success"
    assert resp["count"] >= 1
    patient_ids = [p["patientId"] for p in resp["patients"]]
    assert "U0042" in patient_ids


def test_hospital_get_patient_history(db_session):
    history = main.get_patient_history("U0042", limit=30, db=db_session)
    assert history["status"] == "success"
    assert history["patientId"] == "U0042"
    assert history["totalRecords"] >= 7
    assert len(history["longitudinalHistory"]) >= 7
    assert "clinicalNotes" in history
    assert len(history["clinicalNotes"]) >= 1


def test_hospital_add_and_get_clinical_note(db_session):
    note_req = main.ClinicalNoteCreate(
        patientId="U0042",
        doctorName="Dr. Vance",
        hospitalName="Metro General",
        diagnosis="Autonomic Baseline Evaluation",
        clinicalNotes="Patient shows positive recovery trajectory with consistent 7.5h sleep.",
        treatmentPlan="Maintain active mobility and scheduled sleep windows.",
        advisoryLevel="Normal",
    )
    post_resp = main.add_patient_clinical_note("U0042", note_req, db_session)
    assert post_resp["status"] == "success"
    assert "noteId" in post_resp

    get_resp = main.get_patient_clinical_notes("U0042", db_session)
    assert get_resp["status"] == "success"
    assert any(n["diagnosis"] == "Autonomic Baseline Evaluation" for n in get_resp["notes"])
