# Wearable Health AI - Enterprise Deployment & Status Guide

Welcome to the **Wearable Health AI Enterprise Platform** featuring **Role-Based Access Control (RBAC)**, **Doctor Hospital Portal**, **Patient Health Pass with QR**, and **Animated Biometric HUD Telemetry**.

---

## 🌟 Key New Features & Architecture

### 1. Dual Role-Based Access Control (RBAC)
- **🏃 Normal User / Patient Portal (`role: patient`)**:
  - **Live Biometric Telemetry HUD**: Beating heart rate monitor, nocturnal HRV RMSSD recovery index, restorative sleep hours, and blood oxygen (SpO2).
  - **Physiological Strain Index Gauge**: 0–100 composite risk scoring with color-coded safety tiers.
  - **AI Daily Insights & Action Engine**: Proactive recovery recommendations tailored to recent biomarker deviations.
  - **🎫 Digital Patient Health Pass & Hospital Admission QR**: Allows patients to present a verifiable admission pass and unique medical token (`#AUTH-MED-...`) when visiting any hospital or clinic.
  - **Interactive What-If Simulator**: Real-time multi-signal physiological stress prediction based on sleep, HRV, and activity adjustments.

- **🏥 Hospital / Clinician Portal (`role: hospital`)**:
  - **Instant Patient EHR Lookup**: Doctors can search or select any visiting patient (e.g., `U0042`, `U0001`, `android_device_test_30d`, etc.) and retrieve their **complete past medical records (up to 6 months / 180 days)**.
  - **Longitudinal Medical Trend Analysis**: Multi-chart visualization showing continuous HRV, resting heart rate, sleep duration, and SpO2 nadir.
  - **🩺 Doctor Consultation & Clinical Notes Formulation**: Attending physicians can input clinical impressions, diagnose autonomic fatigue, prescribe deload protocols or medications, set clinical advisory tiers (Normal, Caution, High Alert, Critical Escalation), and save directly into the patient's electronic medical record.
  - **Historical Consultation Timeline**: Chronological log of previous doctor visits, diagnosis, and treatment plans.
  - **Hospital Clinic Cohort Triage**: Real-time status breakdown across monitored patients.

### 2. Next-Level UI & Dynamic Animations
- **Obsidian-Cyan Cyber-Medical Glassmorphism**: High-contrast, clean typography powered by Google Fonts (*Outfit*, *Inter*, and *JetBrains Mono*).
- **Continuous Oscilloscope ECG Waveform**: Animated real-time electrocardiogram wave traversing the HUD.
- **Pulsing Cardiac Heartbeat**: Micro-animated beating heart icon with dynamic BPM pulse.
- **Glowing Holographic Telemetry Cards**: Hover elevation and glowing status accents.

---

## 🚀 Quick Start & How to Run

### 1. Clone & Set Up Environment
```bash
# Ensure Python 3.10+ is installed
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

### 2. Launch FastAPI Enterprise Backend
```bash
# Starts the backend on http://127.0.0.1:5000
python backend/main.py
```
*Swagger API Documentation is accessible at: `http://127.0.0.1:5000/docs`*

### 3. Launch Streamlit Interactive UI
```bash
# In a new terminal window:
streamlit run app/main.py
```
*Opens automatically in your browser at: `http://localhost:8501`*

---

## 🔍 How to Check System Status & Health

Run the automated diagnostic suite at any time:
```bash
python check_status.py
```

### Diagnostic Output Checks:
1. **Dependency Verification**: Confirms all ML libraries, Streamlit, and FastAPI are installed.
2. **Database Integrity**: Validates connection to SQLite/PostgreSQL, counts of health telemetry logs, registered users, and clinical notes.
3. **Data & ML Models**: Confirms the 6-month continuous dataset and GMM/HMM clustering pipeline.
4. **RBAC & Authentication**: Verifies login authentication for both Patient and Doctor accounts.
5. **Backend Server Status**: Pings the live HTTP `/health` endpoint.
6. **Frontend App Verification**: Confirms readiness of `app/main.py`.

### Live API Status Endpoint:
When the backend is running, check via browser or curl:
```bash
curl http://127.0.0.1:5000/health
```
Response:
```json
{
  "status": "healthy",
  "database": "connected",
  "total_health_logs": 184,
  "total_users": 2,
  "total_clinical_notes": 3,
  "timestamp": "2026-10-08T00:11:51.000000Z"
}
```

---

## 👥 Demo Authentication Credentials

| Role | Username | Password | Full Name / Description |
| :--- | :--- | :--- | :--- |
| **Patient / User** | `patient` | `health2026` | Alex Mercer (Patient ID: `U0042`) |
| **Hospital / Doctor** | `doctor` | `clinical2026` | Dr. Elena Vance, MD (Cardiology & Autonomic Medicine) |

*You can also switch roles with 1 click using the **🔄 Switch Role** button at the top of the app bar!*

---

## 🌐 Production Deployment Options

### Option A: Docker Deployment (Recommended)
Create a `docker-compose.yml`:
```yaml
version: '3.8'
services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    ports:
      - "5000:5000"
    environment:
      - DATABASE_URL=sqlite:///./health_database.db

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "8501:8501"
    depends_on:
      - backend
```

### Option B: Streamlit Community Cloud
1. Push repository to GitHub.
2. Log in to [Streamlit Cloud](https://share.streamlit.io).
3. Connect your repository: `Prajwalrameshr/wearable-health-ai`.
4. Main file path: `app/main.py`.
5. Deploy with 1 click!

### Option C: Cloud VPS (AWS / GCP / DigitalOcean)
Use `systemd` or `supervisord` to manage both services:
- Backend: `uvicorn main:app --host 0.0.0.0 --port 5000 --app-dir backend`
- Frontend: `streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0`

---

## 🧪 Automated Testing
Run the complete test suite to ensure 100% test pass rate:
```bash
pytest tests/test_rbac_and_hospital.py tests/test_models.py tests/test_risk_scoring.py
```
All unit tests validate RBAC logins, doctor patient history lookup, consultation note persistence, and ML models.
