# Wearable Health AI - Enterprise Deployment & Operations Guide

Welcome to the **Wearable Health AI Platform**. This guide details the complete end-to-end architecture featuring **Native Android Smartwatch App (with built-in Role-Based Access)**, **FastAPI Machine Learning Backend**, and **Production Database Architecture**.

---

## 🗄️ Database Architecture & Specifications

### 1. Default Embedded Database: SQLite
* **File Location**: `backend/health_database.db`
* **Driver**: `sqlite+pysqlite` via SQLAlchemy ORM
* **Features**:
  * Zero-configuration, local ACID persistence.
  * Auto-initializes schema on startup (`backend/database.py`).
  * Seeds sample patient records (`U0042`, `U0001`, `android_device_test_30d`) and pre-populates with realistic historical data and consultation notes.

### 2. Production Scalable Cloud Database: PostgreSQL (Neon.tech / AWS RDS / Supabase)
* If `DATABASE_URL` is configured in `backend/.env` or system environment, the platform dynamically switches to PostgreSQL without any code changes:
  ```env
  DATABASE_URL=postgresql://user:password@ep-example.neon.tech/neondb?sslmode=require
  ```
* Supports connection pooling via SQLAlchemy `QueuePool`.

### 3. Database Schema Models ([`backend/models.py`](file:///d:/downloads/wearable-health-ai-20260820T054456Z-1-001/wearable-health-ai/backend/models.py))
* **`User`**: Role-based authentication (`role`: `patient` or `hospital`), hashed passwords, patient admission tokens (`#AUTH-MED-...`).
* **`HealthLog`**: 30 to 180-day longitudinal biometric logs containing:
  - `spo2` (Oxygen Saturation %)
  - `heart_rate` & `heart_rate_resting` (BPM)
  - `steps` (Daily pedometer count)
  - `sleep_hours` (Sleep duration)
  - `hrv_rmssd` (Autonomic heart rate variability)
  - `state` (Recovery, Baseline, Strain)
  - `risk_score` (0–100 Physiological Strain Index)
* **`ClinicalNote`**: Hospital EHR doctor consultations (`patient_id`, `doctor_name`, `hospital_name`, `diagnosis`, `clinical_notes`, `treatment_plan`, `advisory_level`).

---

## 📱 Native Android App: Built-In Role-Based Access & The 4 Hero Parameters

Everything is implemented directly in the native Android application ([`android_app/`](file:///d:/downloads/wearable-health-ai-20260820T054456Z-1-001/wearable-health-ai/android_app)):

### 🌟 The 4 Main Biometric Parameters (Hero Grid)
The Android app presents these 4 vital signals prominently in high-contrast cyber-medical telemetry cards:
1. **🫁 SpO2 (Oxygen Saturation)**: Live pulse oximetry, e.g. `98.4%` (Normal > 95%).
2. **💓 Heart Beat (BPM)**: Heart rate pulse, resting heart rate, and autonomic HRV (e.g. `72 BPM`, Resting `64 bpm`, HRV `48 ms`).
3. **🚶 Daily Steps**: Pedometer counter with active distance and burned calories (e.g. `8,420 steps`, `5.8 km`, `435 kcal`).
4. **🌙 Sleep Duration**: Restorative sleep architecture (e.g. `7h 45m`, `465 min total`).

### 👥 Dual Role Switching Inside the Android App
Toggle between roles directly at the top of the mobile screen:
1. **🏃 Normal User / Patient Portal**:
   - Live Health Connect sensor ingestion from smartwatches (Samsung Galaxy Watch, Pixel Watch, WearOS).
   - "⚡ Sync Watch to AI Engine" button communicating directly with FastAPI (`POST /api/health/records`).
   - Live AI Engine HUD: Displays physiological state (`Recovery`, `Baseline`, `Strain`), Strain Index (0–100), and clinical recommendations.
   - Digital Patient Health Pass & Hospital Admission ID (`#AUTH-MED-42`) to present when visiting a hospital clinic.
2. **🏥 Hospital / Doctor Portal**:
   - Clinical EHR station for physicians (e.g., *Dr. Elena Vance, MD*).
   - Instant Patient EHR Lookup: Enter or select Patient ID (`U0042`, `U0001`, `android_user`) to fetch complete past vitals.
   - Doctor's Intake Vitals Review: View patient's historical SpO2, Heart Beat, Steps, and Sleep.
   - Clinical Consultation Entry Form: Formulate clinical diagnosis, doctor notes, treatment plan (Rx), and advisory level.
   - "💾 Save Consultation to EHR Database": Persists consultation directly into the database.
   - Historical Consultation Timeline: Review chronological records of past hospital visits.

---

## 🚀 Complete Step-by-Step Execution Guide (A to Z)

### Step 1: Install Dependencies
Open PowerShell or Terminal in the project root:
```powershell
pip install -r backend/requirements.txt
```

### Step 2: Start the FastAPI Backend Server
Run the FastAPI enterprise backend:
```powershell
python backend/main.py
```
* The server starts on `http://0.0.0.0:5000`
* Interactive API Documentation (Swagger) is available at: `http://127.0.0.1:5000/docs`
* Live Health check endpoint: `http://127.0.0.1:5000/health`

### Step 3: Install the Android App (APK)
The pre-compiled Android APK is available in the project and your Downloads folder:
* **Workspace Location**: [`WearableHealthAI.apk`](file:///d:/downloads/wearable-health-ai-20260820T054456Z-1-001/wearable-health-ai/WearableHealthAI.apk)
* **Downloads Location**: `D:\downloads\WearableHealthAI.apk`

**To Install via ADB (USB / Emulator):**
```powershell
adb install -r WearableHealthAI.apk
```
*Or copy `WearableHealthAI.apk` to your Android phone via USB/WhatsApp/Drive and tap to install.*

### Step 4: Configure Backend Server in the App
1. Open **Wearable Health AI** on your Android phone or emulator.
2. Tap the **⚙️** icon in the top right app bar.
3. Set the FastAPI Server URL:
   - **For Android Emulator**: Tap the **Emulator** button (`http://10.0.2.2:5000/`).
   - **For Physical Android Phone (Wi-Fi)**: Enter your computer's local IP (e.g., `http://192.168.1.15:5000/`).
4. Tap **Save URL**.

### Step 5: Test Patient Portal
1. Select the **🏃 Normal User (Patient)** tab.
2. Observe the 4 core parameters: **SpO2**, **Heart Beat**, **Steps**, and **Sleep**.
3. Tap **⚡ Sync Watch to AI Engine**.
4. The app sends the telemetry payload to the FastAPI backend and displays:
   - AI Engine State (`🟢 Recovery`)
   - Physiological Strain Index (`18.5 / 100`)
   - Clinical Advisory Tier (`Nominal Tier 1`)
   - AI Clinical Recommendations.

### Step 6: Test Hospital / Doctor Portal
1. Tap the **🏥 Hospital (Doctor)** tab at the top of the app.
2. In the EHR lookup box, select or type Patient ID: `U0042` (or `U0001`).
3. Tap **Fetch EHR**.
4. The doctor can instantly view:
   - Patient's past records in the database.
   - The patient's 4 core vitals: SpO2, Heart Beat, Steps, and Sleep.
5. In the **Clinical Consultation Form**, enter:
   - Diagnosis: `Sinus rhythm stable, mild post-exercise fatigue`
   - Clinical Notes: `Vitals review shows normal SpO2 and restorative sleep profile.`
   - Treatment Plan: `Maintain hydration, electrolyte intake, follow up in 14 days.`
6. Tap **💾 Save Consultation to EHR Database**.
7. The note is permanently saved into the database and appears in the consultation timeline!

---

## 🔍 Verification & Automated Status Diagnostics

Run the comprehensive health check script anytime:
```powershell
python check_status.py
```

Run automated backend & RBAC unit tests:
```powershell
pytest tests/test_rbac_and_hospital.py
```
*(All 17 tests validate authentication, patient data querying, doctor note creation, and schema integrity).*
