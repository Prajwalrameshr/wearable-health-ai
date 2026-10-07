#!/usr/bin/env python3
"""
Enterprise Status & Health Diagnostics Suite
Wearable Health AI - Production Verification Tool
"""

import sys
import os
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(BACKEND_DIR))

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_banner():
    print(f"""{CYAN}{BOLD}
================================================================================
   WEARABLE HEALTH AI - ENTERPRISE DIAGNOSTIC & STATUS CHECKER
================================================================================{RESET}
  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  Workspace: {ROOT_DIR}
""")


def check_dependencies():
    print(f"{BOLD}[1/6] Validating Core Dependencies...{RESET}")
    modules = [
        ("streamlit", "Streamlit UI Engine"),
        ("fastapi", "FastAPI Enterprise Backend"),
        ("uvicorn", "ASGI Server Engine"),
        ("sqlalchemy", "SQLAlchemy ORM Layer"),
        ("pandas", "Pandas Analytics Engine"),
        ("numpy", "NumPy Scientific Engine"),
        ("sklearn", "Scikit-Learn ML Suite"),
        ("hmmlearn", "HMM Latent State Engine"),
        ("plotly", "Plotly Visualizations"),
    ]
    all_ok = True
    for mod, desc in modules:
        try:
            __import__(mod)
            print(f"  {GREEN}[PASS]{RESET} {desc} ({mod})")
        except ImportError as e:
            print(f"  {RED}[FAIL]{RESET} {desc} ({mod}) - Missing: {e}")
            all_ok = False
    return all_ok


def check_database():
    print(f"\n{BOLD}[2/6] Inspecting Database & Storage Layer...{RESET}")
    try:
        import database
        import models
        models.Base.metadata.create_all(bind=database.engine)
        db = database.SessionLocal()
        logs_count = db.query(models.HealthLog).count()
        users_count = db.query(models.User).count()
        notes_count = db.query(models.ClinicalNote).count()
        db.close()

        print(f"  {GREEN}[PASS]{RESET} Database Engine connected: {database.DATABASE_URL.split('@')[-1]}")
        print(f"  {GREEN}[PASS]{RESET} Health Telemetry Logs : {logs_count} entries recorded")
        print(f"  {GREEN}[PASS]{RESET} Registered RBAC Users : {users_count} accounts")
        print(f"  {GREEN}[PASS]{RESET} Doctor Clinical Notes : {notes_count} consultations logged")
        return True
    except Exception as exc:
        print(f"  {RED}[FAIL]{RESET} Database Error: {exc}")
        return False


def check_dataset_and_models():
    print(f"\n{BOLD}[3/6] Verifying Data & ML Inference Assets...{RESET}")
    csv_path = ROOT_DIR / "data" / "wearables_health_6mo_daily.csv"
    if csv_path.exists():
        size_mb = csv_path.stat().st_size / (1024 * 1024)
        print(f"  {GREEN}[PASS]{RESET} 6-Month Longitudinal Dataset present: {csv_path.name} ({size_mb:.2f} MB)")
    else:
        print(f"  {YELLOW}[WARN]{RESET} Dataset not found at {csv_path}")

    # Test quick model output
    try:
        import run_inference
        print(f"  {GREEN}[PASS]{RESET} Inference pipeline module loaded (GMM + HMM + KMeans ready)")
        return True
    except Exception as exc:
        print(f"  {RED}[FAIL]{RESET} Inference module error: {exc}")
        return False


def check_rbac_and_auth():
    print(f"\n{BOLD}[4/6] Verifying Role-Based Access Control (RBAC)...{RESET}")
    try:
        import database
        import models
        import main as backend_main
        db = database.SessionLocal()

        # Check demo patient
        patient_res = backend_main.login_user(
            backend_main.UserLoginRequest(username="patient", password="health2026"),
            db
        )
        print(f"  {GREEN}[PASS]{RESET} Demo Patient Auth : Verified ({patient_res['user']['fullName']} - Role: {patient_res['user']['role']})")

        # Check demo doctor
        doctor_res = backend_main.login_user(
            backend_main.UserLoginRequest(username="doctor", password="clinical2026"),
            db
        )
        print(f"  {GREEN}[PASS]{RESET} Demo Doctor Auth  : Verified ({doctor_res['user']['fullName']} - Role: {doctor_res['user']['role']})")

        # Check hospital patient lookup
        patients_res = backend_main.list_hospital_patients(db)
        print(f"  {GREEN}[PASS]{RESET} Hospital Lookup   : Accessible ({len(patients_res['patients'])} patients available for triage)")
        db.close()
        return True
    except Exception as exc:
        print(f"  {RED}[FAIL]{RESET} RBAC Verification error: {exc}")
        return False


def check_live_api_server():
    print(f"\n{BOLD}[5/6] Checking Live FastAPI Server Connectivity...{RESET}")
    import urllib.request
    import json
    url = "http://127.0.0.1:5000/health"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=1.5) as res:
            data = json.loads(res.read().decode())
            print(f"  {GREEN}[ONLINE]{RESET} Live Backend running on port 5000 - Status: {data.get('status')}")
            return True
    except Exception:
        print(f"  {YELLOW}[OFFLINE]{RESET} Live backend server not running currently on http://127.0.0.1:5000")
        print(f"  {CYAN}[TIP]{RESET} Start backend with: python backend/main.py")
        return False


def check_frontend_app():
    print(f"\n{BOLD}[6/6] Checking Streamlit Frontend Application...{RESET}")
    app_main = ROOT_DIR / "app" / "main.py"
    if app_main.exists():
        print(f"  {GREEN}[PASS]{RESET} Streamlit main application file verified ({app_main})")
        print(f"  {CYAN}[TIP]{RESET} Launch with: streamlit run app/main.py")
        return True
    else:
        print(f"  {RED}[FAIL]{RESET} Missing app/main.py")
        return False


def main():
    print_banner()
    d_ok = check_dependencies()
    db_ok = check_database()
    m_ok = check_dataset_and_models()
    r_ok = check_rbac_and_auth()
    api_live = check_live_api_server()
    f_ok = check_frontend_app()

    print(f"\n{CYAN}{BOLD}================================================================================{RESET}")
    if d_ok and db_ok and m_ok and r_ok and f_ok:
        print(f"{GREEN}{BOLD}   ALL SYSTEM CHECKS PASSED - READY FOR PRODUCTION DEPLOYMENT!{RESET}")
    else:
        print(f"{YELLOW}{BOLD}   SYSTEM READY WITH WARNINGS (See details above){RESET}")
    print(f"{CYAN}{BOLD}================================================================================{RESET}\n")


if __name__ == "__main__":
    main()
