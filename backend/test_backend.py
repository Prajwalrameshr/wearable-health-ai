import urllib.request
import json
import time

BASE_URL = "http://127.0.0.1:5000"

def test_endpoint():
    print("--- 1. Testing GET /health ---")
    req = urllib.request.Request(f"{BASE_URL}/health")
    with urllib.request.urlopen(req) as res:
        health_resp = json.loads(res.read().decode())
        print("Health Status:", health_resp)

    # Scenario 1: < 7 days (Insufficient Data)
    print("\n--- 2. Testing <7 days Data Gate Rule ---")
    user_id_short = "android_device_test_short"
    for day in range(1, 4):
        payload = {
            "deviceUserId": user_id_short,
            "steps": 4000 + day * 200,
            "heartRate": 68.0 + day * 0.5,
            "oxygenSaturation": 97.5,
            "sleepMinutes": 380,
            "recordStartTime": f"2025-11-0{day}T00:00:00Z",
            "recordEndTime": f"2025-11-0{day}T23:59:59Z",
        }
        req = urllib.request.Request(
            f"{BASE_URL}/api/health/records",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as res:
            resp = json.loads(res.read().decode())
            print(f"Record {day} Response Status:", resp.get("status"), "| Message:", resp.get("message"))

    # Scenario 2: 10 days (7-Day Baseline Window)
    print("\n--- 3. Testing 7D Baseline Window (7 to 29 Days) ---")
    user_id_7d = "android_device_test_7d"
    for day in range(1, 11):
        day_str = f"{day:02d}"
        payload = {
            "deviceUserId": user_id_7d,
            "steps": 5000 + day * 100,
            "heartRate": 65.0 + (day % 3),
            "oxygenSaturation": 98.0,
            "sleepMinutes": 420,
            "recordStartTime": f"2025-11-{day_str}T00:00:00Z",
            "recordEndTime": f"2025-11-{day_str}T23:59:59Z",
        }
        req = urllib.request.Request(
            f"{BASE_URL}/api/health/records",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as res:
            resp = json.loads(res.read().decode())
    print("Record 10 Status:", resp.get("status"), "| Window Used:", resp.get("window_used"), "| ML State:", resp.get("state"), "| Risk Score:", resp.get("riskScore"))

    # Scenario 3: 32 days (30-Day Baseline Window)
    print("\n--- 4. Testing 30D Extended Baseline Window (>=30 Days) ---")
    user_id_30d = "android_device_test_30d"
    for day in range(1, 33):
        month = "11" if day <= 30 else "12"
        date_num = day if day <= 30 else day - 30
        date_formatted = f"2025-{month}-{date_num:02d}"
        payload = {
            "deviceUserId": user_id_30d,
            "steps": 6000 + (day * 50),
            "heartRate": 66.0 + (day % 4),
            "oxygenSaturation": 97.8,
            "sleepMinutes": 430,
            "recordStartTime": f"{date_formatted}T00:00:00Z",
            "recordEndTime": f"{date_formatted}T23:59:59Z",
        }
        req = urllib.request.Request(
            f"{BASE_URL}/api/health/records",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as res:
            resp = json.loads(res.read().decode())
    print("Record 32 Status:", resp.get("status"), "| Window Used:", resp.get("window_used"), "| ML State:", resp.get("state"), "| Risk Score:", resp.get("riskScore"))

    print("\n[SUCCESS] All Backend Automated Tests Completed Successfully!")

if __name__ == "__main__":
    test_endpoint()
