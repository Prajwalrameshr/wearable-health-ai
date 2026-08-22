from __future__ import annotations

import math
from typing import Any

# Demographic Cohort Reference Metrics: (Mean, Standard Deviation)
# Grouped by (Age Range Key, Gender)
COHORT_REFERENCE_TABLE: dict[tuple[str, str], dict[str, tuple[float, float]]] = {
    ("<30", "male"): {
        "resting_hr_bpm": (62.0, 7.5),
        "hrv_rmssd_ms": (58.0, 15.0),
        "sleep_duration_hours": (7.6, 0.9),
        "steps": (8500.0, 2200.0),
        "spo2_avg_pct": (98.2, 1.1),
    },
    ("<30", "female"): {
        "resting_hr_bpm": (64.0, 7.5),
        "hrv_rmssd_ms": (56.0, 14.5),
        "sleep_duration_hours": (7.8, 0.9),
        "steps": (8200.0, 2100.0),
        "spo2_avg_pct": (98.4, 1.0),
    },
    ("30-45", "male"): {
        "resting_hr_bpm": (65.0, 8.0),
        "hrv_rmssd_ms": (46.0, 12.5),
        "sleep_duration_hours": (7.2, 1.0),
        "steps": (7800.0, 2000.0),
        "spo2_avg_pct": (97.8, 1.2),
    },
    ("30-45", "female"): {
        "resting_hr_bpm": (67.0, 8.0),
        "hrv_rmssd_ms": (44.0, 12.0),
        "sleep_duration_hours": (7.4, 1.0),
        "steps": (7500.0, 1900.0),
        "spo2_avg_pct": (98.0, 1.1),
    },
    ("46-60", "male"): {
        "resting_hr_bpm": (68.0, 8.5),
        "hrv_rmssd_ms": (35.0, 10.0),
        "sleep_duration_hours": (6.9, 1.1),
        "steps": (6800.0, 1800.0),
        "spo2_avg_pct": (97.2, 1.3),
    },
    ("46-60", "female"): {
        "resting_hr_bpm": (70.0, 8.5),
        "hrv_rmssd_ms": (33.0, 9.5),
        "sleep_duration_hours": (7.1, 1.1),
        "steps": (6600.0, 1700.0),
        "spo2_avg_pct": (97.5, 1.2),
    },
    ("60+", "male"): {
        "resting_hr_bpm": (71.0, 9.0),
        "hrv_rmssd_ms": (26.0, 8.0),
        "sleep_duration_hours": (6.6, 1.2),
        "steps": (5500.0, 1600.0),
        "spo2_avg_pct": (96.5, 1.5),
    },
    ("60+", "female"): {
        "resting_hr_bpm": (73.0, 9.0),
        "hrv_rmssd_ms": (25.0, 7.5),
        "sleep_duration_hours": (6.8, 1.2),
        "steps": (5300.0, 1500.0),
        "spo2_avg_pct": (96.8, 1.4),
    },
}

DEFAULT_REFERENCE: dict[str, tuple[float, float]] = {
    "resting_hr_bpm": (66.0, 8.0),
    "hrv_rmssd_ms": (45.0, 12.0),
    "sleep_duration_hours": (7.2, 1.0),
    "steps": (7200.0, 2000.0),
    "spo2_avg_pct": (97.6, 1.2),
}

METRIC_LABELS: dict[str, str] = {
    "resting_hr_bpm": "Resting Heart Rate",
    "hrv_rmssd_ms": "HRV (RMSSD)",
    "sleep_duration_hours": "Sleep Duration",
    "steps": "Daily Steps",
    "spo2_avg_pct": "SpO2 Saturation",
}


def _get_age_key(age: float | int | None) -> str:
    if age is None or math.isnan(float(age)):
        return "30-45"
    a = float(age)
    if a < 30:
        return "<30"
    if a <= 45:
        return "30-45"
    if a <= 60:
        return "46-60"
    return "60+"


def _normal_cdf(x: float) -> float:
    """Standard Normal Cumulative Distribution Function Phi(x)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def benchmark_against_cohort(
    user_metrics: dict[str, Any],
    age: float | int | None = 35,
    gender: str | None = "unknown",
) -> dict[str, Any]:
    """
    Computes exact cohort Z-scores and Percentile Ranks for user metrics
    against demographic peer reference tables.
    """
    age_key = _get_age_key(age)
    g_key = str(gender).strip().lower() if gender else "unknown"
    if g_key not in {"male", "female"}:
        g_key = "male"  # Default reference blend

    ref_table = COHORT_REFERENCE_TABLE.get((age_key, g_key), DEFAULT_REFERENCE)

    results: dict[str, dict[str, Any]] = {}
    for metric_key, (mean_val, std_val) in ref_table.items():
        val = user_metrics.get(metric_key)
        if val is None or math.isnan(float(val)):
            continue

        raw_value = float(val)
        z_score = (raw_value - mean_val) / std_val if std_val > 0 else 0.0

        # For resting_hr_bpm, lower is generally better/higher fitness, so invert percentile ranking
        if metric_key == "resting_hr_bpm":
            percentile = round((1.0 - _normal_cdf(z_score)) * 100.0, 1)
        else:
            percentile = round(_normal_cdf(z_score) * 100.0, 1)

        rating = (
            "Above Average"
            if percentile >= 65.0
            else "Below Average"
            if percentile <= 35.0
            else "Average"
        )

        results[metric_key] = {
            "label": METRIC_LABELS.get(metric_key, metric_key),
            "user_value": round(raw_value, 2),
            "cohort_mean": round(mean_val, 2),
            "z_score": round(z_score, 2),
            "percentile": percentile,
            "rating": rating,
        }

    return {
        "age_group": age_key,
        "gender": g_key,
        "metrics": results,
    }
