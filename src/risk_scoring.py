from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.preprocessing import EARLY_WARNING_FEATURE_COLUMNS, RECOMMENDATION_FEATURE_COLUMNS

RISK_LABELS = {
    "cardiovascular_strain": "Cardiovascular Strain",
    "sleep_deficit": "Sleep Deficit Risk",
    "chronic_stress": "Chronic Stress Risk",
    "recovery_failure": "Recovery Failure Risk",
    "overtraining": "Overtraining Risk",
    "fatigue_accumulation": "Fatigue Accumulation",
    "burnout": "Burnout Risk",
    "circadian_disruption": "Circadian Disruption",
    "metabolic_stress": "Metabolic Stress",
    "autonomic_imbalance": "Autonomic Imbalance",
}


def _ensure_row(features: pd.DataFrame | pd.Series | dict[str, Any]) -> pd.Series:
    if isinstance(features, pd.DataFrame):
        if features.empty:
            raise ValueError("Risk scoring requires at least one row.")
        return features.iloc[0]
    if isinstance(features, pd.Series):
        return features
    return pd.Series(features)


def _clamp_pct(value: float, default: float = 0.0) -> float:
    if pd.isna(value) or np.isinf(value):
        return default
    return round(max(0.0, min(100.0, float(value))), 1)


def _risk_level(score: float) -> str:
    if score >= 70:
        return "High"
    if score >= 40:
        return "Moderate"
    return "Low"


def detect_early_warning_signal(features: pd.DataFrame | pd.Series | dict[str, Any]) -> dict[str, Any]:
    row = _ensure_row(features)
    hrv_slope = float(row.get(EARLY_WARNING_FEATURE_COLUMNS[0], 0.0))
    sleep_slope = float(row.get(EARLY_WARNING_FEATURE_COLUMNS[1], 0.0))
    if hrv_slope < -0.05 and sleep_slope < -0.05:
        return {
            "triggered": True,
            "hrv_dev_slope_7d": round(hrv_slope, 4),
            "sleep_dev_slope_7d": round(sleep_slope, 4),
            "message": "HRV and sleep are both deteriorating.",
            "trend": "Deteriorating",
        }
    if hrv_slope > 0.05 and sleep_slope > 0.05:
        return {
            "triggered": False,
            "hrv_dev_slope_7d": round(hrv_slope, 4),
            "sleep_dev_slope_7d": round(sleep_slope, 4),
            "message": "HRV and sleep are improving together.",
            "trend": "Improving",
        }
    return {
        "triggered": False,
        "hrv_dev_slope_7d": round(hrv_slope, 4),
        "sleep_dev_slope_7d": round(sleep_slope, 4),
        "message": "No strong early warning pattern detected.",
        "trend": "Stable",
    }


def compute_health_risk_intelligence(features: pd.DataFrame | pd.Series | dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Compute sub-dimension physiological strain scores purely from wearable physiological signals."""
    row = _ensure_row(features)
    severity = float(row.get("severity_score", 0.0))
    hrv_dev = abs(float(row.get("hrv_dev", 0.0)))
    hr_dev = abs(float(row.get("hr_dev", 0.0)))
    sleep_dev = abs(float(row.get("sleep_dev", 0.0)))
    sleep_hours = float(row.get("sleep_duration_hours", 7.0))
    steps = float(row.get("steps", 6000.0))
    screen_time = float(row.get("screen_time_min", 0.0))
    caffeine = float(row.get("caffeine_mg", 0.0))

    raw_scores = {
        "cardiovascular_strain": _clamp_pct(severity * 7 + hr_dev * 16),
        "sleep_deficit": _clamp_pct(max(0.0, 7.5 - sleep_hours) * 18 + sleep_dev * 10),
        "chronic_stress": _clamp_pct(severity * 6 + hrv_dev * 14),
        "recovery_failure": _clamp_pct(severity * 5 + sleep_dev * 12 + hrv_dev * 10),
        "overtraining": _clamp_pct(severity * 5 + max(0.0, 12000 - steps) / 180 + hr_dev * 8),
        "fatigue_accumulation": _clamp_pct(severity * 5 + max(0.0, 6.5 - sleep_hours) * 12),
        "burnout": _clamp_pct(severity * 4 + max(0.0, screen_time - 240) * 0.12),
        "circadian_disruption": _clamp_pct(max(0.0, screen_time - 180) * 0.18 + sleep_dev * 8),
        "metabolic_stress": _clamp_pct(severity * 4 + max(0.0, caffeine - 250) / 5 + max(0.0, 5000 - steps) / 120),
        "autonomic_imbalance": _clamp_pct(hrv_dev * 18 + hr_dev * 12),
    }
    return {
        key: {"label": RISK_LABELS[key], "score": value, "level": _risk_level(value)}
        for key, value in raw_scores.items()
    }


def evaluate_clinical_persistence(features: pd.DataFrame | pd.Series | dict[str, Any], user_id: str | None = None) -> dict[str, Any]:
    """
    Evaluates multi-day persistence of physiological strain signals per user.
    Note: These are physiological strain metrics, not medical diagnostic rules.
    """
    if isinstance(features, (pd.Series, dict)):
        row = _ensure_row(features)
        triggers = []
        if float(row.get("hr_dev_z", 0.0)) > 2.0 and float(row.get("hrv_dev_z", 0.0)) < -1.5:
            triggers.append("Cardiovascular strain anomaly detected.")
        if float(row.get("spo2_avg_pct", 98.0)) < 93.0:
            triggers.append("Hypoxia signal (<93% SpO2) detected.")
        if float(row.get("severity_score", 0.0)) > 3.5:
            triggers.append("High physiological severity anomaly (>3.5) detected.")
        if float(row.get("sleep_duration_hours", 7.0)) < 5.5:
            triggers.append("Severe sleep deficit (<5.5h) detected.")
        level = "Attention Needed" if triggers else "Normal"
        return {
            "advisory_level": level,
            "persistent_triggers": triggers,
            "consecutive_high_risk_days": 1 if triggers else 0,
            "clinical_summary_message": " ".join(triggers) or "No persistent multi-day physiological anomaly detected.",
        }

    target_df = features.copy()
    if user_id is not None and "user_id" in target_df.columns:
        target_df = target_df[target_df["user_id"] == user_id]

    if target_df.empty:
        return {
            "advisory_level": "Normal",
            "persistent_triggers": [],
            "consecutive_high_risk_days": 0,
            "clinical_summary_message": "No historical records available.",
        }

    target_df = target_df.sort_values("date").reset_index(drop=True)
    cardio_days, hypoxia_days, severity_days, sleep_days = 0, 0, 0, 0

    for idx in range(len(target_df) - 1, -1, -1):
        r = target_df.iloc[idx]
        is_cardio = float(r.get("hr_dev_z", 0.0)) > 2.0 and float(r.get("hrv_dev_z", 0.0)) < -1.5
        is_hypoxia = float(r.get("spo2_avg_pct", 98.0)) < 93.0
        is_sev = float(r.get("severity_score", 0.0)) > 3.5
        is_sleep = float(r.get("sleep_duration_hours", 7.0)) < 5.5

        if is_cardio:
            cardio_days += 1
        elif idx == len(target_df) - 1:
            cardio_days = 0

        if is_hypoxia:
            hypoxia_days += 1
        elif idx == len(target_df) - 1:
            hypoxia_days = 0

        if is_sev:
            severity_days += 1
        elif idx == len(target_df) - 1:
            severity_days = 0

        if is_sleep:
            sleep_days += 1
        elif idx == len(target_df) - 1:
            sleep_days = 0

    triggers = []
    if cardio_days >= 3:
        triggers.append(f"Cardiovascular Strain anomaly persisting for {cardio_days} consecutive days.")
    if hypoxia_days >= 2:
        triggers.append(f"Hypoxia alert (<93% SpO2) persisting for {hypoxia_days} consecutive days.")
    if severity_days >= 4:
        triggers.append(f"Elevated physiological severity persisting for {severity_days} consecutive days.")
    if sleep_days >= 3:
        triggers.append(f"Severe sleep deficit (<5.5h) persisting for {sleep_days} consecutive days.")

    advisory_level = (
        "Strain Advisory" if (cardio_days >= 3 or hypoxia_days >= 2 or severity_days >= 4)
        else "Attention Needed" if triggers
        else "Normal"
    )
    consecutive_days = max(cardio_days, hypoxia_days, severity_days, sleep_days)
    summary_message = " ".join(triggers) if triggers else "All physiological persistence signals remain within baseline boundaries."

    return {
        "advisory_level": advisory_level,
        "persistent_triggers": triggers,
        "consecutive_high_risk_days": consecutive_days,
        "clinical_summary_message": summary_message,
    }


def calculate_risk_score(
    features: pd.DataFrame | pd.Series | dict[str, Any],
    state: str | None = None,
    state_confidence: float | None = None,
    state_duration: int | None = None,
) -> dict[str, Any]:
    """Calculate physiological risk index strictly based on wearable physiological measurements."""
    row = _ensure_row(features)
    severity_score = float(row.get("severity_score", 0.0))
    recommendation_context = {
        column: float(row.get(column, 0.0)) for column in RECOMMENDATION_FEATURE_COLUMNS if column in row.index
    }
    recommendation_context["sleep_duration_hours"] = float(row.get("sleep_duration_hours", 0.0))
    early_warning = detect_early_warning_signal(row)
    clinical_escalation = evaluate_clinical_persistence(features)
    risk_intelligence = compute_health_risk_intelligence(row)
    score = _clamp_pct(sum(item["score"] for item in risk_intelligence.values()) / len(risk_intelligence))

    if state == "Recovery":
        score = _clamp_pct(score - 12)
    elif state == "Baseline":
        score = _clamp_pct(score + 0)
    elif state == "Strain":
        score = _clamp_pct(score + 12)

    if state_duration is not None:
        score = _clamp_pct(score + max(0, state_duration - 1) * 2.5)
    if state_confidence is not None:
        score = _clamp_pct(score + max(0.0, float(state_confidence) - 0.5) * 10.0)
    if early_warning["triggered"]:
        score = _clamp_pct(score + 6.0)
    if clinical_escalation["advisory_level"] == "Strain Advisory":
        score = _clamp_pct(score + 10.0)

    return {
        "score": round(score, 1),
        "level": _risk_level(score),
        "severity_score": round(severity_score, 2),
        "state_duration_days": int(state_duration or 1),
        "state_confidence": round(float(state_confidence), 4) if state_confidence is not None else None,
        "early_warning": early_warning,
        "clinical_escalation": clinical_escalation,
        "recommendation_context": {key: round(value, 2) for key, value in recommendation_context.items()},
        "multi_risk": {item["label"]: item["score"] for item in risk_intelligence.values()},
        "risk_intelligence": risk_intelligence,
    }


def build_user_state_summary(state_distribution: pd.Series | dict[str, Any] | None) -> dict[str, Any]:
    if state_distribution is None:
        return {"distribution": {}, "summary_lines": []}
    distribution = {
        str(key): float(value)
        for key, value in (state_distribution.to_dict().items() if isinstance(state_distribution, pd.Series) else dict(state_distribution).items())
    }
    ordered_distribution = {key: round(value, 2) for key, value in sorted(distribution.items(), key=lambda item: item[1], reverse=True)}
    summary_lines = [f"User spent {value:.1f}% time in {key}." for key, value in ordered_distribution.items()]
    return {"distribution": ordered_distribution, "summary_lines": summary_lines}
