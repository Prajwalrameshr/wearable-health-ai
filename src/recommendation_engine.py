def generate_recommendations(analysis_result: dict) -> list[str]:
    risk = analysis_result["risk"]
    state = analysis_result["state"]
    context = analysis_result.get("recommendation_context", {})
    early_warning = analysis_result.get("early_warning", {})
    environment = analysis_result.get("environment") or {}
    environment_impact = analysis_result.get("environment_impact") or {}

    steps = float(context.get("steps", 0.0))
    sleep_hours = float(context.get("sleep_duration_hours", 0.0))
    caffeine = float(context.get("caffeine_mg", 0.0))
    spo2 = float(context.get("spo2_avg_pct", 100.0))
    screen_time = float(context.get("screen_time_min", 0.0))
    aqi = float(environment.get("aqi", 1.0))

    recommendations: list[str] = []
    if state == "Strain":
        recommendations.append("Reduce training intensity and prioritize recovery inputs today.")
    elif state == "Recovery":
        recommendations.append("Maintain current habits and use this window for productive training or focused work.")
    else:
        recommendations.append("Hold a balanced routine and monitor whether your physiology drifts toward strain.")

    if sleep_hours < 6.5:
        recommendations.append("Extend sleep opportunity tonight because recovery is being limited by insufficient sleep.")
    if steps < 4000:
        recommendations.append("Add light movement or walking to prevent recovery from stagnating.")
    if early_warning.get("trend") == "Deteriorating":
        recommendations.append("The temporal pattern is deteriorating, so use the next 24 hours as a stabilization window.")
    if spo2 < 95:
        recommendations.append("Monitor oxygen saturation and avoid unusually intense exertion until readings stabilize.")
    if caffeine > 250:
        recommendations.append("Reduce caffeine because it may be amplifying strain and sleep disruption.")
    if screen_time > 240:
        recommendations.append("Cut late-evening screen exposure to improve sleep quality.")
    if aqi >= 4:
        recommendations.append("Poor air quality is present, so keep activity indoors if possible.")
    if risk["level"] == "High":
        recommendations.append("Make the next 24 to 48 hours recovery-focused and escalate clinically if symptoms persist.")
    if environment_impact.get("recommendation"):
        recommendations.append(str(environment_impact["recommendation"]))
    return list(dict.fromkeys(recommendations))
