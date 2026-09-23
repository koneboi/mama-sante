LEVELS = ["wait", "local_scan", "regional_imaging", "urgent_biopsy"]


def decide(risk_score: float, image_result: dict | None, escalate_threshold: float) -> str:
    """Combine the risk score and imaging confidence into a single triage level."""
    conf = (image_result or {}).get("confidence", 0.0)
    above = (image_result or {}).get("above_floor", False)

    if above and conf >= escalate_threshold:
        return "urgent_biopsy"
    if above and risk_score >= 0.5:
        return "regional_imaging"
    if risk_score >= 0.5:
        return "local_scan"
    return "wait"
