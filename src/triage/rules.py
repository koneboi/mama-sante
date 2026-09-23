LEVELS = ["wait", "local_scan", "regional_imaging", "urgent_biopsy"]


def decide(
    risk_score: float,
    image_result: dict | None,
    escalate_threshold: float,
    risk_threshold: float = 0.5,
) -> str:
    """Combine the risk score and imaging confidence into a single triage level.

    ``risk_threshold`` is the decision threshold for the risk score; the API
    resolves it per patient (group-aware fairness thresholds) before calling.
    """
    conf = (image_result or {}).get("confidence", 0.0)
    above = (image_result or {}).get("above_floor", False)

    if above and conf >= escalate_threshold:
        return "urgent_biopsy"
    if above and risk_score >= risk_threshold:
        return "regional_imaging"
    if risk_score >= risk_threshold:
        return "local_scan"
    return "wait"
