GLOBAL_SEVERITY_SCALE = {
    "info": 0.0,
    "low": 0.25,
    "moderate": 0.5,
    "high": 0.75,
    "critical": 1.0,
}

MANDATORY_ESCALATION_THRESHOLD = 0.75


def normalize_severity(sev: str) -> float:
    key = sev.lower().strip()
    return GLOBAL_SEVERITY_SCALE.get(key, GLOBAL_SEVERITY_SCALE["moderate"])


def severity_label(v: float) -> str:
    diffs = {k: abs(v - num) for k, num in GLOBAL_SEVERITY_SCALE.items()}
    # Choose the label with the smallest absolute distance
    return min(diffs.items(), key=lambda item: item[1])[0]
