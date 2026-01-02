import re
from typing import List, Sequence, Dict, Any

from .signals import DetectorSignal


def simple_pattern_detector(
    *,
    name: str,
    text: str,
    regexes: Sequence[str],
    keywords: Sequence[str] | None = None,
    high_keywords: Sequence[str] | None = None,
) -> DetectorSignal:
    """
    Heuristic detector helper that scans text for regex and keyword matches
    and returns a DetectorSignal with severity and confidence derived from hits.
    """
    keywords = keywords or []
    high_keywords = high_keywords or []

    lowered = text.lower()
    matches: List[str] = []
    high_hits: List[str] = []

    for pattern in regexes:
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
            hits = compiled.findall(text)
            if hits:
                if isinstance(hits[0], tuple):
                    hits = [" ".join(h).strip() for h in hits]
                matches.extend([str(h) for h in hits[:5]])
        except re.error:
            continue

    for kw in keywords:
        if kw.lower() in lowered:
            matches.append(kw)

    for kw in high_keywords:
        if kw.lower() in lowered:
            high_hits.append(kw)

    total_hits = len(matches) + len(high_hits)
    violation = total_hits > 0

    # Severity tiers
    if not violation:
        severity = "S0"
    elif len(high_hits) > 0 or total_hits >= 5:
        severity = "S3"
    elif total_hits >= 3:
        severity = "S2"
    else:
        severity = "S1"

    confidence = min(1.0, 0.2 * total_hits + 0.3 * len(high_hits))

    return DetectorSignal(
        detector_name=name,
        severity=severity,
        description=f"{name} risk detected" if violation else f"No {name} indicators detected",
        confidence=confidence,
        violations=[name] if violation else [],
        evidence={
            "matches": matches,
            "high_impact_hits": high_hits,
            "violation": violation,
        },
    )
